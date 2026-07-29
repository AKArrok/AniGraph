# AniGraph 架构文档

> 外部 HTTP / Python 接入见 [接口文档](api.md)，内部模块与状态字段见 [Python API 参考](api_reference.md)。

## 概述

AniGraph 是一个基于 LangGraph 的多 Agent ACG 番剧检索、问答与推荐系统。核心采用 **ExecutionPlan 驱动编排** 的设计：Planner 输出一份执行计划，图引擎根据计划自动选择检索路径、Expert 组合和回答策略；截图请求先经过独立的识图节点。

### v2.5 关键变化

- **检索层拆分**：gents/retrieval.py 汇聚所有检索逻辑，	ools/rag.py 降级为纯工具
- **metadata_fallbacks 共享**：Simple Fact 快速通道与全流程共享 etch_by_studio_year / etch_by_title / ilter_metadata_by_query 兜底规则
- **SessionStore 统一**：main.py / chat.py / server.py 共用 gents/session_store.py 的 SessionStore，同一 	hread_id 的 MemorySaver 与已编译图实例复用
- **answer_planner 独立**：零 LLM 成本回答结构规划器，随机选结构避免千篇一律
- **query_processing 独立**：查询优化节点从 	ools/query_processing.py 迁入 gents/query_processor.py
- **alias_resolve 管道式**：别名解析与实体识别拆为 lias.py + ntity_resolver.py，新增 metadata_index.fuzzy_lookup 管道匹配
- **默认 PINECONE_SEARCH_TYPE=similarity**：实测 MMR 延迟约 29s 而 similarity 仅约 1.4s，降维后的 RRF 融合 + 去重压缩已能处理多样性
- **MAX_REPLANS 默认 0**：v2.5 默认关闭重规划，由 MAX_REPLANS 环境变量控制

---

## 图流程

`
START
  │
  │  route_from_start()
  ├── 有图片 ──────→ image_recognition ─┐
  ├── 需解析别名 ──→ alias_resolve ─────┼→ history_extractor
  └── 可跳过别名 ──→ alias_skip ────────┘
  │
  ▼
history_extractor      ← 从 messages 提取最近 N 轮对话历史
  │
  ▼
context_builder        ← 构建 ConversationContext：追问检测、指代消解、话题推断
  │                       同时生成 history_text（完整版）和 history_text_recent（截断版）
  ▼
planner                ← 4 层路由（Embedding 预过滤 → 缓存 → LLM 分类 → 复杂度分析）
  │                       输出 ExecutionPlan
  │
  │  route_after_planner()
  ├── chat ────────────────────────────────────→ answer（闲聊直达回答）
  │
  └── 其他 ──→ query_processing   ← 查询优化：direct / rewrite / hyde / decompose
                  │
                  ▼
           knowledge_retrieval    ← 知识检索：metadata / semantic / mixed 三路径
                  │
                  │  route_after_retrieval()
                  ├── simple_fact ──→ simple_fact_answer ──→ END（快速通道）
                  ├── 无 Expert  ──→ answer_planner
                  └── 有 Expert  ──→ Expert Dispatcher
                                          ├── parallel=true  → Send(experts) 并行
                                          ├── parallel=false → serial_expert 按顺序循环
                                          └── 单 Expert      → 直接执行
                                                   │
                                                   ▼
                                              merge（仅当前 attempt）
                                                   │
                                                   ▼
                                               evaluator
                                      ┌────────────┼──────────────┐
                                      │ pass       │ replan       │ fallback/exhausted
                                      ▼            ▼              ▼
                              web判断/answer  replanner ─────→ web判断/降级回答
                                                   │
                                                   └→ query_processing（最多一次）
`

chat 与 simple_fact 是显式快速通道，不进入质量闭环。普通查询的 Expert 结果按 xecution_id + attempt 隔离，避免不同请求或重规划轮次的结果污染 Merge。

### Web Trace 面板

**文件**：server.py + static/ + 	race/

基于 FastAPI + SSE（Server-Sent Events）的实时执行追踪面板：
- GET / — Web 面板（聊天气泡 + 流程图）
- POST /chat/stream — SSE 流式推送节点事件 + LLM Token + 回答文本
- GET /chat/stream?query=... — 行为相同的 GET 调试变体
- POST /chat/image — JPEG / PNG / WebP 截图上传 + SSE 输出
- GET /api/models / GET /api/health

**trace/ 模块**：
| 文件 | 职责 |
|------|------|
| collector.py | 单一 stream_events(version="v2") 流收集所有事件 |
| dapter.py | LangGraph 事件 → 前端 TraceEvent 格式适配 |
| models.py | TraceEvent / NodeInfo / NodeRuntime / LLMTrace 类型 |
| pricing.py | DeepSeek Token 计价（.55 / .19 per 1M） |

**启动**：python server.py → http://localhost:9527

---

## 节点详解

### 1. lias_resolve — 别名/实体解析（按需）

**文件**：gents/alias_resolve.py，调用 gents/alias.py + gents/entity_resolver.py

**调用 LLM**：0–2 次（按需，条件触发）

**触发条件**：_route_from_start() 预检——查询中含别名特征（短名称、已知角色/梗名）时路由到 alias_resolve；纯闲聊/精确番剧名查询则跳过。可通过 ENABLE_ALIAS_RESOLVE 配置开关。

**处理流程**：
1. gents/alias.py:resolve_alias() — 先查别名词典，未命中且查询较短时调用 LLM 推断
2. gents/entity_resolver.py:resolve_entity() — 识别角色名、梗名
3. gents/metadata_index.py:fuzzy_lookup() — 精确→去语气词→去前缀管道匹配，命中直接短路（萌蛋呢: 19s → 0.12s）
4. 角色/梗高置信度命中 → 将对应番剧名写入 search_keywords
5. 低置信度或未知实体 → 标记 ntity_confidence，供 Planner 决定是否需要联网

**输出**：
| 字段 | 说明 |
|------|------|
| original_query | 用户原始查询 |
| esolved_query | 别名解析后的查询（若未解析则等于原始） |
| search_keywords | 提取的番剧名列表 |
| ntity_type | 实体类型："alias" / "character" / "meme" / "" |
| ntity_name | 解析出的实体名 |
| ntity_anime | 实体对应的番剧名 |
| ntity_confidence | 置信度 0.0–1.0 |
| ntity_source | 解析来源："dict" / "llm" / "web" |

---

### 2. history_extractor — 对话历史提取

**文件**：gents/history_extractor.py

**调用 LLM**：否（纯函数，零成本）

**输入**：messages（LangGraph 消息列表，含历史轮次）

**处理流程**：遍历 messages，将 HumanMessage 和 AIMessage 按序配对，取最近 config.MEMORY_MAX_ROUNDS（默认 5）轮。

**输出**：{"context": {"history": [{"user": "...", "assistant": "..."}, ...]}}

---

### 3. context_builder — 对话上下文构建

**文件**：gents/context_builder.py

**调用 LLM**：否（纯规则，零成本）

**输入**：messages[-1].content / context.history / ntity_name / ntity_type / ecent_entities / previous_intent

**处理流程**：
1. **追问检测**（_detect_followup）：正则匹配代词开头、追问词（"还有吗""再""继续"）、对比词
2. **指代解析**（_resolve_reference）：序数指代（"第二部的评分"）→ 替换为 ecent_entities[1].name；代词指代（"它的评分"）→ 替换为 ecent_entities[0].name；特殊处理 "那" 字避免误匹配
3. **话题推断**（_infer_topic）：匹配关键词推断当前话题（评分/声优/制作/推荐/对比/闲聊/通用）
4. **history 预拼接**：一次性生成 history_text（完整版，供 planner）和 history_text_recent（截断版=最近 3 轮=6 行，供 answer/simple_fact_answer）

**输出**：{"context": ConversationContext, "resolved_query": "指代消解后的查询"}

---

### 4. planner — 执行计划生成

**文件**：gents/planner.py

**调用 LLM**：1 次 simple_LLM（deepseek-v4-flash），含自动重试；另有 1 次 embedding 计算

**输入**：esolved_query / context.history / context.history_text / ntity_confidence / ntity_type / ntity_source

**4 层路由**：

`
1. Embedding 预过滤 (_prefilter)
   查询 embedding 与 4 类别质心计算余弦相似度
   → 排除低于 EMBEDDING_EXCLUDE_MARGIN 阈值的类别
   → 返回 (best_category, score, all_scores_dict)

2. 分类缓存
   相似历史查询命中 → 直接复用缓存结果（缓存键含 history，追问场景不误命中）

3. LLM 意图分类 (_classify_intent)
   在排除后的候选类别中做精确分类
   → query_category, query_type, experts, parallel, need_web

4. 复杂度分析 (_analyze_complexity)
   LLM 判断查询是否需要多查询扩展
   → 简单查询: direct 策略，跳过 query_processing 的 LLM 调用
   → 复杂查询: rephrase / hyde / decompose
`

**Planner 还会根据实体解析结果调整 plan**：
- ntity_confidence < 0.5 → 
eed_web = True
- ntity_type == "meme" → 
eed_web = True

**缓存**：OrderedDict 真 LRU，move_to_end + popitem(last=False)。_prefilter_cache 让 _should_skip_alias 和 planner._prefilter 共享同一 query 的 embedding 结果。

**输出**（ExecutionPlan）：
`python
{
    "query_type":        "simple_fact | recommendation | comparison | chat",
    "alias_resolved":    False,
    "rewrite_strategy":  "direct | rewrite | hyde | decompose",
    "experts":           ["metadata_reasoner"] | ["similar_expert"] | ["metadata_reasoner", "similar_expert"] | [],
    "parallel":          True | False,
    "query_category":    "metadata | semantic | mixed",
    "need_web":          True | False,
    "reasoning":         "Planner 推理过程简述"
}
`

---

### 5. query_processing — 查询优化

**文件**：gents/query_processor.py

**调用 LLM**：视策略而定（multi_query_rewrite 和 hyde_generate 调 LLM，direct 无）

**输入**：plan.rewrite_strategy / esolved_query

| 策略 | 工具函数 | 适用场景 |
|------|----------|----------|
| direct | 原样返回 | 精确番剧名查询 |
| ewrite | multi_query_rewrite() | 多角度扩展（推荐类） |
| hyde | hyde_generate() | 深度分析/评价类 |
| decompose | decompose() | 含多子问题 |

**输出**：{"shared_context": [查询1, ...], "optimized_queries": [...], "query_strategy": "..."}

---

### 6. knowledge_retrieval — 知识检索

**文件**：gents/retrieval.py（v2.5 拆分自 	ools/query_processing.py）

**调用 LLM**：否（含 rag_optimizer 改写时调 LLM，但检测到上游已优化则跳过）

**输入**：plan.query_category / plan.query_type / esolved_query / shared_context / search_keywords

**三条检索路径**：

| 路径 | 数据源 | 触发条件 |
|------|--------|----------|
| metadata | Metadata Index（本地 JSON） | 公司/声优/年份/评分/标签等结构化查询 |
| semantic | Pinecone（向量）+ Whoosh（稀疏）→ Fusion + Rerank | 相似推荐/评价分析 |
| mixed | 两者全路检索 + 融合 | 标签 + 推荐意图，或默认兜底 |

**检索细节**：
- etrieve_by_keywords()：search_keywords 中的番剧名直接走 Metadata Index 精确查找 + 标签模糊匹配
- _extract_metadata_filters()：提取结构化过滤条件（标签、评分范围、年份）
- etrieve_metadata() / etrieve_semantic()：主检索路径，Semantic 检测上游是否已优化，避免 ag_optimizer 二次改写
- 新增 metadata_fallbacks.py 共享兜底：etch_by_studio_year() / etch_by_title() / ilter_metadata_by_query()

**输出**：{"metadata": [...最多30条], "shared_context": [...最多10条]}

---

### 7. simple_fact_answer — 简单事实快速通道

**文件**：gents/simple_fact_answer.py

**调用 LLM**：是（simple_LLM，轻量模型，一次调用）

**路由条件**：plan.query_type == "simple_fact"

**处理流程**：
1. 优先展示匹配关键词的元数据条目
2. 截断到 5 条，格式化为紧凑文本
3. 追问时注入最近 3 轮对话历史到 system prompt
4. 一次 LLM 调用直接输出回答
5. 更新 ecent_entities（实体追踪）

**注意**：此节点直接连 END，完全跳过 merge → answer_planner → answer 三步流水线。

---

### 8. Expert 节点（metadata_reasoner / similar_expert）

#### metadata_reasoner

**文件**：gents/metadata_reasoner.py

**调用 LLM**：是（默认 nswer_LLM；simple_fact 查询自动切换 simple_LLM）

**职责**：基于结构化元数据（评分/标签/制作/声优等）+ 语义上下文做推理推荐

**输出**：{expert, execution_id, attempt, answer, confidence, evidence}

#### similar_expert

**文件**：gents/similar_expert.py

**调用 LLM**：是（默认 nswer_LLM；simple_fact 查询自动切换 simple_LLM）

**职责**：基于 Embedding 向量 + Metadata Index 发现相似作品，LLM 排序并解释

**工作流**：提取目标番剧 → 同标签/同公司结构相似 → Embedding 语义相似 TopK → 合并去重 → LLM 排序解释

**输出**：{expert, execution_id, attempt, answer, confidence, evidence}

#### 调度执行机制

- 多 Expert 且 plan.parallel == true：为每个计划 Expert 返回一个 Send
- 多 Expert 且 plan.parallel == false：进入 serial_expert 循环，严格按顺序执行
- 串行后一个 Expert 可读取前一个 Expert 的结论，用于核验或补充
- 单 Expert 直接进入对应节点，不受 parallel 值影响
- Expert 输入由 _expert_input 显式构造，不继承完整父 state

---

### 9. merge — 结果合并

**文件**：gents/merge.py

**调用 LLM**：否（纯程序合并，零成本）

**处理流程**：
1. **隔离**：只选择当前 xecution_id + attempt 的 Expert 结果
2. **去重**：基于 answer 5-gram Jaccard 相似度（阈值 0.5）
3. **过滤**：舍弃 confidence < 0.3 的结果
4. **排序**：按置信度降序
5. **格式化**：生成 [Expert N | 置信度: X%]\n... 文本

**输出**：{"merged_results": "合并后的文本"}

---

### 10. valuator — 证据质量评估

**文件**：gents/evaluator.py

**调用 LLM**：通常为 0；仅廉价规则发现潜在结论冲突时调用一次 simple_LLM

**确定性规则**：无有效证据、全部结果低于置信度阈值、计划内 Expert 缺失，以及 comparison/recommendation 证据覆盖不足均返回 eplan。输出为 EvaluationResult {verdict, score, issues, missing_dimensions, feedback}。

**路由规则**：pass 进入联网判断或回答；eplan 且预算未耗尽进入 Replanner；预算耗尽后按需联网，否则附带不确定性回答；allback 直接联网。

---

### 11. eplanner — 受限重规划

**文件**：gents/replanner.py

Replanner 只能修改检索策略、查询类别、Expert 组合、并行模式、联网需求和附加查询。它不允许修改 query_type、原始查询或实体解析结果，也不把重规划结果写入 Planner LRU cache。

每次重规划递增 ttempt，清空当前轮 metadata、shared_context、merged_results 和评估状态。MAX_REPLANS 默认 0（v2.5 默认关闭）。

---

### 12. web_fallback — 联网回退（按需）

**文件**：gents/web_fallback.py

**调用 LLM**：是（simple_LLM 提取关键信息）

**开关控制**：通过 ToolRegistry.is_enabled("search_web") + config.ENABLE_WEB_SEARCH 双重开关，可通过配置关闭整个联网搜索功能。

**触发条件**（should_trigger_web，任一满足）：
1. plan.need_web == True
2. shared_context 为空（检索无结果）
3. 所有 Expert 的 confidence < CONFIDENCE_THRESHOLD（默认 0.5）

---

### 13. nswer_planner — 回答结构规划

**文件**：gents/answer_planner.py

**调用 LLM**：否（随机选择，零成本）

**职责**：为 nswer 节点提供结构指引，避免千篇一律的回答

| query_type | 候选结构 |
|------------|----------|
| ecommendation | top_pick / compare / theme / honest |
| simple_fact | direct / expand |
| comparison | vs / narration |

---

### 14. nswer — 最终回答生成

**文件**：gents/answer.py

**调用 LLM**：是（chat/simple_fact 用 simple_LLM，其他用 nswer_LLM）

**输入**：original_query / plan.query_type / merged_results / nswer_plan.structure / context.history_text_recent

**特殊处理**：
- chat 类：跳过所有分析，直接 simple_LLM.invoke([HumanMessage(content=query)])
- 从 merged_results 正则提取 **粗体** 内的番剧名，写入 ecent_entities
- 同时将 ntity_name（角色/梗名）写入 ecent_entities

---

## 路由函数

### _route_from_start(state) → str

`python
if state.get("image_data"):
    return "image_recognition"
if _should_skip_alias(state):
    return "alias_skip"
return "alias_resolve"
`

### _route_after_planner(state) → str

`python
if plan.query_type == "chat":
    return "answer"          # 闲聊直达回答
return "query_processing"    # 其他走查询优化
`

### _route_after_retrieval(state) → list[Send | str]

`python
if plan.query_type == "simple_fact":
    return "simple_fact_answer"   # 快速通道
if not experts:
    return "answer_planner"       # 无 Expert 直接规划
# 1 个 Expert → 直接返回节点名
# 多 Expert + parallel=true → 多个 Send
# 多 Expert + parallel=false → serial_expert
`

### _route_after_evaluator(state) → str

`python
if verdict == "pass":
    return "web_fallback" if should_trigger_web(state) else "answer_planner"
if verdict == "fallback":
    return "web_fallback"
if attempt < max_replans:
    return "replanner"
return "web_fallback" if should_trigger_web(state) else "answer_planner"
`

---

## AgentState

**文件**：gents/state.py

`python
class AgentState(TypedDict):
    # ── 消息流 ──
    messages:          Annotated[List[BaseMessage], add_messages]

    # ── Planner 输出 ──
    plan:              dict   # ExecutionPlan

    # ── 检索结果 ──
    metadata:          list[dict]          # Metadata Index 结果
    shared_context:    list[str]           # Dense + Sparse 语义文本

    # ── Expert 流水线 ──
    expert_results:    Annotated[list[dict], add]  # 累加保留历史；Merge 按 request/attempt 过滤
    merged_results:    str                # Merge 后综合结果
    execution_id:      str                # 隔离 checkpoint 中不同请求
    attempt:           int                # 当前质量尝试，首次为 0
    max_replans:       int                # 最大重规划次数，默认/上限为 1
    current_expert_index: int             # 串行 Expert 当前下标
    evaluation:        dict               # EvaluationResult
    replan_feedback:   dict               # 附加查询与 plan diff
    execution_mode:    str                # single | parallel | serial
    termination_reason: str               # quality_pass | replan_exhausted | web_fallback
    quality_trace:     Annotated[list[dict], add]

    # ── 查询相关 ──
    original_query:    str                # 用户原始查询
    resolved_query:    str                # 别名 + 指代解析后的查询
    search_keywords:   list[str]          # alias 提取的番剧名
    metadata_cache:    dict               # {name: metadata_dict}
    alias_cache:       dict               # {alias: full_name}
    answer_plan:       dict               # Answer Planner 的结构指引

    # ── 实体解析 ──
    entity_type:       str                # "character" | "meme" | "alias" | ""
    entity_name:       str                # 解析出的实体名
    entity_anime:      str                # 对应番剧名
    entity_confidence: float              # 0.0–1.0
    entity_source:     str                # "dict" | "llm" | "web"

    # ── 对话上下文 ──
    context:           ConversationContext # 当前轮上下文
    recent_entities:   list[dict]         # 持久化: 最近讨论的实体 [{name, type}]
    previous_intent:   str                # 持久化: 上一轮意图

    # ── 截图识别 ──
    image_data:        str                # base64 图片，识别后清空
    image_recognition_result: dict         # 番剧名/集数/时间戳/预览等
`

ttempt 不能单独隔离结果：MemorySaver 下每个新请求都会从 attempt 0 开始，因此必须同时匹配请求级 xecution_id。quality_trace 记录 attempt、实际执行模式、评估分数、问题列表和重规划前后差异；	ermination_reason 区分 quality_pass、eplan_exhausted 与 web_fallback。

---

## ConversationContext

**文件**：gents/state.py

`python
class ConversationContext(TypedDict):
    history:          list[dict]   # 最近 N 轮: [{user: str, assistant: str}]
    history_text:     str          # 完整对话历史文本（供 planner）
    history_text_recent: str       # 截断版（最近3轮），供 answer/simple_fact_answer
    recent_entities:  list[dict]   # 最近讨论的实体: [{name: str, type: str}]
    current_topic:    str          # 当前话题: 评分/声优/制作/推荐/对比/闲聊/通用
    is_followup:      bool         # 是否为追问
    resolved_query:   str          # 指代解析后的查询
    previous_intent:  str          # 上一轮意图: recommend | fact | compare | chat
`

**数据流**：
`
history_extractor → context.history ──┐
alias_resolve     → entity_type/name ─┤
messages[-1]      → query ────────────┤
                                      ├── context_builder ──→ ConversationContext
recent_entities   (上轮持久化) ────────┤
previous_intent   (上轮持久化) ────────┘
`

---

## ToolRegistry

**文件**：	ools/registry.py

统一工具注册表，集中管理所有工具的全生命周期（注册 → 懒加载 → 调用 → 开关控制）。

| 类别 | 工具 | 用途 |
|------|------|------|
| llm_tool | answer_LLM | 主 LLM 实例 |
| llm_tool | simple_LLM | 轻量 LLM 实例 |
| pipeline | retrieve_optimized | RAG 检索优化 |
| pipeline | search_web | Tavily 联网搜索 |
| pipeline | multi_query_rewrite | 多查询改写 |
| pipeline | hyde_generate | HyDE 假设答案生成 |
| pipeline | decompose | 查询分解 |
| pipeline | classify_query | 查询分类 |
| pipeline | rag_search | RAG 搜索 |
| pipeline | metadata_search | 元数据搜索 |
| pipeline | resolve_alias | 别名解析 |
| debug | get_last_debug | 检索调试信息 |

---

## SessionStore

**文件**：gents/session_store.py

SessionStore 为线程安全的会话图实例池，同一 	hread_id 共享 MemorySaver 与已编译图实例。main.py / chat.py / server.py 共用模块级 default_store = SessionStore()。

`python
store = SessionStore()
app = store.get_app(thread_id)       # 获取或创建编译后的 graph
store.clear(thread_id)                # 清空单会话记忆
store.clear_all()                     # 清空全部
`

---

## 记忆系统

### MemorySaver（短期记忆）

**文件**：main.py → gents/session_store.py

- **类型**：内存型 Checkpointer（进程内，重启丢失）
- **存储内容**：每个 	hread_id 的完整 AgentState + messages
- **容量控制**：MEMORY_MAX_ROUNDS = 5（history_extractor 只取最近 5 轮注入上下文；但 messages 全量保留在 checkpointer 中）
- **线程隔离**：不同 	hread_id 的对话互不影响

### 对话上下文层

在 MemorySaver 之上，引入了独立的上下文层：

| 组件 | 作用 | 多轮表现 |
|------|------|----------|
| history_extractor | 提取最近 N 轮配对 | 保证上下文不超长 |
| context_builder | 追问检测 + 指代消解 | "它的评分" → "JOJO的评分" |
| ecent_entities | 跨轮实体追踪 | 支持 "第二部呢" 序号指代 |
| previous_intent | 跨轮意图追踪 | Planner/SimpleFact 消费 |

---

## LLM 健壮性

### Structured Output 降级

**文件**：llms.py → invoke_structured()

部分模型（如 deepseek-v4-flash）不支持 with_structured_output()。降级策略：
1. 首选 llm.with_structured_output(model) 直接获取 Pydantic 对象
2. 失败时自动降级为 _json_fallback_invoke() — 在 prompt 中追加 JSON 格式要求，手动解析
3. 解析失败再抛异常

### 统一重试

**文件**：llms.py → llm_invoke_with_retry() / llm_ainvoke_with_retry()

tenacity 指数退避重试（max 3 次，间隔 1–10s），覆盖所有 13 处 LLM 调用点：
- gents/answer.py / simple_fact_answer.py / metadata_reasoner.py / similar_expert.py / web_fallback.py / lias.py / ntity_resolver.py / query_processor.py / ag_optimizer.py
- llms.py 内部 invoke_structured + JSON fallback

可重试：APIError、APITimeoutError、RateLimitError；不可重试：BadRequestError、Pydantic 校验失败。

---

## 共享 Prompt 组件

**文件**：gents/prompts.py

`python
BANNED_PHRASES = '"推荐理由""综合分析""值得注意的是"...'  # 禁止套话清单
INTERNAL_TERMS = '"元数据""数据库""资料库"...'           # 禁止内部术语
def build_context_section(history_text, is_followup=True) -> str: ...
`

nswer.py 和 simple_fact_answer.py 引用共享组件，消除禁止清单和上下文构建的重复。

---

## 完整数据流示例

以用户查询 "推荐一部类似命运石之门的科幻番" 为例：

`
1. alias_resolve
   检测到 "命运石之门" 是已知番剧名
   → search_keywords: ["命运石之门"]

2. history_extractor
   从 messages 提取最近 5 轮 → context.history

3. context_builder
   无追问 → is_followup=False, resolved_query=原样

4. planner
   规则判断: 含"类似" → semantic 类
   → query_type="recommendation", experts=["similar_expert"]

5. query_processing
   rewrite → multi_query_rewrite(["命运石之门", "科幻 时间旅行", ...])

6. knowledge_retrieval
   mixed: 查 Metadata Index 拿命运石之门元数据 + Pinecone 找语义相似番剧

7. similar_expert
   分析向量检索结果 → LLM 推荐 "Re:0" "夏日重现" "异度侵入"
   → ExpertResult {expert: "similar_expert", execution_id, attempt: 0,
                   answer, confidence: 0.85, evidence}

8. merge
   按 execution_id + attempt 过滤结果；仅 1 个 Expert，直接 format

9. evaluator
   确定性规则检查证据完整性与置信度 → verdict="pass"
   → termination_reason="quality_pass"

10. answer_planner
   recommendation → 随机选 top_pick 结构

11. answer
    输入: merged_results + structure="top_pick"
    → "命运石之门确实是时间旅行题材的标杆……我最想推的是 Re:0……"
`

---

## 项目结构

`
AniGraph/
├── main.py                # 入口：run() / run_stream() + 终端
├── chat.py                # 交互式终端 Chat
├── server.py              # FastAPI Web Trace Server
├── config.py              # 全局配置 + 校验
├── llms.py                # LLM / Embedding 实例
├── static/                # 前端静态文件
│   ├── index.html         # Web Trace 面板
│   ├── app.js             # SSE 客户端 + 聊天 + 流程图
│   └── style.css          # 样式
├── trace/                 # Trace 数据采集模块
│   ├── collector.py       # astream_events 事件收集器
│   ├── adapter.py         # LangGraph 事件 → 前端格式适配
│   ├── models.py          # TraceEvent / NodeInfo 等类型定义
│   └── pricing.py         # DeepSeek Token 计价
├── agents/                # Agent 节点
│   ├── graph.py           # 图定义 + 路由函数
│   ├── state.py           # AgentState / ExecutionPlan / ExpertResult
│   ├── alias_resolve.py   # 别名/实体解析节点
│   ├── alias.py           # 别名词典 + LLM 推断
│   ├── entity_resolver.py # 角色/梗名识别
│   ├── history_extractor.py   # 对话历史提取
│   ├── context_builder.py     # 追问检测 + 指代消解
│   ├── planner.py         # 4 层路由 → ExecutionPlan
│   ├── query_processor.py # 查询优化（direct / rewrite / hyde / decompose）
│   ├── retrieval.py       # 知识检索（metadata / semantic / mixed）
│   ├── metadata_fallbacks.py  # 共享检索兜底（studio_year / title / filter）
│   ├── metadata_reasoner.py   # 元数据推理 Expert
│   ├── similar_expert.py      # 相似推荐 Expert
│   ├── simple_fact_answer.py  # 快速通道回答
│   ├── merge.py           # 结果合并去重
│   ├── evaluator.py       # 证据质量评估
│   ├── replanner.py       # 受限重规划
│   ├── answer_planner.py  # 回答结构规划（零 LLM）
│   ├── answer.py          # 最终回答生成
│   ├── web_fallback.py    # 联网回退
│   ├── image_recognition.py   # 截图识别
│   ├── metadata_index.py      # Metadata Index 局部搜索
│   ├── session_store.py       # 会话记忆管理
│   ├── cache.py           # 缓存工具
│   ├── message_content.py # 消息内容工具
│   └── prompts.py         # 共享 Prompt 组件
├── tools/                 # 检索工具
│   ├── registry.py        # ToolRegistry 统一注册表
│   ├── rag.py             # Pinecone retriever 工厂
│   ├── rag_optimizer.py   # 检索优化
│   ├── query_processing.py    # 查询改写/分解/生成
│   ├── knowledge_retrieval.py # 旧版检索管线（v2.5 前）
│   ├── web_search.py      # Tavily 联网搜索
│   └── image_search.py    # trace.moe API 客户端
├── scripts/               # 辅助脚本
│   ├── audit_seed.py      # 种子数据审计
│   ├── backfill_alias_to_index.py  # 别名回填索引
│   ├── bench_planner.py   # Planner 性能基准
│   ├── fetch_aliases.py   # 别名词典抓取
│   ├── inspect_db.py      # 数据库检查
│   ├── probe_alias_hit.py # 别名命中探查
│   ├── smoke_alias*.py    # 别名冒烟测试
│   └── stream_probe.py    # 流式探查
├── docs/                  # 文档
│   ├── architecture.md    # 架构文档（本文）
│   ├── api.md             # HTTP / Python 接口文档
│   ├── api_reference.md   # 内部 Python 模块参考
│   ├── ablation_report.md # 消融实验报告
│   ├── failure_case.md    # 失败用例分析（多轮）
│   └── failure_case_hard.md # Hard Eval 失败分析
├── tests/                 # 测试与评测
│   ├── test_*.py          # 单元测试
│   ├── ablation.py        # 消融实验框架
│   ├── run_*.py           # 评测运行脚本
│   ├── build_*.py         # 结果构建脚本
│   ├── eval_dataset.json  # 评测数据集
│   ├── eval_hard*.json    # Hard Eval 数据集
│   └── *_results.json     # 评测结果
└── data/                  # 知识库数据（.gitignore）
`

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排框架 | LangGraph 1.2+ |
| 主 LLM | DeepSeek-V4-Pro |
| 轻量 LLM | DeepSeek-V4-Flash |
| Web Server | FastAPI + uvicorn + SSE |
| 向量检索 | Pinecone (similarity / MMR) |
| 稀疏检索 | Whoosh (BM25F) |
| 精排 | bge-reranker-v2-m3 |
| 嵌入模型 | Ark Coding Plan doubao-embedding-vision（默认）/ local Qwen3 / DashScope |
| 结构化索引 | JSON MetadataIndex |
| 联网搜索 | Tavily |
