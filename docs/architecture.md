# 🏗️ AniGraph 架构文档

## 概述

AniGraph 是一个基于 LangGraph 的多 Agent 协作 ACG 番剧推荐系统。核心采用 **ExecutionPlan 驱动编排** 的设计：Planner 输出一份执行计划，图引擎根据计划自动选择检索路径、Expert 组合和回答策略。

关键特性：

- **双层意图分类**：Embedding 粗筛（4 类别质心匹配）排除不相关类别 → LLM 精分类（simple_LLM），含复杂度分析
- **按需节点路由**：alias_resolve 通过条件 START 边按需加入；web_fallback 通过 ToolRegistry 开关控制
- **三层检索路径**：Metadata Index（结构化过滤）、Pinecone + Whoosh（向量 + 稀疏检索）、联网回退
- **Simple Fact 快速通道**：简单事实查询跳过 Expert → Merge → Answer 三步流水线，一次 LLM 调用直接作答
- **对话上下文感知**：追问检测、指代消解、多轮话题推断，history 分版（完整版/截断版）
- **ToolRegistry**：统一工具注册表，12 个工具懒加载、集中开关控制
- **LLM 健壮性**：Structured Output 自动降级 + tenacity 指数退避重试
- **Web Trace 面板**：FastAPI + SSE 实时推送执行过程——聊天气泡 + 流程图 + Token 用量

---

## LLM 健壮性

### Structured Output 降级

**文件**：`llms.py` -> `invoke_structured()`

部分模型（如 deepseek-v4-flash）不支持 `with_structured_output()`，返回 `"response_format type is unavailable"`。

**降级策略**：
1. 首选 `llm.with_structured_output(model)` 直接获取 Pydantic 对象
2. 失败时自动降级为 `_json_fallback_invoke()` - 在 prompt 中追加 JSON 格式要求，手动解析 response 内容
3. 解析失败再抛异常

```python
def invoke_structured(llm, output_class, messages, max_retries=3):
    try:
        structured_llm = llm.with_structured_output(output_class)
        # 用 llm_invoke_with_retry 统一包装，保证可重试异常自动重试
        return llm_invoke_with_retry(structured_llm, messages, max_retries=max_retries)
    except Exception as e:
        if "response_format" in str(e).lower() or "unavailable" in str(e).lower():
            return _json_fallback_invoke(llm, output_class, messages, max_retries)
        raise
```

### LLM 重试（v2.3 全覆盖）

**文件**：`llms.py` -> `llm_invoke_with_retry()` / `llm_ainvoke_with_retry()`

使用 tenacity 库实现指数退避重试（max 3 retries），应对 API 瞬时故障：

```python
def llm_invoke_with_retry(llm, messages, max_retries=3):
    return _make_retry(max_retries)(llm.invoke)(messages)

async def llm_ainvoke_with_retry(llm, messages, max_retries=3):
    @_make_retry(max_retries)
    async def _call():
        return await llm.ainvoke(messages)
    return await _call()
```

**v2.3 关键改进**：所有 13 处 `llm.invoke()` 调用点都替换为 `llm_invoke_with_retry` 或 `await llm_ainvoke_with_retry`，覆盖：
- `agents/answer.py` (2处) / `simple_fact_answer.py` (1处)
- `agents/metadata_reasoner.py` (1处) / `similar_expert.py` (1处)
- `agents/web_fallback.py` (1处) / `alias.py` (1处) / `entity_resolver.py` (1处)
- `tools/query_processing.py` (2处) / `rag_optimizer.py` (1处)
- `llms.py` 内部 `invoke_structured` + JSON fallback (3处)

**异步改造**（v2.3）：5 个 async 节点改用 `await llm_ainvoke_with_retry`，Expert 通过 Send API 真正并行执行（原同步 `llm.invoke` 会阻塞事件循环导致串行，延迟 -3s）。

重试条件：`APIError`、`APITimeoutError`、`RateLimitError`；非可重试异常（如 `BadRequestError`）直接抛出。

---

## 图流程

```
START
  │
  │  route_from_start()  ← v2.2: 条件路由
  ├── 需解析别名 ──→ alias_resolve → history_extractor
  └── 无需解析 ──────────────────→ history_extractor
  │
  ▼
history_extractor      ← 从 messages 提取最近 N 轮对话历史
  │
  ▼
context_builder        ← 构建 ConversationContext：追问检测、指代消解、话题推断
  │                       同时生成 history_text（完整版）和 history_text_recent（截断版）
  ▼
planner                ← v2.2: 4 层路由（Embedding预过滤 → 缓存 → LLM分类 → 复杂度分析）
  │                       输出 ExecutionPlan（query_type / experts / rewrite_strategy 等）
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
                  └── 有 Expert  ──→ [metadata_reasoner || similar_expert]（并行 if 2 个）
                                          │
                                          ├──→ merge（去重 + 排序 + 合并）
                                          │        │
                                          │        │  route_after_merge()
                                          │        ├── web_fallback（按需触发联网搜索）
                                          │        │        │
                                          │        └── answer_planner ←┘（回答结构规划）
                                          │                 │
                                          └──→ answer ←────┘（最终回答生成）
                                                   │
                                                  END
```

- **图定义文件**：`agents/graph.py` → `build_graph()`

### Web Trace 面板

**文件**：`server.py` + `static/` + `trace/`

基于 FastAPI + SSE（Server-Sent Events）的实时执行追踪面板：
- `GET /` — Web 面板（聊天气泡 + 流程图）
- `GET /chat/stream?query=...` — SSE 流式推送节点事件 + LLM Token + 回答文本
- `GET /api/models` / `GET /api/health`

**trace/ 模块**：
| 文件 | 职责 |
|------|------|
| `collector.py` | 单一 `astream_events(version="v2")` 流收集所有事件 |
| `adapter.py` | LangGraph 事件 → 前端 TraceEvent 格式适配 |
| `models.py` | TraceEvent / NodeInfo / NodeRuntime / LLMTrace 类型 |
| `pricing.py` | DeepSeek Token 计价（$0.55 / $2.19 per 1M） |

**启动**：`python server.py` → http://localhost:9527

---

## 节点详解

### 1. `alias_resolve` — 别名/实体解析（按需）

**文件**：`agents/graph.py` → `_alias_resolve_node()`

**调用 LLM**：0-2 次（按需，条件触发）

**触发条件**：START 节点的 `_route_from_start()` 预检——查询中包含别名特征（如短名称、已知角色/梗名）时才路由到 alias_resolve；纯闲聊/精确番剧名查询则跳过。可通过 `ENABLE_ALIAS_RESOLVE` 配置开关。

**输入**：
| 字段 | 说明 |
|------|------|
| `messages[-1].content` | 用户当前查询 |

**处理流程**：
1. 调用 `agents/alias.py:resolve_alias()` — 先查别名词典，未命中且查询较短时调用 LLM 推断
2. 调用 `agents/entity_resolver.py:resolve_entity()` — 识别角色名、梗名
3. 高置信度番剧别名命中 → 直接查 Metadata Cache，提前注入 `metadata`
4. 角色/梗高置信度命中 → 将对应番剧名写入 `search_keywords`，供后续检索使用
5. 低置信度或未知实体 → 标记 `entity_confidence`，供 Planner 决定是否需要联网

**输出**：
| 字段 | 说明 |
|------|------|
| `original_query` | 用户原始查询 |
| `resolved_query` | 别名解析后的查询（若未解析则等于原始） |
| `search_keywords` | 提取的番剧名列表 |
| `entity_type` | 实体类型：`"alias"` / `"character"` / `"meme"` / `""` |
| `entity_name` | 解析出的实体名 |
| `entity_anime` | 实体对应的番剧名 |
| `entity_confidence` | 置信度 0.0–1.0 |
| `entity_source` | 解析来源：`"dict"` / `"llm"` / `"web"` |

---

### 2. `history_extractor` — 对话历史提取

**文件**：`agents/history_extractor.py`

**调用 LLM**：否（纯函数，零成本）

**输入**：
| 字段 | 说明 |
|------|------|
| `messages` | LangGraph 消息列表（含历史轮次） |

**处理流程**：
- 遍历 `messages`，将 `HumanMessage` 和 `AIMessage` 按序配对
- 取最近 `config.MEMORY_MAX_ROUNDS`（默认 5）轮

**输出**：
```python
{"context": {"history": [{"user": "...", "assistant": "..."}, ...]}}
```

---

### 3. `context_builder` — 对话上下文构建

**文件**：`agents/context_builder.py`

**调用 LLM**：否（纯规则，零成本）

**输入**：
| 字段 | 说明 |
|------|------|
| `messages[-1].content` | 当前用户查询 |
| `context.history` | history_extractor 输出的历史 |
| `entity_name` / `entity_type` | alias_resolve 的实体结果 |
| `recent_entities` | 上轮持久化的实体列表 |
| `previous_intent` | 上轮意图 |

**处理流程**：
1. **追问检测**（`_detect_followup`）：正则匹配代词开头、追问词（"还有吗""再""继续"）、对比词（使用模块级预编译常量）
2. **指代解析**（`_resolve_reference`）：
   - 序号指代：`"第二部的评分"` → 从预编译的序数词映射表替换为 `recent_entities[1].name` + "的评分"
   - 代词指代：`"它的评分"` → 替换为 `recent_entities[0].name` + "的评分"
   - 特殊处理 `"那"` 字避免误匹配（如 `"那祢豆子呢"` 不解析）
3. **话题推断**（`_infer_topic`）：匹配关键词推断当前话题（评分/声优/制作/推荐/对比/闲聊/通用）
4. **history 预拼接**（v2.2）：一次性生成 `history_text`（完整版，供 planner）和 `history_text_recent`（截断版=最近3轮=6行，供 answer/simple_fact_answer），避免各节点重复拼接，同时防止 answer 节点 token 膨胀

**输出**：
```python
{"context": ConversationContext, "resolved_query": "指代消解后的查询"}
```

---

### 4. `planner` — 执行计划生成

**文件**：`agents/planner.py`

**调用 LLM**：1 次 `simple_LLM`（deepseek-v4-flash），含自动重试；另有 1 次 embedding 计算（本地 CPU，~50ms）

**输入**：
| 字段 | 说明 |
|------|------|
| `resolved_query` | context_builder 指代消解后的查询 |
| `context.history` | 对话历史 |
| `context.history_text` | v2.2: 预拼接的完整对话历史文本 |
| `entity_confidence` / `entity_type` / `entity_source` | 实体解析结果 |

**v2.2 核心逻辑（4 层路由）**：

```
1. Embedding 预过滤 (_prefilter)
   查询 embedding 与 4 类别质心计算余弦相似度
   → 排除低于 EMBEDDING_EXCLUDE_MARGIN 阈值的类别
   → 返回 (best_category, score, all_scores_dict)

2. 分类缓存
   相似历史查询命中 → 直接复用缓存结果

3. LLM 意图分类 (_classify_intent)
   在排除后的候选类别中做精确分类
   → query_category, query_type, experts, parallel, need_web

4. 复杂度分析 (_analyze_complexity, v2.2)
   LLM 判断查询是否需要多查询扩展
   → 简单查询: direct 策略，跳过 query_processing 的 LLM 调用
   → 复杂查询: rephrase / hyde / decompose
```

**Planner 还会根据实体解析结果调整 plan**：
- `entity_confidence < 0.5` → `need_web = True`
- `entity_type == "meme"` → `need_web = True`

**输出**（`ExecutionPlan`）：
```python
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
```

---

### 5. `query_processing` — 查询优化

**文件**：`agents/graph.py` → `_query_processing_node()`

**调用 LLM**：视策略而定（`multi_query_rewrite` 和 `hyde_generate` 调 LLM，`direct` 无）

**输入**：
| 字段 | 说明 |
|------|------|
| `plan.rewrite_strategy` | 优化策略 |
| `resolved_query` | 当前查询 |

**策略对应**：
| 策略 | 工具函数 | 适用场景 |
|------|----------|----------|
| `direct` | 原样返回 | 精确番剧名查询 |
| `rewrite` | `multi_query_rewrite()` | 多角度扩展（推荐类） |
| `hyde` | `hyde_generate()` | 深度分析/评价类 |
| `decompose` | `decompose()` | 含多子问题 |

**输出**：
```python
{"shared_context": [查询1, 查询2, ...], "optimized_queries": [...], "query_strategy": "..."}
```

---

### 6. `knowledge_retrieval` — 知识检索

**文件**：`agents/graph.py` → `_knowledge_retrieval_node()`

**调用 LLM**：否（含 rag_optimizer 改写时调 LLM，但检测到上游已优化则跳过）

**输入**：
| 字段 | 说明 |
|------|------|
| `plan.query_category` | 检索分类 |
| `plan.query_type` | 查询类型 |
| `resolved_query` | 当前查询 |
| `shared_context` | 优化后的 queries |
| `search_keywords` | alias 提取的关键词 |

**三条检索路径**：

| 路径 | 数据源 | 触发条件 |
|------|--------|----------|
| `metadata` | Metadata Index（本地） | 公司/声优/年份/评分/标签等结构化查询 |
| `semantic` | Pinecone（向量）+ Whoosh（稀疏）→ Fusion + Rerank | 相似推荐/评价分析 |
| `mixed` | 两者全路检索 + 融合 | 标签 + 推荐意图，或默认兜底 |

**检索细节**：
- 关键词优先：`search_keywords` 中的番剧名直接走 Metadata Index 精确查找 + 标签模糊匹配
- 提取结构化过滤条件（`_extract_metadata_filters`）：标签、评分范围、年份
- Semantic 检索会检测上游是否已优化，避免 `rag_optimizer` 二次改写

**输出**：
```python
{"metadata": [...最多30条], "shared_context": [...最多10条]}
```

---

### 7. `simple_fact_answer` — 简单事实快速通道

**文件**：`agents/simple_fact_answer.py`

**调用 LLM**：是（`simple_LLM`，轻量模型，一次调用）

**路由条件**：`plan.query_type == "simple_fact"`

**输入**：
| 字段 | 说明 |
|------|------|
| `resolved_query` / `original_query` | 用户查询 |
| `metadata` | 元数据条目 |
| `search_keywords` | 用于优先展示匹配条目 |
| `context.history_text_recent` | v2.2: 预拼接的截断对话历史（最近3轮） |

**处理流程**：
1. 优先展示匹配关键词的元数据条目
2. 截断到 5 条，格式化为紧凑文本
3. 追问时注入最近 3 轮对话历史到 system prompt
4. 一次 LLM 调用直接输出回答
5. 更新 `recent_entities`（实体追踪）

**状态写入**：
```python
{
    "messages": [AIMessage],
    "previous_intent": "simple_fact",
    "recent_entities": [...]
}
```

**注意**：此节点直接连 `END`，完全跳过 `merge → answer_planner → answer` 三步流水线。

---

### 8. Expert 节点（`metadata_reasoner` / `similar_expert`）

#### `metadata_reasoner`（文件：`agents/metadata_reasoner.py`）

**调用 LLM**：是（默认 `answer_LLM`；`simple_fact` 查询自动切换 `simple_LLM`）

**职责**：基于结构化元数据（评分/标签/制作/声优等）+ 语义上下文做推理推荐

**输入**：`metadata`（结构化数据）+ `shared_context`（语义文本）+ `query`

**输出**：`ExpertResult {answer, confidence, evidence}`

#### `similar_expert`（文件：`agents/similar_expert.py`）

**调用 LLM**：是（默认 `answer_LLM`；`simple_fact` 查询自动切换 `simple_LLM`）

**职责**：基于 Embedding 向量 + Metadata Index 发现相似作品，LLM 排序并解释

**工作流**：提取目标番剧 → 同标签/同公司结构相似 → Embedding 语义相似 TopK → 合并去重 → LLM 排序解释

**输出**：`ExpertResult {answer, confidence, evidence}`

#### 并行执行机制

使用 LangGraph `Send` API 实现并行：
- 2 个 Expert 时，`_route_after_retrieval` 返回 `[Send("metadata_reasoner", {...}), Send("similar_expert", {...})]`
- 1 个 Expert 时，直接返回节点名，走正常边
- Expert 需要从 `expert_input` 字典中显式拿到 state 字段（Send 不自动继承父 state）

---

### 9. `merge` — 结果合并

**文件**：`agents/merge.py`

**调用 LLM**：否（纯程序合并，零成本）

**处理流程**：
1. **去重**：基于 answer 5-gram Jaccard 相似度（阈值 0.5）
2. **过滤**：舍弃 confidence < 0.3 的结果
3. **排序**：按置信度降序
4. **格式化**：生成 `[Expert N | 置信度: X%]\n...` 文本

**输出**：`{"merged_results": "合并后的文本"}`

---

### 10. `web_fallback` — 联网回退（按需）

**文件**：`agents/web_fallback.py`

**调用 LLM**：是（`simple_LLM` 提取关键信息）

**开关控制**：通过 `ToolRegistry.is_enabled("search_web")` + `config.ENABLE_WEB_SEARCH` 双重开关，可通过配置关闭整个联网搜索功能。

**触发条件**（`should_trigger_web`，任一满足）：
1. `plan.need_web == True`
2. `shared_context` 为空（检索无结果）
3. 所有 Expert 的 `confidence < CONFIDENCE_THRESHOLD`（默认 0.5）

**处理流程**：联网搜索 → light LLM 提取关键信息 → 追加到 `merged_results`

---

### 11. `answer_planner` — 回答结构规划

**文件**：`agents/graph.py` → `_answer_planner_node()`

**调用 LLM**：否（随机选择，零成本）

**职责**：为 `answer` 节点提供结构指引，避免千篇一律的回答

**策略选择**（按 `query_type` 随机）：
| query_type | 候选结构 |
|------------|----------|
| `recommendation` | top_pick / compare / theme / honest |
| `simple_fact` | direct / expand |
| `comparison` | vs / narration |

---

### 12. `answer` — 最终回答生成

**文件**：`agents/answer.py`

**调用 LLM**：是（`chat`/`simple_fact` 用 `simple_LLM`，其他用 `answer_LLM`）

**输入**：
| 字段 | 说明 |
|------|------|
| `original_query` | 用户原始查询 |
| `plan.query_type` | 查询类型 |
| `merged_results` | merge 后的综合结果 |
| `answer_plan.structure` | 回答结构指引 |
| `context.history_text_recent` | v2.2: 预拼接的截断对话历史（最近3轮，追问时注入） |

**特殊处理**：
- `chat` 类：跳过所有分析，直接 `simple_LLM.invoke([HumanMessage(content=query)])`
- 从 `merged_results` 正则提取 `**粗体**` 内的番剧名，写入 `recent_entities`
- 同时将 `entity_name`（角色/梗名）写入 `recent_entities`

---

## 路由函数

### `_route_from_start(state) → str` (v2.2 新增)

```python
if _should_skip_alias(state):
    return "history_extractor"   # 跳过别名解析
return "alias_resolve"           # 按需加入
```

### `_route_after_planner(state) → str`

```python
if plan.query_type == "chat":
    return "answer"          # 闲聊直达回答
return "query_processing"    # 其他走查询优化
```

### `_route_after_retrieval(state) → list[Send | str]`

```python
if plan.query_type == "simple_fact":
    return "simple_fact_answer"   # 快速通道
if not experts:
    return "answer_planner"       # 无 Expert 直接规划
# 1 个 Expert → 直接返回节点名
# 2 个 Expert → Send API 并行分发
```

### `_route_after_merge(state) → str`

```python
if should_trigger_web(state):
    return "web_fallback"
return "answer_planner"
```

### `_route_after_expert(state) → str`

```python
return "merge"  # Expert 统一进入 merge
```

---

## AgentState

**文件**：`agents/state.py`

```python
class AgentState(TypedDict):
    # ── 消息流 ──
    messages:          Annotated[List[BaseMessage], add_messages]

    # ── Planner 输出 ──
    plan:              dict   # ExecutionPlan

    # ── 检索结果 ──
    metadata:          list[dict]          # Metadata Index 结果
    shared_context:    list[str]           # Dense + Sparse 语义文本

    # ── Expert 流水线 ──
    expert_results:    Annotated[list[dict], add]  # 并行 Expert 累加写入
    merged_results:    str                # Merge 后综合结果

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
```

---

## ConversationContext

**文件**：`agents/state.py`

由 `context_builder` 生成，供 `planner`、`simple_fact_answer`、`answer` 消费：

```python
class ConversationContext(TypedDict):
    history:          list[dict]   # 最近 N 轮: [{user: str, assistant: str}]
    recent_entities:  list[dict]   # 最近讨论的实体: [{name: str, type: str}]
    current_topic:    str          # 当前话题: 评分/声优/制作/推荐/对比/闲聊/通用
    is_followup:      bool         # 是否为追问
    resolved_query:   str          # 指代解析后的查询
    previous_intent:  str          # 上一轮意图: recommend | fact | compare | chat
    history_text:     str          # v2.2: 完整对话历史文本（供 planner）
    history_text_recent: str       # v2.2: 截断版（最近3轮），供 answer/simple_fact_answer
```

**history_text 分版策略**（v2.2）：
- `history_text`：完整版，供 planner 做意图分类和复杂度分析
- `history_text_recent`：截断版（最近3轮 = 6行），防止 answer 节点上下文过长、token 膨胀

**数据流**：
```
history_extractor → context.history ──┐
alias_resolve     → entity_type/name ─┤
messages[-1]      → query ────────────┤
                                      ├── context_builder ──→ ConversationContext
recent_entities   (上轮持久化) ────────┤
previous_intent   (上轮持久化) ────────┘
```

**跨轮持久化**：`recent_entities` 和 `previous_intent` 是 AgentState 顶层字段，由 `simple_fact_answer` 和 `answer` 在每轮末尾写入，下轮的 `context_builder` 消费。

---

## Simple Fact 快速通道

**设计动机**：评分查询、声优查询、身份介绍等简单事实不需要走 `metadata_reasoner → merge → answer` 三步流水线（约 2–3 次 LLM 调用），一次 LLM 调用即可完成。

**触发条件**：`plan.query_type == "simple_fact"`

**流程差异**：
```
普通查询:  ... → retrieval → [experts] → merge → answer_planner → answer → END
快速通道:  ... → retrieval → simple_fact_answer ──────────────────→ END
```

**优点**：
- 减少 2–3 次 LLM 调用，延迟降低 60%+
- 追问时自动注入对话上下文，不丢失多轮能力
- 自动更新 `recent_entities`，不影响指代消解

---

## ToolRegistry（v2.2 新增）

**文件**：`tools/registry.py`

统一工具注册表，集中管理所有工具的全生命周期（注册 → 懒加载 → 调用 → 开关控制）。

### 设计动机

v2.1 中工具散落在各处：`search_web` 在 `tools/web_search.py`，`retrieve_optimized` 在 `tools/rag_optimizer.py`，LLM 实例在 `llms.py`。依赖关系不清晰，且 web_fallback 需要按需控制是否启用 Tavily。

### 核心类

```python
@dataclass
class ToolSpec:
    name: str              # 工具名
    import_path: str       # 懒加载路径 "module.path:function_name"
    category: str          # llm_tool | pipeline | debug
    enabled: bool = True   # 是否启用
    description: str = ""

class ToolRegistry:
    """单例模式，统一管理"""
    _instance = None
    _tools: Dict[str, ToolSpec]
    _cache: Dict[str, Callable]  # 懒加载缓存

    def register(tool: ToolSpec) -> None
    def get_callable(name: str) -> Callable
    def is_enabled(name: str) -> bool
    def get_llm_tools() -> List[ToolSpec]
```

### 工具清单

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

### 使用方式

```python
# 在 graph.py 启动时注册所有工具
from tools import register_default_tools
register_default_tools()

# 按需调用（带懒加载）
from tools import tool_registry
result = tool_registry.get_callable("search_web")(query)

# 开关检查
if tool_registry.is_enabled("search_web"):
    # 触发 web_fallback
```

---

## v2.3 工程化改进

### 函数拆分（可维护性）

**`_knowledge_retrieval_node`** 拆为 3 个辅助函数：
- `_retrieve_by_keywords(keywords)` - 别名关键词优先查 Metadata Index
- `_retrieve_metadata(query, plan, ...)` - 结构化过滤 / 名称 / 标签搜索
- `_retrieve_semantic(search_queries, state)` - Pinecone + Whoosh 混合检索

主函数从 100 行缩减到 35 行，每个辅助函数职责单一、异常处理独立。

**`plan()`** 拆为 2 个路由函数 + 主编排：
- `_route_embedding(query)` - 层1：Embedding 粗筛
- `_route_complexity(query, intent, history_text)` - 层4：复杂度分析路径
- `plan()` 只做编排，~35 行

### 缓存改进

**Planner 缓存真 LRU**：`OrderedDict` + `move_to_end` + `popitem(last=False)`，淘汰死代码 `_strategy_cache`。

**缓存键含 history**：`md5(query|history_text)`，避免追问场景下误命中（同一查询不同历史可能需要不同分类策略）。

**Embedding 预检缓存**：`_prefilter_cache` 让 `_should_skip_alias` 和 `planner._prefilter` 共享同一 query 的 embedding 结果，避免重复计算（~50ms × 2 -> 1 次）。

### 共享 Prompt 组件

**文件**：`agents/prompts.py`

```python
BANNED_PHRASES = '"推荐理由""综合分析""值得注意的是"...'  # 禁止套话清单
INTERNAL_TERMS = '"元数据""数据库""资料库"...'           # 禁止内部术语
def build_context_section(history_text, is_followup=True) -> str: ...
```

`answer.py` 和 `simple_fact_answer.py` 引用共享组件，消除禁止清单和上下文构建的重复。

### 正确性修复

**`_extract_recent_from_merged` 严格过滤**：原 `\*\*(.+?)\*\*` 会误抓 `**评分**`、`**声优**` 等字段标注。新规则：
- 长度 2-15 字符
- 含中文标点（`：:，。、`）跳过
- 排除 25+ 已知字段名
- 全英文/纯数字跳过
- 8 个测试用例验证

**`_retrieve_semantic` already_optimized 修复**：原判断 `query_strategy in ("rewrite", "hyde", "decompose")` 排除了 direct，导致 planner 判 direct 时下游重新调 `classify` 与 planner 决策冲突。改为 `bool(state.get("query_strategy"))`，direct 也跳过。

**web_fallback 异常不污染**：异常时只记 `logger.warning`，不把错误信息追加到 `merged_results`（原行为会让 answer 把错误当正文输出给用户）。

**`recent_entities` 裁剪**：answer/simple_fact_answer 插入新实体后加 `[:5]`，防止长对话累积导致 context_builder prompt 膨胀。

### 配置清理

删除 4 个全项目无引用的配置项：
- `MAX_ITERATIONS`（LangGraph recursion_limit 没用它）
- `ENABLE_VERIFICATION`
- `PLANNER_MODEL`（planner 用 simple_LLM）
- `PLANNER_TEMPERATURE`

### 模块级常量统一

- `_ANIME_TAGS`：两处重复的 tag 列表（16 + 30 个）合并为模块级常量（30 个）
- `_SCORE_RANGE_RE` / `_YEAR_RE`：`_extract_metadata_filters` 的正则预编译
- `_EXTRACT_PROMPT`：web_fallback 的 prompt 从函数内移到模块级

---

## 记忆系统

### MemorySaver（短期记忆）

**文件**：`main.py`

```python
from langgraph.checkpoint.memory import MemorySaver

g.compile(checkpointer=MemorySaver()).ainvoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": thread_id}}
)
```

- **类型**：内存型 Checkpointer（进程内，重启丢失）
- **存储内容**：每个 `thread_id` 的完整 `AgentState` + `messages`
- **容量控制**：`MEMORY_MAX_ROUNDS = 5`（`history_extractor` 只取最近 5 轮注入上下文；但 `messages` 全量保留在 checkpointer 中）
- **线程隔离**：不同 `thread_id` 的对话互不影响

### 对话上下文层（v1.1）

在 MemorySaver 之上，引入了独立的上下文层：

| 组件 | 作用 | 多轮表现 |
|------|------|----------|
| `history_extractor` | 提取最近 N 轮配对 | 保证上下文不超长 |
| `context_builder` | 追问检测 + 指代消解 | `"它的评分" → "JOJO的评分"` |
| `recent_entities` | 跨轮实体追踪 | 支持 `"第二部呢"` 序号指代 |
| `previous_intent` | 跨轮意图追踪 | Planner/SimpleFact 消费 |

---

## 完整数据流示例

以用户查询 `"推荐一部类似命运石之门的科幻番"` 为例：

```
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
   → ExpertResult {answer, confidence: 0.85, evidence}

8. merge
   仅 1 个 Expert，直接 format

9. answer_planner
   recommendation → 随机选 top_pick 结构

10. answer
    输入: merged_results + structure="top_pick"
    → "命运石之门确实是时间旅行题材的标杆……我最想推的是 Re:0……"
```
