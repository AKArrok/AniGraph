# 多 Agent 协作架构方案（终版）

> 基于《多Agent架构优化建议》+《多Agent架构进一步优化建议》整合

---

## 本轮新增改进点

| # | 建议 | 改进措施 |
|---|------|---------|
| 1 | Metadata Expert "去 LLM 化" | 建立 **Metadata Index**（SQLite/JSON 结构化索引），LLM 只负责推理和表达，不负责查数据 → 更名 **Metadata Reasoner** |
| 2 | Similar Expert 独立检索 | 用 Metadata Index + Embedding 直接做相似召回，不依赖 shared_context |
| 3 | Retrieval Center 拆两层 | **Query Processing**（Alias/Rewrite/HyDE/Decompose）+ **Knowledge Retrieval**（Metadata Index/Dense/Sparse/Fusion/Rerank） |
| 4 | Planner → ExecutionPlan | 输出完整执行计划（alias/rewrite/retrieval/experts/parallel/need_web），图自动执行 |
| 5 | Web Expert 按需触发 | 默认不联网，仅在 Planner 判断需要 / 检索不足 / confidence < 阈值时触发 |
| 6 | 两级 Metadata Cache | Alias Cache → Metadata Cache，昵称解析后直接命中元数据，无需重复查询 |
| 7 | 预留 Tool Agent | 未来 OCR/翻译/识图等功能统一走 Tool Agent，不新增 Expert |

---

## 核心设计理念

```
动漫推荐系统三要素：
  - 元数据查询（Metadata）  ≈ 数据库能力  → Metadata Index（结构化，零 LLM）
  - 相似推荐（Similar）     ≈ 检索能力    → Embedding + Index（批量召回）
  - 回答组织（Answer）      ≈ LLM 能力    → Answer Agent（自然语言表达）

三者彻底解耦 → 性能、准确率、可维护性全面提升
```

---

## 架构图

```
                                User Query
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │    Alias Tool        │  ← 字典优先 → Metadata Cache
                          │  (非 Agent)          │     未命中 → LLM fallback
                          └──────────┬──────────┘
                                     │ resolved_query
                                     ▼
                          ┌─────────────────────┐
                          │      Planner         │  ← LLM 一次调用，输出完整 ExecutionPlan
                          │   (qwen-max t=0.3)   │
                          └──────────┬──────────┘
                                     │ ExecutionPlan {alias, rewrite, retrieval, experts, parallel, need_web}
                                     ▼
              ═══════════════════════════════════════════════════
              ║            Query Processing Layer               ║  ← Alias 已完成
              ║  Rewrite / HyDE / Decompose / Direct            ║     根据 plan.rewrite 决定策略
              ╚══════════════════════╤═══════════════════════════╝
                                     │ optimized_queries
                                     ▼
              ═══════════════════════════════════════════════════
              ║          Knowledge Retrieval Layer              ║
              ║                                                ║
              ║  ┌──────────────┐  ┌────────┐  ┌────────┐     ║
              ║  │Metadata Index│  │ Dense  │  │ Sparse │     ║  ← 三路并行
              ║  │ (SQLite/JSON)│  │(Pinec) │  │(Whoosh)│     ║
              ║  └──────┬───────┘  └───┬────┘  └───┬────┘     ║
              ║         └──────────────┼───────────┘           ║
              ║                        ▼                       ║
              ║                 Fusion + Rerank                ║
              ╚══════════════════════╤══════════════════════════╝
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │ Metadata Reasoner│    │ Similar Expert   │  ← 2 个 Agent 并行
              │ (LLM 推理+表达)   │    │ (Embedding 召回   │
              │ 输入: 结构化元数据 │    │  + LLM 排序解释)  │
              │ + 语义文本 Context │    │                  │
              └────────┬─────────┘    └────────┬─────────┘
                       │ answer                │ answer
                       │ confidence            │ confidence
                       │ evidence              │ evidence
                       └───────────┬───────────┘
                                   │
                          ┌────────┴────────┐
                          │  confidence < 0.5│──── 是 ──→ Web Expert (按需触发)
                          │  或 结果为空？   │              │
                          └────────┬────────┘              │
                                   │ 否                    │
                                   ▼                       ▼
                          ┌─────────────────────────────────┐
                          │         Program Merge            │  ← 去重 + Reranker 排序
                          │         (零 LLM 成本)            │     过滤低置信度结果
                          └──────────────┬──────────────────┘
                                         │ merged_results
                                         ▼
                          ┌─────────────────────┐
                          │   Answer Agent      │  ← qwen-max (t=0.9)
                          │   综合生成自然回答    │
                          └──────────┬──────────┘
                                     ▼
                                    END
```

---

## Agent 角色（最终 4 个 + 1 预留）

| Agent | 模型 | 输入 | 职责 |
|-------|------|------|------|
| **Planner** | qwen-max (t=0.3) | 用户查询 | 输出 ExecutionPlan（分类/策略/Agent列表/并串行/联网） |
| **Metadata Reasoner** | qwen-max (t=0.7) | 结构化元数据 + 语义 Context | 基于精确数据分析推荐，LLM 只负责推理和语言组织 |
| **Similar Expert** | qwen-max (t=0.7) | Embedding 召回 TopK | 相似作品发现 + 排序 + 解释 |
| **Answer Agent** | qwen-max (t=0.9) | 所有 Expert 结果 | 综合生成口语化回答 |
| **Tool Agent**（预留） | qwen-max (t=0.3) | 工具列表 | 未来 OCR/翻译/识图统一入口 |

> Web Expert 降级为**按需触发的回退节点**，不作为常驻 Agent。

---

## 关键设计细节

### 1. Metadata Index 设计

构建知识库时，同时生成结构化索引：

```python
# data/metadata_index.json（构建时自动生成）
[
  {
    "id": "199630",
    "name_cn": "为美好的世界献上祝福！",
    "name": "この素晴らしい世界に祝福を！",
    "score": 8.1,
    "rank": 245,
    "date": "2016-01-14",
    "eps": 10,
    "tags": ["搞笑", "异世界", "奇幻", "冒险", "轻小说改"],
    "studio": "Studio DEEN",
    "director": "金﨑貴臣",
    "writer": "上江洲誠",
    "seiyuu": ["福島潤", "雨宮天", "高橋李依", "茅野愛衣"],
    "alias": ["素晴", "konosuba", "为美好世界献上祝福"]
  },
  ...
]
```

查询方式：

```python
class MetadataIndex:
    def search(self, **filters) -> list[dict]:
        """支持: tag, studio, director, writer, seiyuu, score_min, date_range"""
        # SQLite WHERE 子句或 JSON 列表推导
        pass
    
    def get_by_alias(self, name: str) -> dict | None:
        """别名精确匹配 → 直接返回元数据"""
        pass
    
    def get_by_id(self, sid: str) -> dict | None:
        """ID 精确查询"""
        pass
```

**收益**：类型/评分/公司/导演/编剧/声优查询 **零 LLM 调用**，毫秒级响应。

### 2. Metadata Reasoner 工作流

```
Metadata Index 查询 → 结构化结果（list of dicts）
                         +
         Dense Context（语义文本片段）
                         ↓
              LLM 推理 + 自然语言组织
                         ↓
            {answer, confidence, evidence}
```

Reasoner 的 prompt 范本：

```
你是 ACG 数据分析师。以下是精确的番剧元数据（结构化）和相关文本上下文。

元数据:
{metadata_json}

上下文:
{context_text}

用户问题: {query}

请基于数据给出推荐，输出 JSON:
{ "answer": "推荐理由", "confidence": 0.85, "evidence": ["依据1", "依据2"] }
```

### 3. Similar Expert 工作流

```
用户查询 → 提取目标番剧 ID
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Metadata Index       Embedding 检索
(同标签/同公司等)     (语义相似 TopK)
    │                   │
    └─────────┬─────────┘
              ▼
        合并去重 → LLM 排序 + 解释
              ↓
    {answer, confidence, evidence}
```

**不依赖 shared_context**，用自己的 Embedding 召回。

### 4. ExecutionPlan 定义

```python
class ExecutionPlan(BaseModel):
    query_type: str      # "simple_fact" | "recommendation" | "comparison" | "chat"
    alias_resolved: bool  # 是否已解析别名
    rewrite_strategy: str  # "direct" | "rewrite" | "hyde" | "decompose"
    experts: list[str]   # ["metadata_reasoner", "similar_expert"]
    parallel: bool       # 是否并行执行 Experts
    need_web: bool       # 是否需要联网
    reasoning: str
```

图根据 `ExecutionPlan` 自动编排，不硬编码边。

### 5. Web 按需触发

```python
def should_trigger_web(state) -> bool:
    plan = state["plan"]
    if plan.get("need_web"):
        return True  # Planner 明确要求
    # 检索结果不足
    if not state.get("shared_context"):
        return True
    # 所有 Expert confidence < 0.5
    results = state.get("expert_results", [])
    if results and all(r.get("confidence", 0) < 0.5 for r in results):
        return True
    return False
```

### 6. 两级 Metadata Cache

```python
class MetadataCache:
    """L1: Alias → L2: Metadata"""
    
    def resolve(self, query: str) -> tuple[str, dict | None]:
        # L1: Alias Cache
        full_name = self.alias_cache.get(query_key)
        if full_name:
            # L2: Metadata Cache
            meta = self.metadata_cache.get(full_name)
            if meta:
                return full_name, meta  # 直接返回，零查询
        return query, None
```

常用番剧的别名和元数据始终在内存中，命中率 > 90%。

### 7. Retrieval Center 两层拆分（重构 tools/rag_optimizer.py）

```
当前 rag_optimizer.py（单文件 400+ 行）→ 拆为:

tools/query_processing.py   # Alias / Rewrite / HyDE / Decompose
tools/knowledge_retrieval.py # Metadata Index / Dense / Sparse / Fusion / Rerank
tools/rag_optimizer.py       # 门面，组合上述两个模块
```

---

## State 定义

```python
class AgentState(TypedDict):
    messages:         Annotated[List[BaseMessage], add_messages]
    plan:             dict              # ExecutionPlan
    metadata:         list[dict]        # Metadata Index 查询结果（结构化）
    shared_context:   list[str]         # Dense + Sparse 语义文本
    expert_results:   list[dict]        # [ExpertResult, ...]
    merged_results:   str
    original_query:   str
    resolved_query:   str              # 别名解析后的查询
    metadata_cache:   dict             # {name: metadata_dict}
    alias_cache:      dict             # {alias: full_name}
```

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新建** | `agents/__init__.py` | 模块入口 |
| **新建** | `agents/state.py` | AgentState + ExecutionPlan + ExpertResult |
| **新建** | `agents/cache.py` | 两级 MetadataCache（Alias + Metadata） |
| **新建** | `agents/alias.py` | Alias 解析工具（字典优先，LLM fallback） |
| **新建** | `agents/planner.py` | Planner Agent → ExecutionPlan |
| **新建** | `agents/metadata_index.py` | Metadata Index（SQLite/JSON 结构化索引） |
| **新建** | `agents/metadata_reasoner.py` | Metadata Reasoner Agent |
| **新建** | `agents/similar_expert.py` | Similar Expert Agent（独立 Embedding 召回） |
| **新建** | `agents/answer.py` | Answer Agent |
| **新建** | `agents/web_fallback.py` | Web 按需触发（非 Agent，回退节点） |
| **新建** | `agents/graph.py` | 多 Agent 图（根据 ExecutionPlan 动态编排） |
| **新建** | `agents/merge.py` | Program Merge（去重 + 排序，零 LLM） |
| **新建** | `tools/query_processing.py` | Query Processing 拆分（从 rag_optimizer 抽离） |
| **新建** | `tools/knowledge_retrieval.py` | Knowledge Retrieval 拆分（从 rag_optimizer 抽离） |
| **修改** | `tools/rag_optimizer.py` | 门面，组合 query_processing + knowledge_retrieval |
| **修改** | `data/build_kb.py` | 构建时同步生成 metadata_index.json |
| **重写** | `graph.py` | `from agents.graph import build_graph` |
| **保留** | `llms.py` | 不变 |
| **保留** | `config.py` | 新增 Metadata Index 路径配置 |
| **可删** | `nodes/domain_router.py` | 被 Planner 替代 |
| **可删** | `nodes/dimension_experts.py` | 被 Metadata Index + Reasoner 替代 |

---

## 对比总结

| 维度 | 初版 | 修订版 | **终版** |
|------|------|--------|---------|
| Agent 数 | 12 | 5 | **4**（Planner + Reasoner + Similar + Answer） |
| 元数据查询 | LLM | LLM | **Metadata Index（SQLite，零 LLM）** |
| 相似推荐 | LLM | LLM 看 shared_context | **Embedding 独立召回 + LLM 排序** |
| RAG 结构 | 单文件 | 共享 Center | **Query Processing + Knowledge Retrieval 两层** |
| 调度 | 固定路由 | Agent 列表 | **ExecutionPlan（图自动编排）** |
| 联网 | 常驻 Agent | Agent | **按需触发（回退节点）** |
| Web Expert | 独立 Agent | 独立 Agent | **按需回退（非 Agent）** |
| 缓存 | 无 | 4 层 | **两级 Metadata Cache（Alias → Metadata）** |
| 扩展性 | 加 Expert | 加 Expert | **Tool Agent 统一入口** |
| LLM 参与度 | 全程 | 大量 | **仅推理和表达，查数据走结构化** |

---

## 验证步骤

1. `python agents/graph.py` — 编译通过
2. `python data/build_kb.py` — 同步生成 `metadata_index.json`
3. `python tests/test_agent.py` — 交互测试，输出显示 ExecutionPlan + Expert 置信度
4. 对比测试：旧架构 vs 新架构，相同问题对比延迟和回答质量
5. 边界测试：纯闲聊 / 简单事实 / 多维度推荐 / 昵称查询 / RAG 空结果 / Web fallback
