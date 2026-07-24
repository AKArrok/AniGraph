# AniGraph 内部 Python API 参考

> 本文面向维护者，记录模块、状态和内部函数。HTTP / Python 对外接入方式见 [接口文档](api.md)。

## 入口函数

### `run(query, thread_id, image_data)`

```python
# main.py
import asyncio
from main import run

async def main():
    result = await run("推荐一部类似命运石之门的科幻番", thread_id="user_001")
    print(result)

asyncio.run(main())
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | 必填 | 用户查询文本 |
| `thread_id` | `str` | `"1"` | 对话线程 ID，不同 ID 隔离内存记忆 |
| `image_data` | `str \| None` | `None` | 可选的 base64 图片数据 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 回答 | `str` | LLM 生成的最终自然语言回答 |

**注意**: `main.py` 按 `thread_id` 缓存模块级 `MemorySaver`。同一进程内，相同 ID 自动共享多轮记忆；进程重启后记忆丢失，且多进程部署不会自动共享状态。

### `run_stream(query, thread_id, image_data)` — 流式执行（带 Trace）

```python
# main.py
from main import run_stream

async for evt in run_stream("无职转生的主角是谁？", thread_id="user_001"):
    print(evt)  # TraceEvent dict
```

用于 Web Trace 面板的后端，`server.py` 中的 SSE 端点调用此函数。

---

## 图构建

### `build_graph()`

```python
# graph.py / agents/graph.py
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

g = build_graph()
app = g.compile(checkpointer=MemorySaver())
```

**返回值**: `StateGraph` 实例，已注册所有节点和边。

**节点列表**（按执行顺序）:
| 节点 | 文件 | LLM |
|------|------|:---:|
| `image_recognition` | `agents/image_recognition.py` | 0-1 |
| `alias_resolve` | `agents/graph.py` | 0-2 |
| `alias_skip` | `agents/graph.py` | 0 |
| `history_extractor` | `agents/history_extractor.py` | 0 |
| `context_builder` | `agents/context_builder.py` | 0 |
| `planner` | `agents/planner.py` | 0-1 |
| `query_processing` | `agents/graph.py` | 0-1 |
| `knowledge_retrieval` | `agents/graph.py` | 0 |
| `simple_fact_answer` | `agents/simple_fact_answer.py` | 1 |
| `metadata_reasoner` | `agents/metadata_reasoner.py` | 1 |
| `similar_expert` | `agents/similar_expert.py` | 1 |
| `serial_expert` | `agents/graph.py` | 每轮 1 |
| `merge` | `agents/merge.py` | 0 |
| `evaluator` | `agents/evaluator.py` | 0-1 |
| `replanner` | `agents/replanner.py` | 0 |
| `web_fallback` | `agents/web_fallback.py` | 0-1 |
| `answer_planner` | `agents/graph.py` | 0 |
| `answer` | `agents/answer.py` | 1 |

---

## LLM 实例

```python
# llms.py
from llms import answer_LLM, simple_LLM
```

| 实例 | 模型（.env） | 温度 | 用途 |
|------|------------|:---:|------|
| `answer_LLM` | `LLM_MODEL` | `ANSWER_TEMPERATURE`（默认 0.7） | 主 LLM：Expert、复杂回答 |
| `simple_LLM` | `SIMPLE_LLM_MODEL` | 0.5 | 轻量 LLM：simple_fact、chat 回答 |

`answer_LLM` 的请求超时为 45 秒，`simple_LLM` 为 30 秒；两者的底层客户端 `max_retries=2`，项目调用包装器另提供 tenacity 重试。

---

## Embedding 实例

```python
# llms.py
from llms import embeddings
```

根据 `EMBEDDING_BACKEND` 配置自动选择：

| 后端 | 配置值 | 模型 | 说明 |
|------|--------|------|------|
| Ark Coding Plan | `ark` | `doubao-embedding-vision`（1024 维） | 默认后端，OpenAI 兼容协议 |
| 本地 | `local` | `LOCAL_EMBEDDING_MODEL`（默认 `Qwen/Qwen3-Embedding-0.6B`） | 零 API 成本，CPU 运行 |
| DashScope | `dashscope` | `EMBEDDING_MODELS[0]` | API 模式，配额耗尽自动降级 |

---

## 配置 (config.py)

### LLM 相关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `deepseek-v4-pro` | 主 LLM 模型名 |
| `SIMPLE_LLM_MODEL` | `deepseek-v4-flash` | 轻量 LLM 模型名 |
| `LLM_API_KEY` | - | LLM API Key（必填） |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | API 端点 |

### Embedding 相关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_BACKEND` | `ark` | `ark` / `local` / `dashscope` |
| `LOCAL_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 本地模型路径 |
| `LOCAL_EMBEDDING_DEVICE` | `auto` | 推理设备 |
| `EMBEDDING_MODELS` | `["text-embedding-v4", ...]` | DashScope 模型列表（自动降级，仅 dashscope 后端） |
| `ARK_EMBEDDING_API_KEY` | - | Ark Coding Plan API Key（`EMBEDDING_BACKEND=ark` 时必填） |
| `ARK_EMBEDDING_BASE_URL` | `https://ark.cn-beijing.volces.com/api/coding/v3` | Ark 端点 |
| `ARK_EMBEDDING_MODEL` | `doubao-embedding-vision` | Ark 模型（固定此值） |
| `ARK_EMBEDDING_DIMENSIONS` | `1024` | 输出维度（固定此值） |

### 向量数据库

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PINECONE_API_KEY` | - | Pinecone API Key（必填） |
| `PINECONE_INDEX` | `vector` | Pinecone 索引名 |

### 联网搜索

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TAVILY_API_KEY` | - | Tavily API Key（必填） |

### Agent 调优

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVER_K` | `5` | 最终返回文档数 |
| `RETRIEVER_FETCH_K` | `50` | 粗召回文档数 |
| `HYBRID_DENSE_K` | `20` | Pinecone 密集检索数 |
| `HYBRID_SPARSE_K` | `20` | Whoosh 稀疏检索数 |
| `RERANK_TOP_K` | `10` | CrossEncoder 精排候选数 |
| `EXPERT_TEMPERATURE` | `0.7` | Expert LLM 温度 |
| `ANSWER_TEMPERATURE` | `0.7` | Answer LLM 温度 |
| `CONFIDENCE_THRESHOLD` | `0.5` | Web fallback 触发阈值 |
| `MAX_REPLANS` | `1` | 最大重规划次数；配置值被强制限制在 0-1 |
| `ENABLE_RERANKING` | `true` | 是否启用 CrossEncoder 精排 |

### 短期记忆

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEMORY_MAX_ROUNDS` | `5` | 保留最近 N 轮对话 |

---

## AgentState

完整字段定义见 [agents/state.py](../agents/state.py)：

```python
class AgentState(TypedDict):
    # ── 消息流 ──
    messages: Annotated[List[BaseMessage], add_messages]

    # ── Planner 输出 ──
    plan: dict  # ExecutionPlan

    # ── 检索结果 ──
    metadata: list[dict]
    shared_context: list[str]

    # ── Expert 流水线 ──
    expert_results: Annotated[list[dict], add]  # 保留历史，消费端按 execution_id + attempt 过滤
    merged_results: str
    execution_id: str              # 隔离同一 checkpoint 中的不同请求
    attempt: int                   # 当前质量尝试，首次为 0
    max_replans: int               # 最大重规划次数，默认 1
    current_expert_index: int       # 串行 Expert 下标
    evaluation: dict               # EvaluationResult
    replan_feedback: dict          # 补丁、附加查询和 plan diff
    execution_mode: str            # single | parallel | serial
    termination_reason: str        # quality_pass | replan_exhausted | web_fallback
    quality_trace: Annotated[list[dict], add]

    # ── 识图 (v1.2) ──
    image_data: str                        # base64 图片（仅识图节点消费，识别后清空）
    image_recognition_result: dict          # 识别结果详情（番剧名/集数/时间戳/预览）

    # ── 查询相关 ──
    original_query: str
    resolved_query: str
    search_keywords: list[str]
    metadata_cache: dict
    alias_cache: dict
    answer_plan: dict

    # ── 实体解析 ──
    entity_type: str       # "character" | "meme" | "alias" | ""
    entity_name: str
    entity_anime: str
    entity_confidence: float
    entity_source: str     # "dict" | "llm" | "web"

    # ── 对话上下文 (v1.1) ──
    context: ConversationContext
    recent_entities: list[dict]
    previous_intent: str
```

`attempt` 在每个新请求中从 0 开始，不能独立承担隔离职责。Merge、Evaluator 和 Web fallback 都必须使用 `execution_id + attempt` 选择当前结果。`quality_trace` 保存评估与 plan diff 事件，`termination_reason` 给出 `quality_pass`、`replan_exhausted` 或 `web_fallback`。

## ConversationContext

```python
class ConversationContext(TypedDict):
    history: list[dict]         # [{user: str, assistant: str}]
    recent_entities: list[dict] # [{name: str, type: str}]
    current_topic: str          # 当前话题
    is_followup: bool           # 是否为追问
    resolved_query: str         # 指代解析后的查询
    previous_intent: str        # 上一轮意图
```

---

## 核心 Agent 函数

### Planner

```python
# agents/planner.py
async def planner_node(state: dict) -> dict
def plan(query: str, history_text: str = "") -> dict
```

**输入**: `original_query`, `context.history`, `entity_*`  
**输出**: `ExecutionPlan` — `query_type`, `experts`, `rewrite_strategy`, `parallel`, `need_web` 等。`parallel` 是严格执行协议：多 Expert 时 `true` 使用多个 `Send`，`false` 按 `experts` 顺序串行；单 Expert 不受影响。

### Context Builder

```python
# agents/context_builder.py
async def context_builder_node(state: AgentState) -> dict
```

**输入**: `context.history`, `recent_entities`, `entity_name/type`  
**输出**: `ConversationContext` — 追问检测、指代解析、话题推断

### History Extractor

```python
# agents/history_extractor.py
async def history_extractor_node(state: AgentState) -> dict
```

**输入**: `messages`  
**输出**: `context.history` — 最近 N 轮用户-助手配对

### Simple Fact Answer

```python
# agents/simple_fact_answer.py
async def simple_fact_answer_node(state: dict) -> dict
```

**触发**: `plan.query_type == "simple_fact"`  
**特点**: 单次 LLM 调用直接回答，跳过 Expert → Merge → Answer

### Answer

```python
# agents/answer.py
async def answer_node(state: dict) -> dict
```

**输入**: `merged_results`, `plan.query_type`, `answer_plan.structure`, `context`  
**输出**: `messages`, `recent_entities`, `previous_intent`

### Metadata Reasoner

```python
# agents/metadata_reasoner.py
async def metadata_reasoner_node(state: dict) -> dict
```

**特点**: `simple_fact` 查询自动切换 `simple_LLM`

### Similar Expert

```python
# agents/similar_expert.py
async def similar_expert_node(state: dict) -> dict
```

**特点**: `simple_fact` 查询自动切换 `simple_LLM`

Expert 输出除 `answer`、`confidence`、`evidence` 外，还携带 `expert`、`execution_id` 和 `attempt`。Merge 只消费当前 `execution_id + attempt` 的结果，历史结果仅用于 trace。

### Evaluator

```python
# agents/evaluator.py
async def evaluator_node(state: dict) -> dict
```

确定性检查无证据、全低置信度、计划 Expert 缺失及 comparison/recommendation 证据覆盖不足。只有廉价规则发现潜在结论冲突时才调用 `simple_LLM`。输出 `EvaluationResult {verdict, score, issues, missing_dimensions, feedback}`。

### Replanner

```python
# agents/replanner.py
async def replanner_node(state: dict) -> dict
```

只修正 `rewrite_strategy`、`query_category`、`experts`、`parallel`、`need_web` 和附加查询，不修改原始查询、实体解析结果或 `query_type`。它递增 `attempt`，清理当前轮检索/合并状态，并返回 `query_processing`；补丁不写入 Planner LRU 缓存。

---

## 检索工具

### Knowledge Retrieval

```python
# tools/knowledge_retrieval.py
def fusion(query: str, dense_docs: list, sparse_docs: list, strategy: str) -> list
def rerank(query: str, docs: list[str], top_k: int = 5) -> list[str]
def compress_docs(docs: list[str], query: str, top_k: int = 5) -> list[str]
```

融合策略（RRF / Weighted / Max） + CrossEncoder 精排 + trigram Jaccard 压缩去重。

### RAG 全链路门面

```python
# tools/rag_optimizer.py
# 组合 query_processing + knowledge_retrieval 的全链路入口
def get_last_debug() -> dict
```

查询优化入口：`direct` / `rewrite` / `hyde` / `decompose`

### Web Search

```python
# tools/web_search.py
def web_search(query: str) -> str
```

封装 Tavily 联网搜索。

### Metadata Index

```python
# agents/metadata_index.py
class MetadataIndex:
    def search(self, query: str, ...) -> list[dict]
    def search_by_tags(self, tags: list[str], ...) -> list[dict]
```

结构化元数据索引（JSON/SQLite）。

---

## Web Trace Server

### 启动

```bash
python server.py [port]
# 默认端口 9527
# python server.py 8080  # 自定义端口
```

### SSE 端点: `POST /chat/stream`

实时流式推送节点事件 + LLM Token + 回答文本。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `query` | `str` | 是 | JSON body 中的用户查询 |
| `thread_id` | `str` | 否 | JSON body 中的会话 ID，默认 `"default"` |

**响应**: `text/event-stream`（SSE 格式）

`GET /chat/stream?query=...&thread_id=...` 是行为相同的调试变体。

### 图片端点: `POST /chat/image`

使用 `multipart/form-data` 上传 `file`，可附带 `query` 和 `thread_id`。接受 JPEG、PNG、WebP，返回同样的 SSE 事件流。

**事件类型**:
| event | 说明 |
|-------|------|
| `node_start` | 节点开始执行 |
| `node_end` | 节点执行完毕，含 runtime（耗时、LLM 调用、State 变化） |
| `answer_chunk` | 回答文本片段（流式打字机效果） |
| `done` | 全部完成，含 summary（总耗时、总 Token、图路径） |

### 其他端点

| 端点 | 说明 |
|------|------|
| `GET /` | Web Trace 面板 HTML |
| `GET /static/{file}` | 静态资源（CSS / JS） |
| `GET /api/models` | 返回含 LLM、Embedding 后端、模型、维度及 identity 的对象 |
| `GET /api/health` | 进程存活检查 `{"status": "ok"}`，不探测外部依赖 |

---

## 运行测试

```bash
# 交互式多轮测试（支持短期记忆）
python tests/test_agent.py

# 全链路集成测试
python tests/test_integration.py

# 实体解析测试
python tests/test_entity_resolver.py

# 知识库检查
python tests/check_db.py

# 简单单次查询（run() 入口）
python main.py
```
