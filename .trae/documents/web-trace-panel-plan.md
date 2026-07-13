# Web Trace 面板 — 实施计划

## 摘要

为 AniGraph 构建实时 Trace 面板：一个 **FastAPI + SSE** 后端驱动、**单文件 HTML/JS** 渲染的 Web 界面。用户输入查询后，通过 SSE 流式展示每个 Agent 节点的执行过程——节点名、状态变化、耗时、并行关系、LLM Token 用量——以瀑布图（Gantt chart）形式呈现。

***

## 当前状态分析

### 已有能力

* `build_graph()` 返回 `StateGraph`，调用方用 `.compile(checkpointer)` 编译

* 编译后的 `CompiledStateGraph` 支持 `astream(stream_mode=["tasks", "updates"])`

* `"tasks"` mode 返回 `TaskPayload`（start）和 `TaskResultPayload`（end），含 `name`、`run_id`、`result`

* `"updates"` mode 返回 `{节点名: 输出 dict}`

* 各节点已有 `logger.info(f"  {节点名} 耗时 {t:.1f}s")` 风格日志（但不完整覆盖）

* `main.py` 和 `chat.py` 是现有调用入口，均使用 `ainvoke`（阻塞等待完整结果）

### 缺失项

* 无任何 Web 框架（FastAPI、uvicorn、SSE 等）

* 无 `astream` 调用代码（全部用 `ainvoke`）

* 无统一 Trace 数据模型

* 无前端代码

* 7 个节点缺少耗时日志（alias\_resolve、history\_extractor、context\_builder、query\_processing、merge、answer\_planner、web\_fallback）

### 关键约束

* 并行 Expert（`Send`）通过 `operator.add` 合并 `expert_results`，但 streaming 事件中各自独立出现

* `build_graph()` 返回未编译图，因此 trace 逻辑放在 **编译后调用方**，不侵入图定义

* Windows 环境，注意 SSE 编码问题

***

## 实施计划

### Step 1: 新增依赖

**文件**: `requirements.txt`

添加：

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sse-starlette>=2.1.0
httpx>=0.27.0                   # async HTTP client，用于 config.validate() 联网测试
```

### Step 2: 创建 Trace 数据模型

**新文件**: `trace/models.py`

定义 Trace 事件的数据结构（不使用 Pydantic，用 `TypedDict` 以减少序列化开销）：

```python
class TraceNode(TypedDict):
    id: str           # run_id，用于前端匹配 start/end
    name: str         # 节点注册名 e.g. "planner"
    display: str      # 显示名 e.g. "规划器"
    start: float      # timestamp
    end: float        # timestamp (0 if still running)
    state_before: dict | None   # 节点输入 state (浅层，不传 messages 全链)
    state_after: dict | None    # 节点输出 state
    llm_calls: list[LLMCall]    # 节点内 LLM 调用记录
    error: str | None

class LLMCall(TypedDict):
    model: str
    input_tokens: int
    output_tokens: int
    cost: str          # "$0.0012"
    elapsed: float
```

维护节点名→显示名映射表：

```python
NODE_DISPLAY = {
    "alias_resolve": "别名/实体解析",
    "history_extractor": "历史提取",
    "context_builder": "上下文构建",
    "planner": "规划器",
    "query_processing": "查询优化",
    "knowledge_retrieval": "知识检索",
    "metadata_reasoner": "元数据推理专家",
    "similar_expert": "相似推荐专家",
    "merge": "结果合并",
    "simple_fact_answer": "简单事实回答",
    "web_fallback": "联网兜底",
    "answer_planner": "回答结构规划",
    "answer": "回答生成",
}
```

### Step 3: 创建 Trace 收集器

**新文件**: `trace/collector.py`

核心类 `TraceCollector`，包装 `astream(stream_mode=["tasks", "updates"])`：

```python
class TraceCollector:
    """收集 ainvoke 过程中的节点事件，产出 TraceNode 列表 + 最终回答。"""
    
    def __init__(self):
        self.nodes: dict[str, TraceNode] = {}  # run_id -> TraceNode
        self.final_answer: str = ""
    
    async def collect(self, app, input_state, config) -> AsyncIterator[dict]:
        """SSE 事件生产者。每次有新事件时 yield 给前端。"""
```

实现要点：

1. 用 `astream` 同时监听 `"tasks"` 和 `"updates"` 两种 mode
2. `TaskPayload` → 创建 TraceNode（`end=0`），记录 `start` 时间，转为 SSE 事件
3. `TaskResultPayload` → 匹配同 `run_id` 的 TraceNode，记录 `end` 时间和 `result`
4. `UpdatesStreamPart` → 提取 `__metadata__`（含 `langgraph_node`），确认路由跳转
5. 用 `astream_events` 的子流（以 `run_id` 过滤）监听 `on_chat_model_end` → 提取 `usage_metadata` 中的 token 数

Token 成本计算（DeepSeek 定价）：

```python
DEEPSEEK_PRICING = {
    "deepseek-v4-pro":   {"input": 0.55, "output": 2.19},   # $/1M tokens
    "deepseek-v4-flash": {"input": 0.14, "output": 0.56},
}
```

### Step 4: 重构 `main.py` 的 `run()` 以支持 streaming

**修改文件**: `main.py`

当前 `run()` 使用 `ainvoke` 返回完整 state。新增一个 `run_stream(query, thread_id)` 函数，返回 `AsyncIterator[SSEEvent]`。原有 `run()` 保持不变（向后兼容）。

```python
async def run(query: str, thread_id: str = "1") -> str:
    """阻塞式调用，向后兼容。"""
    # 不变

async def run_stream(query: str, thread_id: str = "1") -> AsyncIterator[dict]:
    """流式调用，yield SSE 事件 dict。"""
    collector = TraceCollector()
    app = _get_app(thread_id)
    input_state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}}
    async for event in collector.collect(app, input_state, config):
        yield event
```

### Step 5: 创建 FastAPI 服务

**新文件**: `server.py`

端点和功能：

| 端点                                    | 方法   | 功能                |
| ------------------------------------- | ---- | ----------------- |
| `/`                                   | GET  | 返回单文件 HTML 界面     |
| `/chat/stream`                        | POST | SSE 流式 Trace + 回答 |
| `/chat/stream?query=xxx&thread_id=t1` | GET  | 同上，GET 参数版本       |
| `/models`                             | GET  | 返回当前模型配置          |
| `/health`                             | GET  | 存活检测              |

SSE 端点核心逻辑：

```python
@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    async def event_generator():
        async for event in run_stream(body.query, body.thread_id):
            yield {"event": "trace", "data": json.dumps(event, ensure_ascii=False)}
        yield {"event": "done", "data": ""}
    
    return EventSourceResponse(event_generator())
```

注意：

* SSE Response 设置 `Cache-Control: no-cache`、`X-Accel-Buffering: no`

* Windows GBK 问题：所有 JSON 用 `ensure_ascii=False`，Response 设 `Content-Type: text/event-stream; charset=utf-8`

### Step 6: 创建前端 HTML 面板

**内嵌在** **`server.py`** **中**（单文件方案，直接 return HTML 字符串），约 300-400 行 JS。

核心组件：

1. **搜索栏** — 输入框 + 发送按钮 + 会话 ID 显示
2. **回复区** — 打字机效果渐进显示最终回答
3. **Trace 瀑布图** — SVG/Canvas 绘制，纵轴节点列表，横轴时间，bar 宽度 = 耗时

   * 颜色：蓝色=运行中，绿色=完成，红色=错误，灰色=跳过

   * 并行 Expert 的 bar 上下叠放但时间重叠
4. **节点详情面板** — 点击任意节点 bar 展开：输入/输出 state diff、LLM Token 用量、成本
5. **汇总栏** — 总耗时、总 Token 用量、总成本、Graph 路径

不使用任何前端框架，纯原生 JS + CSS Grid（面试展示时无需装依赖）。

### Step 7: 更新入口脚本

**修改文件**: `chat.py`

在 `chat_loop` 中加一个 `/trace` 命令，提示用户打开 `http://localhost:8000` 使用 Web Trace 面板。

**修改文件**: `README.md` 和 `docs/`

添加 Trace 面板使用说明：

```markdown
## Web Trace 面板
python server.py
# 打开 http://localhost:8000
```

***

## 设计决策

1. **不侵入图定义** — Trace 逻辑只在 `server.py` 调用层，不影响现有 `ainvoke` 流程和测试
2. **SSE 而非 WebSocket** — SSE 单向推送足够（服务器→前端），实现更简单，HTTP/1.1 兼容好
3. **单文件 HTML** — 内嵌在 `server.py`，零前端安装，直接 `python server.py` 出结果
4. **向后兼容** — 原有 `main.py` 的 `run()` 和 `chat.py` 不变
5. **Token 成本用 DeepSeek 官方定价** — 硬编码定价表，不做 API 查询（避免额外开销）

## 文件清单

| 操作 | 文件                   | 说明                                        |
| -- | -------------------- | ----------------------------------------- |
| 修改 | `requirements.txt`   | 添加 fastapi, uvicorn, sse-starlette, httpx |
| 新建 | `trace/__init__.py`  | 包初始化                                      |
| 新建 | `trace/models.py`    | TraceNode, LLMCall 数据模型 + 节点映射表           |
| 新建 | `trace/collector.py` | TraceCollector 类，包装 astream               |
| 修改 | `main.py`            | 新增 `run_stream()` 函数                      |
| 新建 | `server.py`          | FastAPI 应用 + 内嵌 HTML 前端                   |
| 修改 | `chat.py`            | 加 `/trace` 命令提示                           |

## 验证步骤

1. `python server.py` → 确认启动无报错
2. 浏览器打开 `http://localhost:8000` → 确认 HTML 正常渲染
3. 输入 "进击的巨人评分多少" → 确认：

   * 回复区渐进显示打字机效果

   * Trace 瀑布图显示所有经过的节点

   * 节点耗时与日志序一致

   * 并行 Expert 时间重叠

   * Token 使用量显示正确
4. 输入 "推荐一部类似命运石之门的科幻番" → 确认复杂查询的完整 Trace
5. `python main.py` → 确认原有 CLI 不受影响
6. `python chat.py` → 确认 `/trace` 提示正常

