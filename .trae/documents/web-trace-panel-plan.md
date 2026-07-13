# Web Trace 面板 — 实施计划（评审修订版）

## 摘要

为 AniGraph 构建实时 Trace 面板：FastAPI + SSE 后端，独立 static/ 前端。用户输入查询后，通过 SSE 流式展示 Agent 节点执行过程——瀑布图、Graph Path、State Diff、Prompt Viewer、Token 用量与成本。

---

## 评审吸收

从评审建议中吸收的关键改进：

| 评审建议 | 采纳 | 实施方式 |
|----------|:----:|----------|
| EventAdapter 解耦 | ✅ | `trace/adapter.py` 将 LangGraph raw events → TraceEvent |
| 数据模型分层 | ✅ | 拆为 NodeInfo / NodeRuntime / LLMTrace / StateDiff |
| State 只传 diff | ✅ | 只发送变化字段，不传 full state（尤其是 messages） |
| TokenProvider 抽象 | ✅ | `trace/pricing.py` 支持 DeepSeek / OpenAI |
| 前端独立文件 | ✅ | `static/index.html` + `style.css` + `app.js` |
| Graph Path | ✅ | 节点列表 + 箭头连线展示执行路径 |
| Prompt Viewer | ✅ | 点击 LLM 节点展开 System/User/最终 Prompt |
| State Diff Viewer | ✅ | before → after，高亮变化字段 |
| 现成框架 | 暂不接入 | 目标仍是自己实现以展示工程能力 |

---

## 当前状态分析

### 已有能力
- `build_graph()` → `StateGraph`，`.compile(checkpointer)` → `CompiledStateGraph`
- `CompiledStateGraph.astream(stream_mode=["tasks", "updates"])` 可获取每个节点的 start/end 事件
- 各节点已有 `logger.info(f"  {name} 耗时 {t:.1f}s")` 风格日志
- `main.py` 和 `chat.py` 使用 `ainvoke`

### 缺失项
- 无 Web 框架 / SSE / 前端
- 无 `astream` 调用代码（全部用 `ainvoke`）
- 无统一 Trace 数据模型

### 关键约束
- Trace 逻辑放在编译后调用方，不侵入图定义
- Windows 环境注意 SSE 编码
- 并行 Expert（Send）各自独立产生 streaming 事件

---

## 实施计划

### Step 1: 新增依赖

**文件**: `requirements.txt`

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sse-starlette>=2.1.0
```

### Step 2: 创建 trace/ 模块

```
trace/
├── __init__.py       # re-export TraceCollector
├── models.py         # 数据模型: NodeInfo, NodeRuntime, LLMTrace, StateDiff, TraceEvent
├── adapter.py        # EventAdapter: LangGraph raw events → TraceEvent
├── pricing.py        # TokenProvider: DeepSeek/OpenAI 定价 + 成本计算
└── collector.py      # TraceCollector: 包装 astream, 协调 adapter + pricing
```

**models.py** — 分层数据模型：

```python
class NodeInfo(TypedDict):     # 静态信息
    name: str                  # e.g. "planner"
    display: str               # e.g. "规划器"

class LLMTrace(TypedDict):     # LLM 调用记录
    model: str
    input_tokens: int
    output_tokens: int
    cost: str                  # "$0.0012"
    elapsed: float
    system_prompt: str | None  # Prompt Viewer 用
    user_prompt: str | None

class StateDiff(TypedDict):    # 状态变化
    added: dict                # 新增/修改的字段
    changed: list[str]         # 变化的键名列表

class NodeRuntime(TypedDict):  # 运行时信息
    start: float
    end: float                 # 0 = running
    state_diff: StateDiff | None
    llm_calls: list[LLMTrace]
    error: str | None

class TraceEvent(TypedDict):   # 发送到前端的统一事件
    type: str                  # "node_start" | "node_end" | "answer_chunk" | "done" | "graph_path"
    node: NodeInfo | None
    runtime: NodeRuntime | None
    answer_text: str           # 打字机流用的增量文本
    graph_path: list[str]      # 最终的执行路径
    summary: dict | None       # 总耗时/总token/总成本
```

**adapter.py** — EventAdapter：

```python
class EventAdapter:
    """将 LangGraph astream task events 转为 TraceEvent。
    解耦 LangGraph 版本，以后换版本只需改此类。
    """
    def adapt_task_start(self, payload: TaskPayload) -> TraceEvent: ...
    def adapt_task_end(self, payload: TaskResultPayload) -> TraceEvent: ...
    def adapt_updates(self, data: dict) -> TraceEvent: ...
    def build_summary(self) -> TraceEvent: ...
```

**pricing.py** — TokenProvider：

```python
class TokenProvider:
    PROVIDERS = {
        "deepseek": {
            "deepseek-v4-pro":   {"input": 0.55, "output": 2.19},
            "deepseek-v4-flash": {"input": 0.14, "output": 0.56},
        }
    }
    @classmethod
    def calculate(cls, model: str, input_tokens: int, output_tokens: int) -> str:
        """返回 "$0.0012" 格式的成本字符串。"""
```

**collector.py** — TraceCollector：

```python
class TraceCollector:
    async def collect(self, app, input_state, config) -> AsyncIterator[TraceEvent]:
        """用 astream(stream_mode=["tasks","updates"]) 收集事件，通过 adapter 转换后 yield。"""
```

实现要点：
1. `TaskPayload` → adapter → `node_start` TraceEvent
2. `TaskResultPayload` → adapter → `node_end` TraceEvent（含 state_diff、llm_calls）
3. `UpdatesStreamPart` → 提取 `__metadata__` 判断路由
4. 并行 Expert 通过时间戳重叠自动检测
5. 最终 answer 从 messages[-1].content 提取，分块发送 `answer_chunk`
6. 结束发送 `done` + summary（总耗时、总token、总成本、graph_path）

### Step 3: main.py 新增 run_stream()

**修改文件**: `main.py`

```python
async def run_stream(query: str, thread_id: str = "1") -> AsyncIterator[dict]:
    """流式调用，yield TraceEvent dict。"""
    collector = TraceCollector()
    app = _get_app(thread_id)
    async for event in collector.collect(app, {"messages": [HumanMessage(content=query)]},
                                          {"configurable": {"thread_id": thread_id}}):
        yield event
```

原 `run()` 不变。

### Step 4: 创建 server.py

**新文件**: `server.py`

端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回 `static/index.html` |
| `/chat/stream` | POST | SSE 流式 Trace + answer |
| `/api/models` | GET | JSON: 当前 LLM/Embedding 配置 |
| `/api/health` | GET | JSON: {"status":"ok"} |

SSE 端点：
```python
@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    async def gen():
        async for evt in run_stream(body.query, body.thread_id):
            yield {"event": evt["type"], "data": json.dumps(evt, ensure_ascii=False)}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(gen())
```

静态文件挂载：`app.mount("/static", StaticFiles(directory="static"), name="static")`

启动入口：
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
```

### Step 5: 创建 static/ 前端

```
static/
├── index.html    # 主页面布局
├── style.css     # 样式
└── app.js        # SSE 连接 + 瀑布图 + 详情面板
```

**页面布局**（左右分栏）：

```
┌────────────────────────────────────────────────────────┐
│  [AniGraph Trace]         模型: deepseek-v4-pro        │
├──────────────────────┬─────────────────────────────────┤
│  搜索栏              │                                 │
│  [______________] 🔍 │         Trace 瀑布图            │
│                      │                                 │
│  回复区 (打字机)     │   alias_resolve ████ 0.1s      │
│  Lorem ipsum...      │   planner         ████ 0.3s    │
│                      │   knowledge_ret.. ████████ 2.1s │
│                      │   metadata_reas.. ██████ 1.5s   │
│                      │   answer          ████ 0.4s     │
├──────────────────────┴─────────────────────────────────┤
│  汇总: 总耗时 4.8s | Tokens: 1,234 in + 567 out | $0.0021 │
│  Path: alias → planner → retrieval → expert → answer     │
└────────────────────────────────────────────────────────┘
```

**核心功能**：
1. **SSE 连接** — `EventSource` 监听 `/chat/stream`
2. **瀑布图** — SVG 绘制，纵轴节点，横轴时间，蓝色 bar 实时增长
3. **点击节点** — 右侧展开详情：State Diff（红色删除/绿色新增）、LLM 调用详情、Prompt（collapsible）
4. **颜色**：蓝色=运行中，绿色=完成，红色=错误，灰色=跳过
5. **汇总栏** — 底部显示总耗时、总 Token、总成本、Graph Path
6. **打字机** — `answer_chunk` 事件逐步追加文本

纯原生 HTML/CSS/JS，零框架依赖。

### Step 6: chat.py 加 /trace 命令

**修改文件**: `chat.py`

在 `/help` 和命令处理中加入：
```
/trace   打开 Web Trace 面板 (http://localhost:8000)
```
提示用户先启动 `python server.py`。

### Step 7: 端到端验证

1. `python server.py` 启动成功
2. 浏览器 `http://localhost:8000` 正常渲染
3. "进击的巨人评分多少" → 瀑布图完整，节点耗时合理
4. "推荐一部类似命运石之门的科幻番" → 复杂查询全链路 Trace
5. 点击节点展开 State Diff / Prompt 正常
6. 汇总栏数据正确

---

## 设计决策

1. **EventAdapter 解耦** — 前端不直接消费 LangGraph events，中间加 adapter
2. **数据模型分层** — NodeInfo / NodeRuntime / LLMTrace / StateDiff 独立
3. **State 只传 diff** — 不传输 full state（尤其是 messages），只传变化键
4. **TokenProvider 抽象** — 定价独立模块，方便切换 LLM 提供商
5. **前端独立文件** — `static/` 三文件，利于维护
6. **不侵入图定义** — Trace 全在调用层
7. **SSE 单向推送** — 足够用，实现简单

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `requirements.txt` | +fastapi, uvicorn, sse-starlette |
| 新建 | `trace/__init__.py` | 包初始化 |
| 新建 | `trace/models.py` | TraceEvent, NodeInfo, NodeRuntime, LLMTrace, StateDiff |
| 新建 | `trace/adapter.py` | EventAdapter: LangGraph events → TraceEvent |
| 新建 | `trace/pricing.py` | TokenProvider: 成本计算 |
| 新建 | `trace/collector.py` | TraceCollector: 协调 astream + adapter |
| 修改 | `main.py` | +run_stream() |
| 新建 | `server.py` | FastAPI app |
| 新建 | `static/index.html` | 页面布局 |
| 新建 | `static/style.css` | 样式 |
| 新建 | `static/app.js` | SSE + 瀑布图 + 详情 |
| 修改 | `chat.py` | +/trace 命令 |
