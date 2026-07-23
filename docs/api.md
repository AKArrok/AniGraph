# AniGraph 接口文档

> **版本**: v2.3  
> **更新日期**: 2026-07-16  
> **适用范围**: 前后端对接、二次开发、集成接入

AniGraph 提供三层接口：

| 层级 | 用途 | 入口 |
|------|------|------|
| HTTP API | 前端/第三方系统集成 | `server.py` (FastAPI) |
| Python API | 程序化调用、二次开发 | `main.py` (`run` / `run_stream`) |
| 交互式终端 | 人工调试、演示 | `chat.py` |

---

## 一、HTTP API

### 启动服务

```bash
python server.py              # 默认端口 9527
python server.py 8080        # 指定端口
```

服务启动后访问 `http://localhost:9527`。

### 1.1 POST `/chat/stream` - 流式对话（SSE）

主接口，通过 Server-Sent Events 流式返回 Agent 执行过程和回答。

**请求**

```
POST /chat/stream
Content-Type: application/json
```

```json
{
  "query": "进击的巨人评分多少",
  "thread_id": "default"
}
```

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|:---:|------|------|
| `query` | string | ✅ | - | 用户查询文本 |
| `thread_id` | string | ❌ | `"default"` | 会话 ID，同一 ID 共享对话历史 |

**响应**

`Content-Type: text/event-stream; charset=utf-8`

逐个推送 TraceEvent，事件类型见 [三、TraceEvent 结构](#三traceevent-结构)。

```
event: node_start
data: {"type":"node_start","node":{"name":"planner","display":"规划器"},"runtime":{...}}

event: node_end
data: {"type":"node_end","node":{"name":"planner","display":"规划器"},"runtime":{...}}

event: answer_chunk
data: {"type":"answer_chunk","answer_text":"进击的巨人 Bangumi 评分 "}

event: done
data: {"type":"done","graph_path":["alias_resolve","planner","knowledge_retrieval","answer"],"summary":{"elapsed":3.2,"total_tokens":1234,"total_cost":"$0.0012"}}
```

**异常处理**：发生错误时推送 `error` 事件，随后推送 `done` 结束流。

```
event: error
data: {"type":"error","message":"Pinecone 查询失败: ..."}

event: done
data:
```

### 1.2 GET `/chat/stream` - 流式对话（GET 版）

方便浏览器直接测试，参数通过 query string 传递。

```
GET /chat/stream?query=进击的巨人评分&thread_id=default
```

行为同 POST 版本。

### 1.3 GET `/api/models` - 模型配置

返回当前 LLM / Embedding 配置。

```json
{
  "llm_model": "deepseek-v4-pro",
  "simple_llm_model": "deepseek-v4-flash",
  "embedding_backend": "local",
  "embedding_device": "cuda",
  "embedding_model": "BAAI/bge-m3"
}
```

### 1.4 GET `/api/health` - 健康检查

```json
{"status": "ok"}
```

### 1.5 GET `/` - Trace 面板首页

返回 `static/index.html`，即 Web Trace 可视化面板。

---

## 二、Python API

### 2.1 `run()` - 单次查询

```python
from main import run

answer = await run("推荐一部类似命运石之门的科幻番")
print(answer)
# "推荐《RE：0》--同样是异世界题材，主角都面临生死轮回..."
```

**签名**

```python
async def run(query: str, thread_id: str = "1") -> str
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | str | - | 用户查询 |
| `thread_id` | str | `"1"` | 会话 ID，同 ID 共享 MemorySaver 记忆 |

**返回**：`str` - 模型生成的回答文本

**多轮对话**

```python
# 第一轮
await run("进击的巨人评分多少", thread_id="user-123")
# -> "Bangumi 评分 8.6..."

# 第二轮（同一 thread_id，自动衔接上下文）
await run("它有几季？", thread_id="user-123")
# -> "三季，分别是..."
```

### 2.2 `run_stream()` - 流式查询

```python
from main import run_stream

async for event in run_stream("进击的巨人评分多少"):
    print(event["type"], event.get("node", {}).get("display", ""))
```

**签名**

```python
async def run_stream(query: str, thread_id: str = "1") -> AsyncIterator[dict]
```

**Yields**：`dict` - TraceEvent，按时间顺序：
`node_start` -> `node_end` -> ... -> `answer_chunk` -> `done`

**示例：收集完整回答**

```python
answer_parts = []
async for event in run_stream("推荐催泪番"):
    if event["type"] == "answer_chunk":
        answer_parts.append(event["answer_text"])
    elif event["type"] == "done":
        print(f"总耗时: {event['summary']['elapsed']}s")
        print(f"总 Token: {event['summary']['total_tokens']}")
        print(f"总成本: {event['summary']['total_cost']}")
full_answer = "".join(answer_parts)
```

---

## 三、TraceEvent 结构

所有事件共享 `TraceEvent` TypedDict，通过 `type` 字段区分。

### 3.1 通用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | str | 事件类型: `node_start` / `node_end` / `answer_chunk` / `done` / `error` |
| `node` | NodeInfo | 节点标识（`node_start`/`node_end` 携带） |
| `runtime` | NodeRuntime | 节点运行时信息（`node_end` 携带完整数据） |
| `answer_text` | str | 回答增量文本（`answer_chunk` 携带） |
| `graph_path` | list[str] | 最终执行路径（`done` 携带） |
| `summary` | dict | 汇总信息（`done` 携带） |
| `message` | str | 错误信息（`error` 携带） |

### 3.2 NodeInfo

```typescript
{
  "name": "planner",        // 节点注册名
  "display": "规划器"        // 中文显示名
}
```

**节点名映射表**

| name | display |
|------|---------|
| `alias_resolve` | 别名/实体解析 |
| `history_extractor` | 历史提取 |
| `context_builder` | 上下文构建 |
| `planner` | 规划器 |
| `query_processing` | 查询优化 |
| `knowledge_retrieval` | 知识检索 |
| `metadata_reasoner` | 元数据推理专家 |
| `similar_expert` | 相似推荐专家 |
| `merge` | 结果合并 |
| `simple_fact_answer` | 简单事实回答 |
| `web_fallback` | 联网兜底 |
| `answer_planner` | 回答结构规划 |
| `answer` | 回答生成 |

### 3.3 NodeRuntime

```typescript
{
  "start": 1721000000.123,    // time.time() 时间戳
  "end": 1721000001.456,      // 0 = 仍在运行
  "state_diff": {
    "added": {"plan": {...}},  // 新增/修改的字段
    "changed": ["plan"]        // 变化的字段名列表
  },
  "llm_calls": [               // 该节点内的 LLM 调用记录
    {
      "model": "deepseek-v4-flash",
      "input_tokens": 120,
      "output_tokens": 45,
      "cost": "$0.0001",
      "elapsed": 0.8,
      "system_prompt": "...",  // Prompt Viewer 用
      "user_prompt": "..."
    }
  ],
  "error": ""                  // 非空 = 节点异常
}
```

### 3.4 事件类型详解

#### `node_start`

节点开始执行。

```json
{
  "type": "node_start",
  "node": {"name": "planner", "display": "规划器"},
  "runtime": {"start": 1721000000.0, "end": 0, "llm_calls": []}
}
```

#### `node_end`

节点执行完成，携带完整 runtime（含 LLM 调用记录、状态变化）。

```json
{
  "type": "node_end",
  "node": {"name": "planner", "display": "规划器"},
  "runtime": {
    "start": 1721000000.0,
    "end": 1721000001.5,
    "state_diff": {"added": {"plan": {"query_category": "metadata"}}},
    "llm_calls": [{"model": "deepseek-v4-flash", "input_tokens": 120, ...}]
  }
}
```

#### `answer_chunk`

answer 节点的流式输出增量。

```json
{
  "type": "answer_chunk",
  "answer_text": "进击的巨人 Bangumi 评分 "
}
```

#### `done`

整条查询完成，携带最终路径和汇总。

```json
{
  "type": "done",
  "graph_path": ["alias_resolve", "planner", "knowledge_retrieval", "answer"],
  "summary": {
    "elapsed": 3.2,
    "total_tokens": 1234,
    "total_cost": "$0.0012"
  }
}
```

#### `error`

执行过程中发生错误。

```json
{
  "type": "error",
  "message": "Pinecone 查询失败: ConnectionError"
}
```

---

## 四、交互式终端

### 4.1 启动

```bash
python chat.py                    # 默认会话
python chat.py --session my       # 指定会话 ID（多会话隔离）
python main.py                    # 等效入口
```

### 4.2 内置命令

| 命令 | 说明 |
|------|------|
| `/exit` `/quit` | 退出对话 |
| `/clear` | 清空当前会话记忆（重建 MemorySaver） |
| `/session` | 显示当前会话 ID |
| `/trace` | 提示 Web Trace 面板启动方式 |
| `/help` | 显示帮助 |

### 4.3 对话示例

```
======================================================================
  AniGraph - ACG 番剧推荐与问答
  模型: deepseek-v4-pro / deepseek-v4-flash
  会话: default
======================================================================
  输入问题开始对话，输入 /exit 退出，/clear 清空记忆，/help 帮助

🐱 你 > 进击的巨人评分多少
  ⏳ 思考中 ...
                ─────────────────────────────────────────────────────────
Bangumi 评分 8.6，制作公司 WIT STUDIO...
───────────────────────────────────────────────────────────

🐱 你 > 它有几季？
  ⏳ 思考中 ...
                ─────────────────────────────────────────────────────────
三季，分别是...
───────────────────────────────────────────────────────────
```

---

## 五、配置项

通过环境变量或 `.env` 文件配置。

### 5.1 必填

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` | LLM + Embeddings API Key |
| `PINECONE_API_KEY` | Pinecone 向量数据库 API Key |
| `TAVILY_API_KEY` | Tavily 联网搜索 API Key |

### 5.2 模型配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | - | LLM API 地址 |
| `LLM_MODEL` | `deepseek-v4-pro` | 主 LLM（answer/Expert） |
| `SIMPLE_LLM_MODEL` | `deepseek-v4-flash` | 轻量 LLM（planner/simple_fact） |
| `EMBEDDING_BACKEND` | `local` | `local` / `dashscope` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地 embedding 模型 |
| `LOCAL_EMBEDDING_DEVICE` | `cuda` | `cuda` / `cpu` |

### 5.3 功能开关

| 变量 | 默认 | 说明 |
|------|:---:|------|
| `ENABLE_EMBEDDING_PREFILTER` | `true` | Embedding 粗筛开关 |
| `ENABLE_COMPLEXITY_CHECK` | `true` | LLM 复杂度分析开关 |
| `ENABLE_ALIAS_RESOLVE` | `true` | 别名解析按需开关 |
| `ENABLE_WEB_SEARCH` | `true` | Tavily 联网搜索开关 |
| `ENABLE_QUERY_OPTIMIZATION` | `true` | 查询优化开关 |
| `ENABLE_RERANKING` | `true` | CrossEncoder 重排序开关 |
| `ENABLE_COMPRESSION` | `true` | 文档压缩开关 |

### 5.4 检索参数

| 变量 | 默认 | 说明 |
|------|:---:|------|
| `RETRIEVER_K` | `5` | 最终返回文档数 |
| `RETRIEVER_FETCH_K` | `20` | 初始抓取数 |
| `HYBRID_DENSE_K` | `10` | Pinecone 检索数 |
| `HYBRID_SPARSE_K` | `10` | Whoosh 检索数 |
| `FUSION_STRATEGY` | `rrf` | 融合策略: `rrf` / `weighted` / `max` |
| `EMBEDDING_EXCLUDE_MARGIN` | `0.15` | Embedding 粗筛排除阈值 |
| `EMBEDDING_PREFILTER_THRESHOLD` | - | Embedding 预检高置信度阈值 |
| `CONFIDENCE_THRESHOLD` | `0.5` | Web fallback 触发阈值 |

### 5.5 LLM 参数

| 变量 | 默认 | 说明 |
|------|:---:|------|
| `EXPERT_TEMPERATURE` | `0.7` | Expert 节点温度 |
| `ANSWER_TEMPERATURE` | `0.7` | Answer 节点温度 |

---

## 六、前端集成示例

### 6.1 JavaScript (EventSource)

```javascript
const eventSource = new EventSource(
  `/chat/stream?query=${encodeURIComponent("进击的巨人评分")}&thread_id=user-123`
);

eventSource.addEventListener("node_start", (e) => {
  const data = JSON.parse(e.data);
  console.log(`▶ ${data.node.display} 开始`);
});

eventSource.addEventListener("node_end", (e) => {
  const data = JSON.parse(e.data);
  const elapsed = data.runtime.end - data.runtime.start;
  console.log(`✔ ${data.node.display} 完成 (${elapsed.toFixed(1)}s)`);
  if (data.runtime.llm_calls.length > 0) {
    const llm = data.runtime.llm_calls[0];
    console.log(`  LLM: ${llm.model}, tokens=${llm.input_tokens}+${llm.output_tokens}, cost=${llm.cost}`);
  }
});

eventSource.addEventListener("answer_chunk", (e) => {
  const data = JSON.parse(e.data);
  document.getElementById("answer").textContent += data.answer_text;
});

eventSource.addEventListener("done", (e) => {
  const data = JSON.parse(e.data);
  console.log(`路径: ${data.graph_path.join(" -> ")}`);
  console.log(`总耗时: ${data.summary.elapsed}s, 成本: ${data.summary.total_cost}`);
  eventSource.close();
});

eventSource.addEventListener("error", (e) => {
  console.error("错误:", e.data);
});
```

### 6.2 Python (httpx + SSE)

```python
import httpx
import json

async def chat(query: str, thread_id: str = "default"):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:9527/chat/stream",
            json={"query": query, "thread_id": thread_id},
            timeout=120,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    if line[6:]:
                        data = json.loads(line[6:])
                        yield event_type, data
                    event_type = None

# 使用
async for event_type, data in chat("推荐催泪番"):
    if event_type == "answer_chunk":
        print(data["answer_text"], end="", flush=True)
```

---

## 七、错误处理

### 7.1 HTTP 状态码

| 状态码 | 场景 | 说明 |
|--------|------|------|
| 200 | 正常请求 | SSE 流正常开始 |
| 422 | 参数校验失败 | `query` 为空或类型错误 |
| 500 | 服务异常 | 检查 `.env` 配置和 Pinecone 连通性 |

### 7.2 运行时错误

| 错误 | 原因 | 处理 |
|------|------|------|
| `Pinecone 查询失败` | API Key 无效或网络不通 | 检查 `PINECONE_API_KEY` |
| `LLM API 超时` | 请求超 45s | 检查 `LLM_BASE_URL` 连通性，或降低 `RETRIEVER_K` |
| `Embedding 初始化失败` | 模型未下载 | 执行 `python -c "from llms import embeddings"` 预下载 |
| `Whoosh 索引不存在` | 首次运行未建索引 | 执行 `python scripts/build_index.py` |

### 7.3 重试机制

所有 LLM 调用都经过 `llm_invoke_with_retry` / `llm_ainvoke_with_retry` 包装：

- **可重试错误**：`APIError`、`APITimeoutError`、`RateLimitError` - 指数退避重试（max 3 次，间隔 1-10s）
- **不可重试错误**：`BadRequestError`、Pydantic 校验失败 - 直接抛出
- **Structured Output 降级**：模型不支持 `with_structured_output` 时自动切 JSON prompt + 手动解析

---

## 八、性能参考

基于 `deepseek-v4-pro` + `deepseek-v4-flash` + `BAAI/bge-m3`（CUDA）的典型延迟：

| 查询类型 | 典型路径 | LLM 调用 | 延迟 |
|----------|----------|:---:|:---:|
| 闲聊（"你好"） | alias_skip -> planner(embedding拦截) -> answer | 1 | 0.5-1s |
| 简单事实（"巨人评分"） | planner -> retrieval -> simple_fact_answer | 1-2 | 1-2s |
| 元数据查询（"MAPPA作品"） | planner -> retrieval -> metadata_reasoner -> merge -> answer | 3 | 2-4s |
| 语义推荐（"类似巨人的番"） | planner -> retrieval -> similar_expert -> merge -> answer | 3 | 2-4s |
| 复杂对比（"巨人和鬼灭对比"） | planner -> retrieval -> 2 Experts(并行) -> merge -> answer | 4 | 3-5s |
| 复杂分析（"为什么EVA是神作"） | planner -> retrieval(hyde) -> similar_expert -> merge -> answer | 4 | 3-6s |

**成本优化**：
- Embedding 预检拦截闲聊，零 LLM 调用
- Simple Fact 快速通道跳过 Expert 流水线
- Planner 缓存（LRU 500）命中时跳过 LLM 分类
- Expert 并行执行（Send API + async ainvoke）
- ToolRegistry 按需加载，未用工具零开销
