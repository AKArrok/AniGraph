# 动漫识图功能集成方案

> **状态**: ✅ 已实施 — 在 START 后新增 `image_recognition` 节点，trace.moe 优先 + VLM 补充，识别结果合成为自然语言查询写回 `messages`，并复用 `entity_*` 字段进入现有检索流程。

## Summary

在 AniGraph 现有 LangGraph 工作流中集成动漫截图识别能力。用户上传图片后，通过 trace.moe API 识别番剧名、集数、时间戳，**把识别结果合成为一句中文自然语言查询写回 `messages`**，让下游 `context_builder` / `planner` / `knowledge_retrieval` 自然读到番剧名，实体信息走与 `alias_resolve` 完全一致的 `entity_*` 字段（`entity_type="alias"`），后续流程零改动复用。

## Current State Analysis

### 当前流程
```
START -> [alias_resolve | alias_skip] -> history_extractor -> context_builder
  -> planner -> query_processing -> knowledge_retrieval
  -> [metadata_reasoner || similar_expert] -> merge -> [web_fallback]?
  -> answer_planner -> answer -> END
```

### 关键发现
- **入口**: `main.py` 的 `run()` / `run_stream()` 只接受文本 `query: str`，无图片支持
- **Web 层**: `server.py` 的 `ChatRequest` 是纯文本，前端用 `EventSource` GET 请求，无文件上传
- **State**: `AgentState` TypedDict 无图片相关字段
- **别名解析节点** (`_alias_resolve_node`): 已有完整的"输入 -> 番剧名 + entity_* 字段"模式，识图节点复用其输出格式（但 `entity_type` 必须用 `"alias"`，见下）
- **工具注册**: `tools/registry.py` 支持懒加载注册，新增工具只需加一条 `ToolSpec`
- **配置**: `config.py` 用 `os.getenv` 模式，新增配置简单

### 设计决策
- **集成点**: START 之后、alias_resolve 之前，因为识图本质是另一种"实体解析"
- **API 选择**: trace.moe（免费、无需认证、动漫特化、返回番剧名+集数+时间戳）
- **中文标题映射**: trace.moe 返回的是 AniList 英文/罗马音/原文标题，而知识库（`alias.py` 别名表、`metadata_index.json`）是中文键。必须加一层"英文/罗马音 -> 中文正式名"映射，否则 `search_keywords` 拿英文名查中文索引 0 命中
- **低置信度兜底**: trace.moe 识别失败/低置信度/网络异常时，用多模态 LLM（Qwen-VL）描述图片内容作为 fallback；VLM 未配置时降级为"识别失败"提示
- **文件上传**: 用 POST + multipart/form-data 上传图片，**单步**返回 SSE 流（服务端 `EventSourceResponse` 兼容 POST，前端用 `fetch` + `ReadableStream` 读取）

## 关键设计修正（相对初稿）

初稿假设"识图节点写 `resolved_query` = 番剧名，下游零改动复用"。这与现有 graph 运行机制冲突，必须修正：

1. **`resolved_query` 会被 `context_builder` 覆盖**
   `context_builder.py:100,140` 在非追问时 `resolved = query`（`messages[-1].content`），`return {"resolved_query": resolved}` 直接覆盖。`planner.py:573` 优先读 `resolved_query`，`planner.py:609` 还会把 `original_query` 一起覆盖。alias 场景能跑只是因为原始 query 本身含别名文本；**图片场景 query 可能为空，planner 对空串分类会落到 `chat` 直接跳过检索**。
   **修正**: 识图节点把番剧名合成为中文自然语言查询，**追加一条 `HumanMessage`**，使 `messages[-1]` 成为合成查询，`context_builder` / `planner` 自然读到番剧名。

2. **`entity_type` 必须用 `"alias"`**
   `context_builder.py:105` 和 `answer.py:151` 都判断 `entity_type in ("character","alias")` 才做指代解析 / 写 `recent_entities`。初稿的 `"anime_image"` 不被认，多轮"它的评分"会断裂。**修正**: 用 `entity_type="alias"`，真正复用 alias 输出格式。

3. **`image_data` 识别后必须清空**
   base64 原图进 `AgentState` 会被 `MemorySaver` 逐节点持有并进 trace。**修正**: 节点返回 `image_data=None`，不持久化原图。

4. **trace.moe 标题需中文化**（见设计决策 #2）。

## Proposed Changes

### 1. 新增 `tools/image_search.py` -- 识图工具

**文件**: `tools/image_search.py`（新建）

核心函数 `async def search_anime_by_image(image_b64: str) -> dict`:
- 输入: **base64 图片字节**（纯 base64，无 `data:image/...` 前缀；不接受外部 URL，避免 SSRF）
- 调用 trace.moe `POST https://api.trace.moe/search`，body `{"image": image_b64}`，带 `TRACE_MOE_TIMEOUT` 超时 + 1 次重试
- 返回:
  ```python
  {
    "matched": bool,
    "anilist_id": int,
    "title_raw": str,        # trace.moe 原始标题（英文/罗马音/原文）
    "title_cn": str,         # 中文化后的正式番剧名（核心字段）
    "episode": int | None,
    "timestamp": str,        # "12:36"
    "similarity": float,
    "preview_url": str,
    "source": "trace_moe" | "vlm_fallback" | "failed",
  }
  ```
- 高置信度（`>= IMAGE_CONFIDENCE_THRESHOLD`）: 取 `title_raw` 调 `normalize_to_chinese_title()` 得 `title_cn`，`source="trace_moe"`
- 低置信度 / 网络异常: 调 `describe_image_with_vlm()`（VLM 未配置则 `source="failed"`）
- **缓存**: 用 `sha256(image_b64)` 做 key 的模块级 dict 缓存（`async` 函数不能用 `functools.lru_cache`），限制大小 + LRU 淘汰

辅助函数:
- `async def normalize_to_chinese_title(title_raw: str, anilist_id: int) -> str`
  策略: 本地英中映射表（高频番，可扩展）-> 复用 `simple_LLM` 做一次"番剧标题 -> 中文正式名"翻译（`@lru_cache` 缓存）。这是命中中文知识库的关键。
- `async def describe_image_with_vlm(image_b64: str) -> str`
  用 `langchain_openai.ChatOpenAI`（独立 `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL`，OpenAI 兼容协议支持多模态 `content=[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`）。`VLM_API_KEY` 未配置时直接返回空，跳过 fallback。

注册到 `tools/registry.py` 的 `register_default_tools()`:
```python
ToolSpec(
    name="search_anime_by_image",
    description="动漫截图识别: 上传截图返回番剧名/集数/时间戳(中文化标题)",
    category="pipeline",
    import_path="tools.image_search.search_anime_by_image",
    tags=["image", "recognition"],
)
```

### 2. 新增 `agents/image_recognition.py` -- Graph 节点

**文件**: `agents/image_recognition.py`（新建）

`async def image_recognition_node(state: AgentState) -> dict`:
1. 读 `state["image_data"]`（base64），调 `search_anime_by_image()`
2. **合成中文自然语言查询**（让 `context_builder`/`planner` 真正读到番剧名）:
   - 有用户文本 `user_text`: `synth = f"{user_text}（截图识别：《{title_cn}》第{ep}话 {ts}）"`
   - 无用户文本: `synth = f"这是《{title_cn}》的截图，介绍一下这部番"`
   - 识别完全失败: `synth = "这张动漫截图没能识别出来，请补充文字描述"`，`entity_confidence=0`
3. **追加一条 `HumanMessage(synth)`**（`add_messages` reducer 会追加，使 `messages[-1]=synth`；原始空/短 query 被自然取代，下一轮历史提取不受影响）
4. 写入与 `alias_resolve` 一致的实体字段（`entity_type="alias"`）:
   ```python
   return {
       "messages": [HumanMessage(content=synth)],
       "original_query": user_text or synth,   # 会被 planner 再次覆盖成 synth，无妨
       "resolved_query": synth,                # 会被 context_builder 覆盖成 messages[-1]=synth，无妨
       "search_keywords": [title_cn],          # 番剧名走 metadata 检索的唯一可靠通道
       "entity_type": "alias",                 # 下游 context_builder/answer 认
       "entity_name": title_cn,
       "entity_anime": title_cn,
       "entity_confidence": similarity,
       "entity_source": source,               # "trace_moe" | "vlm_fallback"
       "image_recognition_result": {...},      # 完整结果(含集数/时间戳/预览)
       "image_data": None,                    # 清空，不进 checkpoint/trace
       # 预填 metadata（与 alias_resolve 一致，省一次检索）
       # "metadata": [meta] if (meta := metadata_cache.resolve(title_cn)[1]) else []
   }
   ```
   注: `metadata_cache` 预填同 `graph.py:104,119` 的 alias 做法，命中时直接给 `knowledge_retrieval` 喂数据。

### 3. 修改 `agents/state.py` -- 新增字段

在 `AgentState` TypedDict 末尾新增:
```python
image_data: str                    # base64 图片（仅识图节点消费，识别后清空）
image_recognition_result: dict     # 识别结果详情（番剧名/集数/时间戳/预览）
```

### 4. 修改 `agents/graph.py` -- 接入节点

**4a. 新增 import 和节点注册**:
```python
from agents.image_recognition import image_recognition_node

# 在 build_graph() 中:
g.add_node("image_recognition", image_recognition_node)
```

**4b. 修改 `_route_from_start` 路由函数**（带开关 + image 优先）:
```python
def _route_from_start(state: AgentState) -> str:
    if config.ENABLE_IMAGE_RECOGNITION and state.get("image_data"):
        return "image_recognition"
    query = _get_query(state)
    if _should_skip_alias(query):
        return "alias_skip"
    return "alias_resolve"
```

**4c. 修改条件边映射**:
```python
g.add_conditional_edges(START, _route_from_start, {
    "image_recognition": "image_recognition",  # NEW
    "alias_resolve": "alias_resolve",
    "alias_skip": "alias_skip",
})

# 识图完成后进入 history_extractor（和 alias 路径汇合）
g.add_edge("image_recognition", "history_extractor")
```

### 5. 修改 `main.py` -- 支持图片输入

扩展 `run()` 和 `run_stream()` 签名:
```python
async def run(query: str = "", thread_id: str = "1", image_data: str = None) -> str:
    initial_state = {"messages": [HumanMessage(content=query)]}
    if image_data:
        initial_state["image_data"] = image_data   # 识图节点会追加 synth 查询并清空此字段
    resp = await app.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    ...
```
`run_stream` 同理增加 `image_data` 参数透传。

### 6. 修改 `server.py` -- 图片上传端点

新增 `POST /chat/image` 端点（**单步**，统一初稿"两步/一步"的矛盾）:
- 接收 `UploadFile` + `query` (Form, 可空) + `thread_id` (Form)
- 校验类型（jpeg/png/webp）+ 大小（`<= 10MB`），过大用 Pillow 缩放后再 base64
- 调用 `run_stream(query, thread_id, image_data=base64_str)`
- 直接返回 `EventSourceResponse`（SSE 格式与 `/chat/stream` 一致）

```python
from fastapi import UploadFile, File, Form

@app.post("/chat/image")
async def chat_image(file: UploadFile = File(...),
                    query: str = Form(""),
                    thread_id: str = Form("default")):
    image_b64 = await _read_and_validate_image(file)   # 校验+缩放+base64
    async def event_generator():
        try:
            async for event in run_stream(query=query, thread_id=thread_id, image_data=image_b64):
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"type":"error","message":str(e)}, ensure_ascii=False)}
        finally:
            yield {"event": "done", "data": ""}
    return EventSourceResponse(event_generator(), headers={...同 /chat/stream...})
```

### 7. 修改前端 `static/app.js` + `static/index.html`

- HTML: 加图片上传按钮 + 预览区
- JS: 上传图片用 `fetch` POST 到 `/chat/image`（`FormData`），拿到 SSE 流后用 `ReadableStream` 读取（替代 `EventSource`，因为 `EventSource` 只支持 GET）

### 8. 修改 `config.py` -- 新增配置

```python
ENABLE_IMAGE_RECOGNITION = os.getenv("ENABLE_IMAGE_RECOGNITION", "true").lower() == "true"
TRACE_MOE_API_URL = os.getenv("TRACE_MOE_API_URL", "https://api.trace.moe/search")
TRACE_MOE_TIMEOUT = int(os.getenv("TRACE_MOE_TIMEOUT", "15"))
IMAGE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_CONFIDENCE_THRESHOLD", "0.8"))
IMAGE_MAX_SIZE_MB = int(os.getenv("IMAGE_MAX_SIZE_MB", "10"))

# VLM fallback（OpenAI 兼容协议，如 DashScope qwen-vl-max）
VLM_API_KEY  = os.getenv("VLM_API_KEY", "")           # 留空则禁用 VLM fallback
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLM_MODEL    = os.getenv("VLM_MODEL", "qwen-vl-max")
```
注: VLM 与主 LLM（DeepSeek）协议不同，必须独立 key/base_url，不能复用 `LLM_*`。

### 9. 修改 `requirements.txt`

新增:
```
httpx>=0.28.0       # 异步 HTTP（trace.moe API 调用）
Pillow>=11.0.0      # 图片校验/缩放
```
注: `python-multipart` 已在 requirements.txt，FastAPI `UploadFile` 直接可用。

## Assumptions & Decisions

1. **trace.moe 公共 API 配额**: 有按 IP 的并发/配额限制且偶发不稳定，无固定"每日 N 次"数字。初期够用，后续可自建 Docker 版解除限制。
2. **中文标题映射**: trace.moe 返回 AniList 英文/罗马音/原文，知识库是中文键，必须经 `normalize_to_chinese_title()` 映射；本地表 + LLM 翻译兜底。
3. **VLM fallback**: trace.moe 失败时用 Qwen-VL 描述图片；需独立 `VLM_*` 配置，未配置则降级为"识别失败"提示，不阻塞流程。
4. **合成查询而非覆盖字段**: 因 `context_builder`/`planner` 会覆盖 `resolved_query`/`original_query`，识图节点改为追加 `HumanMessage(synth)`，让番剧名经 `messages` 自然流向下游。
5. **`entity_type="alias"`**: 与 alias_resolve 一致，使 `context_builder`/`answer` 的指代解析与 `recent_entities` 正常工作，下游零改动。
6. **图片不进 Pinecone**: 识图只负责"图片 -> 中文番剧名"，后续检索仍走 Pinecone + Whoosh + Metadata Index。
7. **`image_data` 用后即清**: 识别完返回 `image_data=None`，避免 base64 原图进 checkpointer/trace。
8. **仅接受 base64 上传**: 不接受外部 URL，规避 SSRF；服务端校验类型/大小并缩放。
9. **前端 SSE 兼容**: `EventSource` 不支持 POST，改用 `fetch` + `ReadableStream`。

## Verification Steps

1. **单元测试**: `tools/image_search.py` -- mock trace.moe 响应，验证解析 + 中文标题映射逻辑
2. **空 query 纯图测试**: 上传截图、`query=""`，验证 `messages[-1]` 为合成中文查询、planner 未误判 `chat`、`knowledge_retrieval` 正常检索
3. **文本+图片组合测试**: `query="声优是谁"` + 截图，验证合成查询保留用户意图且含番剧名
4. **多轮追问测试**: 识图后追问"它的评分"，验证 `entity_type="alias"` 使指代解析到识别出的番
5. **中英标题映射测试**: 上传英文标题番剧截图，验证 `search_keywords` 为中文名、MetadataIndex 命中
6. **低置信度/非动漫图**: 上传风景图，验证 fallback 到 VLM 描述；VLM 未配置时降级为"识别失败"
7. **trace.moe 宕机/超时**: mock 超时/5xx，验证不抛异常进 graph、降级路径生效
8. **端到端**: 上传截图 -> 识别 -> 正常推荐流程 -> 返回番剧推荐
9. **前端**: 图片上传 -> 预览 -> 发送 -> SSE 流式展示
10. **graph 执行**: 至少跑一次完整 image 路径（非仅 `py_compile`），确认 `image_data` 识别后被清空、不进 trace
