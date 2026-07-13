"""AniGraph Web Trace Server — FastAPI + SSE 实时 Trace 面板。

启动: python server.py
打开: http://localhost:9527
"""

import json
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from main import run_stream
import config

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="AniGraph Trace", version="1.0")

# 静态文件
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================
# 数据模型
# ============================================================

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


# ============================================================
# 端点
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 Trace 面板首页。"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AniGraph Trace Panel</h1><p>静态文件缺失，请确认 static/index.html 存在。</p>")


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """SSE 端点 — 流式输出 TraceEvent 序列。"""
    async def event_generator():
        try:
            async for event in run_stream(query=body.query, thread_id=body.thread_id):
                # TraceEvent 的所有值都是 JSON 可序列化的
                yield {
                    "event": event["type"],
                    "data": json.dumps(event, ensure_ascii=False),
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False),
            }
        finally:
            yield {"event": "done", "data": ""}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/chat/stream")
async def chat_stream_get(query: str, thread_id: str = "default"):
    """SSE 端点（GET 参数版）— 方便浏览器测试。"""
    return await chat_stream(ChatRequest(query=query, thread_id=thread_id))


@app.get("/api/models")
async def get_models():
    """返回当前 LLM / Embedding 配置。"""
    return JSONResponse({
        "llm_model": config.LLM_MODEL,
        "simple_llm_model": config.SIMPLE_LLM_MODEL,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "embedding_device": config.LOCAL_EMBEDDING_DEVICE,
        "embedding_model": config.LOCAL_EMBEDDING_MODEL,
    })


@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok"})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9527
    uvicorn.run("server:app", host="0.0.0.0", port=port)
