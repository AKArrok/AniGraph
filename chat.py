"""AniGraph 交互式对话终端

使用方式:
    python chat.py              # 默认会话
    python chat.py --session my # 指定会话 ID（多会话隔离）
"""

import asyncio
import sys
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph
from trace import TraceCollector
import config


def _truncate(text: str, max_len: int = 72) -> str:
    """截断过长的单行文本，保持终端整洁。"""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


async def chat_loop(app, thread_id: str):
    """主对话循环。"""
    print("=" * 60)
    print("  AniGraph — ACG 番剧推荐与问答")
    print(f"  模型: {config.LLM_MODEL} / {config.SIMPLE_LLM_MODEL}")
    print(f"  会话: {thread_id[:8]}")
    print("=" * 60)
    print("  输入问题开始对话，输入 /exit 退出，/clear 清空记忆，/help 帮助")
    print()

    while True:
        try:
            query = input("\n🐱 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not query:
            continue

        # ── 内置命令 ──────────────────────────────────────────
        if query == "/exit" or query == "/quit":
            print("再见！")
            break
        if query == "/help":
            print("  /exit, /quit   退出对话")
            print("  /clear         清空本轮会话记忆")
            print("  /session       显示当前会话 ID")
            print("  /trace         提示 Web Trace 面板 (先 python server.py)")
            continue
        if query == "/trace":
            print("  Web Trace 面板: 先在新终端运行 python server.py，然后打开 http://localhost:9527")
            continue
        if query == "/session":
            print(f"  当前会话 ID: {thread_id}")
            continue
        if query == "/clear":
            # 新建 MemorySaver 即清空记忆
            nonlocal_memory = MemorySaver()
            app = build_graph().compile(checkpointer=nonlocal_memory)
            print("  会话记忆已清空，新的 MemorySaver 已就绪。")
            continue

        # ── 发送查询 ──────────────────────────────────────────
        print("  ⏳ 思考中 ...", end="\r", flush=True)
        try:
            collector = TraceCollector()
            streamed_text = ""
            answer_started = False
            async for event in collector.collect(
                app,
                {"messages": [HumanMessage(content=query)]},
                {"configurable": {"thread_id": thread_id}},
            ):
                if event["type"] != "answer_chunk":
                    continue
                chunk = event.get("answer_text", "")
                streamed_text += chunk
                if not answer_started:
                    print(" " * 30, end="\r")
                    print("─" * 60)
                    answer_started = True
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n  ❌ 调用失败: {e}")
            continue

        if not streamed_text:
            print("\n  ⚠️ (系统未返回回答)")
            continue

        print()
        print("─" * 60)


def _warmup():
    """预加载模型，避免首查询时才 Loading weights。"""
    print("  预热中 ...", end=" ")
    from llms import warm_embeddings
    warmers = [("Embedding", warm_embeddings)]
    if config.ENABLE_RERANKING:
        from tools.knowledge_retrieval import warm_reranker
        warmers.append(("Reranker", warm_reranker))
    failures = []
    for name, warmer in warmers:
        try:
            warmer()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        print("降级启动")
        for failure in failures:
            print(f"  [警告] {failure}")
    else:
        print("就绪")


def main():
    config.validate()

    # 解析命令行参数
    thread_id = "default"
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            thread_id = sys.argv[idx + 1]

    # 预加载模型（在交互提示出现前完成）
    _warmup()
    print()

    # 构建图（共享 MemorySaver 实现跨轮记忆）
    memory = MemorySaver()
    app = build_graph().compile(checkpointer=memory)

    asyncio.run(chat_loop(app, thread_id))


if __name__ == "__main__":
    main()
