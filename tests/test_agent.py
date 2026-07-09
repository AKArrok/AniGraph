"""交互式 ACG 番剧推荐问答测试

用法: python tests/test_agent.py
输入 quit / exit / q 退出
"""
import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver


def _get_final_answer(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return m.content.strip()
    return "(未生成回答)"


async def ask(query: str, thread_id: str = "interactive"):
    g, _ = build_graph()
    app = g.compile(checkpointer=MemorySaver())

    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": thread_id}},
    )

    msgs = resp.get("messages", [])
    answer = _get_final_answer(msgs)

    return {
        "answer":            answer,
        "router_decision":   resp.get("router_decision", "?"),
        "dimensions":        resp.get("dimensions", []),
        "processed_dims":    resp.get("processed_dims", []),
    }


def print_result(query: str, result: dict):
    print(f"\n{'─' * 60}")
    print(f"  📋 路由: {result['router_decision']:<10s}  维度: {result['dimensions']}  →  已处理: {result['processed_dims']}")
    print(f"  {'─' * 56}")
    print(f"  {result['answer']}")
    print(f"{'─' * 60}")


async def main():
    print("=" * 60)
    print("  ACG 番剧推荐 — 交互式问答测试")
    print("  输入 quit / exit / q 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        print("⏳ 思考中...", end="\r")
        result = await ask(query)
        print_result(query, result)


if __name__ == "__main__":
    asyncio.run(main())
