"""交互式 ACG 番剧推荐问答测试 — 多 Agent 协作版

用法: python tests/test_agent.py
输入 quit / exit / q 退出
"""
import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
# 只对项目自身模块启用调试日志，抑制第三方库（openai/httpcore/httpx等）的DEBUG噪音
logging.basicConfig(level=logging.WARNING, format="%(name)s [%(levelname)s] %(message)s")
logging.getLogger("agents").setLevel(logging.INFO)
logging.getLogger("tools").setLevel(logging.INFO)

from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from tools.rag_optimizer import get_last_debug


def _get_final_answer(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not (hasattr(m, "tool_calls") and m.tool_calls):
            return m.content.strip()
    return "(未生成回答)"


async def ask(query: str, thread_id: str = "interactive"):
    g = build_graph()
    app = g.compile(checkpointer=MemorySaver())

    resp = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        {"configurable": {"thread_id": thread_id}},
    )

    msgs = resp.get("messages", [])
    answer = _get_final_answer(msgs)

    plan = resp.get("plan", {})
    expert_results = resp.get("expert_results", [])
    metadata_count = len(resp.get("metadata", []))
    context_count = len(resp.get("shared_context", []) if isinstance(resp.get("shared_context"), list) else [])

    return {
        "answer": answer,
        "plan": plan,
        "expert_results": expert_results,
        "metadata_count": metadata_count,
        "context_count": context_count,
    }


def _get_queries_display(debug: dict) -> str:
    queries = debug.get("rewritten_queries", [])
    if not queries or len(queries) <= 1:
        return ""
    lines = []
    for q in queries:
        q_short = q[:60] + "..." if len(q) > 60 else q
        lines.append(f"    {q_short}")
    return "\n".join(lines)


def print_result(query: str, result: dict):
    debug = get_last_debug()
    plan = result.get("plan", {})
    expert_results = result.get("expert_results", [])

    # ── Execution Plan ──
    query_type = plan.get("query_type", "?")
    query_category = plan.get("query_category", "?")
    experts = plan.get("experts", [])
    rewrite = plan.get("rewrite_strategy", "?")
    parallel = plan.get("parallel", False)
    need_web = plan.get("need_web", False)
    reasoning = plan.get("reasoning", "")

    # 检索路径图标
    cat_icons = {"metadata": "\U0001f4ca DB", "semantic": "\U0001f50d Vec", "mixed": "\U0001f500 Mix"}
    cat_tag = cat_icons.get(query_category, query_category)

    print(f"\n{'=' * 60}")
    print(f"  \U0001f9e0 多 Agent 协作链路:")

    # Planner 输出
    web_tag = "\U0001f310 联网" if need_web else "\U0001f512 本地"
    par_tag = "\u26a1 并行" if parallel else "\u2192 串行"
    print(f"  \U0001f4cb Planner: {query_type} | {cat_tag} | {rewrite} | {par_tag} | {web_tag}")
    if reasoning:
        print(f"     Reasoning: {reasoning}")
    print(f"     Experts: {experts}")

    # Alias 解析
    nickname = debug.get("nickname_resolved")
    if nickname:
        print(f"  \U0001f310 Alias: {nickname}")

    # ── RAG 优化链路 ──
    print(f"  {'─' * 56}")
    if debug.get("classification"):
        print(f"  \U0001f52c RAG 优化链路:")
        cls = debug.get("classification", "?")
        print(f"     \U0001f4cc 查询分类: {cls}")

        opt = debug.get("optimization", "?")
        print(f"     \U0001f504 查询优化: {opt}")

        qd = _get_queries_display(debug)
        if qd:
            print(f"     \U0001f50d 重写查询 ({debug.get('query_count', 0)} 条):")
            print(qd)

        d_count = debug.get("dense_retrieved", 0)
        s_count = debug.get("sparse_retrieved", 0)
        print(f"     \U0001f4e5 Dense 检索: {d_count} 条 | Sparse(Whoosh): {s_count} 条")

        fus = debug.get("fusion_strategy", "?")
        fus_count = debug.get("post_fusion_count", 0)
        print(f"     \U0001f517 融合策略: {fus}  →  融合后: {fus_count} 条")

        rr = debug.get("reranking", "?")
        rr_count = debug.get("post_rerank_count", 0)
        print(f"     \U0001f3af 精排: {rr}  →  精排后: {rr_count} 条")

        comp = debug.get("compression", "?")
        final_count = debug.get("final_count", 0)
        print(f"     \u2702 压缩: {comp}  →  最终: {final_count} 条")
    else:
        err = debug.get("error", "")
        if err:
            print(f"  \U0001f52c RAG 链路: \u274c 检索失败 — {err}")
        else:
            print(f"  \U0001f52c RAG 链路: (未触发 RAG 检索)")

    # ── Expert 结果 ──
    print(f"  {'─' * 56}")
    if expert_results:
        print(f"  \U0001f393 Expert 分析结果 ({len(expert_results)} 个):")
        for i, er in enumerate(expert_results, 1):
            conf = er.get("confidence", 0)
            conf_icon = "\u2705" if conf >= 0.7 else "\u26a0" if conf >= 0.5 else "\u274c"
            evidence = er.get("evidence", [])
            print(f"     {conf_icon} Expert {i}: 置信度 {conf:.0%}")
            if evidence:
                for ev in evidence[:2]:
                    print(f"        依据: {ev}")
    else:
        print(f"  \U0001f393 Expert: (未触发)")

    # ── Metadata ──
    md_count = result.get("metadata_count", 0)
    ctx_count = result.get("context_count", 0)
    if md_count or ctx_count:
        print(f"  \U0001f4ca Metadata Index: {md_count} 条 | 检索 Context: {ctx_count} 条")

    # ── 回答 ──
    print(f"  {'─' * 56}")
    print(f"  \U0001f4ac 回答:")
    print(f"  {result['answer']}")
    print(f"{'=' * 60}")


async def main():
    print("=" * 60)
    print("  ACG 番剧推荐 — 多 Agent 协作交互测试")
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

        print("\u23f3 思考中...", end="\r")
        result = await ask(query)
        print_result(query, result)


if __name__ == "__main__":
    asyncio.run(main())
