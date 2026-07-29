"""Ad-hoc smoke to verify the retrieval fallbacks land under A1."""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(".env"))

os.environ["ABLATION_NO_FAST_PATH"] = "true"

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.graph import build_graph

QUERIES = [
    "《幽灵公主》在 Bangumi 上的评分是多少（精确到小数点后一位）？",
    "《阴阳大战记》的评分是多少？上映年份是哪一年？",
    "在 Bangumi 上，《校园迷糊大王》被观众/编辑打上了哪些标签？请列出主要标签。",
    "ufotable 在 2019 年制作了哪些动画？",
]


async def go() -> None:
    for q in QUERIES:
        app = build_graph().compile(checkpointer=MemorySaver())
        t0 = time.time()
        r = await app.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            {"configurable": {"thread_id": f"smoke_{time.time()}"}},
        )
        lat = time.time() - t0
        ans = r["messages"][-1].content if r.get("messages") else ""
        md_n = len(r.get("metadata", []))
        print()
        print(">>>", q)
        print(f"    lat={lat:.1f}s  metadata_len={md_n}")
        print(f"    A: {ans[:200]}")


if __name__ == "__main__":
    asyncio.run(go())
