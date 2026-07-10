"""Entry point."""
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph
import config

config.validate()

async def run(query: str, thread_id: str = "1") -> str:
    g = build_graph()
    resp = await g.compile(checkpointer=MemorySaver()).ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    msgs = resp.get("messages", [])
    return msgs[-1].content if msgs else "(无回答)"

if __name__ == "__main__":
    print(asyncio.run(run("推荐一部类似命运石之门的科幻番")))
