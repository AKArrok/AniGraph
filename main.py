"""Entry point."""
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph
import config

config.validate()

async def run(query: str, thread_id: str = "1") -> str:
    g, _ = build_graph()
    return (await g.compile(checkpointer=MemorySaver()).ainvoke(
        {"messages": [HumanMessage(content=query)], "iteration_count": 0},
        config={"configurable": {"thread_id": thread_id}}
    ))["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(run("推荐一部类似命运石之门的科幻番")))
