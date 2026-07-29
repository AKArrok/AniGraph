"""Answer Planner — 零 LLM 成本随机选回答结构。"""
import random
from agents.state import AgentState


def answer_planner_node(state: AgentState) -> dict:
    """零 LLM 成本的回答结构规划器，随机选结构避免套路化"""
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "recommendation")

    if query_type == "chat":
        return {"answer_plan": {"structure": "简短闲聊"}}

    structures = {
        "recommendation": [
            "top_pick — 先重点安利最推荐的1-2部，多说几句为什么喜欢，后面简略带过",
            "compare — 用对比的方式介绍，突出每部特点，让用户自己选",
            "theme — 按主题/风格归类推荐，先说共同点再展开",
            "honest — 先夸优点再说槽点，显得客观，加一句看你自己口味",
        ],
        "simple_fact": [
            "direct — 直接回答核心问题，顺带讲个相关趣事",
            "expand — 先回答核心问题，再补充1-2个相关维度",
        ],
        "comparison": [
            "vs — 逐项对比，最后一句总结谁更适合什么人",
            "narration — 先分别讲每部特点，最后说更看重X就选A看重Y就选B",
        ],
    }

    options = structures.get(query_type, structures["recommendation"])
    chosen = random.choice(options)

    recommendation_count = state.get("recommendation_count", 0)
    if query_type == "recommendation" and recommendation_count:
        chosen = (
            f"优先推荐约 {recommendation_count} 部作品；每部说明推荐理由；"
            "重点保证相关性，不要堆砌备选"
        )

    return {"answer_plan": {
        "structure": chosen,
        "tone": "casual",
        "recommendation_count": recommendation_count,
    }}
