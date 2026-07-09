"""Dimension expert nodes — 各 ACG 维度专属查询重写"""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State
from llms import answer_LLM

_EXPERT_PROMPTS = {
    "rating": (
        "你是评分维度的 ACG 检索专家。"
        "将用户需求重写为一个强调查找【高分/经典/口碑好的番剧】的检索 query。"
        "提取评分要求（如 8分以上/9分以上），优先按评分排序。"
    ),
    "genre": (
        "你是类型维度的 ACG 检索专家。"
        "提取用户提到的所有类型/标签关键词（科幻/热血/治愈/催泪/恋爱/机战/悬疑/日常/搞笑等），"
        "重写为按类型标签匹配的检索 query。"
    ),
    "director": (
        "你是导演维度的 ACG 检索专家。"
        "提取用户提到的导演/监督人名（如庵野秀明/新海诚/新房昭之/渡边信一郎/押井守/石原立也/水岛努），"
        "重写为按导演匹配的检索 query。注意：制作公司不是导演，不要混淆。"
    ),
    "studio": (
        "你是制作公司维度的 ACG 检索专家。"
        "提取用户提到的动画制作公司/工作室/厂牌名称（如京阿尼/京都动画/骨头社/P.A.WORKS/Ghibli/SHAFT/MAPPA），"
        "重写为按制作公司匹配的检索 query。"
    ),
    "writer": (
        "你是编剧/原作维度的 ACG 检索专家。"
        "提取用户提到的编剧/原作/脚本作者/原作者名称（如虚渊玄/麻枝准/大河内一楼/荒川弘/奈须蘑菇），"
        "重写为按编剧或原作匹配的检索 query。"
    ),
    "seiyuu": (
        "你是声优维度的 ACG 检索专家。"
        "提取用户提到的声优名称（如花泽香菜/梶裕贵/钉宫理惠），"
        "重写为按声优/角色匹配的检索 query。"
    ),
    "similar": (
        "你是相似推荐维度的 ACG 检索专家。"
        "提取用户想找类似作品的番剧名称，重写为语义相似度检索 query，"
        "重点查找风格/题材/受众接近的作品。"
    ),
    "general": (
        "保持原始查询不变，直接返回用户的原始问题。"
    ),
}


async def dimension_expert_node(state: State, active_dim: str):
    """指定维度的专家：用专属 System Prompt 重写用户查询"""
    dim = active_dim if active_dim in _EXPERT_PROMPTS else "general"
    expert_prompt = _EXPERT_PROMPTS[dim]
    user_q = state["messages"][-1].content

    resp = await answer_LLM.ainvoke([
        SystemMessage(content=f"{expert_prompt}\n只输出改写后的检索 query，不要解释。"),
        HumanMessage(content=user_q),
    ])

    return {
        "messages": [HumanMessage(content=f"[{dim}] {resp.content}")],
    }
