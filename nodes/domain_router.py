"""Domain router — 识别用户问题涉及的 ACG 维度（评分/类型/导演/声优/相似推荐）"""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, DimensionDecision
from llms import router_LLM

_DIMENSIONS_MAP = {
    "rating":   "评分/高分/排名/TOP/神作/经典/口碑 相关查询",
    "genre":    "类型/标签/风格（科幻/热血/治愈/催泪/恋爱/机战/悬疑/日常/搞笑/推理）",
    "director": "导演/监督 个人（庵野秀明/新海诚/新房昭之/渡边信一郎/押井守/石原立也/水岛努）",
    "studio":   "制作公司/工作室/厂牌（京阿尼/京都动画/骨头社/Ghibli/吉卜力/P.A.WORKS/J.C.STAFF/A-1/Trigger/MAPPA/ufotable/SHAFT/WIT）",
    "writer":   "编剧/原作/脚本/系列构成（虚渊玄/麻枝准/key社/奈须蘑菇/大河内一楼/荒川弘/谏山创/羽海野千花）",
    "seiyuu":   "声优/CV/配音演员（花泽香菜/梶裕贵/钉宫理惠/林原惠/宫野真守/神谷浩史）",
    "similar":  "类似/相似/像XX/找和XX风格接近/同类型",
}

_DIMS_DESC = "\n".join(f"- {k}: {v}" for k, v in _DIMENSIONS_MAP.items())

_PROMPT = f"""分析用户问题，识别涉及的 ACG 检索维度。输出JSON格式。primary 必须是维度名本身（如 rating/genre），不要写解释。

{_DIMS_DESC}
- general: 问候/闲聊/无特定维度/纯新番推荐

关键规则（逐一检查每个维度）:
1. 动画制作公司/工作室/厂牌名称 → 包含 studio（不是 director！）
   ⚠️ 反例: "京阿尼制作的动画" → studio, "骨头社出品的番剧" → studio, "MAPPA制作的动画" → studio
   ⚠️ 反例: "京都动画的作品" → studio, "P.A.WORKS的番剧" → studio, "SHAFT制作的" → studio
   ⚠️ 反例: "吉卜力的电影" → studio, "WIT STUDIO制作" → studio, "ufotable的作品" → studio
2. 导演/监督 人名 → 包含 director（仅限人名！公司名不算！）
3. 编剧/原作/脚本 人名或原作名 → 包含 writer
4. 评分/高分/排名/TOP/神作/经典/口碑 → 包含 rating
5. 类型/风格词（科幻/热血/治愈/催泪/恋爱/机战/悬疑等）→ 包含 genre
6. 声优/CV + 人名 → 包含 seiyuu
7. 类似XX/像XX/同类型/找和XX一样 → 包含 similar

核心区分: 公司/社/工作室/厂牌名称 → studio; 人名 → 看身份（监督→director, 编剧→writer, 声优→seiyuu）

重要: 一个问题可能涉及多个维度，必须全部列出！
例: "京阿尼的高分催泪作品" → dimensions=["studio","rating","genre"], primary="studio"
例: "京阿尼制作的动画推荐" → dimensions=["studio"], primary="studio"
例: "骨头社出品的番剧有哪些" → dimensions=["studio"], primary="studio"
例: "MAPPA制作的动画" → dimensions=["studio"], primary="studio"
例: "庵野秀明导演的作品" → dimensions=["director"], primary="director"
例: "骨头社出品的热血动画" → dimensions=["studio","genre"], primary="studio"
例: "虚渊玄编剧的黑暗番" → dimensions=["writer","genre"], primary="writer"
例: "推荐9分以上科幻番" → dimensions=["rating","genre"], primary="rating"
纯问候 → dimensions=["general"], primary="general" """


async def domain_router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(DimensionDecision, method="function_calling").ainvoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=state["messages"][-1].content),
        ])
        dims = r.dimensions if r.dimensions else ["general"]
        return {
            "router_decision": r.primary,
            "reasoning":       r.reasoning,
            "dimensions":      dims,
            "active_dimension": r.primary,
            "processed_dims":  [],
            "tool_results":    {},
        }
    except Exception as e:
        return {
            "router_decision": "general",
            "reasoning":       str(e),
            "dimensions":      ["general"],
            "active_dimension": "general",
            "processed_dims":  [],
            "tool_results":    {},
        }
