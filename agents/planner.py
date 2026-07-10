"""Planner Agent — 分析用户查询，输出 ExecutionPlan 驱动图编排

职责:
  1. Query Classifier: 判断查询类别（metadata/semantic/mixed），决定走哪个检索路径
  2. 判断查询类型（简单事实 / 推荐 / 对比 / 闲聊）
  3. 决定是否需要别名解析
  4. 决定查询优化策略（direct/rewrite/hyde/decompose）
  5. 决策需要哪些 Expert（metadata_reasoner/similar_expert）
  6. 决定 Expert 并行还是串行
  7. 决定是否需要联网

输入: 用户原始查询
输出: ExecutionPlan dict
"""
import re
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config

import functools

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=256)
def _quick_is_simple(query: str) -> bool | None:
    """零 LLM 快速判断是否为简单事实查询。
    返回 True/False 表示确定，None 表示需要 LLM 判断。
    """
    # 明确的事实询问词 → simple
    SIMPLE_MARKERS = [
        r"(评分|几分|多少分)",      # 评分查询
        r"(声优|配音|CV|cv)",       # 声优查询
        r"(导演|监督|执导)",         # 导演查询
        r"(公司|制作|出品)",         # 公司查询
        r"(是谁|谁是|是什么|叫什么|介绍)",   # 身份询问
        r"(标签|类型|分类)",         # 标签查询
        r"(什么时候|哪年|播出)",     # 时间查询
    ]
    if any(re.search(p, query) for p in SIMPLE_MARKERS):
        return True

    # 明确的推荐/发现词 → recommend
    RECOMMEND_MARKERS = [
        r"(推荐|求推荐|安利|求安利|找一部|找几部|有哪些|有什么好看的)",
        r"(类似|像.{0,4}一样|同类型|差不多)",
        r"(求番|找番|要几部|来几部)",
    ]
    if any(re.search(p, query) for p in RECOMMEND_MARKERS):
        return False

    return None  # 规则无法确定


@functools.lru_cache(maxsize=256)
def _is_simple_fact(query: str) -> bool:
    """轻量 LLM 判断查询是否为简单事实（问某个已知事物的具体信息）vs 推荐/发现类

    使用 simple_LLM (qwen-flash) 做二分类，速度快、成本极低。
    注意: 此函数仅在 _quick_is_simple 无法确定时才被调用。
    """
    from llms import simple_LLM
    from langchain_core.messages import HumanMessage

    prompt = (
        "判断用户意图: 是在询问一个已知事物的具体信息, 还是在要求推荐/发现新事物?\n"
        "- 简单事实(simple): 问评分、声优、导演、公司、介绍、是谁、有什么标签等具体信息\n"
        "- 推荐(recommend): 要求推荐、找番、求安利、有哪些好看的等\n"
        f"查询: {query}\n"
        "只输出 simple 或 recommend:"
    )
    try:
        resp = simple_LLM.invoke([HumanMessage(content=prompt)])
        text = resp.content.strip().lower()
        return "simple" in text and "recommend" not in text
    except Exception:
        # fallback: 短查询默认为 simple
        return len(query) <= 15

# ── Query Classifier: 规则优先，LLM 补充 ─────────────────────────

# Metadata Query 特征 — 纯结构化过滤，零 Pinecone
_METADATA_PATTERNS = [
    # 制作公司/工作室查询
    r"(京都动画|京阿尼|MAPPA|A-1|骨头社|扳机社|WIT|霸权社|UFO|飞碟社|节操社|J\.C|小丑社|东映|SUNRISE|日升|P\.A\.WORKS|银链|feel\.|LIDENFILMS|动画工房|SILVER LINK|WHITE FOX|Production I\.G|MADHOUSE|疯房子|SHAFT|GAINAX|TRIGGER|David Production|CloverWorks|Kinema Citrus|C2C|8bit|ZEXCS|XEBEC|SATELIGHT|Brain's Base|动画工房|Lerche|diomed.a|TROYCA|NOMAD|OLM).*(作品|动漫|番|动画|有哪些|做过|做过什么)",
    # 年份/季度筛选
    r"(20\d{2}|今年|去年|前年|本季|上季).*(播出|番|动漫|动画|新番)",
    r"\d{4}年.{0,4}(动漫|番|动画|作品)",
    # 评分排序
    r"评分(最|比较).*(高|低|好|差).{0,6}(番|动漫|动画|作品|恋爱|热血|异世界)",
    r"(最好|最差|最强|高分|低分).{0,4}(番|动漫|动画)",
    r"(评分|分数).{0,2}(>|>=|<|<=|高于|低于|超过|以上|以下)\s*\d",
    # 声优查询
    r"(花泽香菜|钉宫理惠|杉田智和|神谷浩史|梶裕贵|宫野真守|福山润|中村悠一|樱井孝宏|松冈祯丞|茅野爱衣|早见沙织|佐仓绫音|水濑祈|悠木碧|雨宫天|高桥李依|花江夏树|石川界人|内田真礼|小野大辅|诹访部顺一).{0,6}(配音|出演|声优|角色|作品|动漫|番)",
    r"([\u4e00-\u9fff]{2,4}).{0,4}(配音|声优|出演).{0,4}(动漫|番|动画|作品)",
    # 导演/编剧查询
    r"(新海诚|新房昭之|押井守|渡边信一郎|庵野秀明|虚渊玄|奈须蘑菇|大河内一楼|花田十辉|冈田磨里|横手美智子|丸户史明).{0,6}(作品|动漫|番|动画|执导|监督|脚本)",
    r"([\u4e00-\u9fff]{2,4}).{0,4}(监督|导演|执导).{0,4}(动漫|番|动画|作品)",
    # 纯标签/类型过滤
    r"(有哪些|有什么|推荐|介绍|推荐一下).{0,4}(标签|类型|分类|题材|风格|标签下)",
    r"^(推荐|找|求|有没有|有什么|哪些).{0,6}(番|动漫|动画).{0,2}$",
]

# Semantic Query 特征 — 需要语义理解，走 Pinecone
_SEMANTIC_PATTERNS = [
    r"(类似|像|相似|相近|同类|同款|同类型).{0,6}(番|动漫|动画|作品|推荐)",
    r"(觉得|认为|推荐|评价|怎么样|好看吗|值得看|好不好看|好看|不好看)",
    r"(深度|分析|解读|影评|评论|讨论|聊聊|说说|谈谈)",
    r"(为什么|怎么|如何|怎么样才算)",
    r"(剧情|故事|结局|角色|人物|画风|制作|音乐|声优|演出).{0,4}(怎么样|如何|评价|表现)",
    r"(口碑|风评|热度|人气|火不火|知名度)",
    r"(感动|催泪|热血|燃|搞笑|治愈|致郁|恐怖|悬疑|烧脑|轻松|愉快).{0,4}(番|动漫|动画|推荐|作品)",
    r"(神作|佳作|名作|经典|必看|入门)",
    r"(和|跟|与|VS|vs).{1,4}(比|对比|比较|区别|不同|差异|哪个)",
]

# 规则特征词（精确到制作公司/声优/年份等实体）
_METADATA_ENTITY_MARKERS = {
    "studio": [
        "京都动画", "京阿尼", "MAPPA", "A-1", "骨头社", "扳机社", "WIT", "霸权社",
        "UFO", "飞碟社", "节操社", "J.C", "小丑社", "东映", "SUNRISE", "日升",
        "P.A.WORKS", "动画工房", "WHITE FOX", "MADHOUSE", "疯房子", "SHAFT",
        "TRIGGER", "David Production", "CloverWorks", "Kinema Citrus",
    ],
    "seiyuu": [
        "花泽香菜", "钉宫理惠", "杉田智和", "神谷浩史", "梶裕贵", "宫野真守",
        "福山润", "中村悠一", "樱井孝宏", "松冈祯丞", "茅野爱衣", "早见沙织",
        "佐仓绫音", "水濑祈", "悠木碧", "雨宫天", "高桥李依", "石川界人",
        "花江夏树", "内田真礼", "小野大辅", "诹访部顺一",
    ],
    "director": [
        "新海诚", "新房昭之", "庵野秀明", "渡边信一郎", "押井守",
        "虚渊玄", "奈须蘑菇", "大河内一楼", "花田十辉", "冈田磨里",
    ],
}


def _classify_query_category(query: str) -> str:
    """规则优先 → 判断查询类别: metadata | semantic | mixed"""
    q = query.strip()

    # 1. 公司/声优/导演 实体 → metadata 为主
    studio_hit = any(m in q for m in _METADATA_ENTITY_MARKERS["studio"])
    seiyuu_hit = any(m in q for m in _METADATA_ENTITY_MARKERS["seiyuu"])
    director_hit = any(m in q for m in _METADATA_ENTITY_MARKERS["director"])

    # 2. 年份筛选
    year_hit = bool(re.search(r"(20\d{2}|今年|去年|本季).*(播出|番|动漫|动画)", q))

    # 3. 评分排序
    score_hit = bool(re.search(r"评分.{0,4}(最|高|低|好|差)", q))

    # 4. 已知标签关键词（热血/动作/搞笑/异世界等）
    tag_keywords = ["热血", "动作", "搞笑", "异世界", "奇幻", "科幻", "恋爱", "日常",
                  "治愈", "悬疑", "推理", "战斗", "冒险", "校园", "机战", "运动",
                  "魔法", "后宫", "百合", "耽美", "美食", "音乐", "竞技",
                  "战争", "历史", "恐怖", "职场", "偶像", "转生", "游戏"]
    tag_kw_hit = any(t in q for t in tag_keywords)

    # 5. 纯标签/条件查询
    tag_only = bool(re.search(r"(有哪些|有什么|有没有|找|求).{0,6}(番|动漫|动画)", q))

    is_metadata = studio_hit or seiyuu_hit or director_hit or year_hit or score_hit or tag_kw_hit or tag_only

    # 5. 语义特征
    is_semantic = any(re.search(p, q) for p in _SEMANTIC_PATTERNS)

    # 5.5 标签 + 推荐意图 → mixed（既需要 Metadata 过滤，也需要语义理解）
    tag_with_recommend = bool(
        (studio_hit or seiyuu_hit or director_hit or year_hit or score_hit or tag_only)
        and is_semantic
    )
    if tag_with_recommend:
        return "mixed"

    # 6. 纯事实查询（单个番剧的信息查询）
    if not is_metadata and not is_semantic:
        # 短查询可能是指定番剧的事实查询
        if len(q) <= 15 and any(kw in q for kw in ["评分", "声优", "导演", "编剧", "制作", "什么", "介绍", "是"]):
            return "metadata"
        # 指定番剧名 + 维度 → metadata
        if re.search(r"的(评分|声优|导演|编剧|标签|类型|制作|公司)", q):
            return "metadata"

    if is_metadata and is_semantic:
        return "mixed"
    elif is_metadata:
        return "metadata"
    elif is_semantic:
        return "semantic"
    else:
        return "mixed"  # 默认走双路


# ── Planner Prompt ─────────────────────────────────────────────

_PLANNER_SYSTEM = """你是 ACG 番剧推荐系统的规划器。分析用户查询，输出执行计划。

## 查询类型分类
- simple_fact: 查询特定番剧的评分/声优/导演/公司等事实信息（例："素晴的评分是多少"）
- recommendation: 根据条件推荐番剧（例："推荐热血动作番"、"有没有类似进击的巨人"）
- comparison: 对比多部番剧（例："巨人vs鬼灭哪个好看"）
- chat: 闲聊/问候/非番剧问题

## 查询优化策略
- direct: 简单查询，不需要重写（如精确番剧名查询）
- rewrite: 需要从多角度扩展查询（如推荐类问题）
- hyde: 深度分析/评价类问题（有"为什么""好在哪""区别"等词时选用）
- decompose: 含多子问题（有"分别""还有"等标记）

## Expert 选择
- metadata_reasoner: 涉及评分/标签/制作公司/导演/编剧/声优等结构化数据查询时必选
- similar_expert: 涉及相似推荐/对比/同一类型作品时必选
- 推荐类问题通常两个 Expert 都选，简单事实通常只选 metadata_reasoner

## 并行 vs 串行
- 两个 Expert 都需要时 → parallel=true（并行执行）
- 只有一个 Expert 或 Expert 间有依赖 → parallel=false

## 联网
- need_web=true: 查询可能超出知识库范围、需要最新数据、或昵称无法解析时
- need_web=false: 知识库能覆盖的常规查询

{history_section}

输出必须是严格 JSON 格式，不要包含 markdown 标记:
{{
  "query_type": "recommendation",
  "alias_resolved": false,
  "rewrite_strategy": "rewrite",
  "experts": ["metadata_reasoner", "similar_expert"],
  "parallel": true,
  "need_web": false,
  "reasoning": "用户要求推荐动作热血番，需要元数据查询标签+评分，同时需要相似推荐"
}}"""

_PLANNER_USER = "用户查询: {query}"


def plan(query: str, history_text: str = "") -> dict:
    """Planner 一次调用，输出完整 ExecutionPlan

    优化: 规则能决定时直接返回，零 LLM 调用
    """
    import json
    from llms import answer_LLM

    # Step 0: 规则分类 query_category（零 LLM）
    query_category = _classify_query_category(query)

    # ── 规则优先：能直接判断的跳过 LLM ──

    # 1. 闲聊/问候
    CHAT_PATTERNS = ["你好", "hi", "hello", "谢谢", "再见", "bye", "你是谁", "你能做什么", "帮我"]
    if any(p in query for p in CHAT_PATTERNS) and len(query) <= 15:
        return {
            "query_type": "chat",
            "alias_resolved": False,
            "rewrite_strategy": "direct",
            "experts": [],
            "parallel": False,
            "need_web": False,
            "query_category": query_category,
            "reasoning": "规则判断：闲聊"
        }

    # 2. 纯 metadata 查询（公司/声优/年份/评分/标签）→ 只需 metadata_reasoner
    if query_category == "metadata":
        qs = _quick_is_simple(query)
        if qs is None:
            qs = _is_simple_fact(query)
        return {
            "query_type": "simple_fact" if qs else "recommendation",
            "alias_resolved": False,
            "rewrite_strategy": "direct",
            "experts": ["metadata_reasoner"],
            "parallel": False,
            "need_web": False,
            "query_category": "metadata",
            "reasoning": "规则判断：纯 metadata 查询"
        }

    # 3. 纯语义查询（相似推荐/评价分析）→ 只需 similar_expert
    if query_category == "semantic":
        return {
            "query_type": "recommendation",
            "alias_resolved": False,
            "rewrite_strategy": "rewrite",
            "experts": ["similar_expert"],
            "parallel": False,
            "need_web": False,
            "query_category": "semantic",
            "reasoning": "规则判断：纯语义查询"
        }

    # ── mixed / 复杂查询走 LLM ──
    llm = answer_LLM.bind(temperature=config.PLANNER_TEMPERATURE)

    # 构建历史段落
    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史（仅供参考，用于理解指代和上下文）\n"
            f"{history_text}\n\n"
            f"注意: 即使有历史，仍需独立判断当前查询的真实意图。followup 不代表延续上一轮的 query_type。"
        )

    system_prompt = _PLANNER_SYSTEM.format(history_section=history_section)

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=_PLANNER_USER.format(query=query)),
    ])

    # 解析 JSON
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        plan_dict = json.loads(text)
    except json.JSONDecodeError:
        plan_dict = {
            "query_type": "recommendation",
            "alias_resolved": False,
            "rewrite_strategy": "rewrite",
            "experts": ["metadata_reasoner", "similar_expert"],
            "parallel": True,
            "need_web": False,
            "reasoning": "JSON 解析失败，使用默认推荐策略"
        }

    # 确保必要字段存在
    plan_dict.setdefault("query_type", "recommendation")
    plan_dict.setdefault("alias_resolved", False)
    plan_dict.setdefault("rewrite_strategy", "rewrite")
    plan_dict.setdefault("experts", ["metadata_reasoner", "similar_expert"])
    plan_dict.setdefault("parallel", True)
    plan_dict.setdefault("need_web", False)
    plan_dict.setdefault("reasoning", "")

    # 注入规则分类的 query_category（覆盖 LLM 判断，确保一致性）
    plan_dict["query_category"] = query_category

    # metadata 类查询 → 不需要 similar_expert
    if query_category == "metadata":
        plan_dict["experts"] = ["metadata_reasoner"]
        plan_dict["parallel"] = False
        plan_dict["rewrite_strategy"] = "direct"

    # 简单事实查询 → 只用 metadata_reasoner
    if plan_dict["query_type"] == "simple_fact":
        plan_dict["experts"] = ["metadata_reasoner"]
        plan_dict["parallel"] = False

    # 闲聊 → 无 Expert
    if plan_dict["query_type"] == "chat":
        plan_dict["experts"] = []
        plan_dict["parallel"] = False
        plan_dict["rewrite_strategy"] = "direct"

    return plan_dict


async def planner_node(state: dict) -> dict:
    """LangGraph 节点: Planner"""
    t0 = time.time()
    # 优先使用 context_builder 解析后的 query（已做指代消解），再 fallback 到原始 query
    query = state.get("resolved_query", "") or state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    # 构建对话历史文本（注入 LLM prompt）
    context = state.get("context", {})
    history_text = ""
    if isinstance(context, dict) and context.get("history"):
        lines = []
        for r in context["history"]:
            lines.append(f"用户: {r['user']}")
            lines.append(f"助手: {r['assistant']}")
        history_text = "\n".join(lines)

    execution_plan = plan(query, history_text=history_text)

    # 根据实体解析结果调整 plan
    entity_confidence = state.get("entity_confidence", 1.0)
    entity_type = state.get("entity_type", "")
    entity_source = state.get("entity_source", "")

    if not isinstance(execution_plan, dict):
        execution_plan = execution_plan.model_dump() if hasattr(execution_plan, "model_dump") else dict(execution_plan)

    # 低置信度实体 → 需要联网搜索
    if entity_confidence < 0.5:
        execution_plan["need_web"] = True

    # 梗实体且非字典来源 → 知识库可能没有相关内容，走联网
    if entity_type == "meme" and entity_source != "dict":
        execution_plan["need_web"] = True

    # 角色实体 + 身份询问（"谁是X"/"X是谁"/"介绍X"）→ 更正为 simple_fact
    if entity_type == "character" and entity_confidence >= 0.5:
        identity_patterns = [r"谁", r"介绍", r"是什么", r"是怎样的", r"是什么人"]
        if any(re.search(p, query) for p in identity_patterns):
            execution_plan["query_type"] = "simple_fact"
            execution_plan["experts"] = ["metadata_reasoner"]
            execution_plan["parallel"] = False
            execution_plan["rewrite_strategy"] = "direct"
            execution_plan["reasoning"] = "规则判断：角色身份查询 → simple_fact"

    logger.info(f"  planner 耗时 {time.time()-t0:.1f}s (query_type={execution_plan.get('query_type')})")
    return {"plan": execution_plan, "original_query": query}
