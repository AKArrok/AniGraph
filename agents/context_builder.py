"""Context Builder — 基于 history + 结构化状态生成 ConversationContext

职责:
  1. 检测追问/指代模式
  2. 解析指代（代词/序号指代）
  3. 推断当前话题
  4. 生成完整 ConversationContext（含预拼接的 history_text）
"""
import re
from agents.message_content import has_image_block, latest_user_message, message_text
from agents.state import AgentState, ConversationContext


# ── 模块级预编译常量（避免每次调用重建）──

_FOLLOWUP_PATTERNS = [
    re.compile(r"^(它|他|她|这个|那个|这部|那部|这|那)"),
    re.compile(r"^(还有|还有吗|还有呢|再|继续|再来)"),
    re.compile(r"^(那|那么|那.*呢)"),
    re.compile(r"^(和|跟|与).{0,3}(比|对比|区别|哪个)"),
]

# 追问线索：ordinal reference 或句中出现明显 follow-up 词
_FOLLOWUP_HINTS_RE = re.compile(
    r"(第一个|第二个|第三个|第四个|第五个|"
    r"第一部|第二部|第三部|第四部|第五部|"
    r"第一|第二|第三|第四|第五|"
    r"再来一部|再来|再推荐|换一部|换一个|其他推荐|更接近|更贴近|更类似)"
)

# 序号指代: 按 key 长度降序排列（最长匹配优先），模块级常量避免每次 sorted
_ORDINAL_MAP: list[tuple[str, int]] = [
    ("第一个", 0), ("第一部", 0), ("第一", 0),
    ("第二个", 1), ("第二部", 1), ("第二", 1),
    ("第三个", 2), ("第三部", 2), ("第三", 2),
    ("第四个", 3), ("第四部", 3), ("第四", 3),
    ("第五个", 4), ("第五部", 4), ("第五", 4),
]

# 代词候选: 按长度降序（长词优先）
_PRNOUN_CANDIDATES = sorted(
    ["它", "他", "她", "这个", "那个", "这部", "那部", "这"],
    key=lambda x: -len(x),
)

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "评分": ["评分", "分数", "几分", "多少分"],
    "声优": ["声优", "配音", "CV"],
    "制作": ["制作", "公司", "动画公司", "制作组"],
    "推荐": ["推荐", "类似", "还有", "有没有", "求", "找"],
    "对比": ["比", "对比", "比较", "哪个", "区别", "差异", "vs"],
    "闲聊": ["你好", "谢谢", "再见", "拜拜", "你是谁"],
}

_RECOMMEND_COUNT_RE = re.compile(
    r"(?:推荐|再推荐|再来|找|给我)(?:[^，。！？]{0,6}?)(\d+|[一二三四五六七八九十两]+)部"
)
_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_VALID_CHINESE_COUNT_RE = re.compile(
    r"(?:[一二两三四五六七八九]|十[一二三四五六七八九]?|二十)"
)
_EXPLICIT_TITLE_RE = re.compile(r"《([^》]{1,40})》")


def _detect_followup(query: str) -> bool:
    if any(p.match(query) for p in _FOLLOWUP_PATTERNS):
        return True
    return bool(_FOLLOWUP_HINTS_RE.search(query))


# ── 约束识别（支持继承与撤销）──
_CONSTRAINT_PATTERNS: list[tuple[str, str, bool]] = [
    # (模式, 约束键, 设置值)
    (r"排除同系列", "exclude_same_series", True),
    (r"不要同系列|不要续作|不看续作|排除续作", "exclude_same_series", True),
    (r"不要剧场版|不看剧场版|排除剧场版", "exclude_movies", True),
    (r"同系列也可以|同系列可以|同系列也行|续作也可以|剧场版也可以", "exclude_same_series", False),
]


def _extract_constraints(query: str, previous: dict | None) -> dict:
    """从当前查询提取显式约束，并在追问时继承未覆盖的已有约束。"""
    constraints: dict = {}
    if previous:
        constraints.update(previous)

    # 显式约束：命中模式则覆盖（包括撤销）
    for pattern, key, value in _CONSTRAINT_PATTERNS:
        if re.search(pattern, query):
            constraints[key] = value

    # 清理：若 exclude_same_series=False，一并清空 excluded_series
    if not constraints.get("exclude_same_series"):
        constraints.pop("excluded_series", None)
        constraints.pop("topic_tags", None)

    return constraints


def _topic_tags(topic_title: str) -> list[str]:
    """从 Metadata Index 获取主题作品的标签，用于同系列排除时的检索扩展。"""
    if not topic_title:
        return []
    try:
        from agents.metadata_index import index
        item = index.get_by_alias(topic_title)
        if not item:
            return []
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in re.split(r"[,，、]", tags) if t.strip()]
        # 优先保留能表达类型的标签，过滤过于泛化或专属的标签
        skip = {"TV", "日本", "动画", "2011", "2011年", "2011年4月", "2018", "2018年", "2018年4月"}
        return [t for t in tags[:12] if t not in skip][:6]
    except Exception:
        return []


def _resolve_reference(query: str, entities: list[dict]) -> str:
    """解析指代

    示例:
      "它的评分" + [{name:"JOJO"}] → "JOJO的评分"
      "第二部的评分" + [{name:"A"}, {name:"B"}] → "B的评分"
    """
    for word, idx in _ORDINAL_MAP:
        if word in query and idx < len(entities):
            return query.replace(word, entities[idx]["name"])

    for p in _PRNOUN_CANDIDATES:
        if query.startswith(p) and entities:
            rest = query[len(p):]
            return entities[0]["name"] + rest

    # 追问中的指代不一定出现在句首，例如“再推荐两部和它气质相近的动画”。
    if entities:
        anchor = entities[0]["name"]
        internal_pronoun = re.search(
            r"(这个|那个|这部|那部|它|他|她)(?=的|是|有|在|和|跟|与|气质|风格|剧情|评分|主角|声优|$)",
            query,
        )
        if internal_pronoun:
            start, end = internal_pronoun.span()
            return query[:start] + anchor + query[end:]

    if query.startswith("那") and entities and len(query) > 1:
        second_char = query[1]
        if second_char in "他她它个部件些有没有好是":
            return entities[0]["name"] + query[1:]
    if query in ("那", "那呢", "那吗") and entities:
        return entities[0]["name"] + query[1:]

    return query


def extract_recommendation_count(query: str) -> int:
    """提取用户明确要求的推荐数量；未指定或异常值返回 0。"""
    match = _RECOMMEND_COUNT_RE.search(query)
    if not match:
        return 0
    raw = match.group(1)
    if raw.isdigit():
        count = int(raw)
    else:
        if not _VALID_CHINESE_COUNT_RE.fullmatch(raw):
            return 0
        if raw == "二十":
            count = 20
        elif raw.startswith("十"):
            count = 10 + _CHINESE_NUMBERS.get(raw[1:], 0)
        else:
            count = _CHINESE_NUMBERS[raw]
    return count if 1 <= count <= 20 else 0


def _infer_topic(query: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return topic
    return "通用"


async def context_builder_node(state: AgentState) -> dict:
    """构建 ConversationContext"""
    # Always read the latest HumanMessage. Tool and AI messages may be
    # appended after it by graph orchestration.
    messages = state.get("messages", [])
    message = latest_user_message(messages)
    query = ""
    if (
        state.get("vision_query")
        and message is not None
        and has_image_block(message)
    ):
        # vision_analyze has already converted the image into searchable text.
        # Do not replace it with the original caption before planner/retrieval.
        query = state["vision_query"]
    elif message is not None:
        query = message_text(message.content)
    if not query:
        query = state.get("original_query", "")
    ctx = state.get("context", {})
    history = ctx.get("history", []) if isinstance(ctx, dict) else []

    is_followup = _detect_followup(query) if history else False

    # 指代解析（不依赖 MetadataIndex，只用已有 entity 信息）
    resolved = query
    topic_entity = state.get("topic_entity", {})
    explicit_title = _EXPLICIT_TITLE_RE.search(query)
    entity_name = state.get("entity_name", "")
    entity_type = state.get("entity_type", "")
    if explicit_title:
        topic_entity = {"name": explicit_title.group(1).strip(), "type": "anime"}
    elif entity_name and entity_type in ("character", "alias") and entity_name != query:
        topic_entity = {"name": entity_name, "type": entity_type}

    if is_followup:
        # 优先用实体信息（角色/梗名）解析"他/她/它"
        entities = []
        if topic_entity.get("name"):
            entities.append(topic_entity)
        entities.extend(
            e for e in state.get("recent_entities", [])
            if e.get("name") != topic_entity.get("name")
        )
        resolved = _resolve_reference(query, entities)
        # “再推荐两部相似的”没有显式代词，也必须带上当前主题。
        if (
            resolved == query
            and topic_entity.get("name")
            and _infer_topic(query) == "推荐"
            and topic_entity["name"] not in query
        ):
            resolved = f"基于《{topic_entity['name']}》，{query}"

    # 推断当前话题
    current_topic = _infer_topic(query)

    # 跨轮约束继承
    previous_ctx = state.get("context", {}) if isinstance(state.get("context"), dict) else {}
    previous_constraints = previous_ctx.get("constraints", {}) if isinstance(previous_ctx, dict) else {}
    constraints = _extract_constraints(query, previous_constraints)

    # 记录被排除的系列来源：主题实体 + 用户明确排除的作品
    if constraints.get("exclude_same_series") and topic_entity.get("name"):
        excluded = set(constraints.get("excluded_series", []))
        excluded.add(topic_entity["name"])
        constraints["excluded_series"] = sorted(excluded)
        # 同系列排除时，保存主题标签用于追问检索，避免只搜主题标题导致结果全是同系列
        if "topic_tags" not in constraints:
            constraints["topic_tags"] = _topic_tags(topic_entity["name"])

    # 预构建 history_text（下游 planner/answer/simple_fact_answer 共用，避免重复拼接）
    # 分两版: 完整版给 planner（需全量上下文做意图分类），截断版给 answer（限制 token）
    history_text = ""
    history_text_recent = ""
    if history:
        lines = []
        for r in history:
            if r.get("user"):
                lines.append(f"用户: {r['user']}")
            if r.get("assistant"):
                lines.append(f"助手: {r['assistant'][:200]}")
        history_text = "\n".join(lines)
        # answer 节点只需最近 3 轮，避免 token 膨胀超过 LLM 输入限制
        history_text_recent = "\n".join(lines[-(3 * 2):])  # 每轮 2 行，3 轮 = 6 行

    context: ConversationContext = {
        "history": history,
        "history_text": history_text,
        "history_text_recent": history_text_recent,
        "recent_entities": state.get("recent_entities", []),
        "topic_entity": topic_entity,
        "current_topic": current_topic,
        "is_followup": is_followup,
        "resolved_query": resolved,
        "previous_intent": state.get("previous_intent", ""),
        "constraints": constraints,
    }

    return {
        "context": context,
        "resolved_query": resolved,
        "topic_entity": topic_entity,
        "recommendation_count": extract_recommendation_count(query),
    }
