"""Context Builder — 基于 history + 结构化状态生成 ConversationContext

职责:
  1. 检测追问/指代模式
  2. 解析指代（代词/序号指代）
  3. 推断当前话题
  4. 生成完整 ConversationContext（含预拼接的 history_text）
"""
import re
from agents.state import AgentState, ConversationContext


# ── 模块级预编译常量（避免每次调用重建）──

_FOLLOWUP_PATTERNS = [
    re.compile(r"^(它|他|她|这个|那个|这部|那部|这|那)"),
    re.compile(r"^(还有|还有吗|还有呢|再|继续|再来)"),
    re.compile(r"^(那|那么|那.*呢)"),
    re.compile(r"^(和|跟|与).{0,3}(比|对比|区别|哪个)"),
]

# 序号指代: 按 key 长度降序排列（最长匹配优先），模块级常量避免每次 sorted
_ORDINAL_MAP: list[tuple[str, int]] = [
    ("第一", 0), ("第一个", 0), ("第一部", 0), ("一", 0),
    ("第二", 1), ("第二个", 1), ("第二部", 1), ("二", 1),
    ("第三", 2), ("第三个", 2), ("第三部", 2), ("三", 2),
    ("第四", 3), ("第四个", 3), ("第四部", 3), ("四", 3),
    ("第五", 4), ("第五个", 4), ("第五部", 4), ("五", 4),
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


def _detect_followup(query: str) -> bool:
    return any(p.match(query) for p in _FOLLOWUP_PATTERNS)


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

    if query.startswith("那") and entities and len(query) > 1:
        second_char = query[1]
        if second_char in "他她它个部件些有没有好是":
            return entities[0]["name"] + query[1:]
    if query in ("那", "那呢", "那吗") and entities:
        return entities[0]["name"] + query[1:]

    return query


def _infer_topic(query: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return topic
    return "通用"


async def context_builder_node(state: AgentState) -> dict:
    """构建 ConversationContext"""
    # 获取当前用户输入：优先从 messages[-1]（跨轮最可靠），fallback 到 original_query
    messages = state.get("messages", [])
    query = ""
    if messages:
        last = messages[-1]
        query = last.content if hasattr(last, 'content') else ""
    if not query:
        query = state.get("original_query", "")
    ctx = state.get("context", {})
    history = ctx.get("history", []) if isinstance(ctx, dict) else []

    is_followup = _detect_followup(query) if history else False

    # 指代解析（不依赖 MetadataIndex，只用已有 entity 信息）
    resolved = query
    if is_followup:
        # 优先用实体信息（角色/梗名）解析"他/她/它"
        entity_name = state.get("entity_name", "")
        entity_type = state.get("entity_type", "")
        if entity_name and entity_type in ("character", "alias") and entity_name != query:
            entities = [{"name": entity_name, "type": entity_type}] + state.get("recent_entities", [])
        else:
            entities = state.get("recent_entities", [])
        resolved = _resolve_reference(query, entities)

    # 推断当前话题
    current_topic = _infer_topic(query)

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
        "current_topic": current_topic,
        "is_followup": is_followup,
        "resolved_query": resolved,
        "previous_intent": state.get("previous_intent", ""),
    }

    return {"context": context, "resolved_query": resolved}
