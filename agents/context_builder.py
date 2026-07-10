"""Context Builder — 基于 history + 结构化状态生成 ConversationContext

职责:
  1. 检测追问/指代模式
  2. 解析指代（代词/序号指代）
  3. 推断当前话题
  4. 生成完整 ConversationContext
"""
import re
from agents.state import AgentState, ConversationContext
import config


def _detect_followup(query: str) -> bool:
    """检测追问/指代模式"""
    patterns = [
        r"^(它|他|她|这个|那个|这部|那部|这|那)",    # 代词开头
        r"^(还有|还有吗|还有呢|再|继续|再来)",       # 追问
        r"^(那|那么|那.*呢)",                        # 衔接追问
        r"^(和|跟|与).{0,3}(比|对比|区别|哪个)",     # 对比追问
    ]
    return any(re.match(p, query) for p in patterns)


def _resolve_reference(query: str, entities: list[dict]) -> str:
    """解析指代

    示例:
      "它的评分" + [{name:"JOJO"}] → "JOJO的评分"
      "第二部的评分" + [{name:"A"}, {name:"B"}] → "B的评分"
    """
    # 序号指代: 第一部, 第二个, 第三个...
    ordinal_map = {
        "一": 0, "第一": 0, "第一个": 0, "第一部": 0,
        "二": 1, "第二": 1, "第二个": 1, "第二部": 1,
        "三": 2, "第三": 2, "第三个": 2, "第三部": 2,
        "四": 3, "第四": 3, "第四个": 3, "第四部": 3,
        "五": 4, "第五": 4, "第五个": 4, "第五部": 4,
    }
    for word, idx in sorted(ordinal_map.items(), key=lambda x: -len(x[0])):
        if word in query and idx < len(entities):
            return query.replace(word, entities[idx]["name"])

    # 代词指代: 替换查询开头的代词为第一个实体名
    # 注意: "那" 字太短且容易误匹配（如"那祢豆子呢"），必须检查后面字符是否为代词延续
    candidate_pronouns = ["它", "他", "她", "这个", "那个", "这部", "那部", "这"]
    for p in sorted(candidate_pronouns, key=lambda x: -len(x)):
        if query.startswith(p) and entities:
            rest = query[len(p):]
            return entities[0]["name"] + rest
    # "那" 单独处理：只在后面紧跟非中文字符或动词时才视为代词
    if query.startswith("那") and entities and len(query) > 1:
        second_char = query[1]
        # 如果第二个字是代词（他/她/它）、数量词（几/些）、疑问词（是/有/好）→ 追问模式
        if second_char in "他她它个部件些有没有好是":
            return entities[0]["name"] + query[1:]
    # 纯 "那"（如"那呢"）→ 也视为追问
    if query in ("那", "那呢", "那吗") and entities:
        return entities[0]["name"] + query[1:]

    return query


def _infer_topic(query: str) -> str:
    """推断当前话题"""
    topics = {
        "评分": ["评分", "分数", "几分", "多少分"],
        "声优": ["声优", "配音", "CV"],
        "制作": ["制作", "公司", "动画公司", "制作组"],
        "推荐": ["推荐", "类似", "还有", "有没有", "求", "找"],
        "对比": ["比", "对比", "比较", "哪个", "区别", "差异", "vs"],
        "闲聊": ["你好", "谢谢", "再见", "拜拜", "你是谁"],
    }
    for topic, keywords in topics.items():
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
        # 安全检查: entity_name 若与原始 query 相同（即 alias_resolve 没解析到真实实体），拒绝使用
        # 支持 character 和 alias 两种实体类型
        if entity_name and entity_type in ("character", "alias") and entity_name != query:
            # 把实体也加入候选列表，排在番剧名前面
            entities = [{"name": entity_name, "type": entity_type}] + state.get("recent_entities", [])
        else:
            entities = state.get("recent_entities", [])
        resolved = _resolve_reference(query, entities)

    # 推断当前话题
    current_topic = _infer_topic(query)

    context: ConversationContext = {
        "history": history,
        "recent_entities": state.get("recent_entities", []),
        "current_topic": current_topic,
        "is_followup": is_followup,
        "resolved_query": resolved,
        "previous_intent": state.get("previous_intent", ""),
    }

    return {"context": context, "resolved_query": resolved}
