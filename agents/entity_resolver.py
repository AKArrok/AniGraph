"""Entity Resolver: 将用户输入中的实体（角色/梗/作品简称）解析为结构化信息。

层级:
  L0: 高频字典 (零成本)
  L1: LLM 推理 (覆盖常见实体)
  L2: 联网兜底 (Tavily, 仅在 confidence < 0.5 时触发)

输出格式:
  {
    "type": "character" | "alias" | "meme",
    "entity": "原始输入中的实体名",
    "anime": "对应番剧正式中文名",
    "confidence": 0.0-1.0,
    "source": "dict" | "llm" | "web"
  }
"""

import json
import re
import functools

from langchain_core.messages import HumanMessage
from llms import simple_LLM

# ── L0: 高频实体字典 ──

ENTITY_DICT: dict[str, tuple[str, str]] = {
    # ── 角色 (character) ──
    "夏亚": ("character", "机动战士高达"),
    "阿姆罗": ("character", "机动战士高达"),
    "刹那": ("character", "机动战士高达00"),
    "基拉": ("character", "机动战士高达SEED"),
    "阿斯兰": ("character", "机动战士高达SEED"),
    "惠惠": ("character", "为美好的世界献上祝福！"),
    "阿库娅": ("character", "为美好的世界献上祝福！"),
    "达克妮丝": ("character", "为美好的世界献上祝福！"),
    "和真": ("character", "为美好的世界献上祝福！"),
    "雷姆": ("character", "Re:从零开始的异世界生活"),
    "拉姆": ("character", "Re:从零开始的异世界生活"),
    "艾米莉亚": ("character", "Re:从零开始的异世界生活"),
    "菜月昴": ("character", "Re:从零开始的异世界生活"),
    "486": ("character", "Re:从零开始的异世界生活"),
    "艾伦": ("character", "进击的巨人"),
    "三笠": ("character", "进击的巨人"),
    "兵长": ("character", "进击的巨人"),
    "利威尔": ("character", "进击的巨人"),
    "炭治郎": ("character", "鬼灭之刃"),
    "祢豆子": ("character", "鬼灭之刃"),
    "善逸": ("character", "鬼灭之刃"),
    "伊之助": ("character", "鬼灭之刃"),
    "承太郎": ("character", "JOJO的奇妙冒险"),
    "迪奥": ("character", "JOJO的奇妙冒险"),
    "dio": ("character", "JOJO的奇妙冒险"),
    "乔鲁诺": ("character", "JOJO的奇妙冒险"),
    "徐伦": ("character", "JOJO的奇妙冒险"),
    "琦玉": ("character", "一拳超人"),
    "埼玉": ("character", "一拳超人"),
    "鲁路修": ("character", "Code Geass 反叛的鲁路修"),
    "cc": ("character", "Code Geass 反叛的鲁路修"),
    "c.c.": ("character", "Code Geass 反叛的鲁路修"),
    "雪之下雪乃": ("character", "我的青春恋爱物语果然有问题。"),
    "比企谷八幡": ("character", "我的青春恋爱物语果然有问题。"),
    "大老师": ("character", "我的青春恋爱物语果然有问题。"),
    "折木奉太郎": ("character", "冰菓"),
    "千反田爱瑠": ("character", "冰菓"),
    "saber": ("character", "Fate/stay night"),
    "阿尔托莉雅": ("character", "Fate/stay night"),
    "金闪闪": ("character", "Fate/stay night"),
    "远坂凛": ("character", "Fate/stay night"),
    "卫宫士郎": ("character", "Fate/stay night"),
    "间桐樱": ("character", "Fate/stay night"),
    "冈部伦太郎": ("character", "命运石之门"),
    "助手": ("character", "命运石之门"),
    "牧濑红莉栖": ("character", "命运石之门"),
    "柯南": ("character", "名侦探柯南"),
    "路飞": ("character", "海贼王"),
    "鸣人": ("character", "火影忍者"),
    "佐助": ("character", "火影忍者"),
    "银时": ("character", "银魂"),

    # ── 梗 (meme) ──
    "典明粥": ("meme", "JOJO的奇妙冒险"),
    "我不做人了": ("meme", "JOJO的奇妙冒险"),
    "欧拉欧拉": ("meme", "JOJO的奇妙冒险"),
    "木大木大": ("meme", "JOJO的奇妙冒险"),
    "jojo立": ("meme", "JOJO的奇妙冒险"),
    "替身使者": ("meme", "JOJO的奇妙冒险"),
    "食堂泼辣酱": ("meme", "JOJO的奇妙冒险"),
    "砸瓦鲁多": ("meme", "JOJO的奇妙冒险"),
    "都是时臣的错": ("meme", "Fate/Zero"),
    "人被殺就會死": ("meme", "Fate/stay night"),
    "人被杀了就会死": ("meme", "Fate/stay night"),
    "正义的伙伴": ("meme", "Fate/stay night"),
    "我变秃了也变强了": ("meme", "一拳超人"),
    "一拳秒了": ("meme", "一拳超人"),
    "错的不是我，是这个世界": ("meme", "东京喰种"),
    "献出心脏": ("meme", "进击的巨人"),
    "一匹不留": ("meme", "进击的巨人"),
    "教练我想打篮球": ("meme", "灌篮高手"),
    "真相只有一个": ("meme", "名侦探柯南"),
    "我的钻头是突破天际的": ("meme", "天元突破 红莲螺岩"),
    "你已经死了": ("meme", "北斗神拳"),
    "人类圣经": ("meme", "紫罗兰永恒花园"),
    "萌王": ("meme", "关于我转生变成史莱姆这档事"),
}

ENTITY_DICT_LOWER = {k.lower(): v for k, v in ENTITY_DICT.items()}


# ── 实体检测（判断用户是否在问角色/梗）──

CHARACTER_PATTERNS = [
    r"(谁|是谁|什么人|哪个角色|出自|登场).{0,8}$",
    r"^.{1,6}是.{0,4}(角色|人物)",
    r"(介绍|说说|讲一下).{1,6}(这|那).{0,2}(角色|人物)",
    r"(什么番|哪个番|哪个动漫|什么动漫).{0,6}(角色|人物|主角)",
    r"(配音|cv|声优).{0,6}是",
]
MEME_PATTERNS = [
    r"(什么梗|啥梗|意思是|出处|为什么说|梗是|这个梗)",
    r"^(这|那).{0,2}(啥|什么)(意思|梗)",
]


def detect_entity_type(query: str) -> str | None:
    """检测用户输入是否在询问角色/梗，返回 type 或 None"""
    q = query.strip()

    # 含代词/追问词的查询不是实体查询（如"他的对手是谁""那个角色的声优"）
    FOLLOWUP_PRONOUNS = [r"他的", r"她的", r"它的", r"他们的", r"她们的", r"它们的",
                          r"这个", r"那个", r"这部", r"那部", r"还有吗", r"还有呢"]
    if any(re.search(p, q) for p in FOLLOWUP_PRONOUNS):
        return None

    # 先检测梗（更明确的关键词）
    for pat in MEME_PATTERNS:
        if re.search(pat, q):
            return "meme"

    # 再检测角色
    for pat in CHARACTER_PATTERNS:
        if re.search(pat, q):
            return "character"

    # 短纯文本（2-8字纯中文+英文），可能是角色名
    if 2 <= len(q) <= 8 and re.match(r'^[\u4e00-\u9fff·\w\-]+$', q):
        if q.lower() not in {"你好", "谢谢", "再见", "请问", "帮我", "你好呀"}:
            return "character"

    return None


# ── 解析入口 ──

def resolve_entity(query: str) -> dict | None:
    """解析用户查询中的实体。

    Returns:
        {"type", "entity", "anime", "confidence", "source"} 或 None
    """
    q = query.strip()

    # ── Step 1: L0 字典匹配 ──
    ql = q.lower()

    # 精确匹配
    if ql in ENTITY_DICT_LOWER:
        etype, anime = ENTITY_DICT_LOWER[ql]
        return _make_result(etype, q, anime, 0.95, "dict")

    # 包含匹配（最长优先，避免 "欧拉" 先于 "欧拉欧拉" 匹配）
    for key, (etype, anime) in sorted(ENTITY_DICT_LOWER.items(), key=lambda x: -len(x[0])):
        if key in ql:
            return _make_result(etype, key, anime, 0.90, "dict")

    # 复用现有番剧别名字典（alias.py）
    from agents.alias import resolve_alias_dict
    alias_result = resolve_alias_dict(q)
    if alias_result:
        return _make_result("alias", q, alias_result, 0.90, "dict")

    # ── Step 2: 检测是否可能是实体查询 ──
    etype = detect_entity_type(q)
    if not etype:
        return None

    # ── Step 3: L1 LLM 推理 ──
    llm_result = _llm_resolve(q, etype)
    if llm_result and llm_result.get("confidence", 0) >= 0.5:
        return _make_result(
            etype, q, llm_result["anime"],
            llm_result.get("confidence", 0.6), "llm",
        )

    # ── Step 4: 低置信度 — 返回标记，由 planner 触发联网 ──
    return _make_result(etype, q, "", 0.3, "llm")


def _make_result(etype: str, entity: str, anime: str, confidence: float, source: str) -> dict:
    return {
        "type": etype,
        "entity": entity,
        "anime": anime,
        "confidence": confidence,
        "source": source,
    }


# ── L1: LLM 推理 ──

_LLM_PROMPT = """你是 ACG 番剧专家。用户提到了一个{type_label}，请写出它对应番剧的正式中文名称及你的置信度。

{type_label}: {query}

输出严格 JSON: {{"anime": "番剧正式中文名", "confidence": 0.85}}
如果无法确认，输出: {{"anime": "", "confidence": 0.0}}"""


@functools.lru_cache(maxsize=128)
def _llm_resolve(query: str, etype: str) -> dict | None:
    type_label = "角色" if etype == "character" else "梗"
    prompt = _LLM_PROMPT.format(type_label=type_label, query=query[:100])

    try:
        resp = simple_LLM.invoke([HumanMessage(content=prompt)])
        text = resp.content.strip()
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None
