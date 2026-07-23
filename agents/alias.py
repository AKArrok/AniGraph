"""Alias 解析工具 — 字典优先，LLM fallback"""
from __future__ import annotations
import functools
import config


def _build_hardcoded_alias_map() -> dict[str, str]:
    """构建常用番剧别名映射表（硬编码高频别名，零 LLM 调用）"""
    return {
        # 缩写/简称 → 正式名称
        "素晴": "为美好的世界献上祝福！",
        "konosuba": "为美好的世界献上祝福！",
        "为美好世界献上祝福": "为美好的世界献上祝福！",
        "re0": "Re:从零开始的异世界生活",
        "re:0": "Re:从零开始的异世界生活",
        "从零开始": "Re:从零开始的异世界生活",
        "巨人": "进击的巨人",
        "进击的巨人最终季": "进击的巨人 The Final Season",
        "刀剑": "刀剑神域",
        "sao": "刀剑神域",
        "fate": "Fate/stay night",
        "fate zero": "Fate/Zero",
        "fz": "Fate/Zero",
        "石头门": "命运石之门",
        "sg": "命运石之门",
        "俺妹": "我的妹妹哪有这么可爱！",
        "春物": "我的青春恋爱物语果然有问题。",
        "果青": "我的青春恋爱物语果然有问题。",
        "大老师": "我的青春恋爱物语果然有问题。",
        "路人女主": "路人女主的养成方法",
        "四月": "四月是你的谎言",
        "四月谎": "四月是你的谎言",
        "未闻花名": "我们仍未知道那天所看见的花的名字。",
        "花名": "我们仍未知道那天所看见的花的名字。",
        "魔禁": "魔法禁书目录",
        "超炮": "某科学的超电磁炮",
        "小圆": "魔法少女小圆",
        "圆神": "魔法少女小圆",
        "物语": "物语系列",
        "cl": "CLANNAD",
        "clannad": "CLANNAD",
        "ab": "Angel Beats!",
        "eva": "新世纪福音战士",
        "EVA": "新世纪福音战士",
        "龙与虎": "龙与虎",
        "ditf": "DARLING in the FRANXX",
        "国家队": "DARLING in the FRANXX",
        "紫罗兰": "紫罗兰永恒花园",
        "京紫": "紫罗兰永恒花园",
        "莉可丽丝": "Lycoris Recoil",
        "蒜": "Lycoris Recoil",
        "孤独摇滚": "孤独摇滚！",
        "滚": "孤独摇滚！",
        "推子": "【我推的孩子】",
        "我推": "【我推的孩子】",
        "芙莉莲": "葬送的芙莉莲",
        "咒术": "咒术回战",
        "咒回": "咒术回战",
        "鬼灭": "鬼灭之刃",
        "间谍": "间谍过家家",
        "过家家": "间谍过家家",
        "夏日重现": "夏日重现",
        "无职": "无职转生 ~到了异世界就拿出真本事~",
        "无职转生": "无职转生 ~到了异世界就拿出真本事~",
        "86": "86 -不存在的战区-",
        "边缘行者": "赛博朋克：边缘行者",
        "2077": "赛博朋克：边缘行者",
        "op": "ONE PIECE",
        "海贼": "ONE PIECE",
        "火影": "火影忍者",
        "死神": "死神",
        "银魂": "银魂",
    }


# 全局别名映射（启动时构建）
HARDCODED_ALIASES: dict[str, str] = _build_hardcoded_alias_map()


def resolve_alias_dict(query: str) -> str | None:
    """纯字典匹配：精确 / 包含 / Cache 三层"""
    q = query.strip()
    q_lower = q.lower()

    # 1. 精确匹配
    if q_lower in HARDCODED_ALIASES:
        return HARDCODED_ALIASES[q_lower]

    # 2. 包含匹配（别名在查询中）
    for alias, full_name in sorted(HARDCODED_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias.lower() in q_lower:
            return full_name

    # 3. Metadata Cache 查询
    from agents.cache import metadata_cache
    full_name = metadata_cache.resolve_alias(q)
    if full_name:
        return full_name

    return None


@functools.lru_cache(maxsize=128)
def _llm_resolve_alias(query: str) -> str | None:
    """LLM fallback: 联网查询番剧简称 → 正式名称"""
    _ALIAS_PROMPT = """你是 ACG 番剧专家。用户用简称/缩写提到了一个番剧，请写出它的正式中文名称。
如果查询不是番剧简称，输出"无"。

简称: {query}

正式名称或"无":"""

    try:
        from llms import simple_LLM, llm_invoke_with_retry
        from langchain_core.messages import HumanMessage

        prompt = _ALIAS_PROMPT.format(query=query)
        resp = llm_invoke_with_retry(simple_LLM, [HumanMessage(content=prompt)])
        name = resp.content.strip()
        if name and name != "无" and name != query and len(name) >= 2:
            return name
    except Exception:
        pass
    return None


def resolve_alias(query: str, use_llm: bool = True) -> tuple[str, bool]:
    """别名解析主入口

    策略: 硬编码字典 → Metadata Cache → LLM fallback (可选)

    Returns:
        (resolved_query, was_resolved)
    """
    # 1. 硬编码字典
    resolved = resolve_alias_dict(query)
    if resolved and resolved != query:
        return resolved, True

    # 2. LLM fallback
    if use_llm:
        resolved = _llm_resolve_alias(query)
        if resolved and resolved != query:
            # 加入缓存
            from agents.cache import metadata_cache
            metadata_cache.add_alias(query, resolved)
            return resolved, True

    return query, False
