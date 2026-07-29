"""Simple Fact Answer - 简单事实查询用单次 LLM 调用直接回答

跳过 metadata_reasoner -> merge -> answer 三步流水线，
对 simple_fact 类查询（评分/声优/是谁/哪部等）一次完成分析和回答。
"""
import time
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
import config
from agents.prompts import BANNED_PHRASES, INTERNAL_TERMS, build_context_section
from agents.metadata_fallbacks import (
    filter_metadata_by_query as _filter_by_query,
    fetch_by_studio_year as _fetch_by_studio_year,
    format_metadata as _format_metadata,
    is_list_query as _is_list_query,
)

logger = logging.getLogger(__name__)

_SIMPLE_FACT_SYSTEM = f"""回答用户的 ACG 事实类问题（评分、声优、导演、日期、作品归属等）。答案就在"元数据"里，缺什么就说不知道。

## 硬性事实约束
- 只能用"元数据"里出现过的字段，不用常识补，不跨条目拼数据。
- 涉及具体数字/名字（评分、声优、日期、集数）时，用"元数据"里给的原样值，不要凑整、不要改写。
- 元数据里没有的字段直接说"这个我没有确切信息"，不要用"应该""大概"糊过去。
- 追问场景：不重述已经聊过的人物/作品名，直接答新问的点。

## 完整性约束（列表类查询：标签/声优/staff）
- 用户问"有哪些标签/tag/被打上""列出主要标签/声优""所有 xxx"时，
  把元数据里出现的对应字段**全部列出**（去重后至少 8 个，实际有多少列多少），不要挑 3-5 个了事。
- 只要元数据里该字段有内容，就不许说"没有收录 xxx 数据"这种否认句。
- 标签清单可以用逗号平铺，不用逐个解释含义。

## 语言风格
- 3-5 句话内给出答案，别铺垫、别总结、不总分。
- 直接说结论，不写问候语，不写"根据资料/资料显示/从数据来看/以下是"这类来源说明。
- 列表类查询例外：完整列出优先于"3-5 句话内"。
- 有余量时再补一句相关背景（同作品的其他信息、常被搞混的对照点），没有就到此为止。
- 不做番剧推荐/安利，只答问题本身。

## 禁语
- 套话：{BANNED_PHRASES}。
- 内部术语：{INTERNAL_TERMS}。
- 元自我描述："作为一个 AI/助手"、"根据我的分析"。

{{context_section}}"""

_SIMPLE_FACT_USER = """## 用户问题
{query}

## 元数据
{metadata}
"""


async def simple_fact_answer_node(state: dict) -> dict:
    """简单事实查询：一次 LLM 调用直接输出回答"""
    t0 = time.time()
    from llms import simple_LLM, llm_ainvoke_with_retry

    query = state.get("resolved_query") or state.get("original_query", "")
    metadata = state.get("metadata", [])
    keywords = state.get("search_keywords", [])
    vision_context = state.get("vision_context", "")
    topic_entity = state.get("topic_entity", {})

    # 构建对话上下文段落（追问时帮助 LLM 理解指代）
    context = state.get("context", {})
    history_text = context.get("history_text_recent", "") if isinstance(context, dict) else ""
    is_followup = context.get("is_followup", False) if isinstance(context, dict) else False
    context_section = build_context_section(history_text, is_followup=is_followup)

    # 先按 query 里出现的公司×年份 / 年份区间过滤（关键：让"京都 2007"这类
    # 查询即便走快速通道也能拿到正确子集，而不是被 30 条噪声淹没）。
    filtered = _filter_by_query(metadata, query)
    # 二次召回：如果 query 里明显有 studio×year 意图，检索层召回不足时，
    # 直接问 metadata_index 补一批。相当于给快速通道一个"结构化兜底"。
    supplemental = _fetch_by_studio_year(query)
    if supplemental:
        seen = {str(m.get("id", "")) for m in filtered}
        for m in supplemental:
            if str(m.get("id", "")) not in seen:
                filtered.append(m)
                seen.add(str(m.get("id", "")))
    if not filtered:
        filtered = _prioritize_metadata(metadata, keywords)
    prioritized = filtered[:12]  # simple_fact_answer 用轻量模型，12 条留出余量
    # 追问场景：如果 metadata 为空但存在 topic_entity，用主题实体补一次检索
    if not prioritized and topic_entity.get("name"):
        try:
            from agents.metadata_index import index
            hit = index.get_by_alias(topic_entity["name"])
            if hit:
                prioritized = [hit]
        except Exception:
            pass

    # KB 全空时按需触发 Web fallback（simple_fact 快速通道不经 evaluator，
    # 否则联网兜底永远不会触发）。
    web_snippet = ""
    if not prioritized:
        web_snippet = await _try_web_fallback(query)

    md_text = _format_metadata(prioritized) if prioritized else "(无相关数据)"
    # 列表类查询（问标签/声优/staff）需要看到完整字段而不是截 8 个，
    # 否则模型看到部分数据会否认字段存在（真实回归案例：T000440 校园迷糊大王）
    if _is_list_query(query):
        md_text = _format_metadata(prioritized, verbose=True) if prioritized else md_text
    if web_snippet:
        md_text = f"{md_text}\n\n---\n联网搜索补充:\n{web_snippet}"
    if vision_context:
        md_text = f"图片分析: {vision_context}\n{md_text}"

    llm = simple_LLM.bind(temperature=config.ANSWER_TEMPERATURE)

    resp = await llm_ainvoke_with_retry(llm, [
        SystemMessage(content=_SIMPLE_FACT_SYSTEM.format(context_section=context_section)),
        HumanMessage(content=_SIMPLE_FACT_USER.format(query=query, metadata=md_text)),
    ])

    # 追加当前实体到 recent_entities（保留历史实体）
    entity_name = state.get("entity_name", "")
    entity_type = state.get("entity_type", "")
    existing_recent = state.get("recent_entities", [])
    if entity_name and entity_type in ("character", "alias"):
        if not any(e.get("name") == entity_name for e in existing_recent):
            # 限制 recent_entities 最多 5 个，防止长对话累积导致 prompt 膨胀
            existing_recent = (
                [{"name": entity_name, "type": entity_type}] + existing_recent
            )[:5]

    topic_entity = state.get("topic_entity", {})
    if entity_name and entity_type in ("character", "alias"):
        topic_entity = {"name": entity_name, "type": entity_type}

    logger.info(f"  simple_fact_answer 耗时 {time.time()-t0:.1f}s")
    return {
        "messages": [resp],
        "previous_intent": "simple_fact",
        "recent_entities": existing_recent,
        "topic_entity": topic_entity,
        "termination_reason": "web_fallback" if web_snippet else "",
    }


def _prioritize_metadata(metadata: list[dict], keywords: list[str]) -> list[dict]:
    """优先保留匹配关键词的元数据条目"""
    if not keywords:
        return metadata
    matched = []
    others = []
    for m in metadata:
        name = str(m.get("name", "") or m.get("title", ""))
        if any(kw.lower() in name.lower() for kw in keywords):
            matched.append(m)
        else:
            others.append(m)
    return matched + others


async def _try_web_fallback(query: str) -> str:
    """快速通道下的按需联网兜底：只在 KB 完全查不到时尝试。

    与 agents.web_fallback.web_fallback_node 逻辑对齐，但不改 merged_results
    （快速通道没有），改为返回一段文本让本节点拼进 metadata。
    """
    try:
        from tools.registry import tool_registry
        if not tool_registry.is_enabled("search_web"):
            return ""
        search_web = tool_registry.get_callable("search_web")
        if not search_web:
            return ""
        raw = search_web.invoke(f"{query} 动漫 番剧")
        if not raw or len(raw) < 30:
            return ""
        # 只截取前 1200 字作为补充上下文，模型自行判断相关性
        return raw[:1200]
    except Exception as e:
        logger.warning(f"  simple_fact web fallback 失败: {e}")
        return ""

