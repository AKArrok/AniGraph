"""Simple Fact Answer - 简单事实查询用单次 LLM 调用直接回答

跳过 metadata_reasoner -> merge -> answer 三步流水线，
对 simple_fact 类查询（评分/声优/是谁/哪部等）一次完成分析和回答。
"""
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config
from agents.prompts import BANNED_PHRASES, INTERNAL_TERMS, build_context_section

logger = logging.getLogger(__name__)

_SIMPLE_FACT_SYSTEM = f"""你是 ACG 番剧专家。回答用户问题时，资料里有直接答案就引用，没有就根据线索推理--标签、声优、番剧名、评分都是线索。

## 核心原则
- 用口语化中文，像跟朋友聊天，简洁直接别啰嗦
- 只说答案和相关补充，不要加任何问候语（你好/路上小心/早上好/欢迎等）
- 只有完全没有任何线索时才说"不太确定"
- 追问场景：别从头介绍已讨论过的人物/作品，直接答问题

## 禁止
- 禁止: 以问候语开头，直接回答问题本身
- 禁止: {BANNED_PHRASES}
- 禁止: 编造不存在的人名、评分、日期
- 禁止: 说{INTERNAL_TERMS}等内部术语
- 禁止: 说"根据资料""资料显示""从数据来看"--直接给答案，别解释来源

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

    # 构建对话上下文段落（追问时帮助 LLM 理解指代）
    context = state.get("context", {})
    history_text = context.get("history_text_recent", "") if isinstance(context, dict) else ""
    is_followup = context.get("is_followup", False) if isinstance(context, dict) else False
    context_section = build_context_section(history_text, is_followup=is_followup)

    # 优先展示匹配关键词的条目，其余截断到 5 条
    prioritized = _prioritize_metadata(metadata, keywords)[:5]
    md_text = _format_metadata(prioritized) if prioritized else "(无相关数据)"

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

    logger.info(f"  simple_fact_answer 耗时 {time.time()-t0:.1f}s")
    return {
        "messages": [resp],
        "previous_intent": "simple_fact",
        "recent_entities": existing_recent,
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


def _format_metadata(entries: list[dict]) -> str:
    """格式化为紧凑文本，每个条目一行关键信息"""
    lines = []
    for m in entries:
        name = m.get("name", "") or m.get("title", "")
        score = m.get("score", "") or m.get("rating", "")
        rank = m.get("rank", "")
        tags = m.get("tags", [])
        staff = m.get("staff", []) or m.get("seiyuu", [])
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(staff, str):
            staff = [staff]
        parts = [name]
        if score:
            parts.append(f"评分{score}")
        if rank:
            parts.append(f"排名{rank}")
        if tags:
            parts.append(f"标签:{','.join(str(t) for t in tags[:5])}")
        if staff:
            parts.append(f"人员:{','.join(str(s) for s in staff[:3])}")
        lines.append(" | ".join(str(p) for p in parts))
    return "\n".join(lines)
