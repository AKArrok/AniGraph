"""Answer Agent — 综合所有 Expert 结果，生成自然口语化回答

输入:
  - merged_results: Merge 后的综合结果文本
  - plan: ExecutionPlan（含 query_type）
  - original_query: 用户原始查询
  - context: ConversationContext（对话上下文）

输出:
  自然语言回答（直接写入 messages）
"""
import re
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config
from agents.prompts import BANNED_PHRASES, INTERNAL_TERMS, build_context_section
from agents.simple_fact_answer import _SIMPLE_FACT_SYSTEM
from agents.metadata_fallbacks import format_metadata, is_list_query

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = f"""把下面的分析结果，用一个刷了很多番的朋友的口吻讲给用户听。

## 硬性事实约束
- 只能使用"Expert 分析结果"里出现过的番剧名、评分、评论。缺什么就说"这部我手上信息不多"，不要靠常识补。
- 用户排除的作品/系列，不得作为推荐、备选、对比对象，也不得以"放宽一下"的名义带出来。
- 候选不够就直说数量不够，不要凑数、不要拉续作/剧场版/同系列充数。
- 不确定就说不确定，别用"应该""大概"糊过去。

## 语言风格
- 直接进入正题，不写"首先/其次/接下来/让我来/以下"这种起手式。
- 结尾不写"希望能帮到你/以上就是我的推荐/欢迎追问"这种收尾式。
- 数据自然嵌进句子里："Bangumi 8.5 分"、"排在前 50"，不要写成"评分：8.5"这种字段行。
- 每部作品用不同的切入角度（剧情钩子/观感/评论片段/制作背景），别用同一句式排比。
- 有观众评论时优先引用一两句原文片段，比空说"很受欢迎"有力。
- 空洞形容词能省就省：不要靠"经典/精彩/精妙/神作/引人入胜/不容错过/值得一看"来撑内容，说不出具体点就不说。
- 允许有个人偏好："这部我私心很推"、"这部我看得有点累"，但要落到具体理由。

## 禁语
- 套话：{BANNED_PHRASES}、"总而言之"。
- 内部术语：{INTERNAL_TERMS}。
- 元自我描述："作为一个 AI/助手/推荐系统"、"根据我的分析"。

{{context_section}}

## 跨轮排除项（用户前几轮已明确要排除的）
{{constraint_section}}
这些项不得以任何形式再次出现（包括续作、剧场版、衍生、"如果放宽一下"）；候选不够就说数量不够。"""

_ANSWER_USER = """## 用户问题
{query}

## 回答结构指引
{structure}

## Expert 分析结果
{merged_results}

请生成回答。"""

_CANDIDATE_USER = """## 用户问题
{query}

## 回答结构指引
{structure}

## 可用候选（按检索顺序，不必全部采用）
{candidates}

只从上述候选里挑最相关的几部。每部至少给一句具体理由，可以引用标签、评分或证据里的原文片段，讲清楚跟用户诉求的相似点或差异点。候选不够就说数量不够，不要用候选之外的作品凑数。

## 跨轮排除项（必须遵守）
{constraint_section}
这些项不得作为推荐、备选、对比对象再次出现（包括续作、剧场版、同系列）。"""

_CHAT_SYSTEM = """你是 AniGraph，一个聊 ACG 的伙伴。
- 闲聊保持简短自然，一句到两句为宜，别做自我介绍式的开场白。
- 用户问你是谁/能做什么时，就说"我可以聊聊番剧、角色、ACG 话题"这类简短说明，不背底层模型名，也不说"作为 AI"。
- 涉及具体番剧信息、评分、推荐时，不要在闲聊分支里编造，回一句"这个我查一下再说"这类引导式回复更合适。"""


async def answer_node(state: dict) -> dict:
    """最终回答节点: 重组 Expert 结果，用口语化方式输出"""
    t0 = time.time()
    from llms import answer_LLM, simple_LLM, llm_ainvoke_with_retry

    query = state.get("resolved_query") or state.get("original_query", "")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "unknown")
    context = state.get("context", {})
    merged_results = state.get("merged_results", "")
    recommendation_candidates = state.get("recommendation_candidates", [])

    # 闲聊 & 简单事实查询用小模型（快 + 省），复杂推理用大模型
    if query_type in ("chat", "simple_fact"):
        if query_type == "chat":
            # 闲聊无 Expert 结果，直接用用户消息回复
            resp = await llm_ainvoke_with_retry(simple_LLM, [
                SystemMessage(content=_CHAT_SYSTEM),
                HumanMessage(content=query),
            ])
            logger.info(f"  answer(chat) 耗时 {time.time()-t0:.1f}s")
            return {
                "messages": [resp],
                "previous_intent": query_type,
            }
        llm = simple_LLM.bind(temperature=config.ANSWER_TEMPERATURE)
    else:
        llm = answer_LLM.bind(temperature=config.ANSWER_TEMPERATURE)
    if not merged_results:
        expert_results = state.get("expert_results", [])
        if expert_results:
            parts = []
            for i, r in enumerate(expert_results, 1):
                answer = r.get("answer", "")
                confidence = r.get("confidence", 0)
                if answer:
                    parts.append(f"[Expert {i} | 置信度: {confidence:.0%}]\n{answer}")
            merged_results = "\n\n".join(parts)
        else:
            # 完整链路兜底：simple_fact 分支 experts=[] 时，把已经检索到的
            # metadata 直接格式化喂 answer，行为对齐 simple_fact_answer 快速通道。
            metadata = state.get("metadata", []) or []
            if metadata:
                verbose = is_list_query(query)
                merged_results = format_metadata(metadata[:12], verbose=verbose)
            else:
                merged_results = "(分析结果为空)"

    # 读取 Answer Planner 输出的结构指引
    answer_plan = state.get("answer_plan", {})
    structure = answer_plan.get("structure", "自由发挥")
    recommendation_count = state.get("recommendation_count", 0)

    # 构建约束段落
    constraints = context.get("constraints", {}) if isinstance(context, dict) else {}
    constraint_section = _build_constraint_section(constraints)

    if query_type == "recommendation" and recommendation_count:
        structure += (
            f"\n数量偏好：优先推荐约 {recommendation_count} 部作品，"
            "重点保证相关性和理由质量，不要堆砌备选。"
        )

    # 构建对话上下文段落
    history_text = context.get("history_text_recent", "") if isinstance(context, dict) else ""
    is_followup = context.get("is_followup", False) if isinstance(context, dict) else False
    context_section = build_context_section(history_text, is_followup=is_followup)

    # simple_fact 用简洁事实型 prompt，其他用推荐型 prompt
    if query_type == "simple_fact":
        system_prompt = _SIMPLE_FACT_SYSTEM.format(context_section=context_section)
    else:
        system_prompt = _ANSWER_SYSTEM.format(
            context_section=context_section,
            constraint_section=constraint_section,
        )

    user_prompt = _ANSWER_USER.format(
        query=query,
        structure=structure,
        merged_results=merged_results,
    )
    if query_type == "recommendation" and recommendation_candidates:
        user_prompt = _CANDIDATE_USER.format(
            query=query,
            structure=structure,
            candidates=_format_recommendation_candidates(recommendation_candidates),
            constraint_section=constraint_section,
        )

    resp = await llm_ainvoke_with_retry(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # 更新对话状态
    result = {"messages": [resp], "previous_intent": query_type}

    # 从 merge_results 提取推荐作品（结构化来源，可靠）
    if query_type == "recommendation":
        recent = [
            {"name": candidate["title"], "type": "anime"}
            for candidate in recommendation_candidates[:5]
            if candidate.get("title")
        ] or _extract_recent_from_merged(merged_results)
        if recent:
            result["recent_entities"] = recent

    # 同时把命名实体（角色/梗名）也写入 recent_entities，供下轮指代解析
    entity_name = state.get("entity_name", "")
    entity_type = state.get("entity_type", "")
    if entity_name and entity_type in ("character", "alias"):
        existing = result.get("recent_entities", [])
        if not any(e.get("name") == entity_name for e in existing):
            # 限制 recent_entities 最多 5 个，防止长对话累积导致 prompt 膨胀
            result["recent_entities"] = (
                [{"name": entity_name, "type": entity_type}] + existing
            )[:5]

    logger.info(f"  answer 耗时 {time.time()-t0:.1f}s")
    return result


def _build_constraint_section(constraints: dict) -> str:
    """把结构化的跨轮约束转成 prompt 中的明确约束段落。"""
    if not constraints:
        return "无额外约束。"
    lines = []
    if constraints.get("exclude_same_series"):
        series = constraints.get("excluded_series", [])
        if series:
            lines.append(f"- 排除同系列作品：{', '.join(series)} 及其续作、剧场版、衍生作品")
        else:
            lines.append("- 排除同系列作品")
    if constraints.get("exclude_movies"):
        lines.append("- 排除剧场版")
    if not lines:
        return "无额外约束。"
    return "\n".join(lines)


def _format_recommendation_candidates(candidates: list[dict]) -> str:
    lines = []
    for index, candidate in enumerate(candidates[:8], 1):
        fields = [f"作品: {candidate.get('title', '未知')}"]
        if candidate.get("score") not in ("", None):
            fields.append(f"评分: {candidate['score']}")
        if candidate.get("tags"):
            fields.append("标签: " + "、".join(candidate["tags"][:8]))
        if candidate.get("studio"):
            fields.append(f"制作公司: {candidate['studio']}")
        if candidate.get("date"):
            fields.append(f"日期: {candidate['date']}")
        if candidate.get("evidence"):
            fields.append("证据: " + " | ".join(candidate["evidence"][:2]))
        lines.append(f"{index}. " + "\n   ".join(fields))
    return "\n".join(lines)


def _extract_recent_from_merged(merged: str) -> list[dict]:
    """从 merge_results 中提取作品名

    merge_results 格式: "**命运石之门**（评分8.7）..."
    用正则提取 **粗体** 内的番剧名，严格过滤避免误抓"**评分**""**声优**"等字段标注。
    """
    if not merged or merged == "(分析结果为空)":
        return []

    names = re.findall(r"\*\*(.+?)\*\*", merged)
    entities = []
    # 已知非番剧名粗体词（Expert 输出里常见的字段标注/章节标题）
    skip_keywords = {
        "推荐", "分析", "总结", "对比", "结论", "注意", "提示",
        "评分", "声优", "制作", "导演", "标签", "排名", "年份",
        "概述", "简介", "详情", "理由", "优缺点", "亮点", "缺点",
        "观众评论", "用户评价", "综合", "推荐理由", "参考",
    }
    for name in names:
        name = name.strip()
        # 长度过滤: 番剧名通常 2-15 字符，过短/过长都不是
        if not (2 <= len(name) <= 15):
            continue
        # 含中文标点的多为标注（"评分:" "声优:"）
        if any(p in name for p in ("：", ":", "，", "。", "、")):
            continue
        # 跳过已知字段名
        if name in skip_keywords:
            continue
        # 含 skip_keywords 的也跳过（如"评分说明""制作组"）
        if any(kw in name for kw in skip_keywords):
            continue
        # 全英文/纯数字的也跳过（番剧名几乎都有中文）
        if name.isascii() or name.isdigit():
            continue
        entities.append({"name": name, "type": "anime"})
        if len(entities) >= 5:
            break
    return entities
