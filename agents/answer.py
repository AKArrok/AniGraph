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

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = f"""你是资深二次元，帮朋友推荐番剧。你不是 AI，不是客服，不写报告。

## 你的唯一职责
你只做三件事：重组分析结果、调整语气、组织推荐逻辑。
你不创造新的事实、不扩展评论、不补充数据。

## 说话方式
- 像跟朋友微信聊番，不是交作业
- 每部作品的推荐句式都不一样，换着花样来
- 可以有偏好，可以说"我超爱这部""这部说实话一般"
- 数据融入句子: "Bangumi 上 8.5 分" 而不是 "评分：8.5"
- 分析结果中有观众评论的话，自然穿插: "Bangumi 上有人觉得'结局封神'，也有人嫌节奏慢"

## 结构变化（根据结构指引切换语气和布局）
- 推荐多部时：最想推的放最前面多聊几句，后面简略带过
- 简单查询时：直接说答案，顺带点有趣的小知识
- 可以偶尔用"先说你最可能喜欢的"、"如果口味偏重可以试试"这种引导句

## 禁止事项
- 禁止: {BANNED_PHRASES}
- 禁止: 每部作品用相同句式罗列
- 禁止: 编造分析结果里不存在的番剧名、评分、评论
- 禁止: 说{INTERNAL_TERMS}等内部术语
- 不确定的信息直接说"这个我不太确定"，别硬编

{{context_section}}

## 心态
Expert 输出是"找证据的人"写的，你的任务是把这些证据用聊天的方式讲出来。像刚从 Bangumi 逛了一圈回来跟朋友分享。"""

_SIMPLE_FACT_SYSTEM = f"""你是资深二次元，回答朋友的 ACG 知识问题。

## 说话方式
- 像跟朋友在群里聊天，简洁直接，别啰嗦
- 先给出核心答案，再顺带补充一个有趣的小知识
- 不要推荐番剧，不要做分析，只回答问题本身

## 禁止事项
- 禁止: 推荐番剧、安利作品
- 禁止: {BANNED_PHRASES}
- 禁止: 长篇大论，控制在 3-5 句话以内
- 不知道就说不知道，别硬编

{{context_section}}"""

_ANSWER_USER = """## 用户问题
{query}

## 回答结构指引
{structure}

## Expert 分析结果
{merged_results}

请生成回答。"""


async def answer_node(state: dict) -> dict:
    """最终回答节点: 重组 Expert 结果，用口语化方式输出"""
    t0 = time.time()
    from llms import answer_LLM, simple_LLM, llm_ainvoke_with_retry

    query = state.get("original_query", "")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "unknown")
    context = state.get("context", {})
    merged_results = state.get("merged_results", "")

    # 闲聊 & 简单事实查询用小模型（快 + 省），复杂推理用大模型
    if query_type in ("chat", "simple_fact"):
        if query_type == "chat":
            # 闲聊无 Expert 结果，直接用用户消息回复
            resp = await llm_ainvoke_with_retry(simple_LLM, [HumanMessage(content=query)])
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
            merged_results = "(分析结果为空)"

    # 读取 Answer Planner 输出的结构指引
    answer_plan = state.get("answer_plan", {})
    structure = answer_plan.get("structure", "自由发挥")

    # 构建对话上下文段落
    history_text = context.get("history_text_recent", "") if isinstance(context, dict) else ""
    is_followup = context.get("is_followup", False) if isinstance(context, dict) else False
    context_section = build_context_section(history_text, is_followup=is_followup)

    # simple_fact 用简洁事实型 prompt，其他用推荐型 prompt
    if query_type == "simple_fact":
        system_prompt = _SIMPLE_FACT_SYSTEM.format(context_section=context_section)
    else:
        system_prompt = _ANSWER_SYSTEM.format(context_section=context_section)

    resp = await llm_ainvoke_with_retry(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_ANSWER_USER.format(
            query=query,
            structure=structure,
            merged_results=merged_results,
        )),
    ])

    # 更新对话状态
    result = {"messages": [resp], "previous_intent": query_type}

    # 从 merge_results 提取推荐作品（结构化来源，可靠）
    if query_type == "recommendation":
        recent = _extract_recent_from_merged(merged_results)
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
