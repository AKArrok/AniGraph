"""Answer Agent — 综合所有 Expert 结果，生成自然口语化回答

输入:
  - merged_results: Merge 后的综合结果文本
  - plan: ExecutionPlan（含 query_type）
  - original_query: 用户原始查询

输出:
  自然语言回答（直接写入 messages）
"""
from langchain_core.messages import HumanMessage, SystemMessage
import config

_ANSWER_SYSTEM = """你是资深二次元，帮朋友推荐番剧。你不是 AI，不是客服，不写报告。

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
- 禁止: "推荐理由""综合分析""值得注意的是""综上所述""根据分析""笔者认为"
- 禁止: 每部作品用相同句式罗列
- 禁止: 编造分析结果里不存在的番剧名、评分、评论
- 不确定的信息直接说"这个我不太确定"，别硬编

## 心态
Expert 输出是"找证据的人"写的，你的任务是把这些证据用聊天的方式讲出来。像刚从 Bangumi 逛了一圈回来跟朋友分享。"""

_ANSWER_USER = """## 用户问题
{query}

## 回答结构指引
{structure}

## Expert 分析结果
{merged_results}

请生成回答。"""


async def answer_node(state: dict) -> dict:
    """最终回答节点: 重组 Expert 结果，用口语化方式输出"""
    from llms import answer_LLM

    query = state.get("original_query", "")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "unknown")

    # 闲聊 & 简单事实查询用小模型（快 + 省），复杂推理用大模型
    if query_type in ("chat", "simple_fact"):
        from llms import simple_LLM
        if query_type == "chat":
            # 闲聊无 Expert 结果，直接用用户消息回复
            resp = simple_LLM.invoke([HumanMessage(content=query)])
            return {"messages": [resp]}
        llm = simple_LLM.bind(temperature=config.ANSWER_TEMPERATURE)
    else:
        llm = answer_LLM.bind(temperature=config.ANSWER_TEMPERATURE)

    merged_results = state.get("merged_results", "")
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

    resp = llm.invoke([
        SystemMessage(content=_ANSWER_SYSTEM),
        HumanMessage(content=_ANSWER_USER.format(
            query=query,
            structure=structure,
            merged_results=merged_results,
        )),
    ])

    return {"messages": [resp]}
