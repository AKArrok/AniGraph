"""Planner Agent — LLM 分析用户查询，输出 ExecutionPlan 驱动图编排

职责:
  1. LLM 分类: 判断查询类别（metadata/semantic/mixed/chat）和查询类型
  2. 决定查询优化策略（direct/rewrite/hyde/decompose）
  3. 决策需要哪些 Expert（metadata_reasoner/similar_expert）
  4. 决定 Expert 并行还是串行
  5. 决定是否需要联网

输入: 用户原始查询
输出: ExecutionPlan dict
"""
import json
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config

logger = logging.getLogger(__name__)


# ── LLM 意图分类 Prompt（simple_LLM 一次性完成全部分类）──────

_CLASSIFIER_SYSTEM = """你是 ACG 番剧查询分类器。分析用户查询，输出分类结果。

## 查询类别 (query_category)
- metadata: 查结构化元数据（评分/声优/导演/公司/标签/年份等），通过数据库能回答
  例: "进击的巨人评分"、"MAPPA作品"、"2024热血番"
- semantic: 开放性问题需语义理解（评价/口碑/观后感/推荐相似作品），需向量检索
  例: "有没有类似钢炼的番"、"为什么EVA是神作"、"催泪番推荐"
- mixed: 指定番剧名 + 评价/推荐意图，需 Metadata 查信息 + Semantic 做推荐
  例: "碧蓝之海怎么样？"、"进击的巨人好看吗"、"RE:0值得看吗"
- chat: 闲聊/问候/非番剧问题
  例: "你好"、"谢谢"、"你是谁"、"你能做什么"

## 查询类型 (query_type)
- simple_fact: 查已知事物的具体属性（评分/声优/导演/介绍/标签等）
- recommendation: 要求推荐/发现新番剧（含"推荐"、"类似的"、"有哪些"等意图）
- comparison: 对比多部番剧
- chat: 闲聊/问候

## 查询优化策略 (rewrite_strategy)
- direct: 简单直接查询，不需重写
- rewrite: 需从多角度扩展查询
- hyde: 深度分析/评价类（含"为什么""好在哪""区别"等）
- decompose: 含多个子问题

## 专家选择 (experts)
- metadata_reasoner: 需查结构化数据（评分/标签/公司/声优等）时必选
- similar_expert: 需语义推荐/对比/评价时必选
- 两者都用: mixed 类型或复杂的推荐对比查询

## 并行 (parallel)
- 两个 expert 都需要 → true
- 只需一个 → false

## 联网 (need_web)
- 查询可能超出知识库范围（冷门番剧/最新资讯/实时数据）→ true
- 知识库能覆盖 → false

{history_section}

输出严格 JSON，不要 markdown 标记:
{{"query_category":"mixed","query_type":"recommendation","rewrite_strategy":"direct","experts":["metadata_reasoner","similar_expert"],"parallel":true,"need_web":false,"reasoning":"简短说明"}}"""


def _classify_with_llm(query: str, history_text: str = "") -> dict:
    """使用 simple_LLM 一次调用完成全部意图分类

    Returns:
        dict: query_category, query_type, rewrite_strategy, experts, parallel, need_web, reasoning
    """
    from llms import simple_LLM

    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史（仅供参考，用于理解指代和上下文）\n"
            f"{history_text}\n\n"
            f"注意: 需独立判断当前查询的意图，不受历史类型影响。"
        )

    system_prompt = _CLASSIFIER_SYSTEM.format(history_section=history_section)

    try:
        resp = simple_LLM.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户查询: {query}"),
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        result = json.loads(text)
    except Exception as e:
        logger.warning(f"LLM 分类失败: {e}，使用默认 mixed 策略")
        result = {
            "query_category": "mixed",
            "query_type": "recommendation",
            "rewrite_strategy": "rewrite",
            "experts": ["metadata_reasoner", "similar_expert"],
            "parallel": True,
            "need_web": False,
            "reasoning": "LLM 分类失败，默认推荐策略",
        }

    # 确保必要字段
    result.setdefault("query_category", "mixed")
    result.setdefault("query_type", "recommendation")
    result.setdefault("rewrite_strategy", "direct")
    result.setdefault("experts", [])
    result.setdefault("parallel", False)
    result.setdefault("need_web", False)
    result.setdefault("reasoning", "")

    # ── 一致性修正 ──
    if result["query_category"] == "chat" or result["query_type"] == "chat":
        result["query_category"] = "chat"
        result["query_type"] = "chat"
        result["experts"] = []
        result["parallel"] = False
        result["rewrite_strategy"] = "direct"
        return result

    if result["query_type"] == "simple_fact":
        result["experts"] = ["metadata_reasoner"]
        result["parallel"] = False

    if result["query_category"] == "metadata":
        result["experts"] = ["metadata_reasoner"]
        result["parallel"] = False
        result["rewrite_strategy"] = "direct"

    if result["query_category"] == "semantic":
        result["experts"] = ["similar_expert"]
        result["parallel"] = False

    return result


# ── Planner 深化 Prompt（mixed/复杂查询用 answer_LLM 细化）──

_PLANNER_SYSTEM = """你是 ACG 番剧推荐系统的规划器。以下已有初步分类，请验证并细化执行计划。

## 查询类型
- simple_fact: 查询特定番剧的评分/声优/导演/公司等事实信息
- recommendation: 根据条件推荐番剧
- comparison: 对比多部番剧
- chat: 闲聊

## 查询优化策略
- direct: 简单查询
- rewrite: 需多角度扩展
- hyde: 深度分析/评价类
- decompose: 含多子问题

## Expert 选择
- metadata_reasoner: 涉及评分/标签/公司/声优等结构化数据
- similar_expert: 涉及相似推荐/对比/语义理解

{history_section}

初步分类: {classification}

输出最终执行计划（严格 JSON，不要 markdown）:
{{"query_type":"recommendation","rewrite_strategy":"rewrite","experts":["metadata_reasoner","similar_expert"],"parallel":true,"need_web":false,"reasoning":"..."}}"""


def plan(query: str, history_text: str = "") -> dict:
    """Planner: simple_LLM 一次分类 → 简单查询直接返回，复杂查询用 answer_LLM 深化"""
    from llms import answer_LLM

    # Step 1: simple_LLM 一次性完成全部意图分类
    plan_dict = _classify_with_llm(query, history_text)

    # 闲聊直接返回（不需要 answer_LLM 深化）
    if plan_dict["query_type"] == "chat":
        return plan_dict

    # 简单查询（metadata/semantic 单路）直接返回 LLM 分类结果
    if plan_dict["query_category"] in ("metadata", "semantic"):
        return plan_dict

    # ── mixed / 复杂查询: 用 answer_LLM 深化计划 ──
    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史\n{history_text}\n\n"
            f"注意: 独立判断当前查询意图，不受历史影响。"
        )

    classification = json.dumps(plan_dict, ensure_ascii=False)
    system_prompt = _PLANNER_SYSTEM.format(
        history_section=history_section,
        classification=classification,
    )

    llm = answer_LLM.bind(temperature=config.PLANNER_TEMPERATURE)
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {query}"),
    ])

    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        refined = json.loads(text)
        # 以 simple_LLM 分类为基础，answer_LLM 细化为补充
        plan_dict.update(refined)
    except json.JSONDecodeError:
        logger.warning("answer_LLM 计划解析失败，使用 simple_LLM 分类")

    # 确保必要字段
    plan_dict.setdefault("query_type", "recommendation")
    plan_dict.setdefault("rewrite_strategy", "rewrite")
    plan_dict.setdefault("experts", ["metadata_reasoner", "similar_expert"])
    plan_dict.setdefault("parallel", True)
    plan_dict.setdefault("need_web", False)
    plan_dict.setdefault("reasoning", "")

    # 一致性修正
    if plan_dict["query_type"] == "chat":
        plan_dict["experts"] = []
        plan_dict["parallel"] = False
        plan_dict["rewrite_strategy"] = "direct"
    elif plan_dict["query_type"] == "simple_fact":
        plan_dict["experts"] = ["metadata_reasoner"]
        plan_dict["parallel"] = False

    return plan_dict


async def planner_node(state: dict) -> dict:
    """LangGraph 节点: Planner"""
    t0 = time.time()
    query = state.get("resolved_query", "") or state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    # 构建对话历史文本
    context = state.get("context", {})
    history_text = ""
    if isinstance(context, dict) and context.get("history"):
        lines = []
        for r in context["history"]:
            lines.append(f"用户: {r['user']}")
            lines.append(f"助手: {r['assistant']}")
        history_text = "\n".join(lines)

    execution_plan = plan(query, history_text=history_text)

    if not isinstance(execution_plan, dict):
        execution_plan = dict(execution_plan)

    # 根据实体解析结果调整 plan（基于数据，非正则）
    entity_confidence = state.get("entity_confidence", 1.0)
    entity_type = state.get("entity_type", "")

    if entity_confidence < 0.5:
        execution_plan["need_web"] = True

    if entity_type == "meme":
        execution_plan["need_web"] = True

    logger.info(
        f"  planner 耗时 {time.time()-t0:.1f}s "
        f"(category={execution_plan.get('query_category')}, "
        f"type={execution_plan.get('query_type')}, "
        f"experts={execution_plan.get('experts')})"
    )

    return {"plan": execution_plan, "original_query": query}
