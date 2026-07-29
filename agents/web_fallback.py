"""Web Fallback - 按需触发的联网搜索回退节点

触发条件（任一满足即触发）:
  1. Planner 明确要求 need_web=true
  2. 检索结果为空（无 shared_context）
  3. 所有 Expert confidence < threshold

注意: 这不是常驻 Agent，是条件触发的回退节点。
"""
import logging
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# 模块级 prompt（避免每次调用重建字符串）
_EXTRACT_PROMPT = """从下面的联网搜索结果里，把跟"查询"相关的 ACG 番剧事实抽出来，按短条目排列。

要求：
- 只抽搜索结果里明确出现过的信息（作品名、评分、播出年份、制作公司、观众评价原文片段）。
- 不要编造搜索结果里没有的数字或结论；没找到就不写这一条。
- 每条一行，尽量短，不要写成推荐话术，不要给评价倾向。
- 如果搜索结果里完全没有跟查询相关的 ACG 内容，直接输出一行：无相关信息。

查询: {query}

搜索结果:
{results}

抽取结果:"""


def should_trigger_web(state: dict) -> bool:
    """判断是否需要触发 Web fallback"""
    from tools.registry import tool_registry

    if not tool_registry.is_enabled("search_web"):
        return False

    plan = state.get("plan", {})
    if plan.get("need_web"):
        return True

    if not state.get("shared_context"):
        return True

    import config
    from agents.evaluator import current_expert_results
    results = current_expert_results(state)
    if results and all(r.get("confidence", 0) < config.CONFIDENCE_THRESHOLD for r in results):
        return True

    return False


async def web_fallback_node(state: dict) -> dict:
    """LangGraph 节点: Web Fallback - 联网搜索补充信息（按需启用）"""
    from tools.registry import tool_registry

    if not tool_registry.is_enabled("search_web"):
        return {
            "merged_results": state.get("merged_results", ""),
            "termination_reason": state.get("termination_reason") or "replan_exhausted",
        }

    query = state.get("resolved_query") or state.get("original_query", "")
    search_web = tool_registry.get_callable("search_web")
    if not search_web:
        return {"merged_results": state.get("merged_results", ""), "termination_reason": "web_fallback"}

    from llms import simple_LLM, llm_ainvoke_with_retry

    try:
        logger.info("web_fallback attempt=%s", state.get("attempt", 0))
        search_text = search_web.invoke(f"{query} 动漫 番剧 推荐 评分 评价")
        if not search_text or len(search_text) < 30:
            # 无有效结果时不追加任何文本，保持 merged_results 原值
            # answer 节点会基于现有 merged_results 生成回答
            logger.info(f"  web_fallback: 联网搜索无有效结果")
            return {"merged_results": state.get("merged_results", ""), "termination_reason": "web_fallback"}

        # 用轻量 LLM 提取关键信息
        resp = await llm_ainvoke_with_retry(simple_LLM, [
            HumanMessage(content=_EXTRACT_PROMPT.format(
                query=query,
                results=search_text[:2000],
            )),
        ])

        extracted = resp.content.strip()
        # 抽取器判定为空时不污染 merged_results
        if not extracted or "无相关信息" in extracted:
            logger.info(f"  web_fallback: LLM 抽取判定无相关信息")
            return {"merged_results": state.get("merged_results", ""), "termination_reason": "web_fallback"}

        web_info = f"\n\n---\n[联网搜索结果]\n{extracted}"
        merged = state.get("merged_results", "") + web_info

        return {"merged_results": merged, "termination_reason": "web_fallback"}

    except Exception as e:
        # 异常时不污染 merged_results（避免错误信息被 answer 当正文输出给用户）
        # 只记日志，answer 节点基于原 merged_results 生成回答
        logger.warning(f"  web_fallback 失败: {e}")
        return {"merged_results": state.get("merged_results", ""), "termination_reason": "web_fallback"}
