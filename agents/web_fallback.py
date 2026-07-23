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
_EXTRACT_PROMPT = """从以下搜索结果中提取与 ACG 番剧相关的关键信息，简洁列出:

查询: {query}

搜索结果:
{results}

关键信息（番剧名、评分、推荐理由等）:"""


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
    results = state.get("expert_results", [])
    if results and all(r.get("confidence", 0) < config.CONFIDENCE_THRESHOLD for r in results):
        return True

    return False


async def web_fallback_node(state: dict) -> dict:
    """LangGraph 节点: Web Fallback - 联网搜索补充信息（按需启用）"""
    from tools.registry import tool_registry

    if not tool_registry.is_enabled("search_web"):
        return {"merged_results": state.get("merged_results", "")}

    query = state.get("resolved_query") or state.get("original_query", "")
    search_web = tool_registry.get_callable("search_web")
    if not search_web:
        return {"merged_results": state.get("merged_results", "")}

    from llms import simple_LLM, llm_ainvoke_with_retry

    try:
        search_text = search_web.invoke(f"{query} 动漫 番剧 推荐 评分 评价")
        if not search_text or len(search_text) < 30:
            # 无有效结果时不追加任何文本，保持 merged_results 原值
            # answer 节点会基于现有 merged_results 生成回答
            logger.info(f"  web_fallback: 联网搜索无有效结果")
            return {"merged_results": state.get("merged_results", "")}

        # 用轻量 LLM 提取关键信息
        resp = await llm_ainvoke_with_retry(simple_LLM, [
            HumanMessage(content=_EXTRACT_PROMPT.format(
                query=query,
                results=search_text[:2000],
            )),
        ])

        web_info = f"\n\n---\n[联网搜索结果]\n{resp.content.strip()}"
        merged = state.get("merged_results", "") + web_info

        return {"merged_results": merged}

    except Exception as e:
        # 异常时不污染 merged_results（避免错误信息被 answer 当正文输出给用户）
        # 只记日志，answer 节点基于原 merged_results 生成回答
        logger.warning(f"  web_fallback 失败: {e}")
        return {"merged_results": state.get("merged_results", "")}
