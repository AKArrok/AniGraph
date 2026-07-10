"""Similar Expert Agent — 基于 Embedding 召回 + Metadata Index 发现相似作品

工作流:
  1. 从用户查询中提取目标番剧（如有）
  2. Metadata Index: 同标签/同公司/同导演等结构相似
  3. Embedding 检索: 语义相似 TopK
  4. 合并去重 → LLM 排序 + 解释

输出:
  ExpertResult {answer, confidence, evidence}
"""
import json
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config

logger = logging.getLogger(__name__)

_SIMILAR_SYSTEM = """你是资深动漫宅，专门帮人找"像 XX 的番"。用聊天的方式推荐。

## 输出格式
严格 JSON:
{
  "answer": "口语化的推荐，每个推荐2-3句",
  "confidence": 0.85,
  "evidence": ["引用来源"]
}

## 推荐角度（每个推荐至少覆盖 2 个维度）
- 剧情走向: 叙事结构、反转密度、节奏感
- 角色塑造: 人设深度、角色弧光
- 世界观: 背景设定复杂度
- 观看体验: 情感冲击、代入感
- 观众反馈: 上下文中如有"观众评论"，引用作为佐证

## 不要做的事
- 别只说"风格相似"，要说具体哪里像
- 别罗列评分标签，融入句子里
- 评分差距大的要提
- 也说说差异点和适合谁/不适合谁"""

_SIMILAR_USER = """## 用户查询
{query}

## 结构相似候选（同标签/同公司/同导演）
{structured_candidates}

## 语义相似候选（Embedding 召回）
{semantic_candidates}

请推荐最相似的作品并说明理由。"""


def _find_structured_similar(query: str, state: dict) -> list[dict]:
    """通过 Metadata Index 查找结构相似作品: 同标签/同公司/同导演"""
    try:
        from agents.metadata_index import index
        metadata = state.get("metadata", [])

        if not metadata:
            return []

        similar = set()
        for item in metadata[:3]:
            # 同标签
            tags = item.get("tags", [])
            if tags:
                for tag in tags[:2]:
                    results = index.search(tag=tag, limit=5)
                    for r in results:
                        similar.add(r.get("name_cn", ""))

            # 同制作公司
            studio = item.get("studio", "")
            if studio:
                results = index.search(studio=studio, limit=3)
                for r in results:
                    similar.add(r.get("name_cn", ""))

        # 返回 Metadata Index 中的完整信息
        candidates = []
        for name in similar:
            item = index.get_by_alias(name)
            if item:
                candidates.append(item)

        return candidates[:10]
    except Exception:
        return []


def _format_candidates(candidates: list[dict]) -> str:
    """格式化候选列表为文本"""
    if not candidates:
        return "(无)"

    lines = []
    for i, c in enumerate(candidates[:10], 1):
        name = c.get("name_cn") or c.get("name", "未知")
        score = c.get("score", "?")
        tags = ", ".join(c.get("tags", [])[:5])
        studio = c.get("studio", "")
        date = c.get("date", "")
        lines.append(
            f"{i}. {name} | 评分: {score} | 标签: {tags} | "
            f"公司: {studio} | 日期: {date}"
        )

    return "\n".join(lines)


async def similar_expert_node(state: dict) -> dict:
    """LangGraph 节点: Similar Expert"""
    t0 = time.time()
    from llms import answer_LLM, simple_LLM

    query = state.get("resolved_query") or state.get("original_query", "")

    # 1. 结构相似候选（从 Metadata Index）
    structured = _find_structured_similar(query, state)
    structured_text = _format_candidates(structured)

    # 2. 语义相似候选（从 shared_context）
    shared_context = state.get("shared_context", [])
    semantic_text = "\n\n---\n\n".join(shared_context[:5]) if shared_context else "(无)"

    logger.debug(
        f"Similar Expert 收到 state: "
        f"metadata={len(state.get('metadata', []))}条, "
        f"context={len(shared_context)}条, "
        f"structured_candidates={len(structured)}个"
    )
    if len(semantic_text) > 2000:
        semantic_text = semantic_text[:2000] + "\n... (truncated)"

    # 3. 如果没有数据，降 confidence
    no_data = (not structured) and (not shared_context)

    if no_data:
        return {
            "expert_results": [{
                "answer": "当前知识库中没有足够的相似作品数据。",
                "confidence": 0.2,
                "evidence": [],
            }],
        }

    llm = answer_LLM.bind(temperature=config.EXPERT_TEMPERATURE)

    # simple_fact 查询用轻量模型（快 + 省）
    plan = state.get("plan", {})
    if plan.get("query_type") == "simple_fact":
        llm = simple_LLM.bind(temperature=config.EXPERT_TEMPERATURE)

    resp = llm.invoke([
        SystemMessage(content=_SIMILAR_SYSTEM),
        HumanMessage(content=_SIMILAR_USER.format(
            query=query,
            structured_candidates=structured_text,
            semantic_candidates=semantic_text,
        )),
    ])

    # 解析
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "answer": text[:500],
            "confidence": 0.5,
            "evidence": ["LLM 输出非 JSON 格式"],
        }

    logger.info(f"  similar_expert 耗时 {time.time()-t0:.1f}s")
    return {
        "expert_results": [result],
        "messages": [resp],
    }
