"""Metadata Reasoner Agent — 基于结构化元数据 + 语义上下文做推理推荐

输入:
  - metadata: Metadata Index 查询结果（list[dict] 结构化数据）
  - shared_context: Dense/Sparse 语义文本片段（list[str]）
  - query: 用户查询

输出:
  ExpertResult {answer, confidence, evidence}
"""
import json
import time
import logging
from langchain_core.messages import HumanMessage, SystemMessage
import config

logger = logging.getLogger(__name__)

_REASONER_SYSTEM = """你是阅番无数的资深二次元。你的任务是根据番剧数据和观众评论，给出有观点的分析。

## 你会收到的数据
1. 结构化元数据（JSON）: 番剧名、Bangumi 评分、排名、标签、制作公司、导演、声优等
2. 语义上下文（文本）: 含"观众评论: xxx"字段、番剧描述等

## 输出格式
严格 JSON:
{
  "answer": "口语化的分析结论 — 每个观点都要有依据",
  "confidence": 0.85,
  "evidence": ["来源A: 具体数据", "来源B: 观众评论原文片段"]
}

## 输出原则
- **证据导向**: 每个判断后面跟依据，如 "这部在 Bangumi 上 8.5 分排前 50，口碑很稳"
- **评论优先**: 上下文里有"观众评论:"的话，优先引用 — 比干数据有说服力
- **指出争议**: 评论明显分歧时如实反映，显得客观
- **不用套话**: 别写"根据数据显示""综合分析认为" — 直接说结论

## 风格
像在动漫群里跟群友聊番，不是写分析报告。数据不够就如实说，别硬编。"""

_REASONER_USER = """## 用户问题
{query}

## 结构化元数据
```json
{metadata_json}
```

## 语义上下文
{context_text}

请基于以上数据给出分析结论。"""


async def metadata_reasoner_node(state: dict) -> dict:
    """LangGraph 节点: Metadata Reasoner"""
    t0 = time.time()
    from llms import answer_LLM, simple_LLM, llm_ainvoke_with_retry

    query = state.get("resolved_query") or state.get("original_query", "")
    metadata = state.get("metadata", [])
    shared_context = state.get("shared_context", [])

    logger.debug(
        f"Metadata Reasoner 收到 state: "
        f"metadata={len(metadata)}条, context={len(shared_context)}条, "
        f"query='{query[:50]}'"
    )

    # 格式化元数据
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else "[]"
    # 限制大小
    if len(metadata_json) > 3000:
        metadata_json = metadata_json[:3000] + "\n... (truncated)"

    # 格式化上下文
    context_text = "\n\n---\n\n".join(shared_context[:5]) if shared_context else "(无)"
    if len(context_text) > 2000:
        context_text = context_text[:2000] + "\n... (truncated)"

    llm = answer_LLM.bind(temperature=config.EXPERT_TEMPERATURE)

    # simple_fact 查询用轻量模型（快 + 省）
    plan = state.get("plan", {})
    if plan.get("query_type") == "simple_fact":
        llm = simple_LLM.bind(temperature=config.EXPERT_TEMPERATURE)

    resp = await llm_ainvoke_with_retry(llm, [
        SystemMessage(content=_REASONER_SYSTEM),
        HumanMessage(content=_REASONER_USER.format(
            query=query,
            metadata_json=metadata_json,
            context_text=context_text,
        )),
    ])

    # 解析结果
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

    logger.info(f"  metadata_reasoner 耗时 {time.time()-t0:.1f}s")
    return {
        "expert_results": [result],
    }
