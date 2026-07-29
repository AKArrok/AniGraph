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
from agents.prompts import BANNED_PHRASES, INTERNAL_TERMS

logger = logging.getLogger(__name__)

_REASONER_SYSTEM = f"""基于结构化元数据和观众评论，给出有依据的分析结论。你的输出会被下游节点原样引用，所以内容质量和语言风格直接影响用户可见回答。

## 你会收到的数据
1. 结构化元数据（JSON）: 番剧名、Bangumi 评分、排名、标签、制作公司、导演、声优等。
2. 语义上下文（文本）: 可能含"观众评论: xxx"字段、番剧描述等。

## 输出格式（严格 JSON）
{{
  "answer": "分析结论文本",
  "confidence": 0.0-1.0,
  "evidence": ["来源A: 具体数据或评论原文片段", "..."]
}}

## 事实约束
- 只能使用元数据和上下文里给出的信息，不用常识补，不跨作品拼数据。
- 每个判断后面必须跟得上依据（评分/排名/标签/评论片段），说不出依据就不写。
- 观众评论有分歧时如实反映两侧观点，不要单选一面。
- 数据不足以支撑结论时，`answer` 直接说"手上数据有限，只能说到这里"，`confidence` 相应压低。

## 语言风格
- `answer` 用简洁中文，直接说结论 + 依据，不铺垫、不做总结段、不写引导话术（"值得一看""可以尝试"）。
- 评论片段引用原文，不改写不加戏。
- 不评价用户品味，不猜用户偏好，不说"适合你"这种代入话。

## 禁语
- 套话：{BANNED_PHRASES}、"根据数据显示"、"综合分析认为"。
- 空形容词：不靠"经典/精彩/神作/引人入胜"来代替具体依据。
- 内部术语：{INTERNAL_TERMS}。
- 元自我描述："作为一个 AI/助手/分析器"、"根据我的分析"。"""

_REASONER_USER = """## 用户问题
{query}

## 结构化元数据
```json
{metadata_json}
```

## 语义上下文
{context_text}

## 前序 Expert 结论（串行核验时提供，"(无)" 表示本次无前序结论，可直接忽略）
{peer_findings}"""


def _format_peer_findings(state: dict) -> str:
    execution_id = state.get("execution_id", "")
    attempt = state.get("attempt", 0)
    findings = [
        r for r in state.get("expert_results", [])
        if r.get("execution_id") == execution_id
        and r.get("attempt", 0) == attempt
        and r.get("expert") != "metadata_reasoner"
    ]
    return "\n".join(r.get("answer", "") for r in findings if r.get("answer")) or "(无)"


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
            peer_findings=_format_peer_findings(state),
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

    result.update({
        "expert": "metadata_reasoner",
        "attempt": state.get("attempt", 0),
        "execution_id": state.get("execution_id", ""),
    })

    logger.info(f"  metadata_reasoner 耗时 {time.time()-t0:.1f}s")
    return {
        "expert_results": [result],
    }
