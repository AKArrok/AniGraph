"""Evidence quality gate between Expert Merge and answer generation."""
import logging

import config
from agents.state import EvaluationResult

logger = logging.getLogger(__name__)


def current_expert_results(state: dict) -> list[dict]:
    """Return only results produced by this request and this attempt."""
    execution_id = state.get("execution_id", "")
    attempt = state.get("attempt", 0)
    return [
        result for result in state.get("expert_results", [])
        if result.get("execution_id") == execution_id
        and result.get("attempt", 0) == attempt
    ]


def _deterministic_evaluation(state: dict) -> EvaluationResult:
    results = current_expert_results(state)
    plan = state.get("plan", {})
    planned = list(dict.fromkeys(plan.get("experts", [])))
    issues: list[str] = []
    missing: list[str] = []

    usable = [r for r in results if r.get("answer") and r.get("evidence")]
    if not usable:
        issues.append("no_evidence")

    if results and all(
        float(r.get("confidence", 0)) < config.CONFIDENCE_THRESHOLD
        for r in results
    ):
        issues.append("low_confidence")

    completed = {r.get("expert") for r in results}
    if len(planned) > 1:
        missing = [expert for expert in planned if expert not in completed]
        if missing:
            issues.append("missing_dimension")

    query_type = plan.get("query_type")
    evidence_count = sum(len(r.get("evidence", [])) for r in results)
    if query_type in ("comparison", "recommendation"):
        minimum = 2 if query_type == "comparison" else 1
        if results and evidence_count < minimum and "no_evidence" not in issues:
            issues.append("missing_dimension")
            missing.append("evidence_coverage")

    confidences = [float(r.get("confidence", 0)) for r in results]
    confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
    coverage = len(completed & set(planned)) / len(planned) if planned else 1.0
    score = round(min(1.0, confidence_score * 0.7 + coverage * 0.3), 3)
    verdict = "replan" if issues else "pass"
    feedback = "; ".join(issues) if issues else "当前 Expert 证据完整且置信度达标"
    return EvaluationResult(
        verdict=verdict,
        score=score,
        issues=list(dict.fromkeys(issues)),
        missing_dimensions=list(dict.fromkeys(missing)),
        feedback=feedback,
    )


def _may_conflict(results: list[dict]) -> bool:
    """Cheap gate for semantic conflict judging; false negatives are preferable to fixed cost."""
    if len(results) < 2:
        return False
    answers = [str(r.get("answer", "")) for r in results]
    positive = ("推荐", "值得", "适合", "优秀")
    negative = ("不推荐", "不值得", "不适合", "较差")
    has_positive = any(any(word in answer for word in positive) for answer in answers)
    has_negative = any(any(word in answer for word in negative) for answer in answers)
    return has_positive and has_negative


async def _llm_conflict_judgement(state: dict, results: list[dict]) -> bool:
    """Use the small model only after the cheap conflict gate fires."""
    from langchain_core.messages import HumanMessage
    from llms import simple_LLM, llm_ainvoke_with_retry

    summaries = "\n\n".join(
        f"{r.get('expert')}: {r.get('answer', '')[:600]}" for r in results
    )
    prompt = (
        "判断下面几个 Expert 对同一个用户问题给出的结论是否存在实质矛盾。\n"
        "实质矛盾指：核心结论互相冲突（一个推荐 vs 另一个反对；指向不同作品；给出的关键事实相反）。\n"
        "只是措辞、语气、评价角度不同，不算实质矛盾。\n"
        "仅输出 JSON：{\"conflict\": true} 或 {\"conflict\": false}，不要其他文字。\n\n"
        f"用户问题：{state.get('resolved_query') or state.get('original_query', '')}\n"
        f"Expert 结论：\n{summaries}"
    )
    try:
        response = await llm_ainvoke_with_retry(simple_LLM, [HumanMessage(content=prompt)])
        import json
        text = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        return bool(json.loads(text).get("conflict", False))
    except Exception as exc:
        logger.warning("Evaluator conflict judgement failed: %s", exc)
        return False


async def evaluator_node(state: dict) -> dict:
    """Apply deterministic quality rules; avoid a fixed per-request LLM cost."""
    evaluation = _deterministic_evaluation(state)
    results = current_expert_results(state)
    if (
        evaluation.verdict == "pass"
        and _may_conflict(results)
        and not getattr(config, "ABLATION_NO_EVALUATOR_CONFLICT", False)
    ):
        if await _llm_conflict_judgement(state, results):
            evaluation = evaluation.model_copy(update={
                "verdict": "replan",
                "issues": [*evaluation.issues, "expert_conflict"],
                "feedback": "Expert conclusions conflict and require serial verification",
            })

    termination_reason = ""
    merged_results = state.get("merged_results", "")
    if evaluation.verdict == "pass":
        termination_reason = "quality_pass"
    elif state.get("attempt", 0) >= state.get("max_replans", 1):
        termination_reason = "replan_exhausted"
        merged_results += (
            "\n\n[质量提示] 当前证据仍存在不确定性："
            + (evaluation.feedback or "未达到质量阈值")
        )
    event = {
        "event": "quality_evaluated",
        "execution_id": state.get("execution_id", ""),
        "attempt": state.get("attempt", 0),
        "mode": state.get("execution_mode", ""),
        "score": evaluation.score,
        "issues": evaluation.issues,
    }
    if evaluation.verdict == "pass":
        logger.info("quality_pass attempt=%s score=%.3f", state.get("attempt", 0), evaluation.score)
    return {
        "evaluation": evaluation.model_dump(),
        "termination_reason": termination_reason,
        "merged_results": merged_results,
        "quality_trace": [event],
    }
