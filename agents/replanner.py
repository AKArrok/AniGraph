"""Constrained execution replanner for one quality-recovery attempt."""
import logging

from agents.state import ReplanPatch

logger = logging.getLogger(__name__)


def build_replan_patch(state: dict) -> ReplanPatch:
    """Build a deterministic patch from Evaluator issues."""
    plan = state.get("plan", {})
    evaluation = state.get("evaluation", {})
    issues = set(evaluation.get("issues", []))
    experts = list(dict.fromkeys(plan.get("experts", [])))
    category = plan.get("query_category", "mixed")
    strategy = plan.get("rewrite_strategy", "rewrite")
    parallel = bool(plan.get("parallel", False))
    need_web = bool(plan.get("need_web", False))
    additional_queries: list[str] = []

    if "no_evidence" in issues or "low_confidence" in issues:
        strategy = "hyde" if strategy in ("direct", "rewrite") else "rewrite"
        additional_queries.append(f"{state.get('resolved_query') or state.get('original_query', '')} 评价 依据")

    if "missing_dimension" in issues:
        for expert in evaluation.get("missing_dimensions", []):
            if expert in ("metadata_reasoner", "similar_expert") and expert not in experts:
                experts.append(expert)
        if len(experts) < 2:
            other = "similar_expert" if experts == ["metadata_reasoner"] else "metadata_reasoner"
            experts.append(other)
        category = "mixed"

    if "expert_conflict" in issues:
        parallel = False
    elif len(experts) > 1:
        parallel = True

    if "query_mismatch" in issues:
        strategy = "rewrite"
        additional_queries.append(state.get("resolved_query") or state.get("original_query", ""))

    return ReplanPatch(
        rewrite_strategy=strategy,
        query_category=category,
        experts=experts,
        parallel=parallel,
        need_web=need_web,
        additional_queries=list(dict.fromkeys(q for q in additional_queries if q)),
        reasoning=f"Evaluator issues: {', '.join(sorted(issues)) or 'none'}",
    )


async def replanner_node(state: dict) -> dict:
    patch = build_replan_patch(state)
    old_plan = dict(state.get("plan", {}))
    new_plan = dict(old_plan)
    new_plan.update(patch.model_dump(exclude={"additional_queries"}))
    attempt = state.get("attempt", 0) + 1
    changed = {
        key: {"before": old_plan.get(key), "after": new_plan.get(key)}
        for key in new_plan
        if old_plan.get(key) != new_plan.get(key)
    }
    feedback = {
        "evaluation": state.get("evaluation", {}),
        "additional_queries": patch.additional_queries,
        "plan_diff": changed,
        "reasoning": patch.reasoning,
    }
    logger.info("replan_started attempt=%s diff=%s", attempt, changed)
    return {
        "plan": new_plan,
        "attempt": attempt,
        "current_expert_index": 0,
        "replan_feedback": feedback,
        "evaluation": {},
        "metadata": [],
        "shared_context": [],
        "merged_results": "",
        "execution_mode": "",
        "termination_reason": "",
        "quality_trace": [{
            "event": "replan_started",
            "execution_id": state.get("execution_id", ""),
            "attempt": attempt,
            "plan_diff": changed,
        }],
    }
