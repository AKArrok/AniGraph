"""Focused tests for strict Expert routing and the Evaluator-Replan loop."""
import unittest
from unittest.mock import AsyncMock, patch

from langgraph.types import Send

from agents.evaluator import evaluator_node
from agents.graph import (
    _route_after_evaluator,
    _route_after_retrieval,
    _route_after_similar_expert,
    _route_after_serial_expert,
    _serial_expert_node,
    build_graph,
)
from agents.merge import merge_expert_results
from agents.replanner import build_replan_patch, replanner_node


def _state(**overrides):
    state = {
        "plan": {
            "query_type": "recommendation",
            "query_category": "mixed",
            "rewrite_strategy": "rewrite",
            "experts": ["metadata_reasoner", "similar_expert"],
            "parallel": True,
            "need_web": False,
        },
        "execution_id": "run-current",
        "attempt": 0,
        "max_replans": 1,
        "current_expert_index": 0,
        "original_query": "推荐两部科幻番",
        "resolved_query": "推荐两部科幻番",
        "metadata": [{"name": "A"}],
        "shared_context": ["观众评论: 节奏很好"],
        "expert_results": [],
        "merged_results": "",
        "evaluation": {},
    }
    state.update(overrides)
    return state


def _result(expert, *, attempt=0, execution_id="run-current", confidence=0.8):
    return {
        "expert": expert,
        "attempt": attempt,
        "execution_id": execution_id,
        "answer": f"{expert} conclusion",
        "confidence": confidence,
        "evidence": [f"{expert} evidence"],
    }


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_parallel_two_experts_creates_two_sends(self):
        route = _route_after_retrieval(_state())
        self.assertEqual(len(route), 2)
        self.assertTrue(all(isinstance(item, Send) for item in route))
        self.assertEqual({item.node for item in route}, {"metadata_reasoner", "similar_expert"})
        self.assertTrue(all(item.arg["execution_id"] == "run-current" for item in route))

    def test_serial_two_experts_starts_controller(self):
        state = _state(plan={**_state()["plan"], "parallel": False})
        self.assertEqual(_route_after_retrieval(state), "serial_expert")

    def test_single_expert_ignores_parallel_flag(self):
        for parallel in (False, True):
            plan = {**_state()["plan"], "experts": ["similar_expert"], "parallel": parallel}
            self.assertEqual(_route_after_retrieval(_state(plan=plan)), "similar_expert")

    def test_single_similar_recommendation_bypasses_quality_loop(self):
        plan = {**_state()["plan"], "experts": ["similar_expert"]}
        self.assertEqual(_route_after_similar_expert(_state(plan=plan)), "answer_planner")

    def test_comparison_with_similar_expert_keeps_quality_loop(self):
        plan = {
            **_state()["plan"],
            "query_type": "comparison",
            "experts": ["similar_expert"],
        }
        self.assertEqual(_route_after_similar_expert(_state(plan=plan)), "merge")

    def test_simple_fact_fast_path_is_unchanged(self):
        plan = {**_state()["plan"], "query_type": "simple_fact"}
        self.assertEqual(_route_after_retrieval(_state(plan=plan)), "simple_fact_answer")

    async def test_serial_second_expert_receives_first_result(self):
        state = _state(plan={**_state()["plan"], "parallel": False})
        first = _result("metadata_reasoner")
        second = _result("similar_expert")
        with patch("agents.graph.metadata_reasoner_node", new=AsyncMock(return_value={"expert_results": [first]})):
            update = await _serial_expert_node(state)
        self.assertEqual(update["current_expert_index"], 1)

        second_state = {**state, **update, "expert_results": [first]}
        self.assertEqual(_route_after_serial_expert(second_state), "serial_expert")
        similar_mock = AsyncMock(return_value={"expert_results": [second]})
        with patch("agents.graph.similar_expert_node", new=similar_mock):
            update = await _serial_expert_node(second_state)
        self.assertEqual(update["current_expert_index"], 2)
        self.assertEqual(similar_mock.await_args.args[0]["expert_results"], [first])
        final_state = {**second_state, **update, "expert_results": [first, second]}
        self.assertEqual(_route_after_serial_expert(final_state), "merge")


class QualityLoopTests(unittest.IsolatedAsyncioTestCase):
    def test_merge_isolates_execution_and_attempt(self):
        current = _result("metadata_reasoner", attempt=1)
        old_attempt = _result("similar_expert", attempt=0)
        old_execution = _result("similar_expert", attempt=1, execution_id="old-run")
        merged = merge_expert_results(_state(
            attempt=1,
            expert_results=[old_attempt, old_execution, current],
        ))["merged_results"]
        self.assertIn("metadata_reasoner conclusion", merged)
        self.assertNotIn("similar_expert conclusion", merged)

    async def test_missing_planned_expert_triggers_replan(self):
        state = _state(expert_results=[_result("metadata_reasoner")])
        update = await evaluator_node(state)
        self.assertEqual(update["evaluation"]["verdict"], "replan")
        self.assertIn("missing_dimension", update["evaluation"]["issues"])
        self.assertEqual(_route_after_evaluator({**state, **update}), "replanner")

    async def test_replan_increments_attempt_and_preserves_intent(self):
        state = _state(evaluation={
            "verdict": "replan",
            "score": 0.2,
            "issues": ["no_evidence"],
            "missing_dimensions": [],
            "feedback": "no evidence",
        })
        update = await replanner_node(state)
        self.assertEqual(update["attempt"], 1)
        self.assertEqual(update["plan"]["query_type"], state["plan"]["query_type"])
        self.assertEqual(update["metadata"], [])
        self.assertEqual(update["shared_context"], [])
        self.assertTrue(update["replan_feedback"]["additional_queries"])

    def test_conflict_patch_changes_execution_to_serial(self):
        state = _state(evaluation={
            "issues": ["expert_conflict"],
            "missing_dimensions": [],
        })
        patch = build_replan_patch(state)
        self.assertFalse(patch.parallel)
        self.assertEqual(patch.experts, ["metadata_reasoner", "similar_expert"])

    async def test_second_failure_exhausts_one_replan_budget(self):
        state = _state(attempt=1, expert_results=[])
        update = await evaluator_node(state)
        routed_state = {**state, **update}
        self.assertEqual(update["termination_reason"], "replan_exhausted")
        self.assertNotEqual(_route_after_evaluator(routed_state), "replanner")
        self.assertIn("不确定性", update["merged_results"])

    async def test_fallback_verdict_routes_to_web(self):
        state = _state(evaluation={"verdict": "fallback", "issues": [], "score": 0.0})
        self.assertEqual(_route_after_evaluator(state), "web_fallback")

    async def test_quality_pass_route_no_web(self):
        state = _state(evaluation={"verdict": "pass", "issues": [], "score": 0.9})
        self.assertEqual(_route_after_evaluator(state), "answer_planner")

    def test_graph_compiles_with_quality_loop(self):
        graph = build_graph().compile()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
