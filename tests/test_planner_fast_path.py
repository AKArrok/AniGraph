from agents.planner import _fast_recommendation_plan


def test_recommendation_followup_uses_zero_llm_fast_path():
    plan = _fast_recommendation_plan("第二部太偏了，再来一部风格更接近命运石之门的")

    assert plan is not None
    assert plan["query_type"] == "recommendation"
    assert plan["rewrite_strategy"] == "direct"
    assert plan["experts"] == ["similar_expert"]
