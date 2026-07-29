"""跨轮约束继承回归测试：排除同系列、追问无需重复、显式撤销。"""

import asyncio

from agents.context_builder import context_builder_node, _extract_constraints
from langchain_core.messages import HumanMessage

from agents.history_extractor import history_extractor_node
from langchain_core.messages import AIMessage


def test_extract_constraints_detects_same_series_exclusion():
    result = _extract_constraints("推荐类似命运石之门的动画，排除同系列作品", None)
    assert result["exclude_same_series"] is True


def test_extract_constraints_inherits_on_followup():
    previous = {"exclude_same_series": True, "excluded_series": ["命运石之门"]}
    result = _extract_constraints("再来一部更接近的", previous)
    assert result["exclude_same_series"] is True
    assert "命运石之门" in result["excluded_series"]


def test_extract_constraints_allows_revocation():
    previous = {"exclude_same_series": True, "excluded_series": ["命运石之门"]}
    result = _extract_constraints("同系列也可以", previous)
    assert result.get("exclude_same_series") is not True
    assert "excluded_series" not in result


def test_context_builder_inherits_exclusion_without_repetition():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="第二部太偏了，再来一部更接近的")],
        "context": {
            "history": [{"user": "推荐几部类似命运石之门的悬疑科幻动画，排除同系列作品", "assistant": "..."}],
            "constraints": {"exclude_same_series": True, "excluded_series": ["命运石之门"]},
        },
        "topic_entity": {"name": "命运石之门", "type": "anime"},
        "previous_intent": "recommendation",
    }))

    assert result["context"]["constraints"]["exclude_same_series"] is True
    assert "命运石之门" in result["context"]["constraints"]["excluded_series"]
    assert result["topic_entity"]["name"] == "命运石之门"


def test_context_builder_keeps_topic_entity_after_recommendation():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="推荐几部类似《命运石之门》的悬疑科幻动画，排除同系列作品")],
        "context": {},
    }))

    assert result["topic_entity"]["name"] == "命运石之门"
    assert result["context"]["constraints"]["exclude_same_series"] is True
    assert "命运石之门" in result["context"]["constraints"]["excluded_series"]
    # 应保存主题标签，便于追问检索
    assert "topic_tags" in result["context"]["constraints"]
    assert "科幻" in result["context"]["constraints"]["topic_tags"]


def test_constraint_prompt_section_lists_excluded_series():
    from agents.answer import _build_constraint_section

    section = _build_constraint_section({
        "exclude_same_series": True,
        "excluded_series": ["命运石之门", "命运石之门 0"],
    })
    assert "排除同系列作品" in section
    assert "命运石之门" in section
    assert "命运石之门 0" in section
    assert "严格" not in section


def test_history_extractor_preserves_existing_constraints():
    messages = [
        HumanMessage(content="推荐类似《命运石之门》的动画，排除同系列"),
        AIMessage(content="推荐《四叠半神话大系》"),
        HumanMessage(content="再来一部"),
    ]
    result = asyncio.run(history_extractor_node({
        "messages": messages,
        "context": {
            "constraints": {"exclude_same_series": True, "excluded_series": ["命运石之门"]},
            "topic_entity": {"name": "命运石之门", "type": "anime"},
        },
    }))

    assert result["context"]["constraints"]["exclude_same_series"] is True
    assert "命运石之门" in result["context"]["constraints"]["excluded_series"]
    assert len(result["context"]["history"]) == 1


def test_context_builder_detects_ordinal_reference_as_followup():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="第二部太偏了，再来一部更接近的")],
        "context": {"history": [{"user": "推荐科幻番", "assistant": "..."}]},
        "topic_entity": {"name": "命运石之门", "type": "anime"},
        "recent_entities": [
            {"name": "命运石之门", "type": "anime"},
            {"name": "全部成为F THE PERFECT INSIDER", "type": "anime"},
        ],
    }))

    assert result["context"]["is_followup"] is True
    assert "第二部" not in result["resolved_query"]
    assert "全部成为F" in result["resolved_query"]
