"""Focused regressions for shared message parsing and tool registration."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.context_builder import context_builder_node, extract_recommendation_count
from agents.history_extractor import _extract_recent_rounds
from agents.message_content import has_image_block, latest_user_message, message_text
from tools.registry import register_default_tools, tool_registry


def test_similar_expert_prepares_deduplicated_candidates_without_llm():
    from agents.similar_expert import similar_expert_node

    semantic = "番剧: 命运石之门\n评分: 8.8\n类型/标签: 科幻、悬疑"
    with (
        patch("agents.similar_expert._find_structured_similar", return_value=[{
            "name_cn": "命运石之门",
            "score": 8.8,
            "tags": ["科幻", "悬疑"],
        }]),
        patch("llms.llm_ainvoke_with_retry", new=AsyncMock()) as llm_call,
    ):
        result = asyncio.run(similar_expert_node({
            "resolved_query": "推荐相似动画",
            "shared_context": [semantic],
            "execution_id": "run",
            "attempt": 0,
        }))

    assert len(result["recommendation_candidates"]) == 1
    candidate = result["recommendation_candidates"][0]
    assert candidate["title"] == "命运石之门"
    assert candidate["sources"] == ["semantic", "metadata"]
    assert candidate["evidence"] == [" ".join(semantic.splitlines())]
    llm_call.assert_not_awaited()


def test_similar_expert_excludes_current_topic():
    from agents.similar_expert import similar_expert_node

    with patch("agents.similar_expert._find_structured_similar", return_value=[
        {"name_cn": "命运石之门", "tags": ["科幻"]},
        {"name_cn": "魔法少女小圆", "tags": ["悬疑"]},
    ]):
        result = asyncio.run(similar_expert_node({
            "resolved_query": "推荐和命运石之门相似的动画",
            "shared_context": [],
            "context": {"topic_entity": {"name": "命运石之门", "type": "anime"}},
            "execution_id": "run",
            "attempt": 0,
        }))

    assert [item["title"] for item in result["recommendation_candidates"]] == ["魔法少女小圆"]


def test_similar_expert_excludes_same_series_after_metadata_enrichment():
    from agents.similar_expert import similar_expert_node

    semantic = [
        "番剧: 命运石之门 0\n类型/标签: 科幻、悬疑",
        "番剧: 命运石之门 负荷领域的既视感\n类型/标签: 科幻、剧场版",
        "番剧: 四叠半神话大系\n类型/标签: 科幻、轮回",
    ]
    structured = [
        {"name_cn": "命运石之门 0", "tags": ["命运石之门", "续作"]},
        {
            "name_cn": "命运石之门 负荷领域的既视感",
            "tags": ["命运石之门", "剧场版"],
        },
        {"name_cn": "四叠半神话大系", "tags": ["科幻", "轮回"]},
    ]
    with (
        patch("agents.similar_expert._find_structured_similar", return_value=structured),
        patch("agents.similar_expert._series_markers", return_value={"命运石之门"}),
    ):
        result = asyncio.run(similar_expert_node({
            "resolved_query": "再推荐两部和它气质相近的动画",
            "shared_context": semantic,
            "context": {
                "topic_entity": {"name": "命运石之门", "type": "anime"},
                "constraints": {"exclude_same_series": True, "excluded_series": ["命运石之门"]},
            },
            "execution_id": "run",
            "attempt": 0,
        }))

    assert [item["title"] for item in result["recommendation_candidates"]] == ["四叠半神话大系"]


def test_series_filter_ignores_ambiguous_short_topic_titles():
    from agents.similar_expert import _series_markers

    # 中文模糊短词（<4字）不能作为子串标记，否则会把所有含"日常"的番误判同系列。
    # 别名库补全后 markers 可能含罗马音/假名（如 nichijou/にちじょう），
    # 这些长度>=4 且不与其它番剧标题子串冲突，是安全标记；真正要挡的是中文短词泄漏。
    markers = _series_markers("日常")
    assert all(len(m) >= 4 for m in markers)
    assert not any("日常" in m for m in markers)


def test_message_helpers_parse_multimodal_content():
    message = HumanMessage(content=[
        {"type": "text", "text": "这是什么番"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}},
    ])

    assert message_text(message.content) == "这是什么番"
    assert has_image_block(message)
    assert latest_user_message([message, AIMessage(content="处理中")]) is message


def test_history_extractor_stores_text_instead_of_content_blocks():
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "识别这张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
        ]),
        AIMessage(content=[{"type": "text", "text": "识别为命运石之门"}]),
    ]

    assert _extract_recent_rounds(messages, 3) == [{
        "user": "识别这张图",
        "assistant": "识别为命运石之门",
    }]


def test_context_builder_ignores_trailing_ai_message():
    result = asyncio.run(context_builder_node({
        "messages": [
            HumanMessage(content="命运石之门讲了什么"),
            AIMessage(content="中间输出"),
        ],
        "context": {},
    }))

    assert result["resolved_query"] == "命运石之门讲了什么"


def test_context_builder_resolves_internal_pronoun_to_topic_entity():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="再推荐两部和它气质相近的动画")],
        "context": {"history": [{"user": "命运石之门讲了什么", "assistant": "..."}]},
        "topic_entity": {"name": "命运石之门", "type": "alias"},
        "recent_entities": [
            {"name": "娜娜", "type": "anime"},
            {"name": "命运石之门", "type": "anime"},
        ],
    }))

    assert "命运石之门气质相近" in result["resolved_query"]
    assert "它" not in result["resolved_query"]
    assert result["topic_entity"]["name"] == "命运石之门"
    assert result["recommendation_count"] == 2


def test_context_builder_anchors_elliptical_recommendation_followup():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="再推荐两部相似的")],
        "context": {"history": [{"user": "聊聊命运石之门", "assistant": "..."}]},
        "topic_entity": {"name": "命运石之门", "type": "alias"},
        "recent_entities": [],
    }))

    assert result["resolved_query"] == "基于《命运石之门》，再推荐两部相似的"


def test_context_builder_does_not_treat_recommendation_count_as_ordinal():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="再推荐三部相似的")],
        "context": {"history": [{"user": "推荐科幻番", "assistant": "..."}]},
        "topic_entity": {"name": "命运石之门", "type": "anime"},
        "recent_entities": [
            {"name": "命运石之门", "type": "anime"},
            {"name": "夏日重现", "type": "anime"},
            {"name": "寒蝉鸣泣之时", "type": "anime"},
        ],
    }))

    assert result["resolved_query"] == "基于《命运石之门》，再推荐三部相似的"
    assert result["recommendation_count"] == 3


def test_context_builder_replaces_complete_ordinal_reference():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="那第一个的评分是多少？")],
        "context": {"history": [{"user": "推荐两部动画", "assistant": "..."}]},
        "recent_entities": [
            {"name": "命运石之门", "type": "anime"},
            {"name": "夏日重现", "type": "anime"},
        ],
    }))

    assert result["resolved_query"] == "那命运石之门的评分是多少？"


def test_extract_recommendation_count_supports_chinese_and_arabic_numbers():
    assert extract_recommendation_count("推荐两部科幻动画") == 2
    assert extract_recommendation_count("给我推荐3部日常番") == 3
    assert extract_recommendation_count("推荐一些动画") == 0
    assert extract_recommendation_count("推荐十二部科幻番") == 12
    assert extract_recommendation_count("再推荐十三部热血番") == 13
    assert extract_recommendation_count("推荐十部日常番") == 10
    assert extract_recommendation_count("推荐一部") == 1
    assert extract_recommendation_count("给我找三部") == 3
    assert extract_recommendation_count("推荐四十二部") == 0  # 超出 20 上限


def test_extract_recommendation_count_rejects_malformed_chinese_numbers():
    assert extract_recommendation_count("推荐一三部") == 0
    assert extract_recommendation_count("推荐十十部") == 0
    assert extract_recommendation_count("推荐两八部") == 0


def test_explicit_anime_title_becomes_persistent_topic():
    result = asyncio.run(context_builder_node({
        "messages": [HumanMessage(content="《命运石之门》的主要剧情是什么？")],
        "context": {},
    }))

    assert result["topic_entity"] == {"name": "命运石之门", "type": "anime"}


def test_default_tool_registration_is_idempotent_and_uses_current_image_tool():
    tool_registry.reset()
    register_default_tools()
    first_count = len(tool_registry)
    register_default_tools()

    image_tool = tool_registry.get("search_anime_by_image")
    assert len(tool_registry) == first_count
    assert image_tool is not None
    assert image_tool.import_path == "tools.image_search.search_anime_by_image"
    assert tool_registry.get("trace_moe_identify") is None


def test_trace_moe_result_is_normalized_for_retrieval():
    from tools import image_search

    trace_result = {
        "anilist": {
            "id": 9253,
            "title": {"english": "Steins;Gate", "romaji": "Steins;Gate"},
        },
        "episode": 1,
        "from": 83.9,
        "similarity": 0.97,
        "video": "https://example.invalid/preview.mp4",
    }
    image_search._SEARCH_CACHE.clear()
    with patch.object(image_search, "_call_trace_moe", AsyncMock(return_value=trace_result)):
        result = asyncio.run(image_search.search_anime_by_image("fake-image"))

    assert result == {
        "matched": True,
        "anilist_id": 9253,
        "title_raw": "Steins;Gate",
        "title_cn": "命运石之门",
        "episode": 1,
        "timestamp": "01:23",
        "similarity": 0.97,
        "preview_url": "https://example.invalid/preview.mp4",
        "source": "trace_moe",
    }


def test_trace_moe_uploads_decoded_image_as_multipart():
    from tools import image_search

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "result": [{"similarity": 0.91, "episode": 2}]
    }
    client = AsyncMock()
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client

    image = base64.b64encode(b"jpeg-bytes").decode("ascii")
    with patch.object(image_search.httpx, "AsyncClient", return_value=context):
        result = asyncio.run(image_search._call_trace_moe(image))

    assert result == {"similarity": 0.91, "episode": 2}
    kwargs = client.post.await_args.kwargs
    assert "json" not in kwargs
    assert kwargs["params"] == {"anilistInfo": "true"}
    assert kwargs["files"]["image"] == (
        "frame.jpg",
        b"jpeg-bytes",
        "image/jpeg",
    )


def test_trace_moe_rejects_invalid_base64_without_http_request():
    from tools import image_search

    client = AsyncMock()
    with patch.object(image_search.httpx, "AsyncClient", return_value=client):
        result = asyncio.run(image_search._call_trace_moe("not-base64!"))

    assert result is None
    client.post.assert_not_awaited()


def test_trace_moe_integer_anilist_id_does_not_crash_normalization():
    from tools import image_search

    trace_result = {
        "anilist": 21034,
        "episode": 1,
        "from": 272.1,
        "similarity": 0.99,
    }
    image_search._SEARCH_CACHE.clear()
    with patch.object(image_search, "_call_trace_moe", AsyncMock(return_value=trace_result)):
        result = asyncio.run(image_search.search_anime_by_image("integer-anilist"))

    assert result["matched"] is True
    assert result["anilist_id"] == 21034
    assert result["title_raw"] == ""


def test_trace_moe_prefers_anilist_chinese_title_without_llm_translation():
    from tools import image_search

    trace_result = {
        "anilist": {
            "id": 21034,
            "title": {
                "chinese": "请问您今天要来点兔子吗？？",
                "english": "Is the Order a Rabbit?? Season 2",
            },
        },
        "episode": 1,
        "from": 272.1,
        "similarity": 0.99,
    }
    image_search._SEARCH_CACHE.clear()
    with (
        patch.object(image_search, "_call_trace_moe", AsyncMock(return_value=trace_result)),
        patch.object(image_search, "normalize_to_chinese_title") as translate,
    ):
        result = asyncio.run(image_search.search_anime_by_image("chinese-title"))

    assert result["title_raw"] == "Is the Order a Rabbit?? Season 2"
    assert result["title_cn"] == "请问您今天要来点兔子吗？？"
    translate.assert_not_called()


def test_image_node_injects_trace_moe_title_and_discards_raw_image():
    from agents.image_recognition import image_recognition_node

    recognition = {
        "matched": True,
        "title_cn": "命运石之门",
        "episode": 1,
        "timestamp": "01:23",
        "similarity": 0.97,
        "source": "trace_moe",
    }
    with patch(
        "tools.image_search.search_anime_by_image",
        AsyncMock(return_value=recognition),
    ):
        result = asyncio.run(image_recognition_node({
            "image_data": "fake-image",
            "messages": [HumanMessage(content="这是什么番")],
        }))

    assert result["search_keywords"] == ["命运石之门"]
    assert "《命运石之门》第1话 01:23" in result["resolved_query"]
    assert result["image_data"] is None
