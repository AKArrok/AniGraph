"""Focused regressions for shared message parsing and tool registration."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.context_builder import context_builder_node
from agents.history_extractor import _extract_recent_rounds
from agents.message_content import has_image_block, latest_user_message, message_text
from tools.registry import register_default_tools, tool_registry


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
