from data.chunking import make_anime_chunks
from data.build_kb import _stale_child_ids
from tools.knowledge_retrieval import _sparse_terms


def test_make_anime_chunks_separates_semantic_fields():
    chunks = make_anime_chunks({
        "id": 42,
        "title": "测试番剧",
        "score": 8.5,
        "score_count": 100,
        "date": "2026-01-01",
        "tags": ["科幻", "悬疑"],
        "studios": ["测试社"],
        "directors": ["甲"],
        "writers": [],
        "seiyuu": ["乙"],
        "comments": ["第一条评论", "第二条评论"],
    })

    assert [chunk.chunk_type for chunk in chunks] == [
        "profile", "staff", "cast", "reviews", "reviews",
    ]
    assert [chunk.id for chunk in chunks] == [
        "anime_42",
        "anime_42_staff_0",
        "anime_42_cast_0",
        "anime_42_reviews_0",
        "anime_42_reviews_1",
    ]
    assert all("番剧: 测试番剧" in chunk.text for chunk in chunks)
    assert "观众评论" not in chunks[0].text
    assert "评分" not in chunks[1].text
    assert "声优" not in chunks[1].text
    assert "声优: 乙" in chunks[2].text


def test_review_chunks_preserve_comment_boundaries():
    comments = ["甲" * 30, "乙" * 30, "丙" * 30]
    chunks = make_anime_chunks({
        "id": 1,
        "title": "边界测试",
        "comments": comments,
    }, review_chunk_chars=75)
    reviews = [chunk for chunk in chunks if chunk.chunk_type == "reviews"]

    assert len(reviews) == 3
    for comment, chunk in zip(comments, reviews):
        assert comment in chunk.text


def test_long_review_is_split_without_losing_content():
    comment = "第一段。" * 30 + "第二段没有句号" * 20
    chunks = make_anime_chunks({
        "id": 2,
        "title": "长评测试",
        "comments": [comment],
    }, review_chunk_chars=80)
    reviews = [chunk for chunk in chunks if chunk.chunk_type == "reviews"]

    assert len(reviews) > 1
    rebuilt = "".join(chunk.text.split("- ", 1)[1] for chunk in reviews)
    assert rebuilt == comment
    assert all(len(chunk.text) <= 80 for chunk in reviews)


def test_sparse_terms_remove_low_information_words():
    assert _sparse_terms("科幻番剧推荐") == ["科幻"]
    assert _sparse_terms("京都动画 导演") == ["京都", "导演"]
    assert _sparse_terms("Production I.G") == ["production", "i.g"]


def test_minimal_record_still_has_a_profile_chunk():
    chunks = make_anime_chunks({"id": 3, "title": "极简", "comments": []})

    assert len(chunks) == 1
    assert chunks[0].id == "anime_3"


def test_stale_child_ids_keep_current_chunks():
    chunks = make_anime_chunks({
        "id": 7,
        "title": "更新测试",
        "studios": ["测试社"],
        "comments": ["仍然存在的评论"],
    })

    stale_ids = _stale_child_ids(7, chunks)

    assert "anime_7_staff_0" not in stale_ids
    assert "anime_7_cast_0" in stale_ids
    assert "anime_7_reviews_0" not in stale_ids
    assert "anime_7_reviews_1" in stale_ids


def test_stale_child_ids_keep_current_cast_chunk():
    chunks = make_anime_chunks({
        "id": 8,
        "title": "声优更新测试",
        "seiyuu": ["甲"],
    })

    stale_ids = _stale_child_ids(8, chunks)

    assert "anime_8_cast_0" not in stale_ids
    assert "anime_8_staff_0" in stale_ids
