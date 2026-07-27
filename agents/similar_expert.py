"""Similar Expert — 基于 Embedding 召回 + Metadata Index 整理相似候选

工作流:
  1. 从用户查询中提取目标番剧（如有）
  2. Metadata Index: 同标签/同公司/同导演等结构相似
  3. Embedding 检索: 语义相似 TopK
  4. 确定性合并、去重和排序

输出:
  recommendation_candidates + 兼容质量闭环的 ExpertResult
"""
import re
import time
import logging

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(r"^(?:番剧|【番剧】)[:：]?\s*([^\n（(]+)", re.MULTILINE)


def _normalize_title(title: str) -> str:
    return re.sub(r"[\s《》【】\[\]（）()·:：;；!！?？._-]", "", title).casefold()


def _series_markers(topic_title: str) -> set[str]:
    """Build conservative franchise markers from the current topic metadata."""
    if not topic_title:
        return set()

    names = {topic_title}
    try:
        from agents.metadata_index import index

        item = index.get_by_alias(topic_title)
        if item:
            names.update({
                str(item.get("name_cn", "")),
                str(item.get("name", "")),
                *(str(alias) for alias in item.get("alias", [])),
            })
    except Exception:
        pass

    # Short names such as "日常" are too ambiguous to use as substring markers.
    return {
        normalized
        for name in names
        if (normalized := _normalize_title(name)) and len(normalized) >= 4
    }


def _belongs_to_series(candidate: dict, markers: set[str]) -> bool:
    if not markers:
        return False
    title = _normalize_title(str(candidate.get("title", "")))
    tags = {
        _normalize_title(str(tag))
        for tag in candidate.get("tags", [])
        if tag
    }
    return any(marker in title or marker in tags for marker in markers)


def _find_structured_similar(query: str, state: dict) -> list[dict]:
    """通过 Metadata Index 查找结构相似作品: 同标签/同公司/同导演"""
    try:
        from agents.metadata_index import index
        metadata = state.get("metadata", [])

        if not metadata:
            return []

        similar: list[str] = []
        seen: set[str] = set()

        def remember(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                similar.append(name)

        for item in metadata[:3]:
            # 同标签
            tags = item.get("tags", [])
            if tags:
                for tag in tags[:2]:
                    results = index.search(tag=tag, limit=5)
                    for r in results:
                        remember(r.get("name_cn", ""))

            # 同制作公司
            studio = item.get("studio", "")
            if studio:
                results = index.search(studio=studio, limit=3)
                for r in results:
                    remember(r.get("name_cn", ""))

        # 返回 Metadata Index 中的完整信息
        candidates = []
        for name in similar:
            item = index.get_by_alias(name)
            if item:
                candidates.append(item)

        return candidates[:10]
    except Exception:
        return []


def _candidate_from_metadata(item: dict) -> dict | None:
    title = str(item.get("name_cn") or item.get("name") or item.get("title") or "").strip()
    if not title:
        return None
    tags = item.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in re.split(r"[,，、]", tags) if tag.strip()]
    return {
        "title": title,
        "score": item.get("score", ""),
        "tags": tags[:8],
        "studio": item.get("studio") or item.get("studios") or "",
        "date": item.get("date", ""),
        "evidence": [],
        "sources": ["metadata"],
    }


def _candidate_from_semantic(text: str) -> dict | None:
    match = _TITLE_RE.search(text or "")
    if not match:
        return None
    title = match.group(1).strip()
    evidence = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return {
        "title": title,
        "score": "",
        "tags": [],
        "studio": "",
        "date": "",
        "evidence": [evidence[:700]],
        "sources": ["semantic"],
    }


def _merge_candidates(
    structured: list[dict],
    semantic: list[str],
    limit: int = 8,
    excluded_titles: set[str] | None = None,
    excluded_series_markers: set[str] | None = None,
) -> list[dict]:
    """Merge evidence by exact normalized title while preserving retrieval order."""
    excluded = {
        _normalize_title(title)
        for title in (excluded_titles or set())
        if title
    }
    merged: dict[str, dict] = {}
    order: list[str] = []
    # Semantic retrieval carries the relevance ranking. Metadata enriches those
    # candidates and contributes fallback candidates without overriding that order.
    raw_candidates = [
        *(_candidate_from_semantic(text) for text in semantic),
        *(_candidate_from_metadata(item) for item in structured),
    ]
    for candidate in raw_candidates:
        if not candidate:
            continue
        key = _normalize_title(candidate["title"])
        if key in excluded:
            continue
        if key not in merged:
            merged[key] = candidate
            order.append(key)
            continue
        current = merged[key]
        for field in ("score", "studio", "date"):
            if not current.get(field) and candidate.get(field):
                current[field] = candidate[field]
        current["tags"] = list(dict.fromkeys([*current["tags"], *candidate["tags"]]))[:8]
        current["evidence"] = list(dict.fromkeys([*current["evidence"], *candidate["evidence"]]))[:2]
        current["sources"] = list(dict.fromkeys([*current["sources"], *candidate["sources"]]))
    candidates = [merged[key] for key in order]
    markers = excluded_series_markers or set()
    if markers:
        candidates = [
            candidate
            for candidate in candidates
            if not _belongs_to_series(candidate, markers)
        ]
    return candidates[:limit]


def _format_candidate_summary(candidates: list[dict]) -> str:
    lines = []
    for candidate in candidates:
        details = []
        if candidate.get("score") not in ("", None):
            details.append(f"评分 {candidate['score']}")
        if candidate.get("tags"):
            details.append("标签 " + "、".join(candidate["tags"][:5]))
        if candidate.get("studio"):
            details.append(f"制作 {candidate['studio']}")
        evidence = candidate.get("evidence", [])
        if evidence:
            details.append("证据 " + evidence[0][:240])
        lines.append(f"**{candidate['title']}**：" + "；".join(details))
    return "\n".join(lines)


async def similar_expert_node(state: dict) -> dict:
    """Prepare recommendation candidates without an LLM call."""
    t0 = time.time()
    query = state.get("resolved_query") or state.get("original_query", "")

    structured = _find_structured_similar(query, state)
    shared_context = state.get("shared_context", [])
    context = state.get("context", {})
    topic = context.get("topic_entity", {}) if isinstance(context, dict) else {}
    topic_title = str(topic.get("name", ""))
    constraints = context.get("constraints", {}) if isinstance(context, dict) else {}
    exclude_same_series = bool(constraints.get("exclude_same_series", False))
    excluded_titles = {
        *state.get("search_keywords", []),
        topic_title,
    }
    candidates = _merge_candidates(
        structured,
        shared_context,
        limit=8,
        excluded_titles=excluded_titles,
        excluded_series_markers=_series_markers(topic_title) if exclude_same_series else set(),
    )

    logger.debug(
        f"Similar Expert 收到 state: "
        f"metadata={len(state.get('metadata', []))}条, "
        f"context={len(shared_context)}条, "
        f"structured_candidates={len(structured)}个"
    )
    summary = _format_candidate_summary(candidates)
    result = {
        "expert": "similar_expert",
        "attempt": state.get("attempt", 0),
        "execution_id": state.get("execution_id", ""),
        "answer": summary or "当前知识库中没有足够的相似作品数据。",
        "confidence": 0.85 if candidates else 0.2,
        "evidence": [
            evidence
            for candidate in candidates
            for evidence in candidate.get("evidence", [])[:1]
        ][:8],
    }

    logger.info(f"  similar_expert 整理 {len(candidates)} 个候选，耗时 {time.time()-t0:.1f}s")
    return {
        "recommendation_candidates": candidates,
        "expert_results": [result],
    }
