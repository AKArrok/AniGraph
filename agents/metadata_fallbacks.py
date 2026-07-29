"""Shared structured fallbacks used by both simple_fact_answer (fast path)
and knowledge_retrieval (full pipeline).

Extracted so the full pipeline retrieval layer no longer lags behind the
fast path. Adding a new fallback rule here now benefits both branches.

Rules currently exported:

  filter_metadata_by_query(metadata, query)
      Trim an existing metadata list down to entries that match a
      year/studio/director mentioned in the query. Returns [] if the
      query has no such hints (caller falls back to the full list).

  fetch_by_studio_year(query)
      When the query mentions BOTH a known studio hint AND a 4-digit
      year, hit MetadataIndex.search directly. This bypasses embedding
      retrieval for the very common "studio X in year Y" pattern.

  fetch_by_title(query, search_keywords)
      When the query points at a specific title (either via a bracketed
      substring like 《...》 or via any of the passed search_keywords),
      resolve it through the alias index and return the full metadata
      record. This is the missing piece that made full-pipeline
      knowledge_retrieval return empty on longtail_fact / bangumi_tags /
      bangumi_score_precise questions.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")

_STUDIO_HINTS = (
    "ufotable", "京都动画", "京都アニメーション", "kyoani", "kyoto animation",
    "mappa", "madhouse", "trigger", "wit studio", "bones",
    "j.c.staff", "j.c. staff", "sunrise", "shaft", "gainax",
    "production i.g", "cloverworks", "a-1 pictures",
    "スタジオジブリ", "吉卜力",
    "东映动画", "东映アニメーション", "东宝",
)

# 提取被《》/「」/『』/"" 包住的完整作品名
_TITLE_BRACKETS_RE = re.compile(
    r"[《「『\"“](?P<t>[^《》「」『』\"”]{2,40})[》」』\"”]"
)

_LIST_QUERY_KEYWORDS = (
    "哪些标签", "什么标签", "主要标签", "全部标签", "所有标签",
    "打上", "打了", "打的", "tag", "标签列表",
    "哪些声优", "所有声优", "全部声优", "声优阵容", "配音阵容",
    "哪些人参与", "主创阵容", "staff", "制作人员",
)


def is_list_query(query: str) -> bool:
    """列表类查询：需要完整列出某个字段的全部值。"""
    if not query:
        return False
    q = query.lower()
    return any(kw in q for kw in _LIST_QUERY_KEYWORDS)


def format_metadata(entries: list[dict], verbose: bool = False) -> str:
    """紧凑格式化元数据供 LLM 消费。"""
    tag_limit = 60 if verbose else 8
    staff_limit = 30 if verbose else 3
    lines: list[str] = []
    for m in entries:
        name_cn = m.get("name_cn", "")
        name = m.get("name", "") or m.get("title", "")
        display_name = name_cn or name
        score = m.get("score", "") or m.get("rating", "")
        rank = m.get("rank", "")
        date = m.get("date", "")
        studio = m.get("studio", "")
        director = m.get("director", "")
        tags = m.get("tags", []) or []
        staff = m.get("staff", []) or m.get("seiyuu", []) or []
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(staff, str):
            staff = [staff]
        parts: list[str] = [str(display_name)]
        if name_cn and name and name_cn != name:
            parts.append("原名:" + str(name))
        if score:
            parts.append("评分" + str(score))
        if rank:
            parts.append("排名" + str(rank))
        if date:
            parts.append("日期:" + str(date))
        if studio:
            parts.append("制作:" + str(studio))
        if director:
            parts.append("导演:" + str(director))
        if tags:
            parts.append("标签:" + ",".join(str(t) for t in tags[:tag_limit]))
        if staff:
            parts.append("人员:" + ",".join(str(s) for s in staff[:staff_limit]))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def filter_metadata_by_query(metadata: list[dict], query: str) -> list[dict]:
    """Filter existing metadata by year/studio/director found in the query.

    Only strict substring match. Returns [] when no filter applies, letting
    the caller keep the original list.
    """
    if not metadata or not query:
        return []
    q = query.lower()
    year_match = _YEAR_RE.search(query)
    year = year_match.group(0) if year_match else None

    studios = {(m.get("studio") or "").strip() for m in metadata if m.get("studio")}
    directors = {(m.get("director") or "").strip() for m in metadata if m.get("director")}
    hit_studio = next((s for s in studios if s and s.lower() in q), None)
    hit_director = next((d for d in directors if d and d.lower() in q), None)

    if not (year or hit_studio or hit_director):
        return []

    def match(item: dict) -> bool:
        if year:
            date = str(item.get("date") or "")
            if not date.startswith(year):
                return False
        if hit_studio and (item.get("studio") or "").strip().lower() != hit_studio.lower():
            return False
        if hit_director and (item.get("director") or "").strip().lower() != hit_director.lower():
            return False
        return True

    return [m for m in metadata if match(m)]


def fetch_by_studio_year(query: str) -> list[dict]:
    """Hit MetadataIndex.search directly when both studio and year appear."""
    year_match = _YEAR_RE.search(query)
    if not year_match:
        return []
    q = query.lower()
    studio_hit = next((s for s in _STUDIO_HINTS if s in q), None)
    if not studio_hit:
        return []
    try:
        from agents.metadata_index import index
        year = year_match.group(0)
        return index.search(
            studio=studio_hit,
            date_from=year,
            date_to=str(int(year) + 1),
            limit=30,
        )
    except Exception as e:
        logger.warning("studio x year fallback failed: %s", e)
        return []


def _title_candidates(query: str, search_keywords: list[str] | None) -> list[str]:
    """Collect probable anime titles hidden in the query.

    Priority: bracketed substrings > planner-produced search_keywords > full query.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for m in _TITLE_BRACKETS_RE.finditer(query or ""):
        t = m.group("t").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            candidates.append(t)
    for kw in search_keywords or []:
        k = str(kw).strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            candidates.append(k)
    q = (query or "").strip()
    if q and q.lower() not in seen:
        candidates.append(q)
    return candidates


def fetch_by_title(query: str, search_keywords: list[str] | None) -> list[dict]:
    """Resolve any full-title mention in the query via the alias index.

    Returns [] when nothing resolves. Only exact alias lookups are used;
    fuzzy_lookup is intentionally called for the top candidates to catch
    trailing particles like "评分是多少?".
    """
    try:
        from agents.metadata_index import index
    except Exception as e:
        logger.warning("metadata_index import failed: %s", e)
        return []

    results: list[dict] = []
    seen_ids: set[str] = set()
    for cand in _title_candidates(query, search_keywords):
        # 1) exact alias
        hit = index.get_by_alias(cand)
        if not hit:
            # 2) fuzzy lookup handles trailing particles / question tails
            hit = index.fuzzy_lookup(cand)
        if hit:
            rid = str(hit.get("id", ""))
            if rid and rid not in seen_ids:
                results.append(hit)
                seen_ids.add(rid)
                # 一次命中通常就够了；继续多轮匹配意义不大且可能引入歧义
                break
    return results
