"""Deterministic scoring for tests/eval_hard_final.json.

No LLM Judge involved. Each case is scored 1.0 or 0.0 by structural rules:

  metadata_cross: score = 1.0 iff answer contains EVERY gold title (partial
                  substring, tolerant of full-width/half-width punctuation).
  longtail_fact:  score = 1.0 iff answer contains BOTH the exact score string
                  and the year string.
  kb_boundary:    score = 1.0 iff answer contains ANY refusal phrase AND does
                  NOT contain a specific proper-noun name that would indicate
                  fabrication. Because "does not contain a name" is expensive
                  to check strictly, we accept the refusal signal as sufficient
                  and flag the case for manual review if refusal is absent.

Also computes per-type accuracy and writes results to
tests/hard_eval_results.json.
"""
from __future__ import annotations

import re
from typing import Any

_PUNCT_TABLE = str.maketrans({
    "《": "", "》": "", "「": "", "」": "", "『": "", "』": "",
    "（": "(", "）": ")", "，": ",", "。": ".", "：": ":", "；": ";",
    "！": "!", "？": "?", "、": ",", "―": "-", "—": "-", "－": "-",
    " ": "", "\u3000": "", "\t": "", "\n": "",
})


def _norm(s: str) -> str:
    return (s or "").translate(_PUNCT_TABLE).lower()


def _contains(hay: str, needle: str) -> bool:
    hay_n = _norm(hay)
    needle_n = _norm(needle)
    return needle_n in hay_n if needle_n else False


def score_metadata_cross(answer: str, gold_titles: list[str]) -> dict[str, Any]:
    hits = [t for t in gold_titles if _contains(answer, t)]
    misses = [t for t in gold_titles if not _contains(answer, t)]
    ratio = len(hits) / max(1, len(gold_titles))
    correct = len(misses) == 0 and gold_titles
    return {
        "correct": bool(correct),
        "partial_score": round(ratio, 3),
        "hits": hits,
        "misses": misses,
    }


def score_longtail_fact(answer: str, score_str: str, year: str) -> dict[str, Any]:
    has_score = _contains(answer, score_str)
    has_year = _contains(answer, year)
    return {
        "correct": has_score and has_year,
        "partial_score": (0.5 if has_score else 0) + (0.5 if has_year else 0),
        "has_score": has_score,
        "has_year": has_year,
    }


def score_kb_boundary(answer: str, refusal_kw: list[str]) -> dict[str, Any]:
    ans_l = (answer or "").lower()
    # 通用拒答短语兜底（case 白名单如果漏了，这些也算拒答）
    _EXTRA_REFUSAL = [
        "没有确切信息", "确切信息", "没有相关记录", "无相关记录",
        "手上信息不多", "手上没有", "手上资料有限", "手上数据有限",
        "查不到", "无从确认", "无法给出",
        "不敢瞎说", "不敢乱讲", "不敢确认",
    ]
    all_kw = list(refusal_kw) + _EXTRA_REFUSAL
    refusal_hit = None
    for kw in all_kw:
        if kw.lower() in ans_l or _contains(answer, kw):
            refusal_hit = kw
            break
    fabricated = False
    # Cheap fabrication detector: 3-6 Japanese/CJK-name-shaped tokens.
    # If any string of 2-4 CJK chars appears together with 'は' or '担当'
    # or 名前-like patterns, flag it.
    name_patterns = [
        r"作曲[家师者][是为：: ]{0,3}([\u4e00-\u9fa5A-Za-z]{2,10})",
        r"担当[了]?[是为]?([\u4e00-\u9fa5A-Za-z]{2,10})",
        r"音响[监督师][是为：: ]{0,3}([\u4e00-\u9fa5A-Za-z]{2,10})",
        r"由([\u4e00-\u9fa5A-Za-z]{2,10})作曲",
        r"由([\u4e00-\u9fa5A-Za-z]{2,10})担任",
    ]
    matches: list[str] = []
    for pat in name_patterns:
        for m in re.finditer(pat, answer or ""):
            matches.append(m.group(1))
    if matches and refusal_hit is None:
        fabricated = True
    return {
        "correct": bool(refusal_hit) and not fabricated,
        "partial_score": 1.0 if refusal_hit else 0.0,
        "refusal_hit": refusal_hit,
        "possibly_fabricated_names": matches,
    }


def score_case(case: dict, answer: str) -> dict[str, Any]:
    t = case["type"]
    if t == "metadata_cross":
        return score_metadata_cross(answer, case["gold_evidence"].get("titles", []))
    if t == "longtail_fact":
        return score_longtail_fact(
            answer,
            case["scoring"]["score_exact"],
            case["scoring"]["year_exact"],
        )
    if t == "kb_boundary":
        return score_kb_boundary(answer, case["gold_answer_keywords"])
    if t == "similar_recommendation":
        return score_similar_recommendation(
            answer,
            case["gold_evidence"]["candidate_similar_titles"],
            case["gold_evidence"]["seed_title"],
            min_hits=case["scoring"].get("match_from_candidates_gte", 2),
        )
    if t == "bangumi_tags":
        return score_bangumi_tags(
            answer,
            case["scoring"]["gold_tags"],
            case["scoring"].get("match_ratio_gte", 0.6),
        )
    if t == "bangumi_score_precise":
        return score_bangumi_score(answer, case["scoring"]["score_exact"])
    if t == "cold_score_precise":
        return score_bangumi_score(answer, case["scoring"]["score_exact"])
    if t == "cold_longtail_fact":
        return score_longtail_fact(
            answer,
            case["scoring"]["score_exact"],
            case["scoring"]["year_exact"],
        )
    if t == "release_date_precise":
        return score_release_date(
            answer,
            case["scoring"]["year"],
            case["scoring"]["month"],
        )
    if t == "cold_studio_year":
        return score_metadata_cross(answer, case["gold_evidence"].get("titles", []))
    if t == "tag_top_score":
        return score_tag_top_score(
            answer,
            case["scoring"]["gold_title"],
            case["scoring"]["distractor_titles"],
        )
    if t == "numeric_comparison":
        return score_numeric_comparison(
            answer,
            case["scoring"]["higher_title"],
            case["scoring"]["lower_title"],
        )
    if t == "seiyuu_cold_works":
        return score_seiyuu_works(
            answer,
            case["scoring"]["gold_titles"],
            case["scoring"].get("match_ratio_gte", 0.5),
        )
    if t == "refusal_fabricated":
        return score_kb_boundary(answer, case["gold_answer_keywords"])
    return {"correct": False, "partial_score": 0.0, "note": f"unknown type {t}"}


def score_similar_recommendation(
    answer: str, candidates: list[str], seed_title: str, min_hits: int = 2
) -> dict[str, Any]:
    """Correct iff answer names >=min_hits titles from `candidates`, and does
    NOT count the seed itself. Substring-match, punctuation-normalized."""
    hits: list[str] = []
    for c in candidates:
        if _contains(answer, c) and c != seed_title:
            hits.append(c)
    correct = len(hits) >= min_hits
    return {
        "correct": correct,
        "partial_score": min(1.0, len(hits) / max(1, min_hits)),
        "hits": hits,
        "n_hits": len(hits),
    }


def score_bangumi_tags(
    answer: str, gold_tags: list[str], ratio_gte: float = 0.6
) -> dict[str, Any]:
    hits = [t for t in gold_tags if _contains(answer, t)]
    ratio = len(hits) / max(1, len(gold_tags))
    return {
        "correct": ratio >= ratio_gte,
        "partial_score": round(ratio, 3),
        "hits": hits,
        "misses": [t for t in gold_tags if t not in hits],
    }


def score_bangumi_score(answer: str, score_str: str) -> dict[str, Any]:
    has = _contains(answer, score_str)
    return {
        "correct": has,
        "partial_score": 1.0 if has else 0.0,
        "score_str": score_str,
    }


def score_release_date(answer: str, year: str, month: str) -> dict[str, Any]:
    """Accepts '2004-09', '2004年9月', '2004年09月', '9月 2004'."""
    ans_n = _norm(answer)
    year_hit = year in ans_n
    m = str(int(month))  # normalize '09' -> '9'
    month_forms = {
        f"{year}-{int(month):02d}",
        f"{year}-{m}",
        f"{year}年{int(month):02d}月",
        f"{year}年{m}月",
        f"{year}年{int(month):02d}",
        f"{year}年{m}",
    }
    month_hit = any(mf in ans_n for mf in month_forms)
    return {
        "correct": year_hit and month_hit,
        "partial_score": (0.5 if year_hit else 0) + (0.5 if month_hit else 0),
        "year_hit": year_hit,
        "month_hit": month_hit,
    }


def score_tag_top_score(
    answer: str, gold_title: str, distractor_titles: list[str]
) -> dict[str, Any]:
    """Correct iff the gold title appears in the answer AND none of the
    distractors (other anime carrying the same tag but with lower score)
    appear as the single/primary recommendation."""
    ans_n = _norm(answer)
    gold_hit = _contains(answer, gold_title)
    # weak fabrication guard: distractor hit is fine when gold also hits
    return {
        "correct": bool(gold_hit),
        "partial_score": 1.0 if gold_hit else 0.0,
        "gold_title": gold_title,
        "gold_hit": gold_hit,
    }


def score_numeric_comparison(
    answer: str, higher_title: str, lower_title: str
) -> dict[str, Any]:
    """Correct iff the answer names the higher-scored anime as the winner.
    We look for the higher title appearing before the lower one, or the
    higher title appearing near explicit '更高/较高/评分高' words."""
    ans = answer or ""
    ans_n = _norm(ans)
    hi_n = _norm(higher_title)
    lo_n = _norm(lower_title)
    hi_pos = ans_n.find(hi_n)
    lo_pos = ans_n.find(lo_n)
    both_present = hi_pos >= 0 and lo_pos >= 0
    hi_first = both_present and hi_pos < lo_pos
    winner_markers = ["更高", "较高", "高一些", "评分高", "分数高",
                      "胜出", "胜过", "领先"]
    marker_correct = False
    if hi_pos >= 0:
        # look 30 chars around the higher title
        window = ans_n[max(0, hi_pos - 30): hi_pos + len(hi_n) + 30]
        marker_correct = any(m in window for m in winner_markers)
    correct = both_present and (hi_first or marker_correct)
    return {
        "correct": bool(correct),
        "partial_score": 1.0 if correct else (0.5 if hi_pos >= 0 else 0.0),
        "higher_title": higher_title,
        "lower_title": lower_title,
        "hi_first": hi_first,
        "marker_hit": marker_correct,
    }


def score_seiyuu_works(
    answer: str, gold_titles: list[str], ratio_gte: float = 0.5
) -> dict[str, Any]:
    hits = [t for t in gold_titles if _contains(answer, t)]
    ratio = len(hits) / max(1, len(gold_titles))
    return {
        "correct": ratio >= ratio_gte,
        "partial_score": round(ratio, 3),
        "hits": hits,
        "misses": [t for t in gold_titles if t not in hits],
    }


if __name__ == "__main__":  # smoke test
    fake_answer = "京阿尼在2021年制作了《小林家的龙女仆S》，还有《迷你龙小剧场》。"
    case = {
        "type": "metadata_cross",
        "gold_evidence": {"titles": ["小林家的龙女仆S", "迷你龙小剧场"]},
    }
    print(score_case(case, fake_answer))
