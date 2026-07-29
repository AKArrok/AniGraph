"""Metadata-filter augmentation for the two failure modes in H1..H6.

Two case types the pure Dense-RAG variants can't handle:
  * cold_studio_year   — needs "list every anime by studio S in year Y"
  * seiyuu_cold_works  — needs "list every anime featuring seiyuu N"

A single vector query can't enumerate all rows that satisfy an exact
metadata predicate.  Solution: bypass the retriever for those queries,
use `agents.metadata_index.MetadataIndex.search()` (already backed by
`data/metadata_index.json`), inject the exact candidate list into the
prompt, and let the LLM format the answer.

This is a targeted comparison run: we only score the 3 + 4 = 7 cases
that failed under Dense-only variants, tag it as H7, and merge into
`tests/hard_eval_results.json`.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import config
from agents.metadata_index import MetadataIndex
from tests.deterministic_scorer import score_case


ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT / "eval_hard_final.json"
OUT_PATH = ROOT / "hard_eval_results.json"

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
TARGET_MODEL = config.LLM_MODEL

META = MetadataIndex()
META.load()


SYSTEM_PROMPT = (
    "You are an anime expert. Answer strictly in Chinese, using ONLY the "
    "candidate list provided below. Do not add anime that are not in the "
    "list. If the list is empty, reply with '未记录'."
)


def llm_call(user: str, max_tokens: int = 1024) -> dict:
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=TARGET_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
                timeout=90,
            )
            return {
                "content": resp.choices[0].message.content or "",
                "latency": time.time() - t0,
                "total": resp.usage.total_tokens,
                "completion": resp.usage.completion_tokens,
            }
        except Exception as e:
            print(f"    llm retry {attempt + 1}/4: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("llm_call exhausted retries")


def resolve_case(case: dict) -> tuple[list[str], dict]:
    """Route a case to MetadataIndex.search() and return the candidate list.

    Returns (titles, filter_used).
    """
    t = case["type"]
    ge = case["gold_evidence"]
    if t == "cold_studio_year":
        studio = ge["studio"]
        year = ge["year"]
        rows = META.search(
            studio=studio,
            date_from=f"{year}-01-01",
            date_to=f"{year}-12-31",
            limit=100,
        )
        titles = [(r.get("name_cn") or r.get("name") or "").strip() for r in rows]
        titles = [t for t in titles if t]
        return titles, {"studio": studio, "year": year}

    if t == "seiyuu_cold_works":
        name = ge["seiyuu_name"]
        rows = META.search(seiyuu=name, limit=100)
        titles = [(r.get("name_cn") or r.get("name") or "").strip() for r in rows]
        titles = [t for t in titles if t]
        return titles, {"seiyuu": name}

    raise ValueError(f"unsupported case type: {t}")


def run_variant(cases: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in cases:
        q = c["query"]
        t0 = time.time()
        try:
            titles, flt = resolve_case(c)
        except Exception as e:
            out.append({
                "id": c["id"], "type": c["type"], "query": q,
                "answer": "", "correct": False, "partial_score": 0.0,
                "score_detail": {"error": f"resolve failed: {e}"},
                "latency": 0, "retrieve_latency": 0, "tokens": 0,
                "completion_tokens": 0, "num_docs": 0,
            })
            continue
        retrieve_lat = time.time() - t0

        if titles:
            candidate_block = "\n".join(f"- {t}" for t in titles)
        else:
            candidate_block = "(空)"
        user = (
            f"下面是知识库结构化过滤器返回的候选作品（过滤条件 = {flt}）:\n"
            f"{candidate_block}\n\n"
            f"问题: {q}\n"
            f"请只依据上述候选作品作答，把每一部都列出来。"
        )
        try:
            r = llm_call(user)
        except Exception as e:
            out.append({
                "id": c["id"], "type": c["type"], "query": q,
                "answer": "", "correct": False, "partial_score": 0.0,
                "score_detail": {"error": str(e)[:200]},
                "latency": retrieve_lat, "retrieve_latency": retrieve_lat,
                "tokens": 0, "completion_tokens": 0, "num_docs": len(titles),
            })
            continue

        answer = r["content"]
        score = score_case(c, answer)
        out.append({
            "id": c["id"], "type": c["type"], "query": q, "answer": answer,
            "correct": bool(score.get("correct")),
            "partial_score": score.get("partial_score", 0.0),
            "score_detail": score,
            "latency": round(retrieve_lat + r["latency"], 3),
            "retrieve_latency": round(retrieve_lat, 3),
            "tokens": r["total"], "completion_tokens": r["completion"],
            "num_docs": len(titles),
            "filter_used": flt,
            "candidate_titles": titles,
        })
        ok = "T" if score.get("correct") else "F"
        print(
            f"  H7 {c['id']:26} type={c['type']:18} "
            f"docs={len(titles):>2} correct={ok} "
            f"partial={score.get('partial_score', 0):.2f} "
            f"lat={retrieve_lat + r['latency']:.1f}s"
        )
    return out


def main() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    subset = [c for c in cases if c["type"] in ("cold_studio_year", "seiyuu_cold_works")]
    print(f"H7 metadata-filter runs on {len(subset)} cases")

    results = run_variant(subset)

    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    payload["details_h7"] = results
    payload.setdefault("summary", {})

    # Aggregate per-type numbers
    by_type: dict[str, dict] = {}
    for r in results:
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "partial": 0.0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
        b["partial"] += r["partial_score"]
    for b in by_type.values():
        b["acc"] = round(b["correct"] / b["n"], 3)
        b["partial"] = round(b["partial"] / b["n"], 3)

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    payload["summary"]["details_h7"] = {
        "name": "H7: MetadataIndex.search + LLM (studio_year & seiyuu only)",
        "n": n,
        "strict_accuracy": correct / n if n else 0.0,
        "partial_score": (sum(r["partial_score"] for r in results) / n) if n else 0.0,
        "by_type": by_type,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== H7 summary ===")
    print(f"  strict acc: {correct}/{n} = {correct/n:.1%}")
    for t, b in by_type.items():
        print(f"  {t}: {b['correct']}/{b['n']} acc={b['acc']:.2f} partial={b['partial']:.2f}")


if __name__ == "__main__":
    main()
