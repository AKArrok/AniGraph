"""Ablation runner for tests/eval_hard_final.json using deterministic scoring.

Experiments:
  H0  Direct LLM (no RAG)
  H1  Dense RAG top_k=20 (baseline: current config)
  H2  Dense RAG top_k=3  (winner from the easy set)

Per case: embed (cached) -> Pinecone -> LLM -> deterministic score.
Saves after each case to tests/hard_eval_results.json.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import config
from pinecone import Pinecone
from llms import embeddings
from tests.deterministic_scorer import score_case

ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "_emb_cache.json"
EVAL_PATH = ROOT / "eval_hard_final.json"
OUT_PATH = ROOT / "hard_eval_results.json"

EMB_CACHE: dict = {}
if CACHE_PATH.exists():
    EMB_CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
pinecone_idx = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
TARGET_MODEL = config.LLM_MODEL  # deepseek-v4-pro


def cached_embed(query: str):
    if query in EMB_CACHE:
        return EMB_CACHE[query]
    vec = embeddings.embed_query(query)
    EMB_CACHE[query] = vec
    CACHE_PATH.write_text(json.dumps(EMB_CACHE), encoding="utf-8")
    time.sleep(4)
    return vec


def retrieve_dense(query: str, top_k: int):
    emb = cached_embed(query)
    resp = pinecone_idx.query(
        vector=emb, top_k=top_k, include_metadata=True,
        namespace=getattr(config, "PINECONE_NAMESPACE", ""),
    )
    return [m["metadata"].get("text", "") for m in resp.get("matches", []) if m.get("metadata")]


def llm_call(system: str, user: str, max_tokens: int = 2048):
    last_err = None
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=TARGET_MODEL,
                messages=[
                    {"role": "system", "content": system},
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
            last_err = e
            backoff = 2 ** attempt
            print(f"    llm retry {attempt + 1}/4 after {backoff}s: {e}")
            time.sleep(backoff)
    raise last_err


SYSTEMS = {
    "H0": "You are an anime expert. Answer the question directly in Chinese. "
           "If you don't know, say '未记录' rather than guessing.",
    "H1": "You are an anime expert. Use the provided search results to answer. "
           "Answer in Chinese. If the search results don't contain the answer, "
           "reply with '未记录' rather than inventing details.",
    "H2": "You are an anime expert. Use the provided search results to answer. "
           "Answer in Chinese. If the search results don't contain the answer, "
           "reply with '未记录' rather than inventing details.",
    # H3: Dense RAG top_k=10 (middle-ground retrieval size)
    "H3": "You are an anime expert. Use the provided search results to answer. "
           "Answer in Chinese. If the search results don't contain the answer, "
           "reply with '未记录' rather than inventing details.",
    # H4: Dense RAG top_k=20 with a no-cite prompt (prompt ablation vs H1)
    "H4": "You are an anime expert. Answer the query in Chinese. "
           "You may use the search snippets below as background, "
           "but do not mention or cite them.",
    # H5: Dense RAG top_k=40 (probe upper end of top_k curve)
    "H5": "You are an anime expert. Use the provided search results to answer. "
           "Answer in Chinese. If the search results don't contain the answer, "
           "reply with '未记录' rather than inventing details.",
    # H6: Dense RAG top_k=60 (probe further; look for plateau / degradation)
    "H6": "You are an anime expert. Use the provided search results to answer. "
           "Answer in Chinese. If the search results don't contain the answer, "
           "reply with '未记录' rather than inventing details.",
}


def run_variant(cases: list[dict], tag: str, top_k: int | None) -> list[dict]:
    system = SYSTEMS[tag]
    results: list[dict] = []
    for c in cases:
        q = c["query"]
        try:
            docs: list[str] = []
            ret_lat = 0.0
            if top_k is not None:
                t0 = time.time()
                docs = retrieve_dense(q, top_k=top_k)
                ret_lat = time.time() - t0
                ctx = "\n\n".join(f"[{i+1}] {d[:400]}" for i, d in enumerate(docs))
                user = f"Search results:\n{ctx}\n\nQuestion: {q}"
            else:
                user = q

            r = llm_call(system, user, max_tokens=2048)
            answer = r["content"]
            score = score_case(c, answer)
            results.append({
                "id": c["id"],
                "type": c["type"],
                "query": q,
                "answer": answer,
                "correct": score.get("correct", False),
                "partial_score": score.get("partial_score", 0),
                "score_detail": score,
                "latency": round(ret_lat + r["latency"], 3),
                "retrieve_latency": round(ret_lat, 3),
                "tokens": r["total"],
                "completion_tokens": r["completion"],
                "num_docs": len(docs),
            })
            ok = "T" if score.get("correct") else "F"
            partial = score.get("partial_score", 0.0)
            print(
                f"  {tag} {c['id']:8} type={c['type']:14} "
                f"correct={ok}  partial={partial:.2f}  "
                f"lat={ret_lat + r['latency']:.1f}s"
            )
        except Exception as e:
            print(f"  {tag} {c['id']}: FAILED - {e}")
            results.append({
                "id": c["id"], "type": c["type"], "query": q,
                "answer": "", "correct": False, "partial_score": 0.0,
                "score_detail": {"error": str(e)[:200]},
                "latency": 0, "tokens": 0, "completion_tokens": 0,
                "num_docs": 0,
            })
    return results


def summarize(name: str, results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    partial = statistics.mean(r["partial_score"] for r in results)
    lats = sorted(r["latency"] for r in results if r["latency"] > 0)
    p95 = lats[int(0.95 * (len(lats) - 1))] if len(lats) > 1 else (lats[0] if lats else 0)
    avg_lat = statistics.mean(lats) if lats else 0
    avg_tok = int(statistics.mean(r["tokens"] for r in results if r["tokens"]))
    by_type: dict[str, dict] = {}
    for r in results:
        by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        by_type[r["type"]]["n"] += 1
        by_type[r["type"]]["correct"] += int(r["correct"])
    for t in by_type.values():
        t["acc"] = round(t["correct"] / t["n"], 3)
    print(f"\n{'=' * 70}")
    print(f"  {name} (n={n})")
    print(f"  Strict accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"  Partial score:   {partial:.3f}")
    print(f"  Latency: avg={avg_lat:.2f}s p95={p95:.2f}s")
    print(f"  Avg tokens: {avg_tok}")
    print(f"  By type: {by_type}")
    print("=" * 70)
    return {
        "name": name, "n": n,
        "strict_accuracy": correct / n,
        "partial_score": partial,
        "avg_latency": avg_lat, "p95_latency": p95, "avg_tokens": avg_tok,
        "by_type": by_type,
    }


def main() -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(cases)} hard cases")

    payload: dict = {"summary": {}, "cases_meta": [{"id": c["id"], "type": c["type"]} for c in cases]}
    if OUT_PATH.exists():
        try:
            payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    payload.setdefault("summary", {})

    def dump():
        OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_and_store(tag: str, key: str, top_k):
        if payload.get(key):
            print(f"--- {tag}: SKIPPED (cached) ---")
            return
        print(f"\n--- {tag}: top_k={top_k} ---")
        results: list[dict] = []
        for c in cases:
            results.extend(run_variant([c], tag=tag, top_k=top_k))
            payload[key] = results
            dump()
        payload["summary"][key] = summarize(tag, results)
        dump()

    run_and_store("H0", "details_h0", None)
    run_and_store("H1", "details_h1", 20)
    run_and_store("H2", "details_h2", 3)
    run_and_store("H3", "details_h3", 10)
    run_and_store("H4", "details_h4", 20)
    run_and_store("H5", "details_h5", 40)
    run_and_store("H6", "details_h6", 60)

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
