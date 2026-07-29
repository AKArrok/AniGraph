"""Step A probe: for each case that returned empty content in B1, rerun the
same prompt with higher max_tokens (2048) to test the "reasoning ate the
output" hypothesis.

No changes to run_ablation.py or the persisted ablation_results.json.
Prints per-case verdict; writes tests/probe_empty_answers.json.
"""
from __future__ import annotations

import json
import os
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

ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "_emb_cache.json"
RESULTS_PATH = ROOT / "ablation_results.json"
OUT_PATH = ROOT / "probe_empty_answers.json"

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    EMB_CACHE = json.load(f)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
pinecone_idx = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
TARGET_MODEL = config.LLM_MODEL  # deepseek-v4-pro
FLASH_MODEL = "deepseek-v4-flash"


def cached_embed(query: str):
    if query in EMB_CACHE:
        return EMB_CACHE[query]
    vec = embeddings.embed_query(query)
    EMB_CACHE[query] = vec
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(EMB_CACHE, f)
    time.sleep(4)
    return vec


def retrieve_dense(query: str, top_k: int = 20):
    emb = cached_embed(query)
    resp = pinecone_idx.query(
        vector=emb, top_k=top_k, include_metadata=True,
        namespace=getattr(config, "PINECONE_NAMESPACE", ""),
    )
    return [m["metadata"].get("text", "") for m in resp.get("matches", []) if m.get("metadata")]


def llm_call(model: str, system: str, user: str, max_tokens: int):
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
                timeout=90,
            )
            return {
                "content": resp.choices[0].message.content or "",
                "latency": time.time() - t0,
                "total": resp.usage.total_tokens,
                "completion": resp.usage.completion_tokens,
                "prompt": resp.usage.prompt_tokens,
            }
        except Exception as e:
            print(f"    retry {attempt+1}/3 after {2**attempt}s: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("llm_call failed after retries")


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    b1 = data.get("details_b1", [])

    # Collect cases whose answer is empty and query is non-empty (skip ED002)
    targets = [r for r in b1 if not (r.get("answer") or "").strip() and (r.get("query") or "").strip()]
    print(f"probing {len(targets)} empty-answer cases from B1")

    system = ("You are an anime expert. Use the provided search results to answer. "
              "Cite sources when possible. Answer in Chinese.")

    out = []
    for r in targets:
        q = r["query"]
        cid = r["id"]
        docs = retrieve_dense(q, top_k=20)
        ctx = "\n\n".join(d[:400] for d in docs[:5])
        user = f"Search results:\n{ctx}\n\nQuestion: {q}"

        # Variant P1: v4-pro + max_tokens 2048 (original prompt)
        p1 = llm_call(TARGET_MODEL, system, user, max_tokens=2048)
        # Variant P2: flash + max_tokens 512 (baseline for control)
        p2 = llm_call(FLASH_MODEL, system, user, max_tokens=512)

        row = {
            "id": cid,
            "query": q,
            "original_completion_tokens": r.get("completion_tokens"),
            "original_answer_len": len(r.get("answer") or ""),
            "p1_v4pro_maxtok2048": {
                "ans_len": len(p1["content"]),
                "completion": p1["completion"],
                "total": p1["total"],
                "latency": round(p1["latency"], 2),
                "answer_preview": p1["content"][:300],
            },
            "p2_flash_maxtok512": {
                "ans_len": len(p2["content"]),
                "completion": p2["completion"],
                "total": p2["total"],
                "latency": round(p2["latency"], 2),
                "answer_preview": p2["content"][:300],
            },
        }
        out.append(row)
        print(
            f"  {cid}: v4pro/2048 -> comp={p1['completion']} len={len(p1['content'])} | "
            f"flash/512 -> comp={p2['completion']} len={len(p2['content'])}"
        )
        # Save after each case
        OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
