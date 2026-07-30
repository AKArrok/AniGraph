"""AniRAG ablation runner: B0 (LLM-only) and B1 (Dense RAG) on eval_dataset."""
import json, os, sys, time, statistics, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
import config
from pinecone import Pinecone
from llms import embeddings

# ── Embedding cache to avoid burning quota on reruns ──
_EMB_CACHE_PATH = "tests/_emb_cache.json"
try:
    with open(_EMB_CACHE_PATH, encoding="utf-8") as _f:
        _EMB_CACHE = json.load(_f)
except Exception:
    _EMB_CACHE = {}

def _cached_embed(query):
    if query in _EMB_CACHE:
        return _EMB_CACHE[query]
    vec = embeddings.embed_query(query)
    _EMB_CACHE[query] = vec
    with open(_EMB_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_EMB_CACHE, f)
    time.sleep(4.0)   # extra guard on top of the 3s interval baked into the client
    return vec

BATCH_SIZE = 25
JUDGE_MODEL = "deepseek-v4-flash"
TARGET_MODEL = config.LLM_MODEL

client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_idx = pc.Index(config.PINECONE_INDEX)

def load_subset(path="tests/eval_dataset.json", n=BATCH_SIZE):
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    by_type = {}
    for c in cases:
        by_type.setdefault(c["type"], []).append(c)
    selected = []
    for t in ["simple_fact","recommendation","comparison","alias_resolution","multi_turn","edge_case","knowledge_boundary"]:
        items = by_type.get(t, [])
        for diff in ["easy","medium","hard"]:
            pool = [c for c in items if c["difficulty"] == diff]
            if pool:
                selected.append(pool[0])
                if len(selected) >= n:
                    break
        if len(selected) >= n:
            break
    return selected

def llm_call(model, system, user, max_tokens=1500, temperature=0):
    last_err = None
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
                timeout=60,
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            return content, elapsed, usage.total_tokens, usage.completion_tokens
        except Exception as e:
            last_err = e
            backoff = 2 ** attempt
            print(f"    llm_call retry {attempt + 1}/4 after {backoff}s: {e}")
            time.sleep(backoff)
    raise last_err

def judge_answer_detailed(query, gold_keywords, gold_entities, answer, docs=None):
    """LLM-as-judge with multi-dimensional scoring. Returns dict with scores and reasoning."""
    ctx = ""
    if docs:
        ctx = "Retrieved context:\n" + "\n---\n".join(d[:500] for d in docs[:3])
    system = (
        "You are a strict evaluator for an anime QA system. "
        "Score the answer on 3 dimensions (0-1 scale) and provide a brief reason. "
        "Output JSON only: "
        '{"relevance": float, "accuracy": float, "completeness": float, '
        '"overall": float, "hallucination": bool, "reason": "brief Chinese"}'
    )
    user = (
        f"Query: {query}\n"
        f"Expected entities: {gold_entities}\n"
        f"Expected keywords: {gold_keywords}\n"
        f"{ctx}\n"
        f"Answer: {answer[:1000]}"
    )
    resp_text, _, _, _ = llm_call(JUDGE_MODEL, system, user, max_tokens=800)
    txt = (resp_text or "").strip()
    # Extract JSON block robustly
    for token in ("```json", "```"):
        if token in txt:
            txt = txt.split(token, 1)[-1]
    txt = txt.rsplit("```", 1)[0].strip()
    # Try to isolate the outermost JSON object
    start = txt.find("{"); end = txt.rfind("}")
    if start >= 0 and end > start:
        txt = txt[start:end+1]
    try:
        parsed = json.loads(txt)
        # Ensure required fields present
        for k in ("relevance","accuracy","completeness","overall"):
            parsed.setdefault(k, 0)
        return parsed
    except Exception as e:
        return {"relevance":0,"accuracy":0,"completeness":0,"overall":0,
                "hallucination":True,"reason":f"parse_error: {str(e)[:80]} | raw={resp_text[:200]}"}


def retrieve_dense(query, top_k=None):
    t0 = time.time()
    if top_k is None:
        top_k = config.HYBRID_DENSE_K
    namespace = getattr(config, 'PINECONE_NAMESPACE', '')
    emb = _cached_embed(query)
    resp = pinecone_idx.query(
        vector=emb, top_k=top_k, include_metadata=True,
        namespace=namespace
    )
    elapsed = time.time() - t0
    docs = [m["metadata"].get("text","") for m in resp.get("matches",[]) if m.get("metadata")]
    return docs, elapsed

def judge_answer(query, gold_keywords, answer):
    system = "You are an evaluator. Judge if the answer addresses the query and contains the expected information. Output JSON only: {\"correct\": true/false, \"reason\": \"brief\"}"
    user = f"Query: {query}\nExpected keywords: {gold_keywords}\nAnswer: {answer[:800]}"
    resp_text, _, _, _ = llm_call(JUDGE_MODEL, system, user, max_tokens=200)
    try:
        resp_text = resp_text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(resp_text).get("correct", False)
    except:
        return False

def faithfulness_score(answer, docs):
    if not docs or not answer:
        return 0.0
    doc_text = " ".join(d[:300] for d in docs[:5])
    ans_words = set(answer)
    doc_words = set(doc_text)
    if not ans_words:
        return 0.0
    return len(ans_words & doc_words) / len(ans_words)

def run_b0(cases):
    results = []
    system = "You are a knowledgeable anime expert. Answer the query directly and concisely in Chinese."
    for c in cases:
        q = c["query"] if isinstance(c["query"], str) else c["query"][0]
        answer, lat, tokens, comp_tok = llm_call(TARGET_MODEL, system, q, max_tokens=2048)
        judge = judge_answer_detailed(
            q, c.get("gold_answer_keywords",[]),
            c.get("gold_entities",[]), answer
        )
        correct = judge.get("overall", 0) >= 0.6
        results.append({
            "id": c["id"], "type": c["type"], "difficulty": c["difficulty"],
            "query": q, "answer": answer,
            "gold_entities": c.get("gold_entities",[]),
            "gold_keywords": c.get("gold_answer_keywords",[]),
            "gold_routing": c.get("gold_routing",""),
            "latency": lat, "tokens": tokens, "completion_tokens": comp_tok,
            "correct": correct, "faithfulness": 0.0, "judge": judge,
        })
        print(f"  B0 {c['id']}: judge={judge.get('overall',0):.2f} rel={judge.get('relevance',0):.2f} acc={judge.get('accuracy',0):.2f} comp={judge.get('completeness',0):.2f} lat={lat:.1f}s")
    return results

def run_b1(cases):
    results = []
    system = "You are an anime expert. Use the provided search results to answer. Cite sources when possible. Answer in Chinese."
    for c in cases:
        q = c["query"] if isinstance(c["query"], str) else c["query"][0]
        docs, ret_lat = retrieve_dense(q)
        context = "\n\n".join(d[:400] for d in docs[:5])
        user = f"Search results:\n{context}\n\nQuestion: {q}"
        answer, llm_lat, tokens, comp_tok = llm_call(TARGET_MODEL, system, user, max_tokens=2048)
        total_lat = ret_lat + llm_lat
        judge = judge_answer_detailed(
            q, c.get("gold_answer_keywords",[]),
            c.get("gold_entities",[]), answer, docs
        )
        correct = judge.get("overall", 0) >= 0.6
        faith = faithfulness_score(answer, docs)
        results.append({
            "id": c["id"], "type": c["type"], "difficulty": c["difficulty"],
            "query": q, "answer": answer,
            "gold_entities": c.get("gold_entities",[]),
            "gold_keywords": c.get("gold_answer_keywords",[]),
            "gold_routing": c.get("gold_routing",""),
            "latency": total_lat, "retrieval_latency": ret_lat,
            "tokens": tokens, "completion_tokens": comp_tok,
            "correct": correct, "faithfulness": faith, "judge": judge,
            "retrieved_docs": [d[:200] for d in docs[:5]],
        })
        print(f"  B1 {c['id']}: judge={judge.get('overall',0):.2f} rel={judge.get('relevance',0):.2f} acc={judge.get('accuracy',0):.2f} faith={faith:.2f} lat={total_lat:.1f}s")
    return results

def summarize(name, results):
    n = len(results)
    if n == 0:
        return
    correct = sum(1 for r in results if r["correct"])
    faith = statistics.mean(r["faithfulness"] for r in results)
    lats = sorted(r["latency"] for r in results)
    p95 = lats[int(0.95 * (n-1))] if n > 1 else lats[0]
    avg_lat = statistics.mean(lats)
    avg_tok = int(statistics.mean(r["tokens"] for r in results))
    avg_comp = int(statistics.mean(r.get("completion_tokens", r.get("tokens", 0)) for r in results))
    print(f"\n{'='*70}")
    print(f"  {name} (n={n})")
    print(f"  Accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"  Faithfulness: {faith:.3f}")
    print(f"  Latency: avg={avg_lat:.1f}s p95={p95:.1f}s")
    print(f"  Tokens: avg={avg_tok} completion={avg_comp}")
    print(f"{'='*70}")
    return {
        "name": name, "n": n, "accuracy": correct/n,
        "faithfulness": faith, "avg_latency": avg_lat,
        "p95_latency": p95, "avg_tokens": avg_tok,
        "avg_completion_tokens": avg_comp,
    }


def run_b_variant(cases, *, top_k, cite_sources, tag):
    """Dense RAG variant with configurable top_k and prompt.

    Reuses `_cached_embed` so no fresh embedding calls are made when the query
    is already in tests/_emb_cache.json.
    """
    if cite_sources:
        system = (
            "You are an anime expert. Use the provided search results to answer. "
            "Cite sources when possible. Answer in Chinese."
        )
    else:
        system = (
            "You are an anime expert. Answer the query in Chinese. "
            "You may use the search snippets below as background, but do not mention them."
        )
    results = []
    for c in cases:
        q = c["query"] if isinstance(c["query"], str) else c["query"][0]
        try:
            docs, ret_lat = retrieve_dense(q, top_k=top_k)
            ctx = "\n\n".join(
                f"[{i+1}] {d[:400]}" for i, d in enumerate(docs[:top_k])
            )
            user = f"Search results:\n{ctx}\n\nQuery: {q}"
            answer, llm_lat, tokens, comp_tok = llm_call(
                TARGET_MODEL, system, user, max_tokens=2048
            )
            judge = judge_answer_detailed(
                q, c.get("gold_answer_keywords", []),
                c.get("gold_entities", []), answer, docs,
            )
            faith = faithfulness_score(answer, docs)
            correct = judge.get("overall", 0) >= 0.6
            results.append({
                "id": c["id"], "type": c["type"], "difficulty": c["difficulty"],
                "query": q, "answer": answer,
                "gold_entities": c.get("gold_entities", []),
                "gold_keywords": c.get("gold_answer_keywords", []),
                "correct": correct, "faithfulness": faith, "judge": judge,
                "latency": ret_lat + llm_lat, "tokens": tokens,
                "completion_tokens": comp_tok,
                "retrieve_latency": ret_lat,
                "num_docs": len(docs),
            })
            print(
                f"  {tag} {c['id']}: judge={judge.get('overall',0):.2f} "
                f"docs={len(docs)} lat={ret_lat + llm_lat:.1f}s"
            )
        except Exception as e:
            print(f"  {tag} {c['id']}: FAILED - {e}")
            results.append({
                "id": c["id"], "type": c["type"], "difficulty": c["difficulty"],
                "query": q, "answer": "",
                "correct": False, "faithfulness": 0.0,
                "judge": {"overall": 0, "relevance": 0, "accuracy": 0,
                          "completeness": 0, "reason": str(e)[:200]},
                "latency": 0, "tokens": 0,
            })
    return results


def main():
    import glob
    cases = load_subset()
    print(f"Running ablation on {len(cases)} cases\n")
    summary = {}
    r0 = None
    r1 = None
    r2 = None
    r3 = None
    r4 = None

    # Check for partial results
    partial_path = "tests/ablation_results.json"
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("details_b0"):
            print("--- B0: SKIPPED (already in results file) ---")
            r0 = prev["details_b0"]
            summary["B0_direct_llm"] = summarize("B0: Direct LLM", r0)
        if prev.get("details_b1"):
            print("--- B1: SKIPPED (already in results file) ---")
            r1 = prev["details_b1"]
            summary["B1_dense_rag"] = summarize("B1: Dense RAG", r1)
        if prev.get("details_b2"):
            print("--- B2: SKIPPED (already in results file) ---")
            r2 = prev["details_b2"]
            summary["B2_dense_topk3"] = summarize("B2: Dense RAG top_k=3", r2)
        if prev.get("details_b3"):
            print("--- B3: SKIPPED (already in results file) ---")
            r3 = prev["details_b3"]
            summary["B3_dense_topk10"] = summarize("B3: Dense RAG top_k=10", r3)
        if prev.get("details_b4"):
            print("--- B4: SKIPPED (already in results file) ---")
            r4 = prev["details_b4"]
            summary["B4_no_cite_prompt"] = summarize("B4: no-cite prompt top_k=20", r4)

    if r0 is None:
        print("--- B0: Direct LLM (no RAG) ---")
        r0 = run_b0(cases)
        summary["B0_direct_llm"] = summarize("B0: Direct LLM", r0)
        # Save intermediate
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details_b0": r0}, f, ensure_ascii=False, indent=2)

    if r1 is None:
        print("\n--- B1: Dense RAG + single LLM ---")
        r1 = []
        for i, c in enumerate(cases):
            try:
                r1_partial = run_b1([c])
                r1.extend(r1_partial)
            except Exception as e:
                print(f"  B1 {c['id']}: FAILED - {e}")
                r1.append({"id": c["id"], "type": c["type"], "difficulty": c["difficulty"],
                          "query": c["query"] if isinstance(c["query"],str) else c["query"][0],
                          "correct": False, "judge": {"overall":0,"relevance":0,"accuracy":0,"completeness":0,"reason":str(e)[:200]},
                          "latency": 0, "tokens": 0, "faithfulness": 0.0})
            # Save after each case
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details_b0": r0, "details_b1_partial": r1}, f, ensure_ascii=False, indent=2)
        summary["B1_dense_rag"] = summarize("B1: Dense RAG", r1)

    def _dump():
        payload = {"summary": summary, "details_b0": r0, "details_b1": r1}
        if r2 is not None:
            payload["details_b2"] = r2
        if r3 is not None:
            payload["details_b3"] = r3
        if r4 is not None:
            payload["details_b4"] = r4
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if r2 is None:
        print("\n--- B2: Dense RAG top_k=3 (tight retrieval) ---")
        r2 = []
        for c in cases:
            r2.extend(run_b_variant([c], top_k=3, cite_sources=True, tag="B2"))
            _dump()
        summary["B2_dense_topk3"] = summarize("B2: Dense RAG top_k=3", r2)
        _dump()

    if r3 is None:
        print("\n--- B3: Dense RAG top_k=10 ---")
        r3 = []
        for c in cases:
            r3.extend(run_b_variant([c], top_k=10, cite_sources=True, tag="B3"))
            _dump()
        summary["B3_dense_topk10"] = summarize("B3: Dense RAG top_k=10", r3)
        _dump()

    if r4 is None:
        print("\n--- B4: Dense RAG top_k=20, no-cite prompt ---")
        r4 = []
        for c in cases:
            r4.extend(run_b_variant([c], top_k=20, cite_sources=False, tag="B4"))
            _dump()
        summary["B4_no_cite_prompt"] = summarize("B4: no-cite prompt top_k=20", r4)
        _dump()

    out_path = partial_path
    _dump()
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
