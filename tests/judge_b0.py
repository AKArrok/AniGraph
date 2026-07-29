"""B0 detailed evaluation: re-run + multi-dimensional LLM-as-judge."""
import json, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
import config

JUDGE_MODEL = "deepseek-v4-flash"
TARGET_MODEL = config.LLM_MODEL
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

def llm_call(model, system, user, max_tokens=512, temperature=0):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""
    return content, elapsed, resp.usage.total_tokens

def load_cases():
    with open("tests/eval_dataset.json", encoding="utf-8") as f:
        all_cases = json.load(f)["cases"]
    by_type = {}
    for c in all_cases:
        by_type.setdefault(c["type"], []).append(c)
    selected = []
    for t in ["simple_fact","recommendation","comparison","alias_resolution","multi_turn","edge_case","knowledge_boundary"]:
        items = by_type.get(t, [])
        for diff in ["easy","medium","hard"]:
            pool = [c for c in items if c["difficulty"]==diff]
            if pool:
                selected.append(pool[0])
                if len(selected) >= 20:
                    break
        if len(selected) >= 20:
            break
    return selected

def judge_detailed(query, gold_keywords, gold_routing, answer):
    system = """You are a strict evaluator of AI anime Q&A responses. Score each dimension 1-5.
1=completely wrong/irrelevant, 3=partially correct, 5=perfect.
Output ONLY valid JSON: {"correctness":N,"completeness":N,"relevance":N,"hallucination_risk":N,"verdict":"pass"|"fail"|"partial","reason":"brief Chinese explanation"}"""
    user = f"""Query: {query}
Expected routing: {gold_routing}
Expected keywords/entities: {gold_keywords}
Answer to evaluate:
{answer[:1000]}

Judge the answer. Does it correctly answer the query? Is it complete? Is every claim grounded or could be hallucinated?"""
    resp, _, _ = llm_call(JUDGE_MODEL, system, user, max_tokens=300, temperature=0)
    try:
        resp = resp.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(resp)
    except:
        return {"correctness":0,"completeness":0,"relevance":0,"hallucination_risk":0,"verdict":"error","reason":resp[:100]}

def main():
    cases = load_cases()
    print(f"Evaluating B0 on {len(cases)} cases with multi-dimensional judge\n")

    system = "You are a knowledgeable anime expert. Answer the query directly and concisely in Chinese."
    results = []

    for i, c in enumerate(cases):
        q = c["query"] if isinstance(c["query"], str) else c["query"][0]
        print(f"[{i+1}/{len(cases)}] {c['id']} {q[:60]}...", end=" ", flush=True)

        # Step 1: Get B0 answer
        answer, lat, tokens = llm_call(TARGET_MODEL, system, q, max_tokens=512)

        # Step 2: Multi-dimensional judge
        gold_kw = c.get("gold_answer_keywords", [])
        gold_route = c.get("gold_routing", "")
        scores = judge_detailed(q, gold_kw, gold_route, answer)

        verdict = scores.get("verdict", "?")
        print(f"{verdict} c={scores.get('correctness',0)} m={scores.get('completeness',0)} r={scores.get('relevance',0)} h={scores.get('hallucination_risk',0)} | {lat:.1f}s")

        results.append({
            "id": c["id"],
            "type": c["type"],
            "difficulty": c["difficulty"],
            "query": q,
            "gold_keywords": gold_kw,
            "gold_routing": gold_route,
            "answer": answer[:500],
            "latency": lat,
            "tokens": tokens,
            "correctness": scores.get("correctness", 0),
            "completeness": scores.get("completeness", 0),
            "relevance": scores.get("relevance", 0),
            "hallucination_risk": scores.get("hallucination_risk", 0),
            "verdict": verdict,
            "judge_reason": scores.get("reason", ""),
        })

    # Summary
    n = len(results)
    avg_c = statistics.mean(r["correctness"] for r in results)
    avg_m = statistics.mean(r["completeness"] for r in results)
    avg_r = statistics.mean(r["relevance"] for r in results)
    avg_h = statistics.mean(r["hallucination_risk"] for r in results)
    passes = sum(1 for r in results if r["verdict"] == "pass")
    partials = sum(1 for r in results if r["verdict"] == "partial")
    fails = sum(1 for r in results if r["verdict"] == "fail")
    avg_lat = statistics.mean(r["latency"] for r in results)
    avg_tok = int(statistics.mean(r["tokens"] for r in results))

    print(f"\n{'='*70}")
    print(f"  B0 Multi-Dimensional Judge Results (n={n})")
    print(f"  Correctness:      {avg_c:.1f}/5")
    print(f"  Completeness:     {avg_m:.1f}/5")
    print(f"  Relevance:        {avg_r:.1f}/5")
    print(f"  Hallucination Risk:{avg_h:.1f}/5 (5=no hallucination)")
    print(f"  Verdicts: {passes} pass, {partials} partial, {fails} fail")
    print(f"  Avg latency: {avg_lat:.1f}s")
    print(f"  Avg tokens: {avg_tok}")
    print(f"{'='*70}")

    # Per-type breakdown
    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, []).append(r)
    print("\n  By query type:")
    for t in ["simple_fact","recommendation","comparison","alias_resolution","multi_turn","edge_case","knowledge_boundary"]:
        items = by_type.get(t, [])
        if items:
            avg = statistics.mean(x["correctness"] for x in items)
            pf = f"{sum(1 for x in items if x['verdict']=='pass')}/{sum(1 for x in items if x['verdict']=='partial')}/{sum(1 for x in items if x['verdict']=='fail')}"
            print(f"    {t:<20}: correctness={avg:.1f} verdicts={pf}")

    out_path = "tests/b0_detailed_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {
            "n": n, "avg_correctness": avg_c, "avg_completeness": avg_m,
            "avg_relevance": avg_r, "avg_hallucination_risk": avg_h,
            "passes": passes, "partials": partials, "fails": fails,
            "avg_latency": avg_lat, "avg_tokens": avg_tok,
        }, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results: {out_path}")

if __name__ == "__main__":
    main()
