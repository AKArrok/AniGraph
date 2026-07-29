import json
d = json.load(open("tests/ablation_results.json", encoding="utf-8"))
b1 = {r["id"]: r for r in d.get("details_b1", d.get("details_b1_partial", []))}
for cid in ["CP004", "ED004", "KB001", "MT001", "AL001"]:
    r = b1.get(cid, {})
    print("=" * 90)
    print(f"{cid}: {(r.get('query') or '')[:80]}")
    ans = r.get("answer") or ""
    print(f"Answer: {ans[:500]}")
    print(f"Judge reason: {r.get('judge',{}).get('reason','')}")
    docs = r.get("retrieved_docs", [])
    if docs:
        print(f"Top doc: {docs[0][:250]}")
    print()
