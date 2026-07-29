import json
d = json.load(open("tests/ablation_results.json", encoding="utf-8"))
b0 = {r["id"]: r for r in d["details_b0"]}
b1 = {r["id"]: r for r in d.get("details_b1", d.get("details_b1_partial", []))}
print(f"{'ID':<8} {'Type':<20} {'Diff':<8} {'B0':>6} {'B1':>6} {'Delta':>7}  Query")
print("-" * 100)
for cid in b0:
    if cid not in b1: continue
    b0_s = b0[cid].get("judge", {}).get("overall", 0)
    b1_s = b1[cid].get("judge", {}).get("overall", 0)
    delta = b1_s - b0_s
    query = b0[cid].get("query", "")[:40]
    diff = b0[cid].get("difficulty", "")
    print(f"{cid:<8} {b0[cid].get('type',''):<20} {diff:<8} {b0_s:>6.2f} {b1_s:>6.2f} {delta:>+7.2f}  {query}")

# Where did B1 regress vs B0?
regressions = [cid for cid in b0 if cid in b1 and (b1[cid].get("judge",{}).get("overall",0) < b0[cid].get("judge",{}).get("overall",0))]
improvements = [cid for cid in b0 if cid in b1 and (b1[cid].get("judge",{}).get("overall",0) > b0[cid].get("judge",{}).get("overall",0))]
print()
print(f"B1 regressions vs B0: {len(regressions)} cases  {regressions}")
print(f"B1 improvements vs B0: {len(improvements)} cases  {improvements}")
