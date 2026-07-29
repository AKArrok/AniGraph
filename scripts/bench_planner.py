"""Benchmark planner layers + metadata coverage for the two proposed optimizations.

Run: python scripts/bench_planner.py
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from collections import Counter

# make repo root importable when invoked as `python scripts/bench_planner.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# ── 1. metadata_index field coverage ────────────────────────────

INDEX_PATH = "data/metadata_index.json"
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

FACT_FIELDS = ["score", "rank", "date", "studio", "director", "writer", "seiyuu", "tags", "alias"]


def _is_nonempty(v):
    if v is None or v == "" or v == []:
        return False
    if isinstance(v, float) and v == 0.0:
        return False
    return True


coverage = {f: sum(1 for r in data if _is_nonempty(r.get(f))) for f in FACT_FIELDS}
total = len(data)
print(f"[metadata_index] total={total}")
for f, n in coverage.items():
    print(f"  {f:<10} {n:>5}  {n/total*100:>5.1f}%")

# ── 2. fuzzy_lookup / alias_dict hit rate on realistic queries ──

from agents.metadata_index import index as md_index
from agents.alias import resolve_alias_dict

SIMPLE_FACT_QUERIES = [
    # 追问 / 短查询 / 别名
    "巨人评分",
    "巨人多少分",
    "进击的巨人评分是多少",
    "鬼灭之刃的声优",
    "鬼灭声优",
    "MAPPA做过什么",
    "咒术回战制作公司",
    "钢炼几集",
    "eva导演",
    "clannad评分",
    "命运石之门评分",
    "sg评分",
    "无职转生豆瓣评分",
    "石头门声优",
    "芙莉莲多少集",
    "路人女主导演",
    "间谍过家家评分",
    "碧蓝之海评分",
    "孤独摇滚评分",
    "86评分",
    # 追问型
    "它评分多少",
    "那部呢",
    "还有别的吗",
]

RECOMMEND_QUERIES = [
    "有没有类似钢炼的番",
    "催泪番推荐",
    "推荐点像巨人的",
    "跟鬼灭一样热血的番",
    "冷门但好看的动漫",
]

CHAT_QUERIES = ["你好", "谢谢", "你能做什么"]

ALL_QUERIES = SIMPLE_FACT_QUERIES + RECOMMEND_QUERIES + CHAT_QUERIES

hit_kinds = Counter()
for q in SIMPLE_FACT_QUERIES:
    if md_index.fuzzy_lookup(q):
        hit_kinds["fuzzy_lookup"] += 1
    elif resolve_alias_dict(q):
        hit_kinds["alias_dict"] += 1
    else:
        hit_kinds["miss"] += 1
print(f"\n[entity hit on simple_fact queries] n={len(SIMPLE_FACT_QUERIES)}")
for k, v in hit_kinds.items():
    print(f"  {k:<14} {v}")

# ── 3. planner layer latency (real LLM) ─────────────────────────

from agents.planner import (
    plan,
    _prefilter,
    _classify_intent,
    _analyze_complexity,
    _refine_strategy,
    _prefilter_cache,
    _plan_cache,
)


def _reset_caches():
    _prefilter_cache.clear()
    _plan_cache.clear()


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


print("\n[planner full plan() cold-cache latency, real LLM]")
cold = []
for q in ALL_QUERIES:
    _reset_caches()
    dt, p = _time(plan, q)
    cold.append((q, dt, p.get("query_category"), p.get("query_type")))
    print(f"  {dt*1000:>7.0f}ms  {p.get('query_category'):<9} {p.get('query_type'):<14} {q}")

dts = [x[1] for x in cold]
print(f"\n  p50={statistics.median(dts)*1000:.0f}ms  p95={sorted(dts)[int(0.95*len(dts))-1]*1000:.0f}ms  mean={statistics.mean(dts)*1000:.0f}ms")

print("\n[planner cache-hit latency (same query)]")
_reset_caches()
for q in ["进击的巨人评分是多少", "有没有类似钢炼的番"]:
    plan(q)  # warm
    dt, _ = _time(plan, q)
    print(f"  {dt*1000:>7.2f}ms  {q}")

print("\n[layer breakdown for one representative simple_fact miss]")
_reset_caches()
q = "石头门声优"
dt1, (route, conf, scores) = _time(_prefilter, q)
print(f"  L1 prefilter (embed): {dt1*1000:.0f}ms route={route} conf={conf:.2f}")
dt2, intent = _time(_classify_intent, q, "", [])
print(f"  L3 intent (simple_LLM): {dt2*1000:.0f}ms -> {intent.query_category}/{intent.query_type}")
dt3, comp = _time(_analyze_complexity, q, intent, "")
print(f"  L4 complexity (simple_LLM): {dt3*1000:.0f}ms is_complex={comp.is_complex}")

print("\n[layer breakdown for a mixed query]")
_reset_caches()
q = "无职转生好看吗"
dt1, (route, conf, scores) = _time(_prefilter, q)
print(f"  L1 prefilter: {dt1*1000:.0f}ms route={route} conf={conf:.2f}")
dt2, intent = _time(_classify_intent, q, "", [])
print(f"  L3 intent: {dt2*1000:.0f}ms -> {intent.query_category}/{intent.query_type}")
if intent.query_category == "mixed":
    dt3, strat = _time(_refine_strategy, q, intent, "")
    print(f"  L4 refine (answer_LLM): {dt3*1000:.0f}ms")
else:
    dt3, comp = _time(_analyze_complexity, q, intent, "")
    print(f"  L4 complexity: {dt3*1000:.0f}ms is_complex={comp.is_complex}")

print("\n[done]")
