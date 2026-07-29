"""Smoke test：真实 LLM/Web 打点，验证 L1/L2/L3 各条路径。

Run: python scripts/smoke_alias.py
"""
from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 独立缓存文件，避免污染 data/alias_cache.json
import tempfile
import config
config.ALIAS_CACHE_PATH = os.path.join(tempfile.mkdtemp(), "alias_cache.json")

import agents.alias as alias_mod
alias_mod._LEARNED_CACHE.clear()
alias_mod._CACHE_LOADED = False


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


CASES = [
    # (query, use_web, 期望 source)
    ("素晴", False, "cache"),        # L1 seed
    ("re0", False, "cache"),          # L1 seed
    ("巨人", False, "cache"),        # L1 seed
    ("石头门", False, "cache"),      # L1 seed
    # L2 真实 LLM: 一个 seed 里没有但 LLM 应能识别的简称
    ("三体", False, None),
    ("赛马娘", False, None),
    # 冷门/编造，L2 大概率给 0，看 L3 是否触发
    ("xyz虚构缩写_test", True, None),
]

print(f"{'query':<20} {'use_web':<8} {'source':<8} {'conf':<6} {'latency':<10} {'full_name'}")
print("-" * 100)
for q, use_web, expected in CASES:
    result, dt = _time(alias_mod.resolve_alias_ex, q, use_web=use_web)
    print(f"{q:<20} {str(use_web):<8} {result['source']:<8} {result['confidence']:<6.2f} {dt*1000:>7.0f}ms   {result['full_name']}")
    if expected and result["source"] != expected:
        print(f"  ! 期望 source={expected}，实际 {result['source']}")

# 二次调用应全部命中 L1
print("\n[二次调用（命中 cache）]")
for q, use_web, _ in CASES:
    result, dt = _time(alias_mod.resolve_alias_ex, q, use_web=use_web)
    print(f"{q:<20} {result['source']:<8} {dt*1000:>7.2f}ms   {result['full_name']}")

print("\n[learned cache 内容]")
for k, v in alias_mod._LEARNED_CACHE.items():
    print(f"  {k:<25} -> {v['full_name']}  (conf={v['confidence']}, src={v['source']})")

print(f"\n[disk cache path] {config.ALIAS_CACHE_PATH}")
if os.path.exists(config.ALIAS_CACHE_PATH):
    print(f"[disk cache size] {os.path.getsize(config.ALIAS_CACHE_PATH)} bytes")
