"""端到端跑 alias_resolve_node，看 state 输出是否符合预期。"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
config.ALIAS_CACHE_PATH = os.path.join(tempfile.mkdtemp(), "alias_cache.json")

import agents.alias as alias_mod
alias_mod._LEARNED_CACHE.clear()
alias_mod._CACHE_LOADED = False

from langchain_core.messages import HumanMessage
from agents.alias_resolve import alias_resolve_node


async def _run(query: str) -> tuple[dict, float]:
    state = {"messages": [HumanMessage(content=query)]}
    t0 = time.perf_counter()
    out = await alias_resolve_node(state)
    return out, time.perf_counter() - t0


CASES = [
    "素晴怎么样",                # L1 seed 命中（包含匹配）
    "巨人评分多少",              # L1 seed 命中
    "石头门声优",                # L1 seed 命中
    "赛马娘好看吗",              # L2 LLM 学习（新词，之前 seed 没有）
    "钢炼几集",                  # L1 seed 没有钢炼，得看 fuzzy/LLM
    "间谍过家家评分",            # L1 seed "间谍" 包含匹配
    "有没有类似钢炼的番",        # 长句：alias 不应劫持
    "你好",                      # 闲聊：alias 应无命中
]

print(f"{'query':<28} {'entity_source':<20} {'entity_name':<32} {'latency':<8}")
print("-" * 100)
for q in CASES:
    out, dt = asyncio.run(_run(q))
    src = out.get("entity_source", "-")
    name = out.get("entity_name") or out.get("resolved_query") or "-"
    print(f"{q:<28} {src:<20} {str(name)[:30]:<32} {dt*1000:>6.0f}ms")

print("\n[learned cache]")
for k, v in alias_mod._LEARNED_CACHE.items():
    print(f"  {k:<25} -> {v['full_name']}  (conf={v['confidence']}, src={v['source']})")
