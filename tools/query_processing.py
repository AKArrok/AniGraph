"""Query Processing Layer — 查询分类 + 优化（分离自 rag_optimizer）

模块:
  QueryClassifier   — 规则分类器
  QueryOptimizer    — Multi-Query Rewrite / HyDE / Decompose
  NicknameResolver  — 昵称/别名解析
"""
import re
import hashlib
import functools
from typing import Literal

import config

# ── 缓存 ────────────────────────────────────────────────────────────
_cache: dict[str, list[str]] = {}
MAX_CACHE = 500


def _cache_key(query: str, prefix: str) -> str:
    return prefix + hashlib.md5(query.encode()).hexdigest()


def _cache_get(query: str, prefix: str) -> list[str] | None:
    return _cache.get(_cache_key(query, prefix))


def _cache_set(query: str, prefix: str, value: list[str]):
    if len(_cache) >= MAX_CACHE:
        _cache.pop(next(iter(_cache)))
    _cache[_cache_key(query, prefix)] = value


# ══════════════════════════════════════════════════════════════════════
# 1. QueryClassifier — 规则分类器
# ══════════════════════════════════════════════════════════════════════

_HYDE_KEYWORDS = [
    "为什么", "比.*更好", "比.*强", "深度分析", "评价", "解读",
    "好在哪", "区别", "有何不同", "解析", "到底", "怎么样才算",
]

_DECOMPOSE_MARKERS = ["分别", "还有.*问题", "另外.*也", "并且.*还", "和.*有什么区别"]

_DIRECT_PATTERNS = [
    r"^(你好|hi|hello|谢谢|再见|bye)[\s!！。.]*$",
    r"^[\u4e00-\u9fff]{1,5}$",
    r"什么是RAG|什么是AI",
]

StrategyType = Literal["direct", "rewrite", "hyde", "decompose"]


def classify(query: str) -> StrategyType:
    """规则分类器，零 LLM 调用"""
    q = query.strip()

    for pat in _DIRECT_PATTERNS:
        if re.match(pat, q, re.I):
            return "direct"

    if len(q) <= 5 and not any(kw in q for kw in ["推荐", "评分", "番剧", "动画", "动漫"]):
        return "direct"

    for m in _DECOMPOSE_MARKERS:
        if re.search(m, q):
            return "decompose"

    for kw in _HYDE_KEYWORDS:
        if re.search(kw, q):
            return "hyde"

    return "rewrite"


# ══════════════════════════════════════════════════════════════════════
# 2. QueryOptimizer — 查询优化
# ══════════════════════════════════════════════════════════════════════

_MULTI_QUERY_PROMPT = """用户想找 ACG 番剧，原始查询是：「{query}」

请生成 3 条不同的中文自然语言查询，用于向量检索。要求：
- 用完整的自然语句，不要用关键词堆砌
- 每条从不同角度切入（如剧情风格、观众评价、制作团队、相似作品等）
- 保留原始查询的核心意图

输出格式：每行一条查询"""

_HYDE_PROMPT = """你是一个 ACG 番剧专家。请针对以下问题，写一段假设性的回答（200-300字），包含具体的番剧名、评分和理由。用这个回答的语气和内容风格。

问题: {query}

假设性回答:"""

_DECOMPOSE_PROMPT = """将以下复杂的 ACG 番剧查询拆分为独立子问题，每行一个，只输出子问题:

查询: {query}

子问题:"""


@functools.lru_cache(maxsize=256)
def _call_llm(prompt: str, temperature: float = 0.7) -> str:
    """轻量 LLM 调用"""
    from llms import answer_LLM
    from langchain_core.messages import HumanMessage
    resp = answer_LLM.bind(temperature=temperature).invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def multi_query_rewrite(query: str) -> list[str]:
    """Multi-Query Rewrite: 生成 3 个不同视角的查询 + 原始查询"""
    cached = _cache_get(query, "rewrite")
    if cached:
        return [query] + cached

    try:
        prompt = _MULTI_QUERY_PROMPT.format(query=query)
        text = _call_llm(prompt, temperature=0.7)
        rewrites = [line.strip("- 1234567890. ") for line in text.split("\n") if line.strip()][:3]
        _cache_set(query, "rewrite", rewrites)
        return [query] + rewrites
    except Exception:
        return [query]


def hyde_generate(query: str) -> list[str]:
    """HyDE: 先生成假设性答案，用答案去检索"""
    cached = _cache_get(query, "hyde")
    if cached:
        return cached

    try:
        prompt = _HYDE_PROMPT.format(query=query)
        text = _call_llm(prompt, temperature=0.8)
        _cache_set(query, "hyde", [text])
        return [text]
    except Exception:
        return [query]


def decompose(query: str) -> list[str]:
    """查询拆分"""
    cached = _cache_get(query, "decompose")
    if cached:
        return cached

    try:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        text = _call_llm(prompt, temperature=0.5)
        subs = [line.strip("- 1234567890. ") for line in text.split("\n") if line.strip()][:5]
        if subs:
            _cache_set(query, "decompose", subs)
            return subs
    except Exception:
        pass
    return [query]
