"""Query Processing Layer — 查询分类 + 优化（分离自 rag_optimizer）

模块:
  QueryClassifier   — 规则分类器
  QueryOptimizer    — Multi-Query Rewrite / HyDE / Decompose
  NicknameResolver  — 昵称/别名解析
"""
import hashlib
import functools
from typing import Literal

from langchain_core.messages import HumanMessage
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
# 1. QueryOptimizer — LLM 结构化分类（替代旧正则分类器）
# ══════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field

StrategyType = Literal["direct", "rewrite", "hyde", "decompose"]


class StrategyClassifyOutput(BaseModel):
    """查询优化策略分类"""
    strategy: StrategyType = Field(
        description=(
            "direct: 简单短查询（闲聊/问候/短关键词），不需重写; "
            "rewrite: 需从多角度扩展查询以提升召回; "
            "hyde: 深度分析/评价类（含'为什么''好在哪''区别''解析'等），先生成假设性答案再检索; "
            "decompose: 含多个独立子问题（含'分别''还有''和X有什么区别'等）"
        )
    )


_CLASSIFY_PROMPT = """你是查询优化策略分类器。判断如何优化给定的 ACG 番剧查询以提升检索效果。

策略说明:
- direct: 简单直接（闲聊/问候/简短关键词），直接检索
- rewrite: 适合多角度扩展，生成多个改写查询
- hyde: 深度分析/评价类，先生成假设性答案再检索
- decompose: 含多个子问题，拆分为独立查询分别检索

只输出策略名。"""


def classify(query: str) -> StrategyType:
    """LLM 结构化分类器（替代旧正则匹配）

    用轻量模型做策略判断，避免正则的误判和漏判。
    """
    try:
        from llms import simple_LLM, invoke_structured
        return invoke_structured(
            simple_LLM, StrategyClassifyOutput,
            [HumanMessage(content=f"{_CLASSIFY_PROMPT}\n\n用户查询: {query}")],
        ).strategy
    except Exception:
        # 降级: 短查询 direct，其他默认 rewrite
        q = query.strip()
        if len(q) <= 5:
            return "direct"
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
    from llms import answer_LLM, llm_invoke_with_retry
    from langchain_core.messages import HumanMessage
    resp = llm_invoke_with_retry(
        answer_LLM.bind(temperature=temperature),
        [HumanMessage(content=prompt)],
    )
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
