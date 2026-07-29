"""Query Processing Layer - 查询分类 + 优化（分离自 rag_optimizer）

模块:
  classify          - LLM 结构化策略分类（direct/rewrite/hyde/decompose）
  QueryOptimizer    - Multi-Query Rewrite / HyDE / Decompose
  NicknameResolver  - 昵称/别名解析
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


_CLASSIFY_PROMPT = """判断这条 ACG 番剧查询应该走哪种检索优化策略：

- direct: 短查询/闲聊/问候/单一关键词，无需扩展，原样检索。
- rewrite: 常规推荐/查找类，从多角度改写扩展召回。
- hyde: 含"为什么/好在哪/怎么样/如何评价/区别/解析"等评价或解释诉求，先生成假设文档再检索。
- decompose: 用"并且/分别/还有/和 X 有什么区别"等把多个子问题串在一起。

仅按上述含义判断，别关心输出格式。"""


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

_MULTI_QUERY_PROMPT = """把这条 ACG 番剧查询扩展成 3 条不同角度的中文查询语句，用于向量检索（不给用户看）。

原始查询：「{query}」

要求：
- 保留原始查询的核心意图，不偏题。
- 三条各选一个不同角度切入：剧情/风格标签、观众评价与口碑、制作团队与制作背景、类似作品的联想等。
- 每条用完整的自然中文句子，不要关键词堆砌，不要英文。
- 每行只写一条查询，不加编号、不加破折号、不加引号。"""

_HYDE_PROMPT = """写一段 200-300 字的 ACG 番剧推荐/评价段落，用来喂给向量检索（不会展示给用户）。文本要模仿真实语料的语气：像论坛长评、推荐帖、播出短评那种。

要求：
- 段落自然，不要标题、不要列表、不要 Markdown。
- 如果对作品名有把握可以直接写出，没把握就用"这类作品/这种类型"泛指，别硬编评分和年份。
- 语言口径贴近实际观众会用的表达（剧情钩子、制作观感、观众反应等）。

问题: {query}

假设性回答:"""

_DECOMPOSE_PROMPT = """把这条 ACG 番剧查询拆成若干个可独立检索的子问题，每行一条，别加编号或标点。
如果查询本身不含独立子问题，就原样输出这一条查询，别硬拆。

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
