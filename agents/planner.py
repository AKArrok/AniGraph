"""Planner Agent - 四层路由驱动图编排

架构:
  ① Embedding 粗筛    - 零 LLM 成本拦截闲聊 + 排除明显不相关类别
  ② LRU 缓存         - 同类查询直接命中
  ③ LLM 意图分类     - with_structured_output 保证 100% 可解析
  ④ 复杂度分析/策略细化 - 用小模型判断是否需要多查询扩展，按需用主力模型细化

职责:
  1. 意图分类: 判断查询类别（metadata/semantic/mixed/chat）和查询类型
  2. 决定查询优化策略（direct/rewrite/hyde/decompose）
  3. 决策需要哪些 Expert（metadata_reasoner/similar_expert）
  4. 决定 Expert 并行还是串行
  5. 决定是否需要联网

输入: 用户原始查询
输出: ExecutionPlan dict
"""
import hashlib
import time
import logging
from collections import OrderedDict
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 0. Pydantic 输出模型（保证结构化输出 100% 可解析）
# ══════════════════════════════════════════════════════════════════════

class IntentOutput(BaseModel):
    """第一阶段: 粗粒度意图分类（用小模型，低成本）"""
    query_category: Literal["metadata", "semantic", "mixed", "chat"] = Field(
        description=(
            "metadata: 查结构化元数据（评分/声优/导演等），数据库能回答; "
            "semantic: 需语义理解（推荐/评价/观后感），需向量检索; "
            "mixed: 指定番剧名+评价/推荐意图; "
            "chat: 闲聊/问候/非番剧问题"
        )
    )
    query_type: Literal["simple_fact", "recommendation", "comparison", "chat"] = Field(
        description="查询类型"
    )
    reasoning: str = Field(description="简短说明分类依据（15字以内）")


class ComplexityOutput(BaseModel):
    """复杂度分析: 用小模型判断是否需要多查询扩展（省去不必要的策略细化）"""
    is_complex: bool = Field(description="是否需要多查询扩展/重写/分解")
    suggested_strategy: Literal["direct", "rewrite", "hyde", "decompose"] = Field(
        description="建议的查询优化策略"
    )
    reasoning: str = Field(description="复杂度判断依据（15字以内）")


class StrategyOutput(BaseModel):
    """第二阶段: 细化执行策略（仅 mixed/复杂查询，用主力模型）"""
    rewrite_strategy: Literal["direct", "rewrite", "hyde", "decompose"] = Field(
        description="direct: 简单查询; rewrite: 多角度扩展; hyde: 深度分析; decompose: 多子问题"
    )
    experts: list[Literal["metadata_reasoner", "similar_expert"]] = Field(
        description="需要调用的 Expert"
    )
    parallel: bool = Field(description="Experts 是否并行执行")
    need_web: bool = Field(description="是否需要联网搜索")


# ── 第一阶段分类 Prompt ──

_INTENT_PROMPT = """你是 ACG 番剧查询分类器，快速判断查询的意图类别。

## 查询类别 (query_category)
- metadata: 查结构化元数据（评分/声优/导演/公司/标签/年份等），数据库能回答
  例: "进击的巨人评分"、"MAPPA作品"、"2024热血番"
- semantic: 开放性问题需语义理解（评价/口碑/观后感/推荐相似作品），需向量检索
  例: "有没有类似钢炼的番"、"为什么EVA是神作"、"催泪番推荐"
- mixed: 指定番剧名 + 评价/推荐意图，需 Metadata 查信息 + Semantic 做推荐
  例: "碧蓝之海怎么样？"、"进击的巨人好看吗"、"RE:0值得看吗"
- chat: 闲聊/问候/非番剧问题
  例: "你好"、"谢谢"、"你是谁"、"你能做什么"

## 查询类型 (query_type)
- simple_fact: 查已知事物的具体属性
- recommendation: 要求推荐/发现新番剧
- comparison: 对比多部番剧
- chat: 闲聊/问候

{history_section}"""


# ── 第二阶段策略细化 Prompt ──

_COMPLEXITY_PROMPT = """你是查询复杂度分析器，判断用户查询是否需要多查询扩展。

## 简单查询 (is_complex=false, direct)
- 查单一已知事实: "巨人评分多少"、"MAPPA有哪些作品"
- 简单闲聊: "你好"、"谢谢"
- 明确指代已知番剧的单项查询

## 复杂查询 (is_complex=true)
- 需要多角度扩展: "类似巨人的番"（需从标签/风格/评分多路检索）
- 深度分析: "为什么EVA是神作"、"巨人好在哪"（需 hyde 生成假设文档）
- 多子问题: "推荐2024热血番并说明理由"（需分解+合成）
- 对比类: "巨人和鬼灭哪个好看"（需分别检索再对比）

{suggested_strategy_section}

{history_section}"""

_SUGGESTED_STRATEGY_SECTION = """## 建议策略 (suggested_strategy)
- direct: 简单查询无需重写
- rewrite: 需多角度扩展（标签/风格/评分等维度）
- hyde: 深度分析/评价类，需生成假设文档再检索
- decompose: 含多个子问题需分解执行"""

_STRATEGY_PROMPT = """你是 ACG 番剧推荐系统的规划器。用户查询已初步分类，请细化执行策略。

## 查询优化策略 (rewrite_strategy)
- direct: 简单查询，不需重写
- rewrite: 需从多角度扩展查询
- hyde: 深度分析/评价类（含"为什么""好在哪""区别"等）
- decompose: 含多个子问题

## 专家选择 (experts)
- metadata_reasoner: 涉及评分/标签/公司/声优等结构化数据
- similar_expert: 涉及相似推荐/对比/语义理解

## 并行 (parallel)
- 两个 expert 都需要 -> true，只需一个 -> false

## 联网 (need_web)
- 查询可能超出知识库范围（冷门番剧/最新资讯）-> true

{history_section}

初步分类: {classification}"""


# ══════════════════════════════════════════════════════════════════════
# ① Embedding 预检 - 零 LLM 成本拦截闲聊/问候
# ══════════════════════════════════════════════════════════════════════

# 预计算各类别样例 embedding 质心（只算一次）
# 覆盖全部 4 个意图类别: chat, metadata, semantic, mixed
_ROUTE_CENTROIDS: dict[str, np.ndarray] = {}
_centroids_initialized = False

_CATEGORY_EXAMPLES: dict[str, list[str]] = {
    "chat": [
        "你好", "谢谢", "再见", "你是谁", "你能做什么",
        "hello", "hi", "有人在吗", "早上好", "晚上好",
        "help", "帮帮我", "怎么用", "开始", "退出",
    ],
    "metadata": [
        "进击的巨人评分是多少", "鬼灭之刃的声优是谁",
        "MAPPA制作了哪些番剧", "2024年有哪些热血番",
        "钢之炼金术师有几集", "巨人讲的是什么",
        "咒术回战豆瓣评分", "ufotable作品列表",
        "2023年10月新番", "有哪些异世界番",
    ],
    "semantic": [
        "有没有类似钢炼的番", "为什么EVA是神作",
        "催泪番推荐", "最好看的校园番",
        "有哪些悬疑推理番", "治愈系动漫推荐",
        "跟鬼灭一样热血的番", "冷门但好看的动漫",
        "画风好看的番推荐", "剧情炸裂的动漫",
    ],
    "mixed": [
        "碧蓝之海怎么样", "进击的巨人好看吗",
        "RE:0值得看吗", "无职转生评价如何",
        "芙莉莲好看吗", "迷宫饭推荐吗",
        "孤独摇滚值得看吗", "86不存在的战区好看吗",
    ],
}


def _init_centroids():
    """预计算 embedding 质心（首次调用时触发）"""
    global _centroids_initialized, _ROUTE_CENTROIDS
    if _centroids_initialized:
        return
    try:
        from llms import embeddings
        for category, examples in _CATEGORY_EXAMPLES.items():
            _ROUTE_CENTROIDS[category] = np.mean(
                embeddings.embed_documents(examples), axis=0
            )
        _centroids_initialized = True
        logger.info(f"  Embedding 预检质心初始化完成 ({len(_ROUTE_CENTROIDS)} 个类别)")
    except Exception as e:
        logger.warning(f"Embedding 预检初始化失败: {e}，降级到 LLM 分类")
        _centroids_initialized = True  # 避免重复尝试


def _prefilter(query: str) -> tuple[str | None, float, dict[str, float]]:
    """Embedding 预检: 返回 (最佳类别, 最高分, 各类别得分排行)

    返回值:
      - best_category: 余弦相似度最高的类别名，置信度不足返回 None
      - best_score: 最佳匹配的相似度
      - all_scores: 所有类别的 {category: score}，用于排除明显不相关类别

    高置信度 -> 直接跳过 LLM 意图分类；低置信度 -> 走正常流程。

    带 LRU 缓存: 同一 query 在一次请求中被多次调用（alias_skip + planner）
    只算一次 embedding，结果共享。
    """
    # 缓存命中检查
    if query in _prefilter_cache:
        return _prefilter_cache[query]

    if not config.ENABLE_EMBEDDING_PREFILTER:
        return None, 0.0, {}

    _init_centroids()
    if not _ROUTE_CENTROIDS:
        return None, 0.0, {}

    try:
        from llms import embeddings
        query_vec = np.array(embeddings.embed_query(query))
        query_norm = np.linalg.norm(query_vec)

        all_scores: dict[str, float] = {}
        best_category = None
        best_score = 0.0

        for category, centroid in _ROUTE_CENTROIDS.items():
            sim = float(np.dot(query_vec, centroid) / (
                query_norm * np.linalg.norm(centroid)
            ))
            all_scores[category] = sim
            if sim > best_score:
                best_score = sim
                best_category = category

        # 按得分降序排列
        all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        if best_score >= config.EMBEDDING_PREFILTER_THRESHOLD:
            result = (best_category, best_score, all_scores)
        else:
            result = (None, best_score, all_scores)

        # 写入缓存（限制大小避免内存增长）
        if len(_prefilter_cache) < 200:
            _prefilter_cache[query] = result
        return result
    except Exception:
        return None, 0.0, {}


# 请求级 LRU 缓存: alias_skip 和 planner 共享同一 query 的预检结果
_prefilter_cache: dict[str, tuple] = {}


# ══════════════════════════════════════════════════════════════════════
# ② LLM 结构化分类 - 两阶段: 小模型意图 -> 主力模型策略
# ══════════════════════════════════════════════════════════════════════

def _classify_intent(query: str, history_text: str = "",
                     excluded_categories: list[str] | None = None) -> IntentOutput:
    """第一阶段: 用轻量模型做意图分类（结构化输出保证格式）
    
    excluded_categories: embedding 粗筛排除的类别，缩小 LLM 决策空间。
    """
    from llms import simple_LLM, invoke_structured

    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史（仅供参考，用于理解指代和上下文）\n"
            f"{history_text}\n\n"
            f"注意: 需独立判断当前查询的意图，不受历史类型影响。"
        )

    excluded_hint = ""
    if excluded_categories:
        excluded_hint = (
            f"\n\n## Embedding 粗筛提示\n"
            f"以下类别与该查询明显不相关，请勿选择: {', '.join(excluded_categories)}"
        )

    system_prompt = _INTENT_PROMPT.format(history_section=history_section) + excluded_hint
    return invoke_structured(simple_LLM, IntentOutput, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {query}"),
    ])


def _refine_strategy(query: str, intent: IntentOutput,
                     history_text: str = "") -> StrategyOutput:
    """第二阶段: 用主力模型细化执行策略"""
    from llms import answer_LLM, invoke_structured

    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史\n{history_text}\n\n"
            f"注意: 独立判断当前查询意图。"
        )

    classification = intent.model_dump_json()
    system_prompt = _STRATEGY_PROMPT.format(
        history_section=history_section,
        classification=classification,
    )

    return invoke_structured(answer_LLM, StrategyOutput, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {query}"),
    ])


def _analyze_complexity(query: str, intent: IntentOutput,
                        history_text: str = "") -> ComplexityOutput:
    """用小模型判断查询复杂度，决定是否需要多查询扩展。

    简单查询 -> direct 策略，跳过昂贵的策略细化；
    复杂查询 -> 进入第二阶段策略细化（rewrite/hyde/decompose）。
    """
    from llms import simple_LLM, invoke_structured

    history_section = ""
    if history_text:
        history_section = (
            f"## 对话历史\n{history_text}\n\n"
            f"注意: 独立判断当前查询复杂度。"
        )

    # 简单查询不需要展示策略建议，减少 prompt 噪声
    suggested_section = _SUGGESTED_STRATEGY_SECTION if not _is_trivially_simple(intent) else ""

    system_prompt = _COMPLEXITY_PROMPT.format(
        suggested_strategy_section=suggested_section,
        history_section=history_section,
    )

    return invoke_structured(simple_LLM, ComplexityOutput, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {query}\n意图分类: {intent.model_dump_json()}"),
    ])


def _is_trivially_simple(intent: IntentOutput) -> bool:
    """判断是否明显为简单查询，可跳过复杂度分析"""
    return (
        intent.query_category == "chat"
        or (intent.query_category == "metadata" and intent.query_type == "simple_fact")
    )


def _intent_to_plan(intent: IntentOutput,
                    strategy: StrategyOutput | None = None,
                    complexity: ComplexityOutput | None = None) -> dict:
    """将结构化分类结果组装为 ExecutionPlan dict"""
    if intent.query_category == "chat":
        return {
            "query_category": "chat",
            "query_type": "chat",
            "rewrite_strategy": "direct",
            "experts": [],
            "parallel": False,
            "need_web": False,
            "alias_resolved": False,
            "reasoning": intent.reasoning,
        }

    # 有复杂度分析且判断为简单 -> 直接用复杂度建议的策略，跳过昂贵的策略细化
    if complexity is not None and not complexity.is_complex:
        experts = (
            ["metadata_reasoner"] if intent.query_category == "metadata"
            else ["similar_expert"] if intent.query_category == "semantic"
            else ["metadata_reasoner", "similar_expert"]
        )
        return {
            "query_category": intent.query_category,
            "query_type": intent.query_type,
            "rewrite_strategy": complexity.suggested_strategy,
            "experts": experts,
            "parallel": len(experts) > 1,
            "need_web": False,
            "alias_resolved": False,
            "reasoning": f"{intent.reasoning} | {complexity.reasoning}",
        }

    if strategy is None:
        # 简单路径: metadata 或 semantic，不需要策略细化
        experts = (
            ["metadata_reasoner"] if intent.query_category == "metadata"
            else ["similar_expert"]
        )
        rewrite = "direct" if intent.query_type == "simple_fact" else "rewrite"
        return {
            "query_category": intent.query_category,
            "query_type": intent.query_type,
            "rewrite_strategy": rewrite,
            "experts": experts,
            "parallel": False,
            "need_web": False,
            "alias_resolved": False,
            "reasoning": intent.reasoning,
        }

    # mixed / 复杂查询: 有策略细化结果
    return {
        "query_category": intent.query_category,
        "query_type": intent.query_type,
        "rewrite_strategy": strategy.rewrite_strategy,
        "experts": strategy.experts,
        "parallel": strategy.parallel,
        "need_web": strategy.need_web,
        "alias_resolved": False,
        "reasoning": intent.reasoning,
    }


# ══════════════════════════════════════════════════════════════════════
# ③ LRU 缓存 - 相同查询直接命中（OrderedDict 实现真 LRU）
# ══════════════════════════════════════════════════════════════════════

def _hash_query(query: str, history_text: str = "") -> str:
    """生成查询缓存键: query + history_text（避免追问场景下误命中）

    同一查询在不同历史上下文中可能需要不同分类策略，所以 history_text 也
    参与缓存键。
    """
    return hashlib.md5(f"{query}|{history_text}".encode()).hexdigest()


# OrderedDict: 访问/写入时 move_to_end，满了 popitem(last=False) 淘汰最久未访问
_plan_cache: "OrderedDict[str, dict]" = OrderedDict()
_plan_cache_max = 500


def _get_cached_plan(query: str, history_text: str = "") -> dict | None:
    """检查计划缓存（命中时移到末尾，实现 LRU）"""
    key = _hash_query(query, history_text)
    plan = _plan_cache.get(key)
    if plan is not None:
        _plan_cache.move_to_end(key)
    return plan


def _set_cached_plan(query: str, plan: dict, history_text: str = ""):
    """写入计划缓存"""
    key = _hash_query(query, history_text)
    _plan_cache[key] = plan
    _plan_cache.move_to_end(key)
    if len(_plan_cache) > _plan_cache_max:
        _plan_cache.popitem(last=False)  # 淘汰最久未访问


# ══════════════════════════════════════════════════════════════════════
# 主入口: plan() - 四层路由编排
# ══════════════════════════════════════════════════════════════════════

def _get_excluded_categories(all_scores: dict[str, float],
                              margin: float | None = None) -> list[str]:
    """根据 embedding 得分排行，找出与最佳匹配差距较大的类别（明显不相关）。"""
    if not all_scores:
        return []
    if margin is None:
        margin = config.EMBEDDING_EXCLUDE_MARGIN
    max_score = max(all_scores.values())
    excluded = [
        cat for cat, score in all_scores.items()
        if max_score - score > margin
    ]
    return excluded


def _route_embedding(query: str) -> tuple[dict | None, list[str]]:
    """层1: Embedding 粗筛 - 拦截闲聊 + 排除明显不相关类别

    返回 (plan_or_none, excluded_categories)
    - plan_or_none: 高置信度 chat 直接返回 plan，否则 None
    - excluded_categories: 明显不相关的类别列表（缩小 LLM 决策空间）
    """
    route, confidence, all_scores = _prefilter(query)
    if route == "chat" and confidence >= config.EMBEDDING_PREFILTER_THRESHOLD:
        logger.info(f"  [预检拦截] chat (confidence={confidence:.2f}) - 零 LLM")
        return _intent_to_plan(IntentOutput(
            query_category="chat", query_type="chat",
            reasoning=f"embedding预检 chat={confidence:.2f}"
        )), []

    excluded = _get_excluded_categories(all_scores)
    if excluded:
        logger.info(f"  [粗筛排除] 不相关类别: {excluded}")
    return None, excluded


def _route_complexity(query: str, intent: IntentOutput,
                       history_text: str) -> dict:
    """层4: metadata/semantic 路径 - 复杂度分析决定是否策略细化

    简单查询 -> direct 策略，跳过昂贵的策略细化;
    复杂查询 -> 主力模型细化执行策略。
    """
    if _is_trivially_simple(intent):
        return _intent_to_plan(intent)

    # 复杂度分析开关关闭时走原始简单路径
    if not config.ENABLE_COMPLEXITY_CHECK:
        return _intent_to_plan(intent)

    complexity = _analyze_complexity(query, intent, history_text)
    if not complexity.is_complex:
        logger.info(
            f"  [简单查询] 跳过策略细化 "
            f"(strategy={complexity.suggested_strategy}, "
            f"reason={complexity.reasoning})"
        )
        return _intent_to_plan(intent, complexity=complexity)

    # 复杂查询 -> 需要主力模型做策略细化
    logger.info(f"  [策略细化] 复杂查询需主力模型")
    strategy = _refine_strategy(query, intent, history_text)
    return _intent_to_plan(intent, strategy)


def plan(query: str, history_text: str = "") -> dict:
    """四层路由: Embedding 粗筛 -> 缓存 -> LLM 分类 -> 复杂度分析/策略细化

    优化:
      1. Embedding 预检: 拦截闲聊 + 排除明显不相关类别缩小 LLM 决策空间
      2. 复杂度分析: 用小模型判断是否需要多查询扩展，省去不必要的策略细化
    """
    # 层1: Embedding 粗筛（拦截闲聊 + 排除不相关类别）
    early_plan, excluded = _route_embedding(query)
    if early_plan is not None:
        return early_plan

    # 层2: 缓存检查
    cached = _get_cached_plan(query, history_text)
    if cached:
        logger.info(f"  [缓存命中] - 跳过 LLM 分类")
        return cached

    # 层3: LLM 意图分类
    logger.info(f"  [LLM 意图分类] query={query[:50]}...")
    intent = _classify_intent(query, history_text, excluded_categories=excluded)

    if intent.query_category == "chat":
        result = _intent_to_plan(intent)
        _set_cached_plan(query, result, history_text)
        return result

    # 层4: 复杂度分析 + 策略细化
    if intent.query_category in ("metadata", "semantic"):
        result = _route_complexity(query, intent, history_text)
        _set_cached_plan(query, result, history_text)
        return result

    # mixed: 本身涉及多维度，始终需要策略细化
    logger.info(f"  [策略细化] mixed 查询需主力模型")
    strategy = _refine_strategy(query, intent, history_text)
    result = _intent_to_plan(intent, strategy)
    _set_cached_plan(query, result, history_text)
    return result


async def planner_node(state: dict) -> dict:
    """LangGraph 节点: Planner"""
    t0 = time.time()
    query = state.get("resolved_query", "") or state.get("original_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    # 构建对话历史文本
    context = state.get("context", {})
    history_text = context.get("history_text", "")

    execution_plan = plan(query, history_text=history_text)

    if not isinstance(execution_plan, dict):
        execution_plan = dict(execution_plan)

    # 根据实体解析结果调整 plan（基于数据，非正则）
    entity_confidence = state.get("entity_confidence", 1.0)
    entity_type = state.get("entity_type", "")

    if entity_confidence < 0.5:
        execution_plan["need_web"] = True

    if entity_type == "meme":
        execution_plan["need_web"] = True

    # 标记别名解析状态（alias_resolve 是否实际运行过）
    if state.get("entity_source"):
        execution_plan["alias_resolved"] = True

    logger.info(
        f"  planner 耗时 {time.time()-t0:.1f}s "
        f"(category={execution_plan.get('query_category')}, "
        f"type={execution_plan.get('query_type')}, "
        f"experts={execution_plan.get('experts')}, "
        f"alias={execution_plan.get('alias_resolved')}, "
        f"web={execution_plan.get('need_web')})"
    )

    return {"plan": execution_plan, "original_query": query}
