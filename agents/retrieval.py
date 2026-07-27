from __future__ import annotations
import re
import time
import logging
import config

logger = logging.getLogger(__name__)

_ANIME_TAGS = (
    '热血', '动作', '搞笑', '异世界', '奇幻', '科幻', '恋爱', '日常',
    '治愈', '悬疑', '推理', '战斗', '冒险', '校园', '机战', '运动',
    '魔法', '后宫', '百合', '耽美', '美食', '音乐', '竞技',
    '战争', '历史', '恐怖', '职场', '偶像', '转生', '游戏',
)
_SCORE_RANGE_RE = re.compile(r'(\d[\d.]*)\s*分?\s*(以上|以下|超过|高于|低于)')
_YEAR_RE = re.compile(r'(20\d{2})')

def retrieve_by_keywords(keywords, errors=None):
    results = []
    if not keywords:
        return results
    try:
        from agents.metadata_index import index
        for kw in keywords[:3]:
            md = index.search_by_name(kw)
            if md:
                results.extend(md)
        for kw in keywords[:3]:
            tag_kw = kw.strip('【】！! ')
            if len(tag_kw) >= 2:
                tag_hits = index.search(name=tag_kw, limit=5)
                known_ids = {str(r.get('id', '')) for r in results if r.get('id')}
                for r in tag_hits:
                    if str(r.get('id', '')) not in known_ids:
                        results.append(r)
                        known_ids.add(str(r.get('id', '')))
    except Exception as e:
        msg = f'关键词 Metadata 查询失败: {e}'
        logger.warning(msg)
        if errors is not None:
            errors.append({'source': 'metadata_keyword', 'type': type(e).__name__, 'message': str(e)})
    return results


def retrieve_metadata(query, plan, search_queries, existing, errors=None):
    results = list(existing)
    try:
        from agents.metadata_index import index
        filters = _extract_metadata_filters(query, plan)
        if filters:
            results = index.search(**filters)
        else:
            for q in search_queries[:2]:
                md = index.search_by_name(q)
                if not results:
                    results = md
            if plan.get('query_type') in ('recommendation', 'comparison'):
                matched_tags = [t for t in _ANIME_TAGS if t in query]
                if matched_tags:
                    tag_results = index.search(tag=matched_tags[0], limit=20)
                    seen_ids = {str(r.get('id', '')) for r in results}
                    for r in tag_results:
                        if str(r.get('id', '')) not in seen_ids:
                            results.append(r)
    except Exception as e:
        msg = f'Metadata Index 查询失败: {e}'
        logger.warning(msg)
        if errors is not None:
            errors.append({'source': 'metadata_filter', 'type': type(e).__name__, 'message': str(e)})
    return results


def retrieve_semantic(search_queries, state, errors=None):
    docs = []
    try:
        from tools.registry import tool_registry
        from tools.rag import get_retriever
        retrieve_opt = tool_registry.get_callable('retrieve_optimized')
        retriever = get_retriever()
        plan = state.get('plan', {})
        k_final = config.RETRIEVER_K
        if plan.get('query_type') == 'recommendation':
            k_final = max(k_final, 10)
        context = state.get('context', {})
        constraints = context.get('constraints', {}) if isinstance(context, dict) else {}
        query_count = 3 if constraints.get('exclude_same_series') else 2
        for q in search_queries[:query_count]:
            if retrieve_opt:
                d, _ = retrieve_opt(q, retriever, k_final=k_final, skip_optimization=True)
                docs.extend(d)
    except Exception as e:
        msg = f'语义检索失败: {e}'
        logger.warning(msg)
        if errors is not None:
            errors.append({'source': 'semantic', 'type': type(e).__name__, 'message': str(e)})
    return docs


def _extract_metadata_filters(query, plan):
    filters = {}
    matched_tags = [t for t in _ANIME_TAGS if t in query]
    if matched_tags and plan.get('query_type') in ('recommendation', 'simple_fact'):
        filters['tag'] = matched_tags[0]
    score_match = _SCORE_RANGE_RE.search(query)
    if score_match:
        val = float(score_match.group(1))
        if score_match.group(2) in ('以上', '超过', '高于'):
            filters['score_min'] = val
        else:
            filters['score_max'] = val
    year_match = _YEAR_RE.search(query)
    if year_match:
        year = year_match.group(1)
        if '之前' in query or '以前' in query:
            filters['date_to'] = year
        elif '之后' in query or '以后' in query:
            filters['date_from'] = year
        else:
            filters['date_from'] = year
            filters['date_to'] = str(int(year) + 1)
    return filters if filters else None

async def knowledge_retrieval_node(state):
    t0 = time.time()
    plan = state.get('plan', {})
    query_category = plan.get('query_category', 'mixed')
    query = state.get('resolved_query', '') or state.get('original_query', '')
    queries = state.get('shared_context', [query])
    if isinstance(queries, str):
        queries = [queries]
    search_queries = [q for q in queries if isinstance(q, str)]
    if not search_queries:
        search_queries = [query]
    context = state.get('context', {})
    constraints = context.get('constraints', {}) if isinstance(context, dict) else {}
    if constraints.get('exclude_same_series') and constraints.get('topic_tags'):
        tag_queries = [f'{tag} 动画 推荐' for tag in constraints['topic_tags'][:3]]
        search_queries = list(dict.fromkeys([*tag_queries, *search_queries]))
    retrieval_errors: list[dict] = []
    metadata_results = retrieve_by_keywords(state.get('search_keywords', []), errors=retrieval_errors)
    if query_category in ('metadata', 'mixed'):
        metadata_results = retrieve_metadata(query, plan, search_queries, metadata_results, errors=retrieval_errors)
    shared_context = []
    if query_category in ('semantic', 'mixed'):
        shared_context = retrieve_semantic(search_queries, state, errors=retrieval_errors)
    logger.info(
        '知识检索完成: metadata %d 条, shared_context %d 条 (耗时 %.1fs)',
        len(metadata_results[:30]), len(shared_context[:10]), time.time() - t0)
    experts = list(dict.fromkeys(plan.get('experts', [])))
    mode = 'single' if len(experts) <= 1 else ('parallel' if plan.get('parallel') else 'serial')
    return {
        'metadata': metadata_results[:30],
        'shared_context': shared_context[:10],
        'execution_mode': mode,
        'current_expert_index': 0,
        'retrieval_errors': retrieval_errors,
    }
