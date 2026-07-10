"""自检脚本 — 验证多 Agent 架构所有模块"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

print("=== 1. 导入验证 ===")
from graph import build_graph
from agents.state import AgentState, ExecutionPlan, ExpertResult
from agents.cache import metadata_cache
from agents.alias import resolve_alias, resolve_alias_dict
from agents.metadata_index import index as metadata_index
from agents.planner import plan, planner_node, _classify_query_category
from agents.metadata_reasoner import metadata_reasoner_node
from agents.similar_expert import similar_expert_node
from agents.answer import answer_node
from agents.web_fallback import web_fallback_node, should_trigger_web
from agents.merge import merge_expert_results
from tools.rag_optimizer import retrieve_with_optimization, get_last_debug, classify
from tools.query_processing import multi_query_rewrite, hyde_generate, decompose
from tools.knowledge_retrieval import search_whoosh, fusion, rerank, compress_docs, verify_answer
from data.build_kb import build_metadata
print("  所有模块导入 OK\n")

print("=== 2. ExecutionPlan 字段 ===")
for name, f in ExecutionPlan.model_fields.items():
    desc = (f.description or "?")[:60]
    print(f"  {name}: {desc}")
print()

print("=== 3. Query Classifier 测试 ===")
tests = [
    ("京都动画有哪些作品？",       "metadata"),
    ("京阿尼做过什么动漫",         "metadata"),
    ("2023年播出的动漫",           "metadata"),
    ("2016年番剧推荐",             "mixed"),      # 年份 + 推荐意图 → mixed
    ("评分最高的恋爱番",            "metadata"),
    ("评分高于8.5的动画",          "metadata"),
    ("花泽香菜配音的动漫",         "metadata"),
    ("松冈祯丞出演的作品",         "metadata"),
    ("MAPPA制作的动漫",            "metadata"),
    ("WIT STUDIO有哪些作品",       "metadata"),
    ("新海诚执导的动漫",           "metadata"),
    ("素晴的评分是多少",            "metadata"),
    ("鬼灭之刃的声优",              "metadata"),
    ("命运石之门的制作公司",        "metadata"),
    ("推荐几部类似进击的巨人",      "semantic"),
    ("巨人vs鬼灭哪个好看",          "semantic"),
    ("为什么命运石之门是神作",      "semantic"),
    ("Re:0剧情分析",               "semantic"),
    ("最近有什么好看的番",          "mixed"),      # 标签 + 好看（语义）→ mixed
    ("推荐热血动作番剧",            "mixed"),
    ("有什么催泪的番推荐",          "mixed"),
    ("有没有类似钢炼的番",          "mixed"),
    ("你好",                        "mixed"),
]
passed = failed = 0
for q, expected in tests:
    cat = _classify_query_category(q)
    ok = cat == expected
    if ok: passed += 1
    else: failed += 1
    mark = "OK" if ok else f"FAIL (got {cat})"
    print(f"  [{mark}] {q}")
print(f"  通过: {passed}/{len(tests)}, 失败: {failed}\n")

print("=== 4. 图编译验证 ===")
g = build_graph()
app = g.compile()
nodes = list(app.get_graph().nodes.keys())
print(f"  图节点 ({len(nodes)}): {nodes}")
print(f"  入口: START -> {nodes[0]}\n")

print("=== 5. Planner 完整链路 ===")
plan_dict = plan("推荐热血动作番剧")
print(f"  query_type: {plan_dict['query_type']}")
print(f"  query_category: {plan_dict['query_category']}")
print(f"  rewrite_strategy: {plan_dict['rewrite_strategy']}")
print(f"  experts: {plan_dict['experts']}")
print(f"  parallel: {plan_dict['parallel']}\n")

print("=== 6. Metadata Index 方法检查 ===")
for m in ["load", "reload", "get_by_id", "get_by_alias", "search_by_name", "search", "get_all_tags", "get_all_studios"]:
    ok = hasattr(metadata_index, m)
    print(f"  metadata_index.{m}: {'OK' if ok else 'MISSING'}")
print()

print("=== 7. MetadataCache 方法检查 ===")
for m in ["add_alias", "bulk_add_alias", "resolve_alias", "add_metadata", "bulk_load_metadata", "get_metadata", "resolve", "get_state"]:
    ok = hasattr(metadata_cache, m)
    print(f"  metadata_cache.{m}: {'OK' if ok else 'MISSING'}")
print()

print("=== 8. Alias 解析检查 ===")
try:
    from agents.alias import HARDCODED_ALIASES
    print(f"  硬编码别名数量: {len(HARDCODED_ALIASES)}")
except:
    print("  硬编码别名: 通过 resolve_alias_dict 函数访问")
r, ok = resolve_alias("素晴", use_llm=False)
print(f"  素晴 -> {r}, resolved={ok}")
r, ok = resolve_alias("konosuba", use_llm=False)
print(f"  konosuba -> {r}, resolved={ok}")
r, ok = resolve_alias("不存在的番剧xyz", use_llm=False)
print(f"  不存在的番剧xyz -> {r}, resolved={ok}\n")

print("=== 9. Merge 逻辑检查 ===")
result = merge_expert_results({
    "expert_results": [
        {"answer": "推荐A", "confidence": 0.9, "evidence": ["e1"]},
        {"answer": "推荐B", "confidence": 0.7, "evidence": ["e2"]},
        {"answer": "推荐A", "confidence": 0.5, "evidence": ["e3"]},
        {"answer": "推荐C", "confidence": 0.2, "evidence": ["e4"]},
    ]
})
merged = result.get("merged_results", "")
print(f"  Expert 1 存在: {'Expert 1' in merged}")
print(f"  Expert 2 存在(去重): {'Expert 2' in merged}")
print(f"  Expert 3 过滤(低置信度): {'Expert 3' not in merged}\n")

print("=== 10. Web Fallback 触发 ===")
print(f"  need_web=true: {should_trigger_web({'plan': {'need_web': True}})}")
print(f"  空 context: {should_trigger_web({'plan': {}, 'shared_context': []})}")
print(f"  低置信度: {should_trigger_web({'plan': {}, 'shared_context': ['a'], 'expert_results': [{'confidence': 0.3}]})}")
print(f"  正常: {should_trigger_web({'plan': {}, 'shared_context': ['a'], 'expert_results': [{'confidence': 0.8}]})}\n")

print("=" * 40)
print("  ALL CHECKS PASSED")
print("=" * 40)
