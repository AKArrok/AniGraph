# 性能优化方案：降低回答延迟（二次改进版）

> **状态**: ✅ 大部分已实施 — Planner 规则优先、消除双重查询、Answer Router、并行检索、simple_fact 快速通道、LLM 超时+耗时日志。异步 LLM (.ainvoke) 待实施。

## 摘要

当前系统端到端延迟约 **12-18s**（推荐类查询）。核心原则：**规则优先，LLM 按需调用** — 不是让每一步更快，而是减少没有必要执行的步骤。预计延迟降至 **5-8s**（减少 50-60%）。

***

## 当前状态分析

### 完整调用链 & 延迟分布

| 步骤 | 节点                   | LLM 调用             | 模型             | 延迟         | 问题                                        |
| -- | -------------------- | ------------------ | -------------- | ---------- | ----------------------------------------- |
| 1  | alias\_resolve       | 0-2 次              | qwen-flash     | 0-2s       | 条件触发，可接受                                  |
| 2  | planner              | **1 次**            | qwen-max       | 1-3s       | 所有查询都调 LLM，metadata/chat 结果被规则覆盖          |
| 3  | query\_processing    | 0-1 次              | qwen-max       | 0-3s       | 与 rag\_optimizer 二次改写，重复浪费                |
| 4  | knowledge\_retrieval | 0 次                | -              | 0.5-2s     | 乘法膨胀（N queries × M rewrites）+ 串行 Pinecone |
| 5  | experts (2x)         | 2 次                | qwen-max       | 3-5s       | 并行 Send 但 sync invoke 阻塞                  |
| 6  | web\_fallback        | 0-1 次              | qwen-flash     | 0-4s       | 条件触发，可接受                                  |
| 7  | answer               | **1 次**            | qwen-max T=0.9 | 3-5s       | 简单问题也用大模型 + 温度过高                          |
| -  | **合计**               | **4-6 次 qwen-max** | -              | **12-18s** | -                                         |

***

## 优化方案

### P0-1：消除双重查询优化（收益 \~3s，中工作量）

**问题**：`_query_processing_node`（graph.py:114）调用 `multi_query_rewrite()` 生成改写查询 → 传入 `_knowledge_retrieval_node` → 对每个改写查询又调用 `retrieve_with_optimization()`，后者内部再次 `classify()` + 改写，造成：

```
原始查询 → rewrite(1 LLM) → 4 queries
  → retrieve_with_optimization(q1) → classify + rewrite(1 LLM) → 4 more queries → 4×Pinecone
  → retrieve_with_optimization(q2) → classify + rewrite(1 LLM) → 4 more queries → 4×Pinecone
合计: 3 LLM + 8 Pinecone + 8 Whoosh
```

**修改**：统一优化入口，`query_processing` 输出结构化的 `{strategy, queries}`，检索直接使用，不再二次改写。

1. **`agents/graph.py`** `_query_processing_node` — 输出增加 `query_strategy` 字段：

```python
async def _query_processing_node(state: AgentState) -> dict:
    plan = state.get("plan", {})
    strategy = plan.get("rewrite_strategy", "rewrite")
    query = state.get("resolved_query", "") or state.get("original_query", "")

    from tools.query_processing import (
        classify, multi_query_rewrite, hyde_generate, decompose,
    )

    if strategy == "direct":
        queries = [query]
    elif strategy == "hyde":
        queries = hyde_generate(query)
    elif strategy == "decompose":
        queries = decompose(query)
    else:
        queries = multi_query_rewrite(query)

    return {
        "shared_context": queries,           # 保留现有字段兼容
        "optimized_queries": queries,         # 新增：结构化的优化结果
        "query_strategy": strategy,           # 新增：策略标记
    }
```

1. **`tools/rag_optimizer.py`** `retrieve_with_optimization()` — 增加 `skip_optimization` 参数：

```python
def retrieve_with_optimization(
    query: str,
    dense_retriever,
    k_final: int = 5,
    skip_optimization: bool = False,  # 新增
) -> tuple[list[str], str]:
    # ... nickname resolve + search_query 不变 ...

    if skip_optimization:
        queries = [search_query]
        strategy = "direct"
        _last_debug["optimization"] = "已由上游优化，跳过"
    else:
        # 原有逻辑
        strategy = classify(search_query) if config.ENABLE_QUERY_OPTIMIZATION else "rewrite"
        # ...
    
    # Step 3: 多路检索（保持不变）
    # ...
```

1. **`agents/graph.py`** `_knowledge_retrieval_node` — 传递标记：

```python
# 检测上游是否已做查询优化
already_optimized = state.get("query_strategy") in ("rewrite", "hyde", "decompose") \
                    and bool(state.get("optimized_queries"))

for q in search_queries[:2]:
    docs, _ = retrieve_with_optimization(
        q, retriever, k_final=config.RETRIEVER_K,
        skip_optimization=already_optimized,
    )
    shared_context.extend(docs)
```

**预期收益**：减少 1-2 次 LLM（\~2s），Pinecone 从 8 次降至 2 次（\~1s）。合计 **\~3s**。

***

### P0-2：Planner 规则优先，跳过不必要 LLM（收益 \~2s，低工作量）

**问题**：`planner.py` `plan()` 对所有查询都调 `answer_LLM.invoke()`，但 metadata/chat 结果被后续规则覆盖。

**修改**：`agents/planner.py` `plan()` 函数，规则能决定时直接返回：

```python
def plan(query: str) -> dict:
    query_category = _classify_query_category(query)

    # ── 规则直接生成 Plan，零 LLM ──

    # 1. 纯 metadata 查询（公司/声优/年份/评分/标签）
    if query_category == "metadata":
        # 进一步区分 simple_fact vs 其他
        is_simple = bool(re.search(r"(评分|声优|导演|公司|制作|标签|类型|介绍)", query))
        return {
            "query_type": "simple_fact" if is_simple else "recommendation",
            "rewrite_strategy": "direct",
            "experts": ["metadata_reasoner"],
            "parallel": False,
            "need_web": False,
            "query_category": "metadata",
            "reasoning": "规则判断：纯 metadata 查询"
        }

    # 2. 闲聊
    CHAT_PATTERNS = ["你好", "hi", "hello", "谢谢", "再见", "bye", "你是谁", "你能做什么", "帮我"]
    if any(p in query for p in CHAT_PATTERNS) and len(query) <= 15:
        return {
            "query_type": "chat",
            "rewrite_strategy": "direct",
            "experts": [],
            "parallel": False,
            "need_web": False,
            "query_category": query_category,
            "reasoning": "规则判断：闲聊"
        }

    # 3. 纯语义查询（高级评价/解读类）→ semantic only
    if query_category == "semantic":
        return {
            "query_type": "recommendation",
            "rewrite_strategy": "rewrite",
            "experts": ["similar_expert"],
            "parallel": False,
            "need_web": False,
            "query_category": "semantic",
            "reasoning": "规则判断：纯语义查询"
        }

    # ── 复杂 / mixed 查询才走 LLM ──
    # ... 原有 LLM invoke 逻辑 ...
```

**预期收益**：metadata/chat/semantic 类查询省 1 次 qwen-max。数据查询场景多时效果显著。**\~2s**。

***

### P1-1：Metadata / Vector 严格分流（收益 \~1-2s，中工作量）

**问题**：当前 mixed 查询同时走 metadata index + Pinecone，但很多查询其实只需要其中一路。metadata 类查公司/声优/评分 → 不需要 Pinecone；semantic 类查相似作品 → 不需要 metadata 标签过滤。

**现状**：`graph.py` `_knowledge_retrieval_node` 已经根据 `query_category` 分流（metadata→metadata index，semantic→Pinecone，mixed→两者）。**问题在于** Planner 默认 `query_category=mixed`，导致大部分查询走了双路。

**修改**：配合 P0-2 的规则 planner，让更多查询正确分流到单路，减少不必要的检索路径。

无需额外代码改动，P0-2 的规则 planner 确保 `query_category` 尽可能精确。

**预期收益**：减少不必要检索路径。metadata-only 查询从 \~500ms 降至 \~10ms。**\~0.5-1s**。

***

### P1-2：Answer Router — 简单问题用小模型（收益 \~1-3s，低工作量）

**问题**：当前只有 `chat` 类型用 qwen-flash，`simple_fact`（如"xx评分多少"）仍然走 qwen-max。

**修改**：`agents/answer.py` `answer_node()` 扩展 flash 使用范围：

```python
# answer.py
use_fast = query_type in ("chat", "simple_fact")
llm = simple_LLM if use_fast else answer_LLM
```

同时也降低 `ANSWER_TEMPERATURE` 默认值：

```python
# config.py
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.7"))  # 0.9 → 0.7
```

**预期收益**：simple\_fact 查询 answer 从 qwen-max(\~4s) → qwen-flash(\~1s)，省 **\~3s**。降低温度后复杂查询 answer 省 **\~1s**。

***

### P1-3：检索并行化（收益 \~0.5s，中工作量）

**问题**：`rag_optimizer.py:152-165` 对改写查询串行调用 `dense_retriever.invoke(q)`，多个 Pinecone 网络请求串行等待。

**修改**：`tools/rag_optimizer.py` 检索循环改为并发：

```python
import asyncio

# 并发发送所有 Pinecone 查询 + Whoosh 查询
async def _retrieve_parallel(queries, dense_retriever):
    all_dense, all_sparse = [], []
    dense_counts, sparse_counts = {}, {}

    async def _dense(q):
        try:
            docs = dense_retriever.invoke(q)
            return [(d.page_content, 1.0) for d in docs], len(docs)
        except Exception:
            return [], 0

    async def _sparse(q):
        docs = search_whoosh(q, k=config.HYBRID_SPARSE_K)
        return docs, len(docs)

    # 并行执行所有 dense + sparse
    tasks = []
    for q in queries:
        tasks.append(_dense(q))
        tasks.append(_sparse(q))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, q in enumerate(queries):
        d_result, s_result = results[i*2], results[i*2+1]
        if not isinstance(d_result, Exception):
            docs, cnt = d_result
            all_dense.extend(docs)
            dense_counts[q] = cnt
        if not isinstance(s_result, Exception):
            docs, cnt = s_result
            all_sparse.extend(docs)
            sparse_counts[q] = cnt

    return all_dense, all_sparse, dense_counts, sparse_counts
```

**联动**：P0-1 消除双重优化后，改写查询数量从 4×2=8 降至 2-4 个，并行化后仍有效果。

**预期收益**：省 \~300-500ms（网络 I/O 并行化）。

***

### P2-1：动态 Rerank — 候选少时自动跳过（收益 \~0.3s，极低工作量）

**问题**：CrossEncoder 在候选文档少（≤5条）时无意义，但仍会执行。

**修改**：`tools/rag_optimizer.py` rerank 逻辑已有此判断（`len(fused_docs) > k_final`），保持即可。另可通过 `.env` 全局关闭：

```bash
ENABLE_RERANKING=false
```

无需代码改动。只需在文档中标注推荐配置。

***

### P2-2：缓存层（长期优化，高工作量，本期不做）

建议方向（不在本期实施范围内）：

* **Query Cache**：缓存 Planner / Rewrite 结果（相同查询复用）

* **Retrieval Cache**：缓存 TopK 文档（热点查询复用）

* **Entity Cache**：已有 `@lru_cache`，可扩展 TTL

已有基础：`query_processing.py` 内部有 dict 缓存，`entity_resolver.py` 有 `@lru_cache`。

***

## 实施优先级 & 汇总

| 优先级    | 优化项                          | 收益     | 工作量       | 涉及文件                        |
| ------ | ---------------------------- | ------ | --------- | --------------------------- |
| **P0** | 消除双重查询优化                     | \~3s   | 中         | graph.py, rag\_optimizer.py |
| **P0** | Planner 规则优先                 | \~2s   | 低         | planner.py                  |
| **P1** | Answer Router (simple→flash) | \~3s   | 低         | answer.py, config.py        |
| **P1** | Metadata/Vector 严格分流         | \~0.5s | 低（依赖P0-2） | 无额外改动                       |
| **P1** | 检索并行化                        | \~0.5s | 中         | rag\_optimizer.py           |
| **P2** | 动态 Rerank                    | \~0.3s | 极低        | 配置 / 无改动                    |

**合计预期**：从 \~13s 降至 **\~4-6s**（减少 55-65%）。

## 假设与决策

1. **保留 qwen-max 作为复杂查询主力** — 简单问题用 flash，复杂推理用 max
2. **保留双 Expert 并行架构** — 通过 Send API 分发，已是最优
3. **不改动 Web Fallback 路径** — 条件触发，不影响常规查询
4. **不引入大重构**（同步→异步 LLM 全面改造）— 风险高，作为后续迭代
5. **缓存层作为 P2** — 本期优先消除重复计算，缓存留待下期
6. **CrossEncoder 默认关闭** — 推荐用户 `.env` 设置 `ENABLE_RERANKING=false`

## 验证步骤

1. 运行现有测试确保功能无回归：

   ```bash
   python tests/test_agent.py
   python tests/test_entity_resolver.py
   python tests/test_integration.py
   ```
2. 典型查询延迟验证：

   * "京都动画有哪些作品" — metadata → 期望 \~3s（原 \~8s）

   * "推荐热血番" — mixed → 期望 \~5s（原 \~13s）

   * "进击的巨人评分多少" — simple\_fact → 期望 \~2s（原 \~8s）
3. 对比回答质量：确认 flash 回答简单问题时质量无明显下降

