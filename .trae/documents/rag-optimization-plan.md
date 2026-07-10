# RAG 多层级检索优化 实施计划（修订版）

> **状态**: ✅ 已实施 — 三路索引（Metadata + Pinecone + Whoosh）+ RRF 融合 + CrossEncoder 精排 已实现。

## 总览

基于建议优化后的三层 RAG 架构：

```
用户查询 → Query Classifier → Query Optimizer → Multi-Query Hybrid Retrieval → Fusion → CrossEncoder → Context Compression → LLM → Answer Verification
```

核心原则：

* Query Optimizer **放入 RAG 工具内部**，不新增 Graph Node

* Rewrite / HyDE **互斥**，按查询类型择一

* 混合检索用 **Whoosh** 替代 BM25 pickle（无需同步维护）

* CrossEncoder **启动时预加载**

* State 只存 `query_strategy`，不污染

***

## 一、文件变更清单

### 新建

| 文件                       | 职责                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `tools/rag_optimizer.py` | QueryClassifier + QueryOptimizer + HybridRetriever + Fusion + CrossEncoder Reranker + ContextCompressor |

### 修改

| 文件                 | 改动                                   |
| ------------------ | ------------------------------------ |
| `tools/rag.py`     | RAG 工具内部集成全链路优化                      |
| `state.py`         | 新增 `query_strategy: str`             |
| `config.py`        | 新增检索优化相关配置项                          |
| `data/build_kb.py` | 构建时同步写入 Whoosh 索引                    |
| `llms.py`          | 启动时预加载 CrossEncoder                  |
| `requirements.txt` | 添加 `whoosh`, `sentence-transformers` |

***

## 二、Query Classifier — 查询分类器

**不调用 LLM**，用规则 + 关键词快速分类：

| 类别          | 判断依据                           | 策略                  |
| ----------- | ------------------------------ | ------------------- |
| `direct`    | 问候/闲聊/简单事实（<8字无维度词）            | 跳过 RAG，直接 LLM       |
| `rewrite`   | 含推荐/筛选/口语化表述                   | Multi-Query Rewrite |
| `hyde`      | 含比较/推理/深度评价词（"为什么/比XX更好/深度分析"） | HyDE                |
| `decompose` | 含多个独立子问题                       | 查询拆分                |

实现：纯规则（正则 + 关键词表），零 LLM 调用，<1ms。

***

## 三、Query Optimizer — 查询优化器

### 3.1 Multi-Query Rewrite

```
原始 query → LLM 生成 3 个不同视角的 rewrite
  - 原始 query
  - rewrite_1（侧重类型/标签）
  - rewrite_2（侧重评分/口碑）
  - rewrite_3（侧重staff/制作）
→ 4 条 query 分别检索
```

### 3.2 HyDE

```
原始 query → LLM 生成假设性答案 → 用答案去检索
```

* 仅用于推理/比较类查询

* 与 Rewrite **互斥**，不叠加

### 3.3 Query Decompose

```
复杂 query → LLM 拆为子问题列表 → 逐个检索 → 汇总
```

### 3.4 缓存

* Rewrite/HyDE/Decompose 结果缓存（LRU，key=query md5）

* 命中缓存直接复用，跳过 LLM 调用

***

## 四、混合检索 Hybrid Retrieval

### 4.1 双路召回

| 路      | 实现                              | 返回        |
| ------ | ------------------------------- | --------- |
| Dense  | Pinecone (dense embedding, MMR) | top\_k=10 |
| Sparse | Whoosh (全文索引, BM25F)            | top\_k=10 |

**Whoosh 方案**（替代 pickle）：

* 纯 Python，无需服务端

* 构建知识库时同步写入 `data/whoosh_index/`

* 无需维护同步，每次 `build_kb.py` 重建索引

* 数据量 5000\*3 chunks ≈ 15000 docs，Whoosh 完全胜任

### 4.2 Fusion 融合排序

三种策略可选，配置切换：

| 策略       | 公式                                           | 适用       |
| -------- | -------------------------------------------- | -------- |
| RRF      | `score = Σ 1/(k+rank)`                       | 默认，无需归一化 |
| Weighted | `score = w_d*dense_score + w_s*sparse_score` | 可调权重     |
| Max      | `score = max(dense_score, sparse_score)`     | 任一命中即高分  |

***

## 五、CrossEncoder 精排

### 5.1 模型

`BAAI/bge-reranker-v2-m3`（多语言，中文友好，568MB）

### 5.2 预加载

```python
# llms.py — 启动时加载
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
```

首次下载后缓存，后续启动直接加载。

### 5.3 流程

```
融合后 ~15 docs → CrossEncoder 逐对(query, doc)打分 → 按分数排序 → top_k=5
```

只在检索阶段调用，不影响非 RAG 路径。

***

## 六、Context Compression

对精排后的文档后处理：

1. **去重** — 相似度 > 0.9 的文档合并
2. **截断** — 每篇文档取最相关的前 300 字
3. **关键句提取** — 用 embedding 找与 query 最相关的句子

降低 Token 消耗，提升 LLM 回答质量。

***

## 七、Answer Verification

生成答案后加一层校验：

```
LLM 回答 → 检查是否有检索依据支撑 → 标记置信度
  - 高置信度：直接返回
  - 低置信度：追加"仅供参考"提示 / 降级到通用回答
```

轻量级，用规则 + 简单 N-gram 重合度判断，不引入额外 LLM 调用。

***

## 八、工作流集成

### 关键决策：Optimizer 放入 RAG 工具内部

**不做**：新增 Graph Node。
**做**：在 `RAG()` 工具函数内部完成全链路。

```python
@tool
def RAG(query: str) -> str:
    # 1. Query Classifier (规则, 0ms)
    strategy = classify(query)
    
    # 2. Query Optimizer (LLM, 可选)
    if strategy == "rewrite":
        queries = multi_query_rewrite(query)  # → 4 queries
    elif strategy == "hyde":
        queries = [hyde_generate(query)]      # → 1 hyde query
    elif strategy == "decompose":
        queries = decompose(query)            # → N sub-queries
    
    # 3. Hybrid Retrieval
    all_docs = []
    for q in queries:
        dense_docs = pinecone_retrieve(q, k=10)
        sparse_docs = whoosh_retrieve(q, k=10)
        all_docs.extend(fusion(dense_docs, sparse_docs))
    
    # 4. CrossEncoder Rerank
    ranked = cross_encoder_rerank(query, all_docs, top_k=5)
    
    # 5. Context Compression
    compressed = compress(ranked)
    
    return format_results(compressed)
```

Graph 零改动，保持简洁。

### State 改动

```python
# state.py — 仅新增一个字段
query_strategy: str  # "direct" | "rewrite" | "hyde" | "decompose"
```

***

## 九、配置项

```python
# config.py 新增
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
FUSION_STRATEGY = os.getenv("FUSION_STRATEGY", "rrf")  # rrf | weighted | max
ENABLE_QUERY_OPTIMIZATION = os.getenv("ENABLE_QUERY_OPTIMIZATION", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "false").lower() == "true"
HYBRID_DENSE_K = int(os.getenv("HYBRID_DENSE_K", "10"))
HYBRID_SPARSE_K = int(os.getenv("HYBRID_SPARSE_K", "10"))
```

***

## 十、检索评估

新增 `data/eval_retrieval.py`：

| 指标         | 说明               |
| ---------- | ---------------- |
| Recall\@5  | 前5个结果中命中相关文档的比例  |
| Recall\@10 | 前10个结果中命中相关文档的比例 |
| MRR        | 第一个相关文档的排名倒数均值   |
| nDCG\@5    | 考虑排序位置的相关性得分     |
| Latency    | 检索全链路耗时          |

每次优化后跑评估，量化对比。

***

## 十一、实施顺序

| 步骤 | 内容                                                                                                 | 可并行      |
| -- | -------------------------------------------------------------------------------------------------- | -------- |
| 1  | `pip install sentence-transformers whoosh`                                                         | -        |
| 2  | `config.py` 新增配置项                                                                                  | ✅ 与 3 并行 |
| 3  | 创建 `tools/rag_optimizer.py`（Classifier + Optimizer + Whoosh + Fusion + CrossEncoder + Compression） | ✅ 与 2 并行 |
| 4  | 修改 `tools/rag.py` 集成优化器                                                                            | 依赖 3     |
| 5  | `llms.py` 预加载 CrossEncoder                                                                         | ✅ 与 4 并行 |
| 6  | `build_kb.py` 构建时写 Whoosh 索引                                                                       | 依赖 4     |
| 7  | `state.py` 新增 `query_strategy`                                                                     | ✅ 与 6 并行 |
| 8  | 测试验证                                                                                               | 依赖全部     |

***

## 十二、降级策略

| 场景               | 降级                   |
| ---------------- | -------------------- |
| CrossEncoder 未下载 | 跳过精排，用 Fusion 结果     |
| Whoosh 索引不存在     | 仅 Dense 检索           |
| Pinecone 不可用     | 仅 Sparse (Whoosh) 检索 |
| LLM 优化超时         | 跳过优化，原始 query 直接检索   |
| 查询 < 5 字         | 跳过所有优化               |

***

## 十三、与建议的对照

| 建议                       | 采纳 | 说明                      |
| ------------------------ | -- | ----------------------- |
| 1. Query Classifier      | ✅  | 规则分类器，零 LLM 成本          |
| 2. Rewrite/HyDE 互斥       | ✅  | 按类型择一                   |
| 3. Multi-Query Retrieval | ✅  | 原始 + 3 个 rewrite        |
| 4. 不用 bm25.pkl           | ✅  | 改用 Whoosh               |
| 5. Fusion 多策略            | ✅  | RRF / Weighted / Max    |
| 6. CrossEncoder 预加载      | ✅  | llms.py 启动加载            |
| 7. Optimizer 放 RAG 内部    | ✅  | 不新增 Graph Node          |
| 8. State 轻量              | ✅  | 仅存 query\_strategy      |
| 9. Cache                 | ✅  | LRU 缓存优化结果              |
| 10. Retrieval Evaluation | ✅  | Recall/MRR/nDCG/Latency |
| 11. Context Compression  | ✅  | 去重+关键句提取                |
| 12. Answer Verification  | ✅  | 检索依据检查                  |

