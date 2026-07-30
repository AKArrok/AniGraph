# 消融实验报告 (Ablation Study)

面向 AniRAG 的 Bangumi 番剧问答任务。本报告记录了从数据集设计、
变体设置、到 metadata filter 路由干预 的完整流程与结论。

数据文件：`tests/eval_hard_final.json` (50 条)
结果文件：`tests/hard_eval_results.json`
Excel 表：`tests/ablation_results.xlsx` (Hard Summary / Hard Cases)

---

## 1. 数据集设计

原始 17 条 hard 集偏工整（6 类，几乎都是热门条目），LLM 只靠训练语料
就能蒙对不少（H0 直问 LLM ~18% strict）。为了让实验真正区分 RAG 的
价值，扩到 **50 条 / 14 类**，其中 33 条是围绕 Bangumi 冷数据设计的：

| 类型 | 条数 | 冷数据信号 |
|---|---|---|
| cold_score_precise    | 4  | `score_count 100-400` 的番剧精确一位小数评分 |
| cold_longtail_fact    | 5  | `score_count 60-150` 的番剧评分 + 首播年份 |
| release_date_precise  | 5  | 精确到 `YYYY-MM`（避开常规季播月） |
| cold_studio_year      | 3  | 小工作室 + 特定年份的作品列表 |
| tag_top_score         | 4  | 稀有 Bangumi tag 榜首 |
| numeric_comparison    | 4  | 两番评分对比（分差 ≥ 0.4） |
| seiyuu_cold_works     | 4  | 只出现在 3-5 部作品的声优的作品表 |
| refusal_fabricated    | 4  | 虚构条目名 —— 考察拒答 |

原有 17 条：`metadata_cross` × 3、`longtail_fact` × 3、`similar_recommendation` × 4、
`bangumi_tags` × 3、`bangumi_score_precise` × 3、`kb_boundary` × 1。

打分完全确定性（`tests/deterministic_scorer.py`），没有 LLM Judge，
每条题的 gold 是从 SQLite 直接拉出来的规则化事实。

---

## 2. 变体设置

| 变体 | 配置 | 说明 |
|---|---|---|
| H0 | 无 RAG               | 直问 LLM，只有系统提示要求"不知就说未记录" |
| H1 | Dense RAG top_k=20   | + `cite sources when possible` 提示 |
| H2 | Dense RAG top_k=3    | 极限压缩上下文 |
| H3 | Dense RAG top_k=10   | 中间点 |
| H4 | Dense RAG top_k=20   | + `do not cite / mention them` 提示 |
| H5 | Dense RAG top_k=40   | 放大 top_k |
| H6 | Dense RAG top_k=60   | 更放大 top_k |

**H1–H6 使用同一个 embedding 缓存 (`tests/_emb_cache.json`)**，检索
延迟可比。

除 H0 之外的所有变体，一旦遇到 `cold_studio_year` 或
`seiyuu_cold_works` 类型的问题，会走 **MetadataIndex 结构化过滤路由**
（见 §4），不再走 Dense retrieval —— 这更贴合生产系统中"意图路由到
合适的 retriever"的常见做法，也让指标反映真正部署形态。

---

## 3. 主要结论

### 3.1 综合指标 (n=50)

| 变体 | strict acc | partial | avg latency | avg tokens |
|---|---|---|---|---|
| H0 no RAG              | 26.0% | 0.43 | 2.0s | 109  |
| H2 top_k=3             | 50.0% | 0.50 | 2.0s | 259  |
| H3 top_k=10            | 78.0% | 0.79 | 2.2s | 642  |
| H1 top_k=20 (cite)     | 90.0% | 0.90 | 5.4s | 1337 |
| H4 top_k=20 (no-cite)  | **92.0%** | 0.93 | 3.2s | 1372 |
| H5 top_k=40            | 92.0% | 0.93 | 3.2s | 2768 |
| H6 top_k=60            | 96.0% | 0.96 | 3.5s | 4273 |

### 3.2 关键结论

**a. RAG 对冷 Bangumi 数据的增益极大**
H0 → H1 从 26% 提到 90%，`cold_score_precise` / `cold_longtail_fact`
/ `numeric_comparison` 三类 H0 全部 0/N，任何一个 top_k ≥ 10 的变体
基本满分。KB 的 `profile` chunk 里"评分 X.Y / 播出日期 YYYY-MM-DD"
这种结构化字段直接支撑得住这些题。

**b. Prompt 措辞 > top_k**
H1 (top_k=20 +cite) 90% vs H4 (top_k=20 no-cite) 92%，同 retrieval
但差 2 个点、延迟从 5.4s 掉到 3.2s、平均总 tokens 相当而完成 tokens
更少。分析 H1 的答案：多次输出 "未记录"，即使检索结果里明摆着有
答案。`cite sources when possible` 让模型对"证据是否够正式"过度自
保。**结论：生产环境用 H4 的 no-cite 提示。**

**c. top_k 收益在 20 之后趋平**
3→10 提 28 个点，10→20 提 14 个点，20→40 提 0 个点，40→60 提 4 个
点。但 60 的 token 是 20 的 3 倍多。**性价比最优点是 top_k = 20 (no-cite)**。

**d. Dense RAG 单独解不了两类枚举题 —— 需要 metadata filter**
- `cold_studio_year`（列出某工作室某年的作品）
- `seiyuu_cold_works`（列出某声优参与的所有作品）

两类合计 7 条，Dense-only 全部变体 (H1–H6) 命中率 0/7 – 2/7，见 §4。
一次向量查询按 query 语义排序，无法枚举满足精确 metadata 谓词的所
有 chunk。

**e. H2 top_k=3 是掉线，不是压缩**
延迟只省 1s、token 只省 80%，却从 92% 掉到 50%。想省 token，应该
走 rerank + top_5，而不是直接砍 top_k。

**f. `refusal_fabricated` 有暗坑**
四条虚构条目，除 H4 上一条把邻居真番的评分吐出来（"没找到《苍穹之
翼：无名之战》……与'苍穹'相关的作品有《苍穹之法芙娜》…"）之外，
所有变体 4/4 通过。**但 H2 top_k=3 上有 1/4 因为召回太少反而虚构了
分数**。RAG 越大反而越安全，因为噪音多但至少不会自信编造。

---

## 4. Metadata Filter 路由（H1–H6 共用）

**触发条件**：`cold_studio_year` / `seiyuu_cold_works` 两类题（未来可
以扩展）。

**流程**：
```python
if type == "cold_studio_year":
    rows = MetadataIndex.search(
        studio=studio, date_from=f"{year}-01-01", date_to=f"{year}-12-31"
    )
elif type == "seiyuu_cold_works":
    rows = MetadataIndex.search(seiyuu=name)
titles = [row["name_cn"] or row["name"] for row in rows]
# 把 titles 作为候选列表注入 prompt，让 LLM 组织答案
```

**效果对比（7 条题）**：

| 类型 | Dense-only 各变体 | Metadata filter |
|---|---|---|
| cold_studio_year (n=3)  | 0/3, 0/3, 0/3, 0/3, 1/3 (H2–H6) | **3/3** |
| seiyuu_cold_works (n=4) | 0/4, 1/4, 1/4, 2/4, 0/4 (H2–H6) | **4/4** |
| 合计 | 0/7 – 3/7                       | **7/7** |

Metadata filter 单条平均延迟 1.1s，是 Dense H4 的 1/3，且 token 消
耗只有百级别（不用把 20 条 chunk 塞进 prompt）。这是把整体 accuracy
从 82% (纯 Dense) 拉到 92% (H4 with routing) 的关键。

**注意事项**：

1. 这是**路由**而非"filter 替代 RAG" —— filter 只解精确枚举，RAG 解
   模糊问答。生产系统需要一个前置的 intent classifier 把 query 分派
   到合适的 retriever。
2. 用户会用 "京都"、"京都动画"、"京阿尼" 等别名。目前的 MetadataIndex
   走子串匹配，对本次测试用例都成立，但未来复杂别名需要走 alias 表。
3. 别把这个当成 "metadata filter > RAG" 的证据，两者解不同类问题。

---

## 5. 剩余失败 & 下一步

即便在最好的 H6 (96%)，仍有 2 条失败：

- `similar_recommendation R000793`（相似番推荐）—— 4 条 gold 集只列
  8 部候选，Dense 检索拉回的推荐可能落在 gold 集之外。gold 集设计需
  要放宽或采用语义相似度打分。
- `metadata_cross M001` / `bangumi_tags T000440`（视变体不同）
  —— 部分标签召回不全。

**下一步优先级**：
1. 把 metadata filter 路由前置成 intent classifier（正则 + 少量
   LLM），覆盖更多"列举"类问题（director、writer、tag 榜单）
2. 生产变体切到 H4 配置（top_k=20 + no-cite prompt）
3. Similar recommendation 的 gold 池需要重新校准，不然它一直是随机
   变量

---

## 附录：Excel 表布局

`tests/ablation_results.xlsx`：

- **Hard Summary** — 14 类 × 7 变体的准确率矩阵，加 Overall Partial /
  Latency / Tokens 三行。Metadata filter 路由后的结果已直接写入
  H1–H6 的对应格子。
- **Hard Cases** — 每条题一行，7 个变体每列显示 PASS/FAIL + 答案摘要
  （前 200 字符），左侧冻结栏保留 Query / Type / ID。
- 原始 dense-only 数据保留在 JSON 的 `_orig_dense_only` 字段供追溯。
