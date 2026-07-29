# Hard Eval 失败用例分析

来源：`tests/hard_eval_results.json`（H1 = Dense top_k=20，DeepSeek v4-pro，`thinking=disabled`，`max_tokens=2048`）。
知识库版本：`data/anime_data.db` + Pinecone `vector`（62 779 条 dense 向量），Bangumi 抓取。

H1 整体准确率 12/17 = 70.6%。以下是 5 个失败中最有代表性的 3 个，
分别对应「RAG 召回缺失」「知识库字段缺失」「评测方法本身的偏差」三种不同根因。

---

## Case 1 — RAG 召回缺失（M003 metadata_cross）

**Query**: 「京都アニメーション 在 2021 年制作了哪些动画？」
**Gold**: `小林家的龙女仆S`、`迷你龙小剧场`
**H1 Answer**:
> 根据搜索结果，京都アニメーション 在 2021 年制作的动画有：紫罗兰永恒花园 特别编集版（播出日期 2021-10-29）。

**判定**：`correct=False`，`hits=[]`，`misses=['小林家的龙女仆S','迷你龙小剧场']`。

**根因**：
1. 检索层没把「制作公司 × 年份」的关系拉出来。当前 chunk 策略把每部番的信息切成
   `basic` / `staff` / `characters` / `comments` 等类型（见 `data/chunking.py`），"京阿尼 2021"
   的跨记录聚合需要 SQL 或 metadata 过滤，dense 相似度顶多命中"名字里带京都/紫罗兰"的 chunk。
2. `top_k=20` 里塞满了紫罗兰特别编集版和相关话题讨论，把小林家龙女仆 S 的 chunk 挤出去了。
3. 生成时模型忠实回答"搜索结果里只有紫罗兰"，反而说明 evaluator 该介入却没介入。

**修复方向**：
- Planner 检测到"公司 × 年份"意图时走 SQL 分支（`SELECT anime_title FROM Anime a JOIN Studio ...
  WHERE studio_name=? AND air_year=?`），dense 只做补充。
- 或在建库时新增 `studio_year_index` 结构化字段供 metadata filter 使用。
- Evaluator 应识别答案里只有 1 条命中但问题问的是"哪些"（复数），触发 replan。

---

## Case 2 — 知识库字段缺失（T000440 bangumi_tags）

**Query**: 「在 Bangumi 上，《校园迷糊大王》被观众/编辑打上了哪些标签？请列出主要标签。」
**Gold**: `喜剧` `恋爱` `恶搞` `搞笑` `日常` 等 Bangumi 社区标签。
**H1 Answer**:
> 根据搜索结果，未记录到 Bangumi 上《校园迷糊大王》被观众或编辑打上的具体标签信息。

**判定**：`correct=False`，`hits=[]`。这里模型没有幻觉，是老实承认知识库里没有。

**根因**：
这是知识库本身的漏洞，不是检索或生成失败：
1. `data/anime_data.db` 里 `校园迷糊大王` 的 tag 列表要么没抓到、要么切分时被过滤掉了
   （查库确认：Bangumi 页面确有标签，但 chunk 里没落地）。
2. Chunking 策略里 tag 是从 `Category` 关系表拼出来的，如果 `score_count < 阈值` 或抓取时
   条目状态异常，就会漏抓。
3. 对照 `T000247 银魂`：H1 命中 5/5 tag，因为它是热门番，抓取时数据完整。

**修复方向**：
- 建库脚本 (`data/build_kb.py`) 加数据完整性检查：任何 anime 若 tag chunk 为空但源库有
  Category 记录，写入告警日志或补一次抓取。
- Evaluator 或 web_fallback 在检测到"未记录到 tag 信息"这类字面 hedging 时应触发 Tavily 兜底。

---

## Case 3 — 评测方法本身的偏差（R000822 FLCL similar_recommendation）

**Query**: 「有没有跟《FLCL》相似的动画推荐？可以基于标签、导演或制作公司相似的角度来推荐几部。」
**Gold pool (rescore 后)**: 与 FLCL 共享 `anchor_tag=神配乐` 的 14 部动画。
**H1 Answer** (节选):
> 1. 《宇宙巡警露露子》——TRIGGER 是从 GAINAX 独立出来的，血脉相承。
> 2. 《心灵游戏》《海马》——同属「意识流」「奇幻」标签。
> ⋯（同时提到鹤卷和哉、GAINAX 血脉）

**判定**：`correct=False`，`hits=[]`。**但答案在人类看来是合理的**：TRIGGER/GAINAX 系脉络是
FLCL 最公认的推荐方向。失败原因不在模型，而在 gold pool 的选取：

1. FLCL 在 Bangumi 上的原始 tag 集里，我们选了 `神配乐` 作 anchor_tag，
   pool_size 只有 14 部（rescore 已用 `score_count >= 500` 放宽）。宇宙巡警露露子、
   天元突破等真正相关的作品共享的其实是「GAINAX 血统」「无厘头」等 tag，不在 pool 里。
2. 这暴露了 `similar_recommendation` 用「共享 anchor_tag」评分本质是脆弱的：
   相似性是多因素（导演/公司/精神传承/年代），单一 tag 无法充分表达。
3. 对比 R000298 天空之城（anchor_tag=经典，pool 147）通过率极高——反过来说，也可能是"沾光"通过：
   任何吉卜力片单都会命中 pool。评分变松了。

**修复方向**：
- 长期：换成 LLM-as-Judge，喂 gold_pool + 允许"合理外延" prompt。人机一致性重新校准。
- 短期：`similar_recommendation` 分成两档打分——严格池 (anchor_tag) 命中 + 语义池 (top-N by
  embedding similarity to seed) 命中，任一 ≥ 2 视为通过。
- 这也是为什么本次没把 similar_recommendation 数据放进主 KPI，只作为副指标。

---

## 面试可讲的三条经验

1. **Recall miss 与 Chunk 结构直接相关**：跨实体聚合类查询（公司×年份、导演×流派）
   dense retrieval 单打独斗必输，需要 metadata filter 或 SQL 分支。
2. **知识库缺字段时的正确姿势**：让模型 hedge > 让模型幻觉；同时 evaluator 侧检出 hedging 触发
   Tavily/web_fallback，是端到端可用性的关键。
3. **评测数据集自己也是被测项**：本次 similar_recommendation 4/4 全挂，但真挂的是评分方法
   不是模型。跑出结果就质疑「是模型烂还是我尺子歪」，是做 offline eval 的基本功。
