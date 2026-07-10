# 优化提示词 + 职责重构：让回答更自然、更可信

> **状态**: ✅ 已实施 — Expert/Answer 职责分离、Answer Planner 随机结构、system prompt 优化均已实现。

## 核心理念

> Expert 提供可信证据，Answer Agent 只负责重组表达，不做事实创造。

当前问题不是单一 prompt 不够好，而是**职责边界模糊**：
- Answer Agent 拿到 Expert 的自由文本后，会二次理解和扩展 → 容易产生幻觉、编造评论、修改事实
- Expert 输出没有结构化证据，Answer 难以分辨哪些是事实、哪些是推断
- 缺乏结构变化的机制，每次回答都是一个套路

---

## 改动范围：5 个文件

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `agents/state.py` | AgentState 新增 `answer_plan` 字段 | 低 |
| `agents/metadata_reasoner.py` | 只改 `_REASONER_SYSTEM` 字符串 | 低 |
| `agents/similar_expert.py` | 只改 `_SIMILAR_SYSTEM` 字符串 | 低 |
| `agents/answer.py` | 改 `_ANSWER_SYSTEM` + `_ANSWER_USER` + `answer_node` 读取 answer_plan | 中 |
| `agents/graph.py` | 新增 `_answer_planner_node` + 改 3 处路由 + 1 条边 | 中 |

---

## 文件 1: `agents/state.py` — 新增字段

```python
class AgentState(TypedDict):
    # ... 现有字段 ...
    answer_plan: dict                       # Answer Planner 输出的结构指引
```

放在 `alias_cache` 后面即可。

---

## 文件 2: `agents/metadata_reasoner.py` — 证据化输出

**`_REASONER_SYSTEM`** 替换为：

```
你是阅番无数的资深二次元。你的任务是根据番剧数据和观众评论，给出有观点的分析。

## 你会收到的数据
1. 结构化元数据（JSON）: 番剧名、Bangumi 评分、排名、标签、制作公司、导演、声优等
2. 语义上下文（文本）: 含"观众评论: xxx"字段、番剧描述等

## 输出格式
严格 JSON:
{
  "answer": "口语化的分析结论 — 每个观点都要有依据",
  "confidence": 0.85,
  "evidence": ["来源A: 具体数据", "来源B: 观众评论原文片段"]
}

## 输出原则
- **证据导向**: 每个判断后面跟依据，如 "这部在 Bangumi 上 8.5 分排前 50，口碑很稳"
- **评论优先**: 上下文里有"观众评论:"的话，优先引用 — 比干数据有说服力
- **指出争议**: 评论明显分歧时如实反映，显得客观
- **不用套话**: 别写"根据数据显示""综合分析认为" — 直接说结论

## 风格
像在动漫群里跟群友聊番，不是写分析报告。数据不够就如实说，别硬编。
```

`_REASONER_USER` 不变。

---

## 文件 3: `agents/similar_expert.py` — 多维相似

**`_SIMILAR_SYSTEM`** 替换为：

```
你是资深动漫宅，专门帮人找"像 XX 的番"。用聊天的方式推荐。

## 输出格式
严格 JSON:
{
  "answer": "口语化的推荐，每个推荐2-3句",
  "confidence": 0.85,
  "evidence": ["引用来源"]
}

## 推荐角度（每个推荐至少覆盖 2 个维度）
- 剧情走向: 叙事结构、反转密度、节奏感
- 角色塑造: 人设深度、角色弧光
- 世界观: 背景设定复杂度
- 观看体验: 情感冲击、代入感
- 观众反馈: 上下文中如有"观众评论"，引用作为佐证

## 不要做的事
- 别只说"风格相似"，要说具体哪里像
- 别罗列评分标签，融入句子里
- 评分差距大的要提
- 也说说差异点和适合谁/不适合谁
```

`_SIMILAR_USER` 不变。

---

## 文件 4: `agents/answer.py` — 只重组不创造

### 4a. `_ANSWER_SYSTEM` 替换为：

```
你是资深二次元，帮朋友推荐番剧。你不是 AI，不是客服，不写报告。

## 你的唯一职责
你只做三件事：重组分析结果、调整语气、组织推荐逻辑。
你不创造新的事实、不扩展评论、不补充数据。

## 说话方式
- 像跟朋友微信聊番，不是交作业
- 每部作品的推荐句式都不一样，换着花样来
- 可以有偏好，可以说"我超爱这部""这部说实话一般"
- 数据融入句子: "Bangumi 上 8.5 分" 而不是 "评分：8.5"
- 分析结果中有观众评论的话，自然穿插: "Bangumi 上有人觉得'结局封神'，也有人嫌节奏慢"

## 结构变化（根据结构指引切换语气和布局）
- 推荐多部时：最想推的放最前面多聊几句，后面简略带过
- 简单查询时：直接说答案，顺带点有趣的小知识
- 可以偶尔用"先说你最可能喜欢的"、"如果口味偏重可以试试"这种引导句

## 禁止事项
- 禁止: "推荐理由""综合分析""值得注意的是""综上所述""根据分析""笔者认为"
- 禁止: 每部作品用相同句式罗列
- 禁止: 编造分析结果里不存在的番剧名、评分、评论
- 不确定的信息直接说"这个我不太确定"，别硬编

## 心态
Expert 输出是"找证据的人"写的，你的任务是把这些证据用聊天的方式讲出来。像刚从 Bangumi 逛了一圈回来跟朋友分享。
```

### 4b. `_ANSWER_USER` 替换为：

```
## 用户问题
{query}

## 回答结构指引
{structure}

## Expert 分析结果
{merged_results}

请生成回答。
```

### 4c. `answer_node` 函数修改：

在现有逻辑基础上，增加读取 `answer_plan` 并传入 prompt：

```python
async def answer_node(state: dict) -> dict:
    from llms import answer_LLM

    query = state.get("original_query", "")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "unknown")

    # 闲聊直接回复（跳过所有）
    if query_type == "chat":
        from llms import simple_LLM
        resp = simple_LLM.invoke([HumanMessage(content=query)])
        return {"messages": [resp]}

    merged_results = state.get("merged_results", "")
    if not merged_results:
        expert_results = state.get("expert_results", [])
        if expert_results:
            parts = []
            for i, r in enumerate(expert_results, 1):
                answer = r.get("answer", "")
                confidence = r.get("confidence", 0)
                if answer:
                    parts.append(f"[Expert {i} | 置信度: {confidence:.0%}]\n{answer}")
            merged_results = "\n\n".join(parts)
        else:
            merged_results = "(分析结果为空)"

    # 读取 Answer Planner 输出的结构指引
    answer_plan = state.get("answer_plan", {})
    structure = answer_plan.get("structure", "自由发挥")

    llm = answer_LLM.bind(temperature=config.ANSWER_TEMPERATURE)

    resp = llm.invoke([
        SystemMessage(content=_ANSWER_SYSTEM),
        HumanMessage(content=_ANSWER_USER.format(
            query=query,
            structure=structure,
            merged_results=merged_results,
        )),
    ])

    return {"messages": [resp]}
```

---

## 文件 5: `agents/graph.py` — 新增 Answer Planner

### 5a. 新增 `_answer_planner_node` 函数：

放在 `_route_after_merge` 函数之后即可：

```python
import random

def _answer_planner_node(state: AgentState) -> dict:
    """零 LLM 成本的回答结构规划器，随机选结构避免套路化"""
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "recommendation")

    if query_type == "chat":
        return {"answer_plan": {"structure": "简短闲聊"}}

    structures = {
        "recommendation": [
            "top_pick — 先重点安利最推荐的1-2部，多说几句为什么喜欢，后面简略带过",
            "compare — 用对比的方式介绍，突出每部特点，让用户自己选",
            "theme — 按主题/风格归类推荐，先说共同点再展开",
            "honest — 先夸优点再说槽点，显得客观，加一句看你自己口味",
        ],
        "simple_fact": [
            "direct — 直接回答核心问题，顺带讲个相关趣事",
            "expand — 先回答核心问题，再补充1-2个相关维度",
        ],
        "comparison": [
            "vs — 逐项对比，最后一句总结谁更适合什么人",
            "narration — 先分别讲每部特点，最后说更看重X就选A看重Y就选B",
        ],
    }

    options = structures.get(query_type, structures["recommendation"])
    chosen = random.choice(options)

    return {"answer_plan": {"structure": chosen, "tone": "casual"}}
```

### 5b. `build_graph()` 中的变更：

```python
# 新增节点
g.add_node("answer_planner", _answer_planner_node)

# 修改 3 处路由目标（answer → answer_planner）：

# 1. merge → answer_planner 或 web_fallback（原来是 answer）
g.add_conditional_edges("merge", _route_after_merge, {
    "web_fallback": "web_fallback",
    "answer_planner": "answer_planner",
})

# 2. web_fallback → answer_planner（原来是 answer）
g.add_edge("web_fallback", "answer_planner")

# 3. _route_after_merge 返回值改为 "answer_planner"
# 注意：chat 路径（planner → answer）保持不变
```

### 5c. `_route_after_merge` 修改：

```python
def _route_after_merge(state: AgentState) -> str:
    """Merge 后 → web_fallback 或 answer_planner"""
    if should_trigger_web(state):
        return "web_fallback"
    return "answer_planner"  # 原来是 "answer"
```

### 5d. 新增边：

```python
g.add_edge("answer_planner", "answer")
```

---

## 图路由全景（变更后）

```
START → alias_resolve → planner
                          ├─ chat → answer → END
                          └─ 其他 → query_processing → knowledge_retrieval
                                      ├─ 有 experts → [Send parallel]
                                      │    → metadata_reasoner ┐
                                      │    → similar_expert    ├→ merge
                                      │                        │
                                      │    ┌───────────────────┘
                                      │    ↓
                                      │   merge → web_fallback? → answer_planner → answer → END
                                      │
                                      └─ 无 experts → answer_planner → answer → END
```

---

## 验证方式

`python tests/test_agent.py`:
1. `推荐一部类似命运石之门的科幻番` → 多次运行，确认回答结构不同（top_pick / compare / theme / honest 随机）
2. `进击的巨人怎么样` → 回答有 Bangumi 评论引用、不模板化
3. `巨人vs鬼灭哪个好看` → 对比式回答，有具体差异点
4. `今天星期几` → chat 路简短友好，不走 answer_planner
