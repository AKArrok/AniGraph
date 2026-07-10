# 短期对话记忆实现计划（v3 — 两轮评审后定稿）

> **状态**: ✅ 已实施 (2026-07-10) — 核心方案按计划实现，并在此基础上扩展了 `simple_fact_answer` 快速通道和 LLM 超时+节点耗时日志  
> **实施分支**: `main`  
> **新增文件**: `agents/history_extractor.py`, `agents/context_builder.py`, `agents/simple_fact_answer.py`  
> **修改文件**: `agents/state.py`, `agents/graph.py`, `agents/planner.py`, `agents/answer.py`, `agents/metadata_reasoner.py`, `agents/similar_expert.py`, `config.py`, `llms.py`, `tests/test_agent.py`

## 摘要

为 AniGraph 添加短期对话记忆。核心思路：新增 `history_extractor` + `context_builder` 两个节点，统一生成结构化 `ConversationContext`，Planner 和 Answer 只消费 Context，不直接读 messages。

---

## 现状分析

| 组件 | 状态 | 问题 |
|------|------|------|
| `messages` (AgentState) | `add_messages` reducer 自动累积 | 从未被任何节点读取 |
| MemorySaver | 每次 `run()` 新建 = 新对话 | 设计正确，无需修改 |
| Planner | 只读 `original_query` | 无法感知指代/追问 |
| Answer | 只读 `merged_results` | 回答不衔接历史 |

---

## 架构变更

### 最终图结构

```
START
  │
  ▼
alias_resolve        ← 别名 + 实体解析（不变）
  │
  ▼
history_extractor    ← 新增: 从 messages 提取最近 N 轮对话
  │
  ▼
context_builder      ← 新增: 构建 ConversationContext + 解析指代
  │
  ▼
planner              ← 改为消费 context
  │
  ...（query_processing → knowledge_retrieval → experts → merge 不变）
  │
  ▼
answer               ← 改为消费 context，衔接历史；更新 recent_entities + previous_intent
  │
 END
```

### 数据流

```
messages ──→ history_extractor ──→ raw_history: [{user, assistant}]
                                        │
                                        ▼
                                  context_builder ──→ context: ConversationContext {
                                                        history: [{user, assistant}],
                                                        recent_entities: [{name, type}],
                                                        current_topic: str,
                                                        is_followup: bool,
                                                        resolved_query: str,
                                                        previous_intent: str,
                                                      }
                                        │
                          ┌─────────────┘
                          ▼
                      planner ──→ 消费 context，followup 仅作参考
                          │
                          ▼
                      ... pipeline ...
                          │
                          ▼
                      answer ──→ 消费 context，更新 recent_entities + previous_intent
```

### 关键设计决策（来自评审）

| 决策 | 理由 |
|------|------|
| `recent_entities: [{name, type}]` 替代 `last_anime` | 覆盖"刚才那两部""第三部"等多实体指代 |
| 推荐列表从 merge_results 提取，不从 Answer 解析 | 避免反向解析自然语言的不确定性 |
| Planner 中 followup 仅作提示，不覆盖判断 | "推荐 JOJO → 评分最高是哪个" 最后一句意图已变 |
| ContextResolver 不依赖 MetadataIndex | 番剧识别由 alias/entity_resolver 负责 |
| ConversationContext 用 TypedDict | 类型清晰，可维护 |

---

## 改动清单

### 1. `agents/state.py` — 新增字段

```python
from typing import TypedDict, Annotated, List
# ... 现有 imports ...

class ConversationContext(TypedDict):
    """对话上下文 — 由 context_builder 生成，Planner/Answer 消费"""
    history: list[dict]             # 最近 N 轮: [{user: str, assistant: str}]
    recent_entities: list[dict]     # 最近讨论的实体: [{name: str, type: str}]
    current_topic: str              # 当前话题
    is_followup: bool               # 是否为追问
    resolved_query: str             # 指代解析后的查询
    previous_intent: str            # 上一轮意图: recommend | fact | compare | chat

class AgentState(TypedDict):
    # ... 现有字段保持不变 ...

    # ── 对话上下文 (v1.1) ──
    context: ConversationContext                    # 当前轮上下文
    recent_entities: list[dict]                     # 持久化: 最近讨论的实体 [{name, type}]
    previous_intent: str                            # 持久化: 上一轮意图
```

### 2. `agents/history_extractor.py` — 新建 History Extractor 节点

**职责单一**: 从 `messages` 中提取最近 N 轮对话，不做任何业务推理。

```python
# agents/history_extractor.py

async def history_extractor_node(state: AgentState) -> dict:
    """从 messages 中提取最近 N 轮对话"""
    messages = state.get("messages", [])
    n = config.MEMORY_MAX_ROUNDS

    rounds = _extract_recent_rounds(messages, n)
    # rounds: [{user: str, assistant: str}, ...]

    return {"context": {"history": rounds}}
```

### 3. `agents/context_builder.py` — 新建 Context Builder 节点

**职责**: 基于 raw_history + state 中的结构化字段，生成完整 `ConversationContext`，包括指代解析。

```python
# agents/context_builder.py
import re
from agents.state import AgentState, ConversationContext
import config

def _detect_followup(query: str) -> bool:
    """检测追问/指代模式"""
    patterns = [
        r"^(它|他|她|这个|那个|这部|那部|这|那)",    # 代词开头
        r"^(还有|还有吗|还有呢|再|继续|再来)",       # 追问
        r"^(那|那么|那.*呢)",                        # 衔接追问
        r"^(和|跟|与).*(比|对比|区别)",               # 对比追问
    ]
    return any(re.match(p, query) for p in patterns)

def _resolve_reference(query: str, entities: list[dict]) -> str:
    """解析指代: "它的评分" + [{name:"JOJO"}] → "JOJO的评分"
       序号指代: "第二部的评分" + [{name:"JOJO"}, {name:"巨人"}] → "巨人的评分"
    """
    # 序号指代: 第一部, 第二个, 第三个...
    ordinal_map = {"一": 0, "第一个": 0, "第一部": 0,
                   "二": 1, "第二个": 1, "第二部": 1,
                   "三": 2, "第三个": 2, "第三部": 2}
    for word, idx in ordinal_map.items():
        if word in query and idx < len(entities):
            return query.replace(word, entities[idx]["name"])
    
    # 代词指代
    pronouns = ["它", "他", "她", "这个", "那个", "这部", "那部", "这", "那"]
    for p in pronouns:
        if query.startswith(p) and entities:
            rest = query[len(p):] if query.startswith(p) else query
            return entities[0]["name"] + rest

    return query

async def context_builder_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    query = state.get("original_query", "") or (
        messages[-1].content if messages else ""
    )
    context = state.get("context", {})
    history = context.get("history", [])

    is_followup = _detect_followup(query) if history else False

    # 指代解析（不依赖 MetadataIndex，只用已有 entity 信息）
    resolved = query
    if is_followup:
        entities = state.get("recent_entities", [])
        resolved = _resolve_reference(query, entities)

    # 推断当前话题
    current_topic = _infer_topic(query)

    result: ConversationContext = {
        "history": history,
        "recent_entities": state.get("recent_entities", []),
        "current_topic": current_topic,
        "is_followup": is_followup,
        "resolved_query": resolved,
        "previous_intent": state.get("previous_intent", ""),
    }

    return {"context": result, "resolved_query": resolved}
```

### 4. `agents/graph.py` — 插入两个新节点

```python
from agents.history_extractor import history_extractor_node
from agents.context_builder import context_builder_node

# 注册节点
g.add_node("history_extractor", history_extractor_node)
g.add_node("context_builder", context_builder_node)

# 边变更
g.add_edge(START, "alias_resolve")
g.add_edge("alias_resolve", "history_extractor")     # 新增
g.add_edge("history_extractor", "context_builder")   # 新增
g.add_edge("context_builder", "planner")             # 修改

# expert_input 补全 context（让 Expert 也能感知上下文）
expert_input = {
    # ... 现有字段 ...
    "context": state.get("context", {}),
}
```

### 5. `agents/planner.py` — 消费 ConversationContext

**改动**:
- `planner_node` 读取 `state["context"]`
- 在 LLM prompt 中注入对话历史文本
- **followup 仅作参考信号**，不直接覆盖 planner 的分类结果

```python
async def planner_node(state: dict) -> dict:
    query = state.get("original_query", "")
    context = state.get("context", {})
    
    # 构建历史文本（注入 LLM prompt）
    history_text = ""
    if context.get("history"):
        lines = []
        for r in context["history"]:
            lines.append(f"用户: {r['user']}")
            lines.append(f"助手: {r['assistant']}")
        history_text = "\n".join(lines)
    
    # 在 LLM prompt 中追加历史
    # ... 调用 plan() 时传入 history_text ...
```

Prompt 追加片段:

```
## 对话历史（仅供参考，用于理解指代和上下文）
{history_text}

注意: 即使有历史，仍需独立判断当前查询的真实意图。followup 不代表延续上一轮的 query_type。
```

### 6. `agents/answer.py` — 消费 Context + 更新实体状态

**改动**:
- 读取 `context`，在 system prompt 追加对话上下文
- 回答后从 **merge_results**（不是从回答文本）中提取推荐作品名，更新 `recent_entities`
- 更新 `previous_intent`

```python
async def answer_node(state: dict) -> dict:
    # ... 现有逻辑 ...

    # 更新对话状态（来源: merge_results，不是 LLM 输出）
    merged_results = state.get("merged_results", "")
    plan = state.get("plan", {})

    result = {"messages": [resp]}
    
    # 从 merge_results 提取推荐作品（结构化来源，可靠）
    recent = _extract_recent_from_merged(merged_results)
    if recent:
        result["recent_entities"] = recent
    
    result["previous_intent"] = plan.get("query_type", "")
    return result

def _extract_recent_from_merged(merged: str) -> list[dict]:
    """从 merge_results 中提取作品名（merge 节点已标注番剧名和元数据）"""
    # merge_results 格式: "**命运石之门**（评分8.7）..." 
    # 用正则提取 **粗体** 内的番剧名
    import re
    names = re.findall(r"\*\*(.+?)\*\*", merged)
    entities = []
    for name in names[:5]:
        if len(name) <= 30 and not any(kw in name for kw in ["推荐", "分析", "总结"]):
            entities.append({"name": name, "type": "anime"})
    return entities
```

### 7. `config.py` — 新增配置

```python
# ── Short-term Memory ──
MEMORY_MAX_ROUNDS = int(os.getenv("MEMORY_MAX_ROUNDS", "5"))
```

### 8. `.env.example` — 新增配置说明

```bash
# ── Short-term Memory ──────────────────────────────────────────
MEMORY_MAX_ROUNDS=5
```

### 9. `tests/test_agent.py` — MemorySaver 生命周期修正

将 `MemorySaver()` 提升到交互循环外部，使同一会话共享 checkpointer:

```python
_memory = MemorySaver()

async def ask(query: str):
    g = build_graph()
    resp = await g.compile(checkpointer=_memory).ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "interactive"}}
    )
    ...
```

---

## 不在此次范围的项

| 项 | 原因 |
|---|------|
| AgentState 拆分子 State | 范围太大，后续重构 |
| 自动摘要压缩 | 5 轮对话 token 可控 |
| Reset 能力 | 切换 thread_id 即可 |
| Expert prompt 改造 | context 通过 expert_input 传递即可 |

---

## 文件改动汇总

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `main.py` | 无需改动 | 0 |
| `tests/test_agent.py` | MemorySaver 提升到 ask() 外部 | ~3 行 |
| `agents/state.py` | 新增 ConversationContext TypedDict + 2 字段 | ~15 行 |
| `agents/history_extractor.py` | **新建** | ~40 行 |
| `agents/context_builder.py` | **新建** | ~80 行 |
| `agents/graph.py` | 插入 2 个节点 + 更新边 + expert_input | ~15 行 |
| `agents/planner.py` | 消费 context + prompt 扩展 | ~25 行 |
| `agents/answer.py` | 消费 context + 从 merge 提取实体 + 状态更新 | ~30 行 |
| `config.py` | 新增 MEMORY_MAX_ROUNDS | ~2 行 |
| `.env.example` | 新增配置说明 | ~2 行 |

---

## 验证方式

1. **单轮兼容**: "推荐科幻番" → 正常返回，无报错
2. **代词指代**: "推荐类似 JOJO 的番" → "它的评分是多少" → 回答 JOJO 的评分
3. **序号指代**: "推荐热血番"（返回 A、B、C）→ "第二部的评分" → 回答 B 的评分
4. **追问省略**: "京都动画有哪些作品" → "还有吗" → 继续推荐京都动画
5. **意图切换**: "推荐 JOJO" → "评分最高是哪个" → 正确切换到 fact 查询，不误判为推荐追问
6. **话题切换**: "推荐热血番" → "你好" → 识别为闲聊，不走检索
