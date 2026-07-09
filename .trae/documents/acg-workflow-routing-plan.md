# ACG 番剧推荐 — 路由工作流（Workflow）架构改造计划

## 摘要

将现有线性 Graph 改造为**多维度路由工作流**：Router 识别问题涉及的 ACG 维度（评分/类型/导演/声优/相似推荐），动态路由到对应维度的专家 Node，各专家用专属 Prompt 重写检索 Query，调度器根据上下文决定是否需要追加维度，最后由 Answer 综合回答。

## 当前架构 vs 目标架构

### 当前（线性 4 节点）
```
START → router → llm_tool → tools → answer → END
         (rag/web_search/direct 三选一)
```

### 目标（动态路由工作流）
```
                        ┌─────────────────────┐
                        │   domain_router      │  ← 一级：维度识别
                        │   输出: dimensions    │     {rating, genre, director,
                        │   + primary_dim       │      seiyuu, similar, general}
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  rating_expert  │  │  genre_expert   │  │ director_expert │  ...
    │  "高分+类型"    │  │  "类型匹配"     │  │  "导演作品集"   │
    │  重写 query     │  │  重写 query     │  │  重写 query     │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ▼
                        ┌─────────────────────┐
                        │    scheduler        │  ← 二级：动态调度
                        │   是否追加维度？     │     有更多维度 → 回到 experts
                        │   否则 → tools       │     无更多 → tools
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │     llm_tool        │  ← 调用 RAG 检索
                        │     bind_tools(RAG) │
                        └──────────┬──────────┘
                                   ▼
                        ┌─────────────────────┐
                        │       answer        │  ← 综合推荐
                        └─────────────────────┘
```

**核心区别**：
- 旧：一个 router 三选一，直接走工具
- 新：domain_router 输出维度列表 → 按需走多个维度专家 → scheduler 动态决定何时停止 → 统一检索 → 综合回答

## State 改动

```python
# state.py — 新增字段
class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    router_decision: str           # 保留：domain_router 的分类结果
    reasoning:       str           # 保留：分类理由
    iteration_count: int           # 保留：防止死循环
    dimensions:      list[str]     # 🆕 问题涉及的维度: ["rating", "genre"]
    active_dimension: str          # 🆕 当前正在处理的维度
    processed_dims:  list[str]     # 🆕 已处理的维度（避免重复）
    tool_results:    dict          # 🆕 各维度检索结果汇总
```

## 文件变更清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `nodes/domain_router.py` | 多维度分类器：输入用户问题，输出维度列表 |
| `nodes/dimension_experts.py` | 5 个维度专家函数：rating / genre / director / seiyuu / similar |

### 修改文件

| 文件 | 改动 |
|------|------|
| `state.py` | 新增 `dimensions`, `active_dimension`, `processed_dims`, `tool_results` 字段 |
| `graph.py` | 重构为动态路由图：domain_router → experts → scheduler → tools → answer |
| `nodes/router.py` | 简化为兼容层，调用 domain_router |
| `nodes/answer.py` | 更新 prompt：按维度综合推荐 |
| `tools/rag.py` | 更新 tool 描述 |
| `main.py` | 更新演示查询 |

### 不改文件

| 文件 | 理由 |
|------|------|
| `config.py`, `llms.py` | 与架构无关 |
| `nodes/llm_tool.py` | 逻辑不变 |
| `tools/web_search.py` | Tavily 搜索保留，用于新番/实时信息 |

## 各文件详细设计

### 1. `state.py` — 新增字段

```python
class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    router_decision: str
    reasoning:       str
    iteration_count: int
    dimensions:      list[str]     # ["rating", "genre", "director"]
    active_dimension: str          # "rating"
    processed_dims:  list[str]     # ["rating"]
    tool_results:    dict          # {"rating": "...", "genre": "...", ...}
```

### 2. `nodes/domain_router.py` — 多维度分类器（新建）

```python
"""Domain router — 识别问题涉及的 ACG 维度"""

_DIMENSIONS = {
    "rating":   "评分/高分/排名/TOP/神作/经典 相关",
    "genre":    "类型/标签/风格 (科幻/热血/治愈/催泪/恋爱/机战/悬疑/日常)",
    "director": "导演/监督/制作人/制作公司 (庵野秀明/新海诚/京阿尼/P.A.WORKS)",
    "seiyuu":   "声优/CV/配音演员 (花泽香菜/梶裕贵/钉宫理惠)",
    "similar":  "类似/相似/像XX一样的/找和XX风格接近的",
    "general":  "不属于以上维度的一般性问题",
}

class DimensionDecision(BaseModel):
    dimensions: list[str] = Field(description="问题涉及的维度列表，至少一个")
    primary: str = Field(description="最主要/优先处理的维度")
    reasoning: str = Field(min_length=5, max_length=200)

_PROMPT = f"""分析用户问题，识别涉及哪些 ACG 维度：

{chr(10).join(f'- {k}: {v}' for k, v in _DIMENSIONS.items())}

返回涉及的维度列表 + 优先维度。普通问题只返回 ["general"]。"""

async def domain_router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(DimensionDecision).ainvoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {
            "router_decision": r.primary,
            "reasoning": r.reasoning,
            "dimensions": r.dimensions,
            "active_dimension": r.primary,
            "processed_dims": [],
            "tool_results": {},
        }
    except Exception as e:
        return {
            "router_decision": "general",
            "dimensions": ["general"],
            "active_dimension": "general",
            "processed_dims": [],
            "tool_results": {},
        }
```

### 3. `nodes/dimension_experts.py` — 维度专家节点（新建）

```python
"""Dimension expert nodes — 各维度专属的查询重写逻辑"""

_EXPERT_PROMPTS = {
    "rating": "重写为按评分排序的检索 query，重点查评分高且匹配用户偏好的番剧",
    "genre":  "重写为按类型/标签匹配的检索 query，提取用户提到的所有标签关键词",
    "director": "重写为按导演/staff 匹配的检索 query，提取导演或制作公司名称",
    "seiyuu":  "重写为按声优匹配的检索 query，提取声优名称",
    "similar": "重写为语义相似度检索 query，提取用户想找类似作品的番剧名",
    "general": "保持原样，不做改写",
}

async def dimension_expert_node(state: State, active_dim: str):
    """指定维度的专家：用专属 prompt 重写用户查询"""
    prompt = _EXPERT_PROMPTS.get(active_dim, _EXPERT_PROMPTS["general"])
    user_q = state["messages"][-1].content

    resp = await answer_LLM.ainvoke([
        SystemMessage(content=f"你是一个{active_dim}维度的ACG检索专家。{prompt}。只输出改写后的检索query，不要解释。"),
        HumanMessage(content=user_q),
    ])

    # 将改写后的 query 作为新消息
    return {
        "messages": [HumanMessage(content=f"[{active_dim}] {resp.content}")],
    }
```

### 4. `graph.py` — 动态路由图（重构）

```python
"""Graph assembly — 动态路由工作流"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from state import State
from nodes import llm_tool_node, answer_node
from nodes.domain_router import domain_router_node
from nodes.dimension_experts import dimension_expert_node
from tools import RAG, search_web
import config

DIM_NODES = ["rating", "genre", "director", "seiyuu", "similar", "general"]

def _next_expert(state: State) -> str:
    """调度器：选择下一个未处理的维度专家"""
    dims = state.get("dimensions", ["general"])
    processed = state.get("processed_dims", [])
    for d in dims:
        if d not in processed:
            return d
    return "scheduler_done"  # 所有维度处理完 → 进入工具调用

def _mark_and_route(state: State):
    """专家处理后：标记完成 → 决定下一步"""
    active = state.get("active_dimension", "")
    processed = list(state.get("processed_dims", []))
    if active and active not in processed:
        processed.append(active)
    return {"processed_dims": processed}

def _scheduler(state: State):
    """调度器：所有维度处理后 → 进工具；还有未处理 → 继续"""
    dims = state.get("dimensions", ["general"])
    processed = state.get("processed_dims", [])
    remaining = [d for d in dims if d not in processed]
    if remaining:
        return {"active_dimension": remaining[0]}
    return {}

def _after_expert(state: State):
    """专家完成后的路由"""
    active = state.get("active_dimension", "")
    processed = list(state.get("processed_dims", []))
    if active and active not in processed:
        processed.append(active)

    dims = state.get("dimensions", ["general"])
    remaining = [d for d in dims if d not in processed]
    if remaining:
        return "experts"  # 还有维度 → 继续走专家
    return "tools"        # 全部处理完 → 走工具

# 路由条件函数
def _route_to_expert(state: State) -> str:
    active = state.get("active_dimension", "general")
    if active in DIM_NODES:
        return active
    return "general"

def _next_dim_or_tools(state: State):
    # 检查是否还有工具调用（从 llm_tool 出来）
    from langchain_core.messages import AIMessage
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and state.get("iteration_count", 0) < config.MAX_ITERATIONS:
        return "tools"
    return "answer"

def build_graph():
    tools = [RAG, search_web]

    g = StateGraph(State)

    # 注册节点
    g.add_node("domain_router", domain_router_node)
    for dim in DIM_NODES:
        g.add_node(dim, lambda s, d=dim: dimension_expert_node(s, d))
    g.add_node("experts", lambda s: s)  # 聚合节点（透传）
    g.add_node("llm_tool", lambda s: llm_tool_node(s, tools))
    g.add_node("tools", ToolNode(tools))
    g.add_node("answer", answer_node)

    # 边
    g.add_edge(START, "domain_router")

    # domain_router → 根据 primary 路由到对应维度专家
    g.add_conditional_edges("domain_router", _route_to_expert, DIM_NODES)

    # 每个维度专家 → 回到 experts 聚合节点
    for dim in DIM_NODES:
        g.add_edge(dim, "experts")

    # experts → 还有维度？去下一个专家；否则去 llm_tool
    g.add_conditional_edges("experts", _next_expert, DIM_NODES + ["scheduler_done"])
    g.add_edge("scheduler_done", "llm_tool")

    # llm_tool → tools 或 answer
    g.add_conditional_edges("llm_tool", _next_dim_or_tools, ["tools", "answer"])
    g.add_edge("tools", "answer")
    g.add_edge("answer", END)

    return g, tools
```

**关键设计：动态调度**

不是固定 A→B→C 的顺序，而是：
1. domain_router 输出 `dimensions = ["genre", "rating"]`
2. 先走 genre 专家 → 完成后回到 scheduler
3. scheduler 发现还有 rating → 走 rating 专家
4. 完成后 scheduler 发现全处理完 → 进 llm_tool（统一 RAG 检索）
5. 单维度查询就只走一个专家，多维度走多个

### 5. `nodes/answer.py` — 多维度综合回答（修改）

```python
_PROMPT = """You are an ACG anime recommendation expert.

Response rules:
- Chinese query → reply in Chinese
- Plain text only, no markdown

When recommending anime:
1. List 3-5 recommendations, each with: title, rating, and matching reason
2. Highlight which dimensions match (评分/类型/导演/声优)
3. Keep each recommendation concise, 1-2 lines
4. Total 5-12 lines

When no results: "抱歉，知识库中没有找到符合要求的番剧" """
```

## 验证测试计划

```python
# tests/test_workflow.py

TEST_CASES = [
    # (输入, 期望维度)
    ("推荐一部9分以上的科幻番",              ["rating", "genre"]),
    ("有没有类似命运石之门的番",             ["similar"]),
    ("庵野秀明导演过哪些动画",              ["director"]),
    ("花泽香菜配音的治愈番推荐",            ["seiyuu", "genre"]),
    ("京阿尼的高分催泪作品",                ["director", "rating", "genre"]),
    ("你好",                              ["general"]),
]

async def test_domain_router():
    for query, expected_dims in TEST_CASES:
        result = await domain_router_node({"messages": [HumanMessage(content=query)]})
        assert set(expected_dims).issubset(set(result["dimensions"])), \
            f"'{query}': expected {expected_dims}, got {result['dimensions']}"

async def test_workflow():
    g, _ = build_graph()
    app = g.compile()
    for query, _ in TEST_CASES[:3]:
        result = await app.ainvoke({"messages": [HumanMessage(content=query)]})
        assert result["messages"][-1].content, f"No response for '{query}'"
```

## 改动汇总

| 类型 | 文件 | 行数变化 |
|------|------|---------|
| 新建 | `nodes/domain_router.py` | ~50 行 |
| 新建 | `nodes/dimension_experts.py` | ~40 行 |
| 修改 | `state.py` | +4 字段 |
| 修改 | `graph.py` | 重构 ~70 行 |
| 修改 | `nodes/answer.py` | ~15 行 |
| 修改 | `nodes/router.py` | 保持兼容 |
| 修改 | `tools/rag.py` | ~5 行 |
| 修改 | `main.py` | ~3 行 |
| 不删 | `tools/web_search.py` | 保留 |

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 多维度查询导致多次 LLM 调用（延迟↑） | 单维度查询只走 1 个专家；多维度最多 5 个，可加 `max_dims` 限制 |
| 动态路由死循环 | `processed_dims` 去重 + `iteration_count` 上限 |
| 多轮检索 token 超限 | answer 节点只取最后 10 条消息，自动丢弃旧维度结果 |
