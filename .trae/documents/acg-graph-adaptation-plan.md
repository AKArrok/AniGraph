# ACG 番剧推荐 — Graph 适配计划

> **状态**: 📦 已归档 — 此为旧版 router-based 架构的计划文档。当前系统已升级为 planner-based 多 Agent 协作架构，详见 [project-summary.md](project-summary.md) 和 [short-term-memory-plan.md](short-term-memory-plan.md)。

## 摘要

现有 Graph 结构（Router → LLM Tool → Tools → Answer）无需改动。只需更新 3 个节点和 1 个工具的 Prompt/描述，让 Agent 从"通用助手"变为"ACG 番剧推荐专家"。

## 当前状态分析

### Graph 结构（不改）

```
START → router ──→ llm_tool ──→ tools ──→ answer → END
            │                    │
            └──→ answer ←────────┘ (direct / 超过迭代次数)
```

结构已经正确：
- Router 分发到 `rag` / `web_search` / `direct`
- `rag` / `web_search` → LLM 调用工具 → 获取结果 → 生成回答
- `direct` → 跳过工具直接回答

### 需要改的文件

| 文件 | 当前问题 | 改动内容 |
|------|---------|---------|
| `tools/rag.py` | RAG 工具描述写着"查找上传的文档" | 改为"搜索 ACG 番剧知识库" |
| `state.py` | `file_uploaded` 字段已无实际用途 | **保留不动**（改 State 类型会影响很多地方，且无害） |
| `nodes/router.py` | 路由规则基于"文档/PDF/上传" | 改为基于"番剧/推荐/标签/导演/声优"等 ACG 关键词 |
| `nodes/answer.py` | 通用助手风格，2-4 行 | 改为"番剧推荐专家"风格，列出推荐 + 理由 |
| `nodes/llm_tool.py` | 强制 tool_call 的 SystemMessage | **不改**（逻辑正确，LLM 需要走工具） |
| `graph.py` | **结构完全正确** | **不改** |
| `main.py` | 示例查询是 `"python version check karo"` | 改为 `"推荐一部科幻番"` |

## 各文件具体改动

### 1. `tools/rag.py` — 更新工具描述

```python
# 改前（第9行）
"""Retrieve relevant content from previously uploaded documents."""

# 改后
"""Search the ACG anime knowledge base for recommendations. 
Use for queries about anime recommendations, genres, tags, directors, voice actors, specific shows, or any ACG-related questions.
Input should be a descriptive Chinese/English search query."""
```

### 2. `nodes/router.py` — 更新路由 Prompt

```python
# 改前
_PROMPT = """Classify the query into one of:
- web_search : latest, news, today, current, price, weather, real-time info
- rag        : document, file, PDF, uploaded, "mera data", questions about content
- direct     : greetings, general knowledge, simple questions

When a file has been uploaded and the question might relate to it → rag.
When the question asks about recent/current/live information → web_search."""

# 改后
_PROMPT = """Classify the query into one of:
- rag        : anime recommendations, "推荐番剧", "找番", "求推", "有没有类似",
               genre/tag queries (科幻/热血/治愈/催泪), director/staff queries,
               voice actor queries, "好看的动漫", "补番", "评价", "评分"
- web_search : latest seasonal anime, "新番", "本季", current airing info, news
- direct     : greetings, chitchat, non-ACG questions

When the user asks for anime recommendations or queries about specific anime → rag.
When the user asks about currently airing / latest seasonal anime → web_search.
Otherwise → direct."""
```

同步更新 `router_node` 中传给 LLM 的上下文信息：

```python
# 改前
SystemMessage(content=_PROMPT + f"\nfile_uploaded={state.get('file_uploaded', False)}")

# 改后（file_uploaded 不再有意义，去掉它）
SystemMessage(content=_PROMPT)
```

### 3. `nodes/answer.py` — 更新回答风格

```python
# 改前
_PROMPT = """You are a helpful AI assistant.
- Roman Urdu query → reply in Roman Urdu | English query → reply in English
- Plain text only — no markdown, bullets, or headers
- 2-4 lines max, no filler phrases like "Sure!" or "Great!"
- If tool result provided → explain it clearly in 1-2 lines"""

# 改后
_PROMPT = """You are an ACG anime recommendation expert. Your knowledge comes from Bangumi and Bilibili data.

Response rules:
- Chinese query → reply in Chinese | Roman Urdu → Roman Urdu | English → English
- Plain text only, no markdown formatting

When recommending anime from tool results:
1. List 3-5 recommendations, each with: title, rating, and why it matches
2. Mention matching tags/genres (e.g., "科幻、时间旅行、悬疑")
3. Keep each recommendation to 1-2 lines
4. If the user asked for a specific genre/director/seiyuu, highlight that match

When no tool results are found:
- Say "抱歉，知识库中没有找到符合你要求的番剧，可以换个关键词试试？"

General tone: enthusiastic, knowledgeable, concise. Total 5-12 lines."""
```

### 4. `main.py` — 更新示例查询

```python
# 改前
print(asyncio.run(run("python version check karo")))

# 改后
print(asyncio.run(run("推荐一部类似命运石之门的科幻番")))
```

## 不改动的文件及理由

| 文件 | 理由 |
|------|------|
| `graph.py` | 4 节点结构完全正确，路由逻辑无需变更 |
| `nodes/llm_tool.py` | "Always call a tool" 的逻辑对 RAG + Web 搜索场景仍然适用 |
| `state.py` | `file_uploaded` 字段虽不再使用但保留它无害，改 TypedDict 会牵连多个文件 |
| `config.py` / `llms.py` | 与领域无关 |
| `tools/web_search.py` | Tavily 搜新番信息仍然合理 |

## 改动前后对比

| 维度 | 改前 | 改后 |
|------|------|------|
| RAG 触发词 | "document, file, PDF, uploaded" | "推荐番剧, 找番, 科幻/热血/治愈, 导演, 声优" |
| Web Search 触发 | "latest, news, today, price" | "新番, 本季, 当前播出" |
| 回答风格 | 通用助手 2-4 行 | 番剧推荐专家 5-12 行，含标签匹配 |
| RAG 工具描述 | "查找上传文档" | "搜索 ACG 番剧知识库" |

## 验证步骤

1. **路由测试**：输入不同类型的查询确认路由正确
   ```
   "推荐一部治愈番"        → rag
   "庵野秀明导演的作品有哪些" → rag
   "这季度有什么新番"        → web_search
   "你好"                  → direct
   ```

2. **端到端测试**：
   ```bash
   # 先确保知识库已构建
   python data/build_kb.py
   # 再测试推荐
   python main.py
   ```

3. **多轮对话测试**：利用 MemorySaver 验证上下文记忆
   ```
   Q1: "推荐类似命运石之门的番" → 推荐列表
   Q2: "有没有评分更高一点的" → 基于上一轮上下文筛选
   ```
