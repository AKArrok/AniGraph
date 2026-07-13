# AniGraph — ACG 番剧智能推荐助手

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-orange?style=for-the-badge)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-blue?style=for-the-badge)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> 基于 Hybrid RAG + Multi-Agent 架构的 ACG 番剧智能推荐系统。在约 5,000 部番剧知识库上，通过 LangGraph 编排多个专业化 Agent 协作，支持多轮对话记忆，实现自然语言问答。

</div>

---

## 功能

| 功能 | 描述 |
|------|------|
| 智能推荐 | "推荐类似 JOJO 的番"、"有没有类似命运石之门的科幻番" |
| 事实查询 | "京都动画有哪些作品"、"素晴的评分是多少" |
| 简单事实 | "夏亚是谁"、"逆袭的夏亚评分" — simple_fact 快速通道，单次 LLM 直接回答 |
| 对比分析 | "巨人 vs 鬼灭哪个好看" |
| 多轮追问 | "推荐 JOJO" → "它的评分是多少" → "那和巨人比呢" |
| 角色 / 梗解析 | "夏亚是谁"、"典明粥是什么梗" |
| 昵称识别 | "凉宫" → 凉宫春日的忧郁、"爱马仕" → 偶像大师 |
| 自然闲聊 | 日常对话、感谢、问候等 |

---

## 架构

```
User Query
    │
    ▼
alias_resolve     — 昵称 / 实体解析（字典 → LLM → Web）
    │
    ▼
history_extractor — 从 messages 提取最近 N 轮对话
    │
    ▼
context_builder   — 追问检测 + 指代解析 + 上下文构建
    │
    ▼
planner           — 规则优先分类（metadata / semantic / mixed / chat / simple_fact）
    │
    ▼
query_processing  — 查询优化（Rewrite / HyDE / Decompose）
    │
    ▼
knowledge_retrieval — MetadataIndex（结构化）+ Pinecone MMR（密集）+ Whoosh BM25F（稀疏）
    │                                    → RRF 融合 → CrossEncoder 精排 → 压缩去重
    │
    ├── simple_fact → simple_fact_answer（单次 LLM 直接回答，跳过 Expert）
    │
    ▼
[metadata_reasoner + similar_expert] — 双 Expert 并行推理（Send API）
    │
    ▼
merge             — 去重 + 置信度过滤 + 排序
    │
    ▼
web_fallback      — 低置信度时联网兜底（条件触发）
    │
    ▼
answer_planner    — 随机选择回答结构（零 LLM）
    │
    ▼
answer            — 口语化自然回答（含对话历史衔接）
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| 主 LLM | DeepSeek-V4-Pro（OpenAI 兼容协议） |
| 轻量 LLM | DeepSeek-V4-Flash（OpenAI 兼容协议） |
| Embedding | Qwen3-Embedding-0.6B（本地, CPU） / DashScope API 可选 |
| 向量数据库 | Pinecone |
| 稀疏检索 | Whoosh BM25F（本地） |
| 精排模型 | bge-reranker-v2-m3（CrossEncoder） |
| 联网搜索 | Tavily |
| 会话记忆 | MemorySaver（内存）+ ConversationContext 结构化上下文 |
| 可观测性 | LangSmith / LangFuse（可选） |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

必填的 API Key：

| 变量 | 说明 | 获取地址 |
|------|------|----------|
| `LLM_API_KEY` | DeepSeek / OpenAI 兼容 API Key | [deepseek.com](https://platform.deepseek.com) |
| `PINECONE_API_KEY` | Pinecone 向量数据库 | [pinecone.io](https://pinecone.io) |
| `TAVILY_API_KEY` | Tavily 联网搜索 | [tavily.com](https://tavily.com) |

### 3. 运行

```python
import asyncio
from main import run

# 智能推荐
result = asyncio.run(run("有没有类似命运石之门的科幻番"))

# 事实查询
result = asyncio.run(run("京都动画有哪些作品"))

# 多轮对话（同一 MemorySaver 实例共享记忆）
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
g = build_graph()
app = g.compile(checkpointer=memory)

resp = await app.ainvoke(
    {"messages": [HumanMessage(content="推荐类似 JOJO 的番")]},
    {"configurable": {"thread_id": "1"}}
)
resp = await app.ainvoke(
    {"messages": [HumanMessage(content="它的评分是多少")]},  # 能理解"它"指 JOJO
    {"configurable": {"thread_id": "1"}}
)
```

或直接：

```bash
python main.py

# 交互式多轮测试（支持短期记忆）
python tests/test_agent.py
```

---

## 项目结构

```
AniGraph/
├── main.py                   # 程序入口
├── graph.py                  # 图构建入口（重导出 agents/graph.py）
├── llms.py                   # LLM & Embedding 实例创建（含 request_timeout=60s）
├── config.py                 # 全局配置（读取所有环境变量）
├── requirements.txt          # Python 依赖
├── agents/                   # 多 Agent 核心逻辑
│   ├── graph.py              # LangGraph 图结构构建（核心编排，13 个节点）
│   ├── state.py              # AgentState + ConversationContext 定义
│   ├── planner.py            # 查询规划器（规则优先 + LLM fallback）
│   ├── history_extractor.py  # 历史提取（从 messages 提取最近 N 轮对话）
│   ├── context_builder.py    # 上下文构建（追问检测 + 指代解析）
│   ├── simple_fact_answer.py # 简单事实快速通道（单次 LLM 直接回答）
│   ├── alias.py              # 番剧别名解析（字典 + LLM + Web）
│   ├── entity_resolver.py    # 实体解析（角色/梗 → 番剧）
│   ├── metadata_reasoner.py  # 元数据推理 Expert
│   ├── similar_expert.py     # 相似推荐 Expert
│   ├── merge.py              # Expert 结果合并/去重/排序
│   ├── answer.py             # 最终回答生成
│   ├── web_fallback.py       # 联网搜索回退
│   └── cache.py              # 别名缓存 + 元数据缓存
├── tools/                    # 检索 & 工具层
│   ├── knowledge_retrieval.py # 混合检索（Whoosh + Fusion + Rerank + 压缩）
│   ├── query_processing.py   # 查询分类 + 改写（Rewrite/HyDE/Decompose）
│   ├── rag_optimizer.py      # RAG 全链路门面
│   └── web_search.py         # Tavily 联网搜索封装
├── tests/                    # 测试
│   ├── test_agent.py         # 交互式全链路测试（含耗时统计）
│   └── test_integration.py   # 集成测试
├── data/                     # 知识库数据
├── models/                   # 本地模型文件
└── docs/                     # 文档
```

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LLM_API_KEY` | ✅ | - | LLM API Key（DeepSeek 等 OpenAI 兼容） |
| `LLM_BASE_URL` | - | `https://api.deepseek.com/v1` | LLM API 端点 |
| `PINECONE_API_KEY` | ✅ | - | Pinecone API Key |
| `PINECONE_INDEX` | - | `vector` | Pinecone 索引名 |
| `TAVILY_API_KEY` | ✅ | - | Tavily 联网搜索 Key |
| `LLM_MODEL` | - | `deepseek-v4-pro` | 主 LLM 模型名 |
| `SIMPLE_LLM_MODEL` | - | `deepseek-v4-flash` | 轻量 LLM 模型名 |
| `EMBEDDING_BACKEND` | - | `local` | Embedding 后端：`local` / `dashscope` |
| `ENABLE_RERANKING` | - | `true` | 是否启用 CrossEncoder 精排 |
| `MAX_ITERATIONS` | - | `3` | 最大迭代次数 |
| `RETRIEVER_K` | - | `5` | 最终返回文档数 |
| `MEMORY_MAX_ROUNDS` | - | `5` | 短期记忆保留轮数 |
| `LANGCHAIN_API_KEY` | - | - | LangSmith 追踪（可选） |
| `LANGFUSE_PUBLIC_KEY` | - | - | LangFuse 可观测（可选） |

---

## 设计亮点

| 决策 | 说明 | 收益 |
|------|------|------|
| Planner 规则优先 | 避免每次查询都调 LLM | 80% 查询零 Planner 成本 |
| 三路索引 | Metadata + Dense + Sparse | 覆盖精确/语义/结构化三种查询 |
| 双 Expert 并行 | Send API 并行执行 | 职责清晰，独立优化 |
| RRF 融合 | Dense/Sparse 分数尺度不同 | 规避归一化问题 |
| 本地 Embedding | Qwen3-Embedding-0.6B CPU 运行 | 零 API 配额消耗 |
| 三层实体解析 | 字典 → LLM → Web | 逐级降级，最大化成本效率 |
| Answer 结构随机化 | 避免回答套路化 | 零额外 LLM 成本 |
| **短期对话记忆** | history_extractor + context_builder，零额外 LLM | 支持多轮追问和指代解析 |
| **Simple Fact 快速通道** | 简单查询单次 LLM 直接回答 | 延迟 -57%（136s → 58s） |
| **LLM 超时保护** | `request_timeout=60s` + 节点耗时日志 | 避免 API 卡死，快速定位瓶颈 |

---

## 许可证

MIT
