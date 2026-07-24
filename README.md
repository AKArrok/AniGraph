# AniGraph - ACG 番剧智能检索与推荐系统

基于 LangGraph 的多 Agent 动漫问答系统。它把结构化元数据、向量检索、关键词检索和联网搜索组合在一条动态工作流中，支持事实查询、推荐、对比、多轮追问和动漫截图识别，并提供 Web Trace 实时执行面板。

```
用户: "无职转生的主角是谁？"
AniGraph: 鲁迪乌斯·格雷拉特，声优是内山夕实。原作是理不尽な孫の手写的轻小说...
```

---

## ✨ 特性

- **多 Agent 协作**: Planner 编排 -> Expert 按计划并行或串行推理 -> 证据评估 -> 融合回答
- **Embedding 粗筛 + LLM 精分类**: 双层意图分类，先过滤不相关类别再精确判断（含预检缓存）
- **复杂度感知路由**: LLM 分析查询复杂度，智能决定是否需要多查询扩展
- **按需节点调度**: alias_resolve 和 web_fallback 按需加入，减少不必要 LLM 调用
- **Hybrid RAG**: Metadata Index（结构化）+ Pinecone（语义）+ Whoosh（关键词）三路检索
- **Simple Fact 快速通道**: 简单事实查询单次 LLM 直接回答，跳过 Expert 流水线
- **对话上下文感知**: 支持多轮追问、指代消解（"它的评分""第二部呢"），缓存键含 history 防误命中
- **ToolRegistry**: 统一工具注册表，12 个工具集中管理、懒加载、开关控制
- **LLM 健壮性**: Structured Output 自动降级 + tenacity 指数退避重试
- **严格 Parallel 路由**: `ExecutionPlan.parallel` 决定多 Expert 使用 Send 并行还是按计划顺序串行；串行后项可读取前项结论
- **Evaluator-Replan 闭环**: Merge 后先评估证据质量，失败时最多重规划一次，再按需联网或带不确定性回答
- **共享 Prompt 组件**: `agents/prompts.py` 统一禁止清单和上下文构建，避免重复
- **Web Trace 面板**: SSE 实时推送执行过程--聊天气泡 + 流程图，每个节点的 LLM Token 用量一目了然
- **动漫截图识别**: 上传 JPEG/PNG/WebP，优先调用 trace.moe；低置信度时可选用 VLM 补充识别

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repo-url> && cd AniGraph

# 安装依赖（推荐 uv）
uv pip install -r requirements.txt

# 配置 .env
cp .env.example .env
# 编辑 .env，至少填入 LLM、Embedding、Pinecone 和 Tavily 所需密钥
```

默认使用 Ark Coding Plan 的 `doubao-embedding-vision`（1024 维）。如需本地 embedding，将 `EMBEDDING_BACKEND` 改为 `local`；如需 DashScope，将其改为 `dashscope`。切换模型后不得复用维度不一致的 Pinecone 索引。

启动时 `config.validate()` 会检查必填配置。当前实现将 `TAVILY_API_KEY` 也视为必填，即使关闭联网搜索也不能省略。

### 2. 构建知识库

```bash
python data/build_kb.py --metadata-only    # 构建 Metadata Index
python data/build_kb.py --whoosh-only      # 构建 Whoosh 索引
python data/build_kb.py                    # 构建 Pinecone 向量库
```

构建脚本读取 `data/anime_data.db`。向量和 Whoosh 索引共用分块规则，将每部番剧拆成 `profile`、`staff`、`cast` 和 `reviews` 等语义块。断点续跑使用 `python data/build_kb.py --resume`。

### 3a. Web Trace 面板（推荐）

```bash
python server.py
# 打开 http://localhost:9527
```

左侧聊天面板输入查询，右侧实时显示流程图——每个节点的执行状态、耗时、LLM Token 用量流式推送。

面板同时支持文字查询和截图上传。截图识别依赖公网访问 trace.moe；VLM fallback 只有在配置 `VLM_API_KEY` 后才可用。

### 3b. 命令行交互

```bash
python main.py
# 或指定会话
python main.py --session demo
```

### 3c. Python 调用

```bash
import asyncio
from main import run

answer = asyncio.run(run("推荐一部科幻悬疑番", thread_id="demo"))
print(answer)
```

---

## 🏗️ 架构

```
用户查询 / 图片上传
  │
  ├── 有图 ──→ image_recognition  ← trace.moe 识图 + VLM 补充
  ├── 低风险跳过 ─→ alias_skip    ← 跳过别名解析
  └── 按需 ──→ alias_resolve      ← 别名/实体解析（番剧别名 → 正名，角色/梗识别）
  │
  ▼
history_extractor   ← 提取最近 N 轮对话历史
  │
  ▼
context_builder     ← 追问检测 + 指代消解 + 话题推断 + history预拼接
  │
  ▼
planner             ← Embedding粗筛 → LLM意图分类 → 复杂度分析 → ExecutionPlan
  │
  ├── chat ──────────────────────────→ answer（闲聊直达）
  │
  └── 其他 ──→ query_processing      ← 查询优化（direct / rewrite / hyde / decompose）
                  │
                  ▼
           knowledge_retrieval        ← 三路检索（metadata / semantic / mixed）
                  │
                  ├── simple_fact ──→ simple_fact_answer → END（快速通道）
                  └── 复杂查询 ──→ expert_dispatcher
                                       ├── parallel=true  ─→ Send(experts) 并行
                                       ├── parallel=false ─→ experts 按计划顺序串行
                                       └── 单 Expert ─────→ 直接执行
                                                │
                                                ▼
                                              merge
                                                │
                                                ▼
                                            evaluator
                                       ┌────────┼──────────┐
                                       │pass    │replan    │exhausted/fallback
                                       ▼        ▼          ▼
                                web判断/answer  replanner  web判断/降级回答
                                                │
                                                └──→ query_processing（最多一次）
```

`chat` 和 `simple_fact` 保留快速通道，不进入 Evaluator-Replan 闭环。Expert 产物按 `execution_id + attempt` 隔离，重规划后旧轮结果不会进入本轮 Merge。

质量链路通过 `execution_mode`、`quality_trace` 和 `termination_reason` 暴露实际模式、评估问题、计划差异与最终终止原因，日志区分 `quality_pass`、`replan_started`、`replan_exhausted` 和 `web_fallback`。

---

## 📡 Web Trace 面板

| 功能 | 说明 |
|------|------|
| SSE 实时推送 | 节点开始/结束、LLM Token 用量、回答文本流式传输 |
| 聊天气泡 | 左栏对话式 UI，打字机流式输出 |
| 流程图 | 右栏显示完整执行链路，每个节点标注耗时和 LLM 调用情况 |
| 节点详情 | 点击节点查看 State 变化、LLM 调用明细 |
| 模型信息 | 显示当前 LLM / 轻量模型配置（只读） |

**启动**: `python server.py` → http://localhost:9527

**API 端点**:
- `GET /` — Web 面板
- `POST /chat/stream` — SSE 流式对话（主接口）
- `GET /chat/stream?query=...&thread_id=...` — SSE 流式对话（GET 调试版）
- `POST /chat/image` — 截图上传 + SSE 流式输出
- `GET /api/models` — 返回 LLM / Embedding 配置对象
- `GET /api/health` — 健康检查（仅报告进程存活）

---

## 📦 项目结构

```
AniGraph/
├── server.py              # FastAPI Web Trace Server
├── main.py                # 图执行入口 + run_stream()
├── chat.py                # 命令行交互 Chat
├── config.py              # 全局配置
├── llms.py                # LLM / Embedding 实例
├── static/                # 前端静态文件
│   ├── index.html         # Web Trace 面板
│   ├── app.js             # SSE 客户端 + 聊天 + 流程图
│   └── style.css          # 样式
├── trace/                 # Trace 数据采集模块
│   ├── collector.py       # astream_events 事件收集器
│   ├── adapter.py         # LangGraph 事件 → 前端格式适配
│   ├── models.py          # TraceEvent / NodeInfo 等类型定义
│   └── pricing.py         # DeepSeek Token 计价
├── agents/                # Agent 节点
│   ├── planner.py         # LLM 意图分类 + ExecutionPlan（4层路由）
│   ├── image_recognition.py   # 截图识别节点（trace.moe + VLM）
│   ├── prompts.py         # 共享 Prompt 组件（禁止清单/上下文构建）
│   ├── simple_fact_answer.py  # 快速通道回答
│   ├── answer.py          # 最终回答生成
│   ├── metadata_reasoner.py   # 元数据推理 Expert（async）
│   ├── similar_expert.py      # 相似推荐 Expert（async）
│   ├── merge.py           # 结果合并去重
│   ├── evaluator.py       # 当前 attempt 的证据质量评估
│   ├── replanner.py       # 受限执行策略重规划
│   ├── graph.py           # LangGraph 图定义
│   └── ...
├── tools/                 # 检索工具
│   ├── registry.py        # ToolRegistry 统一工具注册表
│   ├── image_search.py    # trace.moe API 客户端
│   ├── knowledge_retrieval.py  # Hybrid RAG 管线
│   ├── rag_optimizer.py        # 查询优化
│   └── web_search.py           # Tavily 联网搜索
├── docs/                  # 文档
│   ├── architecture.md    # 架构文档
│   ├── api.md             # 外部接口文档（HTTP + Python API + TraceEvent）
│   └── api_reference.md   # 内部 Python 模块参考
└── tests/                 # 测试
```

---

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| 编排框架 | LangGraph 1.2+ |
| 主 LLM | DeepSeek-V4-Pro |
| 轻量 LLM | DeepSeek-V4-Flash |
| Web Server | FastAPI + uvicorn + SSE |
| 向量检索 | Pinecone (MMR) |
| 稀疏检索 | Whoosh (BM25F) |
| 精排 | bge-reranker-v2-m3 |
| 嵌入模型 | Ark Coding Plan `doubao-embedding-vision`（默认）/ local Qwen3 / DashScope |
| 结构化索引 | JSON MetadataIndex |
| 联网搜索 | Tavily |

---

## ⚠️ 配置约束

验证逻辑见 `config.validate()` 和 `config.validate_retrieval_settings()`：

- `RETRIEVER_FETCH_K` >= `HYBRID_DENSE_K`（MMR 粗召回池必须覆盖密集检索量）
- `HYBRID_DENSE_K` >= `RETRIEVER_K` 且 `HYBRID_SPARSE_K` >= `RETRIEVER_K`（密集/稀疏检索量须 ≥ 最终返回数）
- `RERANK_TOP_K` >= `RETRIEVER_K`（精排候选数须 ≥ 最终返回数）
- Ark Coding Plan 固定 `doubao-embedding-vision`、1024 维，不得切换模型或维度
- 记忆系统为进程内 `MemorySaver`，重启清空；同一 `thread_id` 共享 in-process 实例
- `/api/health` 仅报告进程存活，不检查 Pinecone / Tavily / LLM 连通性

---

## 🧪 测试

```bash
# 运行测试
pytest -q tests/test_quality_loop.py tests/test_config.py tests/test_retry_system.py
```

> 注意：`tests/test_agent.py` 中的集成测试依赖 `pytest-asyncio`，需额外安装后运行。

---

## 📖 更多文档

- [架构文档](docs/architecture.md) — 节点详解、State Schema、路由逻辑
- [外部接口文档](docs/api.md) — HTTP API、Python API、TraceEvent 结构
- [内部 API 参考](docs/api_reference.md) — 模块级 Python 接口参考
- [项目总结](.trae/documents/project-summary.md) — 设计决策、经验教训、Future Work
