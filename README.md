# 🎬 AniGraph — ACG 番剧智能推荐系统

基于 LangGraph 的多 Agent 协作 ACG 番剧推荐系统，支持自然语言查询、智能推荐、对比分析，附带 **Web Trace 实时执行面板**。

```
用户: "无职转生的主角是谁？"
AniGraph: 鲁迪乌斯·格雷拉特，声优是内山夕实。原作是理不尽な孫の手写的轻小说...
```

---

## ✨ 特性

- **多 Agent 协作**: Planner 编排 → Expert 并行推理 → 融合回答
- **Embedding 粗筛 + LLM 精分类**: 双层意图分类，先过滤不相关类别再精确判断
- **复杂度感知路由**: LLM 分析查询复杂度，智能决定是否需要多查询扩展
- **按需节点调度**: alias_resolve 和 web_fallback 按需加入，减少不必要 LLM 调用
- **Hybrid RAG**: Metadata Index（结构化）+ Pinecone（语义）+ Whoosh（关键词）三路检索
- **Simple Fact 快速通道**: 简单事实查询单次 LLM 直接回答，跳过 Expert 流水线
- **对话上下文感知**: 支持多轮追问、指代消解（"它的评分""第二部呢"）
- **ToolRegistry**: 统一工具注册表，12 个工具集中管理、懒加载、开关控制
- **LLM 健壮性**: Structured Output 自动降级 + 指数退避重试
- **Web Trace 面板**: SSE 实时推送执行过程——聊天气泡 + 流程图，每个节点的 LLM Token 用量一目了然

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
# 编辑 .env 填入 LLM_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY
```

### 2. 构建知识库

```bash
python data/build_kb.py --metadata-only    # 构建 Metadata Index
python data/build_kb.py --whoosh-only      # 构建 Whoosh 索引
python data/build_kb.py                    # 构建 Pinecone 向量库
```

### 3a. Web Trace 面板（推荐）

```bash
python server.py
# 打开 http://localhost:9527
```

左侧聊天面板输入查询，右侧实时显示流程图——每个节点的执行状态、耗时、LLM Token 用量流式推送。

### 3b. 命令行交互

```bash
python chat.py
```

### 3c. 单次查询

```bash
python main.py
```

---

## 🏗️ 架构

```
用户查询
  │
  ├── (按需) ──→ alias_resolve     ← 别名/实体解析（番剧别名 → 正名，角色/梗识别）
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
                  └── 复杂查询 ──→ [metadata_reasoner || similar_expert]（并行）
                                       │
                                       └──→ merge → answer → END
```

---

## 📡 Web Trace 面板

| 功能 | 说明 |
|------|------|
| SSE 实时推送 | 节点开始/结束、LLM Token 用量、回答文本流式传输 |
| 聊天气泡 | 左栏对话式 UI，打字机流式输出 |
| 流程图 | 右栏显示完整执行链路，每个节点标注耗时和 LLM 调用情况 |
| 节点详情 | 点击节点查看 State 变化、LLM 调用明细 |
| 模型选择 | 下拉切换 LLM 模型（deepseek-v4-pro / deepseek-v4-flash） |

**启动**: `python server.py` → http://localhost:9527

**API 端点**:
- `GET /` — Web 面板
- `GET /chat/stream?query=...&thread_id=...` — SSE 流式查询
- `GET /api/models` — 可用模型列表
- `GET /api/health` — 健康检查

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
│   ├── planner.py         # LLM 意图分类 + ExecutionPlan
│   ├── simple_fact_answer.py  # 快速通道回答
│   ├── answer.py          # 最终回答生成
│   ├── metadata_reasoner.py   # 元数据推理 Expert
│   ├── similar_expert.py      # 相似推荐 Expert
│   ├── merge.py           # 结果合并去重
│   ├── graph.py           # LangGraph 图定义
│   └── ...
├── tools/                 # 检索工具
│   ├── registry.py        # ToolRegistry 统一工具注册表
│   ├── knowledge_retrieval.py  # Hybrid RAG 管线
│   ├── rag_optimizer.py        # 查询优化
│   └── web_search.py           # Tavily 联网搜索
├── docs/                  # 文档
│   ├── architecture.md    # 架构文档
│   └── api_reference.md   # API 参考
└── tests/                 # 测试
```

---

## 🔧 技术栈

| 层级 | 技术 |
|------|------|
| 编排框架 | LangGraph 0.3+ |
| 主 LLM | DeepSeek-V4-Pro |
| 轻量 LLM | DeepSeek-V4-Flash |
| Web Server | FastAPI + uvicorn + SSE |
| 向量检索 | Pinecone (MMR) |
| 稀疏检索 | Whoosh (BM25F) |
| 精排 | bge-reranker-v2-m3 |
| 嵌入模型 | Qwen3-Embedding-0.6B (本地 CPU) |
| 结构化索引 | JSON MetadataIndex |
| 联网搜索 | Tavily |

---

## 📖 更多文档

- [架构文档](docs/architecture.md) — 节点详解、State Schema、路由逻辑
- [API 参考](docs/api_reference.md) — 完整 API 文档
- [项目总结](.trae/documents/project-summary.md) — 设计决策、经验教训、Future Work
