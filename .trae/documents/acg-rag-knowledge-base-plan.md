# ACG 番剧推荐 RAG 知识库搭建计划（WebBaseLoader 版）

## 摘要

基于现有 LangGraph Multi-Agent 项目，使用 LangChain 内置的 `WebBaseLoader` 直接从 Bangumi 和 Bilibili 网页拉取番剧内容，分块后向量化导入 Pinecone。省去自定义爬虫层，用最少的代码搭建 ACG 番剧推荐 RAG 知识库。

## 当前状态分析

| 模块 | 现状 | 需要改动 |
|------|------|---------|
| `tools/rag.py` | 已实现 MMR 检索，从 Pinecone 读取 | **不改** |
| `config.py` | PINECONE 配置齐全 | **不改** |
| `llms.py` | Qwen text-embedding-v4 | **不改** |
| `nodes/router.py` | 通用路由 prompt | **需更新** 加入番剧推荐关键词 |
| `nodes/answer.py` | 通用回答 prompt | **需更新** 为推荐场景定制 |
| `requirements.txt` | 已有 langchain-community（含 WebBaseLoader） | **不改** |
| Pinecone Index `vector` | 已建好 | **需确认维度** 1024（text-embedding-v4） |

## 数据流设计（WebBaseLoader 版）

```
┌──────────────────────────────────────────┐
│  data/urls.txt                           │
│  https://bgm.tv/subject/265              │
│  https://bgm.tv/subject/876              │
│  https://www.bilibili.com/bangumi/...    │
│  ... (500 条)                            │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  data/build_kb.py                        │
│                                          │
│  ① WebBaseLoader(urls).load()            │
│     └→ 批量加载网页 → LangChain Document  │
│                                          │
│  ② RecursiveCharacterTextSplitter        │
│     chunk_size=800, overlap=100          │
│                                          │
│  ③ PineconeVectorStore.from_documents()  │
│     向量化 (Qwen v4) + 上传             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │   Pinecone   │
            │  vector 索引  │
            └─────────────┘
```

## 详细实施步骤

### 步骤 1：准备 URL 列表 `data/urls.txt`

收集 Bangumi 高分番剧页面的 URL，一行一个。后续可手动扩展。

**Bangumi URL 格式**：`https://bgm.tv/subject/{id}`（服务端渲染，WebBaseLoader 可直接解析）

**Bilibili 番剧页面 URL 格式**：`https://www.bilibili.com/bangumi/media/md{media_id}/`（部分内容 SSR 可解析）

**获取 URL 列表的方式**：
- 手动从 Bangumi 排行榜页面复制高分番剧链接
- 或者用 Bangumi API 快速获取 ID 列表后拼接 URL（一次性脚本）

示例 `data/urls.txt`：
```
https://bgm.tv/subject/265
https://bgm.tv/subject/876
https://bgm.tv/subject/110467
```

### 步骤 2：创建 `data/build_kb.py`

只需一个文件，三件事：加载网页 → 分块 → 上传 Pinecone。

```python
"""构建 ACG 番剧知识库 — WebBaseLoader → 分块 → Pinecone"""
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from llms import embeddings
import config


def load_urls(path="data/urls.txt") -> list[str]:
    """读取 URL 列表"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def build_knowledge_base():
    # 1. 批量加载网页
    urls = load_urls()
    print(f"Loading {len(urls)} pages...")

    all_docs = []
    batch_size = 20  # 每批 20 个，避免内存/网络压力
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        loader = WebBaseLoader(
            web_paths=batch,
            header_template={"User-Agent": "ACG-RAG-Bot/1.0"},
            requests_per_second=2,  # 控制请求频率
        )
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  Loaded {i + len(batch)}/{len(urls)} pages, {len(all_docs)} docs so far")
        time.sleep(1)

    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. 上传 Pinecone（id 去重：重复执行自动覆盖）
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX,
        pinecone_api_key=config.PINECONE_API_KEY,
    )
    print(f"Done! Uploaded {len(chunks)} chunks to Pinecone.")


if __name__ == "__main__":
    build_knowledge_base()
```

**关键设计点**：
- **分批加载**：20 个 URL 一批，避免一次性请求 500 个网页
- **限速**：`requests_per_second=2`，不压垮目标网站
- **自动去重**：`from_documents` 按 URL 生成唯一 id，重复执行自动覆盖

### 步骤 3：更新 Router Prompt

修改 [nodes/router.py](file:///d:/Users/ASUS/Desktop/学习资料/agent/代码/langgraph-multi-agent/nodes/router.py) 中的 `_PROMPT`：

```python
_PROMPT = """Classify the query into one of:
- rag        : anime recommendation, "推荐", "番剧", "动漫", "类似", "求推",
               "好看", "补番", genre queries, "什么番", "有没有", "找番",
               any question about specific anime, characters, or ACG content
- web_search : latest, news, today, current season, new anime, 新番, 本季
- direct     : greetings, general knowledge not about anime, simple questions

When the user asks for anime recommendations → rag.
When asking about currently airing / latest anime news → web_search."""
```

### 步骤 4：更新 Answer Prompt

修改 [nodes/answer.py](file:///d:/Users/ASUS/Desktop/学习资料/agent/代码/langgraph-multi-agent/nodes/answer.py) 中的 `_PROMPT`：

```python
_PROMPT = """You are an ACG anime recommendation expert.
- Chinese query → reply in Chinese | Roman Urdu → Roman Urdu | English → English
- When recommending anime from tool results:
  * List 3-5 recommendations with title, rating, and one-line reason
  * Explain why each matches the user's preferences
  * Mention tags/genres that match the user's request
- Plain text only — no markdown formatting
- Be enthusiastic but concise, 5-8 lines max
- If tool result is empty, say you don't have enough data for that query"""
```

### 步骤 5：目录结构变更

```
langgraph-multi-agent/
├── data/                          # 🆕 新增目录
│   ├── urls.txt                  # 🆕 番剧页面 URL 列表
│   └── build_kb.py               # 🆕 知识库构建脚本（唯一脚本）
├── nodes/
│   ├── router.py                 # ✏️ 更新 prompt
│   └── answer.py                 # ✏️ 更新 prompt
├── (其他文件不变)
└── requirements.txt              # 不改（langchain-community 已有 WebBaseLoader）
```

**无需新增任何依赖** —— `WebBaseLoader` 已在 `langchain-community` 中（项目已依赖）。

## 与旧方案对比

| 维度 | 旧方案（API 爬虫） | 新方案（WebBaseLoader） |
|------|-------------------|------------------------|
| 代码量 | ~300 行（3 个文件） | **~50 行（1 个文件）** |
| 新增依赖 | beautifulsoup4、lxml | **0 个** |
| 数据结构 | 需自己定义 JSON schema | **LangChain Document 原生** |
| 容错处理 | 需手写重试/降级逻辑 | **WebBaseLoader 内置** |
| 可维护性 | 需维护 API 字段映射 | **直接改 URL 列表即可** |
| JS 渲染页面 | 走 API 规避 | **仅限 SSR 页面**（Bangumi 支持好） |

## 验证步骤

1. **小批量测试**（先试 5 个 URL）：
   ```bash
   # 在 data/urls.txt 中只放 5 行
   python data/build_kb.py
   ```

2. **全量构建**：
   ```bash
   # 放满 500 条 URL 后执行
   python data/build_kb.py
   ```

3. **端到端验证**：
   ```bash
   python main.py
   # 输入: "推荐一部类似命运石之门的科幻番"
   # 期望: Router → rag → Pinecone 检索 → 推荐回复
   ```

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| B站页面 JS 渲染，WebBaseLoader 取不到内容 | B站 URL 作为补充，主力靠 Bangumi 的 SSR 页面 |
| Bangumi 页面含大量导航/页脚噪音 | RecursiveCharacterTextSplitter 切分后，噪音分散到不同 chunk，检索时 MMR 自然会优先匹配正文 chunk |
| text-embedding-v4 调用量 | 500 页 × ~3 chunks/页 = 1500 次，免费额度内 |
| 网络不稳定 | 分批加载 + 打印进度，断点可继续 |
