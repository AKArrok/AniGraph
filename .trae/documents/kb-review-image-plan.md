# 知识库扩展 + 评论集成 + 识图功能 实施计划

## 总览

三个增强任务：
1. **扩展知识库** — 爬取 Bangumi 2010年以后的番剧（当前 Top 500 含大量老番）
2. **评论数据集成** — Bangumi 热门吐槽 + B站热门评论，作为推荐引用来源
3. **动漫识图** — 调用 trace.moe API，根据截图识别番剧

---

## 一、当前架构分析

| 模块 | 文件 | 现状 |
|------|------|------|
| 知识库构建 | `data/build_kb.py` | 从 Bangumi 排行榜拉 Top 500，无条件过滤年份 |
| URL 源 | `data/urls.txt` | 500 条，包含 1995 年 EVA 等老番 |
| 评论 | 无 | 知识库文档不含任何评论/吐槽数据 |
| 识图 | 无 | 无图像处理能力 |
| 工具列表 | `graph.py` L42 | `[RAG, search_web]` |
| 工具调用 prompt | `nodes/llm_tool.py` | 只提到 RAG 和 search_web |
| State | `state.py` | `DimensionDecision` description 缺少 studio/writer |

---

## 二、任务 1：扩展知识库（2010+）

### 2.1 问题
- `fetch_bangumi_urls()` 按 `sort=rank` 拉 Top 500，包含大量 2005 年前的经典（EVA、星际牛仔、攻壳等）
- Bangumi `/v0/subjects` 返回的每条数据含 `date` 字段（如 `"2009-04-02"`）

### 2.2 方案

修改 `data/build_kb.py` 的 `fetch_bangumi_urls()`：

1. 增大拉取量（offset 循环到 2000+ 条）
2. 在循环内用 `item.get("date", "")` 过滤，只保留 `>= "2010"` 的番剧
3. 收集到目标数量（如 500 条）后停止
4. 保留现有 `--fetch` 参数行为

```python
def fetch_bangumi_urls(path="data/urls.txt", count=500, min_year="2010"):
    urls = []
    offset = 0
    while len(urls) < count:
        ...
        for item in data:
            date = item.get("date", "")
            if date and date >= min_year:
                urls.append(...)  # 仅添加 >= min_year 的
        offset += 50
```

### 2.3 修改文件
- **`data/build_kb.py`** L28-64：`fetch_bangumi_urls()` 添加年份过滤逻辑
- **`data/urls.txt`**：重新生成（仅 2010+ 番剧）

### 2.4 验证
- `python data/build_kb.py --fetch` 生成新的 urls.txt
- 检查 urls.txt 中最老的不早于 2010 年

---

## 三、任务 2：评论数据集成

### 3.1 数据来源

| 来源 | API | 内容 |
|------|-----|------|
| Bangumi 讨论 | `GET /v0/subjects/{id}/topics` | 热门吐槽帖（按回复数排序，取前 5 条） |
| B站长评 | `https://api.bilibili.com/pgc/review/{media_id}/list?ps=5` | B站番剧页面的用户长评 |

### 3.2 数据获取流程

**Bangumi 评论：**
1. 为每部番剧调用 `GET /v0/subjects/{id}/topics?limit=10`
2. 按 `replies` 数量排序，取 Top 5
3. 对每个 topic 调用 `GET /v0/subjects/{id}/topics/{topic_id}` 获取正文
4. 格式化为：`【Bangumi热评】@{username}: {content[:200]}`

**B站评论：**
1. 用番剧名称搜索 B站：`GET https://api.bilibili.com/x/web-interface/search/type?search_type=media_bangumi&keyword={动画名}`
2. 从搜索结果提取 `media_id`
3. 调用 `GET https://api.bilibili.com/pgc/review/{media_id}/list?ps=5` 获取长评
4. 格式化为：`【B站热评】@{username} (点赞{likes}): {content[:200]}`

### 3.3 集成方式

**新增文件 `data/fetch_reviews.py`** — 独立的评论采集脚本：
- 读取 `urls.txt`
- 为每个 subject_id 拉取 Bangumi 评论
- 可选：拉取 B站评论（通过搜索匹配）
- 输出 JSON 缓存文件 `data/reviews_cache.json`

**修改 `data/build_kb.py`** — 在构建文档时将评论附加到 `page_content`：
- `build_knowledge_base()` 中加载 `reviews_cache.json`
- 组合内容时追加：`combined = api_text + "\n\n---\n" + page_content[:2000] + "\n\n【热门评论】\n" + reviews_text`

### 3.4 回答中引用评论

**修改 `nodes/answer.py`** — 在 System Prompt 中添加规则：
```
5. When tool results contain 【Bangumi热评】or【B站热评】, cite 1-2 relevant ones:
   "有Bangumi用户评价说：'{quote}'"
```

### 3.5 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `data/fetch_reviews.py` | **新建** | 评论采集脚本 |
| `data/build_kb.py` L213 | **修改** | 文档组合时追加评论 |
| `nodes/answer.py` L6-21 | **修改** | prompt 添加引用评论的规则 |

---

## 四、任务 3：动漫识图功能

### 4.1 技术选型

**trace.moe API**（免费，无需 API Key）：
- 端点：`POST https://api.trace.moe/search`
- 参数：`url`（图片 URL）或 `image`（文件上传）
- 返回：`[{anilist_id, filename, episode, from, to, similarity, video, image}]`
- 限制：约 1000 次/月免费

### 4.2 工具封装

**新建文件 `tools/image_search.py`**：

```python
@tool
def identify_anime_image(image_url: str) -> str:
    """识别动漫截图，返回番剧名称、集数、时间戳。用户发图片时调用此工具。"""
```

### 4.3 工作流集成

**Step 1**: 工具注册
- `tools/__init__.py` 添加 `identify_anime_image`
- `graph.py` L42 的 `tools` 列表加入：`tools = [RAG, search_web, identify_anime_image]`

**Step 2**: llm_tool prompt 更新
- `nodes/llm_tool.py` `_SYSTEM` 添加：
```python
"- 用户发图片/截图要识别动漫 → 调用 identify_anime_image 识图\n"
"- 识图结果出来后如果需要详细信息 → 再调用 RAG 检索"
```

**Step 3**: 流程说明
```
用户: "这是什么动漫？" + 图片URL
  → domain_router → genre/general 维度 → llm_tool
  → LLM 判断: 有图片URL → 调用 identify_anime_image
  → tools 节点执行 → 返回识图结果 (ToolMessage)
  → answer 节点: 结合识图结果 + RAG 补充信息 → 最终回答
```

### 4.4 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `tools/image_search.py` | **新建** | trace.moe API 封装 |
| `tools/__init__.py` | **修改** | 导出新工具 |
| `graph.py` L42 | **修改** | tools 列表添加 identify_anime_image |
| `nodes/llm_tool.py` L7-11 | **修改** | _SYSTEM 添加识图使用场景 |

---

## 五、附带修复

### state.py 维度描述更新

`DimensionDecision` 的 Field description 落后于实际：
```python
# 当前（过时）
dimensions: list[str] = Field(description="... rating/genre/director/seiyuu/similar/general")
# 修正后
dimensions: list[str] = Field(description="... rating/genre/director/studio/writer/seiyuu/similar/general")
```

---

## 六、实施顺序

| 步骤 | 任务 | 依赖 | 预估影响范围 |
|------|------|------|-------------|
| 1 | 修复 `state.py` description | 无 | 1 行 |
| 2 | 扩展 `fetch_bangumi_urls()` 加年份过滤 | 无 | `build_kb.py` 1 个函数 |
| 3 | 新建 `data/fetch_reviews.py` | 无 | 新文件 |
| 4 | 修改 `build_kb.py` 集成评论 | 步骤 3 | 1 处追加 |
| 5 | 重新构建知识库（--fetch + build） | 步骤 2,4 | 需运行脚本 |
| 6 | 新建 `tools/image_search.py` | 无 | 新文件 |
| 7 | 注册识图工具（graph.py + __init__.py + llm_tool.py） | 步骤 6 | 3 处小改 |
| 8 | 更新 `nodes/answer.py` prompt 引用评论 | 无 | prompt 文本 |
| 9 | 交互测试验证 | 全部 | `tests/test_agent.py` |

---

## 七、风险与假设

| 风险 | 缓解措施 |
|------|----------|
| Bangumi API 不可达（之前遇到过） | 提供降级方案：仅用已缓存数据 + 错误重试 |
| B站搜索匹配不到 media_id | 只对匹配成功的添加 B站评论，失败跳过不阻塞 |
| trace.moe 免费额度用尽 | 提示用户配置自己的 API Key 或等待额度重置 |
| 知识库重建耗时（评论采集大量请求） | 评论采集独立为可选步骤，基础 KB 构建不变 |
| Pinecone 仍不可用（401） | 降级到 FAISS 本地存储（已有代码） |

## 八、验证方式

1. `python data/build_kb.py --fetch` → 确认 urls.txt 无 2010 年前番剧
2. `python data/fetch_reviews.py` → 确认生成 reviews_cache.json
3. `python data/build_kb.py` → 确认知识库文档含评论段落
4. `python tests/test_agent.py` → 问"鬼灭之刃评价怎么样"，回答应引用 Bangumi/B站评论
5. 发一张动漫截图 URL → 确认返回番剧名 + 详情
