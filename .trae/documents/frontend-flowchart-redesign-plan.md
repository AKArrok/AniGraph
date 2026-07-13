# 前端 UI 重设计计划 — 流程图版

## 摘要

将 Trace 面板的瀑布图替换为垂直流程图，大幅放大字号，展示 Agent 执行链路而非逐个节点的条形图。

---

## 用户需求

1. **字号放大** — 当前 150% 缩放才够用，基础字号需加大 ~50%
2. **流程图替代瀑布图** — 不用时间轴条形图，用节点卡片 + 箭头连接展示执行链路
3. **聊天式对话界面** — 不要"上输入框下回答"，要气泡式多轮对话 UI
4. **看链路不是看输出** — 关注的是"走了哪些节点、什么顺序"，不是逐个节点的原始输出

---

## 当前状态回顾

### 可用数据（来自 `done` 事件）
- `graph_path: ["alias_resolve", "planner", "answer"]` — 按顺序的节点名列表
- `summary: {total_elapsed, total_tokens_in, total_tokens_out, total_cost}` — 汇总

### 当前实时事件
- `node_start` → 节点开始，携带 `node.name`, `node.display`, `runtime.start`
- `node_end` → 节点完成，携带 `node.name`, `node.display`, `runtime.{start,end,state_diff,llm_calls,error}`
- `answer_chunk` → 回答增量文本
- `done` → `graph_path` + `summary`

### 当前字号
- body: 15px, 搜索框 0.95em, 回答区 1em, 右侧 0.9em → 目标: body 22px

### 当前布局
- 左右分栏: 左侧问答区 (flex:1) + 右侧 Trace 区 (480px)

### NODE_DISPLAY 映射（13 个节点）
```
alias_resolve → 别名/实体解析
history_extractor → 历史提取
context_builder → 上下文构建
planner → 规划器
query_processing → 查询优化
knowledge_retrieval → 知识检索
metadata_reasoner → 元数据推理专家
similar_expert → 相似推荐专家
merge → 结果合并
simple_fact_answer → 简单事实回答
web_fallback → 联网兜底
answer_planner → 回答结构规划
answer → 回答生成
```

---

## 实施方案

### Step 1: 重写 CSS — 全局字号放大 + 布局调整

**文件**: `static/style.css`

- `body` font-size: 15px → **22px**
- 头部 padding 同比例放大
- 搜索栏、按钮、输入框字号同步放大
- 右侧面板宽度: 480px → **520px** (流程图需要更宽)
- 节点详情字号同步放大
- 流程图卡片样式新增: `.flow-card`, `.flow-arrow`, `.flow-card.running`, `.flow-card.done`, `.flow-card.error`
- 移除所有瀑布图相关 CSS (`.wf-bar` 等)
- 汇总栏用更大字号

### Step 2: 重写 HTML — 聊天式对话 + 流程图

**文件**: `static/index.html`

左侧面板从"输入框 + 回答区"改为**聊天式对话界面**：

```html
<div id="left-panel">
  <!-- 消息列表（可滚动） -->
  <div id="chat-messages">
    <!-- 空状态 -->
    <div id="chat-empty">输入问题开始对话</div>
    <!-- 动态插入的消息气泡 -->
    <!-- <div class="chat-msg user">...</div> -->
    <!-- <div class="chat-msg assistant">...</div> -->
  </div>

  <!-- 输入栏（底部固定） -->
  <div id="chat-input-bar">
    <input type="text" id="query-input" placeholder="输入查询..." autofocus>
    <button id="send-btn">发送</button>
    <div id="thread-row">会话 <span id="thread-id">default</span> <button id="clear-btn">清空</button></div>
  </div>
</div>
```

右侧面板:
- `<h3>执行链路</h3>` + `<div id="flowchart">` + `<div id="node-detail">`

消息气泡 HTML 结构：
```html
<!-- 用户消息 -->
<div class="chat-msg user">
  <div class="chat-avatar">U</div>
  <div class="chat-bubble">进击的巨人评分多少</div>
  <div class="chat-time">14:32</div>
</div>

<!-- AI 消息 -->
<div class="chat-msg assistant">
  <div class="chat-avatar">AI</div>
  <div class="chat-bubble">
    <!-- 打字机效果流式填充 -->
    进击的巨人（TV动画）在元数据中的评分是...
  </div>
  <div class="chat-time">14:32</div>
  <!-- Token 用量 mini badge -->
  <div class="chat-token-badge">1,234 in · 567 out · $0.0021</div>
</div>
```

### Step 3: 重写 JS — 流程图 + 聊天逻辑

**文件**: `static/app.js`

核心变化:

1. **移除** `renderWaterfall()` 以及 `answerStatus`/`answerContent`/`summaryBar` 的直接操作
2. **新增** 聊天消息管理:
```javascript
const Chat = {
  addUserMsg(text) {},       // 新建用户气泡
  createAssistantBubble() {}, // 新建 AI 空气泡（准备打字机填充）
  appendText(text) {},       // 打字机追加文本到当前 AI 气泡
  reset() {},                // 清空所有消息
};
```
3. **新增** `FlowChart` 对象（同前）
4. **事件处理**:
   - 点击发送 → `Chat.addUserMsg(query)` + SSE 连接
   - `answer_chunk` → `Chat.appendText(text)` 打字机填充当前气泡
   - `node_start`/`node_end` → `FlowChart.addNode/updateNode`
   - `done` → 在 AI 气泡底部追加 token 用量 mini badge，清空流程图重置状态

### Step 4: CSS — 聊天气泡样式

**新增关键样式**:
```css
#chat-messages { flex: 1; overflow-y: auto; padding: 24px; }
#chat-input-bar { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--bg-secondary); }

.chat-msg { display: flex; gap: 12px; margin-bottom: 28px; max-width: 85%; }
.chat-msg.user { margin-left: auto; flex-direction: row-reverse; }
.chat-msg.assistant { margin-right: auto; }

.chat-avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85em; flex-shrink: 0; }
.chat-msg.user .chat-avatar { background: var(--accent-dim); color: var(--accent); }
.chat-msg.assistant .chat-avatar { background: var(--green-dim); color: var(--green); }

.chat-bubble { padding: 14px 18px; border-radius: 12px; line-height: 1.7; }
.chat-msg.user .chat-bubble { background: var(--accent-dim); border: 1px solid rgba(88,166,255,0.3); }
.chat-msg.assistant .chat-bubble { background: var(--bg-card); border: 1px solid var(--border); }

.chat-time { font-size: 0.7em; color: var(--text-muted); margin-top: 4px; }
.chat-token-badge { font-size: 0.7em; color: var(--text-muted); margin-top: 4px; font-family: monospace; }
```

### Step 5: 流程图卡片 HTML 结构

**流程图卡片 HTML 结构**:

```html
<div class="flow-card running">
  <!-- 连接箭头 (第一个节点没有) -->
  <div class="flow-arrow">↓</div>
  <!-- 卡片主体 -->
  <div class="flow-card-body">
    <div class="flow-card-header">
      <span class="flow-step-num">#1</span>
      <span class="flow-node-name">别名/实体解析</span>
      <span class="flow-status">● 执行中</span>
      <span class="flow-duration"></span>
    </div>
    <!-- LLM info mini badge (仅 node_end 后有 llm_calls 时显示) -->
    <div class="flow-llm-badge" style="display:none">
      deepseek-v4-flash · 234 in / 56 out · $0.0003
    </div>
    <!-- Error (仅 node_end 后有 error 时显示) -->
    <div class="flow-error" style="display:none">错误信息</div>
  </div>
</div>
```

5. **点击卡片** → 展开 `#node-detail`（逻辑与当前 `showNodeDetail` 类似，改为接收 flowCard 数据）

6. **颜色方案**:
   - running: 蓝色边框 + 旋转动画标记
   - done: 绿色边框 + 耗时
   - error: 红色边框 + 错误信息

### Step 6: 布局调整

- 流程图卡片之间用箭头 `↓` 或 CSS `::after` 伪元素连接
- 卡片最小高度保证，内边距充足
- `#flowchart` 容器 overflow-y: auto，支持滚动

---

## 设计决策

1. **DOM 流程图而非 SVG** — Canvas/SVG 画箭头太复杂，纯 DOM + CSS 箭头更简单可靠
2. **聊天气泡式对话** — 用户蓝色靠右、AI 绿色靠左，多轮历史保留，打字机流式填充
3. **不引入前端框架** — 保持零依赖
4. **字号统一放大** — body 22px 为基准，各元素按比例缩放
5. **节点详情面板保留** — 点击流程图卡片展开详情（state diff + LLM 调用）

## 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `static/style.css` | 字号放大 + 聊天气泡 + 流程图 CSS + 移除 waterfall CSS |
| 修改 | `static/index.html` | 聊天式对话界面 + 流程图替换瀑布图 |
| 修改 | `static/app.js` | Chat 对象 + FlowChart 对象替换 renderWaterfall |

## 验证

1. 页面 100% 缩放下字号舒适可读
2. 输入查询后，左侧出现聊天气泡（用户蓝色靠右，AI 绿色靠左）
3. AI 气泡显示打字机流式效果
4. 多轮对话时历史消息保留可见
5. 右侧流程图卡片按顺序出现，完成后更新耗时和 LLM 用量
6. 点击流程图卡片展开详情面板
7. 每轮对话的 AI 气泡底部显示 token 用量和成本
