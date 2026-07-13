/* AniGraph Trace Panel — Chat + Flowchart */

// ════════════════════════════════════════════════════════
// State
// ════════════════════════════════════════════════════════

const state = {
  running: false,
  threadId: "default",
  flowCards: [],          // [{name, display, start, end, llmCalls, error, stateDiff, el}]
  currentBubble: null,    // 当前正在流式填充的 AI 气泡 DOM
  currentBubbleContent: "", // 累积的 AI 回答文本
};

const $ = (id) => document.getElementById(id);

// ════════════════════════════════════════════════════════
// DOM refs
// ════════════════════════════════════════════════════════

const queryInput = $("query-input");
const sendBtn = $("send-btn");
const chatMessages = $("chat-messages");
const chatEmpty = $("chat-empty");
const flowchart = $("flowchart");
const flowchartEmpty = $("flowchart-empty");
const nodeDetail = $("node-detail");
const detailTitle = $("detail-title");
const detailBody = $("detail-body");
const threadIdSpan = $("thread-id");
const clearBtn = $("clear-btn");
const modelInfo = $("model-info");

// ════════════════════════════════════════════════════════
// Chat
// ════════════════════════════════════════════════════════

const Chat = {
  addUserMsg(text) {
    chatEmpty.style.display = "none";
    const el = document.createElement("div");
    el.className = "chat-msg user";
    el.innerHTML = `
      <div class="chat-avatar">U</div>
      <div>
        <div class="chat-bubble">${escapeHtml(text)}</div>
        <div class="chat-meta">${now()}</div>
      </div>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  },

  createAssistantBubble() {
    const el = document.createElement("div");
    el.className = "chat-msg assistant";
    el.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div>
        <div class="chat-bubble streaming"></div>
        <div class="chat-meta">${now()}</div>
      </div>`;
    chatMessages.appendChild(el);
    state.currentBubble = el.querySelector(".chat-bubble");
    state.currentBubbleContent = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
  },

  appendText(text) {
    if (!state.currentBubble) return;
    state.currentBubbleContent = text;
    state.currentBubble.textContent = text;
    state.currentBubble.classList.remove("streaming");
    chatMessages.scrollTop = chatMessages.scrollHeight;
  },

  reset() {
    state.currentBubble = null;
    state.currentBubbleContent = "";
  },
};

// ════════════════════════════════════════════════════════
// FlowChart
// ════════════════════════════════════════════════════════

const FlowChart = {
  reset() {
    state.flowCards = [];
    flowchart.innerHTML = '<div id="flowchart-empty">等待查询...</div>';
    nodeDetail.style.display = "none";
  },

  addNode(evt) {
    flowchartEmpty.style.display = "none";

    const card = {
      name: evt.node.name,
      display: evt.node.display,
      start: evt.runtime.start,
      end: evt.runtime.end,
      llmCalls: evt.runtime.llm_calls || [],
      error: evt.runtime.error || "",
      stateDiff: null,
    };

    const idx = state.flowCards.length + 1;
    const el = document.createElement("div");
    el.className = "flow-card running";
    el.innerHTML = `
      <div class="flow-arrow"></div>
      <div class="flow-card-body" data-index="${state.flowCards.length}">
        <div class="flow-card-header">
          <span class="flow-step-num">#${idx}</span>
          <span class="flow-node-name">${escapeHtml(card.display)}</span>
          <span class="flow-status">● 执行中</span>
          <span class="flow-duration"></span>
        </div>
      </div>`;
    // 第一个节点不需要箭头
    if (idx === 1) el.querySelector(".flow-arrow").remove();

    card.el = el;
    state.flowCards.push(card);
    flowchart.appendChild(el);

    // 点击事件
    el.querySelector(".flow-card-body").addEventListener("click", () => {
      showNodeDetail(state.flowCards.length - 1);
    });

    flowchart.scrollTop = flowchart.scrollHeight;
  },

  updateNode(evt) {
    const name = evt.node.name;
    // 从后往前找匹配的 card
    for (let i = state.flowCards.length - 1; i >= 0; i--) {
      if (state.flowCards[i].name === name && state.flowCards[i].end === 0) {
        const card = state.flowCards[i];
        card.end = evt.runtime.end;
        card.llmCalls = evt.runtime.llm_calls || [];
        card.error = evt.runtime.error || "";
        card.stateDiff = evt.runtime.state_diff || null;

        const el = card.el;
        const dur = (card.end - card.start).toFixed(1);

        // 更新状态样式
        el.classList.remove("running");
        if (card.error) {
          el.classList.add("error");
          el.querySelector(".flow-status").textContent = "✕ 出错";
          el.querySelector(".flow-duration").textContent = "";
          // 显示错误信息
          const body = el.querySelector(".flow-card-body");
          const errDiv = document.createElement("div");
          errDiv.className = "flow-error-msg";
          errDiv.textContent = card.error;
          body.appendChild(errDiv);
        } else {
          el.classList.add("done");
          el.querySelector(".flow-status").textContent = "✓ 完成";
          el.querySelector(".flow-duration").textContent = dur + "s";
        }

        // LLM badge
        if (card.llmCalls.length > 0) {
          const body = el.querySelector(".flow-card-body");
          card.llmCalls.forEach(llm => {
            const badge = document.createElement("div");
            badge.className = "flow-llm-badge";
            badge.innerHTML = `${escapeHtml(llm.model)} · ${llm.input_tokens} in / ${llm.output_tokens} out`;
            body.appendChild(badge);
          });
        }
        break;
      }
    }
  },
};

// ════════════════════════════════════════════════════════
// Node Detail Viewer
// ════════════════════════════════════════════════════════

function showNodeDetail(index) {
  const card = state.flowCards[index];
  if (!card) return;

  nodeDetail.style.display = "block";
  detailTitle.textContent = `${card.display}`;

  const dur = ((card.end || Date.now() / 1000) - card.start).toFixed(2);
  let html = `<div class="detail-row"><span class="detail-label">耗时</span><span class="detail-value">${dur}s</span></div>`;

  if (card.error) {
    html += `<div class="detail-row"><span class="detail-label">状态</span><span class="detail-value" style="color:var(--red)">${escapeHtml(card.error)}</span></div>`;
  }

  if (card.stateDiff && card.stateDiff.changed && card.stateDiff.changed.length > 0) {
    html += `<div class="detail-row"><span class="detail-label">State</span><span class="detail-value">`;
    card.stateDiff.changed.forEach(key => {
      const val = card.stateDiff.added ? card.stateDiff.added[key] : "?";
      html += `<div><span class="state-diff-field">+${escapeHtml(key)}</span> <span class="state-diff-added">${escapeHtml(String(val))}</span></div>`;
    });
    html += `</span></div>`;
  }

  if (card.llmCalls && card.llmCalls.length > 0) {
    card.llmCalls.forEach((llm, i) => {
      html += `
        <div class="llm-card">
          <div class="llm-header">
            <span class="llm-model">${escapeHtml(llm.model)}</span>
          </div>
          <div class="llm-tokens">${llm.input_tokens} in &middot; ${llm.output_tokens} out</div>
        </div>`;
    });
  }

  detailBody.innerHTML = html;
}

// ════════════════════════════════════════════════════════
// SSE + Event Handlers
// ════════════════════════════════════════════════════════

function sendQuery() {
  const q = queryInput.value.trim();
  if (!q || state.running) return;

  state.running = true;
  sendBtn.disabled = true;

  // 重置流程图
  FlowChart.reset();
  Chat.reset();

  // 添加用户消息
  Chat.addUserMsg(q);
  queryInput.value = "";
  Chat.createAssistantBubble();

  const url = `/chat/stream?query=${encodeURIComponent(q)}&thread_id=${encodeURIComponent(state.threadId)}`;
  const es = new EventSource(url);

  es.addEventListener("node_start", e => {
    FlowChart.addNode(JSON.parse(e.data));
  });
  es.addEventListener("node_end", e => {
    FlowChart.updateNode(JSON.parse(e.data));
  });
  es.addEventListener("answer_chunk", e => {
    const evt = JSON.parse(e.data);
    if (evt.answer_text) Chat.appendText(evt.answer_text);
  });
  es.addEventListener("done", e => {
    state.running = false;
    sendBtn.disabled = false;
    es.close();
  });
  es.addEventListener("error", e => {
    try {
      const err = JSON.parse(e.data || "{}");
      if (state.currentBubble) {
        state.currentBubble.textContent = "错误: " + (err.message || "未知错误");
        state.currentBubble.classList.remove("streaming");
        state.currentBubble.style.color = "var(--red)";
      }
    } catch {
      if (state.currentBubble) {
        state.currentBubble.textContent = "连接异常，请重试";
        state.currentBubble.classList.remove("streaming");
      }
    }
    state.running = false;
    sendBtn.disabled = false;
    es.close();
  });
}

// ════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════

async function init() {
  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    modelInfo.textContent = `LLM: ${d.llm_model}  |  ${d.simple_llm_model}`;
  } catch { modelInfo.textContent = "LLM: -"; }

  sendBtn.addEventListener("click", sendQuery);
  queryInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !state.running) sendQuery();
  });
  clearBtn.addEventListener("click", () => {
    state.threadId = "clear_" + Date.now();
    threadIdSpan.textContent = state.threadId;
    chatMessages.innerHTML = '<div id="chat-empty">输入问题开始对话</div>';
    FlowChart.reset();
  });
}

// ════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function now() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

window.addEventListener("DOMContentLoaded", init);
