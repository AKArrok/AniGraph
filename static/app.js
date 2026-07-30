/* AniRAG Trace Panel — Chat + Flowchart */

const state = {
  running: false,
  threadId: "default",
  selectedImage: null,
  selectedImageFile: null,
  abortController: null,
  panelCollapsed: false,
  flowCards: [],
  currentBubble: null,
  currentBubbleContent: "",
};

const $ = (id) => document.getElementById(id);

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
const fileInput = $("file-input");
const imageBtn = $("image-btn");
const imagePreviewRow = $("image-preview-row");
const previewImg = $("preview-img");
const removeImgBtn = $("remove-img-btn");
const stopBtn = $("stop-btn");
const errorBanner = $("error-banner");
const errorBannerText = $("error-banner-text");
const errorDismissBtn = $("error-dismiss-btn");
const togglePanelBtn = $("toggle-panel-btn");
const rightPanel = $("right-panel");
const initialEmptyState = chatEmpty.cloneNode(true);

function showError(message) {
  errorBannerText.textContent = message || "发生未知错误";
  errorBanner.classList.add("show");
}

function hideError() {
  errorBanner.classList.remove("show");
  errorBannerText.textContent = "";
}

const Chat = {
  addUserMsg(text, image) {
    const currentEmpty = chatMessages.querySelector("#chat-empty");
    if (currentEmpty) currentEmpty.style.display = "none";
    const el = document.createElement("div");
    el.className = "chat-msg user";
    const bubbleContent = document.createElement("div");
    bubbleContent.className = "chat-bubble";
    if (text) {
      const textNode = document.createElement("div");
      textNode.textContent = text;
      bubbleContent.appendChild(textNode);
    }
    if (image) {
      const imageNode = document.createElement("img");
      imageNode.src = image;
      imageNode.alt = "用户上传的图片";
      bubbleContent.appendChild(imageNode);
    }

    el.innerHTML = `
      <div class="chat-avatar">U</div>
      <div>
        <div class="chat-meta">${now()}</div>
      </div>`;
    el.children[1].insertBefore(bubbleContent, el.children[1].firstChild);
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
    state.currentBubble.innerHTML = renderMarkdown(state.currentBubbleContent);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  },

  finalize() {
    if (state.currentBubble) state.currentBubble.classList.remove("streaming");
    state.currentBubble = null;
    state.currentBubbleContent = "";
  },

  reset() {
    state.currentBubble = null;
    state.currentBubbleContent = "";
  },
};

const FlowChart = {
  reset() {
    state.flowCards = [];
    flowchart.innerHTML = '<div id="flowchart-empty">等待查询...</div>';
    nodeDetail.style.display = "none";
  },

  addNode(evt) {
    const empty = flowchart.querySelector("#flowchart-empty");
    if (empty) empty.style.display = "none";

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
    if (idx === 1) el.querySelector(".flow-arrow").remove();

    card.el = el;
    state.flowCards.push(card);
    flowchart.appendChild(el);

    const cardIndex = state.flowCards.length - 1;
    const cardBody = el.querySelector(".flow-card-body");
    cardBody.tabIndex = 0;
    cardBody.setAttribute("role", "button");
    cardBody.addEventListener("click", () => showNodeDetail(cardIndex));
    cardBody.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showNodeDetail(cardIndex);
      }
    });

    flowchart.scrollTop = flowchart.scrollHeight;
  },

  updateNode(evt) {
    const name = evt.node.name;
    for (let i = state.flowCards.length - 1; i >= 0; i--) {
      if (state.flowCards[i].name === name && state.flowCards[i].end === 0) {
        const card = state.flowCards[i];
        card.end = evt.runtime.end;
        card.llmCalls = evt.runtime.llm_calls || [];
        card.error = evt.runtime.error || "";
        card.stateDiff = evt.runtime.state_diff || null;

        const el = card.el;
        const dur = (card.end - card.start).toFixed(1);

        el.classList.remove("running");
        if (card.error) {
          el.classList.add("error");
          el.querySelector(".flow-status").textContent = "✕ 出错";
          el.querySelector(".flow-duration").textContent = "";
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
          <div class="llm-tokens">${llm.input_tokens} in · ${llm.output_tokens} out</div>
        </div>`;
    });
  }

  detailBody.innerHTML = html;
}

function setRunning(running) {
  state.running = running;
  sendBtn.disabled = running;
  queryInput.disabled = running;
  fileInput.disabled = running;
  imageBtn.classList.toggle("disabled", running);
  stopBtn.classList.toggle("show", running);
}

function clearSelectedImage() {
  state.selectedImage = null;
  state.selectedImageFile = null;
  fileInput.value = "";
  previewImg.removeAttribute("src");
  imagePreviewRow.style.display = "none";
}

function dispatchSseEvent(eventName, data) {
  let evt = {};
  if (data && data.trim()) {
    try {
      evt = JSON.parse(data);
    } catch {
      showError("服务器返回了无法解析的流数据");
      return;
    }
  }

  if (eventName === "node_start") FlowChart.addNode(evt);
  else if (eventName === "node_end") FlowChart.updateNode(evt);
  else if (eventName === "answer_chunk" && evt.answer_text) Chat.appendText(evt.answer_text);
  else if (eventName === "error") showError(evt.message || "服务执行失败");
  else if (eventName === "done") Chat.finalize();
}

function consumeSseBlock(block) {
  let eventName = "message";
  const dataLines = [];
  block.split("\n").forEach(line => {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  });
  dispatchSseEvent(eventName, dataLines.join("\n"));
}

async function sendQuery() {
  const q = queryInput.value.trim();
  const image = state.selectedImage;
  const imageFile = state.selectedImageFile;
  if ((!q && !image) || state.running) return;

  hideError();
  setRunning(true);
  state.abortController = new AbortController();

  FlowChart.reset();
  Chat.reset();

  Chat.addUserMsg(q || "识别这张动漫截图", image);
  queryInput.value = "";
  clearSelectedImage();
  Chat.createAssistantBubble();

  try {
    let response;
    if (imageFile) {
      const form = new FormData();
      form.append("file", imageFile);
      form.append("query", q);
      form.append("thread_id", state.threadId);
      response = await fetch("/chat/image", {
        method: "POST",
        body: form,
        signal: state.abortController.signal,
      });
    } else {
      response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, thread_id: state.threadId }),
        signal: state.abortController.signal,
      });
    }

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        detail = body.detail || body.message || detail;
      } catch { /* keep status message */ }
      throw new Error(detail);
    }
    if (!response.body) throw new Error("浏览器无法读取服务器流");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (block.trim()) consumeSseBlock(block);
      }
      if (done) break;
    }
    if (buffer.trim()) consumeSseBlock(buffer);
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message || "连接异常，请重试");
    Chat.finalize();
  } finally {
    state.abortController = null;
    setRunning(false);
    queryInput.focus();
  }
}

function abortQuery() {
  if (state.abortController) state.abortController.abort();
}

function togglePanel() {
  state.panelCollapsed = !state.panelCollapsed;
  rightPanel.classList.toggle("collapsed", state.panelCollapsed);
  togglePanelBtn.setAttribute("aria-expanded", String(!state.panelCollapsed));
}

async function init() {
  if (window.innerWidth <= 768) togglePanel();
  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    const imageProvider = d.image_recognition?.enabled
      ? `  |  识图: ${d.image_recognition.provider}`
      : "";
    modelInfo.textContent = `LLM: ${d.llm_model}${imageProvider}`;
  } catch { modelInfo.textContent = "LLM: -"; }

  sendBtn.addEventListener("click", sendQuery);
  stopBtn.addEventListener("click", abortQuery);
  errorDismissBtn.addEventListener("click", hideError);
  togglePanelBtn.addEventListener("click", togglePanel);
  queryInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !state.running) sendQuery();
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/gif"];
    if (!allowed.includes(file.type)) {
      clearSelectedImage();
      showError("仅支持 JPEG、PNG、WebP 或 GIF 图片");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      clearSelectedImage();
      showError("图片不能超过 10 MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      state.selectedImage = String(reader.result || "");
      state.selectedImageFile = file;
      previewImg.src = state.selectedImage;
      previewImg.alt = file.name;
      imagePreviewRow.style.display = "block";
    };
    reader.onerror = () => {
      clearSelectedImage();
      showError("读取图片失败");
    };
    reader.readAsDataURL(file);
  });
  removeImgBtn.addEventListener("click", clearSelectedImage);
  clearBtn.addEventListener("click", () => {
    state.threadId = "clear_" + Date.now();
    threadIdSpan.textContent = state.threadId;
    chatMessages.replaceChildren(initialEmptyState.cloneNode(true));
    clearSelectedImage();
    FlowChart.reset();
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderMarkdown(text) {
  const codeBlocks = [];
  let html = escapeHtml(text).replace(/```(?:\w+)?\s*\n?([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(`<pre><code>${code}</code></pre>`);
    return `\u0000CODE${codeBlocks.length - 1}\u0000`;
  });

  html = html
    .replace(/^#### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/^---+$/gm, "<hr>")
    .replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  const lines = html.split("\n");
  const output = [];
  let listType = null;
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };

  lines.forEach(line => {
    const unordered = line.match(/^[-*] (.+)$/);
    const ordered = line.match(/^\d+\. (.+)$/);
    const nextType = unordered ? "ul" : ordered ? "ol" : null;
    if (nextType) {
      if (listType !== nextType) {
        closeList();
        output.push(`<${nextType}>`);
        listType = nextType;
      }
      output.push(`<li>${(unordered || ordered)[1]}</li>`);
      return;
    }
    closeList();
    if (!line.trim()) output.push("");
    else if (/^(?:<(?:h[1-4]|blockquote|hr|pre)|\u0000CODE)/.test(line)) output.push(line);
    else output.push(`<p>${line}</p>`);
  });
  closeList();

  html = output.join("\n").replace(/\u0000CODE(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)]);
  return html;
}

function now() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

window.addEventListener("DOMContentLoaded", init);
