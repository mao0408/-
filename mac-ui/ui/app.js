const API = "http://127.0.0.1:8765/api";
const state = { replies: [], selected: -1, context: "", settings: {} };

const $ = (selector) => document.querySelector(selector);
const setText = (selector, value) => { $(selector).textContent = value ?? ""; };

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function renderContext(context) {
  const view = $("#contextView");
  view.innerHTML = "";
  const lines = String(context || "").split("\n").filter(Boolean);
  if (!lines.length) {
    view.innerHTML = '<div class="empty-context">点击“识别当前聊天”读取微信窗口内容</div>';
    return;
  }
  for (const line of lines) {
    const match = line.match(/^\[([^\]]+)\]\s*(.*)$/);
    const div = document.createElement("div");
    div.className = `context-line ${match?.[1] === "我" ? "own" : ""}`;
    div.textContent = match ? match[2] : line;
    view.appendChild(div);
  }
  view.scrollTop = view.scrollHeight;
}

function renderReplies(replies = []) {
  state.replies = replies;
  state.selected = replies.length ? 0 : -1;
  const list = $("#replyList");
  list.innerHTML = "";
  replies.forEach((reply, index) => {
    const card = document.createElement("div");
    card.className = `reply-card ${index === state.selected ? "selected" : ""}`;
    card.innerHTML = `
      <div class="reply-number">${String(index + 1).padStart(2, "0")}</div>
      <div class="reply-text"></div>
      <div class="reply-actions">
        <button class="edit-mini" title="编辑">⌕</button>
        <button class="send-mini" title="发送">⌁</button>
      </div>`;
    card.querySelector(".reply-text").textContent = reply;
    card.addEventListener("click", () => selectReply(index));
    card.querySelector(".edit-mini").addEventListener("click", (event) => {
      event.stopPropagation();
      selectReply(index);
      $("#replyInput").focus();
    });
    card.querySelector(".send-mini").addEventListener("click", async (event) => {
      event.stopPropagation();
      selectReply(index);
      await sendSelected();
    });
    list.appendChild(card);
  });
  if (replies.length) $("#replyInput").value = replies[0];
}

function selectReply(index) {
  state.selected = index;
  $("#replyInput").value = state.replies[index] || "";
  document.querySelectorAll(".reply-card").forEach((card, current) => {
    card.classList.toggle("selected", current === index);
  });
}

function applyState(data) {
  state.context = data.context || state.context;
  renderContext(state.context);
  renderReplies(data.replies || []);
  setText("#conversationTitle", `当前聊天：${data.platform ? `${data.platform}：` : ""}${data.window_title || "等待聊天窗口"}`);
  setText("#panelTitle", `当前聊天：${data.window_title || "等待聊天窗口"}`);
  setText("#statusText", data.status || "等待聊天窗口");
  setText("#footerStatus", data.status || "聊天内容无变化");
  setText("#progressText", data.status || "等待识别聊天窗口");
  setText("#sourceText", data.reply_source || "未生成");
  $("#ocrStatus").style.background = data.ocr_ready === false ? "#d87d7d" : "var(--green)";
  $("#aiStatus").style.background = data.api_ready ? "var(--green)" : "#e4b24e";
}

async function recognize() {
  $("#recognizeButton").disabled = true;
  setText("#statusText", "正在识别当前聊天...");
  try { applyState(await request("/recognize", { method: "POST", body: "{}" })); }
  catch (error) { setText("#statusText", error.message); }
  finally { $("#recognizeButton").disabled = false; }
}

async function regenerate() {
  if (!state.context.trim()) return;
  try { applyState(await request("/generate", { method: "POST", body: JSON.stringify({ context: state.context }) })); }
  catch (error) { setText("#statusText", error.message); }
}

async function sendSelected() {
  const text = $("#replyInput").value.trim();
  if (!text) return;
  try { applyState(await request("/send", { method: "POST", body: JSON.stringify({ text }) })); }
  catch (error) { setText("#statusText", error.message); }
}

async function copyReply() {
  const text = $("#replyInput").value.trim();
  if (!text) return;
  await navigator.clipboard.writeText(text);
  setText("#statusText", "回复已复制到剪贴板");
}

document.querySelectorAll("[data-toggle]").forEach((button) => {
  button.addEventListener("click", async () => {
    button.classList.toggle("selected");
    const key = button.dataset.toggle === "training" ? "training_mode" : button.dataset.toggle === "auto" ? "auto_recognize" : "managed_auto_reply";
    try { applyState(await request("/settings", { method: "POST", body: JSON.stringify({ [key]: button.classList.contains("selected") }) })); }
    catch (error) { setText("#statusText", error.message); }
  });
});

$("#sourceMode").addEventListener("change", async (event) => {
  try { applyState(await request("/settings", { method: "POST", body: JSON.stringify({ reply_source_mode: event.target.value }) })); }
  catch (error) { setText("#statusText", error.message); }
});
$("#recognizeButton").addEventListener("click", recognize);
$("#regenerateButton").addEventListener("click", regenerate);
$("#sendButton").addEventListener("click", sendSelected);
$("#copyButton").addEventListener("click", copyReply);
$("#replyInput").addEventListener("input", (event) => {
  if (state.selected >= 0) state.replies[state.selected] = event.target.value;
});
$("#settingsButton").addEventListener("click", () => setText("#statusText", "设置页将在下一步接入"));
$("#footerSettings").addEventListener("click", () => setText("#statusText", "设置页将在下一步接入"));

request("/state").then(applyState).catch(() => setText("#statusText", "本地回复服务未启动"));
