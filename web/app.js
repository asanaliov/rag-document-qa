"use strict";

const $ = (id) => document.getElementById(id);
const body = document.body;
const thread = $("thread");
const question = $("question");
const send = $("send");
const notice = $("notice");

let config = { max_upload_mb: 25, extensions: [] };
let history = []; // [question, answer] pairs, sent with each question so follow-ups work
let streaming = null; // AbortController while an answer is arriving

// ---------- Page state ----------

function showNotice(message) {
  notice.textContent = message || "";
  notice.hidden = !message;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function showDocument(doc) {
  if (!doc) {
    body.dataset.state = "empty";
    return;
  }
  const size = (doc.pages ? `${plural(doc.pages, "page")}, ` : "") + plural(doc.chunks, "passage");
  $("doc-name").textContent = doc.name;
  $("doc-meta").textContent = size;
  $("ready-title").textContent = `${doc.name} is ready.`;
  $("ready-meta").textContent = `It was split into ${plural(doc.chunks, "passage")}. Ask your first question below.`;
  question.placeholder = `Ask about ${doc.name}`;
  body.dataset.state = "ready";
  question.focus();
}

function clearConversation() {
  if (streaming) streaming.abort();
  history = [];
  thread.replaceChildren();
  body.classList.remove("talking");
  $("clear").hidden = true;
}

// ---------- Adding a document ----------

async function upload(file) {
  if (!file) return;
  showNotice("");
  if (file.size > config.max_upload_mb * 1024 * 1024) {
    showNotice(`${file.name} is larger than the ${config.max_upload_mb} MB limit.`);
    return;
  }

  const previous = body.dataset.state;
  clearConversation();
  $("reading-title").textContent = `Reading ${file.name}`;
  body.dataset.state = "reading";

  try {
    const response = await fetch(`/api/document?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail);
    showDocument(result);
  } catch (error) {
    // The server drops the old document as soon as a new upload starts.
    body.dataset.state = previous === "ready" ? "empty" : previous;
    showNotice(error.message || "The document could not be read. Is the server still running?");
  }
}

$("file").addEventListener("change", (event) => {
  upload(event.target.files[0]);
  event.target.value = "";
});
$("choose").addEventListener("click", () => $("file").click());
$("replace").addEventListener("click", () => $("file").click());
$("clear").addEventListener("click", clearConversation);

let dragDepth = 0;
const hasFiles = (event) => event.dataTransfer && [...event.dataTransfer.types].includes("Files");

window.addEventListener("dragenter", (event) => {
  if (!hasFiles(event)) return;
  dragDepth += 1;
  body.classList.add("dragging");
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) body.classList.remove("dragging");
});
window.addEventListener("dragover", (event) => {
  if (hasFiles(event)) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  body.classList.remove("dragging");
  if (body.dataset.state !== "reading") upload(event.dataTransfer.files[0]);
});

// ---------- Rendering an answer ----------

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

// [1], [2][3] and [1, 2] become buttons that point at the passage in the margin.
function inline(text, passages) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (match, list) => {
      const numbers = list.split(",").map((n) => Number(n));
      if (numbers.some((n) => n < 1 || n > passages)) return match;
      return numbers
        .map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Show passage ${n}">${n}</button>`)
        .join("");
    });
}

// Just enough Markdown for what a model writes in a short answer: paragraphs, lists, bold, code.
function renderAnswer(text, passages) {
  const bullet = /^\s*[-*•]\s+/;
  const numbered = /^\s*\d+[.)]\s+/;
  return escapeHtml(text.trim())
    .split(/\n{2,}/)
    .map((block) => {
      const lines = block.split("\n");
      for (const [pattern, tag] of [[bullet, "ul"], [numbered, "ol"]]) {
        if (lines.every((line) => pattern.test(line))) {
          const items = lines.map((line) => `<li>${inline(line.replace(pattern, ""), passages)}</li>`);
          return `<${tag}>${items.join("")}</${tag}>`;
        }
      }
      return `<p>${inline(lines.join("<br>"), passages)}</p>`;
    })
    .join("");
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderNotes(notes, sources, query, asked) {
  notes.replaceChildren();
  if (query && query !== asked) {
    notes.append(element("li", "searched", `Searched for “${query}”`));
  }
  sources.forEach((source, i) => {
    const note = element("li", "note");
    note.dataset.n = i + 1;

    const head = element("div", "note-head");
    head.append(element("span", "note-n", String(i + 1)));
    if (source.page !== null && source.page !== undefined) {
      head.append(element("span", "", `Page ${source.page + 1}`));
    }
    // Flatten to one paragraph and drop Markdown heading marks, which read as noise here.
    const flat = source.text.replace(/^#{1,6}\s+/gm, "").replace(/\s+/g, " ").trim();
    const text = element("p", "note-text", flat);
    const toggle = element("button", "quiet", "Show full passage");
    toggle.type = "button";
    toggle.dataset.toggle = "";
    note.append(head, text, toggle);
    notes.append(note);

    // Nothing to expand when the passage already fits.
    toggle.hidden = text.scrollHeight <= text.clientHeight + 1;
  });
}

function setOpen(note, open) {
  note.classList.toggle("open", open);
  const toggle = note.querySelector("[data-toggle]");
  toggle.textContent = open ? "Show less" : "Show full passage";
}

const noteFor = (cite) => cite.closest(".turn").querySelector(`.note[data-n="${cite.dataset.n}"]`);

thread.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) {
    const note = noteFor(cite);
    if (!note) return;
    setOpen(note, true);
    note.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return;
  }
  const toggle = event.target.closest("[data-toggle]");
  if (toggle) {
    const note = toggle.closest(".note");
    setOpen(note, !note.classList.contains("open"));
  }
});

// Pointing at a citation lights up its passage.
for (const [type, lit] of [["mouseover", true], ["mouseout", false], ["focusin", true], ["focusout", false]]) {
  thread.addEventListener(type, (event) => {
    const cite = event.target.closest && event.target.closest(".cite");
    const note = cite && noteFor(cite);
    if (note) note.classList.toggle("lit", lit);
  });
}

// ---------- Asking ----------

function setStreaming(controller) {
  streaming = controller;
  send.textContent = controller ? "Stop" : "Ask";
  send.classList.toggle("stop", Boolean(controller));
}

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 120;
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line) yield JSON.parse(line);
  }
}

async function ask(asked) {
  const turn = element("li", "turn");
  const answer = element("div", "answer");
  const notes = element("ol", "notes");
  notes.setAttribute("aria-label", "Passages used for this answer");
  answer.innerHTML = '<span class="caret" role="status" aria-label="Looking through the document"></span>';
  turn.append(element("p", "ask", asked), answer, notes);
  thread.append(turn);
  body.classList.add("talking");
  $("clear").hidden = false;
  turn.scrollIntoView({ block: "start", behavior: "smooth" });

  const controller = new AbortController();
  setStreaming(controller);
  let text = "";
  let passages = 0;

  const fail = (message) => {
    answer.classList.add("failed");
    answer.textContent = message;
    notes.replaceChildren();
  };

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: asked, history }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = (await response.json()).detail;
      throw new Error(typeof detail === "string" ? detail : "The question could not be sent.");
    }

    for await (const event of readEvents(response)) {
      const follow = nearBottom();
      if (event.type === "sources") {
        passages = event.sources.length;
        renderNotes(notes, event.sources, event.query, asked);
      } else if (event.type === "text") {
        text += event.text;
        if (text.trim()) answer.innerHTML = renderAnswer(text, passages);
      } else if (event.type === "error") {
        fail(event.message);
        return;
      } else if (event.type === "done") {
        history.push([asked, text.trim()]);
        showModelStatus("ready");
        // Dim the passages the answer did not point at, unless it cited nothing at all.
        const cited = new Set([...answer.querySelectorAll(".cite")].map((c) => c.dataset.n));
        if (cited.size) {
          for (const note of notes.querySelectorAll(".note")) {
            note.classList.toggle("uncited", !cited.has(note.dataset.n));
          }
        }
      }
      if (follow) window.scrollTo({ top: document.documentElement.scrollHeight });
    }
  } catch (error) {
    if (error.name === "AbortError") {
      if (!text.trim()) answer.replaceChildren();
      turn.append(element("p", "stopped", "Stopped."));
    } else {
      fail(error.message || "The server did not respond. Is it still running?");
    }
  } finally {
    if (streaming === controller) setStreaming(null);
  }
}

function resize() {
  question.style.height = "auto";
  question.style.height = `${question.scrollHeight}px`;
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (streaming) {
    streaming.abort();
    return;
  }
  const asked = question.value.trim();
  if (!asked) return;
  question.value = "";
  resize();
  ask(asked);
});

question.addEventListener("input", resize);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!streaming) $("composer").requestSubmit();
  }
});

// ---------- Model warm-up ----------

function showModelStatus(state, model) {
  const status = $("model-status");
  status.classList.toggle("failed", state === "failed");
  status.textContent =
    state === "warming-up" ? `Loading ${model}` :
    state === "failed" ? `Could not reach Ollama. Start it with: ollama serve` : "";
  status.hidden = !status.textContent;
}

// Poll until the server's warm-up request finishes; a failure stays visible until an answer succeeds.
async function watchModel(state, model) {
  showModelStatus(state, model);
  while (state === "warming-up") {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    try {
      state = (await (await fetch("/api/session")).json()).model_status;
    } catch {
      return;
    }
    showModelStatus(state, model);
  }
}

// ---------- Start ----------

(async function start() {
  try {
    const session = await (await fetch("/api/session")).json();
    config = session;
    $("model").textContent = session.model;
    $("limits").textContent =
      `or drop a file anywhere on this page. ${session.extensions.map((e) => e.slice(1).toUpperCase()).join(", ")} up to ${session.max_upload_mb} MB.`;
    $("file").accept = session.extensions.join(",");
    showDocument(session.document);
    watchModel(session.model_status, session.model);
  } catch {
    body.dataset.state = "empty";
    showNotice("The server did not respond. Is it still running?");
  }
})();
