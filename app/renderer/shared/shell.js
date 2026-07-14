// Tab switching + small shared utilities used by detect.js and export.js.
// annotate.js is a separately-scoped classic script (see app/renderer/annotate/annotate.js)
// that reads window.__activeView directly, so it doesn't import this module.

export let backendBase = null;

export async function initBackend() {
  const port = await window.sam3.backendPort();
  backendBase = `http://127.0.0.1:${port}`;
  return backendBase;
}

export function toast(message, kind = "") {
  const host = document.getElementById("shell-toasts");
  const el = document.createElement("div");
  el.className = "shell-toast" + (kind ? " " + kind : "");
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

export async function fetchJSON(path, opts) {
  const res = await fetch(backendBase + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

// Consumes a backend SSE stream, calling onEvent for every {type,...} frame.
// Resolves once the stream ends (type "done" | "error" | "cancelled").
export async function streamSSE(path, onEvent) {
  const res = await fetch(backendBase + path);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.split("\n").find(l => l.startsWith("data: "));
      if (!line) continue;
      const evt = JSON.parse(line.slice(6));
      onEvent(evt);
      if (evt.type === "done" || evt.type === "error" || evt.type === "cancelled") return;
    }
  }
}

function setActiveView(name) {
  window.__activeView = name;
  document.querySelectorAll(".shell-tab").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".shell-view").forEach(v => v.classList.toggle("active", v.id === "tab-" + name));
  window.dispatchEvent(new CustomEvent("sam3:view-activated", { detail: { name } }));
}

export function initTabs(defaultView = "detect") {
  document.querySelectorAll(".shell-tab").forEach(btn => {
    btn.addEventListener("click", () => setActiveView(btn.dataset.view));
  });
  setActiveView(defaultView);
}

// Detect -> Annotate handoff: called by detect.js when a job finishes, lets
// the user jump straight to labeling the folder they just detected on.
export function openInAnnotate(folderPath) {
  setActiveView("annotate");
  if (window.__annotateOpenFolder) window.__annotateOpenFolder(folderPath);
}
