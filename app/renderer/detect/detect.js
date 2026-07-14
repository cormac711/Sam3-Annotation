import { fetchJSON, streamSSE, toast, openInAnnotate } from "../shared/shell.js";

const $ = (id) => document.getElementById(id);
let sourceFolder = null;
let running = false;

function fmtMb(mb) {
  if (mb == null) return "—";
  return mb >= 1024 ? (mb / 1024).toFixed(1) + " GB" : Math.round(mb) + " MB";
}

async function refreshGpu() {
  try {
    const gpu = await fetchJSON("/system/gpu");
    if (!gpu.available) {
      $("gpuName").textContent = "no CUDA GPU detected — CPU only";
      $("gpuBarFill").style.width = "0%";
      $("gpuNums").textContent = "—";
      $("device").value = "cpu";
      return;
    }
    $("gpuName").textContent = gpu.name;
    const pct = (gpu.used_mb / gpu.total_mb) * 100;
    $("gpuBarFill").style.width = pct.toFixed(0) + "%";
    $("gpuNums").textContent = `${fmtMb(gpu.used_mb)} / ${fmtMb(gpu.total_mb)}`;
  } catch {
    $("gpuName").textContent = "backend unreachable";
  }
}

async function refreshHfAuth() {
  try {
    const auth = await fetchJSON("/system/hf-auth");
    $("hfCard").style.display = auth.authenticated ? "none" : "block";
  } catch {}
}

function appendLog(line) {
  const el = $("detectLog");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function setRunning(isRunning) {
  running = isRunning;
  $("runBtn").disabled = isRunning;
  $("cancelBtn").disabled = !isRunning;
  $("runStatus").textContent = isRunning ? "running" : "idle";
  $("runStatus").className = "badge" + (isRunning ? " ok" : "");
}

function buildParams() {
  return {
    input_dir: sourceFolder,
    prompt: $("prompt").value || "person",
    threshold: parseFloat($("threshold").value),
    mask_threshold: parseFloat($("maskThreshold").value),
    device: $("device").value,
    precision: $("precision").value,
    batch_size: parseInt($("batchSize").value, 10),
    vram_limit_mb: $("vramLimitEnabled").checked ? parseInt($("vramLimitMb").value, 10) : null,
    recursive: $("recursive").checked,
    save_masks: $("saveMasks").checked,
    no_resume: $("noResume").checked,
    compile_model: $("compileModel").checked,
    preview: 0,
  };
}

async function runDetection() {
  if (!sourceFolder) { toast("pick a folder first", "warn"); return; }
  $("detectLog").textContent = "";
  $("detectDoneRow").style.display = "none";
  $("detectProgressBar").style.width = "0%";
  $("detectProgressLabel").textContent = "starting…";
  setRunning(true);

  try {
    await fetchJSON("/detect/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildParams()),
    });
  } catch (e) {
    toast("could not start: " + e.message, "warn");
    setRunning(false);
    return;
  }

  await streamSSE("/detect/stream", (evt) => {
    if (evt.type === "progress") {
      const pct = evt.total ? (evt.done / evt.total) * 100 : 0;
      $("detectProgressBar").style.width = pct.toFixed(1) + "%";
      $("detectProgressLabel").textContent =
        `${evt.done}/${evt.total} · batch ${evt.batch_size} · ${fmtMb(evt.vram_used_mb)} / ${fmtMb(evt.vram_total_mb)} VRAM` +
        (evt.errors ? ` · ${evt.errors} error(s)` : "");
      appendLog(`[${evt.done}/${evt.total}] ${evt.current_file} — ${evt.matches} match(es)`);
    } else if (evt.type === "done") {
      appendLog(`done -> ${evt.result.images} image(s), ${evt.result.annotations} annotation(s)` +
                (evt.result.errors ? `, ${evt.result.errors} error(s)` : ""));
      toast("detection complete", "ok");
      $("detectDoneRow").style.display = "flex";
      setRunning(false);
    } else if (evt.type === "error") {
      appendLog("ERROR: " + evt.message);
      toast("detection failed: " + evt.message, "warn");
      setRunning(false);
    } else if (evt.type === "cancelled") {
      appendLog("cancelled");
      toast("cancelled — progress up to that point was saved");
      setRunning(false);
    }
  });
  refreshGpu();
}

export function initDetect() {
  ["threshold", "maskThreshold"].forEach((id) => {
    const el = $(id);
    const label = $(id + "Label");
    el.addEventListener("input", () => (label.textContent = parseFloat(el.value).toFixed(2)));
  });
  $("batchSize").addEventListener("input", () => ($("batchSizeLabel").textContent = $("batchSize").value));
  $("vramLimitEnabled").addEventListener("change", () => ($("vramLimitMb").disabled = !$("vramLimitEnabled").checked));

  $("pickFolderBtn").addEventListener("click", async () => {
    const folder = await window.sam3.pickFolder();
    if (!folder) return;
    sourceFolder = folder;
    $("srcFolder").textContent = folder;
  });

  $("runBtn").addEventListener("click", runDetection);
  $("cancelBtn").addEventListener("click", () => fetchJSON("/detect/cancel", { method: "POST" }));
  $("unloadBtn").addEventListener("click", async () => {
    await fetchJSON("/detect/unload", { method: "POST" });
    toast("model unloaded — VRAM released", "ok");
    refreshGpu();
  });
  $("hfLoginBtn").addEventListener("click", async () => {
    const token = $("hfToken").value.trim();
    if (!token) return;
    try {
      await fetchJSON("/system/hf-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      toast("authorized", "ok");
      $("hfToken").value = "";
      refreshHfAuth();
    } catch (e) {
      toast("authorization failed: " + e.message, "warn");
    }
  });
  $("openAnnotateBtn").addEventListener("click", () => openInAnnotate(sourceFolder));
  $("openFolderBtn").addEventListener("click", () => window.sam3.openPath(sourceFolder));

  refreshGpu();
  refreshHfAuth();
  setInterval(() => { if (!running) refreshGpu(); }, 3000);
}
