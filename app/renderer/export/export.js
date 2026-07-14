import { fetchJSON, streamSSE, toast } from "../shared/shell.js";

const $ = (id) => document.getElementById(id);
let annotationsPath = null;
let imagesDir = null;
let outDir = null;

function appendLog(line) {
  const el = $("exportLog");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function setRunning(isRunning) {
  $("exportRunBtn").disabled = isRunning;
  $("exportStatus").textContent = isRunning ? "running" : "idle";
  $("exportStatus").className = "badge" + (isRunning ? " ok" : "");
}

async function runExport() {
  if (!annotationsPath) { toast("pick a labeled annotations JSON first", "warn"); return; }
  if (!outDir) { toast("pick an output folder first", "warn"); return; }
  $("exportLog").textContent = "";
  $("exportDoneRow").style.display = "none";
  $("exportProgressBar").style.width = "0%";
  setRunning(true);

  const format = document.querySelector('input[name=format]:checked').value;
  const body = {
    annotations: annotationsPath,
    images_dir: imagesDir,
    out: outDir,
    format,
    val_ratio: parseFloat($("valRatio").value),
    seed: parseInt($("seed").value, 10),
    pad: parseFloat($("pad").value),
    min_crop: parseInt($("minCrop").value, 10),
    link: $("link").checked,
  };

  try {
    await fetchJSON("/export/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    toast("could not start: " + e.message, "warn");
    setRunning(false);
    return;
  }

  await streamSSE("/export/stream", (evt) => {
    if (evt.type === "progress") {
      const pct = evt.total ? (evt.done / evt.total) * 100 : 0;
      $("exportProgressBar").style.width = pct.toFixed(1) + "%";
      $("exportProgressLabel").textContent = `${evt.done}/${evt.total} image(s)`;
    } else if (evt.type === "done") {
      const r = evt.result;
      appendLog("classes: " + r.classes.map(c => `${c.name} (${c.boxes})`).join(", "));
      if (r.yolo) appendLog(`yolo: ${r.yolo.images} image(s), ${r.yolo.boxes} box(es) -> ${r.yolo.path}`);
      if (r.crops) appendLog(`crops: ${r.crops.count} crop(s) -> ${r.crops.path}`);
      if (r.missing.length) appendLog(`warning: ${r.missing.length} image(s) not found`);
      toast("export complete", "ok");
      $("exportDoneRow").style.display = "flex";
      setRunning(false);
    } else if (evt.type === "error") {
      appendLog("ERROR: " + evt.message);
      toast("export failed: " + evt.message, "warn");
      setRunning(false);
    } else if (evt.type === "cancelled") {
      appendLog("cancelled");
      setRunning(false);
    }
  });
}

export function initExport() {
  $("pickAnnotationsBtn").addEventListener("click", async () => {
    const file = await window.sam3.pickFile({ filters: [{ name: "COCO JSON", extensions: ["json"] }] });
    if (!file) return;
    annotationsPath = file;
    $("expAnnotations").textContent = file;
  });
  $("pickImagesDirBtn").addEventListener("click", async () => {
    const folder = await window.sam3.pickFolder();
    if (!folder) return;
    imagesDir = folder;
    $("expImagesDir").textContent = folder;
  });
  $("clearImagesDirBtn").addEventListener("click", () => {
    imagesDir = null;
    $("expImagesDir").textContent = "(same as JSON)";
  });
  $("pickOutBtn").addEventListener("click", async () => {
    const folder = await window.sam3.pickFolder();
    if (!folder) return;
    outDir = folder;
    $("expOut").textContent = folder;
  });
  $("exportRunBtn").addEventListener("click", runExport);
  $("openDatasetBtn").addEventListener("click", () => window.sam3.openPath(outDir));
}

// Detect tab can hand a folder off here too (e.g. a future "export this" shortcut).
window.__exportSetOutDir = (dir) => { outDir = dir; $("expOut").textContent = dir; };
