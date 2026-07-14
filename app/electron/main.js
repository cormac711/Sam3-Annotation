const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const net = require("net");
const fs = require("fs");
const { spawn, execFile } = require("child_process");
const http = require("http");

let backendProcess = null;
let backendPort = null;
let mainWindow = null;

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"]);

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function waitForHealth(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) return resolve();
        retry();
      });
      req.on("error", retry);
      req.setTimeout(1500, () => req.destroy());
    };
    const retry = () => {
      if (Date.now() > deadline) return reject(new Error("backend did not become healthy in time"));
      setTimeout(tryOnce, 300);
    };
    tryOnce();
  });
}

function spawnBackend(port) {
  if (app.isPackaged) {
    const exe = path.join(process.resourcesPath, "backend", "sam3-backend.exe");
    backendProcess = execFile(exe, ["--port", String(port)], { windowsHide: true });
  } else {
    const script = path.join(__dirname, "..", "..", "backend", "server.py");
    backendProcess = spawn("python", [script, "--port", String(port)], { windowsHide: true });
  }
  backendProcess.stdout && backendProcess.stdout.on("data", (d) => process.stdout.write(`[backend] ${d}`));
  backendProcess.stderr && backendProcess.stderr.on("data", (d) => process.stderr.write(`[backend] ${d}`));
  backendProcess.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProcess = null;
  });
}

function killBackend() {
  return new Promise((resolve) => {
    if (!backendProcess || !backendProcess.pid) return resolve();
    const pid = backendProcess.pid;
    if (process.platform === "win32") {
      execFile("taskkill", ["/pid", String(pid), "/T", "/F"], () => resolve());
    } else {
      backendProcess.kill("SIGTERM");
      resolve();
    }
  });
}

async function createWindow() {
  backendPort = await getFreePort();
  spawnBackend(backendPort);

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  try {
    await waitForHealth(backendPort);
  } catch (e) {
    dialog.showErrorBox("Sam3 Annotation", `The detection backend failed to start:\n${e.message}`);
  }

  mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
  mainWindow.once("ready-to-show", () => mainWindow.show());
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

let quitting = false;
app.on("before-quit", async (e) => {
  if (quitting) return;
  e.preventDefault();
  quitting = true;
  await killBackend();
  app.quit();
});

// ---------------------------------------------------------------------
// IPC: system / backend wiring
// ---------------------------------------------------------------------
ipcMain.handle("backend:port", () => backendPort);

ipcMain.handle("dialog:pickFolder", async () => {
  const res = await dialog.showOpenDialog(mainWindow, { properties: ["openDirectory"] });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.handle("dialog:pickFile", async (_evt, opts) => {
  const res = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: (opts && opts.filters) || [{ name: "All Files", extensions: ["*"] }],
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.handle("fs:readFolder", async (_evt, folderPath) => {
  const entries = await fs.promises.readdir(folderPath, { withFileTypes: true });
  const images = [];
  for (const ent of entries) {
    if (!ent.isFile()) continue;
    const ext = path.extname(ent.name).toLowerCase();
    if (IMAGE_EXTS.has(ext)) images.push(ent.name);
  }
  let jsonFile = null;
  let jsonMtime = -1;
  for (const ent of entries) {
    if (ent.isFile() && ent.name.toLowerCase().endsWith(".json")) {
      const stat = await fs.promises.stat(path.join(folderPath, ent.name));
      if (stat.mtimeMs > jsonMtime) {
        jsonMtime = stat.mtimeMs;
        jsonFile = ent.name;
      }
    }
  }
  return { images: images.sort(), jsonFile };
});

ipcMain.handle("fs:readFileDataUrl", async (_evt, filePath) => {
  const buf = await fs.promises.readFile(filePath);
  const ext = path.extname(filePath).slice(1).toLowerCase();
  const mime = ext === "jpg" ? "jpeg" : ext;
  return `data:image/${mime};base64,${buf.toString("base64")}`;
});

ipcMain.handle("fs:readTextFile", async (_evt, filePath) => {
  try {
    return await fs.promises.readFile(filePath, "utf-8");
  } catch {
    return null;
  }
});

ipcMain.handle("fs:writeTextFile", async (_evt, filePath, contents) => {
  await fs.promises.writeFile(filePath, contents, "utf-8");
  return true;
});

ipcMain.handle("shell:openPath", async (_evt, targetPath) => {
  await shell.openPath(targetPath);
});
