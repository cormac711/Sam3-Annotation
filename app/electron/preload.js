const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("sam3", {
  backendPort: () => ipcRenderer.invoke("backend:port"),
  pickFolder: () => ipcRenderer.invoke("dialog:pickFolder"),
  pickFile: (opts) => ipcRenderer.invoke("dialog:pickFile", opts),
  readFolder: (folderPath) => ipcRenderer.invoke("fs:readFolder", folderPath),
  readFileDataUrl: (filePath) => ipcRenderer.invoke("fs:readFileDataUrl", filePath),
  readTextFile: (filePath) => ipcRenderer.invoke("fs:readTextFile", filePath),
  writeTextFile: (filePath, contents) => ipcRenderer.invoke("fs:writeTextFile", filePath, contents),
  openPath: (targetPath) => ipcRenderer.invoke("shell:openPath", targetPath),
});
