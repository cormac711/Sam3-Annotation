(function(){
"use strict";
/* ============================== state ============================== */
const viewRoot = document.getElementById("view-annotate");
const IGNORE = "__ignore__";
const PALETTE = ["#ffb454","#7fd1f7","#f77fbe","#9ce37d","#c9a2ff","#ff8a5c",
                 "#5ce0c6","#f7e36b","#7fa1ff","#ff7f9c","#8fe05c","#5cb8ff",
                 "#e09e5c","#b8f75c","#da7ff7","#5cf78f"];
const MONO = getComputedStyle(document.documentElement).getPropertyValue("--mono");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  meta: {},            // pass-through metadata from the detect manifest
  images: [],          // {fileName, relPath, width, height, dets:[], file, key}
  labels: [],          // {name, color}
  cur: -1, sel: -1, hover: -1,
  mode: "select",
  view: {scale: 1, x: 0, y: 0},
  dirty: false,
  viewMode: "both",    // "both" | "box" | "mask"
  focusMode: true,     // when true, selecting a box hides every other box/mask ("focus"); toggleFocus turns this off to always show everything
  brushRadius: 24,     // image-space px
  penDrawingNew: false, // true while placing points for a brand-new pen path
};
// undo entries: {kind:"delete", imgKey, det, pos} or {kind:"mask", imgKey, det, snapshot}
const undoStack = [];

/* ============================== dom ============================== */
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const stage = $("stage");
let rowEls = [];       // filmstrip row elements

/* ============================== helpers ============================== */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const norm = s => String(s).trim();

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 2800);
}

function labelColor(name) {
  const l = state.labels.find(l => l.name === name);
  return l ? l.color : "#9aa3b2";
}
function detDone(d) { return d.label !== null && d.label !== undefined && d.label !== ""; }
function hasMask(d) { return !!(d._bmp || d.seg); }
function imgStats(e) {
  const total = e.dets.length;
  const done = e.dets.filter(detDone).length;
  return {total, done, todo: total - done};
}
function curEntry() { return state.images[state.cur] || null; }

/* ============================== segmentation (COCO RLE <-> bitmap) ============================== */
// Matches pycocotools' maskApi.c rleToString/rleFrString exactly (verified by
// round-tripping real SAM 3 masks through both implementations).
function decodeCountsString(s) {
  const counts = [];
  let p = 0;
  while (p < s.length) {
    let x = 0, k = 0, more = 1;
    while (more) {
      const c = s.charCodeAt(p) - 48;
      x |= (c & 0x1f) << (5 * k);
      more = c & 0x20;
      p++; k++;
      if (!more && (c & 0x10)) x |= (-1 << (5 * k));
    }
    if (counts.length > 2) x += counts[counts.length - 2];
    counts.push(x);
  }
  return counts;
}
function encodeCountsString(counts) {
  let s = "";
  counts.forEach((c, i) => {
    let x = i > 2 ? c - counts[i - 2] : c;
    let more = 1;
    while (more) {
      let ch = x & 0x1f;
      x >>= 5;
      more = (ch & 0x10) ? (x !== -1) : (x !== 0);
      if (more) ch |= 0x20;
      s += String.fromCharCode(ch + 48);
    }
  });
  return s;
}
// RLE ({size:[h,w], counts:"..."}) -> Uint8Array, row-major (index = y*w+x), 0/1
function rleToBitmap(rle) {
  const [h, w] = rle.size;
  const counts = typeof rle.counts === "string" ? decodeCountsString(rle.counts) : rle.counts;
  const bmp = new Uint8Array(h * w);
  let pos = 0, val = 0;
  for (const c of counts) {
    if (val) {
      for (let k = 0; k < c; k++) {
        const p = pos + k;                          // column-major position
        const col = (p / h) | 0, row = p % h;
        bmp[row * w + col] = 1;
      }
    }
    pos += c;
    val ^= 1;
  }
  return bmp;
}
// row-major 0/1 bitmap -> COCO RLE
function bitmapToRle(bmp, h, w) {
  const counts = [];
  let val = 0, run = 0;
  for (let col = 0; col < w; col++) {
    for (let row = 0; row < h; row++) {
      const v = bmp[row * w + col] ? 1 : 0;
      if (v === val) run++;
      else { counts.push(run); val = v; run = 1; }
    }
  }
  counts.push(run);
  return {size: [h, w], counts: encodeCountsString(counts)};
}
function maskColorFor(d) {
  return detDone(d) ? (d.label === IGNORE ? "#7d8697" : labelColor(d.label)) : "#ff5d5d";
}
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
// baked-in alpha for mask fills — display-time globalAlpha (see draw()) dims this down
// for non-selected masks and brings it to full strength for the selected/focused one
const MASK_ALPHA = 165;
// (re)builds the whole offscreen mask canvas from d._bmp — only called on
// decode/undo/label-color-change, never per brush-move (see paintMaskRegion)
function repaintMaskCanvas(d, w, h) {
  if (!d._bmp || !d._maskCtx) return;
  const [r, g, b] = hexToRgb(maskColorFor(d));
  const id = d._maskCtx.createImageData(w, h);
  const bmp = d._bmp, data = id.data;
  for (let i = 0, p = 0; i < bmp.length; i++, p += 4) {
    if (bmp[i]) { data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = MASK_ALPHA; }
  }
  d._maskCtx.putImageData(id, 0, 0);
}
// updates only a sub-rectangle of the offscreen canvas — used during brush strokes
function paintMaskRegion(d, w, x0, y0, rw, rh) {
  const [r, g, b] = hexToRgb(maskColorFor(d));
  const id = d._maskCtx.createImageData(rw, rh);
  const data = id.data;
  for (let yy = 0; yy < rh; yy++) {
    for (let xx = 0; xx < rw; xx++) {
      const v = d._bmp[(y0 + yy) * w + (x0 + xx)];
      const p = (yy * rw + xx) * 4;
      if (v) { data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = MASK_ALPHA; }
    }
  }
  d._maskCtx.putImageData(id, x0, y0);
}
function ensureMaskCanvas(d, w, h) {
  if (d._maskCanvas) return;
  d._maskCanvas = document.createElement("canvas");
  d._maskCanvas.width = w; d._maskCanvas.height = h;
  d._maskCtx = d._maskCanvas.getContext("2d", {willReadFrequently: false});
}
// lazily decodes d.seg (COCO RLE) into an editable bitmap; safe to call every frame
function decodeDetMask(d, w, h) {
  if (d._bmp || !d.seg) return;
  const rle = d.seg;
  if (!Array.isArray(rle.size) || rle.size[0] !== h || rle.size[1] !== w) {
    toast("box #" + d.id + ": mask size doesn't match the image — dropped", "warn");
    d.seg = null;
    return;
  }
  try {
    d._bmp = rleToBitmap(rle);
  } catch (e) {
    toast("box #" + d.id + ": could not decode mask — dropped", "warn");
    d.seg = null;
    return;
  }
  ensureMaskCanvas(d, w, h);
  repaintMaskCanvas(d, w, h);
  updateOutline(d, w, h);
}
// creates a blank editable mask for a box that doesn't have one yet (e.g. a
// manually drawn box) so the user can paint one from scratch
function ensureMaskEditable(d, w, h) {
  if (!d._bmp) {
    d._bmp = new Uint8Array(w * h);
    ensureMaskCanvas(d, w, h);
    repaintMaskCanvas(d, w, h);
  }
  return d._bmp;
}
// paints (or erases) a filled circle of radius r (image px) centered at (cx,cy)
function paintAt(d, w, h, cx, cy, r, erase) {
  const x0 = clamp(Math.floor(cx - r), 0, w - 1), x1 = clamp(Math.ceil(cx + r), 0, w - 1);
  const y0 = clamp(Math.floor(cy - r), 0, h - 1), y1 = clamp(Math.ceil(cy + r), 0, h - 1);
  if (x1 < x0 || y1 < y0) return;
  const val = erase ? 0 : 1, r2 = r * r, bmp = d._bmp;
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      if (dx * dx + dy * dy <= r2) bmp[y * w + x] = val;
    }
  }
  paintMaskRegion(d, w, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  d._bmpDirty = true;
}
// strokes a line of brush dabs between two image-space points (fills gaps on fast moves)
function paintStroke(d, w, h, ax, ay, bx, by, r, erase) {
  const dist = Math.hypot(bx - ax, by - ay);
  const steps = Math.max(1, Math.ceil(dist / Math.max(2, r * 0.5)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    paintAt(d, w, h, ax + (bx - ax) * t, ay + (by - ay) * t, r, erase);
  }
}
// flushes any dirty edited bitmap for every det in an entry back into COCO RLE
// (d.seg) and drops the raster working set — called before switching images
// and before export, so only the current image ever holds decoded bitmaps
function flushMasks(entry) {
  if (!entry) return;
  for (const d of entry.dets) {
    if (d._bmp) {
      if (d._bmpDirty) {
        const hasAny = d._bmp.some(Boolean);
        d.seg = hasAny ? bitmapToRle(d._bmp, entry.height, entry.width) : null;
      }
      delete d._bmp; delete d._bmpDirty; delete d._maskCanvas; delete d._maskCtx;
      delete d._outline; delete d._pen; delete d._newPath;
    }
  }
}
function decodeMasks(entry) {
  if (!entry || !entry.width || !entry.height) return;
  for (const d of entry.dets) decodeDetMask(d, entry.width, entry.height);
}

/* ============================== fill holes (flood fill from border) ============================== */
function fillHoles() {
  const e = curEntry();
  if (!e || state.sel < 0) { toast("select a box first", "warn"); return; }
  const d = e.dets[state.sel];
  if (!hasMask(d)) { toast("box has no mask", "warn"); return; }
  ensureMaskEditable(d, e.width, e.height);
  const w = e.width, h = e.height, bmp = d._bmp;
  const before = bmp.slice();

  const outside = new Uint8Array(w * h);
  const stack = [];
  const mark = p => { if (!bmp[p] && !outside[p]) { outside[p] = 1; stack.push(p); } };
  for (let x = 0; x < w; x++) { mark(x); mark((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { mark(y * w); mark(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop();
    const x = p % w, y = (p / w) | 0;
    if (x > 0) mark(p - 1);
    if (x < w - 1) mark(p + 1);
    if (y > 0) mark(p - w);
    if (y < h - 1) mark(p + w);
  }

  let filled = 0;
  for (let i = 0; i < bmp.length; i++) if (!bmp[i] && !outside[i]) { bmp[i] = 1; filled++; }
  if (!filled) { toast("no enclosed gaps found", "ok"); return; }

  undoStack.push({kind: "mask", imgKey: e.key, det: d, snapshot: before});
  repaintMaskCanvas(d, w, h);
  d._bmpDirty = true;
  state.dirty = true;
  updateOutline(d, w, h);
  toast("filled " + filled + " enclosed px — Ctrl+Z to undo", "ok");
  requestDraw();
}

/* ============================== pen tool (outline vertex editing) ============================== */
// Moore-neighbor boundary trace of the connected component containing (startX, startY).
// Returns an ordered list of [x,y] boundary pixel centers forming one closed loop.
function traceContour(bmp, w, h, startX, startY) {
  const NB = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]];
  const at = (x, y) => (x >= 0 && x < w && y >= 0 && y < h) ? bmp[y * w + x] : 0;
  const pts = [];
  let cx = startX, cy = startY, backDir = 0;
  const maxSteps = w * h * 4 + 8;
  let steps = 0;
  do {
    pts.push([cx, cy]);
    let found = false;
    for (let k = 0; k < 8; k++) {
      const dir = (backDir + k) % 8;
      const [dx, dy] = NB[dir];
      if (at(cx + dx, cy + dy)) {
        cx += dx; cy += dy;
        backDir = (dir + 5) % 8;
        found = true;
        break;
      }
    }
    if (!found) break;
    steps++;
  } while (!(cx === startX && cy === startY) && steps < maxSteps);
  return pts;
}
// recursive Douglas-Peucker on an open point chain
function rdpSimplify(points, eps) {
  if (points.length < 3) return points;
  const [x1, y1] = points[0], [x2, y2] = points[points.length - 1];
  const dx = x2 - x1, dy = y2 - y1;
  const norm = Math.hypot(dx, dy) || 1;
  let maxDist = 0, idx = 0;
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = points[i];
    const d = Math.abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm;
    if (d > maxDist) { maxDist = d; idx = i; }
  }
  if (maxDist > eps) {
    const left = rdpSimplify(points.slice(0, idx + 1), eps);
    const right = rdpSimplify(points.slice(idx), eps);
    return left.slice(0, -1).concat(right);
  }
  return [points[0], points[points.length - 1]];
}
// simplifies a CLOSED loop by splitting it into two chains, simplifying each, and rejoining
function simplifyClosedPolygon(pts, eps) {
  if (pts.length < 3) return pts;
  const mid = Math.floor(pts.length / 2);
  const a = rdpSimplify(pts.slice(0, mid + 1), eps);
  const b = rdpSimplify(pts.slice(mid), eps);
  const combined = a.slice(0, -1).concat(b.slice(0, -1));
  return combined.length >= 3 ? combined : pts;
}
// traces the mask's main outline and simplifies it into a manageable set of editable points
function tracePenPolygon(d, w, h) {
  const bmp = d._bmp;
  let sx = -1, sy = -1;
  outer:
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bmp[y * w + x] && !(x > 0 && bmp[y * w + x - 1])) { sx = x; sy = y; break outer; }
    }
  }
  if (sx < 0) return null;
  const raw = traceContour(bmp, w, h, sx, sy);
  if (raw.length < 3) return null;
  let tol = 2, simplified = simplifyClosedPolygon(raw, tol);
  while (simplified.length > 150 && tol < 40) { tol *= 1.6; simplified = simplifyClosedPolygon(raw, tol); }
  return simplified;
}
// rasterizes an arbitrary closed polygon (image-space points) into a fresh 0/1 bitmap
function polygonToBitmap(poly, w, h) {
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d");
  octx.fillStyle = "#fff";
  octx.beginPath();
  octx.moveTo(poly[0][0], poly[0][1]);
  for (let i = 1; i < poly.length; i++) octx.lineTo(poly[i][0], poly[i][1]);
  octx.closePath();
  octx.fill();
  const id = octx.getImageData(0, 0, w, h).data;
  const bmp = new Uint8Array(w * h);
  for (let i = 0, p = 3; i < bmp.length; i++, p += 4) bmp[i] = id[p] > 127 ? 1 : 0;
  return bmp;
}
// fills d._pen (a polygon) back into d._bmp and the mask canvas
function rasterizePenPolygon(d, w, h) {
  const poly = d._pen;
  if (!poly || poly.length < 3) return;
  d._bmp = polygonToBitmap(poly, w, h);
  d._bmpDirty = true;
  repaintMaskCanvas(d, w, h);
}
// (re)traces the mask's outline and caches it on d._outline — the source both the
// always-visible animated silhouette and a fresh starting point for the pen tool
function updateOutline(d, w, h) {
  d._outline = d._bmp ? tracePenPolygon(d, w, h) : null;
}
// lazily makes sure d._pen exists for the currently-selected box, reusing the cached
// outline if there is one (so toggling in/out of pen mode doesn't lose hand-placed
// points) and only re-tracing from the raster when there's nothing cached yet
function ensurePenOutline(d, w, h) {
  if (d._pen) return true;
  if (!hasMask(d)) return false;
  ensureMaskEditable(d, w, h);
  const poly = d._outline || tracePenPolygon(d, w, h);
  if (!poly) return false;
  d._pen = poly.map(p => p.slice());
  penHover = -1; penSelected = new Set();
  return true;
}
function hitPenVertex(sx, sy) {
  const e = curEntry();
  if (!e || state.sel < 0) return -1;
  const d = e.dets[state.sel];
  if (!d._pen) return -1;
  for (let i = 0; i < d._pen.length; i++) {
    const [px, py] = toScreen(d._pen[i][0], d._pen[i][1]);
    if (Math.hypot(sx - px, sy - py) <= 7) return i;
  }
  return -1;
}
// finds the nearest point on the polygon's edges to a screen coord; returns {idx, pt} where
// idx is the vertex index the new point should be inserted after, or null if not close enough
function hitPenEdge(sx, sy) {
  const e = curEntry();
  if (!e || state.sel < 0) return null;
  const d = e.dets[state.sel];
  if (!d._pen || d._pen.length < 2) return null;
  let best = null, bestDist = Infinity;
  for (let i = 0; i < d._pen.length; i++) {
    const [ax, ay] = d._pen[i], [bx, by] = d._pen[(i + 1) % d._pen.length];
    const [asx, asy] = toScreen(ax, ay), [bsx, bsy] = toScreen(bx, by);
    const dx = bsx - asx, dy = bsy - asy;
    const len2 = dx * dx + dy * dy || 1;
    let t = ((sx - asx) * dx + (sy - asy) * dy) / len2;
    t = clamp(t, 0, 1);
    const psx = asx + dx * t, psy = asy + dy * t;
    const dist = Math.hypot(sx - psx, sy - psy);
    if (dist < bestDist) { bestDist = dist; best = {idx: i, pt: [ax + (bx - ax) * t, ay + (by - ay) * t]}; }
  }
  return bestDist <= 8 ? best : null;
}
function deletePenVertex() {
  const e = curEntry();
  const d = e.dets[state.sel];
  if (!d._pen) return;
  const toRemove = penSelected.size > 0 ? [...penSelected] : (penHover >= 0 ? [penHover] : []);
  if (!toRemove.length) return;
  if (d._pen.length - toRemove.length < 3) { toast("need at least 3 points remaining", "warn"); return; }
  undoStack.push({kind: "pen", imgKey: e.key, det: d, prevPen: d._pen.map(p => p.slice()), prevBmp: d._bmp.slice()});
  toRemove.sort((a, b) => b - a).forEach(i => d._pen.splice(i, 1));
  penSelected = new Set();
  penHover = -1;
  rasterizePenPolygon(d, e.width, e.height);
  toast(toRemove.length > 1 ? toRemove.length + " points deleted — Ctrl+Z to undo" : "point deleted — Ctrl+Z to undo");
  requestDraw();
}
// used by Alt+Right-click: delete one specific vertex directly, regardless of hover/selection
function deletePenVertexAt(vi) {
  penSelected = new Set();
  penHover = vi;
  deletePenVertex();
}

/* ============================== file matching ============================== */
function stripRoot(rel) {
  const parts = rel.split("/").filter(Boolean);
  return parts.length > 1 ? parts.slice(1).join("/") : parts.join("/");
}
function attachFile(file, rel) {
  const relNoRoot = rel ? stripRoot(rel) : "";
  let e = relNoRoot && state.images.find(x => x.relPath === relNoRoot && !x.file);
  if (!e) e = state.images.find(x => x.fileName === file.name && !x.file);
  if (!e) e = relNoRoot && state.images.find(x => x.relPath === relNoRoot);
  if (!e) e = state.images.find(x => x.fileName === file.name);
  if (e) { e.file = file; return {matched: true}; }
  state.images.push({
    fileName: file.name, relPath: relNoRoot || file.name,
    width: 0, height: 0, dets: [], file, key: "new:" + file.name + ":" + state.images.length,
  });
  return {matched: false};
}
async function addImageFiles(files) {
  let matched = 0, added = 0;
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    const r = attachFile(f, f.webkitRelativePath || f._relPath || "");
    r.matched ? matched++ : added++;
  }
  if (!matched && !added) { toast("no image files found", "warn"); return; }
  let msg = matched + " image file(s) matched to the manifest";
  if (added) msg += ", " + added + " new (not in JSON)";
  toast(msg, "ok");
  imageCache.clear(true);
  renderFilmstrip();
  if (state.cur < 0 && state.images.length) {
    const first = state.images.findIndex(e => e.file);
    selectImage(first >= 0 ? first : 0);
  } else if (state.cur >= 0) {
    selectImage(state.cur);   // re-decode in case the current entry just got its file
  }
  checkEmpty();
}

/* ============================== json load / export ============================== */
function loadJsonText(text, srcName) {
  let data;
  try { data = JSON.parse(text); } catch { toast("could not parse " + srcName, "warn"); return false; }

  const oldFiles = new Map();
  for (const e of state.images) if (e.file) {
    oldFiles.set(e.relPath, e.file);
    if (!oldFiles.has(e.fileName)) oldFiles.set(e.fileName, e.file);
  }
  const fileFor = (rel, name) => oldFiles.get(rel) || oldFiles.get(name) || null;

  let ok = false;
  if (data && Array.isArray(data.images) && Array.isArray(data.annotations) && Array.isArray(data.categories)) {
    ok = importCoco(data, fileFor);
  } else if (data && Array.isArray(data.images) && data.images.some(im => Array.isArray(im.detections))) {
    ok = importLegacy(data, fileFor);
    if (ok) toast("legacy manifest migrated — the next EXPORT will be COCO");
  } else {
    toast(srcName + " is not a COCO annotations file", "warn");
    return false;
  }
  if (!ok) return false;

  state.cur = -1; state.sel = -1; state.dirty = false;
  undoStack.length = 0;
  imageCache.clear(true);
  const nBox = state.images.reduce((a, e) => a + e.dets.length, 0);
  toast("loaded " + state.images.length + " image entr" + (state.images.length === 1 ? "y" : "ies") +
        " · " + nBox + " box(es)", "ok");
  renderFilmstrip(); renderRoster();
  const first = state.images.findIndex(e => imgStats(e).todo > 0);
  selectImage(first >= 0 ? first : (state.images.length ? 0 : -1));
  checkEmpty();
  return true;
}

function pushLabel(name, color) {
  name = norm(name);
  const lc = name.toLowerCase();
  if (!name || lc === "unlabeled" || lc === "ignore" || name === IGNORE) return;
  if (state.labels.some(l => l.name.toLowerCase() === lc)) return;
  state.labels.push({name, color: color || PALETTE[state.labels.length % PALETTE.length]});
}

function importCoco(data, fileFor) {
  state.meta = {info: data.info || {}, licenses: data.licenses || []};
  state.labels = [];
  const catById = new Map();
  for (const c of data.categories) {
    catById.set(c.id, c);
    pushLabel(c.name, c.color);
  }
  const byId = new Map();
  state.images = data.images.map((im, i) => {
    const rel = String(im.file_name || "image_" + i);
    const {id, file_name, width, height, ...extraImg} = im;
    const e = {
      fileName: rel.split("/").pop(), relPath: rel,
      width: width | 0, height: height | 0,
      dets: [], file: fileFor(rel, rel.split("/").pop()),
      key: rel, cocoId: id ?? i + 1, extraImg,
    };
    byId.set(e.cocoId, e);
    return e;
  });
  let dropped = 0;
  for (const a of data.annotations) {
    const e = byId.get(a.image_id);
    if (!e || !Array.isArray(a.bbox)) { dropped++; continue; }
    const {id, image_id, category_id, bbox, area, score, source, segmentation, ...extra} = a;
    const [x, y, w, h] = bbox.map(Number);
    const cat = catById.get(category_id);
    const lc = cat ? String(cat.name).trim().toLowerCase() : "unlabeled";
    const label = lc === "unlabeled" ? null : lc === "ignore" ? IGNORE : norm(cat.name);
    e.dets.push({
      id: e.dets.length + 1, bbox: [x, y, x + w, y + h],
      score: score ?? null, label, source: source || "detector",
      seg: segmentation || null, extra,
    });
  }
  if (dropped) toast(dropped + " annotation(s) skipped (missing image or bbox)", "warn");
  state.nextImgId = Math.max(0, ...state.images.map(e => e.cocoId)) + 1;
  return true;
}

function importLegacy(data, fileFor) {
  state.meta = {info: {description: "migrated from a legacy CAST/LIST manifest",
                       legacy_detector: data.detector || null},
                licenses: []};
  state.labels = [];
  for (const l of (data.label_set || [])) pushLabel(l.name, l.color);
  state.images = data.images.map((e, i) => ({
    fileName: e.file_name || "image_" + i,
    relPath: e.rel_path || e.file_name || "image_" + i,
    width: e.width | 0, height: e.height | 0,
    dets: (e.detections || []).map((d, j) => ({
      id: j + 1, bbox: d.bbox.map(Number), score: d.score ?? null,
      label: d.label ?? null, source: d.source || "detector",
      seg: d.segmentation || null, extra: {},
    })),
    file: fileFor(e.rel_path || e.file_name, e.file_name),
    key: (e.rel_path || e.file_name || i) + "", cocoId: i + 1, extraImg: {},
  }));
  for (const e of state.images) for (const d of e.dets)
    if (detDone(d) && d.label !== IGNORE) pushLabel(d.label);
  state.nextImgId = state.images.length + 1;
  return true;
}

function exportJSON() {
  if (!state.images.length) { toast("nothing to export yet", "warn"); return; }
  flushMasks(curEntry());
  let nIgn = 0, nUnl = 0;
  for (const e of state.images) for (const d of e.dets) {
    if (d.label === IGNORE) nIgn++;
    else if (!detDone(d)) nUnl++;
  }
  const cats = [];
  const catId = new Map();
  for (const l of state.labels) {
    cats.push({id: cats.length + 1, name: l.name, supercategory: "character", color: l.color});
    catId.set(l.name, cats.length);
  }
  if (nIgn) { cats.push({id: cats.length + 1, name: "ignore", supercategory: "meta"}); catId.set(IGNORE, cats.length); }
  if (nUnl) { cats.push({id: cats.length + 1, name: "unlabeled", supercategory: "meta"}); catId.set(null, cats.length); }

  state.nextImgId = state.nextImgId || 1;
  const cocoImages = [], cocoAnns = [];
  let annId = 0;
  for (const e of state.images) {
    if (e.cocoId == null) e.cocoId = state.nextImgId++;
    cocoImages.push({...(e.extraImg || {}), id: e.cocoId, file_name: e.relPath,
                     width: e.width, height: e.height});
    for (const d of e.dets) {
      const key = !detDone(d) ? null : d.label;
      if (key !== null && key !== IGNORE && !catId.has(key)) {
        cats.push({id: cats.length + 1, name: key, supercategory: "character"});
        catId.set(key, cats.length);
      }
      const [x0, y0, x1, y1] = d.bbox.map(Math.round);
      const w = x1 - x0, h = y1 - y0;
      const ann = {
        ...(d.extra || {}),
        id: ++annId, image_id: e.cocoId,
        category_id: catId.get(key),
        bbox: [x0, y0, w, h], area: w * h,
        iscrowd: (d.extra && d.extra.iscrowd) || 0,
        source: d.source || "detector",
      };
      if (d.score != null) ann.score = d.score;
      if (d.seg) ann.segmentation = d.seg;
      cocoAnns.push(ann);
    }
  }
  const out = {
    info: {...(state.meta.info || {}),
           description: (state.meta.info && state.meta.info.description) ||
                        "CAST/LIST character annotations",
           date_labeled: new Date().toISOString(),
           labeled_by: "2_annotate.html"},
    licenses: state.meta.licenses || [],
    categories: cats,
    images: cocoImages,
    annotations: cocoAnns,
  };
  const json = JSON.stringify(out, null, 1);

  // Electron app shell with a known source folder (picked via the native
  // FOLDER dialog): save straight back into it -- no manual "move the
  // download into the folder" step.
  if (window.sam3 && annotateSaveFolder) {
    window.sam3.writeTextFile(joinPath(annotateSaveFolder, "annotations_labeled.json"), json)
      .then(() => {
        state.dirty = false;
        toast((nUnl ? "saved (" + nUnl + " box(es) still unlabeled) -> " : "saved -- all boxes labeled -> ")
              + annotateSaveFolder, nUnl ? "" : "ok");
      })
      .catch(err => toast("save failed: " + err.message, "warn"));
    return;
  }

  const blob = new Blob([json], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations_labeled.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  state.dirty = false;
  toast(nUnl ? "exported COCO (" + nUnl + " box(es) still unlabeled)"
             : "exported COCO — all boxes labeled", nUnl ? "" : "ok");
}

/* ============================== image cache ============================== */
const imageCache = {
  map: new Map(), // key -> {img, url}
  async get(entry) {
    if (!entry || !entry.file) return null;
    const hit = this.map.get(entry.key);
    if (hit) { this.map.delete(entry.key); this.map.set(entry.key, hit); return hit.img; }
    const url = URL.createObjectURL(entry.file);
    const img = new Image();
    img.src = url;
    try { await img.decode(); }
    catch { URL.revokeObjectURL(url); toast("could not decode " + entry.fileName, "warn"); return null; }
    if (!entry.width || !entry.height) {
      entry.width = img.naturalWidth; entry.height = img.naturalHeight;
    } else if (entry.width !== img.naturalWidth || entry.height !== img.naturalHeight) {
      const sx = img.naturalWidth / entry.width, sy = img.naturalHeight / entry.height;
      for (const d of entry.dets) {
        d.bbox = [d.bbox[0]*sx, d.bbox[1]*sy, d.bbox[2]*sx, d.bbox[3]*sy].map(Math.round);
        delete d._bmp; delete d._bmpDirty; delete d._maskCanvas; delete d._maskCtx; // can't rescale a raster mask; re-decode will drop it (size no longer matches)
      }
      entry.width = img.naturalWidth; entry.height = img.naturalHeight;
      state.dirty = true;
      toast(entry.fileName + ": size differs from manifest — boxes rescaled" +
            (entry.dets.some(d => d.seg) ? " (masks dropped)" : ""), "warn");
    }
    this.map.set(entry.key, {img, url});
    while (this.map.size > 8) {
      const [k, v] = this.map.entries().next().value;
      this.map.delete(k); URL.revokeObjectURL(v.url);
    }
    return img;
  },
  clear(revoke) {
    if (revoke) for (const v of this.map.values()) URL.revokeObjectURL(v.url);
    this.map.clear();
  },
};
let curImg = null;      // decoded Image for current entry
let loadSeq = 0;

/* ============================== navigation ============================== */
async function selectImage(i) {
  const prev = state.cur;
  if (prev >= 0 && prev !== i) flushMasks(state.images[prev]);
  if (i < 0 || i >= state.images.length) { state.cur = -1; curImg = null; renderAll(); requestDraw(); return; }
  state.cur = i; state.sel = -1; state.hover = -1;
  if (prev >= 0 && prev !== i) updateRow(prev);
  const entry = state.images[i];
  const seq = ++loadSeq;
  curImg = null;
  fitView();               // fit from manifest dims first for instant layout
  decodeMasks(entry);
  renderAll(); requestDraw();
  const img = await imageCache.get(entry);
  if (seq !== loadSeq) return;
  curImg = img;
  fitView();
  decodeMasks(entry);       // covers the size-mismatch-rescale case in imageCache.get
  // deliberately no auto-select here: every box/mask should show on load
  // (state.sel stays -1 from above) until the user actually clicks one
  renderAll(); requestDraw();
}
function stepImage(dir) {
  if (!state.images.length) return;
  selectImage(clamp(state.cur + dir, 0, state.images.length - 1));
}
function jumpNextUnlabeled() {
  const n = state.images.length;
  if (!n) return;
  const e = curEntry();
  if (e) {
    const start = state.sel;
    for (let k = 1; k <= e.dets.length; k++) {
      const j = ((start < 0 ? -1 : start) + k) % e.dets.length;
      if (!detDone(e.dets[j])) { state.sel = j; ensureVisible(e.dets[j]); renderAll(); requestDraw(); return; }
    }
  }
  for (let k = 1; k <= n; k++) {
    const j = (state.cur + k) % n;
    if (imgStats(state.images[j]).todo > 0) { selectImage(j); return; }
  }
  toast("every box is labeled — hit EXPORT", "ok");
}

/* ============================== labels ============================== */
function addLabel(nameRaw) {
  const name = norm(nameRaw);
  if (!name) return null;
  const lc = name.toLowerCase();
  if (lc === "ignore" || name === IGNORE) { toast("'ignore' is built in — press 0", "warn"); return null; }
  if (lc === "unlabeled") { toast("'unlabeled' is reserved for pending boxes", "warn"); return null; }
  const exist = state.labels.find(l => l.name.toLowerCase() === name.toLowerCase());
  if (exist) return exist;
  const l = {name, color: PALETTE[state.labels.length % PALETTE.length]};
  state.labels.push(l);
  state.dirty = true;
  renderRoster();
  return l;
}
function assign(name) {
  const e = curEntry();
  if (!e) return;
  if (state.sel < 0) {
    const j = e.dets.findIndex(d => !detDone(d));
    if (j < 0) { toast("no unlabeled box in this image", "warn"); return; }
    state.sel = j;
  }
  const d = e.dets[state.sel];
  d.label = name;
  state.dirty = true;
  if (d._bmp) repaintMaskCanvas(d, e.width, e.height);
  const next = e.dets.findIndex((x, j) => j !== state.sel && !detDone(x));
  updateRow(state.cur);
  applyFilmFilter();
  if (next >= 0) { state.sel = next; ensureVisible(e.dets[next]); }
  else toast("image done — Enter / N for the next one", "ok");
  renderAll(); requestDraw();
}
function renameLabel(l) {
  const to = norm(prompt('Rename "' + l.name + '" to:', l.name) || "");
  if (!to || to === l.name) return;
  if (state.labels.some(x => x !== l && x.name.toLowerCase() === to.toLowerCase())) { toast("that name already exists", "warn"); return; }
  for (const e of state.images) for (const d of e.dets) if (d.label === l.name) d.label = to;
  l.name = to; state.dirty = true;
  const cur = curEntry();
  if (cur) for (const d of cur.dets) if (d._bmp) repaintMaskCanvas(d, cur.width, cur.height);
  renderAll(); requestDraw();
}
function removeLabel(l) {
  const used = state.images.reduce((a, e) => a + e.dets.filter(d => d.label === l.name).length, 0);
  if (used && !confirm('Remove "' + l.name + '"? ' + used + " box(es) will go back to unlabeled.")) return;
  for (const e of state.images) for (const d of e.dets) if (d.label === l.name) d.label = null;
  state.labels = state.labels.filter(x => x !== l);
  state.dirty = true;
  const cur = curEntry();
  if (cur) for (const d of cur.dets) if (d._bmp) repaintMaskCanvas(d, cur.width, cur.height);
  renderFilmstrip(); renderAll(); requestDraw();
}

/* ============================== boxes ============================== */
function deleteSel() {
  const e = curEntry();
  if (!e || state.sel < 0) return;
  const det = e.dets[state.sel];
  undoStack.push({kind: "delete", imgKey: e.key, det, pos: state.sel});
  e.dets.splice(state.sel, 1);
  state.sel = Math.min(state.sel, e.dets.length - 1);
  state.dirty = true;
  toast("box #" + det.id + " deleted — U to undo");
  updateRow(state.cur); renderAll(); requestDraw();
}
function undo() {
  const u = undoStack.pop();
  if (!u) { toast("nothing to undo", "warn"); return; }
  const i = state.images.findIndex(e => e.key === u.imgKey);
  if (i < 0) return;

  if (u.kind === "mask") {
    if (i !== state.cur) selectImage(i).then(() => restoreMaskUndo(u));
    else restoreMaskUndo(u);
    return;
  }
  if (u.kind === "pen") {
    if (i !== state.cur) selectImage(i).then(() => restorePenUndo(u));
    else restorePenUndo(u);
    return;
  }
  if (u.kind === "move") {
    const restore = () => {
      u.det.bbox = u.prevBbox;
      state.dirty = true;
      toast("box move undone");
      renderAll(); requestDraw();
    };
    if (i !== state.cur) selectImage(i).then(restore);
    else restore();
    return;
  }

  const e = state.images[i];
  e.dets.splice(Math.min(u.pos, e.dets.length), 0, u.det);
  state.dirty = true;
  if (i !== state.cur) selectImage(i);
  else { state.sel = Math.min(u.pos, e.dets.length - 1); updateRow(i); renderAll(); requestDraw(); }
}
function restoreMaskUndo(u) {
  const e = curEntry();
  const d = u.det;
  d._bmp = u.snapshot;
  ensureMaskCanvas(d, e.width, e.height);
  repaintMaskCanvas(d, e.width, e.height);
  d._bmpDirty = true;
  state.dirty = true;
  updateOutline(d, e.width, e.height);
  if (state.mode === "pen") d._pen = d._outline ? d._outline.map(p => p.slice()) : undefined;
  toast("mask edit undone");
  requestDraw();
}
function restorePenUndo(u) {
  const e = curEntry();
  const d = u.det;
  d._bmp = u.prevBmp;
  d._outline = u.prevPen;
  d._pen = state.mode === "pen" ? u.prevPen.map(p => p.slice()) : undefined;
  ensureMaskCanvas(d, e.width, e.height);
  repaintMaskCanvas(d, e.width, e.height);
  d._bmpDirty = true;
  state.dirty = true;
  penHover = -1;
  penSelected = new Set();
  toast("pen edit undone");
  requestDraw();
}
function nudgeSel(dx, dy) {
  const e = curEntry();
  if (!e || state.sel < 0) return;
  const d = e.dets[state.sel];
  const [x0, y0, x1, y1] = d.bbox, w = x1 - x0, h = y1 - y0;
  const nx = clamp(x0 + dx, 0, e.width - w), ny = clamp(y0 + dy, 0, e.height - h);
  if (nx === x0 && ny === y0) return;
  undoStack.push({kind: "move", imgKey: e.key, det: d, prevBbox: [...d.bbox]});
  d.bbox = [nx, ny, nx + w, ny + h];
  state.dirty = true;
  requestDraw();
}

/* ============================== view / transforms ============================== */
function toScreen(x, y) { return [x * state.view.scale + state.view.x, y * state.view.scale + state.view.y]; }
function toImage(sx, sy) { return [(sx - state.view.x) / state.view.scale, (sy - state.view.y) / state.view.scale]; }
function stageSize() { const r = stage.getBoundingClientRect(); return [r.width, r.height]; }

function fitView() {
  const e = curEntry();
  if (!e || !e.width) return;
  const [w, h] = stageSize();
  const s = Math.min(w / e.width, h / e.height) * 0.96;
  state.view.scale = s;
  state.view.x = (w - e.width * s) / 2;
  state.view.y = (h - e.height * s) / 2;
  updateZoomLabel();
}
function zoomAt(sx, sy, factor) {
  const v = state.view;
  const ns = clamp(v.scale * factor, 0.03, 40);
  v.x = sx - (sx - v.x) * (ns / v.scale);
  v.y = sy - (sy - v.y) * (ns / v.scale);
  v.scale = ns;
  updateZoomLabel(); requestDraw();
}
function zoomToBox(d) {
  const e = curEntry(); if (!e || !d) return;
  const [w, h] = stageSize();
  const bw = d.bbox[2] - d.bbox[0], bh = d.bbox[3] - d.bbox[1];
  const s = clamp(Math.min(w / bw, h / bh) * 0.55, 0.03, 40);
  state.view.scale = s;
  state.view.x = w / 2 - (d.bbox[0] + bw / 2) * s;
  state.view.y = h / 2 - (d.bbox[1] + bh / 2) * s;
  updateZoomLabel(); requestDraw();
}
function ensureVisible(d) {
  const [w, h] = stageSize();
  const [x0, y0] = toScreen(d.bbox[0], d.bbox[1]);
  const [x1, y1] = toScreen(d.bbox[2], d.bbox[3]);
  if (x0 >= 0 && y0 >= 0 && x1 <= w && y1 <= h) return;
  state.view.x += (w / 2) - (x0 + x1) / 2;
  state.view.y += (h / 2) - (y0 + y1) / 2;
}
function updateZoomLabel() { $("zLabel").textContent = Math.round(state.view.scale * 100) + "%"; }

/* ============================== hit testing ============================== */
const HANDLES = ["nw","n","ne","e","se","s","sw","w"];
function handlePoints(d) {
  const [x0, y0] = toScreen(d.bbox[0], d.bbox[1]);
  const [x1, y1] = toScreen(d.bbox[2], d.bbox[3]);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return {nw:[x0,y0], n:[mx,y0], ne:[x1,y0], e:[x1,my], se:[x1,y1], s:[mx,y1], sw:[x0,y1], w:[x0,my]};
}
function hitHandle(sx, sy) {
  const e = curEntry();
  if (!e || state.sel < 0 || state.viewMode === "mask") return null;
  const pts = handlePoints(e.dets[state.sel]);
  for (const k of HANDLES) {
    const [px, py] = pts[k];
    if (Math.abs(sx - px) <= 7 && Math.abs(sy - py) <= 7) return k;
  }
  return null;
}
function hitBox(ix, iy) {
  const e = curEntry();
  if (!e) return -1;
  let best = -1, bestArea = Infinity;
  e.dets.forEach((d, i) => {
    const [x0, y0, x1, y1] = d.bbox;
    if (ix >= x0 && ix <= x1 && iy >= y0 && iy <= y1) {
      const a = (x1 - x0) * (y1 - y0);
      if (a < bestArea) { bestArea = a; best = i; }
    }
  });
  return best;
}

/* ============================== drawing ============================== */
let needDraw = true;
function requestDraw() { needDraw = true; }
let dashT = 0;
(function loop(ts) {
  const animating = state.sel >= 0 && !reduceMotion;
  if (animating) dashT = (ts / 40) % 20;
  if (needDraw || animating) { draw(); needDraw = false; }
  requestAnimationFrame(loop);
})(0);

new ResizeObserver(() => { requestDraw(); }).observe(stage);

function draw() {
  const [w, h] = stageSize();
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const e = curEntry();
  if (!e) return;

  if (curImg) {
    ctx.imageSmoothingEnabled = state.view.scale < 2.5;
    ctx.imageSmoothingQuality = "high";
    const [ox, oy] = toScreen(0, 0);
    ctx.drawImage(curImg, ox, oy, e.width * state.view.scale, e.height * state.view.scale);
  } else if (e.width) {
    const [ox, oy] = toScreen(0, 0);
    ctx.fillStyle = "rgba(255,255,255,.04)";
    ctx.fillRect(ox, oy, e.width * state.view.scale, e.height * state.view.scale);
    ctx.fillStyle = "#5d6572";
    ctx.font = "12px " + MONO;
    ctx.fillText(e.file ? "decoding…" : "image file not loaded — use IMAGES", ox + 12, oy + 22);
  }

  const showBox = state.viewMode !== "mask";
  const showMask = state.viewMode !== "box";
  const hasSel = state.focusMode && state.sel >= 0;

  if (showMask) {
    const [ox, oy] = toScreen(0, 0);
    e.dets.forEach((d, i) => {
      if (!d._maskCanvas) return;
      const isSel = i === state.sel;
      if (hasSel && !isSel && i !== state.hover) return; // focus: hide other masks while one is selected
      ctx.save();
      ctx.globalAlpha = isSel ? 1 : 0.55;
      ctx.drawImage(d._maskCanvas, ox, oy, e.width * state.view.scale, e.height * state.view.scale);
      ctx.restore();
    });
  }

  e.dets.forEach((d, i) => {
    const sel = i === state.sel, hov = i === state.hover;
    if (hasSel && !sel && !hov) return; // focus: hide other boxes while one is selected
    const done = detDone(d);
    const col = done ? (d.label === IGNORE ? "#7d8697" : labelColor(d.label)) : "#ff5d5d";
    const [x0, y0] = toScreen(d.bbox[0], d.bbox[1]);
    const [x1, y1] = toScreen(d.bbox[2], d.bbox[3]);
    const bw = x1 - x0, bh = y1 - y0;

    ctx.save();
    if (showBox) {
      if (hov || sel) { ctx.fillStyle = col + (sel ? "26" : "14"); ctx.fillRect(x0, y0, bw, bh); }
      ctx.lineWidth = sel ? 3 : hov ? 2.5 : 2;
      ctx.strokeStyle = col;
      ctx.setLineDash(done ? [] : [7, 5]);
      if (sel) { ctx.shadowColor = col; ctx.shadowBlur = 12; }
      ctx.strokeRect(x0, y0, bw, bh);
      ctx.shadowBlur = 0;
      ctx.setLineDash([]);
    }

    if (sel && showBox) {
      const pts = handlePoints(d);
      for (const k of HANDLES) {
        const [px, py] = pts[k];
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#10131a";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.rect(px - 4, py - 4, 8, 8); ctx.fill(); ctx.stroke();
      }
    }

    if (showBox) {
      // chip
      const txt = done
        ? (d.label === IGNORE ? "ignore" : d.label)
        : "#" + d.id + (d.score != null ? " · " + Number(d.score).toFixed(2) : "");
      ctx.font = "700 11px " + MONO;
      const tw = ctx.measureText(txt).width;
      const cw = tw + 12, ch = 17;
      let cx = x0, cy = y0 - ch - 3;
      if (cy < 2) cy = y0 + 3;
      if (cx + cw > w - 2) cx = w - 2 - cw;
      if (cx < 2) cx = 2;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.roundRect(cx, cy, cw, ch, 4);
      ctx.fill();
      ctx.fillStyle = "#10131a";
      ctx.fillText(txt, cx + 6, cy + 12.5);
    }
    ctx.restore();
  });

  if (dragState && dragState.kind === "draw" && dragState.rect) {
    const [a, b, c, dd] = dragState.rect;
    const [x0, y0] = toScreen(a, b), [x1, y1] = toScreen(c, dd);
    ctx.strokeStyle = "#ffb454";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
  }

  if (state.mode === "mask" && lastPointer) {
    const r = state.brushRadius * state.view.scale;
    ctx.beginPath();
    ctx.arc(lastPointer[0], lastPointer[1], r, 0, Math.PI * 2);
    ctx.strokeStyle = dragState && dragState.erase ? "#ff5d5d" : "#ffb454";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // animated (marching-ants) silhouette outline for the selected box's mask — visible in
  // any mode so the segmentation reads clearly, not just while actively pen-editing
  if (showMask && state.mode !== "pen" && state.sel >= 0) {
    const d = e.dets[state.sel];
    if (d && d._outline) {
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineDashOffset = -dashT;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      d._outline.forEach(([px, py], i) => {
        const [osx, osy] = toScreen(px, py);
        if (i === 0) ctx.moveTo(osx, osy); else ctx.lineTo(osx, osy);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }

  if (state.mode === "pen" && state.sel >= 0) {
    const d = e.dets[state.sel];
    if (d && !d._newPath && !d._pen) ensurePenOutline(d, e.width, e.height);

    if (d && d._newPath && d._newPath.length) {
      ctx.save();
      ctx.strokeStyle = "#9ce37d";
      ctx.setLineDash([5, 4]);
      ctx.lineDashOffset = -dashT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      d._newPath.forEach(([px, py], i) => {
        const [nsx, nsy] = toScreen(px, py);
        if (i === 0) ctx.moveTo(nsx, nsy); else ctx.lineTo(nsx, nsy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      d._newPath.forEach(([px, py], i) => {
        const [nsx, nsy] = toScreen(px, py);
        ctx.fillStyle = i === 0 ? "#fff" : "#9ce37d";
        ctx.strokeStyle = "#10131a";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(nsx, nsy, i === 0 ? 5 : 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      });
      ctx.restore();
    } else if (d && d._pen) {
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineDashOffset = -dashT;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      d._pen.forEach(([px, py], i) => {
        const [psx, psy] = toScreen(px, py);
        if (i === 0) ctx.moveTo(psx, psy); else ctx.lineTo(psx, psy);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      d._pen.forEach(([px, py], i) => {
        const [psx, psy] = toScreen(px, py);
        const isSel = penSelected.has(i);
        ctx.fillStyle = i === penHover ? "#fff" : isSel ? "#7fd1f7" : "#ffb454";
        ctx.strokeStyle = "#10131a";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(psx, psy, (i === penHover || isSel) ? 5 : 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      });
      ctx.restore();
    }

    if (dragState && dragState.kind === "pen-marquee") {
      ctx.save();
      ctx.strokeStyle = "#7fd1f7";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      const mx0 = Math.min(dragState.sx0, dragState.sx1), my0 = Math.min(dragState.sy0, dragState.sy1);
      ctx.strokeRect(mx0, my0, Math.abs(dragState.sx1 - dragState.sx0), Math.abs(dragState.sy1 - dragState.sy0));
      ctx.setLineDash([]);
      ctx.restore();
    }
  }
}

/* ============================== pointer interactions ============================== */
let dragState = null;
let spaceDown = false;
let lastPointer = null;   // [sx, sy] screen coords, for the mask-mode brush cursor ring
let penHover = -1;        // index into the selected det's _pen array, or -1
let penSelected = new Set(); // marquee-selected vertex indices into the selected det's _pen
let pathResolveDet = null;   // det awaiting a Replace/Merge/Subtract/Cancel choice for a new path

const CURSORS = {nw:"nwse-resize", se:"nwse-resize", ne:"nesw-resize", sw:"nesw-resize",
                 n:"ns-resize", s:"ns-resize", e:"ew-resize", w:"ew-resize"};

cv.addEventListener("pointerdown", ev => {
  if (!curEntry()) return;
  cv.setPointerCapture(ev.pointerId);
  const sx = ev.offsetX, sy = ev.offsetY;
  const [ix, iy] = toImage(sx, sy);

  if (ev.button === 1 || spaceDown) { dragState = {kind: "pan", lx: ev.clientX, ly: ev.clientY}; cv.classList.add("grabbing"); return; }

  if (state.mode === "mask") {
    if (ev.button !== 0 && ev.button !== 2) return;
    const e = curEntry();
    if (state.sel < 0) return;
    const det = e.dets[state.sel];
    ensureMaskEditable(det, e.width, e.height);
    undoStack.push({kind: "mask", imgKey: e.key, det, snapshot: det._bmp.slice()});
    const erase = ev.button === 2 || ev.altKey;
    dragState = {kind: "mask", det, erase, lx: ix, ly: iy};
    paintAt(det, e.width, e.height, ix, iy, state.brushRadius, erase);
    requestDraw();
    return;
  }

  if (state.mode === "pen") {
    const e = curEntry();
    if (state.sel < 0) return;
    const d = e.dets[state.sel];

    if (state.penDrawingNew) {
      if (ev.button !== 0) return;
      if (!d._newPath) d._newPath = [];
      if (d._newPath.length >= 3) {
        const [fx, fy] = toScreen(d._newPath[0][0], d._newPath[0][1]);
        if (Math.hypot(sx - fx, sy - fy) <= 8) { openPathResolve(d); return; }
      }
      d._newPath.push([clamp(ix, 0, e.width), clamp(iy, 0, e.height)]);
      requestDraw();
      return;
    }

    if (!d._pen && !ensurePenOutline(d, e.width, e.height)) return;

    if (ev.button === 2 && ev.altKey) {
      const vi = hitPenVertex(sx, sy);
      if (vi >= 0) deletePenVertexAt(vi);
      return;
    }
    if (ev.button !== 0) return;

    const vi = hitPenVertex(sx, sy);
    if (vi >= 0) {
      if (!penSelected.has(vi)) penSelected = new Set([vi]);
      undoStack.push({kind: "pen", imgKey: e.key, det: d, prevPen: d._pen.map(p => p.slice()), prevBmp: d._bmp.slice()});
      dragState = {kind: "pen-vertex", det: d, group: [...penSelected],
                   startPts: [...penSelected].map(i => d._pen[i].slice()), ax: ix, ay: iy};
      return;
    }
    const edge = hitPenEdge(sx, sy);
    if (edge) {
      undoStack.push({kind: "pen", imgKey: e.key, det: d, prevPen: d._pen.map(p => p.slice()), prevBmp: d._bmp.slice()});
      d._pen.splice(edge.idx + 1, 0, edge.pt);
      penSelected = new Set([edge.idx + 1]);
      dragState = {kind: "pen-vertex", det: d, group: [edge.idx + 1], startPts: [edge.pt.slice()], ax: ix, ay: iy};
      requestDraw();
      return;
    }
    dragState = {kind: "pen-marquee", sx0: sx, sy0: sy, sx1: sx, sy1: sy};
    return;
  }
  if (ev.button !== 0) return;

  if (state.mode === "draw") {
    const e = curEntry();
    dragState = {kind: "draw", ax: clamp(ix, 0, e.width), ay: clamp(iy, 0, e.height), rect: null};
    return;
  }
  const hk = hitHandle(sx, sy);
  if (hk) {
    const det = curEntry().dets[state.sel];
    undoStack.push({kind: "move", imgKey: curEntry().key, det, prevBbox: [...det.bbox]});
    dragState = {kind: "resize", which: hk, orig: [...det.bbox]};
    return;
  }
  const i = hitBox(ix, iy);
  if (i >= 0) {
    if (state.sel !== i) { state.sel = i; renderBoxlist(); renderFooter(); requestDraw(); }
    // boxes aren't drawn in MASKS-only view, so don't let the (invisible)
    // bbox be dragged/moved there either — clicking still selects it above,
    // it just can't be dragged around unseen.
    dragState = state.viewMode === "mask"
      ? null
      : {kind: "maybe-move", ax: ix, ay: iy, sx: ev.clientX, sy: ev.clientY,
         orig: [...curEntry().dets[i].bbox], undoPushed: false};
  } else {
    dragState = {kind: "maybe-pan", sx: ev.clientX, sy: ev.clientY, lx: ev.clientX, ly: ev.clientY};
  }
});

cv.addEventListener("pointermove", ev => {
  const sx = ev.offsetX, sy = ev.offsetY;
  const [ix, iy] = toImage(sx, sy);
  $("fCoords").textContent = curEntry() ? Math.round(ix) + "," + Math.round(iy) : "—";
  lastPointer = [sx, sy];

  if (!dragState) {
    let cur = "default";
    const hk = hitHandle(sx, sy);
    if (state.mode === "mask") { cur = "none"; requestDraw(); }
    else if (state.mode === "pen") {
      if (state.penDrawingNew) cur = "crosshair";
      else {
        const vi = hitPenVertex(sx, sy);
        if (vi !== penHover) { penHover = vi; requestDraw(); }
        cur = vi >= 0 ? "grab" : (hitPenEdge(sx, sy) ? "copy" : "crosshair");
      }
    }
    else if (state.mode === "draw") cur = "crosshair";
    else if (hk) cur = CURSORS[hk];
    else {
      const i = hitBox(ix, iy);
      if (i !== state.hover) { state.hover = i; requestDraw(); renderBoxlist(); }
      if (i >= 0) cur = "pointer";
      else if (spaceDown) cur = "grab";
    }
    cv.style.cursor = cur;
    return;
  }
  if (dragState.kind === "mask") {
    const e = curEntry();
    const d = dragState;
    paintStroke(d.det, e.width, e.height, d.lx, d.ly, ix, iy, state.brushRadius, d.erase);
    d.lx = ix; d.ly = iy;
    requestDraw();
    return;
  }
  if (dragState.kind === "pen-vertex") {
    const e = curEntry();
    const d = dragState;
    const dx = ix - d.ax, dy = iy - d.ay;
    d.group.forEach((idx, k) => {
      const [ox, oy] = d.startPts[k];
      d.det._pen[idx] = [clamp(ox + dx, 0, e.width), clamp(oy + dy, 0, e.height)];
    });
    rasterizePenPolygon(d.det, e.width, e.height);
    requestDraw();
    return;
  }
  if (dragState.kind === "pen-marquee") {
    dragState.sx1 = sx; dragState.sy1 = sy;
    requestDraw();
    return;
  }

  const d = dragState;
  const e = curEntry();
  if (d.kind === "pan" || d.kind === "maybe-pan") {
    if (d.kind === "maybe-pan" &&
        Math.hypot(ev.clientX - d.sx, ev.clientY - d.sy) > 3) { d.kind = "pan"; cv.classList.add("grabbing"); }
    if (d.kind === "pan") {
      state.view.x += ev.clientX - d.lx;
      state.view.y += ev.clientY - d.ly;
      requestDraw();
    }
    d.lx = ev.clientX; d.ly = ev.clientY;
  } else if (d.kind === "maybe-move" || d.kind === "move") {
    if (d.kind === "maybe-move" &&
        Math.hypot(ev.clientX - d.sx, ev.clientY - d.sy) > 3) {
      d.kind = "move";
      if (!d.undoPushed) {
        undoStack.push({kind: "move", imgKey: e.key, det: e.dets[state.sel], prevBbox: [...d.orig]});
        d.undoPushed = true;
      }
    }
    if (d.kind === "move" && state.sel >= 0) {
      const det = e.dets[state.sel];
      const w = d.orig[2] - d.orig[0], h = d.orig[3] - d.orig[1];
      const nx = clamp(d.orig[0] + (ix - d.ax), 0, e.width - w);
      const ny = clamp(d.orig[1] + (iy - d.ay), 0, e.height - h);
      det.bbox = [nx, ny, nx + w, ny + h];
      state.dirty = true;
      requestDraw();
    }
  } else if (d.kind === "resize" && state.sel >= 0) {
    const det = e.dets[state.sel];
    let [x0, y0, x1, y1] = d.orig;
    const cix = clamp(ix, 0, e.width), ciy = clamp(iy, 0, e.height);
    if (d.which.includes("w")) x0 = cix;
    if (d.which.includes("e")) x1 = cix;
    if (d.which.includes("n")) y0 = ciy;
    if (d.which.includes("s")) y1 = ciy;
    det.bbox = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
    state.dirty = true;
    requestDraw();
  } else if (d.kind === "draw") {
    const bx = clamp(ix, 0, e.width), by = clamp(iy, 0, e.height);
    d.rect = [Math.min(d.ax, bx), Math.min(d.ay, by), Math.max(d.ax, bx), Math.max(d.ay, by)];
    requestDraw();
  }
});

cv.addEventListener("pointerup", ev => {
  const d = dragState;
  dragState = null;
  cv.classList.remove("grabbing");
  if (!d) return;
  const e = curEntry();
  if (d.kind === "maybe-pan") {
    if (state.sel !== -1) { state.sel = -1; renderBoxlist(); renderFooter(); }
    requestDraw();
  } else if (d.kind === "draw" && d.rect && e) {
    const [x0, y0, x1, y1] = d.rect.map(Math.round);
    if (x1 - x0 >= 4 && y1 - y0 >= 4) {
      const id = e.dets.reduce((m, x) => Math.max(m, x.id), 0) + 1;
      e.dets.push({id, bbox: [x0, y0, x1, y1], score: null, label: null, source: "manual"});
      state.sel = e.dets.length - 1;
      state.dirty = true;
      setMode("select");
      updateRow(state.cur);
      renderAll();
    }
    requestDraw();
  } else if ((d.kind === "move" || d.kind === "resize") && e && state.sel >= 0) {
    const det = e.dets[state.sel];
    det.bbox = det.bbox.map(Math.round);
    requestDraw();
  } else if (d.kind === "pen-marquee" && e && state.sel >= 0) {
    const det = e.dets[state.sel];
    const dist = Math.hypot(d.sx1 - d.sx0, d.sy1 - d.sy0);
    if (dist < 4) {
      penSelected = new Set();
    } else if (det._pen) {
      const [ix0, iy0] = toImage(Math.min(d.sx0, d.sx1), Math.min(d.sy0, d.sy1));
      const [ix1, iy1] = toImage(Math.max(d.sx0, d.sx1), Math.max(d.sy0, d.sy1));
      const sel = new Set();
      det._pen.forEach(([px, py], i) => { if (px >= ix0 && px <= ix1 && py >= iy0 && py <= iy1) sel.add(i); });
      penSelected = sel;
    }
    requestDraw();
  }
});

cv.addEventListener("dblclick", ev => {
  const [ix, iy] = toImage(ev.offsetX, ev.offsetY);
  const i = hitBox(ix, iy);
  if (i >= 0) { state.sel = i; zoomToBox(curEntry().dets[i]); renderBoxlist(); renderFooter(); }
});

cv.addEventListener("contextmenu", ev => ev.preventDefault());

cv.addEventListener("wheel", ev => {
  if (!curEntry()) return;
  ev.preventDefault();
  const f = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0016));
  zoomAt(ev.offsetX, ev.offsetY, f);
}, {passive: false});

/* ============================== keyboard ============================== */
document.addEventListener("keydown", ev => {
  if (window.__activeView && window.__activeView !== "annotate") return;
  if ($("help").open || $("pathResolve").open) return;
  const t = ev.target;
  const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
  if (typing) return;

  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") { ev.preventDefault(); exportJSON(); return; }
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") { ev.preventDefault(); undo(); return; }
  if (ev.altKey && ev.key.startsWith("Arrow")) {
    ev.preventDefault();
    if (state.sel >= 0) {
      const step = ev.shiftKey ? 10 : 1;
      nudgeSel(ev.key === "ArrowLeft" ? -step : ev.key === "ArrowRight" ? step : 0,
               ev.key === "ArrowUp" ? -step : ev.key === "ArrowDown" ? step : 0);
    }
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

  const k = ev.key;
  if (k === " ") { spaceDown = true; ev.preventDefault(); return; }
  if (k >= "1" && k <= "9") {
    const l = state.labels[+k - 1];
    if (l) assign(l.name); else toast("no character on key " + k + " — add one on the right", "warn");
    return;
  }
  if (k === "0") { assign(IGNORE); return; }

  switch (k) {
    case "Tab": {
      ev.preventDefault();
      const e = curEntry();
      if (!e || !e.dets.length) break;
      const dir = ev.shiftKey ? -1 : 1;
      state.sel = ((state.sel < 0 ? (dir > 0 ? -1 : 0) : state.sel) + dir + e.dets.length) % e.dets.length;
      ensureVisible(e.dets[state.sel]);
      if (state.mode === "pen" && !state.penDrawingNew && !ensurePenOutline(e.dets[state.sel], e.width, e.height)) {
        setMode("select");
      }
      renderBoxlist(); renderFooter(); requestDraw();
      break;
    }
    case "Enter":
      if (state.mode === "pen" && state.penDrawingNew) {
        const e = curEntry(); const d = e && state.sel >= 0 ? e.dets[state.sel] : null;
        if (d && d._newPath && d._newPath.length >= 3) openPathResolve(d);
        else toast("need at least 3 points to close the path", "warn");
      } else jumpNextUnlabeled();
      break;
    case "x": case "X": case "Delete": case "Backspace":
      ev.preventDefault();
      if (state.mode === "pen" && !state.penDrawingNew && (penHover >= 0 || penSelected.size > 0)) deletePenVertex();
      else deleteSel();
      break;
    case "u": case "U": undo(); break;
    case "b": case "B": setMode("draw"); break;
    case "v": case "V": setMode("select"); break;
    case "m": case "M": setMode("mask"); break;
    case "o": case "O": setMode("pen"); break;
    case "i": case "I": toggleNewPath(); break;
    case "[": state.brushRadius = clamp(state.brushRadius / 1.25, 4, 400); updateBrushLabel(); requestDraw(); break;
    case "]": state.brushRadius = clamp(state.brushRadius * 1.25, 4, 400); updateBrushLabel(); requestDraw(); break;
    case "Escape":
      if (state.mode === "pen" && state.penDrawingNew) {
        const e = curEntry(); if (e && state.sel >= 0) delete e.dets[state.sel]._newPath;
        state.penDrawingNew = false;
        $("mNewPath").classList.remove("on");
        setMode("pen");
      }
      else if (state.mode !== "select") setMode("select");
      else if (state.sel >= 0) { state.sel = -1; renderBoxlist(); renderFooter(); requestDraw(); }
      break;
    case "f": case "F": fitView(); requestDraw(); break;
    case "z": case "Z": if (state.sel >= 0) zoomToBox(curEntry().dets[state.sel]); break;
    case "n": case "N": case "PageDown": ev.preventDefault(); stepImage(1); break;
    case "p": case "P": case "PageUp": ev.preventDefault(); stepImage(-1); break;
    case "+": case "=": { const [w, h] = stageSize(); zoomAt(w / 2, h / 2, 1.25); break; }
    case "-": case "_": { const [w, h] = stageSize(); zoomAt(w / 2, h / 2, 0.8); break; }
    case "?": $("help").showModal(); break;
    case "ArrowLeft": case "ArrowUp": ev.preventDefault(); stepImage(-1); break;
    case "ArrowRight": case "ArrowDown": ev.preventDefault(); stepImage(1); break;
  }
});
document.addEventListener("keyup", ev => { if (ev.key === " ") spaceDown = false; });

function setMode(m) {
  if (m === "mask" && (!curEntry() || state.sel < 0)) { toast("select a box first", "warn"); m = "select"; }
  if (m === "pen") {
    if (!curEntry() || state.sel < 0) { toast("select a box first", "warn"); m = "select"; }
    else if (!ensurePenOutline(curEntry().dets[state.sel], curEntry().width, curEntry().height)) {
      toast("paint a mask first (M), then fine-tune it with the pen tool", "warn");
      m = "select";
    }
  }
  if (m !== "pen") {
    const e = curEntry();
    if (e) for (const dd of e.dets) {
      if (dd._pen) dd._outline = dd._pen;
      delete dd._pen; delete dd._newPath;
    }
    penSelected = new Set();
    state.penDrawingNew = false;
    $("mNewPath").classList.remove("on");
  }
  state.mode = m;
  $("mSelect").classList.toggle("on", m === "select");
  $("mDraw").classList.toggle("on", m === "draw");
  $("mMask").classList.toggle("on", m === "mask");
  $("mPen").classList.toggle("on", m === "pen");
  $("fMode").textContent = m.toUpperCase();
  cv.style.cursor = m === "draw" ? "crosshair" : (m === "mask" || m === "pen") ? "none" : "default";
  requestDraw();
}
function enterNewPath() {
  if (!curEntry() || state.sel < 0) { toast("select a box first", "warn"); return; }
  const e = curEntry(), d = e.dets[state.sel];
  if (d._pen) d._outline = d._pen;
  delete d._pen;
  d._newPath = [];
  state.mode = "pen";
  state.penDrawingNew = true;
  $("mSelect").classList.remove("on"); $("mDraw").classList.remove("on"); $("mMask").classList.remove("on");
  $("mPen").classList.add("on"); $("mNewPath").classList.add("on");
  $("fMode").textContent = "PEN·NEW";
  cv.style.cursor = "none";
  requestDraw();
}
function toggleNewPath() {
  if (state.mode === "pen" && state.penDrawingNew) {
    const e = curEntry();
    if (e && state.sel >= 0) delete e.dets[state.sel]._newPath;
    state.penDrawingNew = false;
    $("mNewPath").classList.remove("on");
    setMode("pen");
  } else {
    enterNewPath();
  }
}
function openPathResolve(d) {
  pathResolveDet = d;
  $("prName").textContent = "#" + d.id;
  $("pathResolve").showModal();
}
function resolvePath(action) {
  const d = pathResolveDet;
  pathResolveDet = null;
  $("pathResolve").close();
  if (!d) return;
  const e = curEntry();
  const path = d._newPath;
  delete d._newPath;
  state.penDrawingNew = false;
  $("mNewPath").classList.remove("on");
  if (action === "cancel" || !path || path.length < 3) { setMode("pen"); requestDraw(); return; }

  const shapeBmp = polygonToBitmap(path, e.width, e.height);
  ensureMaskEditable(d, e.width, e.height);
  const snapshot = d._bmp.slice();
  if (action === "replace") d._bmp.set(shapeBmp);
  else if (action === "merge") { for (let i = 0; i < d._bmp.length; i++) if (shapeBmp[i]) d._bmp[i] = 1; }
  else if (action === "subtract") { for (let i = 0; i < d._bmp.length; i++) if (shapeBmp[i]) d._bmp[i] = 0; }

  undoStack.push({kind: "mask", imgKey: e.key, det: d, snapshot});
  d._bmpDirty = true;
  repaintMaskCanvas(d, e.width, e.height);
  updateOutline(d, e.width, e.height);
  state.dirty = true;
  toast(action + "d new path into the mask — Ctrl+Z to undo", "ok");
  setMode("pen");
  requestDraw();
}
$("pathResolve").addEventListener("close", () => { if (pathResolveDet) resolvePath("cancel"); });
function setViewMode(m) {
  state.viewMode = m;
  $("vBoth").classList.toggle("on", m === "both");
  $("vBox").classList.toggle("on", m === "box");
  $("vMask").classList.toggle("on", m === "mask");
  requestDraw();
}
function toggleFocusMode() {
  state.focusMode = !state.focusMode;
  const btn = $("toggleFocus");
  btn.classList.toggle("on", state.focusMode);
  btn.title = state.focusMode
    ? "Focus mode: ON — selecting a box hides the rest. Click to show every character all the time instead."
    : "Focus mode: OFF — every character always shows. Click to go back to hiding the rest while one is selected.";
  renderAll();
  requestDraw();
}

/* ============================== rendering (dom) ============================== */
function renderAll() { renderHeader(); renderRoster(); renderBoxlist(); renderFooter(); updateRow(state.cur); }

function renderHeader() {
  const e = curEntry();
  $("fname").textContent = e ? e.relPath : "no image";
  $("fname").title = e ? e.relPath : "";
  $("fidx").textContent = (state.cur + 1) + "/" + state.images.length;
  let tot = 0, done = 0, imgsDone = 0;
  for (const im of state.images) {
    const s = imgStats(im);
    tot += s.total; done += s.done;
    if (s.total > 0 && s.todo === 0) imgsDone++;
  }
  $("pnum").textContent = done; $("ptot").textContent = tot;
  $("pimg").textContent = imgsDone; $("pimgtot").textContent = state.images.length;
  $("pbar").style.width = (tot ? (100 * done / tot) : 0) + "%";
}

/* ============================== label filter (AND / OR / NOT, up to 4 terms) ============================== */
// grammar: ['NOT'] label (('AND'|'OR') ['NOT'] label){0,3} — evaluated strictly left-to-right,
// no precedence/parens; a bare "label label" without a connector implies AND
function parseFilterQuery(q) {
  const tokens = String(q || "").trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) return null;
  const terms = [];
  let i = 0;
  while (i < tokens.length && terms.length < 4) {
    let op = null;
    if (terms.length > 0) {
      const opTok = tokens[i] ? tokens[i].toUpperCase() : "";
      if (opTok === "AND" || opTok === "OR") { op = opTok; i++; }
      else op = "AND";
    }
    let neg = false;
    if (tokens[i] && tokens[i].toUpperCase() === "NOT") { neg = true; i++; }
    if (i >= tokens.length) break;
    const label = tokens[i]; i++;
    terms.push({op, neg, label});
  }
  return terms.length ? terms : null;
}
function imageHasLabel(entry, term) {
  const t = String(term).trim().toLowerCase();
  if (!t) return true;
  return entry.dets.some(d => {
    if (t === "unlabeled") return !detDone(d);
    if (t === "ignore") return d.label === IGNORE;
    return detDone(d) && d.label !== IGNORE && d.label.toLowerCase() === t;
  });
}
function evalFilterQuery(terms, entry) {
  if (!terms) return true;
  let result = null;
  for (const {op, neg, label} of terms) {
    let match = imageHasLabel(entry, label);
    if (neg) match = !match;
    result = result === null ? match : (op === "OR" ? (result || match) : (result && match));
  }
  return result;
}
function applyFilmFilter() {
  const input = $("filmFilter");
  const terms = parseFilterQuery(input.value);
  input.classList.toggle("bad", !!input.value.trim() && !terms);
  let shown = 0;
  state.images.forEach((e, i) => {
    const match = !terms || evalFilterQuery(terms, e);
    if (rowEls[i]) rowEls[i].classList.toggle("filtered-out", !match);
    if (match) shown++;
  });
  $("filmcnt").textContent = terms ? (shown + "/" + state.images.length) : state.images.length;
}

function renderFilmstrip() {
  const list = $("filmlist");
  list.textContent = "";
  rowEls = [];
  state.images.forEach((e, i) => {
    const b = document.createElement("button");
    b.className = "frow";
    b.innerHTML = '<span class="num"></span><span class="name"></span><span class="st"></span>';
    b.querySelector(".num").textContent = String(i + 1).padStart(3, "0");
    b.querySelector(".name").textContent = e.fileName;
    b.title = e.relPath;
    b.addEventListener("click", () => selectImage(i));
    list.appendChild(b);
    rowEls.push(b);
    updateRow(i);
  });
  applyFilmFilter();
}
function updateRow(i) {
  const b = rowEls[i], e = state.images[i];
  if (!b || !e) return;
  b.classList.toggle("cur", i === state.cur);
  const st = b.querySelector(".st");
  const s = imgStats(e);
  if (!e.file) { st.textContent = "no file"; st.className = "st nofile"; }
  else if (s.total === 0) { st.textContent = "0 box"; st.className = "st nofile"; }
  else if (s.todo === 0) { st.textContent = "✓ " + s.total; st.className = "st done"; }
  else if (s.done === 0) { st.textContent = s.todo + " left"; st.className = "st todo"; }
  else { st.textContent = s.todo + " left"; st.className = "st part"; }
  if (i === state.cur) b.scrollIntoView({block: "nearest"});
}

function renderRoster() {
  const list = $("rosterlist");
  list.textContent = "";
  const counts = new Map();
  let ignored = 0;
  for (const e of state.images) for (const d of e.dets) {
    if (d.label === IGNORE) ignored++;
    else if (detDone(d)) counts.set(d.label, (counts.get(d.label) || 0) + 1);
  }
  state.labels.forEach((l, i) => {
    const b = document.createElement("button");
    b.className = "crow";
    b.innerHTML = '<span class="key"></span><span class="sw"></span><span class="nm"></span><span class="ct"></span><span class="ren" title="Rename everywhere">✎</span><span class="del" title="Remove character">✕</span>';
    b.querySelector(".key").textContent = i < 9 ? String(i + 1) : "·";
    b.querySelector(".sw").style.background = l.color;
    b.querySelector(".nm").textContent = l.name;
    b.querySelector(".ct").textContent = counts.get(l.name) || 0;
    b.title = "Assign " + l.name + (i < 9 ? " (key " + (i + 1) + ")" : "");
    b.addEventListener("click", () => assign(l.name));
    b.querySelector(".ren").addEventListener("click", ev => { ev.stopPropagation(); renameLabel(l); });
    b.querySelector(".del").addEventListener("click", ev => { ev.stopPropagation(); removeLabel(l); });
    list.appendChild(b);
  });
  const ig = document.createElement("button");
  ig.className = "crow ignore";
  ig.innerHTML = '<span class="key">0</span><span class="sw"></span><span class="nm">ignore / not a character</span><span class="ct"></span>';
  ig.querySelector(".ct").textContent = ignored;
  ig.title = "Background mob, mascot, false positive you still want to keep boxed (key 0)";
  ig.addEventListener("click", () => assign(IGNORE));
  list.appendChild(ig);
  $("rostercnt").textContent = state.labels.length;
}

function renderBoxlist() {
  const list = $("boxlist");
  list.textContent = "";
  const e = curEntry();
  if (!e) { $("boxcnt").textContent = 0; return; }
  e.dets.forEach((d, i) => {
    const b = document.createElement("button");
    b.className = "brow" + (i === state.sel ? " sel" : "");
    b.innerHTML = '<span class="bid"></span><span class="chip"></span><span class="sc"></span>';
    b.querySelector(".bid").textContent = "#" + d.id + (hasMask(d) ? " ◨" : "");
    b.querySelector(".bid").title = hasMask(d) ? "has a segmentation mask" : "";
    const chip = b.querySelector(".chip");
    if (detDone(d)) {
      chip.textContent = d.label === IGNORE ? "ignore" : d.label;
      chip.style.background = d.label === IGNORE ? "#7d8697" : labelColor(d.label);
    } else {
      chip.textContent = "unlabeled";
      chip.className = "chip none";
    }
    b.querySelector(".sc").textContent = d.score != null ? Number(d.score).toFixed(2) : "manual";
    b.addEventListener("click", () => {
      state.sel = i; ensureVisible(d); renderBoxlist(); renderFooter(); requestDraw();
    });
    b.addEventListener("dblclick", () => zoomToBox(d));
    b.addEventListener("mouseenter", () => { state.hover = i; requestDraw(); });
    b.addEventListener("mouseleave", () => { state.hover = -1; requestDraw(); });
    list.appendChild(b);
  });
  $("boxcnt").textContent = e.dets.length;
}

function renderFooter() {
  const e = curEntry();
  const d = e && state.sel >= 0 ? e.dets[state.sel] : null;
  $("fSel").textContent = d
    ? "box #" + d.id + " · " + Math.round(d.bbox[2] - d.bbox[0]) + "×" + Math.round(d.bbox[3] - d.bbox[1]) +
      (detDone(d) ? " · " + (d.label === IGNORE ? "ignore" : d.label) : " · unlabeled") +
      (hasMask(d) ? " · has mask" : "")
    : "no box selected";
}

function checkEmpty() {
  $("empty").classList.toggle("hide", state.images.length > 0);
}

/* ============================== wiring ============================== */
// Electron app shell: prefer the native folder dialog (also captures the
// absolute path so EXPORT can save straight back into it, no manual "move
// the download back into the folder" step). Falls back to the plain
// <input webkitdirectory> picker when running standalone in a browser, or
// for recursive/nested folders which this native path doesn't walk.
let annotateSaveFolder = null;
function joinPath(dir, name) { return dir.replace(/[\\/]+$/, "") + "\\" + name; }

async function loadFolderNative(folder) {
  if (!folder) {
    folder = await window.sam3.pickFolder();
    if (!folder) return;
  }
  const { images, jsonFile } = await window.sam3.readFolder(folder);
  if (!images.length && !jsonFile) { toast("no images or JSON found in that folder", "warn"); return; }
  const files = [];
  for (const name of images) {
    const dataUrl = await window.sam3.readFileDataUrl(joinPath(folder, name));
    const blob = await (await fetch(dataUrl)).blob();
    const f = new File([blob], name, { type: blob.type });
    f._relPath = name;
    files.push(f);
  }
  if (jsonFile) {
    const text = await window.sam3.readTextFile(joinPath(folder, jsonFile));
    if (text != null) files.push(new File([new Blob([text], { type: "application/json" })], jsonFile));
  }
  annotateSaveFolder = folder;
  routeFiles(files);
}
// Exposed so the Detect tab can hand off "open this folder in Annotate"
// after a detection job finishes, without the user re-picking it.
if (window.sam3) window.__annotateOpenFolder = loadFolderNative;

$("btnJson").addEventListener("click", () => $("inJson").click());
$("eJson").addEventListener("click", () => $("inJson").click());
$("btnImgs").addEventListener("click", () => $("inImgs").click());
$("eImgs").addEventListener("click", () => $("inImgs").click());
if (window.sam3) {
  $("btnDir").addEventListener("click", () => loadFolderNative());
  $("eDir").addEventListener("click", () => loadFolderNative());
} else {
  $("btnDir").addEventListener("click", () => $("inDir").click());
  $("eDir").addEventListener("click", () => $("inDir").click());
}
$("btnExport").addEventListener("click", exportJSON);
$("btnHelp").addEventListener("click", () => $("help").showModal());
$("helpX").addEventListener("click", () => $("help").close());
$("prevImg").addEventListener("click", () => stepImage(-1));
$("nextImg").addEventListener("click", () => stepImage(1));
$("mSelect").addEventListener("click", () => setMode("select"));
$("mDraw").addEventListener("click", () => setMode("draw"));
$("mMask").addEventListener("click", () => setMode("mask"));
$("mPen").addEventListener("click", () => setMode("pen"));
$("mNewPath").addEventListener("click", toggleNewPath);
$("mFillHoles").addEventListener("click", fillHoles);
$("prReplace").addEventListener("click", () => resolvePath("replace"));
$("prMerge").addEventListener("click", () => resolvePath("merge"));
$("prSubtract").addEventListener("click", () => resolvePath("subtract"));
$("prCancel").addEventListener("click", () => resolvePath("cancel"));
$("vBoth").addEventListener("click", () => setViewMode("both"));
$("vBox").addEventListener("click", () => setViewMode("box"));
$("vMask").addEventListener("click", () => setViewMode("mask"));
$("toggleFocus").addEventListener("click", toggleFocusMode);
$("zIn").addEventListener("click", () => { const [w, h] = stageSize(); zoomAt(w / 2, h / 2, 1.25); });
$("zOut").addEventListener("click", () => { const [w, h] = stageSize(); zoomAt(w / 2, h / 2, 0.8); });
$("zFit").addEventListener("click", () => { fitView(); requestDraw(); });

function updateBrushLabel() { $("brLabel").textContent = Math.round(state.brushRadius) + "px"; }
$("brDown").addEventListener("click", () => { state.brushRadius = clamp(state.brushRadius / 1.25, 4, 400); updateBrushLabel(); requestDraw(); });
$("brUp").addEventListener("click", () => { state.brushRadius = clamp(state.brushRadius * 1.25, 4, 400); updateBrushLabel(); requestDraw(); });
$("mClearMask").addEventListener("click", () => {
  const e = curEntry();
  if (!e || state.sel < 0) { toast("select a box first", "warn"); return; }
  const d = e.dets[state.sel];
  if (!d._bmp && !d.seg) { toast("box has no mask", "warn"); return; }
  ensureMaskEditable(d, e.width, e.height);
  undoStack.push({kind: "mask", imgKey: e.key, det: d, snapshot: d._bmp.slice()});
  d._bmp.fill(0);
  repaintMaskCanvas(d, e.width, e.height);
  d._bmpDirty = true;
  state.dirty = true;
  d._outline = null;
  if (state.mode === "pen") d._pen = undefined;
  toast("mask cleared — U to undo");
  requestDraw();
});

async function routeFiles(files) {
  const jsons = files.filter(f => f.name.toLowerCase().endsWith(".json"));
  const imgs = files.filter(f => f.type.startsWith("image/"));
  if (jsons.length) {
    // newest first — a moved-back annotations_labeled.json beats the raw step-1 file
    jsons.sort((a, b) => (b.lastModified || 0) - (a.lastModified || 0));
    let loaded = false;
    for (const j of jsons) {
      if (loadJsonText(await j.text(), j.name)) {
        loaded = true;
        if (jsons.length > 1) toast("using " + j.name + " — " + (jsons.length - 1) + " other JSON file(s) ignored");
        break;
      }
    }
    if (!loaded && !imgs.length) return;
  }
  if (imgs.length) addImageFiles(imgs);
  else if (!jsons.length) toast("no JSON or image files found", "warn");
}
$("inJson").addEventListener("change", ev => { routeFiles([...ev.target.files]); ev.target.value = ""; });
$("inImgs").addEventListener("change", ev => { routeFiles([...ev.target.files]); ev.target.value = ""; });
$("inDir").addEventListener("change", ev => { routeFiles([...ev.target.files]); ev.target.value = ""; });

function addFromInput() {
  const name = $("addName").value;
  const l = addLabel(name);
  if (l) {
    $("addName").value = "";
    if (state.sel >= 0 && curEntry() && !detDone(curEntry().dets[state.sel])) assign(l.name);
  }
}
$("addBtn").addEventListener("click", addFromInput);
$("addName").addEventListener("keydown", ev => {
  if (ev.key === "Enter") { ev.preventDefault(); addFromInput(); }
  if (ev.key === "Escape") { $("addName").value = ""; $("addName").blur(); }
});

$("filmFilter").addEventListener("input", applyFilmFilter);
$("filmFilter").addEventListener("keydown", ev => {
  if (ev.key === "Escape") { $("filmFilter").value = ""; applyFilmFilter(); $("filmFilter").blur(); }
});

/* drag & drop (files and folders) -- scoped to this view so dragging over
   the Detect/Export tabs doesn't get routed here */
let dragDepth = 0;
viewRoot.addEventListener("dragenter", ev => { ev.preventDefault(); dragDepth++; viewRoot.classList.add("dragging"); });
viewRoot.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; viewRoot.classList.remove("dragging"); } });
viewRoot.addEventListener("dragover", ev => ev.preventDefault());
viewRoot.addEventListener("drop", async ev => {
  ev.preventDefault();
  dragDepth = 0;
  viewRoot.classList.remove("dragging");
  const items = [...(ev.dataTransfer.items || [])];
  const files = [];
  const walk = entry => new Promise(res => {
    if (entry.isFile) {
      entry.file(f => { f._relPath = entry.fullPath.replace(/^\//, ""); files.push(f); res(); }, () => res());
    } else if (entry.isDirectory) {
      const rd = entry.createReader();
      const readAll = () => rd.readEntries(async ents => {
        if (!ents.length) return res();
        await Promise.all(ents.map(walk));
        readAll();
      }, () => res());
      readAll();
    } else res();
  });
  const entries = items.map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (entries.length) await Promise.all(entries.map(walk));
  else files.push(...ev.dataTransfer.files);

  if (!files.length) { toast("drop a .json manifest or image files", "warn"); return; }
  routeFiles(files);
});

document.addEventListener("click", ev => {
  const b = ev.target.closest("button");
  if (b) b.blur();   // keep Enter/Space on the bench shortcuts, not on the last-clicked button
});

window.addEventListener("beforeunload", ev => {
  if (state.dirty) { ev.preventDefault(); ev.returnValue = ""; }
});

/* boot */
setMode("select");
renderAll();
checkEmpty();

})();
