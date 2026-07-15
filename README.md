# Data Curation Pipeline (SAM 3 edition)

Turn raw anime posters/illustrations into a **named-character training dataset** in three steps —
detection by Meta's **SAM 3**, everything stored as **standard COCO**:

```
                 ┌────────────────────────────── one folder ──────────────────────────────┐
raw_images/ ──▶  Detect  ──▶  raw_images/annotations.json  ──▶  Annotate  ──▶  annotations_labeled.json
                 (SAM 3, text        (COCO, category                (load the folder;         (COCO, characters
                  prompt "person")    "unlabeled")                   name every box)           as categories)
                                                                                                    │
                                                                                                 Export
                                                                                                    │
                                                                                     ├─ dataset/yolo/   detector training
                                                                                     └─ dataset/crops/  classifier training
```

You give step 1 **one thing — the folder** — and the COCO `annotations.json` spawns inside it.
Loading that same folder into Annotate brings boxes and images in together; no separate JSON step.

## The app (recommended)

Detect, Annotate, and Export live together in one desktop app — no Python environment, no
terminal, no separate HTML file to open.

1. **Install**: run `Sam3 Annotation Setup.exe` (built via `installer/build-installer.ps1`,
   see [Building the installer](#building-the-installer) below) — one click, no wizard, no admin
   prompt. It launches itself when done.
2. **First run**: the **Detect** tab checks your Hugging Face authorization for the gated
   `facebook/sam3` weights. If you haven't requested access yet, do so once at
   huggingface.co/facebook/sam3, then paste a token into the app (or run `hf auth login` in a
   terminal) — no need to leave the app for this.
3. **Detect**: pick a folder, set the concept prompt (`person` by default), and hit **RUN
   DETECTION**. Batch size, device (CUDA/CPU), precision (fp16/bf16/fp32), and an optional VRAM
   cap are all in-app controls — a live VRAM gauge tracks usage while a job runs, and if a batch
   doesn't fit, the batch size is halved and retried automatically rather than crashing the job.
   Multiple images are detected in a single batched forward pass, reusing one cached text-prompt
   embedding across the whole run, rather than re-encoding the prompt per image.
4. **Annotate**: click **OPEN IN ANNOTATE →** when detection finishes (or switch tabs and load a
   folder yourself) — this is the same CAST/LIST labeling bench described below, running inside
   the app. Loading a folder here also remembers where to save: **EXPORT** writes
   `annotations_labeled.json` straight back into the folder, no manual "move the download" step.
5. **Export**: point at the labeled JSON and an output folder, pick YOLO / crops / both, and run.

The standalone tools below (`1_detect.py`, `2_annotate.html`, `3_export.py`) still work exactly as
documented — they're the scriptable/headless path for automation or servers without a GUI. The
app is a thin, faster shell around the same COCO pipeline, not a replacement for them.

### Building the installer

```powershell
installer\build-installer.ps1
```

Requires Python (with `pip install -r backend/requirements.txt`), PyInstaller, and Node.js/npm.
Freezes the backend with PyInstaller (`--onedir`, so the installed app starts fast — no
per-launch re-extraction) and packages it with the Electron shell via electron-builder into
`app/dist/Sam3 Annotation Setup.exe`. The installer bundles CUDA-enabled torch so it's a few GB
and genuinely one-click — nothing further to install afterward except the gated SAM 3 weights
themselves (~3.4 GB, downloaded on first Detect run, see step 2 above).

## Advanced: command-line pipeline

The three scripts below are what the app runs under the hood, exposed directly for scripting,
headless/server use, or if you'd rather not install the app at all.

### Install

```bash
pip install -r requirements.txt        # torch + transformers (v5+, has SAM 3) + pillow
```

**SAM 3 is a gated model.** One-time setup:

1. Request access at <https://huggingface.co/facebook/sam3> (instant for most accounts)
2. `hf auth login` with a token from that account

First run downloads ~3.4 GB of weights. A GPU is strongly recommended; CPU works but takes seconds per image.

### Step 1 — detect characters with SAM 3

```bash
python 1_detect.py ./raw_images
# -> ./raw_images/annotations.json  (COCO; every box in category "unlabeled")
```

SAM 3 does *promptable concept segmentation*: you give it a noun phrase and it finds every instance —
and because it's a segmentation model, not just a box detector, every proposal also comes with a
per-instance mask. Masks are stored as COCO RLE alongside each box **by default**, so 2_annotate.html
can show the actual silhouette and let you touch it up, not just a rectangle.

| flag | what it does |
|---|---|
| `--prompt "anime character"` | change the concept (default `"person"`; try `"girl"`, `"chibi character"`, …) |
| `--threshold 0.4` | lower it if **small background characters** are missed (default 0.5) |
| `--preview 8` | draw boxes on the first 8 images into `<folder>/previews` — check before labeling hundreds |
| `--recursive` | scan subfolders (`previews/` is always skipped) |
| `--no-masks` | skip segmentation masks — boxes only, no pycocotools needed, smaller/faster JSON |
| `--device cuda\|cpu` | force a device (default: auto) |
| `--no-resume` | redo everything; by default re-runs skip images already in the JSON |

Masks need `pip install pycocotools`; if it's missing, `1_detect.py` prints a one-time warning and
falls back to boxes-only automatically — nothing breaks.

For "stacked together / too small" posters: run once with `--preview`, look at the boxes, then tune
`--threshold` and `--prompt`. False positives are cheap (one keypress to delete in step 2); the app's
draw tool covers anything SAM 3 missed. The script checkpoints every 20 images, so interrupting is safe.

### Step 2 — name every box (2_annotate.html)

Open **`2_annotate.html`** in any modern browser (double-click the file — no server, fully offline).

1. **FOLDER** → pick `raw_images/` — the `annotations.json` inside is detected and loaded **automatically**, together with the images (drag & dropping the folder onto the window does the same)
2. Add your characters once in the CAST panel (`naruto`, `sakura`, …) — each gets a color and a number key
3. A box is auto-selected → press its character's number key → the next unlabeled box selects itself. Repeat.
4. **EXPORT** (or `Ctrl+S`) → downloads `annotations_labeled.json` (COCO)

The export is also your **save file**: load it back (or drop it into the folder — the app always picks the *newest* JSON it finds) to resume. Export often; progress lives only in the page until you do. Old pre-COCO manifests from the earlier pipeline load fine and are migrated to COCO on the next export.

Selecting a box or mask **focuses** it — everything else on the canvas hides so you can work without
clutter, and the selected one renders brighter/glowing (hover a different box to peek at it without
losing your selection). The toolbar lives across the top of the canvas, grouped into TOOL / BRUSH /
VIEW / ZOOM. Above the filmstrip, a **filter box** narrows which images are listed by character
label — up to 4 terms combined with AND / OR / NOT (e.g. `naruto AND sakura`, `NOT ignore`); it also
understands the special labels `ignore` and `unlabeled`.

Key shortcuts (press `?` in the app for the full list):

| key | action |
|---|---|
| `1`–`9` | assign character to selected box |
| `0` | mark as *ignore* (background mob / mascot / false positive worth keeping boxed) |
| `Enter` | jump to next unlabeled box (crosses into the next image when done) |
| `Tab` | cycle boxes in the image |
| `X` | delete box (false positive) — or, in `PEN` mode, remove the hovered outline point |
| `Ctrl`+`Z` / `U` | undo the last action — box move/resize, delete, mask edit, or pen edit |
| `B` | draw a box SAM 3 missed (drag on the image) |
| `M` | paint/erase the selected box's mask — drag to paint, right-drag or `Alt`+drag to erase |
| `[` / `]` | shrink / grow the mask brush |
| `O` | pen tool — edit the mask outline point-by-point (needs an existing mask) |
| `I` | pen tool, new path — draw an independent new shape from scratch |
| `Alt`+right-click | (in `PEN` mode) delete that outline point directly |
| wheel / drag | zoom at cursor / pan — zoom way in for tiny characters |
| `←` `→` `↑` `↓` | next / previous image; hold `Alt` to nudge the selected box instead (`Shift` = ×10) |
| `N` / `P` | next / previous image (same as `←`/`→`) |

Boxes are **red & dashed** until labeled; the filmstrip shows how many each image still needs.

**Segmentation masks** load in automatically wherever SAM 3 produced one (translucent fill in the
box's color). The view HUD's `BOTH` / `BOXES` / `MASKS` buttons switch what's drawn on the canvas —
labeling still works in any view. A box without a mask — including ones you draw by hand with `B` —
gets a blank one the instant you start painting on it with `M`. `CLEAR` wipes the selected box's
mask; `FILL HOLES` fills any gaps fully enclosed inside it (SAM 3 sometimes leaves small unmasked
speckles inside an otherwise solid silhouette). The selected box's silhouette is always outlined
with an animated marching-ants dash, in any mode, so it reads clearly against the fill.

For finer control than the brush, the `O` **pen tool** traces the mask's outline into draggable
points — drag a point to reshape the silhouette, click an edge to add a point, `X`/`Delete` removes
the hovered point, drag a rectangle over empty space to marquee-select several points and move or
delete them together, `Alt`+right-click deletes one point instantly. `I` (**NEW PATH**) switches to
placing brand-new points from scratch, independent of whatever's already there — click to place
each point, click the start point (or press `Enter`) to close the loop, then choose **Replace**
(swap in this new shape), **Merge** (union it into the mask), **Subtract** (cut it out), or
**Cancel**. `Esc` abandons an in-progress new path. Every edit (brush or pen) is re-encoded to COCO
RLE on `EXPORT`, and is undoable with `Ctrl`+`Z`.

### Step 3 — build the datasets

```bash
python 3_export.py raw_images/annotations_labeled.json -o ./dataset
```

`--images-dir` defaults to the folder the JSON sits in, so if you kept everything in one folder that's the whole command. Produces (pick with `--format yolo|crops|both`):

**`dataset/yolo/`** — Ultralytics detection dataset, *character names as classes*:

```bash
pip install ultralytics
yolo detect train data=dataset/yolo/data.yaml model=yolov8m.pt imgsz=1024 epochs=100
```

**`dataset/crops/<character>/*.png`** — every labeled box cropped out (`--pad 0.05` padding), one folder per character — a ready-made ImageFolder for classifier training (timm, torchvision, …).

Category semantics during export: `ignore` boxes are excluded from classes (an image where *every* box is ignore stays in the YOLO set as a clean negative); images that still contain `unlabeled` boxes are **left out of the YOLO set** (they'd teach the detector those characters are background) but their labeled boxes still go to crops. Other flags: `--val-ratio 0.1`, `--min-crop 32`, `--link`, `--seed`.

### Using it at inference time

**A. Single stage** — your trained YOLO model does detection + identity in one pass. Simple and fast; wants ~50+ boxes per character.

**B. Two stage** (usually more data-efficient for many characters) — keep SAM 3 for *where*, your crop classifier for *who*:

```python
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

img = Image.open("new_poster.jpg").convert("RGB")
inputs = processor(images=img, text="person", return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
res = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist())[0]

for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
    x0, y0, x1, y1 = map(int, box)
    name = my_classifier(img.crop((x0, y0, x1, y1)))   # trained on dataset/crops/
    print(name, (x0, y0, x1, y1), round(score, 2))
```

`crops/` exists for option B, `yolo/` for option A — you get both, so benchmark and keep the winner.

### The COCO file

Standard COCO object-detection layout — readable by pycocotools, FiftyOne, CVAT, etc.:

```jsonc
{
  "info":     { "description": "...", "sam3": {"prompt": "person", "threshold": 0.5, ...} },
  "licenses": [],
  "categories": [
    {"id": 1, "name": "naruto",    "supercategory": "character", "color": "#ffb454"},
    {"id": 2, "name": "sakura",    "supercategory": "character", "color": "#f77fbe"},
    {"id": 3, "name": "ignore",    "supercategory": "meta"},      // only present if used
    {"id": 4, "name": "unlabeled", "supercategory": "meta"}       // only while boxes remain unnamed
  ],
  "images": [
    {"id": 1, "file_name": "sub/poster_001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [x, y, width, height],           // COCO convention: top-left + size, pixels
     "area": 96800, "iscrowd": 0,
     "score": 0.93,                            // SAM 3 confidence (absent on hand-drawn boxes)
     "source": "detector",                     // or "manual"
     "segmentation": {...}}                    // RLE mask, present unless --no-masks / pycocotools missing
  ]
}
```

The two meta categories carry the workflow state: fresh step-1 output is 100% `unlabeled`; a finished, fully-labeled export contains no `unlabeled` category at all. `segmentation` round-trips through the app — viewable and editable with the mask brush (`M`) — and is re-encoded to COCO RLE on export; boxes with no mask simply omit the key.

### Gotchas

- **Gated weights**: `1_detect.py` prints exactly what to do if HF access/login is missing.
- **`file_name` includes subfolder paths** (`sub/a.jpg`) and the app's folder loader matches on them, so duplicate basenames across subfolders are fine *when loading folders*; only the loose-files button falls back to name-only matching.
- If images are **resized after detection**, the app rescales boxes automatically when a loaded file's size differs from the manifest (and tells you).
- Character name spelling is the class identity — use ✎ rename in the CAST panel to fix typos everywhere, rather than adding a second spelling.
