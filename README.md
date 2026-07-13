# Anime Character Curation Pipeline (SAM 3 edition)

Turn raw anime posters/illustrations into a **named-character training dataset** in three steps —
detection by Meta's **SAM 3**, everything stored as **standard COCO**:

```
                 ┌────────────────────────────── one folder ──────────────────────────────┐
raw_images/ ──▶  1_detect.py  ──▶  raw_images/annotations.json  ──▶  2_annotate.html  ──▶  annotations_labeled.json
                 (SAM 3, text        (COCO, category                (load the folder;         (COCO, characters
                  prompt "person")    "unlabeled")                   name every box)           as categories)
                                                                                                    │
                                                                                              3_export.py
                                                                                                    │
                                                                                     ├─ dataset/yolo/   detector training
                                                                                     └─ dataset/crops/  classifier training
```

You give step 1 **one thing — the folder** — and the COCO `annotations.json` spawns inside it.
Loading that same folder into the app brings boxes and images in together; no separate JSON step.

## Install

```bash
pip install -r requirements.txt        # torch + transformers (v5+, has SAM 3) + pillow
```

**SAM 3 is a gated model.** One-time setup:

1. Request access at <https://huggingface.co/facebook/sam3> (instant for most accounts)
2. `hf auth login` with a token from that account

First run downloads ~3.4 GB of weights. A GPU is strongly recommended; CPU works but takes seconds per image.

## Step 1 — detect characters with SAM 3

```bash
python 1_detect.py ./raw_images
# -> ./raw_images/annotations.json  (COCO; every box in category "unlabeled")
```

SAM 3 does *promptable concept segmentation*: you give it a noun phrase and it finds every instance.

| flag | what it does |
|---|---|
| `--prompt "anime character"` | change the concept (default `"person"`; try `"girl"`, `"chibi character"`, …) |
| `--threshold 0.4` | lower it if **small background characters** are missed (default 0.5) |
| `--preview 8` | draw boxes on the first 8 images into `<folder>/previews` — check before labeling hundreds |
| `--recursive` | scan subfolders (`previews/` is always skipped) |
| `--save-masks` | also store SAM 3 segmentation masks as COCO RLE (needs `pip install pycocotools`) |
| `--device cuda\|cpu` | force a device (default: auto) |
| `--no-resume` | redo everything; by default re-runs skip images already in the JSON |

For "stacked together / too small" posters: run once with `--preview`, look at the boxes, then tune
`--threshold` and `--prompt`. False positives are cheap (one keypress to delete in step 2); the app's
draw tool covers anything SAM 3 missed. The script checkpoints every 20 images, so interrupting is safe.

## Step 2 — name every box (the app)

Open **`2_annotate.html`** in any modern browser (double-click the file — no server, fully offline).

1. **FOLDER** → pick `raw_images/` — the `annotations.json` inside is detected and loaded **automatically**, together with the images (drag & dropping the folder onto the window does the same)
2. Add your characters once in the CAST panel (`naruto`, `sakura`, …) — each gets a color and a number key
3. A box is auto-selected → press its character's number key → the next unlabeled box selects itself. Repeat.
4. **EXPORT** (or `Ctrl+S`) → downloads `annotations_labeled.json` (COCO)

The export is also your **save file**: load it back (or drop it into the folder — the app always picks the *newest* JSON it finds) to resume. Export often; progress lives only in the page until you do. Old pre-COCO manifests from the earlier pipeline load fine and are migrated to COCO on the next export.

Key shortcuts (press `?` in the app for the full list):

| key | action |
|---|---|
| `1`–`9` | assign character to selected box |
| `0` | mark as *ignore* (background mob / mascot / false positive worth keeping boxed) |
| `Enter` | jump to next unlabeled box (crosses into the next image when done) |
| `Tab` | cycle boxes in the image |
| `X` / `U` | delete box / undo delete |
| `B` | draw a box SAM 3 missed (drag on the image) |
| wheel / drag | zoom at cursor / pan — zoom way in for tiny characters |
| arrows | nudge selected box (`Shift` = ×10); drag edges/corners to resize |
| `N` / `P` | next / previous image |

Boxes are **red & dashed** until labeled; the filmstrip shows how many each image still needs.

## Step 3 — build the datasets

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

## Using it at inference time

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

## The COCO file

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
     "segmentation": {...}}                    // RLE, only with --save-masks
  ]
}
```

The two meta categories carry the workflow state: fresh step-1 output is 100% `unlabeled`; a finished, fully-labeled export contains no `unlabeled` category at all. Extra per-annotation fields (like `segmentation`) survive the app round-trip untouched.

## Gotchas

- **Gated weights**: `1_detect.py` prints exactly what to do if HF access/login is missing.
- **`file_name` includes subfolder paths** (`sub/a.jpg`) and the app's folder loader matches on them, so duplicate basenames across subfolders are fine *when loading folders*; only the loose-files button falls back to name-only matching.
- If images are **resized after detection**, the app rescales boxes automatically when a loaded file's size differs from the manifest (and tells you).
- Character name spelling is the class identity — use ✎ rename in the CAST panel to fix typos everywhere, rather than adding a second spelling.
