#!/usr/bin/env python3
"""
Step 1 — SAM 3 concept detection over a folder of anime images.

Give it ONE thing: the folder. It runs Meta's SAM 3 (promptable concept
segmentation, text prompt "person" by default) on every image and writes a
COCO-format `annotations.json` INTO that same folder. All boxes start in the
"unlabeled" category — you turn them into character names with 2_annotate.html.

    python 1_detect.py ./raw_images
    python 1_detect.py ./raw_images --prompt "anime character" --threshold 0.4
    python 1_detect.py ./raw_images --recursive --preview 8 --save-masks

Requirements:
  * pip install torch transformers pillow   (transformers with SAM 3, v5+)
  * facebook/sam3 is a GATED model: request access once at
    https://huggingface.co/facebook/sam3 then run `hf auth login`.
  * First run downloads ~3.4 GB of weights. GPU strongly recommended.

Re-running resumes: images already in annotations.json are skipped
(--no-resume to redo). --preview N draws the first N results into
<folder>/previews so you can sanity-check the threshold cheaply.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
UNLABELED = "unlabeled"
SKIP_DIRS = {"previews"}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_sam3(model_id: str, device: str):
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError:
        sys.exit(
            "Missing dependencies. Install with:\n"
            "  pip install torch transformers pillow\n"
            "(SAM 3 needs a transformers release that includes Sam3Model, v5+.)"
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = Sam3Model.from_pretrained(model_id).to(device).eval()
        processor = Sam3Processor.from_pretrained(model_id)
    except Exception as e:
        sys.exit(
            f"Could not load {model_id}: {e}\n\n"
            "facebook/sam3 is a gated repository. If you haven't yet:\n"
            "  1. request access at https://huggingface.co/facebook/sam3\n"
            "  2. run `hf auth login` with a token from that account"
        )
    if device == "cpu":
        print("[warn] running SAM 3 on CPU — expect several seconds per image",
              file=sys.stderr)
    return torch, model, processor, device


def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def rle_encode(mask):
    """Binary mask -> COCO compressed RLE (needs pycocotools + numpy)."""
    import numpy as np
    from pycocotools import mask as maskutils

    m = mask.detach().cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    rle = maskutils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def detect(torch, model, processor, device, image, prompt, threshold, mask_threshold):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    boxes = [to_list(b) for b in to_list(results["boxes"])]   # xyxy, image px
    scores = [float(s) for s in to_list(results["scores"])]
    masks = results.get("masks")
    return boxes, scores, masks


# --------------------------------------------------------------------------
# files / manifest
# --------------------------------------------------------------------------
def find_images(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.glob("*")
    out = []
    for p in sorted(it):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return out


def new_manifest(args):
    return {
        "info": {
            "description": "SAM 3 person proposals for anime character labeling "
                           "(CAST/LIST pipeline, step 1)",
            "version": "2",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "anime-char-pipeline/1_detect.py",
            "sam3": {
                "model": args.model,
                "prompt": args.prompt,
                "threshold": args.threshold,
                "mask_threshold": args.mask_threshold,
            },
        },
        "licenses": [],
        "categories": [{"id": 1, "name": UNLABELED, "supercategory": "meta"}],
        "images": [],
        "annotations": [],
    }


def load_existing(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
            return data
        print(f"[warn] {path} exists but is not a COCO manifest — starting fresh "
              f"(the old file will be overwritten)", file=sys.stderr)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read existing {path}: {e} — starting fresh",
              file=sys.stderr)
    return None


def unlabeled_cat_id(data):
    for c in data["categories"]:
        if str(c.get("name", "")).strip().lower() == UNLABELED:
            return c["id"]
    cid = max((c["id"] for c in data["categories"]), default=0) + 1
    data["categories"].append({"id": cid, "name": UNLABELED, "supercategory": "meta"})
    return cid


def draw_preview(img_path: Path, anns, out_dir: Path):
    from PIL import ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        lw = max(2, round(min(im.size) * 0.004))
        for a in anns:
            x, y, w, h = a["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=(255, 90, 90), width=lw)
            d.text((x + lw + 2, y + lw + 2),
                   f"{a['score']:.2f}", fill=(255, 90, 90))
        im.save(out_dir / f"{img_path.stem}_preview.jpg", quality=90)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder with raw images — annotations.json "
                                      "is created inside it")
    ap.add_argument("--prompt", default="person",
                    help='SAM 3 concept prompt (default: "person"; try '
                         '"anime character", "girl", "chibi character", ...)')
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="detection confidence threshold (default 0.5; lower to "
                         "~0.3-0.4 if small background characters are missed)")
    ap.add_argument("--mask-threshold", type=float, default=0.5,
                    help="mask binarization threshold (default 0.5)")
    ap.add_argument("--model", default="facebook/sam3",
                    help="HF model id (default: facebook/sam3)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="inference device (default: auto)")
    ap.add_argument("--recursive", action="store_true", help="scan subfolders too")
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="draw boxes on the first N images into <folder>/previews")
    ap.add_argument("--save-masks", action="store_true",
                    help="also store SAM 3 segmentation masks as COCO RLE "
                         "(needs `pip install pycocotools`; larger JSON)")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-detect every image even if already in the manifest")
    ap.add_argument("-o", "--output", default=None,
                    help="override the manifest path (default: <input_dir>/annotations.json)")
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    out_path = Path(args.output) if args.output else root / "annotations.json"

    images = find_images(root, args.recursive)
    if not images:
        sys.exit(f"no images found in {root} (extensions: {sorted(IMAGE_EXTS)})")

    if args.save_masks:
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            sys.exit("--save-masks needs pycocotools:  pip install pycocotools")

    data = None if args.no_resume else (load_existing(out_path) if out_path.exists() else None)
    if data is None:
        data = new_manifest(args)
    else:
        print(f"[resume] continuing {out_path} "
              f"({len(data['images'])} image(s) already done; --no-resume to redo)")

    done = {im["file_name"] for im in data["images"]}
    unl_id = unlabeled_cat_id(data)
    next_img_id = max((im["id"] for im in data["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in data["annotations"]), default=0) + 1

    todo = [p for p in images if str(p.relative_to(root)) not in done]
    if not todo:
        print("nothing to do — every image is already in the manifest")
        print(f"manifest -> {out_path.resolve()}")
        return

    torch, model, processor, device = load_sam3(args.model, args.device)
    if device == "cuda":
        # this is a one-shot batch job, not a latency-sensitive service — trade
        # the caching allocator's speed for a flat, predictable VRAM footprint
        # instead of it growing to fit every new image shape it meets
        torch.cuda.caching_allocator_enable(False)
    print(f"SAM 3 ready on {device} · prompt={args.prompt!r} · "
          f"{len(todo)}/{len(images)} image(s) to process")

    previews_left = args.preview
    errors = 0

    def save():
        out_path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    for idx, img_path in enumerate(todo, 1):
        rel = str(img_path.relative_to(root))
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                boxes, scores, masks = detect(
                    torch, model, processor, device, im,
                    args.prompt, args.threshold, args.mask_threshold)
        except Exception as e:
            errors += 1
            print(f"[{idx}/{len(todo)}] {rel}: ERROR {e}", file=sys.stderr)
            continue

        data["images"].append({"id": next_img_id, "file_name": rel,
                               "width": W, "height": H})
        img_anns = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = sorted((max(0.0, min(W, x0)), max(0.0, min(W, x1))))
            y0, y1 = sorted((max(0.0, min(H, y0)), max(0.0, min(H, y1))))
            w, h = x1 - x0, y1 - y0
            if w < 1 or h < 1:
                continue
            ann = {
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": unl_id,
                "bbox": [round(x0, 1), round(y0, 1), round(w, 1), round(h, 1)],
                "area": round(w * h, 1),
                "iscrowd": 0,
                "score": round(score, 4),
                "source": "detector",
            }
            if args.save_masks and masks is not None:
                try:
                    ann["segmentation"] = rle_encode(masks[i])
                except Exception as e:
                    print(f"[warn] mask encode failed on {rel}: {e}", file=sys.stderr)
            data["annotations"].append(ann)
            img_anns.append(ann)
            next_ann_id += 1

        if previews_left > 0:
            draw_preview(img_path, img_anns, root / "previews")
            previews_left -= 1

        print(f"[{idx}/{len(todo)}] {rel}: {len(img_anns)} match(es)")
        next_img_id += 1
        if idx % 20 == 0:      # checkpoint so a crash doesn't lose everything
            save()

    data["images"].sort(key=lambda x: x["file_name"])
    save()

    print(f"\ndone: {len(data['images'])} image(s), {len(data['annotations'])} box(es)"
          + (f", {errors} error(s)" if errors else ""))
    print(f"manifest -> {out_path.resolve()}")
    print("next: open 2_annotate.html and load this folder — the JSON is picked "
          "up automatically")


if __name__ == "__main__":
    main()
