#!/usr/bin/env python3
"""
Step 3 — Turn the labeled COCO annotations into training data.

Consumes the COCO JSON exported from the labeling app (2_annotate.html) —
or the raw step-1 file if you only need person boxes — and produces:

  yolo   Ultralytics detection dataset, character names as classes:
             out/yolo/images/{train,val}/…
             out/yolo/labels/{train,val}/…   (class cx cy w h, normalized)
             out/yolo/data.yaml
  crops  Cropped characters for classifier training:
             out/crops/<character>/<image-stem>_<ann-id>.png
  both   (default)

Category semantics:
  "unlabeled"  boxes are skipped; images that still contain any are left out
               of the YOLO set entirely (they'd teach the detector that those
               characters are background) — finish labeling them first.
  "ignore"     boxes are excluded from classes; images where every box is
               "ignore" are kept in the YOLO set as clean negatives.

--images-dir defaults to the folder the JSON lives in, so after step 1 wrote
annotations.json into your image folder this is just:

    python 3_export.py raw_images/annotations_labeled.json -o ./dataset
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPECIAL = {"unlabeled", "ignore"}


def sanitize(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip().lower())
    s = re.sub(r"[^\w\-.]", "", s, flags=re.UNICODE)
    return s or "unnamed"


def resolve_image(images_dir: Path, file_name: str):
    p = images_dir / file_name
    if p.is_file():
        return p
    hits = list(images_dir.rglob(Path(file_name).name))
    return hits[0] if hits else None


def clamp_xyxy(x0, y0, x1, y1, w, h):
    x0, x1 = sorted((max(0.0, min(w, x0)), max(0.0, min(w, x1))))
    y0, y1 = sorted((max(0.0, min(h, y0)), max(0.0, min(h, y1))))
    return x0, y0, x1, y1


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="COCO JSON (labeled export from the app)")
    ap.add_argument("--images-dir", default=None,
                    help="root folder holding the original images "
                         "(default: the folder the JSON is in)")
    ap.add_argument("-o", "--out", default="dataset", help="output folder")
    ap.add_argument("--format", choices=["yolo", "crops", "both"], default="both")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="fraction of images used for validation (default 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="split seed")
    ap.add_argument("--pad", type=float, default=0.05,
                    help="padding around crops as a fraction of box size (default 0.05)")
    ap.add_argument("--min-crop", type=int, default=32,
                    help="skip crops smaller than this many px on a side (default 32)")
    ap.add_argument("--link", action="store_true",
                    help="symlink images into the YOLO set instead of copying")
    return ap


def run(args, progress_cb=None):
    """Core export logic. Raises ValueError on bad input (caller decides how
    to surface that -- the CLI turns it into sys.exit, the app backend turns
    it into an HTTP error). Returns a summary dict. `progress_cb(done, total)`
    is called periodically if provided."""
    ann_path = Path(args.annotations)
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    if "annotations" not in data and any("detections" in im for im in data.get("images", [])):
        raise ValueError("this looks like the old (pre-COCO) manifest format — open it in "
                         "2_annotate.html and hit EXPORT to migrate it, then re-run")
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"not a COCO file: missing '{key}' list")

    images_dir = Path(args.images_dir) if args.images_dir else ann_path.resolve().parent
    if not images_dir.is_dir():
        raise ValueError(f"--images-dir is not a directory: {images_dir}")
    out = Path(args.out)

    cat_name = {c["id"]: str(c["name"]).strip() for c in data["categories"]}
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # ---- collect classes -------------------------------------------------
    counts, n_unlabeled, n_ignore = Counter(), 0, 0
    for a in data["annotations"]:
        name = cat_name.get(a["category_id"], "unlabeled")
        low = name.lower()
        if low == "unlabeled":
            n_unlabeled += 1
        elif low == "ignore":
            n_ignore += 1
        else:
            counts[name] += 1
    if not counts:
        raise ValueError("no character-labeled boxes found — label boxes in 2_annotate.html first")

    classes = sorted(counts)
    class_id = {c: i for i, c in enumerate(classes)}
    safe_name = {}
    for c in classes:
        s = base = sanitize(c)
        k = 2
        while s in safe_name.values():
            s = f"{base}_{k}"
            k += 1
        safe_name[c] = s

    # ---- pick exportable images ------------------------------------------
    complete, incomplete = [], 0
    for im in data["images"]:
        anns = anns_by_img.get(im["id"], [])
        if any(cat_name.get(a["category_id"], "unlabeled").lower() == "unlabeled"
               for a in anns):
            incomplete += 1
            continue
        complete.append(im)
    if n_unlabeled:
        print(f"[warn] {n_unlabeled} unlabeled box(es) across {incomplete} image(s) "
              f"-> those images are excluded from the YOLO set "
              f"(their labeled boxes still go to crops)", file=sys.stderr)

    rng = random.Random(args.seed)
    order = list(range(len(complete)))
    rng.shuffle(order)
    n_val = max(1, round(len(complete) * args.val_ratio)) \
        if len(complete) > 1 and args.val_ratio > 0 else 0
    val_ids = {complete[j]["id"] for j in order[:n_val]}
    complete_ids = {im["id"] for im in complete}

    do_yolo = args.format in ("yolo", "both")
    do_crops = args.format in ("crops", "both")
    if do_yolo:
        for split in ("train", "val"):
            (out / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
    if do_crops:
        (out / "crops").mkdir(parents=True, exist_ok=True)

    missing, yolo_boxes, yolo_imgs, crop_count, skipped_small = [], 0, 0, 0, 0

    total_images = len(data["images"])
    for idx, im in enumerate(data["images"], 1):
        if progress_cb:
            progress_cb(idx, total_images)
        char_anns = [a for a in anns_by_img.get(im["id"], [])
                     if cat_name.get(a["category_id"], "").lower() not in SPECIAL
                     and cat_name.get(a["category_id"]) in class_id]
        wants_yolo = do_yolo and im["id"] in complete_ids
        wants_crops = do_crops and bool(char_anns)
        if not (wants_yolo or wants_crops):
            continue
        src = resolve_image(images_dir, im["file_name"])
        if src is None:
            missing.append(im["file_name"])
            continue
        W, H = float(im["width"]), float(im["height"])
        stem = Path(im["file_name"]).stem

        if wants_yolo:
            split = "val" if im["id"] in val_ids else "train"
            lines = []
            for a in char_anns:
                x, y, w, h = (float(v) for v in a["bbox"])
                x0, y0, x1, y1 = clamp_xyxy(x, y, x + w, y + h, W, H)
                bw, bh = x1 - x0, y1 - y0
                if bw < 2 or bh < 2:
                    continue
                cid = class_id[cat_name[a["category_id"]]]
                lines.append(f"{cid} {(x0 + bw / 2) / W:.6f} {(y0 + bh / 2) / H:.6f} "
                             f"{bw / W:.6f} {bh / H:.6f}")
            # empty label files are written too: all-ignore images = negatives
            img_dst = out / "yolo" / "images" / split / src.name
            if not img_dst.exists():
                if args.link:
                    try:
                        img_dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, img_dst)
                else:
                    shutil.copy2(src, img_dst)
            (out / "yolo" / "labels" / split / f"{src.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            yolo_boxes += len(lines)
            yolo_imgs += 1

        if wants_crops:
            with Image.open(src) as pil:
                pil = pil.convert("RGB")
                iw, ih = pil.size
                sx, sy = (iw / W, ih / H) if (iw, ih) != (W, H) else (1.0, 1.0)
                for a in char_anns:
                    x, y, w, h = (float(v) for v in a["bbox"])
                    x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
                    pw, ph = (x1 - x0) * args.pad, (y1 - y0) * args.pad
                    box = clamp_xyxy(x0 - pw, y0 - ph, x1 + pw, y1 + ph, iw, ih)
                    box = tuple(int(round(v)) for v in box)
                    if box[2] - box[0] < args.min_crop or box[3] - box[1] < args.min_crop:
                        skipped_small += 1
                        continue
                    folder = out / "crops" / safe_name[cat_name[a["category_id"]]]
                    folder.mkdir(parents=True, exist_ok=True)
                    pil.crop(box).save(folder / f"{stem}_{a['id']:04d}.png")
                    crop_count += 1

    # ---- metadata ----------------------------------------------------------
    if do_yolo:
        yaml_lines = [f"path: {(out / 'yolo').resolve()}",
                      "train: images/train",
                      "val: images/val" if n_val else "val: images/train",
                      "names:"]
        yaml_lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
        (out / "yolo" / "data.yaml").write_text("\n".join(yaml_lines) + "\n",
                                                encoding="utf-8")
        (out / "yolo" / "classes.txt").write_text("\n".join(classes) + "\n",
                                                  encoding="utf-8")

    print(f"\nclasses ({len(classes)}):")
    for c in classes:
        print(f"  {class_id[c]:>3}  {c:<24} {counts[c]} box(es)")
    if n_ignore:
        print(f"       ('ignore' boxes excluded: {n_ignore})")
    if do_yolo:
        print(f"yolo : {yolo_imgs} image(s) "
              f"({n_val} val), {yolo_boxes} box(es) -> {out / 'yolo'}")
        print(f"       train: yolo detect train data={out / 'yolo' / 'data.yaml'} "
              f"model=yolov8m.pt imgsz=1024")
    if do_crops:
        print(f"crops: {crop_count} crop(s)"
              + (f", {skipped_small} skipped (<{args.min_crop}px)" if skipped_small else "")
              + f" -> {out / 'crops'}")
    if missing:
        print(f"[warn] {len(missing)} image file(s) not found under {images_dir}: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)

    return {
        "out": str(out.resolve()),
        "classes": [{"id": class_id[c], "name": c, "boxes": counts[c]} for c in classes],
        "n_ignore": n_ignore,
        "yolo": {"images": yolo_imgs, "val_images": n_val, "boxes": yolo_boxes,
                 "path": str((out / "yolo").resolve())} if do_yolo else None,
        "crops": {"count": crop_count, "skipped_small": skipped_small,
                  "path": str((out / "crops").resolve())} if do_crops else None,
        "missing": missing,
    }


def main():
    args = build_arg_parser().parse_args()
    try:
        run(args)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
