"""Step 08 — Run HALO-SORT on one sequence (a folder of frames) and write MOTChallenge rows.

Per frame: the shared YOLO11m detector (weights/detector/best.pt, imgsz 1536, conf >= 0.10), the photometric
saturation map at 512x512 (glaremot.tracking.tracker.fast_saturation_map), then HALOSORT.step.

Examples:
  python scripts/08_run_halo_sort.py --seq D:/datasets/GlareMOT/204_sequence_81q --out results/halo_sort/204_sequence_81q.txt
  python scripts/08_run_halo_sort.py --seq <dir> --terms inh,own      # another term set
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from glaremot.tracking.tracker import PHOTO_S, fast_saturation_map             # noqa: E402
from halo_sort import DEFAULT_TERMS, HALOSORT                                   # noqa: E402
from halo_sort.boxes import norm_box_to_xyxy                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="sequence folder with images/*.jpg (or the images folder itself)")
    ap.add_argument("--out", required=True, help="output MOT txt")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "detector", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--terms", default=DEFAULT_TERMS)
    a = ap.parse_args()

    img_dir = os.path.join(a.seq, "images") if os.path.isdir(os.path.join(a.seq, "images")) else a.seq
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        sys.exit(f"no frames in {img_dir}")
    from ultralytics import YOLO
    det = YOLO(a.weights)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trk = HALOSORT(a.terms)
    rows = []
    for f, p in enumerate(imgs, 1):
        im = cv2.imread(p)
        H, W = im.shape[:2]
        sat = fast_saturation_map(cv2.resize(im, (PHOTO_S, PHOTO_S), interpolation=cv2.INTER_AREA))
        with torch.no_grad():
            r = det.predict(im, imgsz=a.imgsz, conf=0.10, verbose=False, device=dev)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dets.append({"box": np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                              max(1.0, x2 - x1) / W, max(1.0, y2 - y1) / H], np.float32),
                             "conf": float(c)})
        for t in trk.step(dets, W, H, f, sat=sat):
            x1, y1, x2, y2 = norm_box_to_xyxy(t.out_box(), W, H)
            rows.append(f"{f},{t.id},{max(0., x1):.2f},{max(0., y1):.2f},{max(1., x2 - x1):.2f},{max(1., y2 - y1):.2f},"
                        f"{max(float(t.m), .001):.4f},-1,-1,-1")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{len(imgs)} frames, {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
