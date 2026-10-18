"""Box conversions between normalised (cx, cy, w, h) and pixel (x1, y1, x2, y2)."""
import numpy as np


def xyxy_to_norm_box(xyxy, W, H):
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    x1 = max(0.0, min(float(W - 1), x1)); x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1)); y2 = max(0.0, min(float(H - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w = max(1.0, x2 - x1) / float(W)
    h = max(1.0, y2 - y1) / float(H)
    return np.array([((x1 + x2) * 0.5) / W, ((y1 + y2) * 0.5) / H, w, h], np.float32)


def norm_box_to_xyxy(box, W, H):
    cx, cy, w, h = [float(v) for v in box[:4]]
    return np.array([cx * W - w * W * 0.5, cy * H - h * H * 0.5,
                     cx * W + w * W * 0.5, cy * H + h * H * 0.5], np.float32)
