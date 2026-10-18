"""HALO-SORT base tracker.

IoU association + observation-centric recovery (OC-SORT), duplicate-birth suppression, a glare-aware lifetime
(tracks whose predicted torso is saturated, or whose head region shows a helmet-lamp blob, are kept alive longer
and may emit their Kalman prediction), and a weak Kalman centre update from the helmet lamp.
"""
import numpy as np
from glaremot.external.ocsort.association import iou_batch, linear_assignment
from glaremot.external.ocsort.ocsort import KalmanBoxTracker
from .boxes import norm_box_to_xyxy, xyxy_to_norm_box

# Association, confirmation and lifecycle: 12 scalar design settings.
DET = 0.25
BIRTH_MARGIN = 0.10
IOU = 0.30
NMS = 0.60
MIN_HITS = 3
MAX_AGE = 60
DUP = 0.35
NEAR_IOU = 0.20
NEAR = 0.55
GLARE_E = 0.08
GLARE_MAX_AGE = 160
GLARE_EMIT = 12

# Nine geometry/noise choices of the helmet-lamp beacon and torso exposure.
TORSO_X_MARGIN = 0.20
TORSO_Y_MARGIN = 0.25
HEAD_FRACTION = 0.25
LAMP_MIN_PIXELS = 2
LAMP_MAX_FRACTION = 0.50
LAMP_CENTER_OFFSET = 0.375
LAMP_SIGMA_HEIGHT = 0.08
LAMP_VARIANCE_FLOOR = 25.0
LAMP_Y_VARIANCE_MULTIPLIER = 4.0

# Cache representation and output confidence, not association thresholds.
MASK_S = 512
SCORE_DECAY = 0.90
SCORE_FLOOR = 0.01


def config():
    """Include geometry and output constants in the scalar inventory."""
    return {name: value for name, value in globals().items()
            if name.isupper() and isinstance(value, (int, float))}


def _nms(D):
    if len(D) <= 1:
        return D
    keep = []
    for i in np.argsort(-D[:, 4]):
        if all(iou_batch(D[i:i + 1, :4], D[j:j + 1, :4])[0, 0] <= NMS for j in keep):
            keep.append(int(i))
    keep.sort()
    return D[keep]


def _torso_exposure(sat, b, W, H):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    sx, sy = MASK_S / W, MASK_S / H
    a = int(max(0, (x1 + TORSO_X_MARGIN * w) * sx))
    c = int(min(MASK_S, np.ceil((x2 - TORSO_X_MARGIN * w) * sx)))
    d = int(max(0, (y1 + TORSO_Y_MARGIN * h) * sy))
    e = int(min(MASK_S, np.ceil((y2 - TORSO_Y_MARGIN * h) * sy)))
    return float(sat[d:e, a:c].mean()) if c > a and e > d else 0.0


def _weak_center_update(kf, zx, zy, r):
    """Centre-only update; preserve the OC-SORT observation history for ORU."""
    Hm = kf.H[:2, :]
    R = np.diag([r, LAMP_Y_VARIANCE_MULTIPLIER * r])
    y = np.array([[zx], [zy]], np.float64) - Hm @ kf.x
    PHT = kf.P @ Hm.T
    K = PHT @ np.linalg.inv(Hm @ PHT + R)
    kf.x = kf.x + K @ y
    I_KH = np.eye(kf.dim_x) - K @ Hm
    kf.P = I_KH @ kf.P @ I_KH.T + K @ R @ K.T


def _headlamp(sat, b, W, H):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    sx, sy = MASK_S / W, MASK_S / H
    a, c = int(max(0, x1 * sx)), int(min(MASK_S, np.ceil(x2 * sx)))
    d = int(max(0, y1 * sy))
    e = int(min(MASK_S, np.ceil((y1 + HEAD_FRACTION * h) * sy)))
    if c <= a or e <= d:
        return None
    roi = sat[d:e, a:c]
    ys, xs = np.nonzero(roi)
    if xs.size < LAMP_MIN_PIXELS or xs.size > LAMP_MAX_FRACTION * roi.size:
        return None
    return (xs.mean() + a + 0.5) / sx, (ys.mean() + d + 0.5) / sy


def _hungarian(iou):
    if iou.size == 0 or iou.max() < IOU:
        return []
    return [(int(d), int(t)) for d, t in linear_assignment(-iou) if iou[d, t] >= IOU]


class Track:
    def __init__(self, tid, row, W, H):
        self.id = tid
        self.kf = KalmanBoxTracker(row[:4])
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.W, self.H = W, H
        self.box = row[:4].copy()
        self.conf = float(row[4])
        self.cause = 'plain'

    @property
    def m(self):
        return float(np.clip(self.conf * SCORE_DECAY ** self.kf.time_since_update,
                             SCORE_FLOOR, 1.0))

    def out_box(self):
        b = self.box if self.kf.time_since_update == 0 else self.kf.get_state()[0]
        return xyxy_to_norm_box(b, self.W, self.H)

    def observe(self, row):
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.box, self.conf, self.cause = row[:4].copy(), float(row[4]), 'plain'

    def alive(self):
        cap = GLARE_MAX_AGE if self.cause == 'glare' else MAX_AGE
        return self.kf.time_since_update <= cap


class HALOSORT:
    def __init__(self):
        self.tracks = []
        self.n = 0
        self.frame = 0

    def step(self, dets, W, H, frame_idx, sat=None, byte_dets=None):
        self.frame += 1
        rows = [list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']]
                for d in dets if d['conf'] >= DET]
        D = _nms(np.asarray(rows, np.float32).reshape(-1, 5))
        P = np.asarray([t.kf.predict()[0][:4] for t in self.tracks],
                       np.float32).reshape(-1, 4)
        ud, ut = list(range(len(D))), list(range(len(P)))
        matches = []
        if ud and ut:
            m1 = _hungarian(iou_batch(D[:, :4], P))
            matches += m1
            ud = sorted(set(ud) - {d for d, _ in m1})
            ut = sorted(set(ut) - {t for _, t in m1})

        # Observation-centric recovery against the last detected boxes.
        if ud and ut:
            L = np.asarray([self.tracks[t].box for t in ut], np.float32)
            m2 = _hungarian(iou_batch(D[ud, :4], L))
            matches += [(ud[d], ut[t]) for d, t in m2]
            ud = sorted(set(ud) - {ud[d] for d, _ in m2})
            ut = sorted(set(ut) - {ut[t] for _, t in m2})
        for d, t in matches:
            self.tracks[t].observe(D[d])

        for t in ut:
            tr = self.tracks[t]
            tr.kf.update(None)
            if sat is None or tr.kf.hits < MIN_HITS:
                continue
            pred = tr.kf.get_state()[0][:4]
            blinded = _torso_exposure(sat, pred, W, H) >= GLARE_E
            lamp = self._beacon(tr, pred, sat, W, H)
            if blinded or lamp:
                tr.cause = 'glare'  # Latched until the next detected observation.

        for d in ud:
            r = D[d]
            if r[4] < DET + BIRTH_MARGIN or self._near_track(r[:4]):
                continue
            self.n += 1
            self.tracks.append(Track(self.n, r, W, H))
        self.tracks = [t for t in self.tracks if t.alive()]
        return [t for t in self.tracks
                if (t.kf.hits >= MIN_HITS or self.frame <= MIN_HITS)
                and (t.kf.time_since_update == 0
                     or (t.cause == 'glare' and t.kf.time_since_update <= GLARE_EMIT))]

    @staticmethod
    def _beacon(tr, pred, sat, W, H):
        p = _headlamp(sat, pred, W, H)
        if p is None:
            return False
        h = float(pred[3] - pred[1])
        r = max(LAMP_VARIANCE_FLOOR, (LAMP_SIGMA_HEIGHT * h) ** 2)
        _weak_center_update(tr.kf.kf, p[0], p[1] + LAMP_CENTER_OFFSET * h, r)
        return True

    def _near_track(self, b):
        for t in self.tracks:
            for ref in (t.kf.get_state()[0][:4], t.box):
                v = float(iou_batch(b[None], np.asarray(ref, np.float32)[None])[0, 0])
                if v >= DUP:
                    return True
                if v >= NEAR_IOU:
                    c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (ref[:2] + ref[2:4])
                    s = max(b[2] - b[0], b[3] - b[1], ref[2] - ref[0], ref[3] - ref[1], 1.0)
                    if float(np.hypot(*c)) < NEAR * s:
                        return True
        return False
