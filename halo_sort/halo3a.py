"""HALO-SORT tracker: the base tracker (halo3) plus optional association, identity and glare-coast terms.

HALO_ASSOC (comma-separated terms, empty = halo3):
  hm   Hybrid-SORT height-modulated IoU: affinity = IoU * HIoU (vertical overlap / vertical union)
  ocm  OC-SORT observation-centric momentum: + vdc * conf * (pi/2 - angle)/pi, delta_t=3, vdc=0.2 (OC-SORT defaults)
  ri   GlareTrack's _resolve_identity on the stage-1 matches, with its published constants (AMBIG_IOU .2,
       AMBIG_MARGIN .1, SWAP_SEP_FRAC .25): a detection contested by an overlapping unmatched track goes to the
       track whose last observation explains it best; a matched pair that inverts its recent order in both x
       and y is swapped back. Recent order uses the last 6 observed centres (GlareTrack uses output centres).
  inh  confirm-time identity inheritance: when a new track is confirmed (its MIN_HITS-th hit, before it was ever
       emitted) and exactly one confirmed, currently lost track is near it by halo3's own birth-suppression
       geometry (centre distance < NEAR * size, against the lost track's prediction or last observation),
       the new track takes that identity and the lost track is retired. No constants beyond halo3's NEAR.
  inh3 (with inh) ambiguity is judged on all geometric candidates; the unique candidate is then vetoed unless its
       evidence covers its absence (hits >= frames lost). Same comparison as inh2, used as a veto only.
  inh2 (with inh) only lost tracks that were never observed together with the new track (last observation before
       its birth), and whose evidence covers their absence (hits >= frames lost).
  post output the Kalman posterior box instead of the raw detection (no new constant; same filter as halo3).
  emitall emit every confirmed coasting track for up to GLARE_EMIT frames, not only glare-caused ones (removes a rule).
  earn evidence-earned lifetime: a track survives while it has been observed at least as many frames as it has
       been lost (hits >= frames lost). Replaces MAX_AGE and GLARE_MAX_AGE (no constant).
  earng the same rule for glare-caused coasts only (replaces GLARE_MAX_AGE; plain coasts keep MAX_AGE).
  Glare-branch ablations (removal only; each drops constants, adds none):
  noexp   no torso-exposure cause (drops GLARE_E, TORSO_X_MARGIN, TORSO_Y_MARGIN)
  nobeacon no headlamp beacon: no cause, no KF centre update (drops the 7 lamp constants)
  noglare both of the above: no glare cause at all, so GLARE_MAX_AGE/GLARE_EMIT are unused (drops 12)
  noemit  glare coasts are kept alive (GLARE_MAX_AGE) but never emitted (drops GLARE_EMIT)
  noage   glare coasts are emitted (GLARE_EMIT) but live only MAX_AGE (drops GLARE_MAX_AGE)
  emitlife glare coasts are emitted for their whole glare lifetime (drops GLARE_EMIT; lifetime GLARE_MAX_AGE)
  TrackTrack (CVPR 2025) inheritances, no new constants:
  mnn  track-perspective association: instead of Hungarian, repeatedly accept pairs that are each other's best
       (mutual nearest) among gated pairs (IoU >= IOU), as TrackTrack's iterative_assignment, without its
       threshold-reduction schedule (the gate stays halo3's IOU). Used in both stages.
  del  detections removed by halo3's NMS are offered, after OCR, to still-unmatched confirmed tracks (IoU >= IOU
       with the prediction); they never start tracks (TrackTrack's deleted-detection candidates; stage order replaces
       its penalty).
  nsa  TrackTrack's Noise Scale Adaptive Kalman update (from GIAOTracker): the measurement covariance of each detected
       update is scaled by (1 - confidence). No constant.
  own  output ownership: a coasting track is not emitted when its predicted box overlaps (IoU >= DUP) a box
       observed this frame by another track. Output only; tracking state is unchanged.
  Glare-coast evidence terms (2026-10-18, round 3):
  neg   negative evidence: a glare coast is emitted only while its predicted torso is in glare / under a headlamp
        *now* (absence of a detection outside glare is evidence the person is gone). No constant.
  negl  neg + the glare cause is un-latched when the prediction leaves glare (lifetime falls back to MAX_AGE).
  ev50  a coast is emitted only while IoU(prediction, last observed box) >= 0.5 (the evaluation match threshold).
  ev30  the same with halo3's association gate IOU (.3): emitted only while it could still re-match its own evidence.
  nocap drop the GLARE_EMIT frame cap (use with neg/ev*, which then end emission).
  Low-confidence recovery (round 4):
  byte  ByteTrack second stage: detections below DET (down to the cache floor .1) not suppressed by a kept detection
        (IoU > NMS) are offered after OCR to still-unmatched confirmed tracks (IoU >= IOU); they never start tracks.
  gbyte the same, but only low detections whose own torso is in glare / under a headlamp (glare explains the low score).
The gate is unchanged: a pair is accepted only if its raw IoU >= IOU, so the new terms only re-rank
pairs that halo3 already considers valid. Everything else (OCR stage, glare, beacon, birth, lifetime) is halo3.
"""
import os
import numpy as np
from . import halo3
from .halo3 import HALOSORT as _Base, Track, _nms, iou_batch, linear_assignment, DET, BIRTH_MARGIN, MIN_HITS, IOU
from glaremot.external.ocsort.ocsort import k_previous_obs
from glaremot.external.ocsort.association import speed_direction_batch
from .boxes import norm_box_to_xyxy

VDC, DELTA_T = 0.2, 3
AMBIG_IOU, AMBIG_MARGIN, SWAP_SEP_FRAC = 0.20, 0.10, 0.25


def _hiou(a, b):
    lo = np.maximum(a[:, None, 1], b[None, :, 1]); hi = np.minimum(a[:, None, 3], b[None, :, 3])
    u = np.maximum(a[:, None, 3], b[None, :, 3]) - np.minimum(a[:, None, 1], b[None, :, 1])
    return np.clip(hi - lo, 0, None) / np.maximum(u, 1e-6)


def _assign(score, gate, mnn=False):
    if gate.size == 0 or gate.max() < IOU:
        return []
    if not mnn:
        return [(int(d), int(t)) for d, t in linear_assignment(-score) if gate[d, t] >= IOU]
    s = np.where(gate >= IOU, score, -np.inf).astype(np.float64); out = []
    while np.isfinite(s).any():
        bt = s.argmax(1); bd = s.argmax(0)
        new = [(d, int(bt[d])) for d in range(s.shape[0]) if np.isfinite(s[d, bt[d]]) and bd[bt[d]] == d]
        if not new:
            break
        for d, t in new:
            out.append((int(d), t)); s[d, :] = -np.inf; s[:, t] = -np.inf
    return out


class PostTrack(Track):
    """Emit the Kalman posterior (the filter's own fused estimate) instead of the raw detection box."""
    def out_box(self):
        return halo3.xyxy_to_norm_box(self.kf.get_state()[0], self.W, self.H)


class NSATrack(Track):
    """Measurement noise R scaled by (1 - confidence) for every detected update (TrackTrack / GIAOTracker NSA)."""
    def observe(self, row):
        kf = self.kf.kf; R0 = kf.R.copy(); kf.R = R0 * (1.0 - float(row[4]))
        try:
            super().observe(row)
        finally:
            kf.R = R0


class HALOSORT(_Base):
    def __init__(self, terms=None):
        super().__init__()
        terms = os.environ.get("HALO_ASSOC", "") if terms is None else terms
        self.terms = set(t for t in terms.split(",") if t)

    def _affinity(self, D, P, idx):
        iou = iou_batch(D[:, :4], P)
        s = iou.copy()
        if "hm" in self.terms:
            s = s * _hiou(D[:, :4], P)
        if "ocm" in self.terms and len(idx):
            trs = [self.tracks[t] for t in idx]
            prev = np.array([k_previous_obs(t.kf.observations, t.kf.age, DELTA_T) for t in trs], float).reshape(-1, 5)
            vel = np.array([t.kf.velocity if t.kf.velocity is not None else (0.0, 0.0) for t in trs], float).reshape(-1, 2)
            Y, X = speed_direction_batch(D[:, :4].astype(float), prev)          # tracks x dets
            cos = np.clip(vel[:, 1:] * X + vel[:, :1] * Y, -1, 1)      # velocity = (dy, dx)
            ang = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi
            ang = ang * (prev[:, 4:5] >= 0)
            s = s + (VDC * ang * D[None, :, 4]).T
        return s, iou

    def _resolve(self, m1, D, P, ut):
        if not m1 or len(self.tracks) < 2:
            return m1
        iou_dt = iou_batch(D[:, :4], P); iou_tt = iou_batch(P, P)
        unmatched = set(ut); out = []
        for di, ti in m1:
            win = ti
            for tj in sorted(unmatched):
                if iou_tt[win, tj] < AMBIG_IOU or iou_dt[di, tj] < iou_dt[di, win] - AMBIG_MARGIN:
                    continue
                lw = iou_batch(D[di:di + 1, :4], self.tracks[win].box[None])[0, 0]
                lj = iou_batch(D[di:di + 1, :4], self.tracks[tj].box[None])[0, 0]
                rec = abs(lj - lw) <= 1e-6 and self.tracks[tj].kf.time_since_update < self.tracks[win].kf.time_since_update
                if lj > lw + 1e-6 or rec:
                    win = tj
            if win != ti:
                unmatched.discard(win); unmatched.add(ti)
            out.append((di, win))

        def med(t):
            h = self.tracks[t].kf.history_observations[-6:]
            c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in h], float)
            return np.median(c, 0), len(h)
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                (di, ti), (dj, tj) = out[a], out[b]
                if iou_tt[ti, tj] < AMBIG_IOU:
                    continue
                (hi, ni), (hj, nj) = med(ti), med(tj)
                if ni < 3 or nj < 3:
                    continue
                ci = (D[di, :2] + D[di, 2:4]) / 2; cj = (D[dj, :2] + D[dj, 2:4]) / 2
                ref = (max(D[di, 2] - D[di, 0], D[dj, 2] - D[dj, 0], 1.0), max(D[di, 3] - D[di, 1], D[dj, 3] - D[dj, 1], 1.0))
                flip = all(abs(hi[k] - hj[k]) > SWAP_SEP_FRAC * ref[k] and abs(ci[k] - cj[k]) > SWAP_SEP_FRAC * ref[k]
                           and np.sign(ci[k] - cj[k]) == -np.sign(hi[k] - hj[k]) for k in (0, 1))
                if flip:
                    out[a], out[b] = (di, tj), (dj, ti)
        return out

    @staticmethod
    def _close(b, ref):
        c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (ref[:2] + ref[2:4])
        sz = max(b[2] - b[0], b[3] - b[1], ref[2] - ref[0], ref[3] - ref[1], 1.0)
        return float(np.hypot(*c)) < halo3.NEAR * sz

    def _inherit(self):
        if self.frame <= MIN_HITS:
            return
        new = [t for t in self.tracks if t.kf.hits == MIN_HITS and t.kf.time_since_update == 0 and not getattr(t, "seen", False)]
        gone = set()
        for nt in new:
            cand = [o for o in self.tracks if o is not nt and id(o) not in gone and getattr(o, "seen", False)
                    and o.kf.time_since_update > 0 and o.alive()
                    and ("inh2" not in self.terms or (o.kf.time_since_update > nt.kf.age          # never co-observed
                                                      and o.kf.hits >= o.kf.time_since_update))  # earned its absence
                    and any(self._close(nt.box, np.asarray(r, np.float32)) for r in (o.kf.get_state()[0][:4], o.box))]
            if len(cand) == 1 and ("inh3" not in self.terms or cand[0].kf.hits >= cand[0].kf.time_since_update):
                o = cand[0]; nt.id = o.id; gone.add(id(o))
        for t in self.tracks:
            if t.kf.hits >= MIN_HITS:
                t.seen = True       # confirmed (emitted) identity
        if gone:
            self.tracks = [t for t in self.tracks if id(t) not in gone]

    def _alive(self, t):
        g = t.kf.time_since_update
        if "earn" in self.terms:
            return g <= t.kf.hits
        if "earng" in self.terms and t.cause == 'glare':
            return g <= t.kf.hits
        if "noage" in self.terms:
            return g <= halo3.MAX_AGE
        return t.alive()

    def _emit(self, ts):
        if "own" not in self.terms:
            return ts
        obs = np.asarray([t.box for t in ts if t.kf.time_since_update == 0], np.float32).reshape(-1, 4)
        keep = []
        for t in ts:
            if t.kf.time_since_update > 0 and len(obs):
                if iou_batch(t.kf.get_state()[0][None, :4].astype(np.float32), obs).max() >= halo3.DUP:
                    continue
            keep.append(t)
        return keep

    def step(self, dets, W, H, frame_idx, sat=None, byte_dets=None):
        self.frame += 1
        rows = [list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if d['conf'] >= DET]
        A = np.asarray(rows, np.float32).reshape(-1, 5); D = _nms(A)
        X = A[[i for i in range(len(A)) if not any(np.array_equal(A[i], r) for r in D)]] if "del" in self.terms else A[:0]
        P = np.asarray([t.kf.predict()[0][:4] for t in self.tracks], np.float32).reshape(-1, 4)
        ud, ut = list(range(len(D))), list(range(len(P)))
        matches = []
        if ud and ut:
            s, g = self._affinity(D, P, ut)
            m1 = _assign(s, g, "mnn" in self.terms)
            if "ri" in self.terms:
                m1 = self._resolve(m1, D, P, sorted(set(ut) - {t for _, t in m1}))
            matches += m1
            ud = sorted(set(ud) - {d for d, _ in m1}); ut = sorted(set(ut) - {t for _, t in m1})
        if ud and ut:
            L = np.asarray([self.tracks[t].box for t in ut], np.float32)
            iL = iou_batch(D[ud, :4], L)
            m2 = _assign(iL, iL, True) if "mnn" in self.terms else halo3._hungarian(iL)
            matches += [(ud[d], ut[t]) for d, t in m2]
            ud = sorted(set(ud) - {ud[d] for d, _ in m2}); ut = sorted(set(ut) - {ut[t] for _, t in m2})
        for d, t in matches:
            self.tracks[t].observe(D[d])
        if ("byte" in self.terms or "gbyte" in self.terms) and ut:
            lo = np.asarray([list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if d['conf'] < DET],
                            np.float32).reshape(-1, 5)
            if len(lo) and len(D):
                lo = lo[iou_batch(lo[:, :4], D[:, :4]).max(1) <= halo3.NMS]
            if len(lo) and "gbyte" in self.terms:
                lo = lo[[sat is not None and (halo3._torso_exposure(sat, r[:4], W, H) >= halo3.GLARE_E
                                              or halo3._headlamp(sat, r[:4], W, H) is not None) for r in lo]]
            cu = [t for t in ut if self.tracks[t].kf.hits >= MIN_HITS]
            if len(lo) and cu:
                iB = iou_batch(lo[:, :4], P[cu])
                mb = halo3._hungarian(iB)
                for d, t in mb:
                    self.tracks[cu[t]].observe(lo[d])
                ut = sorted(set(ut) - {cu[t] for _, t in mb})
        if len(X) and ut:
            cu = [t for t in ut if self.tracks[t].kf.hits >= MIN_HITS]
            if cu:
                iX = iou_batch(X[:, :4], P[cu])
                m3 = _assign(iX, iX, "mnn" in self.terms)
                for d, t in m3:
                    self.tracks[cu[t]].observe(X[d])
                ut = sorted(set(ut) - {cu[t] for _, t in m3})
        for t in self.tracks:
            t.now = False
        for t in ut:
            tr = self.tracks[t]
            tr.kf.update(None)
            if sat is None or tr.kf.hits < MIN_HITS:
                continue
            if "noglare" in self.terms:
                continue
            pred = tr.kf.get_state()[0][:4]
            blinded = "noexp" not in self.terms and halo3._torso_exposure(sat, pred, W, H) >= halo3.GLARE_E
            lamp = "nobeacon" not in self.terms and self._beacon(tr, pred, sat, W, H)
            tr.now = bool(blinded or lamp)
            if blinded or lamp:
                tr.cause = 'glare'
            elif "negl" in self.terms:
                tr.cause = 'plain'
        for d in ud:
            r = D[d]
            if r[4] < DET + BIRTH_MARGIN or self._near_track(r[:4]):
                continue
            self.n += 1
            self.tracks.append((PostTrack if "post" in self.terms else NSATrack if "nsa" in self.terms else Track)(self.n, r, W, H))
        if "inh" in self.terms:
            self._inherit()
        self.tracks = [t for t in self.tracks if self._alive(t)]
        return self._emit([t for t in self.tracks
                if (t.kf.hits >= MIN_HITS or self.frame <= MIN_HITS)
                and (t.kf.time_since_update == 0
                     or ((t.cause == 'glare' or "emitall" in self.terms) and "noemit" not in self.terms
                         and ("emitlife" in self.terms or "nocap" in self.terms or t.kf.time_since_update <= halo3.GLARE_EMIT)
                         and self._evidence(t)))])

    def _evidence(self, t):
        if ("neg" in self.terms or "negl" in self.terms) and not getattr(t, "now", False):
            return False
        thr = 0.5 if "ev50" in self.terms else IOU if "ev30" in self.terms else None
        if thr is not None:
            p = t.kf.get_state()[0][None, :4].astype(np.float32)
            if iou_batch(p, np.asarray(t.box, np.float32)[None])[0, 0] < thr:
                return False
        return True
