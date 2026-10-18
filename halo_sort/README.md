# HALO-SORT

Glare-aware, appearance-free multi-object tracking for mine cameras, on the same perception stack as PEG
(shared YOLO11m detector, photometric saturation map). No re-identification, no learned tracker component.

## How it works (per frame)

1. **Detections.** Keep boxes with confidence ≥ 0.25, then NMS at 0.6.
2. **Prediction.** Each track carries the OC-SORT Kalman box filter (centre, area, aspect, velocities).
3. **Association, two stages.**
   - Hungarian on IoU(detection, predicted box), accepted at IoU ≥ 0.3.
   - Observation-centric recovery: leftover detections against each leftover track's *last observed* box,
     with the same gate.
4. **Missed tracks: burnt or gone?** For a confirmed track without a detection:
   - *torso exposure*: fraction of saturated pixels in the torso of the predicted box (box shrunk by 20 % left/right,
     25 % top/bottom); ≥ 0.08 means the body is washed out;
   - *helmet lamp*: a small saturated blob in the head region (top 25 % of the box) is taken as the worker's lamp, and
     its centroid is a weak Kalman centre measurement, so the track keeps following the person while the detector is blind.

   Either cue marks the track **glare**, which lives up to 160 frames instead of 60.
5. **Birth.** An unmatched detection starts a track if its confidence is ≥ 0.35 and it is not a duplicate of an
   existing track (IoU ≥ 0.35, or IoU ≥ 0.2 with centres closer than 0.55 × box size).
6. **Output.** A confirmed track (≥ 3 hits) that was detected this frame outputs the detection box. A glare
   track that is coasting outputs its Kalman prediction for up to 12 frames, subject to the terms below.

## Terms on top of the base tracker (default `inh,own,neg,ev50`)

| term | what it does |
|---|---|
| `inh` | **Identity inheritance.** When a new track is confirmed, and exactly one confirmed, currently lost track is near it, the new track takes that identity and the lost track is retired. With two or more candidates nothing is inherited. |
| `own` | **Output ownership.** A coasting box is not output when it overlaps (IoU ≥ 0.35) a box observed this frame by another track. |
| `neg` | **Negative evidence.** A glare coast is output only while its predicted torso is in glare, or under a lamp, *now*. A missed detection in a clear region means the person has gone. |
| `ev50` | **Evidence consistency.** A coast is output only while IoU(prediction, last observed box) ≥ 0.5. |

`halo_sort/halo3a.py` documents further optional terms (`ri`, `hm`, `ocm`, `mnn`, `nsa`, `negl`, `nocap`, …).

## Constants

| group | constants |
|---|---|
| association / lifecycle | DET 0.25, BIRTH_MARGIN 0.10, IOU 0.30, NMS 0.60, MIN_HITS 3, MAX_AGE 60, DUP 0.35, NEAR_IOU 0.20, NEAR 0.55 |
| glare | GLARE_E 0.08, GLARE_MAX_AGE 160, GLARE_EMIT 12, TORSO_X_MARGIN 0.20, TORSO_Y_MARGIN 0.25 |
| helmet lamp | HEAD_FRACTION 0.25, LAMP_MIN_PIXELS 2, LAMP_MAX_FRACTION 0.50, LAMP_CENTER_OFFSET 0.375, LAMP_SIGMA_HEIGHT 0.08, LAMP_VARIANCE_FLOOR 25, LAMP_Y_VARIANCE_MULTIPLIER 4 |
| ev50 | 0.5 |

## Results (GlareMOT, official GT, TrackEval)

| tracker | split | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| HALO-SORT (`inh,own,neg,ev50`) | val | 72.84 | 69.31 | 77.15 | 89.99 | 83.18 | 9 |
| HALO-SORT (`inh,own,neg,ev50`) | test | **69.48** | 66.38 | 73.67 | 85.23 | 81.36 | 19 |

Test ablation:

| configuration | HOTA | IDSW |
|---|---|---|
| base tracker (no terms) | 68.67 | 23 |
| + `own` | 68.72 | 25 |
| + `own` + `inh` | 69.17 | 19 |
| + `own` + `inh` + `neg` + `ev50` | **69.48** | 19 |

## Run

```bash
python scripts/08_run_halo_sort.py --seq D:/datasets/GlareMOT/204_sequence_81q --out results/halo_sort/204_sequence_81q.txt
python scripts/08_run_halo_sort.py --seq <dir> --terms inh,own      # another term set
```

```python
from halo_sort import HALOSORT
trk = HALOSORT()                                  # inh,own,neg,ev50
for f, (dets, sat) in enumerate(stream, 1):       # dets: [{"box": (cx,cy,w,h) normalised, "conf": c}], sat: 512x512 bool
    for t in trk.step(dets, W, H, f, sat=sat):
        box, tid, score = t.out_box(), t.id, t.m
```

Evaluate the written MOT files with `scripts/06_eval_mot.py`.
