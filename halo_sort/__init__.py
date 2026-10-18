"""HALO-SORT: glare-aware, appearance-free multi-object tracking.

    from halo_sort import HALOSORT
    tracker = HALOSORT()                 # default terms: inh,own,neg,ev50
    tracks = tracker.step(dets, W, H, frame_idx, sat=saturation_mask)

dets: list of {"box": normalised (cx, cy, w, h), "conf": float}; sat: boolean saturation map (512x512).
"""
from .halo3a import HALOSORT as _HALOSORT

DEFAULT_TERMS = "inh,own,neg,ev50"


def HALOSORT(terms=DEFAULT_TERMS):
    return _HALOSORT(terms)
