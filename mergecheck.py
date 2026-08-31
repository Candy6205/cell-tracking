"""mergecheck.py -- is the arm we are about to submit the arm we measured?

mergescore's M15 read +3.94% nodes on 44b6 and +4.07% on 6bba. The
submit_m15 visible run, whose four datasets are 2x 44b6 + 2x 6bba, added
only 2.06%. Density does not explain it: offline 44b6 sits at 365
points/frame and 6bba at 157, a 2.3x spread that moved the add rate by
0.13 points, so a jump to 410/frame cannot halve it.

Two candidate explanations, and they have different consequences:

  A. COMPOSITION. The add rate is a per-dataset quantity with a wide
     spread, and the visible four happen to sit low. Nothing is wrong;
     the offline domain means were just never a prediction for four
     specific movies.

  B. THE ARMS DIFFER. submit_m15 recomputes ep05 peaks from hs[3]
     inside detect(); mergescore read them from dccache3's cached
     dcdet_ep05_*.npz. If those two disagree, the submission is not the
     thing that scored +0.0060 / +0.0104 and must not be submitted.

This kernel prints the per-dataset add count for all 24 cached
competition datasets using mergescore's exact admission rule. It settles
A on its own -- if the spread covers 2%, composition is sufficient. And
44b6_0b24845f is in BOTH the cached set and the visible four, so its
number here is a direct comparison point against the instrumented
submission rerun, which settles B.

CPU only, seconds of work, no GPU quota.
"""
import os
import pathlib
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-08-31a-mergecheck"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MERGE_TAU = 0.15
MERGE_UM = 4.0


def _walk_npz(prefix):
    """mergescore's bounded walk, verbatim (trap 20: prune the
    competition tree and the zarr/geff trees or this takes minutes)."""
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


FDET = defaultdict(dict)
for _k, _p in _walk_npz("fdet_").items():
    _arm, _nm = _k.split("_", 1)
    FDET[_arm][_nm] = _p
print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)

DCDET_EP05 = _walk_npz("dcdet_ep05_")
print(f"dcdet_ep05 caches: {len(DCDET_EP05)}", flush=True)
assert len(FDET.get("F4", {})) >= 24, "attach biohub-fusecache"
assert len(DCDET_EP05) >= 24, "attach biohub-dccache3"


def add_count(name):
    """mergescore.merge_dets' admission rule, counting only. An ep05
    point enters iff its own score >= MERGE_TAU and it is farther than
    MERGE_UM from every F4 point at that frame."""
    zf = np.load(FDET["F4"][name])
    zd = np.load(DCDET_EP05[name])
    base = added = frames = 0
    per_frame = []
    for t in range(10000):
        if f"c{t}" not in zf.files:
            break
        frames += 1
        c = np.asarray(zf[f"c{t}"], float).reshape(-1, 3)
        base += len(c)
        dc = np.zeros((0, 3))
        if f"c{t}" in zd.files:
            dc = np.asarray(zd[f"c{t}"], float).reshape(-1, 3)
            ds_ = np.asarray(zd[f"s{t}"], float).reshape(-1)
            dc = dc[ds_ >= MERGE_TAU]
        if len(dc) and len(c):
            d, _ = cKDTree(c * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > MERGE_UM]
        added += len(dc)
        per_frame.append(len(c))
    return base, added, frames, float(np.mean(per_frame)) if per_frame else 0.0


names = sorted(n for n in FDET["F4"] if n in DCDET_EP05
               and n.startswith(("6bba", "44b6")))
print(f"\n=== per-dataset add rate at tau {MERGE_TAU}, "
      f"{len(names)} cached competition datasets ===", flush=True)
print(f"  {'dataset':<20} {'F4 pts':>9} {'+ep05':>8} {'add%':>7} "
      f"{'pts/frame':>10}", flush=True)
rows = []
for n in names:
    b, a, fr, dens = add_count(n)
    rows.append((n, b, a, 100.0 * a / max(b, 1), dens))
    print(f"  {n:<20} {b:>9,} {a:>8,} {rows[-1][3]:>6.2f}% "
          f"{dens:>10.0f}", flush=True)

print("\n=== domain totals (must reproduce mergescore) ===", flush=True)
for dom, want in (("44b6", 3.94), ("6bba", 4.07)):
    sel = [r for r in rows if r[0].startswith(dom)]
    b = sum(r[1] for r in sel)
    a = sum(r[2] for r in sel)
    got = 100.0 * a / max(b, 1)
    print(f"  {dom}: +{a:,} on {b:,} = {got:.2f}% "
          f"(mergescore read {want:.2f}%) "
          f"{'OK' if abs(got - want) < 0.02 else 'MISMATCH'}", flush=True)

print("\n=== spread: does composition explain the visible-4's 2.06%? ===",
      flush=True)
pct = np.array([r[3] for r in rows])
print(f"  min {pct.min():.2f}%  p25 {np.percentile(pct, 25):.2f}%  "
      f"median {np.median(pct):.2f}%  p75 {np.percentile(pct, 75):.2f}%  "
      f"max {pct.max():.2f}%", flush=True)
print(f"  datasets at or below 2.06%: {int((pct <= 2.06).sum())} of "
      f"{len(pct)}", flush=True)
print("  If several cached datasets sit at or under 2.06%, the visible-4 "
      "number is ordinary composition and the arm is fine. If NONE do, "
      "the submission's recomputed ep05 peaks differ from the cache and "
      "the arm must not be submitted until that is explained.", flush=True)

print("\n=== the direct comparison point ===", flush=True)
for n in ("44b6_0b24845f",):
    if n in dict((r[0], r) for r in rows):
        r = [x for x in rows if x[0] == n][0]
        print(f"  {n} is in the cached set AND the visible four: "
              f"+{r[2]:,} ep05 on {r[1]:,} F4 = {r[3]:.2f}%. The "
              f"instrumented submit_m15 rerun must print the same "
              f"add% for this dataset.", flush=True)
    else:
        print(f"  {n} NOT in the cached set -- no direct comparison "
              f"available, conclusion rests on the spread alone",
              flush=True)

print("\nMERGECHECK DONE", flush=True)
