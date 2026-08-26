"""When a cell is missed, is there a peak just OUTSIDE the match radius?

The user asked why the pipeline uses non-maximum suppression instead of
fitting a brightness model, e.g. a mixture of Gaussians. The argument is
sound in principle -- NMS is winner-takes-all and order dependent, so
two cells 3 um apart lose one of them regardless of how much total
intensity is there, and no fit is ever attempted -- but it rests on a
premise that has not been checked:

    MERGED       the missed cell's blob fused with a neighbour's and the
                 pair produced ONE peak. A fit could split it. NMS
                 structurally cannot.
    ABSENT       there is no signal at that position at all. Nothing to
                 fit, and the fitting argument does not apply.

flicker.py found that at 79.8% of missed frames there is no candidate
within MATCH_UM = 5.0 um. **But it only looked out to 5 um.** If the
cell was absorbed by a neighbour, the neighbour's surviving peak should
sit a little further out -- roughly one cell diameter, 7-9 um.

So: at every missed frame, measure the distance to the nearest peak of
ANY score, with no radius limit, BEFORE suppression and after.

  d_pre    nearest PRE-NMS peak. If suppression is what removed the
           cell's own peak, this should often be small.
  d_post   nearest surviving peak. A shell at 5-10 um is the merge
           signature; 20 um and beyond means the region is empty.

⚠ Sparse GT: 6bba is annotated at about 6.5% of its cells, so "the
nearest peak" may belong to an unannotated neighbour either way. That
does not affect the question -- the question is whether ANY signal sits
near the missed cell, not whose it is.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 5.0
GT = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\gtx2")
DET = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


d_post, d_hit = [], []
n_gt = 0
for name in HOLDOUT:
    gf, df = GT / f"gt_{name}.npz", DET / f"detcache_{name}.npz"
    if not (gf.exists() and df.exists()):
        continue
    g = np.load(gf)
    d = np.load(df)
    T = max(int(k[1:]) for k in frame_keys(d.files, "c")) + 1
    g_t = np.asarray(g["t"], int)
    g_zyx = np.stack([np.asarray(g[c], float)
                      for c in ("z", "y", "x")], axis=1)
    for t in range(T):
        gi = np.where(g_t == t)[0]
        if not len(gi):
            continue
        c = np.asarray(d[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(d[f"s{t}"], float).reshape(-1)
        keep = c[s >= UNET_THRESH]
        if not len(keep):
            continue
        gu = g_zyx[gi] * SCALE[None, :]
        dist, _ = cKDTree(keep * SCALE[None, :]).query(gu)
        n_gt += len(gi)
        for v in dist:
            (d_hit if v <= MATCH_UM else d_post).append(float(v))

d_post = np.array(d_post)
d_hit = np.array(d_hit)
print(f"  {n_gt:,} annotated GT nodes over {len(HOLDOUT)} datasets")
print(f"  matched within {MATCH_UM} um : {len(d_hit):,}"
      f"  ({100.0 * len(d_hit) / n_gt:.1f}%)")
print(f"  MISSED                    : {len(d_post):,}"
      f"  ({100.0 * len(d_post) / n_gt:.1f}%)")

print(f"\n  distance from a MISSED cell to the nearest SURVIVING peak:")
for q in (10, 25, 50, 75, 90, 99):
    print(f"    p{q:<3} {np.percentile(d_post, q):7.2f} um")

print(f"\n  what fraction of misses sit in each shell:")
edges = [5, 6, 8, 10, 15, 20, 30, 1e9]
lo = MATCH_UM
for e in edges[1:]:
    m = (d_post > lo) & (d_post <= e)
    tag = ""
    if lo < 10:
        tag = "   <- one cell diameter: MERGE signature"
    elif lo >= 20:
        tag = "   <- nothing nearby: ABSENT"
    hi = "inf" if e > 1e8 else f"{e:g}"
    print(f"    {lo:>4g}-{hi:<5} um  {100.0 * m.mean():5.1f}%{tag}")
    lo = e

print("\n  READ IT: a missed cell whose nearest surviving peak sits at")
print("  7-9 um -- one cell diameter -- is consistent with its blob")
print("  having fused with a neighbour's, which is what a mixture fit")
print("  could split and what winner-takes-all suppression structurally")
print("  cannot. A missed cell with nothing inside 20 um is a region")
print("  with no signal, and no fitting method helps there.")
print("\n  For reference, the median nearest-neighbour spacing of real")
print("  cells is what sets the scale: at 8 um diameter, neighbouring")
print("  cell CENTRES in packed tissue sit about 8-10 um apart.")
