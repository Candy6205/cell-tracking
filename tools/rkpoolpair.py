"""rkpoolpair -- pair rkpool's two arms over the 128 6bba movies.

`rkpool_ref` (RANK_TRAIN=BASE, what ships) against `rkpool` (RANK_TRAIN=F4,
the ranker retrained on the coordinates it is deployed on). Same 128 movies,
same folds, same order, same stage B; the ONLY difference is which table the
fold rankers learned from.

WHY THIS AND NOT eprank3's 12. eprank3 measured the same question on the
12-movie holdout and read -0.0011 / -0.0022 with both intervals crossing
zero. The day before, hparabench round 2 established that a 12-movie holdout
can disagree in SIGN with the 128 that contain it (adj_J -0.0012 on the 12
against +0.0013 on the 128). So the 12 could not settle it. This can.

AND AT THE RIGHT FLOOR. eprank3 inherited `ILP_FLOOR = 0.10` and has no
`PROD_FLOOR` at all, so its verdict was measured at an operating point that
does not ship. rkpool's stage B is the production solve: floor 0.48,
disappearance 1.5, DIV_W 0.55.

THE AGGREGATION IS THE SCORER'S, NOT A MEAN -- `summarise` micro-averages
edge_J and div_J and weight-averages adj_J per sample by w_i = tp+fp+fn.
`tools/hparapair.py` got that wrong on 2026-09-08 and read four times the
effect being measured, so the control's aggregate is ASSERTED here against
its own log before anything is compared.

ANCHOR (rkpool_ref's log, line 379, and divbench3 before it):
    REF over the 128:  div 50/249/75  div_J 0.1337  adj_J 0.9133
                       edge_J 0.9104  score 0.9266

The F4 arm has no anchor -- it is the new thing. That asymmetry is the point:
if the control reproduces the record, the delta is the retrain.

    py tools/rkpoolpair.py bench_csv/rkpool_BASE.csv bench_csv/rkpool_F4.csv
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import benchagg as BA  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "bench_csv/rkpool_BASE.csv"
F4 = sys.argv[2] if len(sys.argv) > 2 else "bench_csv/rkpool_F4.csv"
B = 4000

ANCHOR = dict(edge_jaccard=0.9104, adj_edge_jaccard=0.9133,
              division_jaccard=0.1337, score=0.9266)
TOL = 0.0001


A, H = BA.load(BASE), BA.load(F4)
NAMES = sorted(set(A) & set(H))
print(f"{len(A)} BASE rows, {len(H)} F4 rows, {len(NAMES)} paired")
assert len(NAMES) == 128, len(NAMES)


agg = BA.agg
KEYS = BA.KEYS

print("\n=== ANCHOR: the CONTROL must reproduce its own log and divbench3 ===")
gA = agg([A[n] for n in NAMES])
fail = []
for k, want in ANCHOR.items():
    ok = abs(gA[k] - want) <= TOL
    print(f"  BASE  {k:<20} got {gA[k]:.4f}  want {want:.4f}  "
          f"{'OK' if ok else 'MISS'}")
    if not ok:
        fail.append((k, gA[k], want))
assert not fail, f"the control does not reproduce the record: {fail}"
dtp = sum(A[n]["division_tp"] for n in NAMES)
dfp = sum(A[n]["division_fp"] for n in NAMES)
dfn = sum(A[n]["division_fn"] for n in NAMES)
assert (dtp, dfp, dfn) == (50, 249, 75), (dtp, dfp, dfn)
print("  division cells 50/249/75 OK -- the aggregation is the scorer's")

gH = agg([H[n] for n in NAMES])
print(f"\n=== UNGATED aggregate, 128 movies, DIV_W 0.55, floor 0.48 ===")
print(f"  {'':<10}" + "".join(f"{k:>22}" for k in KEYS))
print(f"  {'BASE':<10}" + "".join(f"{gA[k]:>22.4f}" for k in KEYS))
print(f"  {'F4':<10}" + "".join(f"{gH[k]:>22.4f}" for k in KEYS))
print(f"  {'delta':<10}" + "".join(f"{gH[k] - gA[k]:>+22.4f}" for k in KEYS))
print(f"  divisions  BASE {int(dtp)}/{int(dfp)}/{int(dfn)}   F4 "
      f"{int(sum(H[n]['division_tp'] for n in NAMES))}/"
      f"{int(sum(H[n]['division_fp'] for n in NAMES))}/"
      f"{int(sum(H[n]['division_fn'] for n in NAMES))}")

print(f"\n=== PAIRED BOOTSTRAP over the 128 movies, {B} draws, the SAME "
      f"resample for both arms ===")
rng = np.random.default_rng(0)
D = {k: [] for k in KEYS}
for _ in range(B):
    idx = rng.integers(0, len(NAMES), len(NAMES))
    nm = [NAMES[i] for i in idx]
    a, h = agg([A[n] for n in nm]), agg([H[n] for n in nm])
    for k in KEYS:
        D[k].append(h[k] - a[k])
print(f"  {'metric':<22}{'mean d':>10}{'90% interval':>26}{'P(>0)':>8}")
for k in KEYS:
    d = np.asarray(D[k])
    lo, hi = np.percentile(d, [5, 95])
    print(f"  {k:<22}{np.mean(d):>+10.5f}"
          f"{f'[{lo:+.5f}, {hi:+.5f}]':>26}{np.mean(d > 0):>8.3f}")

print(f"\n=== PER-MOVIE SIGN, because an aggregate cannot show "
      f"concentration ===")
for k in ("adj_edge_jaccard", "edge_jaccard", "node_recall",
          "num_pred_nodes"):
    d = np.array([H[n][k] - A[n][k] for n in NAMES])
    d = d[d == d]
    if not len(d):
        continue
    print(f"  {k:<22} {int((d > 0).sum())}/{len(d)} movies up, "
          f"{int((d == 0).sum())} flat, median {np.median(d):+.5f}, "
          f"p10 {np.percentile(d, 10):+.5f}, p90 {np.percentile(d, 90):+.5f}")

d = np.array([H[n]["adj_edge_jaccard"] - A[n]["adj_edge_jaccard"]
              for n in NAMES])
w = np.array([A[n]["edge_tp"] + A[n]["edge_fp"] + A[n]["edge_fn"]
              for n in NAMES])
o = np.argsort(-np.abs(d * w))
print(f"\n  the 8 movies carrying the most WEIGHTED adj_J movement "
      f"(|d| x its scorer weight):")
print(f"  {'movie':<18}{'weight':>9}{'d adj_J':>10}{'contribution':>14}")
for i in o[:8]:
    print(f"  {NAMES[i]:<18}{w[i]:>9.0f}{d[i]:>+10.5f}"
          f"{d[i] * w[i] / w.sum():>+14.6f}")
print(f"  top 8 account for "
      f"{abs(d[o[:8]] * w[o[:8]]).sum() / max(1e-9, abs(d * w).sum()) * 100:.0f}"
      f"% of the total absolute movement, over {len(NAMES)} movies.")
print(f"  leave-one-out: the largest single mover flips the sign of the "
      f"weighted delta? "
      f"{'YES' if np.sign(np.sum(np.delete(d * w, o[0])) ) != np.sign(np.sum(d * w)) else 'no'}")

print("""
=== THE PRE-REGISTERED READING (progress.md, "Pre-registered reading") ===
  1 F4 wins, interval excludes 0 -> the mismatch costs score; retrain the
    production pkl, then re-check PROD_FLOOR / ILP_DISAPP / DIV_W / the gate,
    all of which were tuned against the mismatched calibration.
  2 flat -> HistGB absorbs the shift; the TTA retrain is unlikely to rescue
    TTA either, and the peer-blend prior weakens with it.
  3 BASE wins -> training on slightly noisier coordinates regularises.
  4 (added after eprank3, because 1-3 were not exhaustive) score moves while
    the held-out AUC does not -> the ranking is unchanged and the damage is
    downstream, at constants tuned for the other calibration. Read the AUC
    line from each arm's log before choosing between 1-3 and 4.""")
