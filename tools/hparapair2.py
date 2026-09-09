"""hparapair2 -- the paired interval hparabench does not print itself.

Both arms bootstrap WITHIN their own cross, but neither compares ACROSS the
arms, and that is the comparison the whole hpara line is about. This pairs
`hparabench_F4.csv` against `hparabench_F4HP.csv` movie by movie -- same 128
movies, same folds, same order -- and resamples the movies once per draw for
both arms, so the movie draw cancels and what is left is the placement.

THE AGGREGATION IS THE SCORER'S, NOT A MEAN. `summarise` in
`tracking_cellmot.metrics` computes `edge_jaccard` and `division_jaccard`
MICRO-averaged (sum TP/FP/FN over movies, divide once) but
`adj_edge_jaccard` as the PER-SAMPLE adjusted value weight-averaged by
w_i = tp_i + fp_i + fn_i. Reproducing that exactly is what
`tools/hparapair.py` got wrong on 2026-09-08 -- it read 0.9126 where the
record says 0.9133, four times the effect being measured -- so the shipped
aggregate is asserted here before anything is compared.

ANCHORS, both from the runs' own logs:
  F4    edge_J 0.9104  adj_J 0.9133  div_J 0.1337  score 0.9266  (divbench3)
  F4HP  edge_J 0.9118  adj_J 0.9146  div_J 0.1519  score 0.9298

⚠ THESE ARE THE UNGATED ROWS. What ships is the joint fork gate on top, and
its cells live only in the logs (F4 46/39/79 score 0.9424, F4HP 49/51/76
score 0.9433). The CSV carries the ungated per-movie rows, so the interval
below is on the UNGATED comparison. Say so rather than letting it be read as
the shipped delta.
"""
import csv
import sys

import numpy as np

F4 = sys.argv[1] if len(sys.argv) > 1 else "hparabench_F4.csv"
HP = sys.argv[2] if len(sys.argv) > 2 else "hparabench_F4HP.csv"
B = 4000

ANCHOR = {"F4": dict(edge_jaccard=0.9104, adj_edge_jaccard=0.9133,
                     division_jaccard=0.1337, score=0.9266),
          "F4HP": dict(edge_jaccard=0.9118, adj_edge_jaccard=0.9146,
                       division_jaccard=0.1519, score=0.9298)}
TOL = 0.0001


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = {}
        for k, v in r.items():
            if k in ("dataset", "fold"):
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = float("nan")
        out[r["dataset"]] = d
    return out


A, H = load(F4), load(HP)
NAMES = sorted(set(A) & set(H))
print(f"{len(A)} F4 rows, {len(H)} F4HP rows, {len(NAMES)} paired")
assert len(NAMES) == 128, len(NAMES)


def agg(rows):
    """summarise()'s aggregation, verbatim.  edge_J and div_J micro-averaged;
    adj_J weight-averaged per sample by tp+fp+fn."""
    etp = sum(r["edge_tp"] for r in rows)
    efp = sum(r["edge_fp"] for r in rows)
    efn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    ej = etp / max(1e-9, etp + efp + efn)
    dj = dtp / max(1e-9, dtp + dfp + dfn)
    ok = [r for r in rows if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in ok]
    tw = sum(w)
    aj = (sum(wi * r["adj_edge_jaccard"] for wi, r in zip(w, ok)) / tw
          if tw else float("nan"))
    return dict(edge_jaccard=ej, division_jaccard=dj, adj_edge_jaccard=aj,
                score=aj + 0.1 * dj)


KEYS = ["edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"]
print("\n=== ANCHOR: the aggregate must reproduce each run's own log ===")
fail = []
for tag, D in (("F4", A), ("F4HP", H)):
    g = agg([D[n] for n in NAMES])
    for k, want in ANCHOR[tag].items():
        ok = abs(g[k] - want) <= TOL
        print(f"  {tag:<5} {k:<20} got {g[k]:.4f}  want {want:.4f}  "
              f"{'OK' if ok else 'MISS'}")
        if not ok:
            fail.append((tag, k, g[k], want))
assert not fail, f"aggregation does not reproduce the runs: {fail}"
print("  the aggregation is the scorer's; the deltas below are readable")

gA, gH = agg([A[n] for n in NAMES]), agg([H[n] for n in NAMES])
print(f"\n=== UNGATED aggregate, 128 movies, DIV_W 0.55 ===")
print(f"  {'':<8}" + "".join(f"{k:>22}" for k in KEYS))
print(f"  {'F4':<8}" + "".join(f"{gA[k]:>22.4f}" for k in KEYS))
print(f"  {'F4HP':<8}" + "".join(f"{gH[k]:>22.4f}" for k in KEYS))
print(f"  {'delta':<8}" + "".join(f"{gH[k] - gA[k]:>+22.4f}" for k in KEYS))

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
for k in ("adj_edge_jaccard", "edge_jaccard"):
    d = np.array([H[n][k] - A[n][k] for n in NAMES])
    d = d[d == d]
    up = int((d > 0).sum())
    print(f"  {k:<22} {up}/{len(d)} movies up, "
          f"median {np.median(d):+.5f}, "
          f"p10 {np.percentile(d, 10):+.5f}, p90 {np.percentile(d, 90):+.5f}")
d = np.array([H[n]["adj_edge_jaccard"] - A[n]["adj_edge_jaccard"]
              for n in NAMES])
w = np.array([A[n]["edge_tp"] + A[n]["edge_fp"] + A[n]["edge_fn"]
              for n in NAMES])
o = np.argsort(-np.abs(d * w))
print(f"\n  the 6 movies carrying the most WEIGHTED adj_J movement "
      f"(|d| x its scorer weight):")
print(f"  {'movie':<18}{'weight':>9}{'d adj_J':>10}{'contribution':>14}")
for i in o[:6]:
    print(f"  {NAMES[i]:<18}{w[i]:>9.0f}{d[i]:>+10.5f}"
          f"{d[i] * w[i] / w.sum():>+14.6f}")
print(f"  top 6 account for "
      f"{abs(d[o[:6]] * w[o[:6]]).sum() / max(1e-9, abs(d * w).sum()) * 100:.0f}"
      f"% of the total absolute movement, over {len(NAMES)} movies.")
