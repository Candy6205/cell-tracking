"""ttapair -- what edge-feature TTA does to the SCORE, on the 12 movies that
carry it, with an interval.

The user: *"只看12部的分数会死？"* and, when I answered that it would probably
come back "cannot tell": *"你都知道测不出来了，还在这里诚实的测不出来"*. Both
fair. The ±0.005 I was quoting came from hparabench round 1, where the arms
changed the COORDINATES -- detections moved, the gate re-formed, the
candidate pairs changed. Here the detections are byte-identical, the
(t, i, j) rows are identical, and the only difference is the value in
`p_fwd` / `p_rev`. Borrowing an interval from a much looser comparison is not
measuring this one.

WHAT IS PAIRED. `hparabench_F4.csv` (the control, DET_SRC=F4, 128 movies)
against `ttabench_tta4.csv` (the same run with `edgeprob_F4_` replaced by
`ttaprobe`'s `edgeprob_tta4_` on the 12 holdout movies). 116 movies are
byte-identical between them by construction, so:

  * the 128-movie aggregate is the DILUTED number, reported because it is
    what a careless read would take;
  * the 12-movie CHANGED subset is the measurement, and it gets the
    bootstrap.

THE RUN'S OWN ANCHOR, already passed: ttaprobe's noTTA arm reproduced
`edgeprob_F4_` on 12 of 12 movies with worst |dp| = 0.000e+00, so the
wrapper is transparent at view 1 and the control is legitimate.

⚠ This is the FIRST edge head only; `edgeprob2_F4_` stays single-view.

Usage:  py tools/ttapair.py hparabench_F4.csv ttabench_tta4.csv
"""
import csv
import sys

import numpy as np

CTRL = sys.argv[1] if len(sys.argv) > 1 else "hparabench_F4.csv"
TTA = sys.argv[2] if len(sys.argv) > 2 else "ttabench_tta4.csv"
B = 4000
KEYS = ["edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"]

# ttaprobe's 12, from its own job list.
HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]


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


A, H = load(CTRL), load(TTA)
NAMES = sorted(set(A) & set(H))
print(f"{len(A)} control rows, {len(H)} tta4 rows, {len(NAMES)} paired")


def agg(rows):
    """summarise()'s aggregation: edge_J and div_J micro-averaged, adj_J
    weight-averaged per sample by tp+fp+fn."""
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


# ---- WHICH movies actually differ, checked rather than assumed ----------
diff = [n for n in NAMES
        if any(abs(A[n][k] - H[n][k]) > 1e-12 for k in
               ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                "division_fp", "division_fn", "adj_edge_jaccard"))]
print(f"\nmovies whose rows actually differ: {len(diff)}")
extra = sorted(set(diff) - set(HOLDOUT_6BBA))
missing = sorted(set(HOLDOUT_6BBA) - set(diff))
if extra:
    print(f"  ⚠ {len(extra)} outside the 12 that carry a tta4 cache: "
          f"{extra[:5]} -- the arms differ somewhere they should not")
if missing:
    print(f"  {len(missing)} of the 12 came out identical: {missing}")
CHANGED = [n for n in NAMES if n in set(diff)]
assert CHANGED, "nothing changed; the tta4 caches were not used"

for label, subset in (("ALL " + str(len(NAMES)) + " (DILUTED)", NAMES),
                      ("CHANGED " + str(len(CHANGED)), CHANGED)):
    a, h = agg([A[n] for n in subset]), agg([H[n] for n in subset])
    print(f"\n=== {label} ===")
    print(f"  {'':<10}" + "".join(f"{k:>22}" for k in KEYS))
    print(f"  {'noTTA':<10}" + "".join(f"{a[k]:>22.4f}" for k in KEYS))
    print(f"  {'tta4':<10}" + "".join(f"{h[k]:>22.4f}" for k in KEYS))
    print(f"  {'delta':<10}" + "".join(f"{h[k] - a[k]:>+22.4f}" for k in KEYS))

print(f"\n=== PAIRED BOOTSTRAP over the {len(CHANGED)} CHANGED movies, "
      f"{B} draws, the same resample for both arms ===")
rng = np.random.default_rng(0)
D = {k: [] for k in KEYS}
for _ in range(B):
    idx = rng.integers(0, len(CHANGED), len(CHANGED))
    nm = [CHANGED[i] for i in idx]
    a, h = agg([A[n] for n in nm]), agg([H[n] for n in nm])
    for k in KEYS:
        D[k].append(h[k] - a[k])
print(f"  {'metric':<22}{'mean d':>10}{'90% interval':>26}{'P(>0)':>8}")
for k in KEYS:
    d = np.asarray(D[k])
    lo, hi = np.percentile(d, [5, 95])
    print(f"  {k:<22}{np.mean(d):>+10.5f}"
          f"{f'[{lo:+.5f}, {hi:+.5f}]':>26}{np.mean(d > 0):>8.3f}")

print(f"\n=== PER MOVIE ===")
print(f"  {'movie':<18}{'weight':>9}{'noTTA adj':>11}{'tta4 adj':>10}"
      f"{'d adj':>10}{'d edge':>10}")
up = 0
for n in CHANGED:
    da = H[n]["adj_edge_jaccard"] - A[n]["adj_edge_jaccard"]
    de = H[n]["edge_jaccard"] - A[n]["edge_jaccard"]
    w = A[n]["edge_tp"] + A[n]["edge_fp"] + A[n]["edge_fn"]
    up += int(da > 0)
    print(f"  {n:<18}{w:>9.0f}{A[n]['adj_edge_jaccard']:>11.4f}"
          f"{H[n]['adj_edge_jaccard']:>10.4f}{da:>+10.5f}{de:>+10.5f}")
print(f"  movies where tta4 beats noTTA on adj_J: {up}/{len(CHANGED)}")
