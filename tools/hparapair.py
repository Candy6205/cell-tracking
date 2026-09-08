"""Pair hparabench's two runs and bootstrap the per-movie score difference.

    py tools/hparapair.py <hparabench_F4.csv> <hparabench_F4HP.csv>

WHY OFFLINE. The two arms are two runs of ONE script (`hparabench_ref.py` with
DET_SRC=F4, `hparabench.py` with DET_SRC=F4HP), each dumping one row per
movie. Pairing them here rather than growing divbench9 a third det-arm
dimension keeps the diff at ~60 lines and makes the control a genuine control:
same code path, same order, same folds, one input map different.

WHAT IT PRINTS

  1. the anchor: the F4 run's aggregate must match divbench9's recorded
     numbers, or the harness moved and nothing below is about hpara;
  2. per-movie delta, so a single movie carrying the whole effect is visible
     -- the failure this comparison is most exposed to, with 24 movies;
  3. the aggregate score of each arm, recomputed the way the bench does it
     (edge-weighted, `_with_ratio`), and their difference;
  4. a PAIRED bootstrap over movies, 2,000 draws, the SAME resample applied
     to both arms so the movie draw cancels -- divbench8's block, which is
     what every recent verdict on this project has used;
  5. the same split by domain and by fold, because a gain that is one-sided
     on either is a different claim from a gain that holds on both.

READING IT. The pre-registered rule, written before the numbers exist:
hpara adds no nodes, so the penalty term cannot move and the whole delta is
edges and divisions. A 90% interval excluding zero on the score, with both
domains the same sign, is a result. An interval spanning zero is NOT a
refutation of `refinecmp` -- refinecmp measured placement, this measures
whether the ranker and the ILP convert it, and the ranker pkl was fitted on
refine coordinates (see make_hparabench.py). Negative or null therefore
leaves the retrain-on-hpara question open at a cost of four GPU hours.
"""
import sys
import csv
import math

import numpy as np

A_PATH = sys.argv[1] if len(sys.argv) > 2 else "hparabench_F4.csv"
B_PATH = sys.argv[2] if len(sys.argv) > 2 else "hparabench_F4HP.csv"
NDRAW = 2000
SEED = 8080


def load(p):
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                if k in ("dataset", "fold"):
                    continue
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = float("nan")
            d["_fold"] = int(r["fold"])
            rows[r["dataset"]] = d
    return rows


def agg(rows):
    """The bench's own weighting: edge counts, not a mean of per-movie J.

    `_with_ratio` sums the edge TP/FP/FN across datasets and divides once,
    so one large movie counts more than one small one. A plain mean of the
    per-movie scores answers a different question and would not compare to
    any recorded number.
    """
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r.get("division_tp", 0.0) for r in rows)
    dfp = sum(r.get("division_fp", 0.0) for r in rows)
    dfn = sum(r.get("division_fn", 0.0) for r in rows)
    ej = tp / max(1.0, tp + fp + fn)
    dj = dtp / max(1.0, dtp + dfp + dfn)
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    tw = sum(w)
    good = [(r, wi) for r, wi in zip(rows, w)
            if r.get("total_node_ratio") == r.get("total_node_ratio")]
    ratio = (sum(r["total_node_ratio"] * wi for r, wi in good) / tw
             if tw and good else float("nan"))
    adj = ej * max(0.0, 1.0 - 0.1 * ratio) if ratio == ratio else ej
    return dict(edge_jaccard=ej, adj=adj, div=dj, score=adj + 0.1 * dj)


A, B = load(A_PATH), load(B_PATH)
names = sorted(set(A) & set(B))
print(f"paired on {len(names)} movies "
      f"({len(A)} in the F4 run, {len(B)} in the F4HP run)")
only = sorted((set(A) | set(B)) - set(A & B.keys()))
if only:
    print(f"  ! not in both, dropped: {only}")
assert len(names) >= 20, f"only {len(names)} paired movies -- refusing"
for n in names:
    assert A[n]["_fold"] == B[n]["_fold"], f"{n}: folds differ between runs"

print("\n=== 1. the F4 arm against divbench9's recorded aggregate ===")
aF4 = agg([A[n] for n in names if n.startswith("6bba")])
print(f"  F4-6bba adj_J {aF4['adj']:.4f}   (divbench9 +BIDIR F4-6bba 0.9261;"
      f" a drift here means the harness moved, not hpara)")

print("\n=== 2. per-movie, F4HP minus F4 ===")
print(f"  {'movie':<16} {'fold':>4} {'F4 score':>9} {'F4HP':>9} {'delta':>9}")
d_one = {}
for n in names:
    sa = agg([A[n]])["score"]
    sb = agg([B[n]])["score"]
    d_one[n] = sb - sa
    print(f"  {n:<16} {A[n]['_fold']:>4} {sa:>9.4f} {sb:>9.4f} "
          f"{sb - sa:>+9.4f}")
up = sum(1 for v in d_one.values() if v > 0)
dn = sum(1 for v in d_one.values() if v < 0)
print(f"  up {up}, down {dn}, flat {len(names) - up - dn}"
      f"   (sign test p = {2 * sum(math.comb(up + dn, k) for k in range(min(up, dn) + 1)) / 2 ** (up + dn):.4f})"
      if up + dn else "  all flat")

print("\n=== 3. aggregates ===")
for lab, R in (("F4  (refine)", A), ("F4HP (hpara)", B)):
    a = agg([R[n] for n in names])
    print(f"  {lab:<14} edge_J {a['edge_jaccard']:.4f}  adj_J {a['adj']:.4f}  "
          f"div_J {a['div']:.4f}  score {a['score']:.4f}")
dA, dB = agg([A[n] for n in names]), agg([B[n] for n in names])
print(f"  {'delta':<14} edge_J {dB['edge_jaccard'] - dA['edge_jaccard']:+.4f}  "
      f"adj_J {dB['adj'] - dA['adj']:+.4f}  "
      f"div_J {dB['div'] - dA['div']:+.4f}  "
      f"score {dB['score'] - dA['score']:+.4f}")

print(f"\n=== 4. PAIRED bootstrap over movies, {NDRAW} draws, same resample "
      f"for both arms ===")
rng = np.random.default_rng(SEED)
idx = np.arange(len(names))


def boot(sub):
    sub = list(sub)
    if not sub:
        return None
    draws = [rng.choice(len(sub), size=len(sub), replace=True)
             for _ in range(NDRAW)]
    da = np.array([agg([A[sub[i]] for i in d])["score"] for d in draws])
    db = np.array([agg([B[sub[i]] for i in d])["score"] for d in draws])
    return db - da


print(f"  {'group':<18} {'n':>3} {'mean d score':>13} {'90% interval':>22}  "
      f"excludes 0")
groups = [("all", names),
          ("6bba", [n for n in names if n.startswith("6bba")]),
          ("44b6", [n for n in names if n.startswith("44b6")]),
          ("fold 0", [n for n in names if A[n]["_fold"] == 0]),
          ("fold 1", [n for n in names if A[n]["_fold"] == 1])]
for lab, sub in groups:
    if len(sub) < 3:
        continue
    rng = np.random.default_rng(SEED)     # same draws per group, reproducible
    d = boot(sub)
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"  {lab:<18} {len(sub):>3} {float(np.mean(d)):>+13.4f}   "
          f"[{lo:>+8.4f}, {hi:>+8.4f}]  {'YES' if (lo > 0 or hi < 0) else 'no'}")

print("\n  Read it against the rule fixed before the numbers: hpara adds no\n"
      "  nodes, so the penalty cannot move and the delta is edges and\n"
      "  divisions. An interval spanning zero does NOT refute refinecmp --\n"
      "  that measured PLACEMENT; this measures whether the ranker and the\n"
      "  ILP convert it, with a ranker pkl fitted on refine coordinates.")
