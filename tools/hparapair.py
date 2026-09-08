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
import re
import sys
import csv
import math

import numpy as np

A_PATH = sys.argv[1] if len(sys.argv) > 2 else "hparabench_F4.csv"
B_PATH = sys.argv[2] if len(sys.argv) > 2 else "hparabench_F4HP.csv"
NDRAW = 2000
SEED = 8080


_NPWRAP = re.compile(r"^np\.\w+\(([^)]*)\)$")


def _num(v):
    """Parse one cell, tolerating numpy's repr, and REFUSE to guess.

    The bench dumps with `repr(value)`. Under numpy 2.x that is
    `np.float64(0.91)`, not `0.91`, so a naive `float(v)` raises -- and the
    obvious except-clause turns the whole column into NaN while every table
    below still prints. That is the silent-wrong-number failure this project
    has a rule about, so anything unparseable raises here instead.
    """
    if v is None:
        raise ValueError("missing cell")
    s = v.strip()
    m = _NPWRAP.match(s)
    if m:
        s = m.group(1)
    if s in ("nan", "np.float64(nan)", ""):
        return float("nan")
    return float(s)


def load(p):
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                if k in ("dataset", "fold"):
                    continue
                try:
                    d[k] = _num(v)
                except (TypeError, ValueError) as e:
                    raise SystemExit(
                        f"{p}: cannot parse column {k!r} of "
                        f"{r['dataset']}: {v!r} ({e}). Fix the parser -- "
                        f"turning this into NaN would print a full set of "
                        f"plausible tables built on a dropped column.")
            d["_fold"] = int(r["fold"])
            rows[r["dataset"]] = d
    return rows


def agg(rows):
    """`tracking_cellmot.metrics.summarise`, reimplemented from its source.

    TWO DIFFERENT aggregations, and mixing them up costs 0.0007 on the
    anchor -- which is how the first draft of this file was caught:

      * `edge_jaccard` and `division_jaccard` are MICRO-averaged: sum the
        TP/FP/FN across datasets, divide once. One large movie counts more
        than one small one.
      * `adj_edge_jaccard` is NOT `edge_J * (1 - 0.1 * node_ratio)` on the
        aggregate. The node penalty is applied PER SAMPLE, and the
        per-sample adjusted values are weight-averaged by
        `w_i = tp_i + fp_i + fn_i`. Recomputing it from an edge-weighted
        mean node ratio reads 0.9126 where divbench9 records 0.9133 --
        close enough to look right, wrong enough to swamp a delta of the
        size this comparison exists to measure.

    `score = adj_edge_jaccard + 0.1 * division_jaccard`. NaN rows drop out
    of the adjusted average only, exactly as summarise does.
    """
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r.get("division_tp", 0.0) for r in rows)
    dfp = sum(r.get("division_fp", 0.0) for r in rows)
    dfn = sum(r.get("division_fn", 0.0) for r in rows)
    ej = tp / max(1.0, tp + fp + fn)
    dj = dtp / max(1.0, dtp + dfp + dfn)
    adj_rows = [r for r in rows
                if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    tw = sum(w)
    adj = (sum(wi * r["adj_edge_jaccard"] for wi, r in zip(w, adj_rows)) / tw
           if tw > 0 else float("nan"))
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
_all = agg(list(A.values()))
print(f"  F4 over its own {len(A)} rows: edge_J {_all['edge_jaccard']:.4f}  "
      f"adj_J {_all['adj']:.4f}  div_J {_all['div']:.4f}  "
      f"score {_all['score']:.4f}")
print( "  divbench9 REF over the 128:   edge_J 0.9104  adj_J 0.9133  "
       "div_J 0.1337  score 0.9266")
_want = dict(edge_jaccard=0.9104, adj=0.9133, div=0.1337, score=0.9266)
if len(A) == 128:
    _off = {k: round(_all[k] - v, 4) for k, v in _want.items()
            if abs(_all[k] - v) >= 5e-5}
    print("  ANCHOR EXACT" if not _off else
          f"  !! ANCHOR DRIFT {_off} -- the harness moved, so nothing below "
          f"is about hpara")
else:
    print(f"  (control has {len(A)} rows, not 128 -- anchor not checkable)")
aF4 = agg([A[n] for n in names if n.startswith("6bba")])
print(f"  paired 6bba subset adj_J {aF4['adj']:.4f}")

# WHICH MOVIES ACTUALLY CHANGED. `hparabench` scores all 128 6bba movies
# (SIX = every 6bba_ in ALL_TRAIN) but the hpara caches cover only the 24
# EVALUATION movies, of which just 12 are 6bba -- the other 12 are 44b6 and
# never enter this CSV. So ~116 of 128 rows are identical between the arms
# BY CONSTRUCTION. They do not bias the paired delta, but they dilute the
# aggregate by roughly 10x and they are not what the question is about, so
# everything below is reported BOTH ways.
_cols = [k for k in A[names[0]] if k != "_fold"]


def _same(a, b):
    for k in _cols:
        x, y = a[k], b[k]
        if x != x and y != y:          # both NaN: not a difference
            continue
        if x != y:
            return False
    return True


CHANGED = [n for n in names if not _same(A[n], B[n])]
print(f"\n  movies whose rows DIFFER between the arms: {len(CHANGED)} of "
      f"{len(names)}" + (f"\n    {CHANGED}" if 0 < len(CHANGED) <= 14 else ""))
if not CHANGED:
    raise SystemExit(
        "the two arms are identical on every paired movie -- the F4HP run "
        "did not pick up the hpara caches. A table of zeros would read as a "
        "null RESULT when what happened is a null RUN.")

print("\n=== 2. per-movie, F4HP minus F4 (only the rows that differ) ===")
print(f"  {'movie':<16} {'fold':>4} {'F4 score':>9} {'F4HP':>9} {'delta':>9}")
d_one = {}
for n in names:
    sa = agg([A[n]])["score"]
    sb = agg([B[n]])["score"]
    d_one[n] = sb - sa
    if n in CHANGED:
        print(f"  {n:<16} {A[n]['_fold']:>4} {sa:>9.4f} {sb:>9.4f} "
              f"{sb - sa:>+9.4f}")
up = sum(1 for n in CHANGED if d_one[n] > 0)
dn = sum(1 for n in CHANGED if d_one[n] < 0)
print(f"  up {up}, down {dn}, flat {len(CHANGED) - up - dn} of the "
      f"{len(CHANGED)} that changed"
      f"   (sign test p = {2 * sum(math.comb(up + dn, k) for k in range(min(up, dn) + 1)) / 2 ** (up + dn):.4f})"
      if up + dn else "  every changed movie is flat on SCORE")

print("\n=== 3. aggregates ===")
print(f"  -- all {len(names)} paired movies: the PIPELINE-level number, what "
      f"swapping hpara in where it is available does to the 6bba bench --")
for lab, R in (("F4  (refine)", A), ("F4HP (hpara)", B)):
    a = agg([R[n] for n in names])
    print(f"  {lab:<14} edge_J {a['edge_jaccard']:.4f}  adj_J {a['adj']:.4f}  "
          f"div_J {a['div']:.4f}  score {a['score']:.4f}")
dA, dB = agg([A[n] for n in names]), agg([B[n] for n in names])
print(f"  {'delta':<14} edge_J {dB['edge_jaccard'] - dA['edge_jaccard']:+.4f}  "
      f"adj_J {dB['adj'] - dA['adj']:+.4f}  "
      f"div_J {dB['div'] - dA['div']:+.4f}  "
      f"score {dB['score'] - dA['score']:+.4f}")


print(f"\n  -- the {len(CHANGED)} movies hpara actually touched: does the "
      f"placement help WHERE APPLIED --")
for lab, R in (("F4  (refine)", A), ("F4HP (hpara)", B)):
    a = agg([R[n] for n in CHANGED])
    print(f"  {lab:<14} edge_J {a['edge_jaccard']:.4f}  adj_J {a['adj']:.4f}  "
          f"div_J {a['div']:.4f}  score {a['score']:.4f}")
_cA, _cB = agg([A[n] for n in CHANGED]), agg([B[n] for n in CHANGED])
print(f"  {'delta':<14} edge_J {_cB['edge_jaccard'] - _cA['edge_jaccard']:+.4f}  "
      f"adj_J {_cB['adj'] - _cA['adj']:+.4f}  "
      f"div_J {_cB['div'] - _cA['div']:+.4f}  "
      f"score {_cB['score'] - _cA['score']:+.4f}")
print(f"\n  NOTE: the 44b6 half of the hpara caches (12 movies) is NOT in this\n"
      f"  CSV -- `SIX` is 6bba only. Read 44b6 off the run's own aggregate in\n"
      f"  the kernel log; there is no per-movie pairing for it here.")
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
groups = [("all paired", names),
          ("CHANGED only", CHANGED),
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
