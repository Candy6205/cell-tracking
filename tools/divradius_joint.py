"""divradius.py with the JOINT gate -- the one that actually ships.

`divradius.py` trains on the 6bba counted forks alone (divbench3's
`REF + gate6`). What ships is divbench8's joint gate, which adds divbench5's
67 counted 44b6 forks to BOTH training folds, and it is a better classifier
(pooled out-of-fold AUC 0.9016 against 0.8798). The radius question has to be
answered against the gate that is actually in front of it.

ANCHOR, exact: 46 / 39 / 79, div_J 0.2805 -- divbench8's recorded
`REF + joint`, the configuration at LB 0.944 (ref 56086066).

The 44b6 rows are read out of `divbench9.py`'s own `ROWS44` literal, which is
the same literal divbench8 trained on; that is what makes this the same gate
rather than a re-derivation of it.

Usage:  py tools/divradius_joint.py path/to/forks_dw55.csv [path/to/divbench9.py]

⚠ The same approximation as `divradius.py`: this filters the proposals a run
at the shipped radii already made, rather than re-running the patch at a
smaller radius, so it does not model the caps freeing a slot. See that file.
"""
import csv
import pathlib
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
CUT = 0.03
FN0 = 75          # GT divisions with no fork proposed at all (divbench3)
ANCHOR = (46, 39, 79)

CSV = sys.argv[1] if len(sys.argv) > 1 else "forks_dw55.csv"
DB9 = sys.argv[2] if len(sys.argv) > 2 else "divbench9.py"


def f(s):
    try:
        return float(s)
    except Exception:
        return np.nan


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
X = np.array([[f(r[c]) for c in FEAT20] for r in rows], np.float64)
lab = np.array([int(r["label"]) for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
d_pq = np.array([f(r["d_pq"]) for r in rows])
d_s = np.array([f(r["d_s"]) for r in rows])
cnt = lab >= 1
print(f"{len(rows)} forks, {int(cnt.sum())} counted "
      f"({int((lab == 2).sum())} TP, {int((lab == 1).sum())} FP)")

# ---- the 44b6 half of the joint gate, out of divbench9's own literal ----
src = pathlib.Path(DB9).read_text(encoding="utf-8")
_i = src.index("ROWS44 = [")
_ns = {"np": np}
exec(src[_i:src.index("\n]", _i) + 2], _ns)
FEAT44 = eval(re.search(r"FEAT44 = (\[.*?\])\n", src, re.S).group(1))
_idx = [FEAT44.index(c) for c in FEAT20]
X44 = np.array([[b[i] for i in _idx] for _, b in _ns["ROWS44"]], np.float64)
y44 = np.array([1 if a == 2 else 0 for a, _ in _ns["ROWS44"]])
print(f"joint training adds {len(y44)} 44b6 counted forks "
      f"({int(y44.sum())} TP) to BOTH folds")

p = np.full(len(rows), np.nan)
for k in (0, 1):
    tr = (fold != k) & cnt
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                       min_samples_leaf=10, random_state=0)
    m.fit(np.vstack([X[tr], X44]),
          np.concatenate([(lab[tr] == 2).astype(int), y44]))
    p[fold == k] = m.predict_proba(X[fold == k])[:, 1]
print(f"joint gate: pooled out-of-fold AUC "
      f"{roc_auc_score((lab[cnt] == 2).astype(int), p[cnt]):.4f}")
gate = p > CUT


def sc(mask):
    tp = int(((lab == 2) & mask).sum())
    fp = int(((lab == 1) & mask).sum())
    fn = FN0 + int((lab == 2).sum()) - tp
    return tp, fp, fn, tp / max(1, tp + fp + fn)


B = sc(gate)
ok = (B[0], B[1], B[2]) == ANCHOR
print(f"\nANCHOR  parent 12.0 sister 15.0 (SHIPS): {B[0]}/{B[1]}/{B[2]}  "
      f"div_J {B[3]:.4f}   want {'/'.join(map(str, ANCHOR))} 0.2805 "
      f"(divbench8) -> {'OK' if ok else 'MISS'}")
assert ok, "this is not the gate that ships; the rows below are not readable"

print(f"\n{'arm':<28}{'TP/FP/FN':>16}{'div_J':>9}{'d div_J':>10}{'d score':>10}"
      f"{'cut':>7}{'TP cut':>8}")
ARMS = ([(f"DIV_PARENT_UM {t}", d_pq <= t) for t in (12.0, 11.0, 10.0, 9.5, 9.0)]
        + [(f"DIV_SISTER_UM {t}", d_s <= t)
           for t in (15.0, 14.0, 13.0, 12.0, 11.0, 10.0)]
        + [("theirs: 9.0 / 14.0", (d_pq <= 9.0) & (d_s <= 14.0)),
           ("ours + sister 12.0", (d_pq <= 12.0) & (d_s <= 12.0))])
for name, m in ARMS:
    t_, f_, n_, j_ = sc(gate & m)
    print(f"  {name:<26}{f'{t_}/{f_}/{n_}':>16}{j_:>9.4f}{j_ - B[3]:>+10.4f}"
          f"{0.1 * (j_ - B[3]):>+10.5f}{int(((~m) & cnt).sum()):>7}"
          f"{int(((~m) & (lab == 2)).sum()):>8}")

print("\n=== WHY, in one table: the gate is NOT blind past GATE_UM ===")
print("  The record's reading was that a fork beyond the ranker's 10.0 um "
      "pair gate carries NaN in 4 of the 20 gate features, so the gate "
      "cannot judge it. Survival rates say otherwise.")
print(f"  {'band':<26}{'counted':>9}{'TP':>5}{'FP':>5}"
      f"{'TP kept':>9}{'FP kept':>9}{'TP surv':>9}{'FP surv':>9}")
for name, m in (("d_pq <= 10 (inside gate)", d_pq <= 10.0),
                ("d_pq > 10 (NaN features)", d_pq > 10.0),
                ("d_s > 12", d_s > 12.0)):
    c = m & cnt
    tp = int(((lab == 2) & c).sum())
    fp = int(((lab == 1) & c).sum())
    tk = int(((lab == 2) & c & gate).sum())
    fk = int(((lab == 1) & c & gate).sum())
    print(f"  {name:<26}{int(c.sum()):>9}{tp:>5}{fp:>5}{tk:>9}{fk:>9}"
          f"{(tk / tp if tp else float('nan')):>9.2f}"
          f"{(fk / fp if fp else float('nan')):>9.2f}")

print("\n=== the separation the radii are cutting on ===")
for nm, v in (("d_pq", d_pq), ("d_s", d_s)):
    print(f"  {nm:<6} TP p50 {np.percentile(v[lab == 2], 50):5.2f} "
          f"p90 {np.percentile(v[lab == 2], 90):5.2f} | "
          f"FP p50 {np.percentile(v[lab == 1], 50):5.2f} "
          f"p90 {np.percentile(v[lab == 1], 90):5.2f}")
