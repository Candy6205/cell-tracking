"""Would tightening the division patch's reach pay, GIVEN the fork gate?

Answered off `forks_dw55.csv`, which hparabench_ref already wrote, with the
gate retrained locally to divbench9's recipe. No compute to buy.

ANCHORS, both exact as of 2026-09-09:
  * 20 features / 2 alternating movie folds / p > 0.03, trained on the 6bba
    counted forks only -> 42 / 47 / 83, div_J 0.2442 = divbench3's recorded
    `REF + gate6`.
  * the same with divbench5's 67 counted 44b6 forks added to BOTH training
    folds -> 46 / 39 / 79, div_J 0.2805 = divbench8's `REF + joint`, which
    is what SHIPS (LB 0.944, ref 56086066).
Run `divradius_joint.py` for the second. If either misses, this is not the
gate that ships and none of the rows below mean anything.

⚠ WHAT THIS APPROXIMATES. It filters the fork PROPOSALS that a run at the
shipped radii already produced, rather than re-running `_add_safe_divisions`
at a smaller radius. The two differ where the caps bind: `DIV_GLOBAL_CAP`,
`DIV_FRAME_CAP` and the `used_p`/`used_q` exclusivity mean a proposal that
is dropped can free a slot for another. Proposals sort by
`d_pq + 0.15 * d_s`, so the far ones sort LAST and win contested slots least
often, which bounds how much that can matter -- but it is not zero, and a
result inside a few thousandths should not be read off this alone.
"""
import csv
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CSV = sys.argv[1] if len(sys.argv) > 1 else "forks_dw55.csv"
FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
CUT = 0.03
FN0 = 75          # GT divisions with no fork proposed at all (divbench3)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))


def f(s):
    try:
        return float(s)
    except Exception:
        return np.nan


X = np.array([[f(r[c]) for c in FEAT20] for r in rows], np.float64)
lab = np.array([int(r["label"]) for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
d_pq = np.array([f(r["d_pq"]) for r in rows])
d_s = np.array([f(r["d_s"]) for r in rows])
cnt = lab >= 1
print(f"{len(rows)} forks, {int(cnt.sum())} counted "
      f"({int((lab == 2).sum())} TP, {int((lab == 1).sum())} FP)")

p = np.full(len(rows), np.nan)
for k in (0, 1):
    tr = (fold != k) & cnt
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                       min_samples_leaf=10, random_state=0)
    m.fit(X[tr], (lab[tr] == 2).astype(int))
    p[fold == k] = m.predict_proba(X[fold == k])[:, 1]
auc = roc_auc_score((lab[cnt] == 2).astype(int), p[cnt])
print(f"gate: pooled out-of-fold AUC {auc:.4f} over the counted forks")


def score(keep_mask, tag, note=""):
    """div TP/FP/FN and div_J for a set of surviving forks."""
    tp = int(((lab == 2) & keep_mask).sum())
    fp = int(((lab == 1) & keep_mask).sum())
    fn = FN0 + (int((lab == 2).sum()) - tp)
    j = tp / max(1, tp + fp + fn)
    print(f"  {tag:<22}{tp:>5}/{fp:>4}/{fn:<5} div_J {j:>7.4f}"
          f"  d {j - BASE[0]:>+8.4f}   {note}")
    return j


gate = p > CUT
BASE = [0.0]
print("\n=== ANCHOR: the shipped radii (12.0 / 15.0), gate at p > 0.03 ===")
print("  divbench3 recorded REF + gate6 as 44 / 54 / 81. Anything else and "
      "this reproduction is not the gate that ships.")
tp0 = int(((lab == 2) & gate).sum())
fp0 = int(((lab == 1) & gate).sum())
fn0 = FN0 + 50 - tp0
BASE[0] = tp0 / max(1, tp0 + fp0 + fn0)
print(f"  got {tp0}/{fp0}/{fn0}  div_J {BASE[0]:.4f}   "
      f"{'OK' if (tp0, fp0, fn0) == (44, 54, 81) else 'MISS (want 44/54/81)'}")

print("\n=== PARENT RADIUS, gate held fixed (the fork is never proposed) ===")
print("  ours 12.0 ships; the ranker's own pair gate is 10.0, so a fork "
      "beyond that carries NaN in 4 of the 20 gate features; theirs is 9.0.")
for thr in (12.0, 11.0, 10.0, 9.5, 9.0, 8.0):
    n_rm = int(((d_pq > thr) & cnt).sum())
    score(gate & (d_pq <= thr), f"DIV_PARENT_UM {thr}",
          f"{n_rm} counted forks never proposed")

print("\n=== SISTER RADIUS, gate held fixed ===")
print("  ours 15.0 ships, theirs is 14.0.")
for thr in (15.0, 14.0, 13.0, 12.0, 11.0):
    n_rm = int(((d_s > thr) & cnt).sum())
    score(gate & (d_s <= thr), f"DIV_SISTER_UM {thr}",
          f"{n_rm} counted forks never proposed")

print("\n=== THE CROSS: both radii at once ===")
for pu in (12.0, 10.0, 9.0):
    for su in (15.0, 13.0, 12.0):
        score(gate & (d_pq <= pu) & (d_s <= su), f"parent {pu} sister {su}")

print("\n=== WHAT THE GATE ALREADY DOES OUT THERE ===")
print("  If the gate already rejects the far forks, tightening the radius "
      "can only cost true ones. Survival = share passing p > 0.03.")
print(f"  {'band':<26}{'counted':>9}{'TP':>5}{'FP':>5}"
      f"{'TP kept':>9}{'FP kept':>9}{'TP surv':>9}{'FP surv':>9}")
for name, m in (("d_pq <= 10 (in gate)", d_pq <= 10.0),
                ("d_pq > 10 (NaN feats)", d_pq > 10.0),
                ("d_s > 13", d_s > 13.0),
                ("d_s > 14", d_s > 14.0)):
    c = m & cnt
    tp = int(((lab == 2) & c).sum())
    fp = int(((lab == 1) & c).sum())
    tk = int(((lab == 2) & c & gate).sum())
    fk = int(((lab == 1) & c & gate).sum())
    print(f"  {name:<26}{int(c.sum()):>9}{tp:>5}{fp:>5}{tk:>9}{fk:>9}"
          f"{(tk / tp if tp else float('nan')):>9.2f}"
          f"{(fk / fp if fp else float('nan')):>9.2f}")
