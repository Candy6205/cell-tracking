"""Test the user's hypothesis: xl leads, DC speaks ONLY at extremes
(strong veto / strong support), silent otherwise.

All from judge_rows.npz, CPU. Held-out protocol identical to
judge_audit.py: every second zh crop, ALL rows, matched labels.

Four measurements:
  1. VETO quality: among candidates xl itself would keep (mean3 >= tau_xl),
     if DC's response hd < tau_veto, how often is the candidate FAKE?
     (veto precision) and how many true cells die (collateral).
  2. RESCUE quality: among candidates xl would drop (mean3 < 0.15),
     if hd >= tau_up, how often is it REAL?
  3. Collateral on COMPETITION positives, split 6bba vs 44b6: of real
     annotated comp cells, what fraction would a DC veto kill? This is
     the 6bba-risk proxy (6bba is where blending died).
  4. Gated score AUC on the held-out zh rows vs mean3 / DCheavy.
"""
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

d = np.load(pathlib.Path(__file__).parent / "judge_rows.npz",
            allow_pickle=True)
X, y = d["X"], d["y"].astype(int)
dom, grp = d["dom"], d["grp"]
FEATS = ["h0", "h1", "h2", "hd", "mean3", "n05", "n15"]
hd = X[:, 3]
mean3 = X[:, 4]

zh = dom == "zh"
crops = sorted(set(grp[zh].tolist()))
hold = set(crops[::2])
is_hold = np.asarray([g in hold for g in grp]) & zh
yh, hdh, m3h = y[is_hold], hd[is_hold], mean3[is_hold]
print(f"held-out zh rows {int(is_hold.sum()):,} (pos {int(yh.sum()):,})")

print("\n=== 1. VETO: candidates xl keeps (mean3 >= 0.15), hd < tau ===")
keep = m3h >= 0.15
print(f"  xl-kept candidates: {int(keep.sum()):,}, of which real "
      f"{int(yh[keep].sum()):,} ({yh[keep].mean():.1%})")
print(f"  {'tau_veto':>8} {'vetoed':>8} {'fake%':>7} {'real killed':>12} "
      f"{'%of all real':>13}")
for tau in (0.01, 0.02, 0.05, 0.10, 0.15):
    v = keep & (hdh < tau)
    if v.sum() == 0:
        continue
    fake = 1 - yh[v].mean()
    killed = int(yh[v].sum())
    print(f"  {tau:>8.2f} {int(v.sum()):>8,} {fake:>7.1%} {killed:>12,} "
          f"{killed / max(1, int(yh[keep].sum())):>13.2%}")

print("\n=== 2. RESCUE: candidates xl drops (mean3 < 0.15), hd >= tau ===")
dropcand = m3h < 0.15
print(f"  xl-dropped candidates: {int(dropcand.sum()):,}, of which real "
      f"{int(yh[dropcand].sum()):,} ({yh[dropcand].mean():.1%})")
print(f"  {'tau_up':>8} {'rescued':>8} {'real%':>7}")
for tau in (0.15, 0.30, 0.50, 0.70, 0.90):
    r = dropcand & (hdh >= tau)
    if r.sum() == 0:
        continue
    print(f"  {tau:>8.2f} {int(r.sum()):>8,} {yh[r].mean():>7.1%}")

print("\n=== 3. COLLATERAL on comp GT positives (the 6bba-risk proxy) ===")
comp = dom == "comp"
is6 = comp & np.asarray([g.startswith("6bba") for g in grp])
is4 = comp & np.asarray([g.startswith("44b6") for g in grp])
print(f"  comp positives: 6bba {int(is6.sum()):,}, 44b6 {int(is4.sum()):,}")
print(f"  {'tau_veto':>8} {'6bba killed%':>13} {'44b6 killed%':>13}")
for tau in (0.01, 0.02, 0.05, 0.10, 0.15):
    k6 = (hd[is6] < tau).mean() if is6.sum() else float("nan")
    k4 = (hd[is4] < tau).mean() if is4.sum() else float("nan")
    print(f"  {tau:>8.2f} {k6:>13.2%} {k4:>13.2%}")
# and how many comp positives DC STRONGLY supports (sanity: DC sees comp)
for tau in (0.5, 0.9):
    print(f"  hd >= {tau}: 6bba {float((hd[is6] >= tau).mean()):.1%}, "
          f"44b6 {float((hd[is4] >= tau).mean()):.1%}")

print("\n=== 4. gated-score AUC on held-out zh (vs the audit table) ===")
Xh = X[is_hold]
def gated(m3, h, lo, hi, bonus=0.2, malus=0.2):
    s = m3.copy()
    s[h >= hi] += bonus          # strong support
    s[h < lo] -= malus           # strong veto
    return s
scores = {
    "mean3 (current)": m3h,
    "DCheavy blend": (Xh[:, 0] + Xh[:, 1] + 0.5 * Xh[:, 2]
                      + 2.0 * Xh[:, 3]) / 4.5,
    "gate lo.02 hi.5": gated(m3h, hdh, 0.02, 0.5),
    "gate lo.05 hi.5": gated(m3h, hdh, 0.05, 0.5),
    "gate lo.02 hi.9": gated(m3h, hdh, 0.02, 0.9),
    "gate lo.05 hi.9": gated(m3h, hdh, 0.05, 0.9),
}
for name, s in scores.items():
    print(f"  {name:<18} AUC {roc_auc_score(yh, s):.4f}")
