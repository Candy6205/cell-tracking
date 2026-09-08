"""gatelab7 -- the JOINT fork gate over both imaging batches, offline.

This is the last open half of the 44b6 item: the gate ships trained on 6bba
alone. divbench5 measured that it transfers (AUC 0.8607 on 44b6, +0.0066 of
local score there). The question left is whether TRAINING on both is better
than training on 6bba alone, and it has to be asked in both directions,
because 6bba is 85% of the score weight and 44b6 is 15%:

  (a) on 44b6, out of fold: does adding 44b6's own forks to the training set
      beat the pure-6bba gate?
  (b) on 6bba, out of fold: does adding 44b6's forks COST anything at home?

Nothing here can be selected on 44b6's div_J alone -- 26 GT events. The
readout is the fork ranking (AUC on counted forks, which has 6 TP / 61 FP on
44b6) and, on 6bba, the same paired bootstrap gatelab used.
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")


def load(fn):
    lines = open(SC + fn).read().strip().split("\n")
    hdr = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    return (hdr[5:], np.array([r[0] for r in rows]),
            np.array([int(r[1]) for r in rows]),
            np.array([int(r[4]) for r in rows]),
            np.array([[float(v) for v in r[5:]] for r in rows]))


F6, m6, f6, l6, X6 = load("forks_f4_v4.csv")
F4, m4, f4_, l4, X4 = load("forks_44b6.csv")
assert F6 == F4, "the two fork tables do not share a column layout"
FEATS = F6
J20 = list(range(20))
c6, c4 = l6 >= 1, l4 >= 1
print(f"6bba: {len(l6)} forks, {int(c6.sum())} counted, {int((l6 == 2).sum())} TP")
print(f"44b6: {len(l4)} forks, {int(c4.sum())} counted, {int((l4 == 2).sum())} TP")
PM = json.load(open(SC + "db3_per_movie.json"))
gt6 = {m: PM[m]["gt"] for m in PM}
N6 = sum(gt6.values())
N4 = 26
print(f"GT divisions: 6bba {N6}, 44b6 {N4} (divbench5's REF: 6 TP + 20 FN)")


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if not y.any() or y.all():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    i, ss = 0, s[o]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def fit(Xtr, ytr, cols=J20):
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                       min_samples_leaf=10, random_state=0)
    m.fit(Xtr[:, cols], ytr)
    return m


def divJ(lab, p, thr, ngt):
    keep = np.isfinite(p) & (p > thr)
    tp, fp = int((lab[keep] == 2).sum()), int((lab[keep] == 1).sum())
    return tp / max(1, tp + fp + (ngt - tp)), tp, fp


print("\n" + "=" * 76)
print("(a) ON 44b6, out of fold: 6bba-only against 6bba + the other 44b6 fold")
print("=" * 76)
# the production gate: all 299 counted 6bba forks, nothing from 44b6
g6 = fit(X6[c6], (l6[c6] == 2).astype(int))
p_pure = g6.predict_proba(X4[:, J20])[:, 1]
# joint: all counted 6bba forks + the OTHER 44b6 fold's counted forks
p_joint = np.full(len(l4), np.nan)
for k in (0, 1):
    tr4 = c4 & (f4_ != k)
    Xtr = np.vstack([X6[c6], X4[tr4]])
    ytr = np.concatenate([(l6[c6] == 2).astype(int), (l4[tr4] == 2).astype(int)])
    gj = fit(Xtr, ytr)
    p_joint[f4_ == k] = gj.predict_proba(X4[f4_ == k][:, J20])[:, 1]
# 44b6-only, out of fold, for contrast (divbench5 could not even train it)
p_only = np.full(len(l4), np.nan)
for k in (0, 1):
    tr4 = c4 & (f4_ != k)
    if (l4[tr4] == 2).sum() >= 2:
        g = fit(X4[tr4], (l4[tr4] == 2).astype(int))
        p_only[f4_ == k] = g.predict_proba(X4[f4_ == k][:, J20])[:, 1]

print(f"  {'gate':<34} {'AUC on 44b6 counted':>20} {'p>0.03 TP/FP':>14} {'div_J':>8}")
for nm, p in (("6bba only (SHIPPED)", p_pure),
              ("6bba + other 44b6 fold (JOINT)", p_joint),
              ("44b6 only, out of fold", p_only)):
    a = auc((l4[c4] == 2).astype(int), p[c4])
    J, tp, fp = divJ(l4, p, 0.03, N4)
    print(f"  {nm:<34} {a:>20.4f} {str(tp) + '/' + str(fp):>14} {J:>8.4f}")
print("  (divbench5 measured the shipped gate at AUC 0.8607, 4/7, div_J 0.1212 --"
      "\n   p_pure here refits it from the same rows, so it should land on that)")

print("\n" + "=" * 76)
print("(b) ON 6bba, out of fold: does adding 44b6's forks cost anything at home?")
print("=" * 76)
p_home = np.full(len(l6), np.nan)
p_home_j = np.full(len(l6), np.nan)
for k in (0, 1):
    tr6 = c6 & (f6 != k)
    g = fit(X6[tr6], (l6[tr6] == 2).astype(int))
    p_home[f6 == k] = g.predict_proba(X6[f6 == k][:, J20])[:, 1]
    Xtr = np.vstack([X6[tr6], X4[c4]])
    ytr = np.concatenate([(l6[tr6] == 2).astype(int), (l4[c4] == 2).astype(int)])
    gj = fit(Xtr, ytr)
    p_home_j[f6 == k] = gj.predict_proba(X6[f6 == k][:, J20])[:, 1]
print(f"  {'gate':<34} {'AUC on 6bba counted':>20} {'p>0.03 TP/FP':>14} {'div_J':>8}")
for nm, p in (("6bba only (SHIPPED)", p_home),
              ("6bba + all 44b6 (JOINT)", p_home_j)):
    a = auc((l6[c6] == 2).astype(int), p[c6])
    J, tp, fp = divJ(l6, p, 0.03, N6)
    print(f"  {nm:<34} {a:>20.4f} {str(tp) + '/' + str(fp):>14} {J:>8.4f}")

print("\n  PAIRED BOOTSTRAP over 6bba movies (2000 draws), cut p > 0.03:")
rng = np.random.default_rng(20260908)
movies = np.array(sorted(gt6))
idx_of = {m: np.where(m6 == m)[0] for m in movies}
draws = [rng.choice(movies, size=len(movies), replace=True) for _ in range(2000)]
JS = {}
for nm, p in (("home", p_home), ("joint", p_home_j)):
    arr = []
    for d in draws:
        ii = np.concatenate([idx_of[m] for m in d])
        ngt = sum(gt6[m] for m in d)
        keep = p[ii] > 0.03
        tp, fp = int((l6[ii][keep] == 2).sum()), int((l6[ii][keep] == 1).sum())
        arr.append(tp / max(1, tp + fp + (ngt - tp)))
    JS[nm] = np.array(arr)
d = JS["joint"] - JS["home"]
lo, hi = np.quantile(d, .05), np.quantile(d, .95)
print(f"    joint minus 6bba-only, on 6bba: {np.mean(d):+.4f}  90% [{lo:+.4f}, {hi:+.4f}]  "
      f"distinguishable: {'YES' if (lo > 0 or hi < 0) else 'no'}")

print("\n" + "=" * 76)
print("(c) the weighted picture: 6bba 85% + 44b6 15% of the score")
print("=" * 76)
for nm, ph, p4 in (("6bba only (SHIPPED)", p_home, p_pure),
                   ("joint", p_home_j, p_joint)):
    J6, _t, _f = divJ(l6, ph, 0.03, N6)
    J4, _t, _f = divJ(l4, p4, 0.03, N4)
    d6 = 0.1 * (J6 - 0.1337)
    d4 = 0.1 * (J4 - 0.0690)
    print(f"  {nm:<24} 6bba d score {d6:+.4f}  44b6 d score {d4:+.4f}  "
          f"weighted {0.85 * d6 + 0.15 * d4:+.4f}")
print("  (division term only; adj_J moves too and is measured in the benches)")
print(f"\ndone in {time.time() - t0:.1f}s")
