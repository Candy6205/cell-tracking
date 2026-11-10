"""gatelab -- offline work on divbench3's fork table (forks_f4.csv).

The pruning curve in divbench3 is a pure function of (label, gate p): div_J
= TP/(TP+FP+FN) with FN = N_GT - TP and N_GT = 125 fixed. So every gate
variant can be priced here without a kernel. adj_J cannot -- that needs the
real pruning -- but divbench3 measured adj_J RISING under every cut, so
div_J is the lever being optimised.

ANCHOR (must reproduce divbench3, entry 2026-09-06):
  2-fold HistGB(100, lr .05, msl 10, rs 0) on the other fold's counted forks
  -> out-of-fold AUC 0.9283 (fold0, 122 counted) / 0.8326 (fold1, 177)
  -> q70 44/79/81 div_J 0.2157 | q80 42/46/83 0.2456 | p>0.03 42/47/83 0.2442
  -> oracle 0.4000
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

CSV = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
       "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/forks_f4.csv")
N_GT = 125

hdr = open(CSV).readline().strip().split(",")
raw = np.genfromtxt(CSV, delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
movie = np.array([r[0] for r in raw])
fold = np.array([int(r[1]) for r in raw])
lab = np.array([int(r[4]) for r in raw])
FEATS = hdr[5:]
X = np.array([[float(r[5 + j]) for j in range(len(FEATS))] for r in raw])
counted = lab >= 1
tp_all = int((lab == 2).sum())
print(f"forks {len(lab)}  counted {int(counted.sum())}  TP {tp_all}  "
      f"FP {int((lab == 1).sum())}  N_GT {N_GT}  features {len(FEATS)}")
print(f"folds: 0 -> {int((fold == 0).sum())} forks / "
      f"{int((counted & (fold == 0)).sum())} counted / "
      f"{int(((lab == 2) & (fold == 0)).sum())} TP; "
      f"1 -> {int((fold == 1).sum())} / {int((counted & (fold == 1)).sum())} / "
      f"{int(((lab == 2) & (fold == 1)).sum())}")


def auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if not y.any() or y.all():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    i = 0
    ss = s[o]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def divJ(keep):
    tp = int((lab[keep] == 2).sum())
    fp = int((lab[keep] == 1).sum())
    return tp / max(1, tp + fp + (N_GT - tp)), tp, fp, N_GT - tp


def curve(p, title, cuts=("q70", "q80", 0.01, 0.02, 0.03, 0.05), quiet=False):
    ok = np.isfinite(p)
    out = {}
    best = (-1, None)
    for c in cuts:
        thr = (np.quantile(p[ok], float(c[1:]) / 100) if isinstance(c, str)
               else float(c))
        keep = ok & (p > thr)
        J, tp, fp, fn = divJ(keep)
        out[c] = (J, tp, fp, fn, thr)
        if J > best[0]:
            best = (J, c)
    # the best achievable threshold on this ranking, and where it sits
    order = np.argsort(-p[ok])
    lo = np.sort(np.unique(p[ok]))
    bJ, bthr = -1, None
    for thr in lo:
        J, *_ = divJ(ok & (p > thr))
        if J > bJ:
            bJ, bthr = J, thr
    if not quiet:
        print(f"\n{title}")
        for c, (J, tp, fp, fn, thr) in out.items():
            print(f"   {str(c):>5} (p>{thr:.4f}): {tp:>3}/{fp:>4}/{fn:<3} div_J {J:.4f}"
                  f"  d score {0.1 * (J - 0.1337):+.4f}")
        print(f"   ORACLE THRESHOLD on this ranking: p>{bthr:.4f} div_J {bJ:.4f}"
              f"  (ceiling for this gate = how good the RANKING is)")
    return out, bJ


def oof(make, cols, groups, n_splits, seed=None, label="", quiet=False):
    """Out-of-fold p over every fork, grouped by movie."""
    p = np.full(len(lab), np.nan)
    Xc = X[:, cols]
    if n_splits == 2:                      # divbench3's own alternating split
        parts = [(fold != k, fold == k) for k in (0, 1)]
    else:
        gk = GroupKFold(n_splits=n_splits)
        parts = [(np.isin(np.arange(len(lab)), tr), np.isin(np.arange(len(lab)), te))
                 for tr, te in gk.split(Xc, lab, groups)]
    aucs = []
    for tr_m, te_m in parts:
        tr = tr_m & counted
        if (lab[tr] == 2).sum() < 3:
            continue
        m = make()
        Xtr = Xc[tr]
        if isinstance(m, LogisticRegression):
            Xtr = np.nan_to_num(Xtr, nan=0.0)
        m.fit(Xtr, (lab[tr] == 2).astype(int))
        Xte = Xc[te_m]
        if isinstance(m, LogisticRegression):
            Xte = np.nan_to_num(Xte, nan=0.0)
        p[te_m] = m.predict_proba(Xte)[:, 1]
        tc = te_m & counted
        if tc.sum() and (lab[tc] == 2).any() and (lab[tc] == 1).any():
            aucs.append((int(tc.sum()), auc((lab[tc] == 2).astype(int), p[tc])))
    pooled = auc((lab[counted] == 2).astype(int), p[counted])
    if not quiet:
        print(f"\n--- {label}: folds "
              + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
              + f"   POOLED out-of-fold AUC {pooled:.4f}")
    return p, pooled


HGB = lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)
ALL = list(range(len(FEATS)))

print("\n" + "=" * 78)
print("STEP 1 -- ANCHOR: reproduce divbench3's 2-fold gate exactly")
print("=" * 78)
p2, a2 = oof(HGB, ALL, movie, 2, label="divbench3 2-fold, 20 features, HistGB")
c2, ceil2 = curve(p2, "divbench3's own curve (want q70 44/79/81 .2157, "
                      "q80 42/46/83 .2456, p>0.03 42/47/83 .2442)")
ok = np.isfinite(p2)
for k in (0, 1):
    m = (fold == k) & counted
    print(f"   fold{k} out-of-fold AUC on its {int(m.sum())} counted forks: "
          f"{auc((lab[m] == 2).astype(int), p2[m]):.4f}   "
          f"(divbench3: {'0.9283' if k == 0 else '0.8326'})")
print(f"   oracle (all FP gone, all TP kept): {tp_all / N_GT:.4f}")

print("\n" + "=" * 78)
print("STEP 2 -- how much is the CUT worth, and how much is the RANKING worth")
print("=" * 78)
print(f"   shipped p>0.03 on the 2-fold ranking : div_J 0.2442")
print(f"   best cut on the SAME ranking        : div_J {ceil2:.4f}  "
      f"(+{0.1 * (ceil2 - 0.2442):+.4f} of score -- all that cut-tuning can buy)")
print(f"   oracle (a perfect ranking)          : div_J 0.4000")
print(f"   => {0.4 - ceil2:.4f} of the {0.4 - 0.2442:.4f} gap is RANKING, "
      f"{ceil2 - 0.2442:.4f} is the cut.")

print("\n" + "=" * 78)
print("STEP 3 -- more folds = more training data per gate (production trains on ALL)")
print("=" * 78)
RES = {}
for ns in (2, 3, 5, 8, 16):
    p, a = oof(HGB, ALL, movie, ns, label=f"HistGB, {ns}-fold grouped by movie")
    _, ce = curve(p, f"   curve, {ns}-fold", cuts=("q70", "q80", 0.02, 0.03, 0.05))
    RES[f"hgb-{ns}"] = (a, ce)

print("\n" + "=" * 78)
print("STEP 4 -- other models, 5-fold grouped by movie")
print("=" * 78)
MODELS = {
    "HistGB 100it lr.05 msl10 (shipped)": HGB,
    "HistGB 300it lr.03 msl5": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, min_samples_leaf=5, random_state=0),
    "HistGB 60it lr.10 msl20 depth3": lambda: HistGradientBoostingClassifier(
        max_iter=60, learning_rate=0.10, min_samples_leaf=20, max_depth=3,
        random_state=0),
    "HistGB 200it lr.05 msl10 l2=1": lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, min_samples_leaf=10, l2_regularization=1.0,
        random_state=0),
    "LogReg (nan->0, no scaling)": lambda: LogisticRegression(max_iter=2000, C=1.0),
}
for nm, mk in MODELS.items():
    p, a = oof(mk, ALL, movie, 5, label=nm)
    _, ce = curve(p, f"   curve: {nm}", cuts=("q70", "q80", 0.02, 0.03, 0.05))
    RES[nm] = (a, ce)

print("\n" + "=" * 78)
print("STEP 5 -- seed sensitivity of the shipped model (is 0.88 real or noise?)")
print("=" * 78)
for rs in (0, 1, 2, 3, 4):
    mk = lambda rs=rs: HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=rs)
    p, a = oof(mk, ALL, movie, 5, label=f"HistGB rs={rs}", quiet=True)
    _, ce = curve(p, "", cuts=("q80",), quiet=True)
    J03 = divJ(np.isfinite(p) & (p > 0.03))[0]
    print(f"   rs={rs}: pooled AUC {a:.4f}  best-cut div_J {ce:.4f}  p>0.03 div_J {J03:.4f}")
