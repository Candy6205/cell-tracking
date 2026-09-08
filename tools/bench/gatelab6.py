"""gatelab6 -- the 28-column fork table from divbench4, offline.

Two questions the kernel's single-point rows cannot settle:
  (a) is any 28-feature gate distinguishable from the shipped 20-feature one,
      under the SAME paired bootstrap over movies that gatelab used?
  (b) is dv2 -- the one new feature with real signal, AUC 0.798 against dv's
      0.787 -- worth anything ON TOP of the incumbents, or is it the same
      information one frame later?
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
lines = open(SC + "forks_f4_v4.csv").read().strip().split("\n")
hdr = lines[0].split(",")
FEATS = hdr[5:]
rows = [ln.split(",") for ln in lines[1:]]
movie = np.array([r[0] for r in rows])
fold = np.array([int(r[1]) for r in rows])
lab = np.array([int(r[4]) for r in rows])
X = np.array([[float(v) for v in r[5:]] for r in rows])
counted = lab >= 1
PM = json.load(open(SC + "db3_per_movie.json"))
gt_of = {m: PM[m]["gt"] for m in PM}
N_GT = sum(gt_of.values())
print(f"{len(lab)} forks, {int(counted.sum())} counted, {int((lab == 2).sum())} TP, "
      f"N_GT {N_GT}, {len(FEATS)} features   ({time.time() - t0:.1f}s)")

F20 = FEATS[:20]
NEW = FEATS[20:]
print(f"  incumbents: {' '.join(F20)}")
print(f"  new:        {' '.join(NEW)}")


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


def oof2(cols):
    """divbench3's alternating 2-fold movie split, the shipped scheme."""
    p = np.full(len(lab), np.nan)
    for k in (0, 1):
        tr = (fold != k) & counted
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10, random_state=0)
        m.fit(X[tr][:, cols], (lab[tr] == 2).astype(int))
        te = fold == k
        p[te] = m.predict_proba(X[te][:, cols])[:, 1]
    return p


IDX = {f: i for i, f in enumerate(FEATS)}
SETS = {
    "20 (SHIPPED)": [IDX[f] for f in F20],
    "20 + dv2": [IDX[f] for f in F20 + ["dv2"]],
    "20 + dv2 + p_q_marg": [IDX[f] for f in F20 + ["dv2", "p_q_marg"]],
    "28 (all)": list(range(len(FEATS))),
    "20 - dv + dv2": [IDX[f] for f in F20 if f != "dv"] + [IDX["dv2"]],
    "6 strong + dv2": [IDX[f] for f in ("sc_q", "dv", "sc_p", "t_frac", "pf_c1",
                                        "n_orph", "dv2")],
}
P = {}
print(f"\n{'feature set':<24} {'pooled AUC':>11} {'TP':>4} {'FP':>5} {'div_J':>8} "
      f"{'d score':>9}")
for nm, cols in SETS.items():
    p = oof2(cols)
    P[nm] = p
    a = auc((lab[counted] == 2).astype(int), p[counted])
    keep = p > 0.03
    tp, fp = int((lab[keep] == 2).sum()), int((lab[keep] == 1).sum())
    J = tp / (tp + fp + (N_GT - tp))
    print(f"{nm:<24} {a:>11.4f} {tp:>4} {fp:>5} {J:>8.4f} {0.1 * (J - 0.1337):>+9.4f}")

print(f"\nPAIRED BOOTSTRAP over movies (2000 draws), cut p > 0.03, exact per-movie N_GT")
rng = np.random.default_rng(20260908)
movies = np.array(sorted(gt_of))
idx_of = {m: np.where(movie == m)[0] for m in movies}
draws = [rng.choice(movies, size=len(movies), replace=True) for _ in range(2000)]
JS = {}
for nm, p in P.items():
    arr = []
    for d in draws:
        ii = np.concatenate([idx_of[m] for m in d])
        ngt = sum(gt_of[m] for m in d)
        keep = p[ii] > 0.03
        tp = int((lab[ii][keep] == 2).sum())
        fp = int((lab[ii][keep] == 1).sum())
        arr.append(tp / max(1, tp + fp + (ngt - tp)))
    JS[nm] = np.array(arr)
base = "20 (SHIPPED)"
print(f"  {'feature set':<24} {'d div_J':>9} {'90% interval':>22}  distinguishable")
for nm in SETS:
    if nm == base:
        continue
    d = JS[nm] - JS[base]
    lo, hi = np.quantile(d, .05), np.quantile(d, .95)
    print(f"  {nm:<24} {np.mean(d):>+9.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]  "
          f"{'YES' if (lo > 0 or hi < 0) else 'no'}")

print("\nIs dv2 the same information as dv, one frame later?")
dv, dv2 = X[:, IDX["dv"]], X[:, IDX["dv2"]]
m = np.isfinite(dv) & np.isfinite(dv2)
print(f"  correlation over all {int(m.sum())} forks with both: "
      f"{np.corrcoef(dv[m], dv2[m])[0, 1]:.3f}")
mc = m & counted
print(f"  over the {int(mc.sum())} counted ones: {np.corrcoef(dv[mc], dv2[mc])[0, 1]:.3f}")
print(f"  dv2 is missing on {int((~np.isfinite(dv2)).sum())} forks "
      f"({int((~np.isfinite(dv2) & counted).sum())} counted) -- a chain that ends or "
      f"branches within two frames")
print(f"  AUC on counted: dv {auc((lab[counted] == 2).astype(int), dv[counted]):.4f}  "
      f"dv2 {auc((lab[counted] == 2).astype(int), dv2[counted]):.4f}  "
      f"max(dv,dv2) {auc((lab[counted] == 2).astype(int), np.fmax(dv[counted], dv2[counted])):.4f}")
print(f"\ndone in {time.time() - t0:.1f}s")

print("\nFAIR COMPARISON: each model at its OWN best threshold, so a shifted"
      "\nprobability scale cannot be mistaken for a worse ranking.")
print(f"  {'feature set':<24} {'best p':>8} {'TP':>4} {'FP':>5} {'div_J':>8} {'vs shipped':>11}")
BEST = {}
for nm, p in P.items():
    ok = np.isfinite(p)
    bJ, bthr, btp, bfp = -1, None, 0, 0
    for thr in np.unique(p[ok]):
        keep = ok & (p > thr)
        tp = int((lab[keep] == 2).sum())
        fp = int((lab[keep] == 1).sum())
        J = tp / max(1, tp + fp + (N_GT - tp))
        if J > bJ:
            bJ, bthr, btp, bfp = J, thr, tp, fp
    BEST[nm] = (bJ, bthr)
    ref = BEST["20 (SHIPPED)"][0]
    print(f"  {nm:<24} {bthr:>8.4f} {btp:>4} {bfp:>5} {bJ:>8.4f} "
          f"{(bJ - ref):>+11.4f}")
print("  (the best threshold is chosen ON the test labels, so these are upper"
      "\n   bounds for each ranking, not achievable scores -- they compare RANKINGS)")
