"""gatelab9 -- WHY does adding 44b6's forks help the gate on 6bba?

gatelab8 ruled out capacity (padding with a bootstrap of the same rows is
distinguishably WORSE) and label balance (negatives alone, from either batch,
do nothing). The remaining hypothesis has a mechanism: two of the twenty
features are DENSITY counts -- dens10 (primaries within 10 um of the mother)
and n_orph (orphans in the frame) -- both read INVERSE (AUC 0.316 and 0.245),
and 44b6 has 327 true cells per frame against 6bba's 97. A gate trained on one
batch can use "few neighbours" as a proxy for "true division"; the other batch
breaks that proxy.

The test: drop the two density features from BOTH gates. If the joint gate's
advantage collapses, density was the thing 44b6 was correcting.
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
F20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
       "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
       "dens10", "n_orph", "t_frac", "rank_gap"]


def load(fn):
    L = open(SC + fn).read().strip().split("\n")
    R = [l.split(",") for l in L[1:]]
    return (np.array([r[0] for r in R]), np.array([int(r[1]) for r in R]),
            np.array([int(r[4]) for r in R]),
            np.array([[float(v) for v in r[5:25]] for r in R]))


m6, f6, l6, X6 = load("forks_f4_v4.csv")
m4, f4_, l4, X4 = load("forks_44b6.csv")
c6, c4 = l6 >= 1, l4 >= 1
PM = json.load(open(SC + "db3_per_movie.json"))
gt6 = {m: PM[m]["gt"] for m in PM}
N6 = sum(gt6.values())
movies = np.array(sorted(gt6))
idx_of = {m: np.where(m6 == m)[0] for m in movies}
rng0 = np.random.default_rng(20260908)
draws = [rng0.choice(movies, size=len(movies), replace=True) for _ in range(2000)]

print("the two density features, by batch (median over counted forks):")
for j, f in ((F20.index("dens10"), "dens10"), (F20.index("n_orph"), "n_orph")):
    for nm, X, l, c in (("6bba", X6, l6, c6), ("44b6", X4, l4, c4)):
        v = X[:, j]
        tp = np.nanmedian(v[(l == 2)])
        fp = np.nanmedian(v[(l == 1)])
        print(f"  {f:<8} {nm}: TP {tp:>6.1f}  FP {fp:>6.1f}  "
              f"all forks {np.nanmedian(v):>6.1f}")


def oof(cols, joint, nfold=2):
    p = np.full(len(l6), np.nan)
    parts = ([(f6 != k, f6 == k) for k in (0, 1)] if nfold == 2 else None)
    for tr_m, te_m in parts:
        tr = tr_m & c6
        Xtr, ytr = X6[tr][:, cols], (l6[tr] == 2).astype(int)
        if joint:
            Xtr = np.vstack([Xtr, X4[c4][:, cols]])
            ytr = np.concatenate([ytr, (l4[c4] == 2).astype(int)])
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10, random_state=0)
        m.fit(Xtr, ytr)
        p[te_m] = m.predict_proba(X6[te_m][:, cols])[:, 1]
    return p


def J6(p, thr=0.03):
    keep = np.isfinite(p) & (p > thr)
    tp, fp = int((l6[keep] == 2).sum()), int((l6[keep] == 1).sum())
    return tp / max(1, tp + fp + (N6 - tp)), tp, fp


def boot(p, thr=0.03):
    out = []
    for d in draws:
        ii = np.concatenate([idx_of[m] for m in d])
        ngt = sum(gt6[m] for m in d)
        keep = p[ii] > thr
        tp, fp = int((l6[ii][keep] == 2).sum()), int((l6[ii][keep] == 1).sum())
        out.append(tp / max(1, tp + fp + (ngt - tp)))
    return np.array(out)


ALL = list(range(20))
NODENS = [i for i in ALL if F20[i] not in ("dens10", "n_orph")]
print(f"\n{'feature set':<24} {'shipped':>9} {'joint':>9} {'paired d':>10} "
      f"{'90% interval':>22}")
for nm, cols in (("all 20", ALL), ("18, no density", NODENS)):
    ps, pj = oof(cols, False), oof(cols, True)
    d = boot(pj) - boot(ps)
    lo, hi = np.quantile(d, .05), np.quantile(d, .95)
    print(f"{nm:<24} {J6(ps)[0]:>9.4f} {J6(pj)[0]:>9.4f} {np.mean(d):>+10.4f} "
          f"[{lo:>+7.4f}, {hi:>+7.4f}]")
print("\n  If the joint advantage survives without dens10/n_orph, density is NOT"
      "\n  the mechanism and the entry must say the mechanism is unknown.")
print(f"\ndone in {time.time() - t0:.1f}s")
