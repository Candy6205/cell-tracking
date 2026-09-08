"""gatelab8 -- controls on gatelab7's joint-gate result.

gatelab7: a gate trained on 6bba's other fold PLUS all 67 counted 44b6 forks
reads div_J 0.2805 on 6bba out of fold against the shipped 0.2442, paired
bootstrap +0.0360 [+0.0163, +0.0597]. That is the first gate variant to clear
the +-0.03 bar. Before believing it, rule out the boring explanations:

  (A) MORE ROWS, not 44b6 rows. Pad the 6bba training fold with a bootstrap
      resample of ITSELF to the same row count. If that gains too, the effect
      is capacity or regularisation, not information.
  (B) LABEL BALANCE. 44b6 brings 6 TP and 61 FP -- far more negative than
      6bba's 1:5. Pad instead with 6bba negatives only.
  (C) THE FOLD SPLIT. Repeat under 3-fold and 4-fold over movies.
  (D) THE MODEL. Repeat at two other hyperparameter settings.
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")


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


def mk(kind="std"):
    if kind == "deep":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=5, random_state=0)
    if kind == "shallow":
        return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.10,
                                              min_samples_leaf=20, max_depth=3,
                                              random_state=0)
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                          min_samples_leaf=10, random_state=0)


def oof6(extra, nfold=2, kind="std", seed=0):
    """out-of-fold p on 6bba; `extra(tr_mask, rng)` returns (X, y) to append."""
    p = np.full(len(l6), np.nan)
    if nfold == 2:
        parts = [(f6 != k, f6 == k) for k in (0, 1)]
    else:
        um = sorted(set(m6))
        g = {m: i % nfold for i, m in enumerate(um)}
        gg = np.array([g[m] for m in m6])
        parts = [(gg != k, gg == k) for k in range(nfold)]
    rng = np.random.default_rng(seed)
    for tr_m, te_m in parts:
        tr = tr_m & c6
        Xtr, ytr = X6[tr], (l6[tr] == 2).astype(int)
        ex = extra(tr, rng)
        if ex is not None:
            Xtr = np.vstack([Xtr, ex[0]])
            ytr = np.concatenate([ytr, ex[1]])
        m = mk(kind)
        m.fit(Xtr, ytr)
        p[te_m] = m.predict_proba(X6[te_m])[:, 1]
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


N44 = int(c4.sum())
ARMS = {
    "6bba only (SHIPPED)": lambda tr, rng: None,
    "+ all 44b6 (JOINT)": lambda tr, rng: (X4[c4], (l4[c4] == 2).astype(int)),
    "(A) + bootstrap of ITSELF": lambda tr, rng: (
        X6[tr][rng.integers(0, int(tr.sum()), N44)],
        (l6[tr] == 2).astype(int)[rng.integers(0, int(tr.sum()), N44)]),
    "(B) + 6bba NEGATIVES only": lambda tr, rng: (
        X6[tr & (l6 == 1)][rng.integers(0, int((tr & (l6 == 1)).sum()), N44)],
        np.zeros(N44, int)),
    "(B2) + 44b6 NEGATIVES only": lambda tr, rng: (
        X4[c4 & (l4 == 1)], np.zeros(int((c4 & (l4 == 1)).sum()), int)),
}
print(f"6bba 299 counted / 50 TP; 44b6 adds {N44} counted / "
      f"{int((l4 == 2).sum())} TP, {int((c4 & (l4 == 1)).sum())} FP")
print(f"\n{'arm':<28} {'TP':>4} {'FP':>5} {'div_J':>8} {'paired d':>10} "
      f"{'90% interval':>22}  distinguishable")
B = {}
for nm, ex in ARMS.items():
    p = oof6(ex)
    B[nm] = boot(p)
    J, tp, fp = J6(p)
    if nm == "6bba only (SHIPPED)":
        print(f"{nm:<28} {tp:>4} {fp:>5} {J:>8.4f} {'--':>10} {'--':>22}")
        continue
    d = B[nm] - B["6bba only (SHIPPED)"]
    lo, hi = np.quantile(d, .05), np.quantile(d, .95)
    print(f"{nm:<28} {tp:>4} {fp:>5} {J:>8.4f} {np.mean(d):>+10.4f} "
          f"[{lo:>+7.4f}, {hi:>+7.4f}]  {'YES' if (lo > 0 or hi < 0) else 'no'}")

print("\n(C) the fold split, and (D) the model -- JOINT against SHIPPED each time")
print(f"  {'setting':<28} {'shipped':>9} {'joint':>9} {'paired d':>10} {'90% interval':>22}")
for nfold in (2, 3, 4):
    for kind in ("std", "deep", "shallow"):
        if nfold != 2 and kind != "std":
            continue
        ps = oof6(ARMS["6bba only (SHIPPED)"], nfold, kind)
        pj = oof6(ARMS["+ all 44b6 (JOINT)"], nfold, kind)
        bs, bj = boot(ps), boot(pj)
        d = bj - bs
        lo, hi = np.quantile(d, .05), np.quantile(d, .95)
        print(f"  {f'{nfold}-fold, {kind}':<28} {J6(ps)[0]:>9.4f} {J6(pj)[0]:>9.4f} "
              f"{np.mean(d):>+10.4f} [{lo:>+7.4f}, {hi:>+7.4f}]")
print(f"\ndone in {time.time() - t0:.1f}s")
