"""gatelab10 -- is the joint gate's gain INFORMATION or REGULARISATION?

gatelab8 ruled out row count (a bootstrap of the same rows is worse) and
label balance. gatelab9 refuted the density hypothesis (the advantage grows
when the two density features are removed). What is left is the possibility
that ANY 67 distinct rows from a different distribution would regularise a
20-feature model fitted on 299 rows with 50 positives, whatever their labels.

Two decisive controls:
  (E) the SAME 44b6 rows, labels SHUFFLED. If the gain survives, the labels
      carry nothing and it is regularisation.
  (F) 67 rows of 6bba features with each COLUMN independently permuted, given
      44b6's label mix (6 positive, 61 negative). Pure noise rows.
Both are averaged over 20 shuffles so a single lucky permutation cannot
decide it.
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
X44, y44 = X4[c4], (l4[c4] == 2).astype(int)
N44 = len(y44)


def oof(extra):
    p = np.full(len(l6), np.nan)
    for k in (0, 1):
        tr = (f6 != k) & c6
        Xtr, ytr = X6[tr], (l6[tr] == 2).astype(int)
        if extra is not None:
            Xtr = np.vstack([Xtr, extra[0]])
            ytr = np.concatenate([ytr, extra[1]])
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10, random_state=0)
        m.fit(Xtr, ytr)
        p[(f6 == k)] = m.predict_proba(X6[f6 == k])[:, 1]
    return p


def J6(p, thr=0.03):
    keep = np.isfinite(p) & (p > thr)
    tp, fp = int((l6[keep] == 2).sum()), int((l6[keep] == 1).sum())
    return tp / max(1, tp + fp + (N6 - tp))


def boot(p, thr=0.03):
    out = []
    for d in draws:
        ii = np.concatenate([idx_of[m] for m in d])
        ngt = sum(gt6[m] for m in d)
        keep = p[ii] > thr
        tp, fp = int((l6[ii][keep] == 2).sum()), int((l6[ii][keep] == 1).sum())
        out.append(tp / max(1, tp + fp + (ngt - tp)))
    return np.array(out)


b_ship = boot(oof(None))
b_join = boot(oof((X44, y44)))
print(f"{'arm':<38} {'div_J':>8} {'paired d vs shipped':>21} {'90% interval':>22}")
print(f"{'6bba only (SHIPPED)':<38} {J6(oof(None)):>8.4f}")
d = b_join - b_ship
print(f"{'+ all 44b6 (JOINT)':<38} {J6(oof((X44, y44))):>8.4f} {np.mean(d):>+21.4f} "
      f"[{np.quantile(d, .05):>+7.4f}, {np.quantile(d, .95):>+7.4f}]")

rng = np.random.default_rng(7)
for tag, gen in (
    ("(E) 44b6 rows, labels SHUFFLED",
     lambda r: (X44, y44[r.permutation(N44)])),
    ("(F) noise rows, 44b6 label mix",
     lambda r: (np.column_stack([X6[c6][r.permutation(int(c6.sum())), j][:N44]
                                 for j in range(X6.shape[1])]), y44.copy())),
):
    ds, Js = [], []
    for _ in range(20):
        p = oof(gen(rng))
        Js.append(J6(p))
        ds.append(np.mean(boot(p) - b_ship))
    print(f"{tag:<38} {np.mean(Js):>8.4f} {np.mean(ds):>+21.4f} "
          f"[{np.min(ds):>+7.4f}, {np.max(ds):>+7.4f}]  (mean of 20 shuffles, "
          f"range not an interval)")
print("\n  If (E) and (F) sit near zero while JOINT is +0.036, the 44b6 LABELS are"
      "\n  what carry it and the gain is information, not regularisation.")
print(f"\ndone in {time.time() - t0:.1f}s")
