"""gatelab5 -- self-contained (no exec of gatelab.py, no full-threshold scan):
what the 47 false forks that SURVIVE the shipped cut look like, and what the 8
true forks it kills look like."""
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
CSV = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
       "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/forks_f4.csv")
lines = open(CSV).read().strip().split("\n")
hdr = lines[0].split(",")
FEATS = hdr[5:]
rows = [ln.split(",") for ln in lines[1:]]
movie = np.array([r[0] for r in rows])
fold = np.array([int(r[1]) for r in rows])
lab = np.array([int(r[4]) for r in rows])
X = np.array([[float(v) for v in r[5:]] for r in rows])
counted = lab >= 1
print(f"loaded {len(lab)} forks in {time.time() - t0:.1f}s", flush=True)

p = np.full(len(lab), np.nan)
for k in (0, 1):
    tr = (fold != k) & counted
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                       min_samples_leaf=10, random_state=0)
    m.fit(X[tr], (lab[tr] == 2).astype(int))
    p[fold == k] = m.predict_proba(X[fold == k])[:, 1]
print(f"gate fitted in {time.time() - t0:.1f}s", flush=True)

keep = p > 0.03
surv = keep & (lab == 1)
kill = ~keep & (lab == 2)
kept = keep & (lab == 2)
print(f"\nAT THE SHIPPED CUT p>0.03: {int(kept.sum())} true forks kept, "
      f"{int(kill.sum())} killed, {int(surv.sum())} false forks survive "
      f"(of {int((lab == 1).sum())})", flush=True)

c = Counter(movie[surv])
print(f"\n--- the {int(surv.sum())} surviving false forks sit in {len(c)} movies "
      f"(of 128) ---")
print("   top 8: " + ", ".join(f"{m[5:]}:{n}" for m, n in c.most_common(8)))
big = {m: n for m, n in c.items() if n >= 3}
print(f"   movies with >=3 of them: {len(big)}, carrying {sum(big.values())} "
      f"of {int(surv.sum())}")
ct = Counter(movie[kept])
print(f"   movies that carry a surviving false fork AND a kept true one: "
      f"{sum(1 for m in c if m in ct)}")

print("\n--- feature medians ---")
print(f"   {'feature':<9} {'kept TP':>9} {'surv FP':>9} {'killed TP':>10} {'all FP':>9}")
for j, f in enumerate(FEATS):
    col = X[:, j]

    def md(m, _c=col):
        v = _c[m & np.isfinite(_c)]
        return np.median(v) if len(v) else np.nan
    print(f"   {f:<9} {md(kept):>9.3f} {md(surv):>9.3f} {md(kill):>10.3f} "
          f"{md(lab == 1):>9.3f}")

src = X[:, FEATS.index("src")].astype(int)
print("\n--- by source, and where the probabilities sit ---")
for s, nm in ((1, "patch"), (0, "ilp")):
    print(f"   {nm:<6} surviving FP {int((surv & (src == s)).sum()):>3}   "
          f"kept TP {int((kept & (src == s)).sum()):>3}")
print(f"   surviving FP p: median {np.median(p[surv]):.3f}  q75 "
      f"{np.quantile(p[surv], .75):.3f}  max {p[surv].max():.3f}")
print(f"   kept TP      p: median {np.median(p[kept]):.3f}  q25 "
      f"{np.quantile(p[kept], .25):.3f}  min {p[kept].min():.3f}")
print(f"   killed TP    p: median {np.median(p[kill]):.4f}  max {p[kill].max():.4f}")

print("\n--- the 8 true forks the cut kills, one row each ---")
show = ["src", "sc_q", "sc_p", "sc_c1", "dv", "d_s", "d_pq", "p_q", "dens10", "len_q"]
js = [FEATS.index(f) for f in show]
print("   " + f"{'movie':>10} {'p':>7} " + " ".join(f"{f:>7}" for f in show))
for i in np.where(kill)[0]:
    print(f"   {movie[i][5:]:>10} {p[i]:>7.4f} "
          + " ".join(f"{X[i, j]:>7.3f}" for j in js))
print(f"\n   for comparison, the median KEPT true fork: "
      + " ".join(f"{np.nanmedian(X[kept][:, j]):.3f}" for j in js))
print(f"   and the median SURVIVING false fork:        "
      + " ".join(f"{np.nanmedian(X[surv][:, j]):.3f}" for j in js))
print(f"\ndone in {time.time() - t0:.1f}s")
