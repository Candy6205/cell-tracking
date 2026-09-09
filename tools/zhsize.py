"""zhsize -- are the 3 zebrahub crops that never scored a biased loss?

`zhpara` v3 measured hpara on 38 of 41 crops (+0.0011 adj_J, [+0.0005,
+0.0017], P 0.998) and was SIGKILLed on crop 39. `zhpara4` then established
why: `compute_metric` allocates a DENSE N_pred x N_gt cost matrix, and on
`ZSNS001_z2y2x7` that is 40,936 x 54,073 x 8 B = 17.7 GB on a box with 31.7
GB available. The three crops that never scored are the three LARGEST
(39/41, 40/41, 41/41 by cache size).

So the missing crops are not missing at random -- they are missing BY SIZE.
That is only a problem for the hpara verdict if the effect depends on size.
This checks, on the 38 that did score:

  * dense matrix size per crop, from the CSV's own n_gt and num_pred_nodes,
    to place the three lost ones inside or outside the measured range;
  * Spearman correlation between crop size and the per-crop hpara delta;
  * the delta recomputed on the largest third alone, against the smallest.

    py tools/zhsize.py
"""
import csv
import sys

import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "bench_csv/zhpara3/zhpara.csv"
# From zhpara4's rss.log -- the only direct measurement of a lost crop.
LOST = {"ZSNS001_z2y2x7": (40936, 54073)}

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["crop"], {})[r["arm"]] = r
crops = sorted(k for k, v in by.items() if len(v) == 2)
arms = sorted({r["arm"] for r in rows})
print(f"{len(rows)} rows, {len(crops)} crops with both arms, arms {arms}")
assert len(arms) == 2, arms
REF, HP = arms if arms[0] in ("REF", "BASE") else arms[::-1]
print(f"  control = {REF!r}, treatment = {HP!r}")


def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float("nan")


size = np.array([f(by[c][REF], "n_gt") * f(by[c][REF], "num_pred_nodes") * 8
                 / 1e9 for c in crops])
d = np.array([f(by[c][HP], "adj_edge_jaccard")
              - f(by[c][REF], "adj_edge_jaccard") for c in crops])
w = np.array([f(by[c][REF], "edge_tp") + f(by[c][REF], "edge_fp")
              + f(by[c][REF], "edge_fn") for c in crops])

print(f"\n=== DENSE COST MATRIX PER CROP (n_gt x n_pred x 8 B) ===")
o = np.argsort(-size)
print(f"  {'crop':<20}{'n_gt':>9}{'n_pred':>9}{'dense GB':>10}{'d adj_J':>10}")
for i in o[:5]:
    print(f"  {crops[i]:<20}{f(by[crops[i]][REF], 'n_gt'):>9.0f}"
          f"{f(by[crops[i]][REF], 'num_pred_nodes'):>9.0f}"
          f"{size[i]:>10.2f}{d[i]:>+10.5f}")
print(f"  ... {len(crops) - 5} more, smallest {size.min():.2f} GB")
print(f"  the 38 that SCORED span {size.min():.2f} - {size.max():.2f} GB")
for name, (npred, ngt) in LOST.items():
    print(f"  {name} (LOST): {ngt:,} x {npred:,} = "
          f"{npred * ngt * 8 / 1e9:.1f} GB  <- "
          f"{npred * ngt * 8 / 1e9 / max(1e-9, size.max()):.0f}x the largest "
          f"crop that finished")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


print(f"\n=== DOES THE EFFECT DEPEND ON SIZE? ===")
print(f"  Spearman(size, d adj_J) over {len(crops)} crops: "
      f"{spearman(size, d):+.3f}")
print(f"  Spearman(size, |d| adj_J)                      : "
      f"{spearman(size, np.abs(d)):+.3f}")
k = len(crops) // 3
big, small = o[:k], o[-k:]


def wavg(idx):
    return float((d[idx] * w[idx]).sum() / max(1e-9, w[idx].sum()))


print(f"  weighted d adj_J, largest {k} crops : {wavg(big):+.5f}")
print(f"  weighted d adj_J, smallest {k} crops: {wavg(small):+.5f}")
print(f"  weighted d adj_J, all {len(crops)}        : {wavg(np.arange(len(crops))):+.5f}")
print(f"\n  Reading: if the correlation is near zero and the two thirds agree "
      f"in sign,\n  the three lost crops are missing for a mechanical reason "
      f"and the 38-crop\n  verdict carries. If the largest third disagrees, "
      f"the verdict is about small\n  crops only and should say so.")
