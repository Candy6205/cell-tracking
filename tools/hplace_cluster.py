"""hplace_cluster -- do the four F4 heatmaps put a cell's peak in the same
place, or in different places? Reads bench_csv/hplace/hplace_sites.csv (v2:
per-map peak coordinates), chains the four maps' peaks at 2 um (single
linkage) and reports, per kind: how many clusters they form, where the
annotated centroid A sits relative to the clusters, and whether the mean
map's peak coincides with one of them or lands between them.

    py tools/hplace_cluster.py
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

d = pd.read_csv("bench_csv/hplace/hplace_sites.csv")
PS = 1.625                       # pooled grid, isotropic um per voxel
M4 = ["b0", "b1", "b2", "dc"]


def P(o, m):
    return np.array([o[f"pz_{m}"], o[f"py_{m}"], o[f"px_{m}"]]) * PS


for kind in ("swap", "ctrl"):
    s = d[d.kind == kind].dropna(subset=[f"pz_{m}" for m in M4 + ["f4"]])
    ncl, f4_major, f4_between, A_near, A_far, f4_nearest = [], [], [], [], [], []
    for _, o in s.iterrows():
        pts = np.array([P(o, m) for m in M4])
        pf = P(o, "f4")
        pA = np.array([o.gz, (o.gy - 1.5) / 4, (o.gx - 1.5) / 4]) * PS
        lab = fcluster(linkage(pts, "single"), 2.0, "distance")
        k = lab.max()
        ncl.append(k)
        f4_nearest.append(np.linalg.norm(pts - pf, axis=1).min())
        if k >= 2:
            cents = np.array([pts[lab == c].mean(0) for c in range(1, k + 1)])
            sizes = np.array([(lab == c).sum() for c in range(1, k + 1)])
            dA = np.sort(np.linalg.norm(cents - pA, axis=1))
            A_near.append(dA[0])
            A_far.append(dA[1])
            df4 = np.linalg.norm(cents - pf, axis=1)
            f4_major.append(sizes[np.argmin(df4)] == sizes.max())
            f4_between.append(df4.min() > 1.5)
    ncl = np.array(ncl)
    print(f"{kind} n={len(s)}: the four peaks form 1 cluster {np.mean(ncl == 1):.2f}, 2 clusters "
          f"{np.mean(ncl == 2):.2f}, 3+ {np.mean(ncl >= 3):.2f} (single linkage, 2 um)")
    print(f"   f4's peak to the nearest of the four: median {np.median(f4_nearest):.2f} um, "
          f"> 1.5 um share {np.mean(np.array(f4_nearest) > 1.5):.2f}")
    if A_near:
        print(f"   multi-cluster sites: A to the nearest cluster median {np.median(A_near):.2f} um, to the other "
              f"{np.median(A_far):.2f}; f4's peak in the majority cluster {np.mean(f4_major):.2f}, "
              f"f4's peak > 1.5 um from every cluster {np.mean(f4_between):.2f}")
