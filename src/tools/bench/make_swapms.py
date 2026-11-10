"""make_swapms.py -> swapms.py: mean shift on the raw image from every detection of a pair -- do the two walk together?

WHY (2026-09-23). The swap FN + FP pair is the scorer's one-to-one match: a (3.6 um from the annotated centroid A)
takes A, the linked b (4.6 um) is left unmatched. swapimg: the background-subtracted intensity centroid within
4 um of A sits 0.37 um from A (99% within 1 um) -- the annotation is the nucleus' mass centre and it is stable.
Two detections on one nucleus should both walk to that centre under mean shift (a Gaussian-weighted intensity
centroid re-centred until it stops); two detections on two nuclei should stop on their own centres. The merge
trigger is then "the two walked together" -- no learned gate, no profile dip. Priced on the scored population:
the cost falls on pairs with an annotated end (both-annotated near pairs ~2,187, one-end ~9,600), the gain on
the ~1,643 swaps, so the convergence rate on true two-cell pairs (f) against on swap pairs (g) decides it.
Sites: swapdip's table (swap 1,643 / gtpair 511 / rand 1,348 / rand_z1 1,850) + ~1,900 control detections
(shipped primaries with a GT node within 5 um, 15 per movie). Bandwidths 1.5 / 2.0 / 3.0 um, 10 steps max.
Output swapms_sites.csv + the summary. Attach: the competition data + faustine649/cell-tracking. CPU, no internet.
"""
import base64
import gzip
import io
import pathlib
import re

import numpy as np
import pandas as pd

N_CTL_PER_MOVIE, SEED = 15, 0
ln = next(l_ for l_ in open("swapdip.py", encoding="utf-8").read().splitlines() if l_.startswith("SITES_B64 = "))
blob = re.search(r'"([^"]+)"', ln).group(1)
sites = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(blob)).decode()))
print(sites.kind.value_counts().to_dict())
# the swap rows need A (gz, gy, gx): take them from hsaddle's table via swapimg's embedded copy
ln2 = open("swapimg.py", encoding="utf-8").read().splitlines()[69]
emb = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(re.search(r'"([^"]+)"', ln2).group(1))).decode()))
sw = emb[emb.kind == "swap"][["dataset", "t", "az", "ay", "ax", "bz", "by", "bx", "gz", "gy", "gx"]].round(3)
m = sites[sites.kind == "swap"].round(3).merge(sw, on=["dataset", "t", "az", "ay", "ax", "bz", "by", "bx"], how="left")
assert m.gz.notna().all(), int(m.gz.isna().sum())
sites = sites.copy()
for c in ("gz", "gy", "gx"):
    sites[c] = np.nan
sites.loc[sites.kind == "swap", ["gz", "gy", "gx"]] = m[["gz", "gy", "gx"]].values

rng = np.random.default_rng(SEED)
rows = []
for f in sorted(pathlib.Path("bench_csv/divmiss/divmiss_prims").glob("*.npz")):
    z = np.load(f)
    ok = np.where(z["d_gt"] <= 5.0)[0]
    if len(ok) == 0:
        continue
    for k in rng.choice(ok, size=min(N_CTL_PER_MOVIE, len(ok)), replace=False):
        rows.append(("ctl", f.stem, int(z["t"][k]), *z["zyx"][k], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
ctl = pd.DataFrame(rows, columns=sites.columns)
sites = pd.concat([sites, ctl], ignore_index=True)
print(sites.kind.value_counts().to_dict())
csv = sites.to_csv(index=False, float_format="%.3f", lineterminator="\n")
b64 = base64.b64encode(gzip.compress(csv.encode())).decode()

TEMPLATE = r'''"""swapms: mean shift on the raw image from each detection of a pair; do two detections on one nucleus walk together?

For every site (swap a/b, gtpair a/b, rand a/b, rand_z1 a/b, ctl a): a crop (+-4 planes, +-24 voxels) around the
pair's midpoint (the point itself for ctl), background = the crop's 20th percentile subtracted and clipped at 0;
from each detection, repeat: new position = the intensity-weighted, Gaussian-weighted (bandwidth h um) centroid of
the crop around the current position; stop when the move < 0.2 um or after 10 steps. Bandwidths 1.5 / 2 / 3 um.
Recorded per site and bandwidth: the moves, a-b distance before / after, and the distance to the nearest annotated
centroid at that frame before / after (swap: to A). Output swapms_sites.csv. CPU, no internet.
"""
import base64
import gzip
import io
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-23a-swapms"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, open_image, read_geff, zarr_path   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SCALE = np.array([1.625, 0.40625, 0.40625])
HZ, HY = 4, 24
BANDS = (1.5, 2.0, 3.0)
MAX_STEPS, TOL_UM, CONV_UM = 10, 0.2, 2.0
SITES_B64 = "__B64__"


def mean_shift(vol, grid_um, p_um, h):
    """vol: background-subtracted crop (>= 0); grid_um: (N, 3) voxel positions in um; p_um: start (3,) um.
    Returns (final position um, steps, total move um)."""
    p = p_um.copy()
    for k in range(MAX_STEPS):
        d2 = ((grid_um - p) ** 2).sum(1)
        w = vol * np.exp(-0.5 * d2 / (h * h))
        s = w.sum()
        if s <= 1e-9:
            return p, k, float(np.linalg.norm(p - p_um))
        q = (w[:, None] * grid_um).sum(0) / s
        mv = float(np.linalg.norm(q - p))
        p = q
        if mv < TOL_UM:
            return p, k + 1, float(np.linalg.norm(p - p_um))
    return p, MAX_STEPS, float(np.linalg.norm(p - p_um))


KEYS = ["h", "d_ab0", "d_ab1", "mv_a", "mv_b", "st_a", "st_b", "ga0", "ga1", "gb0", "gb1", "gm1", "same_gt"]

if __name__ == "__main__":
    t_all = time.time()
    sites = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(SITES_B64)).decode()))
    print(f"{len(sites)} sites: {sites.kind.value_counts().to_dict()}", flush=True)
    fh = open(OUT / "swapms_sites.csv", "w", encoding="utf-8", newline="\n")
    fh.write("kind,dataset,t," + ",".join(KEYS) + "\n")
    n_done = 0
    for k_, (name, grp) in enumerate(sites.groupby("dataset", sort=True)):
        t0 = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        T, Z, Y, X = img.shape
        gt = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gt["t"], float).round().astype(int)
        g_um = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
        trees = {tt: cKDTree(g_um[g_t == tt]) for tt in np.unique(g_t)}

        def near_gt(tt, p):
            if tt not in trees:
                return np.nan, -1
            d, i = trees[tt].query(p)
            return float(d), int(i)
        for _, r in grp.iterrows():
            t = int(r.t)
            a = np.array([r.az, r.ay, r.ax], float)
            b = np.array([r.bz, r.by, r.bx], float) if np.isfinite(r.bz) else None
            mid = 0.5 * (a + b) if b is not None else a
            z0, z1 = max(0, int(round(mid[0])) - HZ), min(Z, int(round(mid[0])) + HZ + 1)
            y0, y1 = max(0, int(round(mid[1])) - HY), min(Y, int(round(mid[1])) + HY + 1)
            x0, x1 = max(0, int(round(mid[2])) - HY), min(X, int(round(mid[2])) + HY + 1)
            crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1], dtype=np.float32)
            bg = float(np.percentile(crop, 20.0))
            vol = np.clip(crop - bg, 0, None).ravel()
            zz, yy, xx = np.indices(crop.shape)
            grid_um = (np.stack([zz.ravel() + z0, yy.ravel() + y0, xx.ravel() + x0], 1).astype(float)) * SCALE
            a_um = a * SCALE
            b_um = b * SCALE if b is not None else None
            A_um = np.array([r.gz, r.gy, r.gx], float) * SCALE if np.isfinite(r.gz) else None
            for h in BANDS:
                pa, sa, mva = mean_shift(vol, grid_um, a_um, h)
                o = dict(h=h, mv_a=mva, st_a=sa, d_ab0=np.nan, d_ab1=np.nan, mv_b=np.nan, st_b=np.nan,
                         gb0=np.nan, gb1=np.nan, gm1=np.nan, same_gt=np.nan)
                if A_um is not None:
                    o["ga0"], o["ga1"] = float(np.linalg.norm(a_um - A_um)), float(np.linalg.norm(pa - A_um))
                else:
                    o["ga0"], ia0 = near_gt(t, a_um)
                    o["ga1"], ia1 = near_gt(t, pa)
                if b_um is not None:
                    pb, sb, mvb = mean_shift(vol, grid_um, b_um, h)
                    o.update(mv_b=mvb, st_b=sb, d_ab0=float(np.linalg.norm(a_um - b_um)), d_ab1=float(np.linalg.norm(pa - pb)))
                    if A_um is not None:
                        o["gb0"], o["gb1"] = float(np.linalg.norm(b_um - A_um)), float(np.linalg.norm(pb - A_um))
                        o["gm1"] = float(np.linalg.norm(0.5 * (pa + pb) - A_um))
                    else:
                        o["gb0"], ib0 = near_gt(t, b_um)
                        o["gb1"], ib1 = near_gt(t, pb)
                        o["same_gt"] = int(ia0 == ib0) if ia0 >= 0 else np.nan
                fh.write(f"{r.kind},{name},{t}," + ",".join(f"{o[k]:.4g}" if o[k] == o[k] else "nan" for k in KEYS) + "\n")
            n_done += 1
        fh.flush()
        print(f"  [{k_ + 1}/{sites.dataset.nunique()}] {name}: {len(grp)} sites, {time.time() - t0:.0f}s", flush=True)
        del img
    fh.close()
    d = pd.read_csv(OUT / "swapms_sites.csv")
    print("\n=== SWAPMS: mean shift on the raw image ===", flush=True)
    for h in BANDS:
        dh = d[d.h == h]
        print(f"\n  bandwidth {h} um", flush=True)
        c = dh[dh.kind == "ctl"]
        print(f"    ctl ({len(c)}): distance to the nearest annotated centroid before / after, median {c.ga0.median():.2f} / {c.ga1.median():.2f} um; "
              f"within 2 um {(c.ga0 <= 2).mean():.2f} / {(c.ga1 <= 2).mean():.2f}; moved > 3 um {(c.mv_a > 3).mean():.2f}; steps med {c.st_a.median():.0f}", flush=True)
        for kind in ("gtpair", "swap", "rand", "rand_z1"):
            g = dh[dh.kind == kind]
            if len(g) == 0:
                continue
            conv = g.d_ab1 <= CONV_UM
            line = (f"    {kind:<8} ({len(g)}): a-b before / after median {g.d_ab0.median():.2f} / {g.d_ab1.median():.2f} um; "
                    f"walked together (<= {CONV_UM} um) {conv.mean():.3f}; <= 3 um {(g.d_ab1 <= 3).mean():.3f}; moves a / b median {g.mv_a.median():.2f} / {g.mv_b.median():.2f}")
            if kind == "swap":
                line += (f"; to A after: a {g.ga1.median():.2f}, b {g.gb1.median():.2f}, midpoint {g.gm1.median():.2f} um; "
                         f"together AND midpoint <= 7 um of A {(conv & (g.gm1 <= 7)).mean():.3f}")
            if kind == "gtpair":
                line += (f"; each end to its nearest annotation before / after a {g.ga0.median():.2f} / {g.ga1.median():.2f}, b {g.gb0.median():.2f} / {g.gb1.median():.2f}; "
                         f"ends still on different annotations after {(g.same_gt == 0).mean():.3f}")
            print(line, flush=True)
        f_ = (dh[dh.kind == "gtpair"].d_ab1 <= CONV_UM).mean()
        g_ = (dh[dh.kind == "swap"].d_ab1 <= CONV_UM).mean()
        print(f"    account: g (swap together) {g_:.3f} / f (two-cell together) {f_:.3f} = {g_ / max(f_, 1e-6):.1f}; break-even ~7", flush=True)
    print(f"\nSWAPMS DONE in {(time.time() - t_all) / 60:.1f} min, {n_done} sites", flush=True)
    print("SWAPMS OK", flush=True)
'''
src = TEMPLATE.replace("__B64__", b64)
open("swapms.py", "w", encoding="utf-8", newline="\n").write(src)
print("written swapms.py", src.count("\n"), "lines")
