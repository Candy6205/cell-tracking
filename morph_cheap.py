"""morph_cheap.py -- can morphology be had WITHOUT masks, and therefore
without the detector?

biohub-morph-probe settled two things. Our `regionprops` on the cached
masks reproduces HOCT's `create_graph` at 0.000e+00 on all five columns,
so the extractor is ours. And GT-matched detections are 28% bigger and
31% brighter than unmatched ones -- the first signal this project has
ever had against over-detection, where N/N_true runs to 1.30.

The blocker is cost. Masks need the detection heatmap, the heatmap needs
the detector, and that is 65 s/dataset of GPU. Across 199 datasets, 3.6 h.

But `cache_train._refine` already cuts a small crop around every detection
to compute an intensity-weighted centroid, and reading raw zarr is CPU
work. If crop statistics track the mask-based ones, morphology becomes
free on every dataset with no detector pass at all.

The user stopped an earlier version of exactly this proposal, correctly:
computing morphology a new way while asking a new question makes a
negative result unattributable. That objection is now satisfied -- there
is a VERIFIED reference to align against, and this run only asks whether
the cheap version agrees with it.

The judgement is different from morph_probe's, and deliberately weaker. A
crop is a box; a watershed region has a shape. They cannot be equal. What
matters is whether they RANK cells the same way, so this reports
correlation, not maximum absolute error.

Also fixed here: morph_probe step 2b drew 114 pairs of which 111 were
true, because it required BOTH endpoints to be GT-matched and sparse GT
means the only matched cell near a matched cell is its own successor.
Three negatives is no measurement. Negatives are now "matched parent, any
candidate that is not its GT successor", the candidate need not be
matched.

CPU, no quota, no detector, one dataset.
"""

SCRIPT_VERSION = "2026-08-25f-morph-cheap"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.measure import regionprops_table

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS if Path(c, "viz_cells.py").exists()),
             None)
assert _code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, _code)
from viz_cells import read_geff, geff_path                     # noqa: E402

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
MATCH_UM = 5.0
GATE_UM = 15.0
CROP_R = 3          # crop half-width in POOLED voxels (1.625 um each)

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/"
             "biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
assert ROOT is not None, "competition not attached"

hits = glob.glob("/kaggle/input/**/hoct_masks.npz", recursive=True)
assert hits, "attach biohub-hoct-gate for hoct_masks.npz"
z = np.load(hits[0])
labels = z["labels"]
POOLED = np.asarray(z["pooled"], float)
NAME = str(z["name"][0])
T = labels.shape[0]
print(f"masks {labels.shape}; dataset {NAME}", flush=True)


def _pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm(p):
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def _read_meta(zp):
    import json as _j
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = _j.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


def _load_volume(zp, t, meta):
    try:
        import zarr
        return np.asarray(zarr.open(str(zp), mode="r")["0"][t])
    except Exception:
        import blosc2
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(
            meta["shape"][1:])


zp = Path(ROOT) / "train" / f"{NAME}.zarr"
meta = _read_meta(zp)
images = np.stack([_norm(_pool_xy(_load_volume(zp, t, meta)))
                   for t in range(T)])
print(f"images {images.shape}", flush=True)


def crop_stats(img, zyx, r=CROP_R):
    """What `_refine`'s crop can say about a cell, with no mask at all.

    Background is the crop minimum, as _refine already does. `mass` is the
    background-subtracted sum -- the natural size-times-brightness proxy --
    and `r_gyr` is the intensity-weighted radius of gyration, the size
    proxy that does not need a threshold."""
    Z, Y, X = img.shape
    z0, y0, x0 = (int(round(v)) for v in zyx)
    a0, a1 = max(0, z0 - r), min(Z, z0 + r + 1)
    b0, b1 = max(0, y0 - r), min(Y, y0 + r + 1)
    c0, c1 = max(0, x0 - r), min(X, x0 + r + 1)
    crop = img[a0:a1, b0:b1, c0:c1].astype(np.float32)
    if crop.size == 0:
        return None
    bg = float(crop.min())
    w = np.clip(crop - bg, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return None
    zz, yy, xx = np.mgrid[a0:a1, b0:b1, c0:c1]
    cz = float((zz * w).sum() / s)
    cy = float((yy * w).sum() / s)
    cx = float((xx * w).sum() / s)
    d2 = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_gyr = float(np.sqrt((d2 * w).sum() / s))
    half = w >= 0.5 * float(w.max())
    return {
        "mass": s,
        "r_gyr": r_gyr,
        "n_half": float(half.sum()),
        "c_mean": float(crop.mean()),
        "c_max": float(crop.max()),
        "c_std": float(crop.std()),
    }


# ---- the verified reference, and the cheap proxy, on the same nodes -----
PROPS = ("label", "centroid", "equivalent_diameter_area",
         "intensity_mean", "intensity_max", "intensity_std")
ref, cheap, tvec, pos = [], [], [], []
for t in range(T):
    if labels[t].max() == 0:
        continue
    tb = regionprops_table(labels[t], intensity_image=images[t],
                           properties=PROPS)
    for i in range(len(tb["label"])):
        c = (float(tb["centroid-0"][i]), float(tb["centroid-1"][i]),
             float(tb["centroid-2"][i]))
        st = crop_stats(images[t], c)
        if st is None:
            continue
        ref.append((float(tb["equivalent_diameter_area"][i]),
                    float(tb["intensity_mean"][i]),
                    float(tb["intensity_max"][i]),
                    float(tb["intensity_std"][i])))
        cheap.append(st)
        tvec.append(t)
        pos.append(c)
ref = np.asarray(ref)
tvec = np.asarray(tvec, int)
pos = np.asarray(pos)
print(f"\n{len(ref)} nodes with both a mask measurement and a crop "
      f"measurement", flush=True)

print("\n=== DOES THE CROP TRACK THE MASK? (Spearman, ranking only) ===",
      flush=True)
print("  A crop is a box and a watershed region has a shape, so these "
      "cannot be equal -- the question is whether they order cells the "
      "same way.", flush=True)
REF_NAMES = ["diam", "i_mean", "i_max", "i_std"]
best = {}
for ri, rn in enumerate(REF_NAMES):
    line = []
    for ck in ("mass", "r_gyr", "n_half", "c_mean", "c_max", "c_std"):
        v = np.asarray([c[ck] for c in cheap], float)
        rho = float(spearmanr(ref[:, ri], v).statistic)
        line.append((abs(rho), rho, ck))
    line.sort(reverse=True)
    best[rn] = line[0]
    pretty = "  ".join(f"{ck} {rho:+.3f}" for _, rho, ck in
                       sorted(line, key=lambda x: x[2]))
    print(f"  {rn:<8} {pretty}", flush=True)
print("\n  best proxy per reference column:", flush=True)
for rn, (a, rho, ck) in best.items():
    verdict = ("usable" if a >= 0.90 else
               "weak" if a >= 0.75 else "NOT a proxy")
    print(f"    {rn:<8} <- {ck:<7} rho {rho:+.3f}   {verdict}", flush=True)

usable = all(v[0] >= 0.75 for v in best.values())
print(f"\n  VERDICT: the crop route is "
      f"{'VIABLE' if usable else 'NOT VIABLE'} as a mask-free stand-in. "
      f"Viable means morphology costs one CPU pass over raw zarr on all "
      f"199 datasets instead of 3.6 h of detector GPU.", flush=True)


# ---- 2b, with the negative class fixed ----------------------------------
gt = read_geff(geff_path(Path(ROOT), "train", NAME))
g_t = np.asarray(gt["t"], int)
g_pos = np.stack([np.asarray(gt[c], float)
                  for c in ("z", "y", "x")], axis=1) * SCALE
row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
gt_edges = set()
for a, b in gt["edges"]:
    ra, rb = row_of.get(int(a)), row_of.get(int(b))
    if ra is not None and rb is not None:
        gt_edges.add((ra, rb))
gt_by_t = defaultdict(list)
for i in range(len(g_t)):
    gt_by_t[int(g_t[i])].append(i)

pos_um = pos * POOLED[None, :]
match = {}
for t in range(T):
    idx = gt_by_t.get(t, [])
    sel = np.where(tvec == t)[0]
    if not idx or not len(sel):
        continue
    tree = cKDTree(g_pos[idx])
    d, gi = tree.query(pos_um[sel])
    for k, s in enumerate(sel):
        if d[k] <= MATCH_UM:
            match[int(s)] = idx[int(gi[k])]

print(f"\n=== 2b REDONE: morphology change across an edge ===", flush=True)
print("  morph_probe required BOTH endpoints matched and got 3 negatives. "
      "Negatives here are: matched parent, any candidate in the gate that "
      "is NOT its GT successor -- the candidate need not be matched.",
      flush=True)
y, d_diam, d_int, d_mass = [], [], [], []
for t in range(T - 1):
    a_sel = np.where(tvec == t)[0]
    b_sel = np.where(tvec == t + 1)[0]
    if not len(a_sel) or not len(b_sel):
        continue
    tree = cKDTree(pos_um[b_sel])
    for ai in a_sel:
        if int(ai) not in match:
            continue                      # parent must be a real cell
        for bj in tree.query_ball_point(pos_um[ai], GATE_UM):
            bi = int(b_sel[bj])
            lab = 1 if (bi in match and
                        (match[int(ai)], match[bi]) in gt_edges) else 0
            y.append(lab)
            da, db = ref[int(ai), 0], ref[bi, 0]
            d_diam.append(abs(db - da) / max(da, 1e-6))
            d_int.append(abs(ref[bi, 1] - ref[int(ai), 1]))
            ma = cheap[int(ai)]["mass"]
            mb = cheap[bi]["mass"]
            d_mass.append(abs(mb - ma) / max(ma, 1e-6))
y = np.asarray(y)
print(f"  {len(y)} pairs, {int(y.sum())} true "
      f"({y.mean() if len(y) else float('nan'):.3f} positive)", flush=True)
if len(y) > 50 and y.sum() >= 10 and (len(y) - y.sum()) >= 10:
    from sklearn.metrics import roc_auc_score
    for nm, v in (("|d diam| / diam", np.asarray(d_diam)),
                  ("|d intensity|", np.asarray(d_int)),
                  ("|d mass| / mass  (crop-only)",
                   np.asarray(d_mass))):
        a = float(np.median(v[y == 1]))
        b = float(np.median(v[y == 0]))
        auc = roc_auc_score(y, -v)
        print(f"  {nm:<28} true {a:.4f}  false {b:.4f}  "
              f"ratio {a / b if b else float('nan'):.2f}   "
              f"AUC(-v) {auc:.4f}", flush=True)
    print("  AUC near 0.5 means the change does not separate a real link "
          "from a near miss and morphology is not an edge feature, "
          "whatever it does for detection filtering.", flush=True)
else:
    print("  still too few of one class to judge", flush=True)

print("\nMORPH CHEAP DONE", flush=True)
