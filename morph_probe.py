"""morph_probe.py -- step 1: own the morphology extractor before believing
anything it says.

The HOCT tracker is closed (its `similarity` lost to a distance baseline by
0.21 AUC and cannot be fine-tuned). What survived is that `create_graph`
computes per-cell morphology from masks, and those column names --
`equivalent_diameter_area`, `inertia_tensor`, `intensity_min/max/mean/std`
-- are skimage `regionprops` properties. So the morphology route needs
regionprops and the cached masks, not hoct.

But that is a claim, and the user's objection to testing it the fast way
is the right one: computing morphology a NEW way while asking a NEW
question means a negative result cannot be attributed. So this run does
one thing first.

STEP 1, the only thing that gates the rest: run OUR regionprops and HOCT's
create_graph on the SAME cached masks and compare node by node. Match and
we own the extractor and can drop hoct entirely. Mismatch and the rest of
the output is suppressed, because features we cannot reproduce are not
features we can reason about.

STEP 2 runs only if step 1 passes, and asks two things that are answerable
on one dataset:

  2a  do GT-matched detections differ in size or brightness from
      unmatched ones? That is the over-detection problem -- N/N_true runs
      to 1.30 and nothing has ever measured whether the invented nodes
      look different.
  2b  across a TRUE GT edge, how much does a cell's size and brightness
      change, against the same numbers for false candidate pairs? That is
      the miniature of "morphology as an edge-ranker feature", the one
      genuinely new information channel available -- every feature this
      project owns is a coordinate or a detection score.

Not answerable here and deliberately not attempted: whether the mother
shrinks before dividing. 30 frames of one dataset holds far too few
divisions; the run reports the count so the next one can be sized.

CPU, no quota. Masks come from biohub-hoct-gate's cached npz.
"""

SCRIPT_VERSION = "2026-08-25e-morph-probe"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

import glob
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

NP_VER = np.__version__
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hoct",
                    f"numpy=={NP_VER}"], capture_output=True, text=True)
print(f"pip install hoct (numpy pinned {NP_VER}) exit {r.returncode}",
      flush=True)
assert np.__version__ == NP_VER, "numpy moved despite the pin"

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
print(f"masks {labels.shape} from {hits[0]}; dataset {NAME}", flush=True)


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
assert images.shape == labels.shape
print(f"images {images.shape}, range [{images.min():.3f}, "
      f"{images.max():.3f}]", flush=True)

PROPS = ("label", "centroid", "equivalent_diameter_area",
         "intensity_min", "intensity_max", "intensity_mean",
         "intensity_std")


# ---- OUR extractor ------------------------------------------------------
rows = []
for t in range(T):
    if labels[t].max() == 0:
        continue
    tb = regionprops_table(labels[t], intensity_image=images[t],
                           properties=PROPS)
    n = len(tb["label"])
    for i in range(n):
        rows.append({
            "t": t,
            "z": float(tb["centroid-0"][i]),
            "y": float(tb["centroid-1"][i]),
            "x": float(tb["centroid-2"][i]),
            "diam": float(tb["equivalent_diameter_area"][i]),
            "i_min": float(tb["intensity_min"][i]),
            "i_max": float(tb["intensity_max"][i]),
            "i_mean": float(tb["intensity_mean"][i]),
            "i_std": float(tb["intensity_std"][i]),
        })
print(f"\nOUR regionprops: {len(rows)} regions over {T} frames", flush=True)


def _summ(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return (float(np.median(v)), float(np.quantile(v, 0.1)),
            float(np.quantile(v, 0.9))) if len(v) else (np.nan,) * 3


OURS = {k: _summ([r[k] for r in rows])
        for k in ("diam", "i_mean", "i_min", "i_max", "i_std")}


# ---- STEP 1: the anchor -------------------------------------------------
print("\n=== STEP 1: does OUR extractor reproduce HOCT's, node for node? "
      "===", flush=True)
ok = False
try:
    import hoct
    from hoct.features.graph import create_graph
    # v1 MISMATCHED on all four intensity columns while
    # equivalent_diameter_area agreed to 0.000e+00 -- same masks, same 770
    # nodes, so the difference had to be the intensity IMAGE, not the
    # regions. It was: create_graph defaults to normalize_images=True and
    # these images are already normalised by _norm, so HOCT normalised
    # them a second time. Ours read a median i_mean of 0.6807 against its
    # 0.4948. Turn its normalisation off and both read the same array.
    g = create_graph(labels, images=images, distance_threshold=GATE_UM,
                     n_neighbors=5, delta_t=3,
                     normalize_images=False,
                     scale=(1.0,) + tuple(float(v) for v in POOLED))
    keys = ["t", "z", "y", "x", "equivalent_diameter_area",
            "intensity_mean", "intensity_min", "intensity_max",
            "intensity_std"]
    na = g.node_attrs(attr_keys=keys)
    H = {k: np.asarray(na[k].to_list(), float) for k in keys}
    print(f"  HOCT nodes {len(H['t'])}, ours {len(rows)}", flush=True)

    # pair them by (frame, centroid); the mask is identical so the
    # centroids must land on top of each other
    mine = np.array([[r["t"], r["z"], r["y"], r["x"]] for r in rows])
    hs = np.stack([H["t"], H["z"], H["y"], H["x"]], axis=1)
    tree = cKDTree(mine)
    d, j = tree.query(hs)
    paired = d < 1e-3
    print(f"  {int(paired.sum())}/{len(hs)} HOCT nodes sit on one of ours "
          f"(centroid distance < 1e-3 voxels)", flush=True)

    NAMEMAP = {"equivalent_diameter_area": "diam",
               "intensity_mean": "i_mean", "intensity_min": "i_min",
               "intensity_max": "i_max", "intensity_std": "i_std"}
    worst = 0.0
    for hk, ok_ in NAMEMAP.items():
        a = H[hk][paired]
        b = np.array([rows[int(k)][ok_] for k in j[paired]])
        m = np.isfinite(a) & np.isfinite(b)
        e = float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")
        worst = max(worst, e if np.isfinite(e) else 0.0)
        print(f"    {hk:<26} max |ours - HOCT| = {e:.3e}", flush=True)
    ok = bool(paired.mean() > 0.95 and worst < 1e-4)
    print(f"\n  VERDICT: {'MATCH' if ok else 'MISMATCH'} "
          f"(worst {worst:.3e})", flush=True)
except Exception:
    print("  could not run HOCT's extractor for comparison:", flush=True)
    traceback.print_exc()

if not ok:
    print("\n  Step 2 SUPPRESSED. Features we cannot reproduce are not "
          "features we can reason about -- fix the extractor before "
          "asking what the numbers mean.", flush=True)
    print("\n  ours, for the record:", flush=True)
    for k, (m_, lo, hi) in OURS.items():
        print(f"    {k:<8} median {m_:.4f}  p10 {lo:.4f}  p90 {hi:.4f}",
              flush=True)
    print("\nMORPH PROBE DONE", flush=True)
    raise SystemExit(0)


# ---- STEP 2: what the verified features actually say --------------------
gt = read_geff(geff_path(Path(ROOT), "train", NAME))
g_t = np.asarray(gt["t"], int)
g_pos = np.stack([np.asarray(gt[c], float)
                  for c in ("z", "y", "x")], axis=1) * SCALE
row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
gt_edges, kids = set(), defaultdict(list)
for a, b in gt["edges"]:
    ra, rb = row_of.get(int(a)), row_of.get(int(b))
    if ra is not None and rb is not None:
        gt_edges.add((ra, rb))
        kids[ra].append(rb)
n_div = sum(1 for v in kids.values() if len(v) >= 2)
gt_by_t = defaultdict(list)
for i in range(len(g_t)):
    gt_by_t[int(g_t[i])].append(i)

pos_um = np.array([[r["z"], r["y"], r["x"]] for r in rows]) * POOLED
tvec = np.array([r["t"] for r in rows], int)
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

print(f"\n=== STEP 2a: GT-matched vs unmatched detections ===", flush=True)
print(f"  {len(match)}/{len(rows)} matched; the rest are what the "
       f"node-count penalty pays for", flush=True)
print(f"  {'feature':<8} {'matched':>22} {'unmatched':>22} {'ratio':>7}",
      flush=True)
m_idx = [i for i in range(len(rows)) if i in match]
u_idx = [i for i in range(len(rows)) if i not in match]
for k in ("diam", "i_mean", "i_max", "i_std"):
    a = _summ([rows[i][k] for i in m_idx])
    b = _summ([rows[i][k] for i in u_idx])
    ratio = a[0] / b[0] if b[0] else float("nan")
    print(f"  {k:<8} {a[0]:>8.4f} [{a[1]:.3f},{a[2]:.3f}] "
          f"{b[0]:>8.4f} [{b[1]:.3f},{b[2]:.3f}] {ratio:>7.2f}",
          flush=True)

print(f"\n=== STEP 2b: morphology CHANGE across an edge ===", flush=True)
print("  true GT edges vs false candidate pairs inside the same "
      f"{GATE_UM} um gate", flush=True)
y, dd, di = [], [], []
for t in range(T - 1):
    a_sel = np.where(tvec == t)[0]
    b_sel = np.where(tvec == t + 1)[0]
    if not len(a_sel) or not len(b_sel):
        continue
    tree = cKDTree(pos_um[b_sel])
    for ai in a_sel:
        for bj in tree.query_ball_point(pos_um[ai], GATE_UM):
            bi = int(b_sel[bj])
            if int(ai) not in match or bi not in match:
                continue
            lab = 1 if (match[int(ai)], match[bi]) in gt_edges else 0
            y.append(lab)
            da = rows[int(ai)]["diam"]
            db = rows[bi]["diam"]
            dd.append(abs(db - da) / max(da, 1e-6))
            di.append(abs(rows[bi]["i_mean"] - rows[int(ai)]["i_mean"]))
y = np.asarray(y)
print(f"  {len(y)} candidate pairs, {int(y.sum())} true "
      f"({y.mean() if len(y) else float('nan'):.3f})", flush=True)
if len(y) > 20 and 0 < y.sum() < len(y):
    for nm, v in (("|d diam| / diam", np.asarray(dd)),
                  ("|d intensity|", np.asarray(di))):
        a, b = _summ(v[y == 1]), _summ(v[y == 0])
        print(f"  {nm:<16} true {a[0]:.4f}  false {b[0]:.4f}  "
              f"ratio {a[0] / b[0] if b[0] else float('nan'):.2f}",
              flush=True)
    print("  A ratio near 1.0 means morphology change does not separate a "
          "real link from a near miss, and the edge-ranker idea dies "
          "here rather than after a pipeline run.", flush=True)
else:
    print("  too few pairs to judge", flush=True)

print(f"\n  divisions in this dataset's {T} frames: {n_div} -- the "
      f"'does the mother shrink' question needs far more, and this "
      f"number is what to size the next run against", flush=True)
print("\nMORPH PROBE DONE", flush=True)
