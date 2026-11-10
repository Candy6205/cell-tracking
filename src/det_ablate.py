"""det_ablate.py -- which of the xl detector's 12 forward passes
actually earn their place?

The xl detector is three UNet3D branches x four flip views, averaged. Two
facts make that worth auditing:

- branches 1 and 2 (`unet3d_traintophat`, `unet3d_v2_tophat_b32`) receive
  PIXEL-IDENTICAL input -- same top-hat preprocessing -- so their errors
  should be correlated and they should cancel each other's mistakes badly.
  Branch 0 (`unet3d_bright`, no top-hat) is the only genuinely different
  view of the data, and it carries the WORST solo val_recall. Solo recall
  is not what earns a place in an ensemble.
- the two averaging levels happen in different spaces: flip views average
  in LOGIT space (inside the branch), branches average in PROBABILITY
  space. Combined with threshold 0.15 on the mean, the branch average is
  already a soft two-of-three agreement rule.

The only per-model numbers anyone has are the val_recall each checkpoint
carries from its original author's split -- bright 0.573, traintophat
0.659, v2_tophat_b32 0.579 -- measured on a DIFFERENT ruler from the 0.766
the ensemble scores on ours. Those two were never comparable, and no
ablation has ever been run here.

ONE inference pass covers everything: the 12 pieces are already computed,
so per branch the logits accumulate and the accumulator is snapshot after
1, 2 and 4 views. Eleven configurations -- all seven non-empty branch
subsets at four views, plus all-three at one and at two views -- then come
out of nine held heatmaps with no second forward pass.

Reports detection recall / precision / F1 / N-over-N_true against
competition GT on the 12 holdout datasets. DETECTION ONLY. Twice today an
offline number failed to survive the pipeline (scale-invariant edge
features; division-model fork pruning), so whatever wins here has to be
re-measured through the full linker before it is believed.
"""
import gc
import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-24e-det-ablate"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        _p = nn.Conv3d(1, 1, 3).to("cuda")
        _ = _p(torch.zeros(1, 1, 4, 4, 4, device="cuda")).cpu()
        DEVICE = "cuda"
        del _p
    except Exception as e:                                     # noqa: BLE001
        print("GPU present but conv3d unusable -> CPU:", str(e)[:80])

# ---- detect-side constants (xl-repro verbatim) ------------------------
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- pipeline constants (ranker-diag verbatim: the 0.9345 config) -----
GATE_UM = 10.0
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
GAP1_GATE_UM = 9.0
GAP1_SNAP_UM = 3.2
GAP1_MIN_CAND_SCORE = 0.10
GAP1_CAP_FRAC = 0.003
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375
ILP_FLOOR = 0.10                            # candidates fed to the ILP
ILP_DISAPP = 1.5
ILP_DIV_W = 0.55   # fine sweep argmax
ILP_WINDOW = 25    # wincal: 0.9382 vs global 0.9368, 2.08x faster

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
assert ROOT is not None, "competition not attached"
TEST_DIR = Path(ROOT) / "test"

# ---- xl detect stack (xl-repro verbatim) ------------------------------

def _find_weight(name):
    cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert cands, f"{name} not found"
    return cands[0]

def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _load_model(name):
    ck = torch.load(_find_weight(name), map_location=DEVICE)
    m = UNet3D(base=ck.get("base", 24)).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print("loaded", name, "| val_recall", ck.get("val_recall"), flush=True)
    return m

MODELS = [_load_model("unet3d_bright.pt"), _load_model("unet3d_traintophat.pt")]

def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))

_ZC = {}

def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])

def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def pool_norm(vol, preproc=""):
    p = pool_xy(vol)
    if preproc == "tophat":
        from scipy.ndimage import grey_opening
        p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)

def _refine(vol, zyx, rz=2, ryx=5):
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bg = float(crop.min())
    w = np.clip(crop - bg, 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return (np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s,
            float(crop.max() - bg))

def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]

_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]
_FLIP_VIEWS = (0, 1, 2, 3)

def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x

def detect(vol):
    hs = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc = None
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, i)
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ------------------------------------------------------------- graph ops

def avail_T(zp):
    meta = read_array_meta(zp)
    T = meta["shape"][0]
    present = [t for t in range(T)
               if (Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    return max(present) + 1 if present else 0

# ---------------- cache xl detections for the rest of train --------------
# The census settled it: 151 GT divisions across all 199 train datasets, and
# only 31 of them sit in the 44 datasets we have detection caches for. This
# caches the other 155 with the SAME xl detector (same three branches, same
# flip views, same thresholds), so the pools can simply be merged.
#
# Two payoffs, both independent of the division work:
#   - ~127 usable in-domain divisions instead of 26 (the anchor set that the
#     Zebrahub / synthetic question needs in order to be answerable at all)
#   - the edge ranker's training pool goes from 32 datasets to 187
#
# Output format matches biohub-edge-ranker exactly: one detcache_<name>.npz
# per dataset with keys c{t} (coords Nx3) and s{t} (scores N), so the
# existing `CACHES` glob picks both up with no code change.

TRAIN_DIR = Path(ROOT) / "train"


# ---- the ablation ------------------------------------------------------
# The xl detector runs THREE branches x FOUR flip views = 12 forward passes
# per frame, then averages. Nobody here has ever measured which of those 12
# earns its place: the only per-model numbers are the val_recall each
# checkpoint carries from its ORIGINAL author's own split
# (bright 0.573, traintophat 0.659, v2_tophat_b32 0.579), which is a
# different ruler from the 0.766 the ensemble scores on ours -- the two
# were never comparable and this fills that gap.
#
# Structurally suspicious: branches 1 and 2 (traintophat, v2_tophat_b32)
# receive PIXEL-IDENTICAL input, so their errors should be correlated and
# they should cancel each other's mistakes poorly. Branch 0 (bright) is the
# only genuinely different view. Its solo recall is the worst -- but solo
# recall is not what earns a place in an ensemble.
#
# ONE inference pass answers everything: the 12 pieces are already being
# computed. Per branch the logits are accumulated over views, and the
# accumulator is SNAPSHOT after 1, 2 and 4 views, so nine heatmaps cover
# every configuration below without a second forward pass.

import itertools                                                # noqa: E402

# cache_train.py never reads ground truth, so its preamble does NOT put the
# code dataset on sys.path the way ranker_fine.py and div_train.py do.
# v1 died here with ModuleNotFoundError after 9 minutes, all of it spent
# loading the three models. A static undefined-name pass cannot catch this:
# pyflakes assumes any import resolves.
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS if Path(c, "viz_cells.py").exists()),
             None)
assert _code is not None, \
    f"viz_cells.py not found under any of {CODE_ROOTS} -- attach " \
    f"faustine649/cell-tracking"
sys.path.insert(0, _code)
from viz_cells import read_geff, geff_path                       # noqa: E402

BR_NAMES = ["bright", "traintophat", "v2_tophat_b32"]
VIEW_SNAPSHOTS = (1, 2, 4)
MATCH_UM = 5.0

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# (label, branch subset, number of flip views)
CONFIGS = []
for r in (1, 2, 3):
    for sub in itertools.combinations(range(3), r):
        CONFIGS.append(("+".join(BR_NAMES[i] for i in sub), sub, 4))
CONFIGS.append(("ALL @1 view", (0, 1, 2), 1))
CONFIGS.append(("ALL @2 views", (0, 1, 2), 2))
print(f"\n=== {len(CONFIGS)} CONFIGURATIONS, ONE INFERENCE PASS ===",
      flush=True)


def branch_heatmaps(vol):
    """Return hm[b][v] for v in VIEW_SNAPSHOTS: branch b's heatmap using
    the first v flip views. Logits accumulate (that is where the flip TTA
    averages); the sigmoid is applied at each snapshot."""
    out = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc, snaps = None, {}
        for i, view in enumerate(_FLIP_VIEWS):
            xv = np.ascontiguousarray(_flip(x, view))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, view)
            acc = lg if acc is None else acc + lg
            n = i + 1
            if n in VIEW_SNAPSHOTS:
                snaps[n] = 1.0 / (1.0 + np.exp(-(acc / n)))
        out.append(snaps)
    return out


def peaks_from(h, vol):
    """The rest of the xl path, unchanged: peak, un-pool, full-resolution
    centroid refinement, physical NMS."""
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


tally = {lab: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for lab, _, _ in CONFIGS}
n_true_total = 0
t0 = time.time()

for k_ds, name in enumerate(HOLDOUT):
    zp = TRAIN_DIR / f"{name}.zarr"
    if not zp.exists():
        print(f"  WARN {name}: no zarr at {zp} -> SKIPPED, so every "
              f"configuration below is measured on fewer datasets than "
              f"the {len(HOLDOUT)} intended", flush=True)
        continue
    T = avail_T(zp)
    gt = read_geff(geff_path(Path(ROOT), "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    gt_by_t = defaultdict(list)
    for i in range(len(g_t)):
        gt_by_t[int(g_t[i])].append(i)

    meta = read_array_meta(zp)
    for t in range(T):
        vol = load_volume(zp, t, meta)
        hm = branch_heatmaps(vol)
        idx = gt_by_t.get(t, [])
        gpts = g_pos[idx] * SCALE[None, :] if idx else np.zeros((0, 3))
        n_true_total += len(idx)
        gtree = cKDTree(gpts) if len(gpts) else None
        for lab, sub, nv in CONFIGS:
            h = hm[sub[0]][nv] if len(sub) == 1 else \
                np.mean([hm[b][nv] for b in sub], axis=0)
            coords, _sc = peaks_from(h, vol)
            d = tally[lab]
            d["n"] += len(coords)
            if gtree is None or len(coords) == 0:
                d["fp"] += len(coords)
                d["fn"] += len(gpts)
                continue
            # one GT node may claim at most one detection
            dd, gi = gtree.query(coords * SCALE[None, :])
            claimed = set()
            tp = 0
            for k in np.argsort(dd):
                if dd[k] <= MATCH_UM and int(gi[k]) not in claimed:
                    claimed.add(int(gi[k]))
                    tp += 1
            d["tp"] += tp
            d["fp"] += len(coords) - tp
            d["fn"] += len(gpts) - tp
        del hm
    print(f"  {k_ds + 1}/{len(HOLDOUT)} {name} "
          f"({time.time() - t0:.0f}s)", flush=True)

print(f"\n=== DETECTION ON {len(HOLDOUT)} HOLDOUT DATASETS "
      f"({n_true_total} GT nodes) ===", flush=True)
print("  Detection quality only. Today's lesson, twice over: an offline "
      "number is not an adj_J. Whatever wins here must be re-measured "
      "through the full pipeline before it is believed.", flush=True)
print(f"  {'configuration':<34} {'recall':>7} {'prec':>7} {'F1':>7} "
      f"{'N/N_true':>9}", flush=True)
rows = []
for lab, _sub, _nv in CONFIGS:
    d = tally[lab]
    rec = d["tp"] / max(d["tp"] + d["fn"], 1)
    pre = d["tp"] / max(d["tp"] + d["fp"], 1)
    f1 = 2 * rec * pre / max(rec + pre, 1e-9)
    ratio = d["n"] / max(n_true_total, 1)
    rows.append((f1, rec, pre, ratio, lab))
    print(f"  {lab:<34} {rec:>7.4f} {pre:>7.4f} {f1:>7.4f} "
          f"{ratio:>9.3f}", flush=True)

print("\n  ranked by F1:", flush=True)
for f1, rec, pre, ratio, lab in sorted(rows, reverse=True):
    print(f"    {f1:.4f}  {lab}", flush=True)

full = next(r for r in rows if r[4] == "+".join(BR_NAMES))
print(f"\n  the shipped configuration (all three, 4 views) is "
      f"F1 {full[0]:.4f}", flush=True)
better = [r for r in rows if r[0] > full[0]]
if better:
    print(f"  {len(better)} configuration(s) beat it; best is "
          f"{max(better)[4]} at F1 {max(better)[0]:.4f} "
          f"({max(better)[0] - full[0]:+.4f})", flush=True)
else:
    print("  nothing beat it -- the shipped ensemble is already the best "
          "of the 11, and the 12 forward passes are all earning their "
          "place.", flush=True)

print("\nDET ABLATION DONE", flush=True)
