"""Does flip TTA change the prediction at all? Three ways to say no.

SCRIPT_VERSION 2026-08-25a-tta-probe

THE CLAIM UNDER TEST, and it is mine, stated too strongly.
`biohub-det-ablate3` v2 reported F1 0.7546 for three branches with ONE
flip view against 0.7539 for the same three branches with all FOUR, and
I told the user "the four flip views buy nothing". **That was a 0.0007
difference reported with no error estimate.** On 101,615 GT points the
standard error of a proportion is about 0.0013 -- larger than the
difference itself. Neither "buys nothing" nor "is slightly better" is
supported by that number.

The user asked for the reverse proof. So: ASSUME flip TTA does nothing,
derive what that forces, and go look.

WHAT IT FORCES. If averaging four flipped predictions equals one
prediction, the network must be nearly FLIP-EQUIVARIANT -- flip the
input, run it, flip the output back, and get the same volume. That is
directly measurable at the source, before peak-finding throws
information away.

THREE FALSIFIERS, in increasing order of how much they would overturn.

  A. PER-VOXEL. Correlate view 0's logits against views 1, 2, 3, and
     measure the spread ACROSS views relative to the spread across
     voxels. Near-identical volumes confirm the claim at the mechanism:
     TTA has nothing to average. Substantially different volumes
     falsify the MECHANISM even if F1 does not move, and that is the
     more interesting outcome -- it would mean TTA changes the heatmap
     but the change dies at peak-finding, so it could still matter at a
     different gate or to a downstream that consumes the heatmap
     itself, such as the watershed masks in `zh_masks`.

  B. PAIRED ERROR BARS. v2 compared two pooled F1 numbers. Here the
     comparison is paired per crop, 41 of them, which removes the
     between-crop variance that dominates. If the mean paired
     difference is inside its own standard error, the honest statement
     is "indistinguishable at this sample size", not "buys nothing".

  C. DETECTION SET OVERLAP, the sharpest. v2 already shows the two
     configurations are NOT producing the same detections: 106,982 for
     four views against 108,336 for one, a 1.3% gap in count alone.
     Count agreement is not set agreement. This matches the two
     detection sets to each other within 2 um and reports what each has
     that the other does not. If the symmetric difference is a few
     percent, the configurations really are near-identical. If it is
     15% while F1 happens to agree, then F1 agreement is a coincidence
     of two different detectors scoring alike, and "buys nothing" is
     false in every sense that matters.

NOTE ON WHAT WOULD STILL BE UNTESTED. All three ask about DETECTION.
None of them says what flip TTA is worth to adj_J, which is the only
thing the competition scores, and `biohub-node-filter` established that
detection quality and adj_J do not move together.
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

SCRIPT_VERSION = "2026-08-25a-tta-probe"
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


# ---- watershed masks for the Zebrahub crops ----------------------------
# Divisions cannot be learned from competition data: 151 GT divisions in
# the whole 199-dataset train split, 18 positives in the 44b6 training
# half, which trains to AUC 0.43 -- below chance. Zebrahub carries 2001
# and div-kinematics showed it alone reaches 0.808. So the division model
# lives on Zebrahub, and anything it is to learn from morphology needs
# masks HERE.
#
# What morphology buys that the 49 existing features cannot:
#
#   MASS CONSERVATION. At a true division V_a + V_b ~= V_mother, so the
#   ratio is ~1.0; for two arbitrary neighbours each about mother-sized it
#   is ~2.0. A physical constraint a spurious pair has no reason to meet.
#   Each daughter should also sit at (1/2)^(1/3) = 0.794 of the mother's
#   diameter, and the two should match each other.
#
# This is NOT sep_slope's situation. sep_slope failed because the filter
# stack already encodes it -- DIV_DIVERGE_UM >= 2.25 um is a hand-set
# one-frame ancestor, and the 14,159 -> 769 cut was already made with it
# before the model was consulted. The filter stack checks sister distance,
# mutual-nearest, both-children-continue and divergence. NOT ONE OF THEM
# TOUCHES SIZE, and neither does any of the division model's 49 columns:
# 25 geometric, 10 mother history, 10 daughter future, 4 coupling, every
# one of them positional.
#
# Masks are built exactly as biohub-hoct-gate built them and verified
# against: peaks at UNET_THRESH, watershed on the pooled heatmap seeded
# there. The pooled grid is ISOTROPIC by construction -- POOL=4 on Y/X
# turns (1.625, 0.40625, 0.40625) into (1.625, 1.625, 1.625) -- so no
# resampling is needed and regionprops reads real geometry.
#
# One GPU pass, saved per crop. Everything downstream is then free CPU:
# biohub-morph-probe proved skimage.regionprops on these masks reproduces
# HOCT's create_graph at 0.000e+00.

from skimage.segmentation import watershed                      # noqa: E402

MASK_FLOOR = 0.05
OUT = pathlib.Path("/kaggle/working")
BUDGET_S = 9.0 * 3600

CROPS = None
for c in ["/kaggle/input/biohub-zebrahub-extract/crops",
          "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
          "crops"]:
    if Path(c).exists():
        CROPS = Path(c)
        break
if CROPS is None:
    hits = list(Path("/kaggle/input").glob("*/**/crops"))
    CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
assert files, f"{CROPS} is empty"
POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
# =======================================================================
# THE TTA PROBE
# =======================================================================
FRAME_STRIDE = 20
MATCH_UM = 5.0
PAIR_UM = 2.0            # detection-to-detection matching, falsifier C
GATE = 0.15

CROPS = None
hits = sorted(Path("/kaggle/input").glob("*/**/crops"))
CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
assert files, f"{CROPS} is empty"

print(f"\n=== TTA PROBE over {len(files)} crops, every {FRAME_STRIDE}th "
      f"frame ===", flush=True)


def logits_all(vol):
    out = {}
    for b, (models, ws, pp) in enumerate(_BRANCHES):
        x = pool_norm(vol, pp)
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            out[(b, i)] = _flip(lg, i).astype(np.float32)
    return out


def dets_from(lg, vol, flips):
    """All three branches, a chosen set of flip views. Same order of
    operations as detect(): mean the logits over flips, sigmoid per
    branch, mean the branches."""
    hs = []
    for b in range(len(_BRANCHES)):
        acc = None
        for i in flips:
            acc = lg[(b, i)] if acc is None else acc + lg[(b, i)]
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(flips)))))
    h = np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0), h
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    rc, rs = _physical_nms(ref, sc, NMS_UM)
    keep = rs >= GATE
    return rc[keep], rs[keep], h


def greedy_hits(coords, sc, gt_um):
    if len(coords) == 0 or len(gt_um) == 0:
        return 0
    det_um = coords * SCALE[None, :]
    tree = cKDTree(gt_um)
    claimed, hit = set(), 0
    for i in np.argsort(-sc):
        near = tree.query_ball_point(det_um[i], MATCH_UM)
        if not near:
            continue
        near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
        free = [g for g in near if g not in claimed]
        if free:
            claimed.add(free[0])
            hit += 1
    return hit


def set_overlap(a, b):
    """Greedy 1:1 between two DETECTION sets within PAIR_UM. Returns
    (paired, only_a, only_b)."""
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)
    au = a * SCALE[None, :]
    bu = b * SCALE[None, :]
    tree = cKDTree(bu)
    used, pair = set(), 0
    for i in range(len(au)):
        near = tree.query_ball_point(au[i], PAIR_UM)
        near = [j for j in near if j not in used]
        if not near:
            continue
        near.sort(key=lambda j: float(np.linalg.norm(bu[j] - au[i])))
        used.add(near[0])
        pair += 1
    return pair, len(a) - pair, len(b) - pair


# accumulators
vox_r = []          # per-frame Pearson r, view0 vs views 1..3
vox_rel = []        # std across views / std across voxels
h_mad = []          # mean |h_full - h_one| over voxels
per_crop = []       # (F1_full, F1_one) per crop, falsifier B
ov_pair = ov_a = ov_b = 0

t0 = time.time()
done = 0
for f in files:
    if time.time() - t0 > BUDGET_S:
        print(f"\n  WARN budget reached after {done} crops", flush=True)
        break
    z = np.load(f)
    if "image" not in z.files or "nodes" not in z.files:
        print(f"  WARN {f.stem}: missing image or nodes -> skipped",
              flush=True)
        continue
    img = z["image"]
    nodes = np.asarray(z["nodes"], float)
    T = img.shape[0]
    gbt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < T:
            gbt[tt].append(r)
    tp_f = nd_f = tp_o = nd_o = ng = 0
    for t in range(0, T, FRAME_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        vol = np.asarray(img[t])
        gt_um = nodes[gi, 1:] * SCALE[None, :]
        ng += len(gi)
        lg = logits_all(vol)

        # ---- falsifier A: are the four views the same volume? --------
        for b in range(len(_BRANCHES)):
            v0 = lg[(b, 0)].ravel()
            for i in (1, 2, 3):
                vi = lg[(b, i)].ravel()
                s0, si = v0.std(), vi.std()
                if s0 > 1e-6 and si > 1e-6:
                    vox_r.append(float(np.corrcoef(v0, vi)[0, 1]))
            stack = np.stack([lg[(b, i)] for i in _FLIP_VIEWS])
            across = float(stack.std(axis=0).mean())
            within = float(stack.mean(axis=0).std())
            if within > 1e-6:
                vox_rel.append(across / within)

        cf, sf, h_full = dets_from(lg, vol, _FLIP_VIEWS)
        co, so, h_one = dets_from(lg, vol, (0,))
        h_mad.append(float(np.abs(h_full - h_one).mean()))

        tp_f += greedy_hits(cf, sf, gt_um)
        nd_f += len(cf)
        tp_o += greedy_hits(co, so, gt_um)
        nd_o += len(co)

        # ---- falsifier C: do they find the SAME cells? ---------------
        p_, a_, b_ = set_overlap(cf, co)
        ov_pair += p_
        ov_a += a_
        ov_b += b_
        del lg, vol

    if ng and nd_f and nd_o:
        def _f1(tp, nd):
            r = tp / ng
            p = tp / nd
            return 2 * r * p / max(1e-9, r + p)
        per_crop.append((_f1(tp_f, nd_f), _f1(tp_o, nd_o)))
    done += 1
    print(f"  {done}/{len(files)} {f.stem}  ({time.time() - t0:.0f}s)",
          flush=True)
    del z, img
    gc.collect()

print(f"\n=== A. are the four flip views the same volume? ===", flush=True)
vr = np.asarray(vox_r, float)
vl = np.asarray(vox_rel, float)
hm = np.asarray(h_mad, float)
print(f"  Pearson r, view0 vs views 1-3, per branch per frame: "
      f"n={len(vr)}  median {np.median(vr):.5f}  "
      f"[p10 {np.percentile(vr, 10):.5f}, p90 {np.percentile(vr, 90):.5f}]",
      flush=True)
print(f"  spread ACROSS the 4 views / spread across voxels: "
      f"median {np.median(vl):.5f}  "
      f"[p10 {np.percentile(vl, 10):.5f}, p90 {np.percentile(vl, 90):.5f}]",
      flush=True)
print(f"  mean |heatmap(4 views) - heatmap(1 view)| per voxel: "
      f"median {np.median(hm):.6f}", flush=True)
print("  Near-1.0 correlation with a tiny across-view spread means the "
      "network is flip-equivariant and TTA has nothing to average -- "
      "the claim holds AT THE MECHANISM. A visible spread means TTA "
      "does change the heatmap and the change dies later, which is a "
      "different statement and leaves the heatmap's other consumers "
      "(the watershed masks) still in question.", flush=True)

print(f"\n=== B. is the F1 difference distinguishable from zero? ===",
      flush=True)
pc = np.asarray(per_crop, float)
if len(pc) > 1:
    d = pc[:, 1] - pc[:, 0]          # one view minus four views
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  paired over {len(d)} crops: mean {d.mean():+.5f}  "
          f"SE {se:.5f}  -> {d.mean() / max(se, 1e-9):+.2f} SE", flush=True)
    print(f"  F1 four views {pc[:, 0].mean():.4f}, one view "
          f"{pc[:, 1].mean():.4f}", flush=True)
    print(f"  crops where ONE view is better: {int((d > 0).sum())}/"
          f"{len(d)}", flush=True)
    print("  v2 pooled these and read 0.7546 vs 0.7539 with no error "
          "bar. Inside one SE means 'indistinguishable at this sample "
          "size', which is what should have been said.", flush=True)

print(f"\n=== C. do the two configurations find the SAME cells? ===",
      flush=True)
tot = ov_pair + ov_a + ov_b
print(f"  paired within {PAIR_UM} um: {ov_pair}", flush=True)
print(f"  only in FOUR-view: {ov_a}   only in ONE-view: {ov_b}", flush=True)
print(f"  symmetric difference: {ov_a + ov_b} of {tot} "
      f"({100 * (ov_a + ov_b) / max(1, tot):.2f}%)", flush=True)
print("  v2 saw counts 106,982 vs 108,336, a 1.3% gap. If the symmetric "
      "difference here is far larger than that, the two configurations "
      "are different detectors whose summary scores happen to agree, "
      "and 'the flips buy nothing' is false in the sense that matters.",
      flush=True)

print(f"\n  {time.time() - t0:.0f}s total", flush=True)
print("\nTTA PROBE DONE", flush=True)