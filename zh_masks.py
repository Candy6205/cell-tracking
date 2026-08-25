"""zh_masks.py -- watershed masks for the Zebrahub crops, so the
morphology line can go to the division model.

Divisions cannot be learned from competition data -- 18 positives in the
44b6 training half, which trains to AUC 0.43, below chance. Zebrahub
carries 2001 and div-kinematics showed it alone reaches 0.808, so the
division model lives there, and morphology for it needs masks there.

What morphology offers that the 49 existing division features cannot:
MASS CONSERVATION. At a true division V_a + V_b is about V_mother, ratio
~1.0, while two arbitrary mother-sized neighbours give ~2.0. Each daughter
should also sit near (1/2)^(1/3) = 0.794 of the mother diameter and match
its sibling. That is a physical constraint a spurious pair has no reason
to satisfy.

Crucially this is NOT sep_slope's situation. sep_slope failed because the
orphan patch's filter stack already encodes it: DIV_DIVERGE_UM >= 2.25 um
is a hand-set one-frame ancestor, and the 14,159 -> 769 cut was made with
it before the model was ever consulted. That filter stack checks sister
distance, mutual-nearest, both-children-continue and divergence -- NOT ONE
touches size. Neither does any of the division model's 49 columns, which
are 25 geometric, 10 mother-history, 10 daughter-future and 4 coupling,
every one positional.

Masks are built exactly as biohub-hoct-gate built them: peaks at
UNET_THRESH, watershed on the pooled heatmap seeded there. The pooled grid
is isotropic by construction, so regionprops reads real geometry with no
resampling.

ONE GPU pass. Everything downstream is free CPU afterwards, because
biohub-morph-probe proved skimage.regionprops on these masks reproduces
HOCT create_graph at 0.000e+00 on all five columns.

Saved per crop, with nodes / track_id / parent_track_id carried through so
a consumer needs this output and not the 7.5 GB of crops.
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

SCRIPT_VERSION = "2026-08-25h-zh-masks"
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
print(f"\n=== WATERSHED MASKS FOR {len(files)} ZEBRAHUB CROPS ===",
      flush=True)
print(f"  pooled spacing {tuple(POOLED)} um -- isotropic "
      f"{np.allclose(POOLED, POOLED[0])}", flush=True)


def heatmap(vol):
    """detect()'s averaged heatmap, returned before peak-finding."""
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
    return np.mean(hs, axis=0)


t0 = time.time()
done = skipped = 0
for f in files:
    if time.time() - t0 > BUDGET_S:
        print(f"\n  WARN wall-clock budget reached: {len(files) - done} "
              f"crops NOT processed. The ones written are complete and "
              f"usable; re-run to finish the rest.", flush=True)
        break
    z = np.load(f)
    if "image" not in z.files:
        print(f"  WARN {f.stem}: no 'image' -> skipped, so it will be "
              f"absent downstream", flush=True)
        skipped += 1
        continue
    img = z["image"]
    if img.ndim != 4:
        print(f"  WARN {f.stem}: image ndim {img.ndim} != 4 -> skipped",
              flush=True)
        skipped += 1
        continue
    T = img.shape[0]
    labs, npk = [], 0
    for t in range(T):
        h = heatmap(np.asarray(img[t]))
        pk = peak_local_max(h, min_distance=1, threshold_abs=UNET_THRESH,
                            exclude_border=False)
        mk = np.zeros(h.shape, np.int32)
        for i, p in enumerate(pk):
            mk[p[0], p[1], p[2]] = i + 1
        labs.append(watershed(-h, mk, mask=h >= MASK_FLOOR).astype(np.int32))
        npk += len(pk)
        if t % 25 == 0:
            gc.collect()
    labels = np.stack(labs)
    out = {"labels": labels, "pooled": np.asarray(POOLED),
           "name": np.array([f.stem])}
    # carry the GT through so one file trains without the 7.5 GB crops
    for k in ("nodes", "track_id", "parent_track_id", "voxel_um", "t0"):
        if k in z.files:
            out[k] = z[k]
        else:
            print(f"  WARN {f.stem}: crop has no '{k}'; downstream code "
                  f"that needs it must skip this crop", flush=True)
    np.savez_compressed(OUT / f"zhmask_{f.stem}.npz", **out)
    mb = (OUT / f"zhmask_{f.stem}.npz").stat().st_size / 2 ** 20
    done += 1
    print(f"  {done}/{len(files)} {f.stem}: T={T}, {npk} peaks, "
          f"{mb:.1f} MB ({time.time() - t0:.0f}s)", flush=True)
    del z, img, labs, labels
    gc.collect()

print(f"\n  {done} written, {skipped} skipped, "
      f"{time.time() - t0:.0f}s total", flush=True)
print("  Attach this kernel and the whole morphology line is CPU from "
      "here: regionprops on these masks IS create_graph (verified at "
      "0.000e+00 by biohub-morph-probe).", flush=True)
print("\nZH MASKS DONE", flush=True)
