"""Cache the peaks BEFORE non-maximum suppression, for the 12 holdout sets.

SCRIPT_VERSION 2026-08-25a-prenms-cache

WHY. `biohub-det-floor` measured, on Zebrahub, where the missing
detections actually are:

    detected today (gate 0.15)          0.7738
    HAS A PEAK BUT IS DISCARDED         0.0950
    no peak at any threshold            0.1312

**Nine and a half points of recall are peaks that exist and are then
thrown away** -- by `_physical_nms` at `NMS_UM` = 4.0 um, or by the gate
at 0.15. `biohub-thr-sweep` already swept the gate and found +0.0033
adj_J at 0.12. **`NMS_UM` has never been swept at all**, and it is five
times the size of what the gate moved.

WHY THIS RUN ONLY CACHES. `detcache_*.npz` stores POST-NMS detections, so
every NMS radius would otherwise need its own GPU pass over the data.
Instead this writes the peaks BEFORE suppression, once. Every radius is
then a CPU operation on the cache, exactly the pattern `biohub-zh-masks`
used to make the whole morphology line free.

WHY THE 12 HOLDOUT DATASETS AND NOT ZEBRAHUB. The user asked whether a
Zebrahub score transfers, and the honest answer on record is that
NOBODY HAS EVER TESTED IT -- `biohub-node-filter` failed at TRAINING
transfer, which is a different question, and no configuration has ever
been ranked on Zebrahub and then re-ranked on competition data. Rather
than assume, this caches the competition holdout directly so the sweep
measures **adj_J**, the thing the competition scores, with no transfer
assumption anywhere in the chain.

WHAT ELSE THE CACHE ANSWERS FOR FREE. `det_floor`'s ceiling of 0.8688 is
OPTIMISTIC: it counts a GT point as reachable if any peak lies within
5 um, with **no 1:1 constraint**, so two adjacent cells sharing one peak
both count. With pre-NMS peaks on disk the true 1:1 ceiling is a CPU
calculation, and the gap between the two numbers is a direct measure of
how much NMS merges neighbouring cells. That gap is also the most likely
explanation for `biohub-div-mass`'s unexplained **44.7% of divisions
whose two daughters shared one watershed label**.

OUTPUT. One `prenms_<name>.npz` per holdout dataset, keys `p{t}` (peaks,
Nx3, refined coordinates) and `q{t}` (heatmap scores, N) -- deliberately
NOT `c{t}`/`s{t}`, so the existing `CACHES` glob cannot pick these up by
accident and feed unsuppressed peaks into a pipeline that expects
suppressed ones.
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

SCRIPT_VERSION = "2026-08-25a-prenms-cache"
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
    # NO SUPPRESSION HERE. That is the entire point of this kernel:
    # NMS is applied downstream on CPU, once per radius under test.
    return ref, sc


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
OUT = pathlib.Path("/kaggle/working")

# the 12 held-out 6bba datasets, COPIED VERBATIM from
# poolsize.py -- an invented list here would silently
# evaluate on the wrong datasets
HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
_present = {zp.name.replace(".zarr", "")
            for zp in TRAIN_DIR.glob("*.zarr")}
todo = [(n, TRAIN_DIR / f"{n}.zarr") for n in HOLDOUT if n in _present]
missing = [n for n in HOLDOUT if n not in _present]
if missing:
    print(f"  WARN {len(missing)} holdout names are not in train/: "
          f"{missing[:4]} -- they will be ABSENT from the cache and the "
          f"sweep will silently run on fewer datasets than the anchor",
          flush=True)
assert todo, ("no holdout dataset resolved; the HOLDOUT list here must "
              "match poolsize.py's exactly or the sweep is not comparable "
              "to the recorded 0.9386 / 0.9254")
print(f"\ncaching PRE-NMS peaks for {len(todo)} holdout datasets",
      flush=True)

BUDGET_S = 10.0 * 3600
t_start = time.time()
done = 0
n_peak = 0
for i, (name, zp) in enumerate(todo):
    if time.time() - t_start > BUDGET_S:
        print(f"\nWARN budget reached: {len(todo) - i} datasets NOT "
              f"cached. This is a PARTIAL cache and the sweep must not "
              f"be compared against the 12-dataset anchor.", flush=True)
        break
    t0 = time.time()
    try:
        T = avail_T(zp)
    except Exception as e:                                   # noqa: BLE001
        print(f"  WARN {name}: avail_T failed ({str(e)[:70]}) -- skipped",
              flush=True)
        continue
    if T == 0:
        print(f"  WARN {name}: no frames present -- skipped", flush=True)
        continue
    meta = read_array_meta(zp)
    out = {}
    npk = 0
    for t in range(T):
        p, q = detect(load_volume(zp, t, meta))
        out[f"p{t}"] = np.asarray(p, np.float32)
        out[f"q{t}"] = np.asarray(q, np.float32)
        npk += len(q)
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"prenms_{name}.npz", **out)
    mb = (OUT / f"prenms_{name}.npz").stat().st_size / 2 ** 20
    n_peak += npk
    done += 1
    print(f"  {done}/{len(todo)} {name}: T={T}, {npk} pre-NMS peaks, "
          f"{mb:.1f} MB ({time.time() - t0:.0f}s)", flush=True)
    gc.collect()

print(f"\n  {done} datasets, {n_peak} pre-NMS peaks total, "
      f"{time.time() - t_start:.0f}s", flush=True)
print("  For reference the POST-NMS caches hold far fewer; the ratio is "
      "how many peaks _physical_nms removes at NMS_UM 4.0, which is the "
      "quantity the sweep is about.", flush=True)
print("  Attach this kernel and every NMS radius is a CPU operation.",
      flush=True)
print("\nPRENMS CACHE DONE", flush=True)