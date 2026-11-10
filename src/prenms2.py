"""Pre-NMS peaks for the OTHER two domains: 44b6 and Zebrahub.

SCRIPT_VERSION 2026-08-25a-prenms2

`biohub-prenms-cache` wrote pre-NMS peaks for the 12 `6bba` holdout
datasets -- 189,938 peaks, 3.2 MB, 23 min. It was built BEFORE the
three-domain standard was set, so a sweep on that cache alone would
produce exactly the kind of `6bba`-only number the leaderboard has now
refused twice. This fills the gap.

WHAT IT ADDS.

  44b6      12 datasets from the other imaging batch, chosen at runtime
            and EXCLUDING D32's training pool, so the ranker that scores
            them has not seen them. 327 cells/frame against `6bba`'s 97
            and duplicate-frame rate 0% against 7.47%.
  zebrahub  a different instrument that no public model has seen. The
            crops carry their own dense GT, so the pipeline can be
            scored there with the OFFICIAL metric.

WHY PRE-NMS AT ALL. `biohub-det-floor` measured where the missing recall
is: 0.7738 detected, **0.0950 has a peak and is discarded**, 0.1312 has
no peak at any threshold. `_physical_nms` at `NMS_UM` = 4.0 is one of the
two things doing the discarding and it has never been swept. Caching
peaks BEFORE suppression makes every radius a CPU operation instead of a
GPU pass.

⚠ `_physical_nms` IS ORDER DEPENDENT, so a radius cannot be applied to
an already-suppressed set. That is the whole reason for this cache, and
it is also why the outputs use keys `p{t}`/`q{t}` rather than
`c{t}`/`s{t}`: the pipeline's `CACHES` glob must not pick up
unsuppressed peaks by accident and feed them to code that assumes
suppression has happened.
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

SCRIPT_VERSION = "2026-08-25a-prenms2"
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
BUDGET_S = 10.0 * 3600
t_start = time.time()

# ---- which 44b6 datasets: outside D32's training pool -------------------
_pk = None
for _p in pathlib.Path("/kaggle/input").glob("*/**/edge_ranker_d32.pkl"):
    _pk = _p
    break
assert _pk is not None, ("edge_ranker_d32.pkl not found -- attach "
                         "faustine649/biohub-poolsize. Without it the 44b6 "
                         "datasets cannot be filtered to ones the ranker "
                         "has not trained on, and the arm would be biased.")
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
POOL32 = set(_blob.get("pool") or [])
assert 0 < len(POOL32) <= 60, (
    f"the pkl at {_pk} lists {len(POOL32)} training datasets; expected "
    f"D32's small pool")
print(f"  D32 pool: {len(POOL32)} datasets, excluded from the 44b6 set",
      flush=True)

N_44B6 = 12
_all44 = sorted(zp.name.replace(".zarr", "")
                for zp in TRAIN_DIR.glob("44b6_*.zarr"))
_clean44 = [n for n in _all44 if n not in POOL32]
SET_44B6 = _clean44[:N_44B6]
print(f"  44b6: {len(_all44)} present, {len(_all44) - len(_clean44)} in "
      f"the pool, {len(SET_44B6)} selected", flush=True)
assert len(SET_44B6) == N_44B6, (
    f"only {len(SET_44B6)} clean 44b6 datasets; the second domain would be "
    f"thinner than the first and the comparison unbalanced")

done = 0
n_peak = 0
for name in SET_44B6:
    if time.time() - t_start > BUDGET_S:
        print(f"\nWARN budget reached during 44b6; PARTIAL cache",
              flush=True)
        break
    zp = TRAIN_DIR / f"{name}.zarr"
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
    n_peak += npk
    done += 1
    print(f"  44b6 {done}/{len(SET_44B6)} {name}: T={T}, {npk} pre-NMS "
          f"peaks ({time.time() - t0:.0f}s)", flush=True)
    gc.collect()
print(f"\n  44b6: {done} datasets, {n_peak} peaks, "
      f"{time.time() - t_start:.0f}s", flush=True)

# ---- zebrahub: the crops carry raw volumes, not zarr --------------------
N_ZH = 8
hits = sorted(pathlib.Path("/kaggle/input").glob("*/**/crops"))
CROPS = hits[0] if hits else None
if CROPS is None:
    print("  WARN zebrahub crops are not mounted -> the third domain is "
          "ABSENT from this cache and any sweep built on it covers two "
          "domains, not three. Say so rather than reporting it as three.",
          flush=True)
else:
    files = sorted(CROPS.glob("*.npz"))[:N_ZH]
    print(f"\n  zebrahub: {len(files)} crops", flush=True)
    zdone = 0
    zpk = 0
    for f in files:
        if time.time() - t_start > BUDGET_S:
            print(f"\nWARN budget reached during zebrahub; PARTIAL",
                  flush=True)
            break
        t0 = time.time()
        z = np.load(f)
        if "image" not in z.files:
            print(f"  WARN {f.stem}: no image -> skipped", flush=True)
            continue
        img = z["image"]
        if img.ndim != 4:
            print(f"  WARN {f.stem}: image ndim {img.ndim} != 4 -> skipped",
                  flush=True)
            continue
        T = img.shape[0]
        out = {}
        npk = 0
        for t in range(T):
            p, q = detect(np.asarray(img[t]))
            out[f"p{t}"] = np.asarray(p, np.float32)
            out[f"q{t}"] = np.asarray(q, np.float32)
            npk += len(q)
            if t % 25 == 0:
                gc.collect()
        # the GT travels with the peaks so a consumer needs one file
        for k in ("nodes", "track_id", "parent_track_id"):
            if k in z.files:
                out[k] = z[k]
            else:
                print(f"  WARN {f.stem}: crop has no '{k}'; a consumer "
                      f"cannot score this crop and must skip it",
                      flush=True)
        np.savez_compressed(OUT / f"prenmszh_{f.stem}.npz", **out)
        zpk += npk
        zdone += 1
        print(f"  zh {zdone}/{len(files)} {f.stem}: T={T}, {npk} pre-NMS "
              f"peaks ({time.time() - t0:.0f}s)", flush=True)
        del z, img
        gc.collect()
    print(f"\n  zebrahub: {zdone} crops, {zpk} peaks", flush=True)

print(f"\n  {time.time() - t_start:.0f}s total", flush=True)
print("  With biohub-prenms-cache attached alongside this, all three "
      "domains have pre-NMS peaks and every NMS radius is CPU.",
      flush=True)
print("\nPRENMS2 DONE", flush=True)