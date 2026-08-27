"""submit_ranker.py -- the 0.917 xl pipeline with its linker replaced.

Detection side is biohub-xl-repro verbatim (3-branch ensemble + flip TTA);
the linker is the learned edge re-ranker feeding a capacity-2 tracksdata
ILP, and every post-link patch is taken from biohub-ranker-diag, which is
the code path that measured 0.9345 on the 12-dataset 6bba holdout against
the xl baseline's 0.9318 (+0.0027, same detector, same datasets).

Single variable vs LB 0.917 (ref 55690436): the linker.

Internet is OFF at submission time, so tracksdata/pyscipopt come from the
support-pack wheels and `predict_unet_transformer` from the pack's repo
copy -- no git clone anywhere.
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

SCRIPT_VERSION = "2026-08-28a-detcmp"
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
if ROOT is None:
    print("note: competition data not mounted "
          "(not needed here)", flush=True)
TEST_DIR = (Path(ROOT) / "test") if ROOT else None   # unused here

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

# ---------------- xl detections on the Zebrahub crops --------------------
# Everything for the division model is now in place except this. Training
# must use DETECTED positions, not GT ones, or the model learns a geometry
# that inference never sees (the ranker already does it this way: detected
# positions matched to GT within MATCH_UM).
#
# The crops already carry `nodes`, `track_id` and `parent_track_id`, so a
# division is recoverable as "two track_ids sharing a parent" (the extract
# header states it survives the crop). This adds the detector's own view.
#
# Output per crop: zhdet_<crop>.npz with c{t}/s{t} in the SAME layout as
# detcache_*.npz, plus the GT arrays copied through so one file is
# self-contained for training.

CROPS = None
for c in ["/kaggle/input/biohub-zebrahub-extract/crops",
          "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(c).exists():
        CROPS = pathlib.Path(c)
        break
if CROPS is None:
    hits = list(pathlib.Path("/kaggle/input").glob("*/**/crops"))
    CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
print(f"crops: {len(files)} under {CROPS}", flush=True)
assert files, "crops directory is empty"

# ======================= DeepCenter ====================================
# The architecture is lifted verbatim from the public notebook's inline
# definition (a plain UNet3D: GroupNorm + SiLU blocks, MaxPool down,
# ConvTranspose up, 1x1x1 head). Its checkpoint stores the weights under
# "model_state", not "state_dict" -- a different convention from ours,
# which is the first thing that would fail silently if guessed.
class _DCConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DCDeepCenterUNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        c = int(base_channels)
        self.enc1 = _DCConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(2, 2)
        self.enc2 = _DCConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(2, 2)
        self.enc3 = _DCConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2, 2)
        self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
        self.dec3 = _DCConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
        self.dec2 = _DCConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, 2)
        self.dec1 = _DCConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def _dc_pool_xy(volume, factor=POOL):
    """DeepCenter's XY pooling -- VERIFIED IDENTICAL to our `pool_xy`.

    Both crop to a multiple of the factor and mean-pool; neither pads,
    and neither touches z. Kept as a separate function only so the diff
    is on the record. **So the only thing that differs between the two
    detection paths is the NETWORK and its NORMALISATION** -- pooling,
    peak extraction, sub-voxel refine and 4.0 um NMS are all shared.
    """
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor,
                           x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize(volume, lo_pct=50.0, hi_pct=99.5,
                  clip_lo=-0.5, clip_hi=6.0):
    """DeepCenter's dynamic-range normalisation, defaults from its cfg.

    ⚠ Note how different this is from ours: a percentile stretch between
    the MEDIAN and the 99.5th, clipped to [-0.5, 6.0]. Feeding it our
    normalisation would be testing the wrong thing."""
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, lo_pct))
    hi = float(np.percentile(vol, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


def _dc_find_checkpoint():
    """checkpoint_last.pt (epoch 500) is preferred over best.pt (epoch 2).

    reference.md records that the epoch-500 checkpoint has SEVEN TIMES
    the validation loss and warns not to read that as "got worse": the
    loss is positive-unlabeled BCE, so it PUNISHES the model for finding
    cells that were never annotated. A checkpoint that scores worse on
    that loss is one that finds more unannotated cells -- which is
    exactly what a repair pool wants. This run is not looking for a
    better primary detector, it is looking for coverage of our misses.
    """
    pats = ["**/full_frame_center/**/checkpoint_last.pt",
            "**/full_frame_center/**/best.pt",
            "**/biohub-deepcenter-unet3d-center-prior-v1/**/*.pt"]
    for p in pats:
        for f in sorted(pathlib.Path("/kaggle/input").glob(p)):
            return f
    return None


DC_CKPT = _dc_find_checkpoint()
assert DC_CKPT is not None, (
    "DeepCenter checkpoint not mounted. Attach "
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1")
_dc_ck = torch.load(DC_CKPT, map_location=DEVICE)
assert isinstance(_dc_ck, dict) and "model_state" in _dc_ck, (
    f"{DC_CKPT} has no 'model_state' key; keys are "
    f"{list(_dc_ck)[:8] if isinstance(_dc_ck, dict) else type(_dc_ck)}")
DC = _DCDeepCenterUNet3D(base_channels=int(_dc_ck.get("base_channels", 24)))
DC.load_state_dict(_dc_ck["model_state"])
DC = DC.to(DEVICE).eval()
print(f"DeepCenter loaded from {DC_CKPT}", flush=True)
for _k in ("epoch", "val_loss", "val_recall", "best_metric"):
    if _k in _dc_ck:
        print(f"  {_k}: {_dc_ck[_k]}", flush=True)


def dc_detect(vol, flips=False):
    """DeepCenter's network, OUR post-processing.

    Everything downstream of the sigmoid is identical to `detect()` --
    the same peak extraction, the same sub-voxel refine, the same 4.0 um
    physical NMS -- so the comparison isolates the network and its
    normalisation. `flips` adds the same 4-view averaging our detector
    gets, because otherwise this compares an ensemble to a single pass.
    """
    x = _dc_normalize(_dc_pool_xy(vol))
    views = _FLIP_VIEWS if flips else _FLIP_VIEWS[:1]
    acc = None
    for i in views:
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = DC(torch.from_numpy(xv)[None, None].to(DEVICE)
                    )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    h = 1.0 / (1.0 + np.exp(-(acc / len(views))))
    pk = peak_local_max(h, min_distance=1, threshold_abs=CAND_THR,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ======================= the comparison ================================
MATCH_UM = 7.0          # the OFFICIAL matcher; 5.0 is a legacy mistake
N_CROPS = 8             # matches the zebrahub arm in gap4/span4/rank5
DC_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)

files = sorted(files)[:N_CROPS]
print(f"{chr(10)}=== DETECTOR COMPARISON ON ZEBRAHUB, {len(files)} crops ===",
      flush=True)
print("  DeepCenter is pilkwang's OWN addition on top of the official "
      "pipeline, with its own split_manifest.json -- not an external "
      "detector with data we lack. It trains on the same competition "
      "images we have, so its only possible value is model diversity. "
      "Venue chosen because it TRAINED on all 71 44b6 and was "
      "model-selected on all 128 6bba, so neither competition domain is "
      "a fair test of it. Zebrahub is a different instrument, no public "
      "model has seen it, and its GT is DENSE -- the only arm whose "
      "rates describe every cell rather than the annotated fraction.",
      flush=True)
print("  ⚠ THE PUBLIC SOLUTION HAS THIS MECHANISM AND DISABLES IT: "
      "BIOHUB_USE_DEEPCENTER_VETO='0'. It is a VETO there (reject a "
      "repair), the opposite of supplying candidates, and their "
      "end-to-end is 0.915 against our 0.922 -- but it lowers the prior "
      "and is said here before the numbers.", flush=True)
print("  ⚠ our detector is 3 branches x 4 flip views; DeepCenter as "
      "shipped is ONE forward pass. Both are reported.", flush=True)
print(f"  recall/precision at MATCH_UM {MATCH_UM} (official). Our arm is "
      f"the SHIPPED threshold {UNET_THRESH}.", flush=True)

_ROWS = []
_t0 = time.time()
for _i, f in enumerate(files):
    z = np.load(f)
    img = z["image"]
    if img.ndim != 4 or "nodes" not in z.files:
        print(f"  WARN {f.stem}: unusable -> skipped", flush=True)
        continue
    T = img.shape[0]
    gnodes = np.asarray(z["nodes"], float)
    g_t = np.asarray(gnodes[:, 0], int)
    g_zyx = np.asarray(gnodes[:, 1:], float)
    tid = (np.asarray(z["track_id"]).reshape(-1)
           if "track_id" in z.files else None)

    ours, dc1, dc4 = {}, {}, {}
    for t in range(T):
        v = np.asarray(img[t])
        ours[t] = detect(v)
        dc1[t] = dc_detect(v, flips=False)
        dc4[t] = dc_detect(v, flips=True)
        if t % 10 == 0:
            gc.collect()

    _ROWS.append((f.stem, T, g_t, g_zyx, tid, ours, dc1, dc4))
    print(f"  {_i + 1}/{len(files)} {f.stem}: T={T}, GT {len(g_t):,} nodes "
          f"({time.time() - _t0:.0f}s)", flush=True)
    del img, z
    gc.collect()


def _hit(cs, ss, thr, gxyz):
    """Which GT rows have a kept detection within MATCH_UM, and how many
    detections were kept. Greedy nearest-neighbour, not the scorer's 1:1
    assignment -- this is a detection diagnostic, not the metric."""
    c = np.asarray(cs, float).reshape(-1, 3)
    m = np.asarray(ss, float).reshape(-1) >= thr
    c = c[m]
    if len(c) == 0 or len(gxyz) == 0:
        return np.zeros(len(gxyz), bool), len(c)
    tr = cKDTree(c * SCALE[None, :])
    d, _ = tr.query(gxyz * SCALE[None, :])
    return d <= MATCH_UM, len(c)


print(f"{chr(10)}=== RECALL AND OUTPUT SIZE ===", flush=True)
print(f"  {'arm':<22} {'recall':>8} {'kept/frame':>12}", flush=True)


def _sweep(get, thr, label):
    hit_tot = n_tot = kept = frames = 0
    for _st, T, g_t, g_zyx, _tid, *dd in _ROWS:
        d = dict(ours=dd[0], dc1=dd[1], dc4=dd[2])[get]
        for t in range(T):
            sel = g_t == t
            h, k = _hit(d[t][0], d[t][1], thr, g_zyx[sel])
            hit_tot += int(h.sum())
            n_tot += int(sel.sum())
            kept += k
            frames += 1
    print(f"  {label:<22} {hit_tot / max(1, n_tot):>8.4f} "
          f"{kept / max(1, frames):>12.1f}", flush=True)
    return hit_tot / max(1, n_tot)


_r_ours = _sweep("ours", UNET_THRESH, f"OURS @ {UNET_THRESH}")
for _thr in DC_THRESHOLDS:
    _sweep("dc1", _thr, f"DeepCenter 1x @ {_thr}")
for _thr in DC_THRESHOLDS:
    _sweep("dc4", _thr, f"DeepCenter 4-flip @ {_thr}")

# ---- THE ACTIONABLE NUMBER --------------------------------------------
print(f"{chr(10)}=== DOES DEEPCENTER COVER WHAT WE MISS? ===", flush=True)
print("  The denominator is the GT cells OUR detector misses at the "
      "shipped threshold. flicker.py measured the same quantity for our "
      "own sub-threshold band [0.05, 0.15) and got 20.2%. Anything much "
      "above that is a reason to pipe DeepCenter into _gap_close's "
      "repair pool; anything at or below it is not.", flush=True)
print(f"  {'DeepCenter arm':<24} {'covers OUR misses':>18} "
      f"{'we cover ITS misses':>20}", flush=True)
for _tag, _lab in (("dc1", "1x"), ("dc4", "4-flip")):
    for _thr in DC_THRESHOLDS:
        a = b = c_ = d_ = 0
        for _st, T, g_t, g_zyx, _tid, *dd in _ROWS:
            mine, theirs = dd[0], dict(dc1=dd[1], dc4=dd[2])[_tag]
            for t in range(T):
                sel = g_t == t
                gx = g_zyx[sel]
                hm, _ = _hit(mine[t][0], mine[t][1], UNET_THRESH, gx)
                ht, _ = _hit(theirs[t][0], theirs[t][1], _thr, gx)
                a += int((~hm & ht).sum())
                b += int((~hm).sum())
                c_ += int((hm & ~ht).sum())
                d_ += int((~ht).sum())
        print(f"  DeepCenter {_lab} @ {_thr:<8} "
              f"{100.0 * a / max(1, b):>17.1f}% "
              f"{100.0 * c_ / max(1, d_):>19.1f}%", flush=True)

# ---- flicker, which is what we actually lose ---------------------------
print(f"{chr(10)}=== FLICKER: always / intermittent / never ===", flush=True)
print("  Followed along each Zebrahub track (its own track_id). This is "
      "the population _gap_close exists to patch, and the only place a "
      "TEMPORAL model should beat a per-frame one.", flush=True)
print(f"  {'arm':<22} {'always':>8} {'inter':>8} {'never':>8} "
      f"{'gap=1':>7} {'gap>=2':>7}", flush=True)


def _flick(get, thr, label):
    cnt = [0, 0, 0]
    g1 = g2 = 0
    for _st, T, g_t, g_zyx, tid, *dd in _ROWS:
        if tid is None:
            continue
        d = dict(ours=dd[0], dc1=dd[1], dc4=dd[2])[get]
        by = defaultdict(list)
        for k in range(len(tid)):
            by[int(tid[k])].append(k)
        trees = {}
        for t in range(T):
            c = np.asarray(d[t][0], float).reshape(-1, 3)
            m = np.asarray(d[t][1], float).reshape(-1) >= thr
            c = c[m]
            trees[t] = cKDTree(c * SCALE[None, :]) if len(c) else None
        for _k, rows in by.items():
            rows.sort(key=lambda r_: g_t[r_])
            if len(rows) < 3:
                continue
            seen = []
            for r_ in rows:
                tr = trees.get(int(g_t[r_]))
                seen.append(bool(tr is not None
                                 and tr.query(g_zyx[r_] * SCALE)[0] <= MATCH_UM))
            if all(seen):
                cnt[0] += 1
            elif not any(seen):
                cnt[2] += 1
            else:
                cnt[1] += 1
                run = 0
                for v in seen:
                    if not v:
                        run += 1
                    elif run:
                        g1 += (run == 1)
                        g2 += (run >= 2)
                        run = 0
                if run:
                    g1 += (run == 1)
                    g2 += (run >= 2)
    n = max(1, sum(cnt))
    print(f"  {label:<22} {100.0 * cnt[0] / n:>7.1f}% "
          f"{100.0 * cnt[1] / n:>7.1f}% {100.0 * cnt[2] / n:>7.1f}% "
          f"{g1:>7,} {g2:>7,}", flush=True)


_flick("ours", UNET_THRESH, f"OURS @ {UNET_THRESH}")
for _thr in DC_THRESHOLDS:
    _flick("dc4", _thr, f"DeepCenter 4-flip @ {_thr}")

print(f"{chr(10)}=== reading it ===", flush=True)
print("  Recall is NOT the number to read: logbase put our detector at "
      "0.9813 against the best classical filter's 0.9247, so there is "
      "1.87% of recall left, not 5%. Read the coverage table and the "
      "flicker table.", flush=True)
print("  A high 'covers OUR misses' with a LOW 'we cover ITS misses' "
      "would mean the two detectors fail on different cells, which is "
      "the case for a union. Both high would mean they fail on the same "
      "cells and there is nothing to gain.", flush=True)
print(f"{chr(10)}  {time.time() - _t0:.0f}s", flush=True)
print("DETCMP DONE", flush=True)
