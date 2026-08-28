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

SCRIPT_VERSION = "2026-08-28c-dcprobe"
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
# v1 GUESSED the normalisation from the public notebook's fallback
# defaults and got 0.4 detections per frame with a logit overflow. Read
# what is actually recorded before assuming anything.
print("  checkpoint keys:", flush=True)
for _k, _v in (_dc_ck.items() if isinstance(_dc_ck, dict) else []):
    if hasattr(_v, "shape") or (isinstance(_v, dict) and len(_v) > 20):
        print(f"    {_k}: <{type(_v).__name__}"
              f"{' len ' + str(len(_v)) if isinstance(_v, dict) else ''}>",
              flush=True)
    else:
        print(f"    {_k}: {str(_v)[:200]}", flush=True)
for _mf in sorted(DC_CKPT.parent.parent.parent.rglob("*.json"))[:8]:
    try:
        _txt = _mf.read_text()[:1200]
    except Exception:                                          # noqa: BLE001
        continue
    if any(w in _txt for w in ("norm", "pct", "clip", "pool", "percentile")):
        print(f"  --- {_mf.name} ---", flush=True)
        print("   ", _txt.replace(chr(10), " ")[:900], flush=True)
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


# The candidate normalisations. v1 bet the run on the first one.
# `ours` is our own pool_norm, included because if DeepCenter was trained
# on the official pipeline's preprocessing rather than its own, that is
# the one that will work.
DC_NORMS = {
    "pct50_995": lambda v: _dc_normalize(v, 50.0, 99.5, -0.5, 6.0),
    "pct1_999": lambda v: _dc_normalize(v, 1.0, 99.9, -0.5, 6.0),
    "pct0_100": lambda v: _dc_normalize(v, 0.0, 100.0, 0.0, 1.0),
    "ours": lambda v: v,          # replaced below by pool_norm's output
}


# The block that used to choose this was cut away for the probe, and
# dc_detect still names it, so bind the shipped default.
_DC_NORM = "pct50_995"


def dc_heatmap(vol, norm="pct50_995", flips=False):
    """DeepCenter's heatmap, plus the statistics that diagnose it.

    Returns (heatmap, stats). A saturated heatmap is visible in the
    logit range alone and needs no recall number to be recognised.
    """
    if norm == "ours":
        x = pool_norm(vol, "")
    else:
        x = DC_NORMS[norm](_dc_pool_xy(vol))
    views = _FLIP_VIEWS if flips else _FLIP_VIEWS[:1]
    acc = None
    for i in views:
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = DC(torch.from_numpy(xv)[None, None].to(DEVICE)
                    )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    lg = acc / len(views)
    h = 1.0 / (1.0 + np.exp(-np.clip(lg, -60.0, 60.0)))
    st = dict(in_lo=float(x.min()), in_hi=float(x.max()),
              in_mean=float(x.mean()),
              lg_lo=float(lg.min()), lg_hi=float(lg.max()),
              h_hi=float(h.max()), h_mean=float(h.mean()),
              n_over_05=int((h > 0.05).sum()))
    return h, st


def _peaks_from(h, vol):
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


def dc_detect(vol, flips=False):
    """DeepCenter's network, OUR post-processing.

    Everything downstream of the sigmoid is identical to `detect()` --
    the same peak extraction, the same sub-voxel refine, the same 4.0 um
    physical NMS -- so the comparison isolates the network and its
    normalisation. `flips` adds the same 4-view averaging our detector
    gets, because otherwise this compares an ensemble to a single pass.
    """
    h, _ = dc_heatmap(vol, _DC_NORM, flips=flips)
    return _peaks_from(h, vol)


# ======================= the probe =====================================
MATCH_UM = 7.0
N_FRAMES = 3

print(f"{chr(10)}=== DOES DEEPCENTER WORK AT ALL? ===", flush=True)
print("  Two runs were spent reading a table of zeros because this "
      "branch never had an anchor. 6bba is pilkwang's own validation "
      "split for DeepCenter, so it MUST work there. If it does and "
      "Zebrahub does not, Zebrahub is out of distribution for it and "
      "the venue is wrong. If it fails on both, the integration is.",
      flush=True)
print("  ⚠ v2 disproved the normalisation hypothesis: four different "
      "input scalings gave byte-identical logits, because GroupNorm "
      "after a bias-free convolution erases any affine input transform. "
      "Our pool_norm is in fact the same function as DeepCenter's.",
      flush=True)


def _stats(tag, vol):
    h, st = dc_heatmap(vol, "pct50_995", flips=False)
    c, sc = _peaks_from(h, vol)
    oc, osc = detect(vol)
    print(f"  {tag:<34} in {st['in_lo']:>6.2f}..{st['in_hi']:<6.2f} "
          f"logit {st['lg_lo']:>8.2f}..{st['lg_hi']:<8.2f} "
          f"h_max {st['h_hi']:>7.4f}  DC peaks {len(c):>6,}  "
          f"OURS peaks {int((np.asarray(osc) >= UNET_THRESH).sum()):>6,}",
          flush=True)
    return st


# ---- 6bba: DeepCenter's OWN validation split --------------------------
print(f"{chr(10)}  -- 6bba (DeepCenter was model-selected here; it MUST "
      f"work) --", flush=True)
_zs = []
for _pat in ("**/6bba_*.zarr", "**/6bba_*/**/zarr.json"):
    for _p in sorted(pathlib.Path("/kaggle/input").glob(_pat)):
        _zp = _p if _p.is_dir() else _p.parent.parent
        if _zp not in _zs:
            _zs.append(_zp)
    if _zs:
        break
print(f"  found {len(_zs)} 6bba zarr paths", flush=True)
for _zp in _zs[:2]:
    try:
        _meta = read_array_meta(_zp)
    except Exception as _e:                                    # noqa: BLE001
        print(f"  WARN {_zp.name}: {type(_e).__name__}: {str(_e)[:70]}",
              flush=True)
        continue
    for _t in range(min(N_FRAMES, _meta["shape"][0])):
        try:
            _v = np.asarray(load_volume(_zp, _t, _meta))
        except Exception as _e:                                # noqa: BLE001
            print(f"  WARN {_zp.name} t{_t}: {type(_e).__name__}: "
                  f"{str(_e)[:60]}", flush=True)
            continue
        print(f"  [{_zp.name} t{_t}] dtype {_v.dtype} shape {_v.shape} "
              f"range {float(_v.min()):.4g}..{float(_v.max()):.4g}",
              flush=True)
        _stats(f"{_zp.name} t{_t}", _v)

# ---- Zebrahub --------------------------------------------------------
print(f"{chr(10)}  -- Zebrahub (a different instrument) --", flush=True)
for _f in sorted(files)[:2]:
    _img = np.load(_f)["image"]
    for _t in range(min(N_FRAMES, _img.shape[0])):
        _v = np.asarray(_img[_t])
        print(f"  [{_f.stem} t{_t}] dtype {_v.dtype} shape {_v.shape} "
              f"range {float(_v.min()):.4g}..{float(_v.max()):.4g}",
              flush=True)
        _stats(f"{_f.stem} t{_t}", _v)
    del _img
    gc.collect()

print(f"{chr(10)}=== reading it ===", flush=True)
print("  A working DeepCenter has logits reaching POSITIVE values "
      "somewhere and a h_max near 1. Both runs so far gave h_max 0.049, "
      "i.e. the largest logit anywhere was -2.97 -- 'no cell anywhere', "
      "confidently. The OURS column is the positive control: a frame "
      "where OURS also finds nothing is a bad frame, not a bad model.",
      flush=True)
print("DCPROBE DONE", flush=True)
