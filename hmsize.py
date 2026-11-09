"""fusecache -- full-frame detection caches for the three winning blends.

SCRIPT_VERSION 2026-08-29a-fusecache

detfuse2 promoted F4 / W1101 / DCheavy (zh dense-GT F1 0.7903 / 0.7913
/ 0.7929 against the shipped detector's 0.7774, with dets/frame at or
below it). This kernel turns those blends into detcache-layout files
(fdet_<arm>_<name>.npz, c{t}/s{t}, floor 0.05) for the 32 eval datasets
-- 12 6bba holdout + 12 manifest-excluded 44b6 + 8 zh crops -- so the
dcpool-style solve harness can score them in three domains with no code
change. One shared 4-heatmap pass per frame feeds all three arms.
Model stack and blending are detfuse2 verbatim (header below).

WHY. edgediag charged the pipeline's loss to detection COLD misses --
GT cells with no response >= 0.05 anywhere within 7 um of them in the
three-branch mean -- at 14.49% of zh edges, ~8x the entire linking-side
loss. dcmerge and dcpool both tried to use DeepCenter AFTER peak
extraction (as points) and closed. Never tried: DeepCenter as a FOURTH
HEATMAP inside the weighted average, BEFORE peak finding -- a dim cell
with no xl response but a DC response can surface as a peak in the
blend, while averaging suppresses DC's own false responses (the same
mechanism that made averaging beat unioning, 0.661, in the public
recipe's own measurement).

THE MODEL is dctrain's epoch-5 checkpoint: clean eval everywhere
(dccache3), covers 48.7% / 67.6% of xl's misses on 6bba / 44b6.

MEASURED IN TWO PLACES:
  * Zebrahub, dense GT (the honest denominator): full recall/precision/
    F1 per weight vector at its own gate, 5.0 um (anchor radius) and
    7.0 um (scorer radius). ANCHORS: FULL3 (1,1,1,0) must reproduce
    detfuse's 0.7774 F1* at 7 um / 0.7539 at 5 um; W110h its 0.7816.
  * The 24 competition eval datasets (12 6bba holdout + 12 excluded
    44b6), sparse GT: recall over ANNOTATED cells per arm, detections
    per frame (the surplus proxy), and the cold-miss rescue count --
    of the annotated cells the three-branch mean misses at floor 0.05,
    how many have a DC window response.

This is DETECTION evidence. A winning arm buys a pipeline test
(re-cache + ranker retrain + three-domain adj_J), not a submission.

Attach: xiaoleilian/biohub-unet3d-weights, ...-v2models,
faustine649/biohub-dctrain-175-weights, faustine649/cell-tracking,
kernel output faustine649/biohub-zebrahub-extract, competition. T4.
"""

import base64
import gc
import glob
import gzip
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-09-25a-hmsize"  # hsaddle was "2026-09-12a-hsaddle"
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
print("device:", DEVICE, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- our code (zarr-free competition IO) ------------------------------
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff, open_image  # noqa: E402

ROOT = find_root(None)
TRAIN_DIR = pathlib.Path(ROOT) / "train"

# ---- xl detect stack (det_ablate3 / detfuse verbatim) -----------------

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
        return np.array([z, y, x], float)
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s

def _nms_keep(coords, scores, radius_um, scale=SCALE):
    """The indices `_physical_nms` would have kept, in the same order.

    Factored out so both placement arms can be built from ONE suppression --
    see the module docstring of make_hparacache.py. `_physical_nms` below is
    now defined in terms of this, so the two cannot drift apart.
    """
    if len(coords) <= 1:
        return np.arange(len(coords))
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
    return np.array(keep)


def _physical_nms(coords, scores, radius_um, scale=SCALE):
    keep = _nms_keep(coords, scores, radius_um, scale)
    if len(coords) <= 1:
        return coords, scores
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


# ---- the fourth branch: dctrain epoch-5 DeepCenter --------------------
class _DCConvBlock3d(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        groups = min(8, co)
        self.block = nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1, bias=False),
            nn.GroupNorm(groups, co), nn.SiLU(inplace=True),
            nn.Conv3d(co, co, 3, padding=1, bias=False),
            nn.GroupNorm(groups, co), nn.SiLU(inplace=True))

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


_ep05 = sorted(pathlib.Path("/kaggle/input").glob(
    "datasets/*/biohub-dctrain-175-weights/best_ep05.pt")) or sorted(
    pathlib.Path("/kaggle/input").glob(
        "*/biohub-dctrain-175-weights/best_ep05.pt"))
assert _ep05, "best_ep05.pt not found -- attach biohub-dctrain-175-weights"
_ck = torch.load(_ep05[0], map_location=DEVICE, weights_only=False)
assert int(_ck.get("epoch", -1)) == 5, f"expected epoch 5, got {_ck.get('epoch')}"
_cfg = dict(_ck.get("config", {}))
DC = _DCDeepCenterUNet3D(base_channels=int(_cfg.get("base_channels", 24)))
DC.load_state_dict(_ck["model_state"])
DC = DC.to(DEVICE).eval()
DC_NORM = (float(_cfg.get("norm_lo_pct", 50.0)),
           float(_cfg.get("norm_hi_pct", 99.5)),
           float(_cfg.get("norm_clip_lo", -0.5)),
           float(_cfg.get("norm_clip_hi", 6.0)))
print(f"ep05 DeepCenter loaded from {_ep05[0]} (epoch "
      f"{_ck.get('epoch')}, norm {DC_NORM})", flush=True)
del _ck

# the training split, to assert the eval sets are outside it
_man = _ep05[0].parent / "split_manifest.json"
assert _man.exists(), "split_manifest.json missing beside best_ep05.pt"
_SEEN = set(json.loads(_man.read_text())["all"])


def _dc_normalize(volume):
    lo_pct, hi_pct, clip_lo, clip_hi = DC_NORM
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, lo_pct))
    hi = float(np.percentile(vol, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


def branch_maps4(vol):
    """Four probability maps: the three xl branches (detfuse verbatim)
    plus ep05 DeepCenter, each the sigmoid of its 4-flip logit mean."""
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
    x = _dc_normalize(pool_xy(vol))
    acc = None
    for i in _FLIP_VIEWS:
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = DC(torch.from_numpy(xv)[None, None].to(DEVICE)
                    )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    hs.append(1.0 / (1.0 + np.exp(-np.clip(acc / len(_FLIP_VIEWS),
                                           -60.0, 60.0))))
    return hs


def _para1(a, b, c):
    """Sub-voxel offset of the vertex of the parabola through (-1,a) (0,b)
    (1,c). Zero when the three are colinear or b is not the largest.

    Copied verbatim from refinecmp.py, which is where the measurement that
    justifies this kernel was made. Do not "simplify" it: the `d >= -1e-12`
    guard is what stops a flat window or an upward cusp from dragging the
    detection into a MINIMUM.
    """
    d = a - 2.0 * b + c
    if d >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def _hpara_shift(h, pk):
    """pk: (n,3) pooled integer peaks -> (n,3) pooled float coordinates."""
    Zp, Yp, Xp = h.shape
    zyx = pk.astype(float)
    for i in range(len(pk)):
        z0, y0, x0 = (int(v) for v in pk[i])
        b = float(h[z0, y0, x0])
        for ax, (hi, c0) in enumerate(((Zp, z0), (Yp, y0), (Xp, x0))):
            if c0 - 1 < 0 or c0 + 1 >= hi:
                continue
            im = [z0, y0, x0]
            ip = [z0, y0, x0]
            im[ax] -= 1
            ip[ax] += 1
            zyx[i][ax] += _para1(float(h[tuple(im)]), b, float(h[tuple(ip)]))
    return zyx


def detect_both(hs, vol, weights):
    """(refine coords, hpara coords, scores) for ONE shared NMS.

    Returns two placements of the SAME surviving detections, so the arms
    differ in position and in nothing else.
    """
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        z3 = np.zeros((0, 3))
        return z3, z3, np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)

    def _unpool(zyx):
        c = zyx.copy()
        c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
        c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
        return c

    ref = np.array([_refine(vol, c) for c in _unpool(pk.astype(float))])
    hpa = np.array([_refine(vol, c) for c in _unpool(_hpara_shift(h, pk))])
    keep = _nms_keep(ref, sc, NMS_UM)
    return ref[keep], hpa[keep], sc[keep]


def detect_weighted(hs, vol, weights):
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c) for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


GATES_UNUSED = None   # no gate sweep here; the cache carries the floor

# The three arms detfuse2 promoted, cached from ONE shared 4-heatmap
# pass per frame. Weights over (b0, b1, b2, ep05-DC).
ARMS = [
    ("F4",      (1.0, 1.0, 1.0, 1.0)),
]

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# ---- HSADDLE: the heatmap between the two peaks of a pair -------------------

UNPOOL_OFF = (POOL - 1) / 2.0          # detect_both: c_pooled * POOL + 1.5
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])   # pooled grid, um per voxel
assert abs(PSCALE[0] - PSCALE[1]) < 1e-9, PSCALE                  # isotropic 1.625 um


def to_pooled(zyx):
    """full-resolution voxel coords (z, y, x) -> pooled grid coords."""
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def hval(h, p):
    return float(ndi.map_coordinates(h, np.asarray(p, float).reshape(3, 1), order=1, mode="nearest")[0])



# ---- hmsize: the heatmap measures -----------------------------------------------------------------------------------
HM_FLOOR = 0.05
HM_KEYS = ["hpk", "hr50", "hr20", "hmass3", "hmass5", "hprof", "hmom"]
HM_RATIO = ["hr50", "hr20", "hmass5", "hprof", "hmom"]
BUDGET_S = 10.0 * 3600


def hm_measures(h, zyx_full):
    """the heatmap measures of one node; h = the pooled F4 map, zyx_full = full-resolution voxel coords"""
    out = {k_: np.nan for k_ in HM_KEYS}
    p = to_pooled(zyx_full)
    Z, Y, X = h.shape
    r = int(np.ceil(8.0 / PSCALE[0]))
    z0, y0, x0 = (int(round(float(v))) for v in p)
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return out
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = np.asarray(h[zs, ys, xs], np.float32)
    zz, yy, xx = np.mgrid[zs, ys, xs]
    d = np.sqrt(((zz - p[0]) * PSCALE[0]) ** 2 + ((yy - p[1]) * PSCALE[1]) ** 2 + ((xx - p[2]) * PSCALE[2]) ** 2)
    near = d <= 2.0
    if not near.any():
        return out
    j = np.unravel_index(int(np.argmax(np.where(near, sub, -np.inf))), sub.shape)
    pk = float(sub[j])
    out["hpk"] = pk
    if pk <= HM_FLOOR:
        return out
    dp = np.sqrt(((zz - zz[j]) * PSCALE[0]) ** 2 + ((yy - yy[j]) * PSCALE[1]) ** 2 + ((xx - xx[j]) * PSCALE[2]) ** 2)
    vox = float(np.prod(PSCALE))

    def _rad(mask):
        if not mask[j]:
            return np.nan
        lab_, _n = ndi.label(mask)
        return float((3.0 * int((lab_ == lab_[j]).sum()) * vox / (4.0 * np.pi)) ** (1.0 / 3.0))

    out["hr50"] = _rad(sub >= 0.5 * pk)
    out["hr20"] = _rad(sub >= 0.2 * pk)
    w_ = np.clip(sub - HM_FLOOR, 0, None)
    out["hmass3"] = float((w_ * (dp <= 3.0)).sum())
    out["hmass5"] = float((w_ * (dp <= 5.0)).sum())
    lab_, _n = ndi.label(sub >= 0.5 * pk)
    comp = lab_ == lab_[j]
    wc = sub[comp]
    out["hmom"] = float(np.sqrt((wc * dp[comp] ** 2).sum() / wc.sum())) if wc.sum() > 0 else np.nan
    shells = np.arange(0.0, 8.0, 1.0)
    prof = np.full(len(shells) - 1, np.nan)
    for i_, (a_, b_) in enumerate(zip(shells[:-1], shells[1:])):
        m_ = (dp >= a_) & (dp < b_)
        if m_.any():
            prof[i_] = float(sub[m_].mean())
    far_ = prof[5:7]
    far = float(np.nanmean(far_)) if np.isfinite(far_).any() else np.nan
    if np.isfinite(far) and pk > far:
        half = far + 0.5 * (pk - far)
        r_ = np.nan
        for i_ in range(len(prof)):
            if np.isfinite(prof[i_]) and prof[i_] < half:
                if i_ == 0:
                    r_ = 0.5
                else:
                    p0, p1 = prof[i_ - 1], prof[i_]
                    x0_, x1_ = shells[i_ - 1] + 0.5, shells[i_] + 0.5
                    r_ = (x0_ + (p0 - half) / (p0 - p1) * (x1_ - x0_)) if (np.isfinite(p0) and p0 > p1) else x1_
                break
        out["hprof"] = float(r_)
    return out


def fork_tables():
    """forkprop_sz2's fork tables with coordinates: [(table, movie, p, q, label, t, zyx_p, zyx_c1, zyx_q)]"""
    rows = []
    for tab in ("CR_NOCAP_DV150", "dw55"):
        hit = sorted(pathlib.Path("/kaggle/input").glob(f"**/forkssz2_{tab}.csv"))
        assert hit, f"attach the kernel faustine649/biohub-forkprop-sz2 (forkssz2_{tab}.csv)"
        with open(hit[0], encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split(",")
            ci = {c_: i_ for i_, c_ in enumerate(hdr)}
            need = ["movie", "p", "q", "label", "t", "z_p", "y_p", "x_p", "z_c1", "y_c1", "x_c1", "z_q", "y_q", "x_q"]
            assert all(c_ in ci for c_ in need), (tab, [c_ for c_ in need if c_ not in ci])
            for line in fh:
                f_ = line.rstrip("\n").split(",")
                if len(f_) < len(hdr):
                    continue
                g = lambda c_: f_[ci[c_]]                                    # noqa: E731
                rows.append((tab, g("movie"), int(g("p")), int(g("q")), int(g("label")), int(float(g("t"))),
                             tuple(float(g(c_)) for c_ in ("z_p", "y_p", "x_p")),
                             tuple(float(g(c_)) for c_ in ("z_c1", "y_c1", "x_c1")),
                             tuple(float(g(c_)) for c_ in ("z_q", "y_q", "x_q"))))
    return rows


def node_key(t, zyx):
    return (int(t),) + tuple(int(round(float(v) * 4)) for v in zyx)          # quarter-voxel rounding


if __name__ == "__main__":
    t_all = time.time()
    ROWS = fork_tables()
    NODES = defaultdict(set)                                  # movie -> {node_key}
    for (_tab, mv, _p, _q, _l, t, zp, zc, zq) in ROWS:
        NODES[mv].add(node_key(t, zp))
        NODES[mv].add(node_key(t + 1, zc))
        NODES[mv].add(node_key(t + 1, zq))
    n_nodes = sum(len(v) for v in NODES.values())
    print(f"fork rows {len(ROWS):,} ({sum(1 for r in ROWS if r[0] == 'CR_NOCAP_DV150'):,} arm + {sum(1 for r in ROWS if r[0] == 'dw55'):,} REF) "
          f"-> {n_nodes:,} distinct nodes in {len(NODES)} movies", flush=True)
    OUT = pathlib.Path("/kaggle/working")
    MEAS = {}                                                 # (movie, node_key) -> measures
    fn = open(OUT / "hmsize_nodes.csv", "w", encoding="utf-8", newline="\n")
    fn.write("movie,t,z,y,x," + ",".join(HM_KEYS) + "\n")
    t0 = time.time()
    n_frames = 0
    done_movies = 0
    for k_, mv in enumerate(sorted(NODES)):
        if time.time() - t_all > BUDGET_S:
            print(f"[budget] stopped before {mv}: {done_movies} movies done", flush=True)
            break
        img = open_image(TRAIN_DIR / f"{mv}.zarr")
        by_t = defaultdict(list)
        for nk in NODES[mv]:
            by_t[nk[0]].append(nk)
        tm = time.time()
        for t in sorted(by_t):
            if t < 0 or t >= img.shape[0]:
                continue
            vol = np.asarray(img[t])
            hs_ = branch_maps4(vol)
            h = sum(hs_) / 4.0                                # F4: equal weights
            for nk in by_t[t]:
                zyx = np.array(nk[1:], float) / 4.0
                m_ = hm_measures(h, zyx)
                MEAS[(mv, nk)] = m_
                fn.write(f"{mv},{t},{zyx[0]:.3f},{zyx[1]:.3f},{zyx[2]:.3f}," + ",".join(f"{m_[c_]:.5g}" if m_[c_] == m_[c_] else "nan" for c_ in HM_KEYS) + "\n")
            n_frames += 1
            del hs_, h, vol
        fn.flush()
        done_movies += 1
        gc.collect()
        print(f"  [{k_ + 1}/{len(NODES)}] {mv}: {len(by_t)} frames, {len(NODES[mv])} nodes, {time.time() - tm:.0f}s "
              f"(total {(time.time() - t0) / 60:.1f} min)", flush=True)
    fn.close()
    with open(OUT / "hmsize_forks.csv", "w", encoding="utf-8", newline="\n") as ff:
        cols = [f"{k_}_{s_}" for k_ in HM_KEYS for s_ in ("p", "c1", "q")] + [f"{k_}_{s_}" for k_ in HM_RATIO for s_ in ("max", "sum")]
        ff.write("table,movie,p,q,label,t," + ",".join(cols) + "\n")
        n_w = 0
        for (tab, mv, p, q, lab, t, zp, zc, zq) in ROWS:
            mp, mc, mq = MEAS.get((mv, node_key(t, zp))), MEAS.get((mv, node_key(t + 1, zc))), MEAS.get((mv, node_key(t + 1, zq)))
            if mp is None or mc is None or mq is None:
                continue
            vals = []
            for k_ in HM_KEYS:
                vals += [mp[k_], mc[k_], mq[k_]]
            for k_ in HM_RATIO:
                a_, b_, c_ = mp[k_], mc[k_], mq[k_]
                ok_ = np.isfinite(a_) and np.isfinite(b_) and np.isfinite(c_) and a_ > 0
                vals += [(max(b_, c_) / a_) if ok_ else np.nan, ((b_ + c_) / a_) if ok_ else np.nan]
            ff.write(f"{tab},{mv},{p},{q},{lab},{t}," + ",".join(f"{v:.5g}" if v == v else "nan" for v in vals) + "\n")
            n_w += 1
    print(f"\nhmsize_forks.csv: {n_w:,} fork rows of {len(ROWS):,}; hmsize_nodes.csv: {len(MEAS):,} nodes over {n_frames:,} frames", flush=True)
    # a first look: the counted forks of the arm, TP against FP, medians per measure
    arm = [(r, MEAS.get((r[1], node_key(r[5], r[6]))), MEAS.get((r[1], node_key(r[5] + 1, r[7]))), MEAS.get((r[1], node_key(r[5] + 1, r[8]))))
           for r in ROWS if r[0] == "CR_NOCAP_DV150" and r[4] >= 1]
    arm = [a_ for a_ in arm if a_[1] is not None and a_[2] is not None and a_[3] is not None]
    print(f"\n=== THE ARM'S COUNTED FORKS ({len(arm)}; TP {sum(1 for a_ in arm if a_[0][4] == 2)}): median of each measure, "
          f"mother / daughter c1 / daughter q, TP against FP; AUC of the max ratio (TP vs FP) ===", flush=True)
    from sklearn.metrics import roc_auc_score
    for k_ in HM_KEYS:
        line = f"  {k_:<7}"
        for lab_, nm_ in ((2, "TP"), (1, "FP")):
            sel = [a_ for a_ in arm if a_[0][4] == lab_]
            med = [float(np.nanmedian([a_[i_][k_] for a_ in sel])) if sel else float("nan") for i_ in (1, 2, 3)]
            line += f"  {nm_} {med[0]:.3g} / {med[1]:.3g} / {med[2]:.3g}"
        if k_ in HM_RATIO:
            y_, s_ = [], []
            for a_ in arm:
                v0, v1, v2 = a_[1][k_], a_[2][k_], a_[3][k_]
                if np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2) and v0 > 0:
                    y_.append(int(a_[0][4] == 2)); s_.append(max(v1, v2) / v0)
            if y_ and 0 < sum(y_) < len(y_):
                line += f"  | max-ratio AUC {roc_auc_score(y_, s_):.3f} (n {len(y_)})"
        print(line, flush=True)
    print(f"HMSIZE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("HMSIZE OK", flush=True)
