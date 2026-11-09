"""detfuse3 -- maps 5 and 6: the OFFICIAL detector heatmaps (two seeds) in the blend.

SCRIPT_VERSION 2026-09-03b-detfuse3

Generated from detfuse2.py by make_detfuse3.py. Same models, same
measurement, same frame sampling; two probability maps are added to
the four the shipped F4 detector averages:

  map 5  the official UNetNodeTransformer's detection head, pack split_0
         weights (the primary model of the public 0.948 notebook);
  map 6  the same architecture, the independently trained seed
         pilkwang/biohub-temporal-unet3d-seed314159-v1 (its "secondary",
         which it weights 0.80 in detection after mean/std alignment).

Both are run the official way: frame strided [::1, ::4, ::4], the
dataset's 0.001 / 0.999 intensity quantiles from the zarr attrs (zh
crops carry none -> per-frame quantiles, warned once), a 2-frame
window (t, t+1) (the last frame uses (T-2, T-1) at index 1, as
predict_video does), 4-view flip TTA of the logits, sigmoid. The
official grid is the strided sample at full-res g*4; ours is the 4x4
block mean centred at g*4+1.5, so the two are offset by 1.5 full-res
voxels (0.6 um) in y and x -- inside the +-5-voxel refine window and
the 4 um NMS. Official maps are cropped to our (Z, Y//4, X//4).

nopred (2026-09-03) put 54% of zebrahub's no-predecessor cells on
"parent never detected", 86% of those cold. This is the detection line
again, with a detector the pipeline has never seen.

ARMS over (b0, b1, b2, dc, off_p, off_s): FULL3 and F4 are anchors
(zh F1* at 7 um 0.7774 / 0.7903, detfuse2); F5p / F5s add one official
map, F6 both, F6h both at half weight; OFFp / OFFs / OFF28 are the
official maps alone (OFF28 = the 0.948 notebook's 0.2 / 0.8 mix, in
probability space here rather than aligned logit space).

Attach, in addition to detfuse2's: pilkwang/biohub-tracking-support-
pack-50ep-v1 (code + primary weights + wheels) and pilkwang/biohub-
temporal-unet3d-seed314159-v1. T4.

--- detfuse2's docstring follows ---
detfuse2 -- a FOURTH heatmap: does ep05 DeepCenter rescue the cold misses?

SCRIPT_VERSION 2026-08-29a-detfuse2

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

import gc
import glob
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-09-04a-detfuse3-curves"
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

# ---- the official model code (pack repo) and its wheels -----------------
import os                                                       # noqa: E402
import shutil                                                   # noqa: E402
import subprocess                                               # noqa: E402
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached (push with kag.py)"
SECONDARY = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                         "biohub-temporal-unet3d-seed314159-v1")
if not SECONDARY.exists():
    SECONDARY = pathlib.Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1")
assert SECONDARY.exists(), "second-seed pack not attached"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
_r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                     "--no-deps"] + [str(w) for w in sorted(WH.glob("*.whl"))],
                    capture_output=True, text=True)
print(f"pip exit {_r.returncode}", flush=True)
if _r.returncode != 0:
    print(_r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)
for _p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    assert _p.exists(), _p
    sys.path.insert(0, str(_p))
import predict_unet_transformer as P                            # noqa: E402

_OFF_DEV = torch.device(DEVICE)
OFF = []
for _lab, _wp in (("off_p", PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"),
                  ("off_s", SECONDARY / "weights/unet_transformer/split_0/edge_predictor_best.pth")):
    assert _wp.exists(), _wp
    _m, _win, _dsm = P.load_model(_wp, _OFF_DEV)
    assert _win == 2, (_lab, _win)
    assert tuple(_dsm) == (1, 4, 4), (_lab, _dsm)   # POOL is defined below
    OFF.append((_lab, _m))
    print(f"official detector {_lab}: {_wp} window {_win} downsample {tuple(_dsm)}",
          flush=True)
_OFF_FLIPS = ((), (-1,), (-2,), (-2, -1))


def _off_frame(v):
    """The official strided load: [::1, ::4, ::4] of the full-res frame."""
    return torch.from_numpy(np.ascontiguousarray(
        np.asarray(v)[:, ::POOL, ::POOL]).astype(np.float32))


def official_maps(vol, vol_other, idx, q, pooled_shape):
    """Two sigmoid maps on our pooled grid (cropped), 4-view TTA, from a
    2-frame window in which our frame sits at `idx` (0: (t, t+1);
    1: (t-1, t), the last frame). q = (q_low, q_high)."""
    if vol_other is None:
        warn_once("official_single_frame",
                  "a movie with one frame: the official window needs two -> "
                  "official maps are ZERO there")
        return [np.zeros(pooled_shape, np.float32) for _ in OFF]
    frames = [_off_frame(vol), _off_frame(vol_other)] if idx == 0 \
        else [_off_frame(vol_other), _off_frame(vol)]
    imgs = torch.stack(frames)
    q_low, q_high = q
    imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
    imgs = imgs.unsqueeze(0).to(_OFF_DEV)
    out = []
    Zp, Yp, Xp = pooled_shape
    with torch.no_grad():
        for _lab, m in OFF:
            acc = None
            for dims in _OFF_FLIPS:
                x = imgs.flip(dims) if dims else imgs
                _u, det = m.encode(x)
                d = det[idx]
                if dims:
                    d = d.flip(dims)
                d = d[0, 0].float()
                acc = d if acc is None else acc + d
                del _u, det
            lg = (acc / len(_OFF_FLIPS)).cpu().numpy()
            assert lg.shape[0] == Zp and lg.shape[1] >= Yp and lg.shape[2] >= Xp, \
                (lg.shape, pooled_shape)
            lg = lg[:, :Yp, :Xp]
            out.append((1.0 / (1.0 + np.exp(-np.clip(lg, -60.0, 60.0)))
                        ).astype(np.float32))
    del imgs
    return out

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
from viz_cells import read_quantiles, warn_once                 # noqa: E402

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


def branch_maps6(vol, vol_other, idx, q):
    """Six probability maps: the three xl branches (detfuse verbatim),
    ep05 DeepCenter, then the two official detectors (official_maps),
    each the sigmoid of its 4-flip logit mean."""
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
    hs.extend(official_maps(vol, vol_other, idx, q, hs[0].shape))
    assert len(hs) == 6 and all(h.shape == hs[0].shape for h in hs), \
        [h.shape for h in hs]
    return hs


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


GATES = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
         0.90, 0.95, 0.965, 0.98, 0.99)
RADII = (5.0, 7.0)

# (name, weights over (b0, b1, b2, dc, off_p, off_s)).
ARMS = [
    ("FULL3", (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)),   # anchor: detfuse2 0.7774@7um
    ("F4",    (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)),   # anchor: detfuse2 0.7903 (ships)
    # v3 (2026-09-03) measured F5p / F5s / F6 / F6h: all below F4; dropped
    ("OFFp",  (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),   # official primary alone
    ("OFFs",  (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),   # second seed alone
    ("OFF28", (0.0, 0.0, 0.0, 0.0, 0.2, 0.8)),   # the 0.948 notebook's mix
]
ANCHORS_7UM = {"FULL3": 0.7774, "F4": 0.7903}


def match_flags(coords, sc, gt_um, radius):
    matched = np.zeros(len(coords), bool)
    claimed = set()
    if len(coords) and len(gt_um):
        det_um = coords * SCALE[None, :]
        tree = cKDTree(gt_um)
        for i in np.argsort(-sc):
            near = tree.query_ball_point(det_um[i], radius)
            if not near:
                continue
            near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
            free = [g for g in near if g not in claimed]
            if free:
                claimed.add(free[0])
                matched[i] = True
    tp = np.zeros(len(GATES), np.int64)
    nd = np.zeros(len(GATES), np.int64)
    for k, g in enumerate(GATES):
        m = sc >= g
        nd[k] = int(m.sum())
        tp[k] = int((m & matched).sum())
    return matched, claimed, tp, nd


def run_domain(domain, frame_iter, dense):
    """frame_iter yields (crop_tag, vol, gt_vox). Accumulates per-arm
    per-gate TP/ND at both radii, plus the cold-miss rescue table
    against the FULL3 mean at floor 0.05."""
    TP = {(nm, r): np.zeros(len(GATES), np.int64) for nm, _ in ARMS
          for r in RADII}
    ND = {(nm, r): np.zeros(len(GATES), np.int64) for nm, _ in ARMS
          for r in RADII}
    NG = 0
    NF = 0
    miss = []          # per F4-missed GT: window max of mean4, off_p, off_s
    for tag, vol, gt_vox, vol_other, idx, q in frame_iter:
        gt_um = gt_vox * SCALE[None, :]
        NG += len(gt_vox)
        NF += 1
        hs = branch_maps6(vol, vol_other, idx, q)
        per = {}
        for nm, wts in ARMS:
            coords, sc = detect_weighted(hs, vol, wts)
            per[nm] = (coords, sc)
            for r in RADII:
                _m, _c, tp_, nd_ = match_flags(coords, sc, gt_um, r)
                TP[(nm, r)] += tp_
                ND[(nm, r)] += nd_
        coords, sc = per["F4"]
        _m, claimed, _t, _n = match_flags(coords, sc, gt_um, 7.0)
        h3 = (hs[0] + hs[1] + hs[2] + hs[3]) / 4.0
        hd = hs[4]
        hs_ = hs[5]
        Zp, Yp, Xp = h3.shape
        for j in range(len(gt_vox)):
            if j in claimed:
                continue
            zc = int(round(gt_vox[j, 0]))
            yc = int(round(gt_vox[j, 1] / POOL))
            xc = int(round(gt_vox[j, 2] / POOL))
            sl = (slice(max(0, zc - 2), min(Zp, zc + 3)),
                  slice(max(0, yc - 2), min(Yp, yc + 3)),
                  slice(max(0, xc - 2), min(Xp, xc + 3)))
            if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop \
                    or sl[2].start >= sl[2].stop:
                continue
            miss.append((float(h3[sl].max()), float(hd[sl].max()),
                         float(hs_[sl].max())))
        del hs, per
    # ---- report ------------------------------------------------------
    kind = "ALL cells" if dense else "ANNOTATED cells only"
    print(f"\n=== {domain}: {NG:,} GT points over {NF} frames "
          f"({kind}) ===", flush=True)
    for r in RADII:
        print(f"\n--- match radius {r} um ---", flush=True)
        print(f"  {'arm':<8} {'gate*':>6} {'dets/fr':>8} {'recall':>8} "
              f"{'prec*':>8} {'F1*':>8} {'F1@.15':>8}", flush=True)
        i15 = GATES.index(0.15)
        for nm, _ in ARMS:
            rec = TP[(nm, r)] / max(1, NG)
            pre = TP[(nm, r)] / np.maximum(1, ND[(nm, r)])
            f1 = 2 * rec * pre / np.maximum(1e-9, rec + pre)
            k = int(np.argmax(f1))
            anc = ""
            if dense and r == 7.0 and nm in ANCHORS_7UM:
                ok = abs(float(f1[k]) - ANCHORS_7UM[nm]) <= 0.0005
                anc = f"   anchor {ANCHORS_7UM[nm]:.4f} {'OK' if ok else 'MISS'}"
                if not ok:
                    warn_once(f"anchor_{nm}", f"zh {nm} F1* {f1[k]:.4f} vs "
                              f"detfuse2's {ANCHORS_7UM[nm]:.4f} -- the four "
                              f"shipped maps or the sampling drifted")
            print(f"  {nm:<8} {GATES[k]:>6.2f} "
                  f"{ND[(nm, r)][k] / max(1, NF):>8.1f} {rec[k]:>8.4f} "
                  f"{pre[k]:>8.4f} {f1[k]:>8.4f} {f1[i15]:>8.4f}{anc}",
                  flush=True)
        if r == 7.0:
            print(f"\n  full gate curves at {r} um (gate, dets/fr, recall, prec, F1):",
                  flush=True)
            for nm, _ in ARMS:
                rec = TP[(nm, r)] / max(1, NG)
                pre = TP[(nm, r)] / np.maximum(1, ND[(nm, r)])
                f1 = 2 * rec * pre / np.maximum(1e-9, rec + pre)
                print(f"  {nm:<8} " + " | ".join(
                    f"{g:.3f} {ND[(nm, r)][k] / max(1, NF):.1f} {rec[k]:.4f} "
                    f"{pre[k]:.4f} {f1[k]:.4f}" for k, g in enumerate(GATES)),
                    flush=True)
        if not dense:
            print("  ⚠ sparse GT: `prec*`/F1 are against annotated cells "
                  "only and NOT comparable across domains; within this "
                  "table the ordering is what matters. dets/fr is the "
                  "honest surplus proxy.", flush=True)
    M = np.asarray(miss, np.float32)
    if len(M):
        print(f"\n  cold-miss rescue: {len(M):,} GT points unmatched by "
              f"F4 at 7 um (floor 0.05)", flush=True)
        for g in (0.15, 0.05):
            n4 = int((M[:, 0] >= g).sum())
            npp = int((M[:, 1] >= g).sum())
            nss = int((M[:, 2] >= g).sum())
            either = int(((M[:, 0] >= g) | (M[:, 1] >= g) | (M[:, 2] >= g)).sum())
            print(f"  window-max >= {g:.2f}: 4-map mean {n4:,} "
                  f"({100.0 * n4 / len(M):.1f}%)   off_p {npp:,} "
                  f"({100.0 * npp / len(M):.1f}%)   off_s {nss:,} "
                  f"({100.0 * nss / len(M):.1f}%)   any {either:,} "
                  f"({100.0 * either / len(M):.1f}%)", flush=True)
        print("  an official column minus its overlap with the mean is what a "
              "5/6-map blend can NEWLY surface; window = +-2 pooled "
              "voxels ~ 3.25 um", flush=True)
    return TP, ND, NG


# ======================= zebrahub ======================================
CROPS = None
for _c in ("/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
           "crops", "/kaggle/input/biohub-zebrahub-extract/crops"):
    if pathlib.Path(_c).exists():
        CROPS = pathlib.Path(_c)
        break
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
assert files, f"{CROPS} is empty"
FRAME_STRIDE = 10
t0 = time.time()


def zh_frames():
    done = 0
    for f in files:
        z = np.load(f)
        if "image" not in z.files or "nodes" not in z.files:
            continue
        img = z["image"]
        nodes = np.asarray(z["nodes"], float)
        if img.ndim != 4:
            continue
        T = img.shape[0]
        gbt = defaultdict(list)
        for r_ in range(len(nodes)):
            tt = int(nodes[r_, 0])
            if 0 <= tt < T:
                gbt[tt].append(r_)
        for t in range(0, T, FRAME_STRIDE):
            gi = gbt.get(t, [])
            if not gi:
                continue
            vol = np.asarray(img[t])
            if T < 2:
                other, idx = None, 0
            elif t + 1 < T:
                other, idx = np.asarray(img[t + 1]), 0
            else:
                other, idx = np.asarray(img[t - 1]), 1
            warn_once("zh_quantiles",
                      "zebrahub crops carry no zarr quantiles -> the official "
                      "maps are normalised by each frame's own 0.001/0.999 "
                      "quantiles (the official path uses the dataset's)")
            q = tuple(float(v) for v in np.quantile(vol, [0.001, 0.999]))
            yield f.stem, vol, nodes[gi, 1:], other, idx, q
        done += 1
        print(f"  zh {done}/{len(files)} {f.stem} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del z, img
        gc.collect()


print(f"\n=== zebrahub: {len(files)} crops, every {FRAME_STRIDE}th frame, "
      f"anchors FULL3 0.7774 / F4 0.7903 at 7 um (detfuse2) ===", flush=True)
run_domain("zh", zh_frames(), dense=True)

# ======================= competition ===================================
HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
names_all = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
EX_44B6 = sorted(n for n in names_all
                 if n.startswith("44b6_") and n not in _SEEN)
assert len(EX_44B6) == 12, f"expected 12 excluded 44b6, got {len(EX_44B6)}"
assert all(n not in _SEEN for n in HOLDOUT_6BBA)
COMP_STRIDE = 10


def comp_frames(names):
    for k, name in enumerate(names):
        img = open_image(TRAIN_DIR / f"{name}.zarr")
        gt = read_geff(geff_path(ROOT, "train", name))
        gbt = defaultdict(list)
        for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
            gbt[int(tt)].append([float(z_), float(y_), float(x_)])
        T = img.shape[0]
        qd = read_quantiles(TRAIN_DIR / f"{name}.zarr")
        assert 0.001 in qd and 0.999 in qd, (name, sorted(qd))
        q = (float(qd[0.001]), float(qd[0.999]))
        for t in range(0, T, COMP_STRIDE):
            gi = gbt.get(t)
            if not gi:
                continue
            vol = np.asarray(img[t])
            if T < 2:
                other, idx = None, 0
            elif t + 1 < T:
                other, idx = np.asarray(img[t + 1]), 0
            else:
                other, idx = np.asarray(img[t - 1]), 1
            yield name, vol, np.asarray(gi, float), other, idx, q
        print(f"  comp {k + 1}/{len(names)} {name} "
              f"({time.time() - t0:.0f}s)", flush=True)
        gc.collect()


print(f"\n=== competition: 12 6bba holdout + 12 excluded 44b6, every "
      f"{COMP_STRIDE}th frame, sparse GT ===", flush=True)
run_domain("6bba", comp_frames(HOLDOUT_6BBA), dense=False)
run_domain("44b6", comp_frames(EX_44B6), dense=False)

print("\n=== reading it ===", flush=True)
print("  The decision variable is the competition-domain cold-miss "
      "rescue (edgediag's fat bucket) TOGETHER with dets/fr (the node "
      "surplus the blend would feed the pipeline) and the zh F1 order "
      "(the only full-population check). A blend that rescues misses "
      "while holding dets/fr near FULL3's earns a pipeline test; one "
      "that buys recall by inflating dets/fr is dcmerge again by "
      "another road.", flush=True)
print(f"\nDETFUSE3 DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
