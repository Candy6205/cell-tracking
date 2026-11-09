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

SCRIPT_VERSION = "2026-09-17a-locnorm"
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

# ---- LOCNORM ------------------------------------------------------------------------
LN_MODE = "GLOBAL"
LN_SIGMA = {"BGSUB15": 15.0, "LOCZ15": 15.0, "LOCZ30": 30.0}
PSCALE_UM = SCALE[0]                       # the pooled grid is isotropic 1.625 um
ARMS_LN = ["GLOBAL", "BGSUB15", "LOCZ15", "LOCZ30"]
MISS_UM = 7.0
CTRL_MATCH_UM = 2.5
LONE_PRIM_UM = 8.0
LONE_GT_UM = 10.0
N_CTRL_PER_FRAME = 10
MIN_MISSED_PER_MOVIE = 30
_pool_xy_raw = pool_xy


def _gauss3d(p, sigma_vox):
    """Separable Gaussian blur of a (Z, Y, X) float32 array on DEVICE, reflect padding."""
    t = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))[None, None].to(DEVICE)
    r = int(np.ceil(2.5 * sigma_vox))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32, device=DEVICE) / sigma_vox) ** 2)
    k = k / k.sum()
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = r
        pad[2 * (4 - dim) + 1] = r
        n = t.shape[dim]
        if n <= r:                                  # a thin axis: pad by replication instead
            t = torch.nn.functional.pad(t, pad, mode="replicate")
        else:
            t = torch.nn.functional.pad(t, pad, mode="reflect")
        t = torch.nn.functional.conv3d(t, k.view(*shape))
    return t[0, 0].cpu().numpy()


def local_norm(p, mode):
    if mode == "GLOBAL":
        return p
    s_vox = LN_SIGMA[mode] / PSCALE_UM
    mu = _gauss3d(p, s_vox)
    if mode == "BGSUB15":
        return (p - mu).astype(np.float32)
    var = _gauss3d(p * p, s_vox) - mu * mu
    sd = np.sqrt(np.clip(var, 0.0, None))
    return ((p - mu) / (sd + 0.1 * float(p.std()) + 1e-6)).astype(np.float32)


def pool_xy(vol, f=POOL):                   # shadows the name pool_norm / branch_maps4 look up at call time
    return local_norm(_pool_xy_raw(vol, f), LN_MODE)


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    off = (POOL - 1) / 2.0
    return np.array([zyx[0], (zyx[1] - off) / POOL, (zyx[2] - off) / POOL])


def map_value(h, p_vox):
    z, y, x = (int(round(v)) for v in to_pooled(p_vox))
    Z, Y, X = h.shape
    if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
        return float(h[z, y, x])
    return float("nan")


def raw_max_within(h, p_vox, R_um, thr=0.05):
    pA = to_pooled(p_vox)
    r = int(np.ceil(R_um / PSCALE_UM)) + 1
    Z, Y, X = h.shape
    z0, y0, x0 = (int(round(v)) for v in pA)
    zs, ys, xs = slice(max(0, z0 - r), min(Z, z0 + r + 1)), slice(max(0, y0 - r), min(Y, y0 + r + 1)), slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = h[zs, ys, xs]
    if sub.size == 0:
        return 0.0, float("nan")
    pk = peak_local_max(sub, min_distance=1, threshold_abs=thr, exclude_border=False)
    if len(pk) == 0:
        return 0.0, float("nan")
    pk = pk + np.array([zs.start, ys.start, xs.start])
    d = np.linalg.norm((pk - pA[None, :]) * PSCALE_UM, axis=1)
    k = int(np.argmin(d))
    return float(d[k] <= R_um), float(sub.max())


# ---- sites -------------------------------------------------------------------------------
FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
RNG = np.random.default_rng(0)
SITES = {}          # name -> {t: [(kind, g_vox)]}
COUNT = defaultdict(int)
print("building sites from the GT and the F4HP caches...", flush=True)
for name in SIX:
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1)
    per_t = defaultdict(list)
    n_miss = 0
    for t in range(T):
        rows = np.where(g_t == t)[0]
        if len(rows) == 0:
            continue
        c = np.asarray(z[f"c{t}"], float)
        s = np.asarray(z[f"s{t}"], float)
        C = c[s >= UNET_THRESH]
        if len(C) == 0:
            continue
        Cu = C * SCALE
        Gu = g_pos[rows] * SCALE
        ptree, gtree = cKDTree(Cu), cKDTree(Gu)
        d_near, i_near = ptree.query(Gu)
        miss = np.where(d_near > MISS_UM)[0]
        for k in miss:
            per_t[t].append(("missed", g_pos[rows[k]]))
        n_miss += len(miss)
        if len(miss):
            pool = []
            for k in np.where(d_near <= CTRL_MATCH_UM)[0]:
                i = int(i_near[k])
                if len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1 == 0 and len(gtree.query_ball_point(Gu[k], LONE_GT_UM)) - 1 == 0:
                    pool.append(("ctrl", g_pos[rows[k]]))
            if len(pool) > N_CTRL_PER_FRAME:
                pool = [pool[i] for i in RNG.choice(len(pool), N_CTRL_PER_FRAME, replace=False)]
            per_t[t].extend(pool)
    if n_miss >= MIN_MISSED_PER_MOVIE:
        SITES[name] = dict(per_t)
        COUNT["movies"] += 1
        COUNT["missed"] += n_miss
        COUNT["ctrl"] += sum(1 for t in per_t for k_, _g in per_t[t] if k_ == "ctrl")
        COUNT["frames"] += len(per_t)
print(f"sites: {dict(COUNT)} (movies with >= {MIN_MISSED_PER_MOVIE} missed cells)", flush=True)
assert COUNT["movies"] > 0

# ---- the pass ------------------------------------------------------------------------------
OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "locnorm_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t,gz,gy,gx," + ",".join(f"{a}_{k}" for a in ARMS_LN for k in ("d_det", "h_g", "rawmax5", "hmax5")) + "\n")
ff = open(OUT / "locnorm_frames.csv", "w", encoding="utf-8", newline="\n")
ff.write("dataset,t,n_gt," + ",".join(f"{a}_{k}" for a in ARMS_LN for k in ("n_det", "n_new", "n_new_ann", "n_det_ann")) + "\n")
RES = {a: defaultdict(list) for a in ARMS_LN}          # arm -> kind -> rows
FR = {a: defaultdict(float) for a in ARMS_LN}
ANCHOR_BAD = 0
n_frames = 0
t0 = time.time()
for k_, name in enumerate(sorted(SITES)):
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    zc = np.load(FDET[name])
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos_all = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    tm = time.time()
    for t in sorted(SITES[name]):
        vol = np.asarray(img[t])
        gt_t = g_pos_all[g_t == t]
        gtree = cKDTree(gt_t) if len(gt_t) else None
        det = {}
        for arm in ARMS_LN:
            LN_MODE = arm
            hs = branch_maps4(vol)
            h = sum(hs) / 4.0
            cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
            det[arm] = (ch[s >= UNET_THRESH] if len(ch) else np.zeros((0, 3)), h)
            if arm == "GLOBAL":
                o = np.asarray(zc[f"c{t}"], np.float32)
                if o.shape != np.asarray(ch, np.float32).shape or not np.array_equal(o, np.asarray(ch, np.float32)):
                    ANCHOR_BAD += 1
            del hs
        LN_MODE = "GLOBAL"
        g_tree = cKDTree(det["GLOBAL"][0] * SCALE) if len(det["GLOBAL"][0]) else None
        frame_row = [name, str(t), str(len(gt_t))]
        for arm in ARMS_LN:
            D, h = det[arm]
            Du = D * SCALE
            n_det = len(D)
            if g_tree is not None and n_det:
                dd, _ = g_tree.query(Du)
                new = dd > MISS_UM
            else:
                new = np.ones(n_det, bool)
            n_new = int(new.sum())
            if gtree is not None and n_det:
                dg, _ = gtree.query(Du)
                n_det_ann = int((dg <= MISS_UM).sum())
                n_new_ann = int(((dg <= MISS_UM) & new).sum())
            else:
                n_det_ann = n_new_ann = 0
            for k2, v in (("n_det", n_det), ("n_new", n_new), ("n_new_ann", n_new_ann), ("n_det_ann", n_det_ann)):
                FR[arm][k2] += v
            frame_row += [str(n_det), str(n_new), str(n_new_ann), str(n_det_ann)]
        ff.write(",".join(frame_row) + "\n")
        for kind, g in SITES[name][t]:
            row = [kind, name, str(t)] + [f"{v:.3f}" for v in g]
            for arm in ARMS_LN:
                D, h = det[arm]
                d_det = float(cKDTree(D * SCALE).query(g * SCALE)[0]) if len(D) else float("nan")
                h_g = map_value(h, g)
                rm5, hmax5 = raw_max_within(h, g, 5.0)
                RES[arm][kind].append((d_det, h_g, rm5, hmax5))
                row += [f"{d_det:.3f}", f"{h_g:.4f}", f"{rm5:.0f}", f"{hmax5:.4f}"]
            fh.write(",".join(row) + "\n")
        n_frames += 1
        del vol, det
        if n_frames % 50 == 0:
            gc.collect()
    fh.flush()
    ff.flush()
    print(f"  [{k_ + 1}/{len(SITES)}] {name}: {len(SITES[name])} frames, anchor misses so far {ANCHOR_BAD}, "
          f"{time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if k_ == 0:
        assert ANCHOR_BAD == 0, "ANCHOR FAILED on the first movie: GLOBAL does not reproduce fdet_F4hp_"
        print("  ANCHOR OK on movie 1: GLOBAL reproduces the fdet_F4hp_ cache exactly", flush=True)
fh.close()
ff.close()

# ---- summary ---------------------------------------------------------------------------------
print(f"\n=== LOCNORM: {n_frames} frames, {len(SITES)} movies; anchor misses {ANCHOR_BAD} ===", flush=True)
print(f"  {'arm':<8}{'missed found<=7um':>18}{'found<=5um':>11}{'h at cell med':>14}{'raw max<=5um':>13}{'hmax5 med':>10} | "
      f"{'ctrl found<=2.5':>15}{'ctrl d med':>11}{'ctrl lost>7':>12}", flush=True)
for arm in ARMS_LN:
    M = np.array(RES[arm]["missed"], float).reshape(-1, 4)
    C = np.array(RES[arm]["ctrl"], float).reshape(-1, 4)
    print(f"  {arm:<8}{np.mean(M[:, 0] <= 7):>18.3f}{np.mean(M[:, 0] <= 5):>11.3f}{np.nanmedian(M[:, 1]):>14.3f}{np.mean(M[:, 2] == 1):>13.3f}{np.nanmedian(M[:, 3]):>10.3f} | "
          f"{np.mean(C[:, 0] <= 2.5):>15.3f}{np.nanmedian(C[:, 0]):>11.2f}{np.mean(C[:, 0] > 7):>12.3f}", flush=True)
print(f"\n  per frame, summed: detections >= 0.15; 'new' = not within 7 um of a GLOBAL detection; annotated = within 7 um of a GT node", flush=True)
print(f"  {'arm':<8}{'n_det':>10}{'vs GLOBAL':>10}{'n_new':>9}{'new annotated':>14}{'new ann share':>14}{'all ann share':>14}", flush=True)
base = FR["GLOBAL"]["n_det"]
for arm in ARMS_LN:
    f_ = FR[arm]
    print(f"  {arm:<8}{f_['n_det']:>10,.0f}{f_['n_det'] / max(base, 1):>10.3f}{f_['n_new']:>9,.0f}{f_['n_new_ann']:>14,.0f}"
          f"{f_['n_new_ann'] / max(f_['n_new'], 1):>14.3f}{f_['n_det_ann'] / max(f_['n_det'], 1):>14.3f}", flush=True)
print("\n  reading: a normalisation that finds a real share of the missed cells (found <= 7 um) while its new peaks are "
      "annotated at about the frame's overall annotated share is adding cells; new peaks annotated far below that share "
      "are noise. Placement at ctrl cells (d med, lost > 7) is the cost on cells already found.", flush=True)
print(f"\nLOCNORM DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("LOCNORM OK" if ANCHOR_BAD == 0 else f"LOCNORM ANCHOR DRIFT on {ANCHOR_BAD} frames", flush=True)
