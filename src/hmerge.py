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
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-09-12a-hmerge"
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
names_all = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
EX_44B6 = sorted(n for n in names_all
                 if n.startswith("44b6_") and n not in _SEEN)
assert len(EX_44B6) == 12, f"expected 12 excluded 44b6, got {len(EX_44B6)}"
assert all(n not in _SEEN for n in HOLDOUT_6BBA)

NAMES_6BBA = [n for n in names_all
              if n.startswith("6bba_") and n not in set(HOLDOUT_6BBA)]
assert len(NAMES_6BBA) == 116, len(NAMES_6BBA)

# dctrain.py:158-163 -- the seeded 44b6 draw the whole bench is built on.
# Hardcoded here AND cross-checked against the names that actually carry an
# fdet_F4_ cache, because a silent change of this list would compare two
# different movie sets and still print a plausible number.
SET_44B6 = [
    "44b6_0b24845f", "44b6_0db75fae", "44b6_33b596bf", "44b6_5740d24b",
    "44b6_87bba6c4", "44b6_8f5ab931", "44b6_9bfa6a0a", "44b6_a2bb48bb",
    "44b6_c15fded2", "44b6_c96cfa10", "44b6_d2f34f90", "44b6_e29f0176",
]
assert len(SET_44B6) == 12 and len(set(SET_44B6)) == 12

# 2026-09-08: EXTENDED FROM 24 TO 140. The 24-movie run answered "does the
# placement change anything" and could not answer "by how much": hparabench
# scores all 128 6bba movies, only 12 of which had hpara coordinates, so 116
# rows were identical between the arms and the score delta landed inside the
# holdout's own 0.002-0.003 noise -- 90% interval [-0.0050, +0.0015], which
# is equally consistent with a real regression and with a small gain. The
# only way to resolve that is to give every scored movie hpara coordinates.
# The 12 44b6 stay in even though this bench never scores them: 18 extra
# minutes, and dropping them would break the downstream override asserts.
NAMES = HOLDOUT_6BBA + NAMES_6BBA + SET_44B6
assert len(NAMES) == 140 and len(set(NAMES)) == 140, len(NAMES)
print(f"NAMES: {len(HOLDOUT_6BBA)} holdout + {len(NAMES_6BBA)} pool 6bba "
      f"+ {len(SET_44B6)} 44b6 = {len(NAMES)}", flush=True)

# ---- the anchor caches ----------------------------------------------------
OLD = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4_*.npz"):
    OLD[_d.name[len("fdet_F4_"):-len(".npz")]] = _d
print(f"anchor caches found: {len(OLD)} fdet_F4_*.npz", flush=True)
_missing = [n for n in NAMES if n not in OLD]
assert not _missing, (
    f"{len(_missing)} of the {len(NAMES)} bench movies have no fdet_F4_ "
    f"cache to "
    f"anchor against -- attach faustine649/biohub-fusecache as a "
    f"kernel_source. Missing: {_missing[:4]}")
ANCHOR_FAILS = []
OUT = pathlib.Path("/kaggle/working")
t0 = time.time()
done = 0

# ---- HMERGE: saddle-keyed merging of hpara primaries ---------------------------
HM_D = 9.5                                  # pair gate, um (hsaddle's site gate)
HM_ARMS = [("hm90", 0.90), ("hm80", 0.80)]
HM_SEG_UM = 2.5                             # segment-centroid radius
UNPOOL_OFF = (POOL - 1) / 2.0
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def from_pooled(p):
    p = np.asarray(p, float)
    return np.array([p[0], p[1] * POOL + UNPOOL_OFF, p[2] * POOL + UNPOOL_OFF])


def hm_pairs(h, coords, scores, thr_prim=UNET_THRESH, D=HM_D):
    """Every pair of primaries within D um: (saddle, i, j), i < j indices into
    coords. Saddle = min of h along the pooled segment / min of the two end
    values (25 samples, trilinear)."""
    prim = np.where(np.asarray(scores) >= thr_prim)[0]
    if len(prim) < 2:
        return []
    P = np.asarray(coords, float)[prim] * SCALE
    pairs = cKDTree(P).query_pairs(D)
    if not pairs:
        return []
    out = []
    ts = np.linspace(0.0, 1.0, 25)
    for i_, j_ in pairs:
        a = to_pooled(coords[prim[i_]])
        b = to_pooled(coords[prim[j_]])
        pts = a[None, :] * (1 - ts)[:, None] + b[None, :] * ts[:, None]
        prof = ndi.map_coordinates(h, pts.T, order=1, mode="nearest")
        ref = float(min(prof[0], prof[-1]))
        sad = float(prof.min() / ref) if ref > 0 else 0.0
        out.append((sad, int(prim[i_]), int(prim[j_])))
    return out


def segment_centroid(h, pa, pb, r_um=HM_SEG_UM, floor=CAND_THR):
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    r = int(np.ceil(r_um / PSCALE[0])) + 1
    lo = np.floor(np.minimum(pa, pb)).astype(int) - r
    hi = np.ceil(np.maximum(pa, pb)).astype(int) + r + 1
    Z, Y, X = h.shape
    zs = slice(max(0, lo[0]), min(Z, hi[0]))
    ys = slice(max(0, lo[1]), min(Y, hi[1]))
    xs = slice(max(0, lo[2]), min(X, hi[2]))
    sub = h[zs, ys, xs]
    zz, yy, xx = np.mgrid[zs, ys, xs]
    P = np.stack([zz, yy, xx], -1).astype(float) * PSCALE
    A_, B_ = pa * PSCALE, pb * PSCALE
    ab = B_ - A_
    L2 = float(ab @ ab)
    tt = np.clip(((P - A_) @ ab) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(P.shape[:-1])
    d = np.linalg.norm(P - (A_ + tt[..., None] * ab), axis=-1)
    w = np.clip(sub - floor, 0, None) * (d <= r_um)
    if w.sum() <= 0:
        return 0.5 * (pa + pb)
    return np.array([(w * zz).sum(), (w * yy).sum(), (w * xx).sum()]) / w.sum()


def hm_apply(h, vol, coords, scores, cand, thr):
    """Greedy merge, highest saddle first, each detection at most once.
    Returns (coords, scores, n_merged): the merged point replaces detection i
    (the pair's first index) and detection j is dropped."""
    coords = np.asarray(coords, float)
    scores = np.asarray(scores, float)
    used = set()
    drop = set()
    new_c = coords.copy()
    new_s = scores.copy()
    n = 0
    for sad, i_, j_ in sorted(cand, key=lambda x: -x[0]):
        if sad < thr:
            break
        if i_ in used or j_ in used:
            continue
        seg = segment_centroid(h, to_pooled(coords[i_]), to_pooled(coords[j_]))
        c_full = from_pooled(seg)
        new_c[i_] = _refine(vol, c_full)
        new_s[i_] = max(scores[i_], scores[j_])
        used.add(i_)
        used.add(j_)
        drop.add(j_)
        n += 1
    keep = np.array([k for k in range(len(coords)) if k not in drop], int)
    return new_c[keep], new_s[keep], n


t0 = time.time()
done = 0
STAT = {a_: defaultdict(int) for a_, _t in HM_ARMS}
print(f"\n=== HMERGE: {len(NAMES)} movies, ALL frames; pairs of primaries within {HM_D} um "
      f"merged at the segment mass centroid when the heatmap saddle >= "
      f"{' / '.join(str(t_) for _a, t_ in HM_ARMS)} ===", flush=True)
for nm in NAMES:
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    T = img.shape[0]
    ref_old = np.load(OLD[nm])
    outs = {a_: {} for a_, _t in HM_ARMS}
    bad = nfr = 0
    tm = time.time()
    for t in range(T):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        h = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        key = f"c{t}"
        if key in ref_old.files:
            o = np.asarray(ref_old[key], np.float32)
            n_ = np.asarray(cr, np.float32)
            nfr += 1
            if o.shape != n_.shape or not np.array_equal(o, n_):
                bad += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append((nm, t, o.shape, n_.shape))
        cand = hm_pairs(h, ch, s)
        n_prim = int((s >= UNET_THRESH).sum())
        for a_, thr in HM_ARMS:
            c2, s2, nmg = hm_apply(h, vol, ch, s, cand, thr)
            outs[a_][f"c{t}"] = np.asarray(c2, np.float32)
            outs[a_][f"s{t}"] = np.asarray(s2, np.float32)
            STAT[a_]["primaries"] += n_prim
            STAT[a_]["pairs"] += len(cand)
            STAT[a_]["merged"] += nmg
        del hs, h, vol
        if t % 25 == 0:
            gc.collect()
    for a_, _t in HM_ARMS:
        np.savez_compressed(OUT / f"fdet_F4{a_}_{nm}.npz", **outs[a_])
    done += 1
    print(f"  {done}/{len(NAMES)} {nm}: T={T}, anchor {nfr - bad}/{nfr} frames exact, "
          + ", ".join(f"{a_} merged {STAT[a_]['merged']:,}/{STAT[a_]['primaries']:,} primaries "
                      f"({STAT[a_]['merged'] / max(1, STAT[a_]['primaries']):.3%}, cumulative)"
                      for a_, _t in HM_ARMS)
          + f"  {time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if done == 1:
        assert bad == 0, (f"ANCHOR FAILED on the first movie: {bad} of {nfr} frames differ from the "
                          f"shipped fdet_F4_ cache. {ANCHOR_FAILS[:3]}")
        print("  ANCHOR OK on movie 1: every frame reproduces fdet_F4_ exactly", flush=True)
    ref_old.close()
    del outs
    gc.collect()

print(f"\n=== HMERGE TOTALS over {done} movies ===", flush=True)
for a_, thr in HM_ARMS:
    d_ = STAT[a_]
    print(f"  {a_} (saddle >= {thr}): primaries {d_['primaries']:,}, pairs within {HM_D} um {d_['pairs']:,}, "
          f"merged {d_['merged']:,} ({d_['merged'] / max(1, d_['primaries']):.3%} of primaries, "
          f"{d_['merged'] / max(1, d_['pairs']):.3%} of pairs)", flush=True)
if ANCHOR_FAILS:
    print(f"\n  !! ANCHOR DRIFT on {len(ANCHOR_FAILS)}+ frames after movie 1. First few: {ANCHOR_FAILS[:4]}", flush=True)
else:
    print("\n  ANCHOR: every frame of every movie reproduced the shipped fdet_F4_ cache exactly.", flush=True)
print(f"\nHMERGE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("HMERGE OK" if not ANCHOR_FAILS and done == len(NAMES) else "HMERGE INCOMPLETE OR ANCHOR FAILED", flush=True)
