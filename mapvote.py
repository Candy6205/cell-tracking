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

SCRIPT_VERSION = "2026-09-19a-mapvote"
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

# ---- MAPVOTE ----------------------------------------------------------------------------------------
VOTE_FLOOR = 0.05        # peak-finding floor on each single map
AGREE_UM = 2.5           # maxima of different maps within this of each other are one candidate
RESCUE_UM = 7.0          # a candidate within this of a shipped primary (s >= 0.15) is not new
NEW_NMS_UM = 4.0         # the shipped NMS radius, among the new candidates
MIN_VOTES = 2
SCORE_OF_K = {2: 0.17, 3: 0.18, 4: 0.19}
BUDGET_S = 11.0 * 3600
PSCALE_ARR = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])


def _unpool(zyx):
    c = np.asarray(zyx, float).reshape(-1, 3).copy()
    c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
    c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
    return c


def single_maxima(m):
    """(pooled sub-voxel coords, values) of every local maximum of one map at VOTE_FLOOR, hpara-shifted."""
    pk = peak_local_max(m, min_distance=1, threshold_abs=VOTE_FLOOR, exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    v = m[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    return np.asarray(_hpara_shift(m, pk), float).reshape(-1, 3), v


def vote_candidates(hs, f4):
    """Maxima of distinct maps within AGREE_UM of each other, greedy from the highest value, nearest per map.
    Returns (pooled mean coords (n, 3), votes (n,), mean member value (n,), f4 value at the mean (n,)) for
    the clusters with >= MIN_VOTES maps."""
    P, V, L = [], [], []
    for i, m in enumerate(hs):
        p, v = single_maxima(m)
        P.append(p)
        V.append(v)
        L.append(np.full(len(v), i, int))
    P, V, L = np.concatenate(P), np.concatenate(V), np.concatenate(L)
    empty = (np.zeros((0, 3)), np.zeros(0, int), np.zeros(0), np.zeros(0))
    if len(P) == 0:
        return empty
    Pu = P * PSCALE_ARR
    tree = cKDTree(Pu)
    claimed = np.zeros(len(P), bool)
    cc, kk, vv = [], [], []
    for i in np.argsort(-V, kind="stable"):
        if claimed[i]:
            continue
        near = [j for j in tree.query_ball_point(Pu[i], AGREE_UM) if j != i and not claimed[j] and L[j] != L[i]]
        best = {}
        for j in near:
            d = float(np.linalg.norm(Pu[j] - Pu[i]))
            if L[j] not in best or d < best[L[j]][0]:
                best[L[j]] = (d, j)
        idx = [i] + [j for _d, j in best.values()]
        claimed[idx] = True
        if len(idx) >= MIN_VOTES:
            cc.append(P[idx].mean(0))
            kk.append(len(idx))
            vv.append(float(V[idx].mean()))
    if not cc:
        return empty
    C = np.asarray(cc, float)
    Z, Y, X = f4.shape
    ci = np.clip(np.round(C).astype(int), 0, [Z - 1, Y - 1, X - 1])
    ff = f4[ci[:, 0], ci[:, 1], ci[:, 2]].astype(float)
    return C, np.asarray(kk, int), np.asarray(vv, float), ff


def select_new(refined_vox, k, v, f, prim_um):
    """Keep the candidates with no shipped primary within RESCUE_UM, then the shipped NMS among them
    (votes first, then value). Returns the kept index array."""
    refined_vox = np.asarray(refined_vox, float).reshape(-1, 3)
    if len(refined_vox) == 0:
        return np.zeros(0, int)
    keep = np.ones(len(refined_vox), bool)
    if len(prim_um):
        d, _ = cKDTree(prim_um).query(refined_vox * SCALE)
        keep &= d > RESCUE_UM
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return idx
    order_score = k[idx] * 10.0 + v[idx]
    kept = _nms_keep(refined_vox[idx], order_score, NEW_NMS_UM)
    return idx[np.asarray(kept, bool)] if np.asarray(kept).dtype == bool else idx[np.asarray(kept, int)]


FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
OUT = pathlib.Path("/kaggle/working")
KS = (2, 3, 4)
TOT = defaultdict(float)
DGT = defaultdict(list)          # annotated new candidates: distance to the GT node, by k
F4V = defaultdict(list)          # f4 value at the new candidates, by k
t_all = time.time()
done = n_bad = 0
ANCHOR_FAILS = []
for k_, name in enumerate(SIX):
    if time.time() - t_all > BUDGET_S:
        print(f"BUDGET: stopping before {name}", flush=True)
        break
    tm = time.time()
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    out = {}
    stat = defaultdict(float)
    for t in range(T):
        c0 = np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3)
        s0 = np.asarray(z[f"s{t}"], np.float32).reshape(-1)
        prim = c0[s0 >= UNET_THRESH] * SCALE
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        f4 = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        n_ = np.asarray(ch, np.float32)
        if c0.shape != n_.shape or not np.array_equal(c0, n_):
            n_bad += 1
            if len(ANCHOR_FAILS) < 6:
                ANCHOR_FAILS.append((name, t, c0.shape, n_.shape))
        C, K, Vm, Ff = vote_candidates(hs, f4)
        del hs, f4
        if len(C):
            refined = np.array([_refine(vol, c) for c in _unpool(C)], float).reshape(-1, 3)
            sel = select_new(refined, K, Vm, Ff, prim)
        else:
            refined, sel = np.zeros((0, 3)), np.zeros(0, int)
        del vol
        cn = refined[sel].astype(np.float32)
        kn = K[sel].astype(np.int8)
        sn = np.asarray([SCORE_OF_K[int(min(kk, 4))] for kk in kn], np.float32)
        out[f"c{t}"] = np.concatenate([c0, cn]) if len(cn) else c0
        out[f"s{t}"] = np.concatenate([s0, sn]) if len(sn) else s0
        out[f"nnew{t}"] = np.int32(len(cn))
        out[f"k{t}"] = kn
        out[f"f{t}"] = Ff[sel].astype(np.float32)
        out[f"v{t}"] = Vm[sel].astype(np.float32)
        gt_t = g_pos[g_t == t]
        gtree = cKDTree(gt_t) if len(gt_t) else None
        stat["shipped"] += len(prim)
        stat["raw_clusters"] += len(C)
        if gtree is not None and len(prim):
            stat["shipped_ann"] += int((gtree.query(prim)[0] <= RESCUE_UM).sum())
        if gtree is not None:
            d_gp = cKDTree(prim).query(gt_t)[0] if len(prim) else np.full(len(gt_t), np.inf)
            miss = d_gp > RESCUE_UM
            stat["gt"] += len(gt_t)
            stat["gt_miss"] += int(miss.sum())
            if miss.any() and len(cn):
                ctree_all = cKDTree(cn * SCALE)
                for kmin in KS:
                    m_ = kn >= kmin
                    if m_.any():
                        d_ = cKDTree(cn[m_] * SCALE).query(gt_t[miss])[0]
                        stat[f"miss_hit{kmin}"] += int((d_ <= RESCUE_UM).sum())
                del ctree_all
        if len(cn):
            dg = gtree.query(cn * SCALE)[0] if gtree is not None else np.full(len(cn), np.inf)
            for kk in KS:
                m_ = kn == kk
                stat[f"new_{kk}"] += int(m_.sum())
                stat[f"newann_{kk}"] += int((m_ & (dg <= RESCUE_UM)).sum())
                if m_.any():
                    F4V[kk].extend(Ff[sel][m_].tolist())
                    DGT[kk].extend(dg[m_ & (dg <= RESCUE_UM)].tolist())
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"fdet_F4vt_{name}.npz", **out)
    done += 1
    for k2, v2 in stat.items():
        TOT[k2] += v2
    n_new = sum(stat[f"new_{kk}"] for kk in KS)
    n_new_ann = sum(stat[f"newann_{kk}"] for kk in KS)
    print(f"  [{done}/{len(SIX)}] {name}: T={T}, shipped {stat['shipped']:,.0f} (annotated {stat['shipped_ann'] / max(1, stat['shipped']):.3f}), "
          f"clusters {stat['raw_clusters']:,.0f}, new {n_new:,} (+{n_new / max(1, stat['shipped']):.1%}; annotated {n_new_ann / max(1, n_new):.3f}); by votes "
          + " ".join(f"[{kk}: {stat[f'new_{kk}']:.0f}/{stat[f'newann_{kk}']:.0f}]" for kk in KS)
          + f"; missed GT {stat['gt_miss']:.0f}, recovered k>=2 {stat['miss_hit2']:.0f} k>=3 {stat['miss_hit3']:.0f}; anchor misses {n_bad}"
          f"  {time.time() - tm:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    if done == 1:
        assert n_bad == 0, f"ANCHOR FAILED on the first movie: {ANCHOR_FAILS[:3]}"
        print("  ANCHOR OK on movie 1: every frame reproduces fdet_F4hp_ exactly", flush=True)
    del img, out, z
    gc.collect()


def _q(vals, f):
    v = np.asarray(vals, float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


print(f"\n=== MAPVOTE over {done} movies: shipped primaries {TOT['shipped']:,.0f}, annotated share {TOT['shipped_ann'] / max(1, TOT['shipped']):.4f}; "
      f"vote clusters before exclusion {TOT['raw_clusters']:,.0f}; anchor misses {n_bad} ===", flush=True)
print(f"  {'votes':<8}{'count':>9}{'+% of shipped':>14}{'annotated':>10}{'ann share':>10}{'x shipped share':>16}{'f4 med':>8}{'f4 p90':>8}{'d->GT med':>10}", flush=True)
base = TOT["shipped_ann"] / max(1, TOT["shipped"])
for kk in KS:
    n, a = TOT[f"new_{kk}"], TOT[f"newann_{kk}"]
    print(f"  {kk:<8}{n:>9,.0f}{n / max(1, TOT['shipped']):>14.2%}{a:>10,.0f}{a / max(1, n):>10.4f}{(a / max(1, n)) / max(base, 1e-9):>16.2f}"
          f"{_q(F4V[kk], .5):>8.3f}{_q(F4V[kk], .9):>8.3f}{_q(DGT[kk], .5):>10.2f}", flush=True)
for lab, kmin in (("k >= 2", 2), ("k >= 3", 3), ("k >= 4", 4)):
    n = sum(TOT[f"new_{kk}"] for kk in KS if kk >= kmin)
    a = sum(TOT[f"newann_{kk}"] for kk in KS if kk >= kmin)
    print(f"  admit {lab}: {n:,.0f} candidates (+{n / max(1, TOT['shipped']):.2%} of shipped), annotated {a:,.0f} ({a / max(1, n):.4f}, "
          f"{(a / max(1, n)) / max(base, 1e-9):.2f}x shipped); missed annotated cells recovered within 7 um: {TOT[f'miss_hit{kmin}']:,.0f} of {TOT['gt_miss']:,.0f} "
          f"({TOT[f'miss_hit{kmin}'] / max(1, TOT['gt_miss']):.1%}); annotated GT nodes {TOT['gt']:,.0f}", flush=True)
print("  reading: 'x shipped share' near 1 means the agreed maxima are cells as often as the shipped detections are; "
      "well below 1 means the vote admits mostly texture. The recovery column is the ceiling the bench can reach; the "
      "bench (heads on this set + lnbench's harness, lntrace's accounting) says how many the ranker links.", flush=True)
print(f"\nMAPVOTE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("MAPVOTE OK" if done == len(SIX) and n_bad == 0 else f"MAPVOTE INCOMPLETE {done}/{len(SIX)} anchor misses {n_bad}", flush=True)
