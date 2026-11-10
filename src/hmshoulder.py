"""hmshoulder -- is a missed annotated cell a second blob under its host's
heatmap response, findable without the annotation? See
tools/bench/make_hmshoulder.py for the design. Built from hplace.py: the same
branch_maps4 / detect_both pass, anchored against biohub-hparacache.

SCRIPT_VERSION 2026-09-18a-hmshoulder
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
from scipy import ndimage as _ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-09-18a-hmshoulder"
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

UNPOOL_OFF = (POOL - 1) / 2.0          # detect_both: c_pooled * POOL + 1.5
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])   # pooled grid, um per voxel
assert abs(PSCALE[0] - PSCALE[1]) < 1e-9, PSCALE                  # isotropic 1.625 um
MAPS = ["b0", "b1", "b2", "dc", "f4"]
R_MATCH = 7.0                          # the scorer's match radius, um
R_CROWD = 9.5


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def from_pooled(p):
    p = np.asarray(p, float)
    return np.array([p[0], p[1] * POOL + UNPOOL_OFF, p[2] * POOL + UNPOOL_OFF])


def um_p(p, q):
    return float(np.linalg.norm((np.asarray(p, float) - np.asarray(q, float)) * PSCALE))


def um_full(p, q):
    return float(np.linalg.norm((np.asarray(p, float) - np.asarray(q, float)) * SCALE))


def place_one(m, vol, pA, thr=DETECT_THRESH, R=R_MATCH):
    """Nearest local maximum of map m within R um of pooled point pA.
    Returns dict(d, dz, dxy, v, n7, dr) with NaN when there is none."""
    Z, Y, X = m.shape
    r = int(np.ceil(R / PSCALE[0])) + 1
    z0, y0, x0 = (int(round(v)) for v in pA)
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = m[zs, ys, xs]
    nan = float("nan")
    if sub.size == 0:
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    pk = peak_local_max(sub, min_distance=1, threshold_abs=thr, exclude_border=False)
    if len(pk) == 0:
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    pk = pk + np.array([zs.start, ys.start, xs.start])
    d = np.linalg.norm((pk - pA[None, :]) * PSCALE[None, :], axis=1)
    inR = d <= R
    if not inR.any():
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    n7 = int(inR.sum())
    k = int(np.argmin(np.where(inR, d, np.inf)))
    p = _hpara_shift(m, pk[k:k + 1])[0]
    dv = (p - pA) * PSCALE
    out = dict(d=float(np.linalg.norm(dv)), dz=float(abs(dv[0])), dxy=float(np.hypot(dv[1], dv[2])),
               v=float(m[tuple(pk[k])]), n7=n7, pz=float(p[0]), py=float(p[1]), px=float(p[2]))
    pr = _refine(vol, from_pooled(p))
    out["dr"] = um_full(pr, from_pooled(pA))
    return out



# ---- HMSHOULDER: a second blob under the host's response, blind to the annotation ------------------
MISS_UM = 7.0            # a cell with no shipped primary within this is missed (the scorer's radius)
HOST_UM = 12.0           # the host = the nearest shipped primary, within this
CTRL_MATCH_UM = 2.5
LONE_PRIM_UM = 8.0
LONE_GT_UM = 10.0
N_CTRL_PER_FRAME = 2     # clean singles and dense singles, from the host frames only
PATCH_UM = 13.0          # fit window: the pooled f4 map within this of the host
MASK_UM = 6.0            # voxels within this of ANOTHER shipped primary are left out of the fit
EXCL_UM = 3.0            # the second blob is looked for at least this far from the host
FIRE_UM = 4.0            # a second blob within this of any shipped primary is not a fire
HIT_UM = 3.0             # a second blob within this of the missed cell is a hit
SIG0 = 1.3               # starting sigma, pooled voxels (2.1 um)
LM_FLOOR = 0.05          # peak-finding floor for 'does map X have a local maximum near the cell'
LM_UM = 3.0
RNG = np.random.default_rng(0)


def build_sites(gt, prim):
    """gt: t -> annotated positions (um); prim: per t (coords_vox, scores) of the shipped primaries.
    host: an annotated cell with no primary within MISS_UM and one within HOST_UM (its host);
    ctrl: a primary holding an annotated cell (2.5 um), no other primary within 8, no other GT within 10;
    dense: a primary holding an annotated cell with another primary within 8 um.
    Controls are drawn from the host frames only, N_CTRL_PER_FRAME of each, seeded."""
    hosts, ctrl, dense = defaultdict(list), defaultdict(list), defaultdict(list)
    for t in sorted(gt):
        if t >= len(prim):
            continue
        C = np.asarray(prim[t][0], float).reshape(-1, 3)
        S = np.asarray(prim[t][1], float).reshape(-1)
        if len(C) == 0:
            continue
        g_pos = np.asarray(gt[t], float).reshape(-1, 3)
        if len(g_pos) == 0:
            continue
        Cu = C * SCALE
        ptree = cKDTree(Cu)
        gtree = cKDTree(g_pos)
        d_near, i_near = ptree.query(g_pos)
        for k in np.where(d_near > MISS_UM)[0]:
            if d_near[k] <= HOST_UM:
                i = int(i_near[k])
                n8 = len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1
                hosts[t].append(dict(kind="host", t=t, d=C[i].copy(), s=float(S[i]), g=g_pos[k] / SCALE,
                                     d_gnear=float(d_near[k]), n8=n8))
        for k in np.where(d_near <= CTRL_MATCH_UM)[0]:
            i = int(i_near[k])
            n8 = len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1
            n_gt = len(gtree.query_ball_point(g_pos[k], LONE_GT_UM)) - 1
            row = dict(kind="", t=t, d=C[i].copy(), s=float(S[i]), g=None, d_gnear=float(d_near[k]), n8=n8)
            if n8 == 0 and n_gt == 0:
                row["kind"] = "ctrl"
                ctrl[t].append(row)
            elif n8 >= 1:
                row["kind"] = "dense"
                dense[t].append(row)
    out = []
    for t in sorted(hosts):
        out.extend(hosts[t])
        for pool in (ctrl[t], dense[t]):
            if len(pool) > N_CTRL_PER_FRAME:
                pick = RNG.choice(len(pool), N_CTRL_PER_FRAME, replace=False)
                pool = [pool[i] for i in sorted(pick)]
            out.extend(pool)
    return out


def um_grid(P):
    """Full-resolution um of pooled grid points (N, 3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    return np.stack([P[:, 0] * SCALE[0], (P[:, 1] * POOL + UNPOOL_OFF) * SCALE[1],
                     (P[:, 2] * POOL + UNPOOL_OFF) * SCALE[2]], 1)


def um_pooled(p):
    return um_grid(np.asarray(p, float)[None, :])[0]


def _gauss(P, c, sg):
    return np.exp(-0.5 * ((P - c) ** 2).sum(1) / (sg * sg))


def fit_site(m, pd_, others_um):
    """m: the pooled f4 map; pd_: the host in pooled coords; others_um: every other shipped primary (um).
    Blind to the annotation. Returns the one-blob fit, the residual maximum, the radial-median residual
    maximum and the two-blob fit; positions in pooled coords. None when the window is too small."""
    Z, Y, X = m.shape
    r = int(np.ceil(PATCH_UM / PSCALE[0]))
    z0, y0, x0 = (int(round(float(v))) for v in pd_)
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return None
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = np.asarray(m[zs, ys, xs], np.float64)
    gz, gy, gx = np.mgrid[zs, ys, xs]
    P = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1).astype(np.float64)
    v = sub.ravel()
    Pu = um_grid(P)
    hu = um_pooled(pd_)
    keep = np.linalg.norm(Pu - hu, axis=1) <= PATCH_UM
    others_um = np.asarray(others_um, float).reshape(-1, 3)
    if len(others_um):
        keep &= cKDTree(others_um).query(Pu)[0] >= MASK_UM
    P, v, Pu = P[keep], v[keep], Pu[keep]
    if len(v) < 200:
        return None
    a0 = max(float(m[z0, y0, x0]), 0.05)
    pd_ = np.asarray(pd_, float)

    def f1(p):
        return p[0] * _gauss(P, p[1:4], p[4]) + p[5]

    lo1 = [0.0, pd_[0] - 2.0, pd_[1] - 2.0, pd_[2] - 2.0, 0.6, 0.0]
    hi1 = [2.0, pd_[0] + 2.0, pd_[1] + 2.0, pd_[2] + 2.0, 4.0, 0.6]
    x1 = np.clip(np.array([a0, pd_[0], pd_[1], pd_[2], SIG0, 0.0]), lo1, hi1)
    r1 = least_squares(lambda p: f1(p) - v, x1, bounds=(lo1, hi1), method="trf", max_nfev=300)
    p1 = r1.x
    res = v - f1(p1)
    sse1 = float((res ** 2).sum())
    c1 = p1[1:4]
    rr = np.linalg.norm(um_grid(P) - um_pooled(c1), axis=1)
    far = rr >= EXCL_UM
    if not far.any():
        return None
    k = int(np.argmax(np.where(far, res, -np.inf)))
    out = dict(n_fit=int(len(v)), A1=float(p1[0]), s1=float(p1[4]), b1=float(p1[5]), sse1=sse1,
               c1=c1.copy(), r_res=float(res[k]), c_res=P[k].copy())
    # the radial-median residual: the map minus its shell median around the fitted centre
    sh = np.round(rr / PSCALE[0]).astype(int)
    prof = np.zeros(int(sh.max()) + 1)
    for j in range(len(prof)):
        mj = sh == j
        prof[j] = float(np.median(v[mj])) if mj.any() else 0.0
    rad = v - np.interp(rr, np.arange(len(prof)) * PSCALE[0], prof)   # the shell profile, interpolated in radius
    k2 = int(np.argmax(np.where(far, rad, -np.inf)))
    out.update(r_rad=float(rad[k2]), c_rad=P[k2].copy())

    # two blobs, started from each residual maximum; the lower SSE wins
    def f2(p):
        return p[0] * _gauss(P, p[1:4], p[4]) + p[5] + p[6] * _gauss(P, p[7:10], p[10])

    lo2 = lo1 + [0.0, float(zs.start), float(ys.start), float(xs.start), 0.6]
    hi2 = hi1 + [2.0, float(zs.stop - 1), float(ys.stop - 1), float(xs.stop - 1), 4.0]
    starts = [(res[k], P[k])]
    if np.linalg.norm((P[k] - P[k2]) * PSCALE) > 2.0:
        starts.append((rad[k2], P[k2]))
    best = None
    for amp, c in starts:
        x2 = np.clip(np.concatenate([p1, [max(float(amp), 0.01)], c, [SIG0]]), lo2, hi2)
        r2 = least_squares(lambda p: f2(p) - v, x2, bounds=(lo2, hi2), method="trf", max_nfev=400)
        sse2 = float(((f2(r2.x) - v) ** 2).sum())
        if best is None or sse2 < best[0]:
            best = (sse2, r2.x)
    sse2, p2 = best
    A1b, c1b, s1b, bb, A2, c2, s2 = p2[0], p2[1:4], p2[4], p2[5], p2[6], p2[7:10], p2[10]
    if np.linalg.norm((c2 - pd_) * PSCALE) < np.linalg.norm((c1b - pd_) * PSCALE):   # the fit swapped the blobs
        A1b, c1b, s1b, A2, c2, s2 = A2, c2, s2, A1b, c1b, s1b
    out.update(A1b=float(A1b), s1b=float(s1b), bb=float(bb), A2=float(A2), s2=float(s2), c2=np.asarray(c2, float).copy(),
               sse2=sse2, dsse=float((sse1 - sse2) / max(sse1, 1e-12)))
    return out


def map_at(m, p):
    z, y, x = (int(round(float(v))) for v in p)
    Z, Y, X = m.shape
    if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
        return float(m[z, y, x])
    return float("nan")


def max_near(m, p, R):
    """Maximum of the map within R um of pooled point p."""
    Z, Y, X = m.shape
    r = int(np.ceil(R / PSCALE[0])) + 1
    z0, y0, x0 = (int(round(float(v))) for v in p)
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = m[zs, ys, xs]
    if sub.size == 0:
        return float("nan")
    gz, gy, gx = np.mgrid[zs, ys, xs]
    d = np.sqrt(((gz - p[0]) * PSCALE[0]) ** 2 + ((gy - p[1]) * PSCALE[1]) ** 2 + ((gx - p[2]) * PSCALE[2]) ** 2)
    inR = d <= R
    return float(sub[inR].max()) if inR.any() else float("nan")


def along(m, pa, pb, step_um=0.5):
    """The map sampled (trilinear) from pooled pa to pooled pb."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    n = max(3, int(np.ceil(np.linalg.norm((pb - pa) * PSCALE) / step_um)) + 1)
    pts = pa[None, :] + (pb - pa)[None, :] * np.linspace(0.0, 1.0, n)[:, None]
    return _ndi.map_coordinates(m, pts.T, order=1, mode="nearest")


def dip_g(m, pg, pd_):
    """min of the map along cell -> host, over the value at the cell (1 = monotone rise, no dip)."""
    vals = along(m, pg, pd_)
    return float(vals.min() / max(float(vals[0]), 1e-6))


def far2_g(m, pg, pd_, um=2.0):
    """The map 2 um beyond the cell, away from the host, over the value at the cell."""
    pg, pd_ = np.asarray(pg, float), np.asarray(pd_, float)
    u = (pg - pd_) * PSCALE
    nrm = np.linalg.norm(u)
    if nrm < 1e-6:
        return float("nan")
    q = pg + (u / nrm) * um / PSCALE
    vals = along(m, pg, q)
    return float(vals[-1] / max(float(vals[0]), 1e-6))


def ball_mean(vol_p, p, r=1):
    z, y, x = (int(round(float(v))) for v in p)
    Z, Y, X = vol_p.shape
    sub = vol_p[max(0, z - r):min(Z, z + r + 1), max(0, y - r):min(Y, y + r + 1), max(0, x - r):min(X, x + r + 1)]
    return float(sub.mean()) if sub.size else float("nan")


# ---- the caches, the sites, the frame loop ------------------------------------------------------------
FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
print(f"F4hp caches: {len(SIX)} 6bba movies", flush=True)

MAP_NAMES = ["f4", "b0", "b1", "b2", "dc"]
COLS = (["s_host", "n8", "d_gnear", "d_dg", "f4_g", "f4_g2", "f4_d", "dip_g", "far2_g"]
        + [f"lm3_{mp}" for mp in MAP_NAMES] + ["I_g", "I_d", "p40", "n_fit", "A1", "s1", "b1", "sse1",
        "r_res", "d_res_g", "d_res_d", "d_res_prim", "r_rad", "d_rad_g", "d_rad_d", "d_rad_prim",
        "A1b", "s1b", "A2", "s2", "dsse", "d_c2_g", "d_c2_d", "d_c2_prim"])
OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "hmshoulder_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t,dz,dy,dx,gz,gy,gx," + ",".join(COLS) + "\n")
RES = defaultdict(list)
COUNT = defaultdict(int)
t0 = time.time()
n_frames = n_bad = n_ann = n_missed = n_prim_all = 0
ANCHOR_FAILS = []
nan = float("nan")
for k_, nm in enumerate(SIX):
    tm = time.time()
    zf = np.load(FDET[nm])
    T = max(int(k[1:]) for k in zf.files if k.startswith("c")) + 1
    prim = []
    for t in range(T):
        c = np.asarray(zf[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(zf[f"s{t}"], float).reshape(-1)
        prim.append((c[s >= UNET_THRESH], s[s >= UNET_THRESH]))
        n_prim_all += int((s >= UNET_THRESH).sum())
    gtg = read_geff(geff_path(ROOT, "train", nm))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    gt = {int(tt): g_pos[g_t == tt] for tt in np.unique(g_t)}
    n_ann += len(g_t)
    sites = build_sites(gt, prim)
    n_missed += sum(1 for r_ in sites if r_["kind"] == "host")
    by_t = defaultdict(list)
    for r_ in sites:
        by_t[r_["t"]].append(r_)
    if not by_t:
        print(f"  [{k_ + 1}/{len(SIX)}] {nm}: no host site", flush=True)
        continue
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    for t in sorted(by_t):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        maps = dict(zip(["b0", "b1", "b2", "dc"], hs))
        maps["f4"] = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        o_ = np.asarray(zf[f"c{t}"], np.float32)
        n_ = np.asarray(ch, np.float32)
        if o_.shape != n_.shape or not np.array_equal(o_, n_):
            n_bad += 1
            if len(ANCHOR_FAILS) < 6:
                ANCHOR_FAILS.append((nm, t, o_.shape, n_.shape))
        praw = pool_xy(vol)
        p40 = float(np.percentile(praw[::2, ::2, ::2], 40))
        C_t = np.asarray(prim[t][0], float).reshape(-1, 3)
        Cu_t = C_t * SCALE
        ptree = cKDTree(Cu_t) if len(Cu_t) else None
        f4 = maps["f4"]
        for row in by_t[t]:
            kind = row["kind"]
            d = np.asarray(row["d"], float)
            du = d * SCALE
            pd_ = to_pooled(d)
            others = Cu_t[np.linalg.norm(Cu_t - du, axis=1) > 0.5] if len(Cu_t) else np.zeros((0, 3))
            o = {c_: nan for c_ in COLS}
            o["s_host"], o["n8"], o["d_gnear"] = row["s"], row["n8"], row["d_gnear"]
            g = row["g"]
            if g is not None:
                gu = np.asarray(g, float) * SCALE
                pg = to_pooled(g)
                o["d_dg"] = float(np.linalg.norm(gu - du))
                o["f4_g"] = map_at(f4, pg)
                o["f4_g2"] = max_near(f4, pg, 2.0)
                o["f4_d"] = map_at(f4, pd_)
                o["dip_g"] = dip_g(f4, pg, pd_)
                o["far2_g"] = far2_g(f4, pg, pd_)
                for mp in MAP_NAMES:
                    o[f"lm3_{mp}"] = place_one(maps[mp], vol, pg, thr=LM_FLOOR, R=LM_UM)["d"]
                o["I_g"] = ball_mean(praw, pg)
            o["I_d"] = ball_mean(praw, pd_)
            o["p40"] = p40
            fit = fit_site(f4, pd_, others)
            if fit is not None:
                for kk in ("n_fit", "A1", "s1", "b1", "sse1", "r_res", "r_rad", "A1b", "s1b", "A2", "s2", "dsse"):
                    o[kk] = fit[kk]
                for tag, ck in (("res", "c_res"), ("rad", "c_rad"), ("c2", "c2")):
                    cu = um_pooled(fit[ck])
                    o[f"d_{tag}_d"] = float(np.linalg.norm(cu - du))
                    o[f"d_{tag}_prim"] = float(ptree.query(cu)[0]) if ptree is not None else nan
                    if g is not None:
                        o[f"d_{tag}_g"] = float(np.linalg.norm(cu - gu))
            RES[kind].append(o)
            COUNT[kind] += 1
            gs = ",".join(f"{v:.3f}" for v in g) if g is not None else "nan,nan,nan"
            fh.write(f"{kind},{nm},{t},{d[0]:.3f},{d[1]:.3f},{d[2]:.3f},{gs},"
                     + ",".join(f"{o[c_]:.5g}" if o[c_] == o[c_] else "nan" for c_ in COLS) + "\n")
        n_frames += 1
        del hs, maps, vol, praw, f4
    fh.flush()
    zf.close()
    del img
    gc.collect()
    print(f"  [{k_ + 1}/{len(SIX)}] {nm}: {len(by_t)} frames, sites host {sum(1 for r_ in sites if r_['kind'] == 'host')} "
          f"ctrl {sum(1 for r_ in sites if r_['kind'] == 'ctrl')} dense {sum(1 for r_ in sites if r_['kind'] == 'dense')}, "
          f"anchor misses so far {n_bad}, {time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if n_frames and k_ == 0:
        assert n_bad == 0, f"ANCHOR FAILED on the first movie: {ANCHOR_FAILS[:3]}"
        print("  ANCHOR OK on movie 1: every computed frame reproduces fdet_F4hp_ exactly", flush=True)
fh.close()


# ---- summary -----------------------------------------------------------------------------------------
def q(vals, f):
    v = np.array([x for x in vals if x == x], float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


def share(vals, pred):
    v = [x for x in vals if x == x]
    return float(np.mean([pred(x) for x in v])) if v else float("nan")


def col(kind, c_):
    return [o[c_] for o in RES[kind]]


from sklearn.metrics import roc_auc_score  # noqa: E402

H, Cc, Dd = RES["host"], RES["ctrl"], RES["dense"]
print(f"\n=== HMSHOULDER: {n_frames} frames, sites host {len(H)} / ctrl {len(Cc)} / dense {len(Dd)}; anchor misses {n_bad}; "
      f"annotated GT nodes {n_ann:,}, missed at {MISS_UM} um with a host within {HOST_UM} um: {n_missed:,} "
      f"({n_missed / max(1, n_ann):.2%}); shipped primaries in these movies {n_prim_all:,} ===", flush=True)
print(f"\n  --- the missed cell g against its host d (n = {len(H)}) ---", flush=True)
print(f"  d(d, g) median {q(col('host', 'd_dg'), .5):.2f} um (p25 {q(col('host', 'd_dg'), .25):.2f}, p75 {q(col('host', 'd_dg'), .75):.2f}); "
      f"host score median {q(col('host', 's_host'), .5):.2f}; other primaries within 8 um of the host: median {q(col('host', 'n8'), .5):.0f}, "
      f">= 1 share {share(col('host', 'n8'), lambda x: x >= 1):.2f}", flush=True)
print(f"  f4 at g median {q(col('host', 'f4_g'), .5):.3f} (p25 {q(col('host', 'f4_g'), .25):.3f}, p75 {q(col('host', 'f4_g'), .75):.3f}), "
      f"max within 2 um {q(col('host', 'f4_g2'), .5):.3f}, >= 0.05 share {share(col('host', 'f4_g2'), lambda x: x >= 0.05):.2f}; "
      f"f4 at d median {q(col('host', 'f4_d'), .5):.3f}", flush=True)
print(f"  dip along g -> d (min / value at g) median {q(col('host', 'dip_g'), .5):.3f}, >= 0.97 (no dip: a monotone rise to the host) "
      f"share {share(col('host', 'dip_g'), lambda x: x >= 0.97):.2f}, < 0.8 share {share(col('host', 'dip_g'), lambda x: x < 0.8):.2f}; "
      f"value 2 um beyond g / at g median {q(col('host', 'far2_g'), .5):.3f}, < 0.9 share {share(col('host', 'far2_g'), lambda x: x < 0.9):.2f}", flush=True)
lm_any = [min([o[f'lm3_{mp}'] for mp in MAP_NAMES[1:] if o[f'lm3_{mp}'] == o[f'lm3_{mp}']] or [nan]) for o in H]
print("  a local maximum (floor 0.05) within 3 um of g: " + ", ".join(
    f"{mp} {share([0.0 if o[f'lm3_{mp}'] != o[f'lm3_{mp}'] else 1.0 for o in H], lambda x: x == 1.0):.2f}" for mp in MAP_NAMES)
    + f"; any single map {share([0.0 if x != x else 1.0 for x in lm_any], lambda x: x == 1.0):.2f}", flush=True)
bg = [1.0 if o["I_g"] <= o["p40"] else 0.0 for o in H if o["I_g"] == o["I_g"]]
print(f"  pooled raw intensity at g <= the frame's 40th percentile (trained as background) share {np.mean(bg) if bg else nan:.2f}; "
      f"I(g) / I(d) median {q([o['I_g'] / o['I_d'] for o in H if o['I_d'] == o['I_d'] and o['I_d'] > 0], .5):.3f}; "
      f"controls' I(d) <= p40 share ctrl {share([1.0 if o['I_d'] <= o['p40'] else 0.0 for o in Cc], lambda x: x == 1.0):.2f}, "
      f"dense {share([1.0 if o['I_d'] <= o['p40'] else 0.0 for o in Dd], lambda x: x == 1.0):.2f}", flush=True)
print(f"\n  --- the fits (medians host / ctrl / dense) ---", flush=True)
for c_, lab in (("n_fit", "voxels fitted"), ("A1", "one-blob amplitude"), ("s1", "one-blob sigma (pooled vox)"), ("b1", "one-blob floor"),
                ("A2", "second blob amplitude"), ("s2", "second blob sigma"), ("dsse", "SSE drop, two blobs vs one"),
                ("r_res", "residual max (>= 3 um from host)"), ("r_rad", "radial-median residual max")):
    print(f"  {lab:<36}{q(col('host', c_), .5):>9.3f}{q(col('ctrl', c_), .5):>9.3f}{q(col('dense', c_), .5):>9.3f}", flush=True)
print(f"\n  --- where the second point lands, at the hosts (d -> g in um; fire = >= {FIRE_UM} um from every shipped primary; "
      f"hit = fire and within {HIT_UM} um of g) ---", flush=True)
print(f"  {'method':<24}{'d->g med':>10}{'p75':>8}{'<=3um':>8}{'fire':>8}{'hit':>8}{'d->host med':>13}{'fire ctrl':>11}{'fire dense':>12}", flush=True)
METHODS = (("two-blob centre c2", "c2", "A2"), ("one-blob residual max", "res", "r_res"), ("radial residual max", "rad", "r_rad"))
for lab, tag, sc in METHODS:
    dg = col("host", f"d_{tag}_g")
    fire_h = [1.0 if o[f"d_{tag}_prim"] >= FIRE_UM else 0.0 for o in H if o[f"d_{tag}_prim"] == o[f"d_{tag}_prim"]]
    hit_h = [1.0 if (o[f"d_{tag}_prim"] >= FIRE_UM and o[f"d_{tag}_g"] <= HIT_UM) else 0.0 for o in H if o[f"d_{tag}_g"] == o[f"d_{tag}_g"]]
    print(f"  {lab:<24}{q(dg, .5):>10.2f}{q(dg, .75):>8.2f}{share(dg, lambda x: x <= HIT_UM):>8.2f}"
          f"{np.mean(fire_h) if fire_h else nan:>8.2f}{np.mean(hit_h) if hit_h else nan:>8.2f}{q(col('host', f'd_{tag}_d'), .5):>13.2f}"
          f"{share([o[f'd_{tag}_prim'] for o in Cc], lambda x: x >= FIRE_UM):>11.2f}{share([o[f'd_{tag}_prim'] for o in Dd], lambda x: x >= FIRE_UM):>12.2f}", flush=True)
print(f"\n  --- separation and price (a site's score is 0 unless its second point fires) ---", flush=True)
N_PRIM = 2_091_008
frac_host = n_missed / max(1, n_ann)
print(f"  base rate: hosts ~ {frac_host:.2%} of primaries (~{frac_host * N_PRIM:,.0f} of {N_PRIM:,}); a wrong fire puts a second point "
      f"next to a real cell (~1 edge at risk), a hit recovers ~2 edges (break-even ~2 wrong per hit)", flush=True)


def scored(rows, tag, sc):
    out = []
    for o in rows:
        v = o[sc]
        f = o[f"d_{tag}_prim"]
        out.append(v if (v == v and f == f and f >= FIRE_UM) else 0.0)
    return np.asarray(out, float)


for lab, tag, sc in METHODS + (("two-blob SSE drop", "c2", "dsse"),):
    sh_, sc_, sd_ = scored(H, tag, sc), scored(Cc, tag, sc), scored(Dd, tag, sc)
    hit = np.asarray([1.0 if (o[f"d_{tag}_g"] == o[f"d_{tag}_g"] and o[f"d_{tag}_g"] <= HIT_UM) else 0.0 for o in H], float)
    if len(sh_) < 30 or len(sc_) < 30 or len(sd_) < 30:
        continue
    auc_c = roc_auc_score(np.r_[np.ones(len(sh_)), np.zeros(len(sc_))], np.r_[sh_, sc_])
    auc_d = roc_auc_score(np.r_[np.ones(len(sh_)), np.zeros(len(sd_))], np.r_[sh_, sd_])
    print(f"  {lab} ({sc}): AUC host vs clean {auc_c:.3f}, host vs dense {auc_d:.3f}; host fires {np.mean(sh_ > 0):.2f}, "
          f"hits {np.mean((sh_ > 0) & (hit > 0)):.2f}", flush=True)
    hs_ = sh_[(sh_ > 0) & (hit > 0)]
    for rec in (0.3, 0.5, 0.7):
        need = rec * len(sh_)
        if len(hs_) < need:
            print(f"    {rec:.0%} hit recall unreachable: at most {len(hs_) / len(sh_):.0%} of the hosts get a hit at any threshold", flush=True)
            continue
        thr = float(np.sort(hs_)[::-1][int(np.ceil(need)) - 1])
        fc, fd = float((sc_ >= thr).mean()), float((sd_ >= thr).mean())
        f_all = 0.5 * (fc + fd)
        wrong = f_all * (1 - frac_host) / max(rec * frac_host, 1e-9)
        print(f"    at {rec:.0%} hit recall (thr {thr:.3f}): fires on {fc:.3f} of clean singles, {fd:.3f} of dense singles "
              f"-> ~{wrong:.1f} wrong fires per hit (break-even ~2)", flush=True)
print("\n  reading: if the two-blob centre lands within 3 um of g for most hosts while ordinary cells rarely fire at that "
      "amplitude, the shoulder is a placement source and a heatmap-side splitter has a price; if the fires at "
      "ordinary cells cost more than ~2 per hit, the shoulder is not separable from the map's own texture, as the raw "
      "image was not (blobsplit). Control fires are an upper bound: an unannotated cell next to a control is a real "
      "second blob the annotation cannot confirm.", flush=True)
print(f"\nHMSHOULDER DONE in {(time.time() - t0) / 60:.1f} min; hmshoulder_sites.csv written", flush=True)
print("HMSHOULDER OK" if n_bad == 0 else f"HMSHOULDER ANCHOR DRIFT on {n_bad} frames", flush=True)
