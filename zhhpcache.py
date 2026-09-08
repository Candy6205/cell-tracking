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

SCRIPT_VERSION = "2026-09-08a-zhhpcache"
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

def _nms_keep(coords, scores, radius_um, scale=SCALE):
    """_physical_nms's own loop, returning the KEPT INDICES instead of the
    rows. Both placements must survive or die together: if each arm ran its
    own NMS on its own coordinates the two would keep different detections
    and the comparison would confound placement with detection."""
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


def _para1(a, b, c):
    """Vertex of the parabola through (-1,a) (0,b) (1,c). Copied verbatim
    from hparacache.py, itself verbatim from refinecmp.py. The
    `d >= -1e-12` guard is what stops a flat window or an upward cusp from
    dragging the detection into a MINIMUM -- do not simplify it away."""
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
            if c0 <= 0 or c0 >= hi - 1:
                continue
            im = [z0, y0, x0]
            ip = [z0, y0, x0]
            im[ax] -= 1
            ip[ax] += 1
            zyx[i][ax] += _para1(float(h[tuple(im)]), b, float(h[tuple(ip)]))
    return zyx


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


def detect_both(vol):
    """(refine coords, hpara coords, scores) from ONE peak set and ONE NMS.

    The shipped `detect` runs NMS on the REFINED coordinates, so the survivor
    set depends on placement. Deciding it once, on refine, is what keeps the
    two arms scoring the same cells -- the same rule hparacache uses.
    """
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
        z = np.zeros((0, 3))
        return z, z.copy(), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    hp = _hpara_shift(h, pk)
    hp[:, 1] = hp[:, 1] * POOL + (POOL - 1) / 2
    hp[:, 2] = hp[:, 2] * POOL + (POOL - 1) / 2
    keep = _nms_keep(ref, sc, NMS_UM)
    return ref[keep], hp[keep], sc[keep]


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
OLD = {}
for _p in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz"):
    if _p.name.startswith("zhdet_hp_"):
        continue
    OLD[_p.name[len("zhdet_"):-len(".npz")]] = np.load(_p)
print(f"anchor caches found: {len(OLD)} zhdet_*.npz", flush=True)
assert OLD, ("attach faustine649/biohub-zh-detect -- without it the refine "
             "arm cannot be checked and the hpara column is unverifiable")
ANCHOR_FAILS, NFR, NBAD, SHIFT = [], [0], [0], []

files = sorted(CROPS.glob("*.npz"))
print(f"crops: {len(files)} under {CROPS}", flush=True)
assert files, "crops directory is empty"

OUT = pathlib.Path("/kaggle/working")
BUDGET_S = 10.0 * 3600
t_start = time.time()
done = 0
t_per = []

for i, f in enumerate(files):
    if time.time() - t_start > BUDGET_S:
        print(f"\nWARN wall-clock budget reached: {len(files)-i} crops NOT "
              f"detected ({f.stem} onwards). PARTIAL output -- re-run for "
              f"the rest.", flush=True)
        break
    t0 = time.time()
    z = np.load(f)
    img = z["image"]
    if img.ndim != 4:
        print(f"  WARN {f.stem}: image ndim {img.ndim}, expected 4 "
              f"-- skipped", flush=True)
        continue
    T = img.shape[0]
    out = {}
    n_seed = 0
    for t in range(T):
        c, chp, s = detect_both(np.asarray(img[t]))
        out[f"c{t}"] = np.asarray(chp, np.float32)      # hpara is the OUTPUT
        out[f"s{t}"] = np.asarray(s, np.float32)
        _key = f"c{t}"
        _old = OLD.get(f.stem)
        if _old is not None and _key in _old.files:
            _o = np.asarray(_old[_key], np.float32)
            _n = np.asarray(c, np.float32)              # refine is the ANCHOR
            NFR[0] += 1
            if _o.shape != _n.shape or not np.array_equal(_o, _n):
                NBAD[0] += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append(
                        (f.stem, t, _o.shape, _n.shape,
                         float(np.abs(_o - _n).max())
                         if _o.shape == _n.shape else float("nan")))
        if len(c):
            SHIFT.append(np.linalg.norm((chp - c) * SCALE[None, :], axis=1))
        n_seed += int((out[f"s{t}"] >= UNET_THRESH).sum())
        if t % 25 == 0:
            gc.collect()
    # carry the labels through so one file trains without the 7.5 GB crop
    for k in ("nodes", "track_id", "parent_track_id", "origin_um",
              "voxel_um", "t0"):
        if k in z.files:
            out[k] = z[k]
        else:
            print(f"  WARN {f.stem}: crop has no '{k}'", flush=True)
    np.savez_compressed(OUT / f"zhdet_hp_{f.stem}.npz", **out)
    dt = time.time() - t0
    t_per.append(dt)
    done += 1
    eta = (len(files) - i - 1) * (sum(t_per) / len(t_per))
    print(f"  [{i+1}/{len(files)}] {f.stem}: T={T} gt_nodes="
          f"{len(z['nodes']) if 'nodes' in z.files else -1} "
          f"seeds={n_seed} {dt:.0f}s  eta {eta/60:.0f}m", flush=True)
    del z, img
    gc.collect()

el = time.time() - t_start
print(f"\ndetected {done}/{len(files)} crops in {el/3600:.2f} h", flush=True)
mb = sum(p.stat().st_size for p in OUT.glob("zhdet_*.npz")) / 2 ** 20
print(f"output {mb:.0f} MB", flush=True)
print(f"\nANCHOR: {NFR[0] - NBAD[0]} of {NFR[0]} frames reproduce the "
      f"shipped zhdet_ cache exactly on the REFINE placement", flush=True)
if ANCHOR_FAILS:
    print(f"  [degraded] first mismatches: {ANCHOR_FAILS}", flush=True)
if SHIFT:
    _s = np.concatenate(SHIFT)
    print(f"|hpara - refine|: p50 {np.median(_s):.3f} p90 "
          f"{np.percentile(_s, 90):.3f} max {_s.max():.3f} um "
          f"({len(_s):,} detections)", flush=True)
print("ZHHPCACHE " + ("OK" if NBAD[0] == 0 and NFR[0] > 0 else "ANCHOR FAILED"),
      flush=True)
print("ZH DETECT DONE", flush=True)
