"""dccache3.py -- WHICH training stage keeps DeepCenter's value? ep5 vs ep7 vs ep12.

SCRIPT_VERSION 2026-08-29a-dccache3

THE QUESTION. dccache2 measured the retrained DeepCenter (epoch 12, the
val-loss optimum) and found the model in-domain-adequate but
out-of-domain COLLAPSED: zh recall 0.5293@0.05 against the published
DeepCenter's 0.9784, coverage of xl's misses 18.8% against 69.5%. The
leading explanation: the published model's value came from being barely
trained -- its own run diverged and its best checkpoint is epoch 2, a
generic bright-blob detector -- while our epoch 12 fit the competition
distribution and lost the transfer. That is testable with checkpoints
we already hold: epoch 5 (ep0005 snapshot) and epoch 7 (ep0010
snapshot's best), against epoch 12.

If ep5/ep7 restore the zh coverage: the "second detector" value is an
UNDERTRAINING artifact we can now dial deliberately, with all three
eval domains clean. If they do not: the published model's coverage is a
property of that particular artifact, and this line closes.

ALSO FIXED from dccache2: the 12 6bba holdout xl caches live in
biohub-edge-ranker's output (post_sweep.py's own note), which dccache2
did not attach -- its 6bba coverage column was empty. Attached now, so
the ep12 6bba row completes as well.

MECHANICS. ep12 detections are NOT recomputed: dccache2's dcdet_*.npz
are attached and read back. ep5/ep7 run fresh inference (one pooled
normalisation per frame, both models on it). ep5/ep7 detections are
saved as dcdet_ep05_*/dcdet_ep07_* so the winning stage is usable
downstream without another GPU pass. Post-processing identical
throughout (peaks at 0.05, sub-voxel refine, 4.0 um NMS); matching
greedy 1:1 at the scorer's 7.0 um.

⚠ Denominators: 6bba/44b6 recall is over annotated cells (~9.7% /
~0.77%); zh over all cells. Compare checkpoints and xl WITHIN a row.

Attach: competition, faustine649/cell-tracking,
faustine649/biohub-dctrain-175-weights (v2: three checkpoints), kernel
outputs biohub-cache-train, biohub-edge-ranker, biohub-zh-detect,
biohub-zebrahub-extract, biohub-dccache2. T4 pinned.
"""
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-29a-dccache3"
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
CAND_THR = 0.05
NMS_UM = 4.0
MATCH_UM = 7.0

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
from viz_cells import find_root, geff_path, read_geff  # noqa: E402

ROOT = find_root(None)
TRAIN_DIR = pathlib.Path(ROOT) / "train"


def _walk_files(pred):
    import os
    hits = []
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if pred(fn):
                hits.append(pathlib.Path(root_) / fn)
    return sorted(hits)


_FILES = _walk_files(lambda fn: fn in ("best.pt", "best_ep05.pt",
                                       "best_ep07.pt", "split_manifest.json")
                     or fn.startswith(("detcache_", "zhdet_", "dcdet_")))

# ---- checkpoints ------------------------------------------------------
WEIGHT_DIR = None
for p in _FILES:
    if p.name == "best_ep05.pt":
        WEIGHT_DIR = p.parent
        break
assert WEIGHT_DIR is not None, (
    "best_ep05.pt not found -- attach biohub-dctrain-175-weights v2")
CKPTS = [("ep05", WEIGHT_DIR / "best_ep05.pt", 5),
         ("ep07", WEIGHT_DIR / "best_ep07.pt", 7),
         ("ep12", WEIGHT_DIR / "best.pt", 12)]

MANIFEST = WEIGHT_DIR / "split_manifest.json"
assert MANIFEST.exists(), f"{MANIFEST} missing"
seen = set(json.loads(MANIFEST.read_text())["all"])
names = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
assert len(names) == 199
EXCLUDED = sorted(n for n in names if n not in seen)
EX_6BBA = [n for n in EXCLUDED if n.startswith("6bba_")]
EX_44B6 = [n for n in EXCLUDED if n.startswith("44b6_")]
assert len(EX_6BBA) == 12 and len(EX_44B6) == 12

# ---- model ------------------------------------------------------------
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


MODELS = {}
NORM = None
for tag, path, want_ep in CKPTS:
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    got = int(ck.get("epoch", -1))
    assert got == want_ep, (
        f"{tag}: checkpoint at {path} is epoch {got}, expected {want_ep} -- "
        f"the snapshot layout is not what this script assumes")
    cfg = dict(ck.get("config", {}))
    n = (float(cfg.get("norm_lo_pct", 50.0)), float(cfg.get("norm_hi_pct", 99.5)),
         float(cfg.get("norm_clip_lo", -0.5)), float(cfg.get("norm_clip_hi", 6.0)))
    assert NORM is None or n == NORM, "checkpoints disagree on normalisation"
    NORM = n
    m = _DCDeepCenterUNet3D(base_channels=int(cfg.get("base_channels", 24)))
    m.load_state_dict(ck["model_state"])
    MODELS[tag] = m.to(DEVICE).eval()
    print(f"{tag}: epoch {got}, val best_score {ck.get('best_score')}",
          flush=True)
    del ck

# ---- shared pieces ----------------------------------------------------
def read_array_meta(zp):
    with open(pathlib.Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


def load_volume(zp, t, meta=None):
    try:
        import zarr
        return np.asarray(zarr.open(str(zp), mode="r")["0"][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(pathlib.Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])


def _dc_pool_xy(volume, factor=POOL):
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor,
                           x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize(volume):
    lo_pct, hi_pct, clip_lo, clip_hi = NORM
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, lo_pct))
    hi = float(np.percentile(vol, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


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


def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x


def dc_detect_multi(vol, tags):
    """One pooled+normalised volume, N models over it. Returns
    {tag: (coords, scores)} with the shared post-processing."""
    x = _dc_normalize(_dc_pool_xy(vol))
    out = {}
    for tag in tags:
        acc = None
        for i in (0, 1, 2, 3):
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = MODELS[tag](torch.from_numpy(xv)[None, None].to(DEVICE)
                                 )[0, 0].float().cpu().numpy()
            lg = _flip(lg, i)
            acc = lg if acc is None else acc + lg
        h = 1.0 / (1.0 + np.exp(-np.clip(acc / 4.0, -60.0, 60.0)))
        pk = peak_local_max(h, min_distance=1, threshold_abs=CAND_THR,
                            exclude_border=False)
        if len(pk) == 0:
            out[tag] = (np.zeros((0, 3)), np.zeros(0))
            continue
        sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
        coords = pk.astype(float)
        coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
        coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
        ref = np.array([_refine(vol, c) for c in coords])
        out[tag] = _physical_nms(ref, sc, NMS_UM)
    return out


# ---- caches -----------------------------------------------------------
XL_COMP, XL_ZH, DCDET2 = {}, {}, {}
for p in _FILES:
    nm = p.name
    if nm.startswith("detcache_") and nm.endswith(".npz"):
        XL_COMP.setdefault(nm[len("detcache_"):-len(".npz")], p)
    elif nm.startswith("zhdet_") and nm.endswith(".npz"):
        XL_ZH.setdefault(nm[len("zhdet_"):-len(".npz")], p)
    elif nm.startswith("dcdet_") and nm.endswith(".npz") \
            and not nm.startswith(("dcdet_ep05_", "dcdet_ep07_")):
        DCDET2.setdefault(nm[len("dcdet_"):-len(".npz")], p)
print(f"caches: {len(XL_COMP)} xl competition, {len(XL_ZH)} xl zh, "
      f"{len(DCDET2)} ep12 dcdet", flush=True)
missing_xl = [n for n in EX_6BBA + EX_44B6 if n not in XL_COMP]
assert not missing_xl, (
    f"xl detcache missing for {missing_xl} -- attach biohub-edge-ranker "
    f"AND biohub-cache-train")
missing_dc = [n for n in EX_6BBA + EX_44B6 if n not in DCDET2]
assert not missing_dc, (
    f"ep12 dcdet missing for {missing_dc} -- attach biohub-dccache2")

CROPS = None
for c in ["/kaggle/input/biohub-zebrahub-extract/crops",
          "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(c).exists():
        CROPS = pathlib.Path(c)
        break
if CROPS is None:
    import os
    for root_, dirs, _fn in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        if "crops" in dirs:
            CROPS = pathlib.Path(root_) / "crops"
            break
assert CROPS is not None, "zebrahub crops not mounted"
ZH_FILES = sorted(CROPS.glob("*.npz"))[:8]
assert len(ZH_FILES) == 8

# ---- matching ---------------------------------------------------------
def _claimed(coords, sc, gate, gt_um):
    keep = sc >= gate
    coords, sc = coords[keep], sc[keep]
    claimed = set()
    if len(coords) == 0 or len(gt_um) == 0:
        return claimed
    det_um = coords * SCALE[None, :]
    tree = cKDTree(gt_um)
    for i in np.argsort(-sc):
        near = tree.query_ball_point(det_um[i], MATCH_UM)
        if not near:
            continue
        near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
        free = [g for g in near if g not in claimed]
        if free:
            claimed.add(free[0])
    return claimed


AGG = {}


def _tally(tag, domain, gt_um, dc_c, dc_s, xl_c, xl_s):
    a = AGG.setdefault((tag, domain), dict(gt=0, dc05=0, dc15=0, xl15=0,
                                           xl_miss=0, dc_cover=0))
    n = len(gt_um)
    a["gt"] += n
    cl05 = _claimed(dc_c, dc_s, 0.05, gt_um)
    a["dc05"] += len(cl05)
    a["dc15"] += len(_claimed(dc_c, dc_s, 0.15, gt_um))
    cl_xl = _claimed(xl_c, xl_s, 0.15, gt_um)
    a["xl15"] += len(cl_xl)
    xl_miss = set(range(n)) - cl_xl
    a["xl_miss"] += len(xl_miss)
    a["dc_cover"] += len(xl_miss & cl05)


def _at(npz, t):
    kc, ks = f"c{t}", f"s{t}"
    if npz is not None and kc in npz.files:
        return np.asarray(npz[kc], float), np.asarray(npz[ks], float)
    return np.zeros((0, 3)), np.zeros(0)


# ---- run --------------------------------------------------------------
OUT = pathlib.Path("/kaggle/working")
FRESH = ("ep05", "ep07")
t0 = time.time()
done = 0

for nm in EX_6BBA + EX_44B6:
    zp = TRAIN_DIR / f"{nm}.zarr"
    meta = read_array_meta(zp)
    T = int(meta["shape"][0])
    gt = read_geff(geff_path(ROOT, "train", nm))
    gbt = {}
    for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
        gbt.setdefault(int(tt), []).append([float(z_), float(y_), float(x_)])
    xl = np.load(XL_COMP[nm])
    dc12 = np.load(DCDET2[nm])
    domain = "6bba" if nm.startswith("6bba_") else "44b6"
    outs = {tag: {} for tag in FRESH}
    for t in range(T):
        vol = np.asarray(load_volume(zp, t, meta))
        dets = dc_detect_multi(vol, FRESH)
        for tag in FRESH:
            outs[tag][f"c{t}"] = np.asarray(dets[tag][0], np.float32)
            outs[tag][f"s{t}"] = np.asarray(dets[tag][1], np.float32)
        gi = gbt.get(t)
        if gi:
            gt_um = np.asarray(gi, float) * SCALE[None, :]
            xc, xs = _at(xl, t)
            for tag in FRESH:
                _tally(tag, domain, gt_um, *dets[tag], xc, xs)
            _tally("ep12", domain, gt_um, *_at(dc12, t), xc, xs)
        if t % 25 == 0:
            gc.collect()
    for tag in FRESH:
        np.savez_compressed(OUT / f"dcdet_{tag}_{nm}.npz", **outs[tag])
    done += 1
    print(f"  {done}/32 {nm} ({time.time() - t0:.0f}s)", flush=True)
    gc.collect()

for f in ZH_FILES:
    z = np.load(f)
    img = z["image"]
    T = img.shape[0]
    nodes = np.asarray(z["nodes"], float)
    gbt = {}
    for r in range(len(nodes)):
        gbt.setdefault(int(nodes[r, 0]), []).append(nodes[r, 1:].tolist())
    xl = np.load(XL_ZH[f.stem]) if f.stem in XL_ZH else None
    dc12 = np.load(DCDET2[f.stem]) if f.stem in DCDET2 else None
    if dc12 is None:
        print(f"  WARN {f.stem}: no ep12 dcdet -> that row will be short",
              flush=True)
    outs = {tag: {} for tag in FRESH}
    for t in range(T):
        vol = np.asarray(img[t])
        dets = dc_detect_multi(vol, FRESH)
        for tag in FRESH:
            outs[tag][f"c{t}"] = np.asarray(dets[tag][0], np.float32)
            outs[tag][f"s{t}"] = np.asarray(dets[tag][1], np.float32)
        gi = gbt.get(t)
        if gi:
            gt_um = np.asarray(gi, float) * SCALE[None, :]
            xc, xs = _at(xl, t)
            for tag in FRESH:
                _tally(tag, "zh", gt_um, *dets[tag], xc, xs)
            if dc12 is not None:
                _tally("ep12", "zh", gt_um, *_at(dc12, t), xc, xs)
        if t % 25 == 0:
            gc.collect()
    for tag in FRESH:
        for k in ("nodes", "track_id", "parent_track_id"):
            if k in z.files:
                outs[tag][k] = z[k]
        np.savez_compressed(OUT / f"dcdet_{tag}_{f.stem}.npz", **outs[tag])
    done += 1
    print(f"  {done}/40 {f.stem} ({time.time() - t0:.0f}s)", flush=True)
    del img, z
    gc.collect()

# ---- report -----------------------------------------------------------
print(f"\n=== EPOCH SWEEP, greedy 1:1 at {MATCH_UM} um ===", flush=True)
print("  anchors: published DeepCenter on zh read 0.9784@0.05 with 69.5% "
      "coverage of xl misses (detcmp); ep12 rows must reproduce "
      "dccache2's 0.9420/0.8751/0.5293 @0.05.", flush=True)
print(f"  {'domain':<6} {'ckpt':<5} {'GT n':>8} {'DC@.05':>8} {'DC@.15':>8} "
      f"{'xl@.15':>8} {'DCcov of xl-miss':>17}", flush=True)
for d in ("6bba", "44b6", "zh"):
    for tag in ("ep05", "ep07", "ep12"):
        a = AGG.get((tag, d))
        if not a:
            print(f"  {d:<6} {tag:<5} ABSENT", flush=True)
            continue
        n = max(1, a["gt"])
        print(f"  {d:<6} {tag:<5} {a['gt']:>8} {a['dc05'] / n:>8.4f} "
              f"{a['dc15'] / n:>8.4f} {a['xl15'] / n:>8.4f} "
              f"{a['dc_cover'] / max(1, a['xl_miss']):>8.4f} "
              f"(n={a['xl_miss']:>6})", flush=True)
print("\n  reading: if ep05/ep07 restore zh recall and coverage toward "
      "the published model's numbers, undertraining is the mechanism and "
      "the dial is ours; if all three sit together, the published "
      "model's transfer came from something else and this line closes.",
      flush=True)
print(f"\nDCCACHE3 DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
