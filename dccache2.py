"""dccache2.py -- cache the RETRAINED DeepCenter's detections, three clean domains.

SCRIPT_VERSION 2026-08-29a-dccache2

WHAT CHANGED vs dccache (2026-08-28g): the checkpoint. That one loaded
pilkwang's published DeepCenter, trained on all 71 44b6 and selected on
all 128 6bba, so 44b6 could never be evaluated and 6bba was tainted.
This one loads `biohub-dctrain`'s best.pt, trained on 167 movies with
BOTH eval sets excluded (12 6bba holdout + 12 seeded 44b6) and selected
on 8 val movies that are in neither. All three domains are clean.

THE EXCLUSION LIST IS READ, NOT RECOMPUTED: split_manifest.json from the
dctrain output records exactly what training saw. The 24 eval names are
derived as {199 train movies} minus {manifest's `all`} and asserted to
be 12+12. If dctrain is ever re-run with a different split, this kernel
fails loudly here instead of silently caching a contaminated eval.

WHAT IT PRODUCES:
  * dcdet_<name>.npz per dataset (c{t}/s{t}, floor 0.05, our
    post-processing verbatim: same peak extraction, sub-voxel refine,
    4.0 um NMS -- only the network and normalisation are DeepCenter's);
    zh files carry the crop labels through, as dccache did.
  * an inline coverage report per domain: recall vs GT at 7.0 um (the
    scorer's radius) for DC@0.05 / DC@0.15 / xl@0.15, and the mutual
    miss coverage (of the GT cells xl misses, how many DC claims, and
    the reverse). xl detections come from the existing caches
    (biohub-cache-train for competition sets, biohub-zh-detect for
    crops) -- no xl inference here.

⚠ Denominators (working-agreement rule): 6bba recall is over ~9.7%
annotated cells, 44b6 over ~0.77%, Zebrahub over ALL cells. Within a
domain the DC-vs-xl comparison shares the denominator and is clean;
across domains only Zebrahub describes the true population.

Attach: competition (frames + geffs), faustine649/cell-tracking (code),
kernel outputs biohub-dctrain (weights), biohub-cache-train (xl caches),
biohub-zh-detect (xl zh caches), biohub-zebrahub-extract (crop images).
T4 pinned.
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

SCRIPT_VERSION = "2026-08-29a-dccache2"
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
UNET_THRESH = 0.15
NMS_UM = 4.0
MATCH_UM = 7.0        # the official scorer's constant, NOT 5.0

# ---- our code (zarr-free GT reader) -----------------------------------
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "viz_cells.py not found -- attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff  # noqa: E402

ROOT = find_root(None)
TRAIN_DIR = pathlib.Path(ROOT) / "train"

# ---- the retrained checkpoint and its split ---------------------------
def _bounded_glob(filename_pred):
    """Walk /kaggle/input pruning the competition tree and every .zarr /
    .geff directory (trap 20: a blind recursive walk over their chunk
    files appears to hang). `filename_pred` filters file names."""
    import os
    hits = []
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if filename_pred(fn):
                hits.append(pathlib.Path(root_) / fn)
    return sorted(hits)


_ALL_FILES = _bounded_glob(lambda fn: fn == "best.pt"
                           or fn.startswith("detcache_")
                           or fn.startswith("zhdet_"))
CK_HITS = [p for p in _ALL_FILES if p.name == "best.pt"
           and p.parent.name == "full_frame_center_175"]
assert CK_HITS, ("full_frame_center_175/best.pt not found -- attach the "
                 "biohub-dctrain kernel output")
DC_CKPT = CK_HITS[0]
MANIFEST = DC_CKPT.parent / "split_manifest.json"
HISTORY = DC_CKPT.parent / "history.csv"
assert MANIFEST.exists(), f"{MANIFEST} missing -- cannot verify the split"
man = json.loads(MANIFEST.read_text())
seen = set(man["all"])          # train + val = the 175 the model ever saw
names = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
assert len(names) == 199, f"expected 199 train movies, found {len(names)}"
EXCLUDED = sorted(n for n in names if n not in seen)
EX_6BBA = [n for n in EXCLUDED if n.startswith("6bba_")]
EX_44B6 = [n for n in EXCLUDED if n.startswith("44b6_")]
assert len(EX_6BBA) == 12 and len(EX_44B6) == 12, (
    f"excluded split is {len(EX_6BBA)} 6bba + {len(EX_44B6)} 44b6, "
    f"expected 12+12 -- the dctrain output does not match this kernel")
print(f"checkpoint: {DC_CKPT}", flush=True)
print(f"eval 6bba ({len(EX_6BBA)}): " + " ".join(EX_6BBA), flush=True)
print(f"eval 44b6 ({len(EX_44B6)}): " + " ".join(EX_44B6), flush=True)
if HISTORY.exists():
    lines = HISTORY.read_text().strip().splitlines()
    print(f"training history ({len(lines) - 1} epochs):", flush=True)
    for ln in lines[:1] + lines[1::5] + lines[-1:]:
        print("   ", ln[:120], flush=True)

# ---- DeepCenter architecture (dccache verbatim) -----------------------
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


_ck = torch.load(DC_CKPT, map_location=DEVICE, weights_only=False)
assert isinstance(_ck, dict) and "model_state" in _ck, (
    f"{DC_CKPT} has no 'model_state'; keys "
    f"{list(_ck)[:8] if isinstance(_ck, dict) else type(_ck)}")
_cfg = dict(_ck.get("config", {}))
print("checkpoint epoch:", _ck.get("epoch"), "best_score:",
      _ck.get("best_score"), flush=True)
print("config:", {k: _cfg[k] for k in sorted(_cfg)
                  if k in ("pool_factor", "base_channels", "norm_lo_pct",
                           "norm_hi_pct", "norm_clip_lo", "norm_clip_hi",
                           "seed", "epochs")}, flush=True)
DC = _DCDeepCenterUNet3D(base_channels=int(_cfg.get("base_channels", 24)))
DC.load_state_dict(_ck["model_state"])
DC = DC.to(DEVICE).eval()

NORM_LO = float(_cfg.get("norm_lo_pct", 50.0))
NORM_HI = float(_cfg.get("norm_hi_pct", 99.5))
CLIP_LO = float(_cfg.get("norm_clip_lo", -0.5))
CLIP_HI = float(_cfg.get("norm_clip_hi", 6.0))
assert int(_cfg.get("pool_factor", POOL)) == POOL, "pool_factor mismatch"

# ---- shared post-processing (dccache verbatim) ------------------------
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
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, NORM_LO))
    hi = float(np.percentile(vol, NORM_HI))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), CLIP_LO, CLIP_HI).astype(np.float32)


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


def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x


def dc_detect(vol):
    """Retrained DeepCenter, our post-processing, 4-flip averaging --
    dccache's dc_detect(flips=True) with the normalisation constants
    read from the checkpoint's own config instead of assumed."""
    x = _dc_normalize(_dc_pool_xy(vol))
    acc = None
    for i in (0, 1, 2, 3):
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = DC(torch.from_numpy(xv)[None, None].to(DEVICE)
                    )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    lg = acc / 4.0
    h = 1.0 / (1.0 + np.exp(-np.clip(lg, -60.0, 60.0)))
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


# ---- xl caches for the coverage comparison ----------------------------
XL_COMP = {}
XL_ZH = {}
for p in _ALL_FILES:
    if p.name.startswith("detcache_") and p.name.endswith(".npz"):
        XL_COMP.setdefault(p.name[len("detcache_"):-len(".npz")], p)
    elif p.name.startswith("zhdet_") and p.name.endswith(".npz"):
        XL_ZH.setdefault(p.name[len("zhdet_"):-len(".npz")], p)
print(f"xl caches found: {len(XL_COMP)} competition, {len(XL_ZH)} zh",
      flush=True)
assert XL_COMP, "no detcache_*.npz -- attach biohub-cache-train output"
assert XL_ZH, "no zhdet_*.npz -- attach biohub-zh-detect output"

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
assert len(ZH_FILES) == 8, f"expected 8 zh crops, found {len(ZH_FILES)}"

# ---- coverage bookkeeping ---------------------------------------------
def _claimed(coords, sc, gate, gt_um):
    """GT indices claimed by a greedy 1:1 match at MATCH_UM, highest
    score first, over detections with score >= gate."""
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


AGG = {}   # domain -> dict of counters


def _tally(domain, gt_um, dc_c, dc_s, xl_c, xl_s):
    a = AGG.setdefault(domain, dict(gt=0, dc05=0, dc15=0, xl15=0,
                                    xl_miss=0, dc_cover=0,
                                    dc_miss=0, xl_cover=0,
                                    nd_dc15=0, nd_xl15=0))
    n = len(gt_um)
    a["gt"] += n
    cl_dc05 = _claimed(dc_c, dc_s, 0.05, gt_um)
    cl_dc15 = _claimed(dc_c, dc_s, 0.15, gt_um)
    cl_xl15 = _claimed(xl_c, xl_s, 0.15, gt_um)
    a["dc05"] += len(cl_dc05)
    a["dc15"] += len(cl_dc15)
    a["xl15"] += len(cl_xl15)
    xl_miss = set(range(n)) - cl_xl15
    dc_miss = set(range(n)) - cl_dc15
    a["xl_miss"] += len(xl_miss)
    a["dc_cover"] += len(xl_miss & cl_dc05)
    a["dc_miss"] += len(dc_miss)
    a["xl_cover"] += len(dc_miss & _claimed(xl_c, xl_s, 0.05, gt_um))
    a["nd_dc15"] += int((dc_s >= 0.15).sum())
    a["nd_xl15"] += int((xl_s >= 0.15).sum())


def _xl_at(npz, t):
    kc, ks = f"c{t}", f"s{t}"
    if kc in npz.files:
        return np.asarray(npz[kc], float), np.asarray(npz[ks], float)
    return np.zeros((0, 3)), np.zeros(0)


# ======================= run ===========================================
OUT = pathlib.Path("/kaggle/working")
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
    xl = np.load(XL_COMP[nm]) if nm in XL_COMP else None
    if xl is None:
        print(f"  WARN {nm}: no xl detcache -> coverage for it skipped, "
              f"recall still reported", flush=True)
    out = {}
    domain = "6bba" if nm.startswith("6bba_") else "44b6"
    for t in range(T):
        c, s = dc_detect(np.asarray(load_volume(zp, t, meta)))
        out[f"c{t}"] = np.asarray(c, np.float32)
        out[f"s{t}"] = np.asarray(s, np.float32)
        gi = gbt.get(t)
        if gi:
            gt_um = np.asarray(gi, float) * SCALE[None, :]
            xc, xs = _xl_at(xl, t) if xl is not None else (np.zeros((0, 3)),
                                                          np.zeros(0))
            _tally(domain, gt_um, c, s, xc, xs)
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"dcdet_{nm}.npz", **out)
    done += 1
    npk = sum(int((out[f"s{t_}"] >= 0.15).sum()) for t_ in range(T))
    print(f"  {done}/32 {nm}: T={T}, {npk:,} peaks >=0.15 "
          f"({time.time() - t0:.0f}s)", flush=True)
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
    if xl is None:
        print(f"  WARN {f.stem}: no zhdet cache -> coverage skipped",
              flush=True)
    out = {}
    for t in range(T):
        c, s = dc_detect(np.asarray(img[t]))
        out[f"c{t}"] = np.asarray(c, np.float32)
        out[f"s{t}"] = np.asarray(s, np.float32)
        gi = gbt.get(t)
        if gi:
            gt_um = np.asarray(gi, float) * SCALE[None, :]
            xc, xs = _xl_at(xl, t) if xl is not None else (np.zeros((0, 3)),
                                                          np.zeros(0))
            _tally("zh", gt_um, c, s, xc, xs)
        if t % 25 == 0:
            gc.collect()
    for k in ("nodes", "track_id", "parent_track_id", "origin_um",
              "voxel_um", "t0"):
        if k in z.files:
            out[k] = z[k]
    np.savez_compressed(OUT / f"dcdet_{f.stem}.npz", **out)
    done += 1
    npk = sum(int((out[f"s{t_}"] >= 0.15).sum()) for t_ in range(T))
    print(f"  {done}/40 {f.stem}: T={T}, {npk:,} peaks >=0.15 "
          f"({time.time() - t0:.0f}s)", flush=True)
    del img, z
    gc.collect()

# ======================= report ========================================
print(f"\n=== COVERAGE, greedy 1:1 at {MATCH_UM} um ===", flush=True)
print("  denominators: 6bba/44b6 = ANNOTATED cells only (~9.7% / ~0.77% "
      "of true); zh = all cells. Compare DC to xl WITHIN a row only.",
      flush=True)
print(f"  {'domain':<6} {'GT n':>8} {'DC@.05':>8} {'DC@.15':>8} "
      f"{'xl@.15':>8} {'DCcov of xl-miss':>17} {'xlcov of DC-miss':>17} "
      f"{'DC/xl dets@.15':>15}", flush=True)
for d in ("6bba", "44b6", "zh"):
    if d not in AGG:
        print(f"  {d:<6} ABSENT", flush=True)
        continue
    a = AGG[d]
    n = max(1, a["gt"])
    print(f"  {d:<6} {a['gt']:>8} {a['dc05'] / n:>8.4f} "
          f"{a['dc15'] / n:>8.4f} {a['xl15'] / n:>8.4f} "
          f"{a['dc_cover'] / max(1, a['xl_miss']):>8.4f} (n={a['xl_miss']:>6}) "
          f"{a['xl_cover'] / max(1, a['dc_miss']):>8.4f} (n={a['dc_miss']:>6}) "
          f"{a['nd_dc15']:>7}/{a['nd_xl15']}", flush=True)
print("\n  the published DeepCenter (dccache, 2026-08-28) read on zh: "
      "recall 0.9784@0.05, covering 69.5% of xl misses. If the retrained "
      "model lands far below that, the retraining lost something the "
      "original had; if 44b6/6bba coverage is now measurable AND "
      "comparable, the retrain did its job.", flush=True)
print(f"\nDCCACHE2 DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
