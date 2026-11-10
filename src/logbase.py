"""What does the UNet actually buy over classical blob detection?

SCRIPT_VERSION 2026-08-26a-logbase

THE QUESTION. The detector is three UNet3D branches on a preprocessed
volume, and the preprocessing is already a classical blob enhancer:

    tophat:  p - grey_opening(p, size=(1, 7, 7))

which removes slowly-varying background and leaves small locally-bright
structures -- cells. Two of the three branches use it, and the recorded
val_recall of the three weights is 0.573 (no tophat), 0.659 (tophat),
0.579 (tophat). **The tophat branch is the best of the three by 8.6
points.** Nobody has ever asked how much of the detector's performance
survives without the network.

⚠ THE USER'S PRIOR, recorded before the run: classical will not match
the UNet, "otherwise everyone would not be using UNets". Probably right.
The value is not the headline -- it is the DECOMPOSITION:

    raw peaks           what naive local maxima on brightness give
    tophat + peaks      what the preprocessing alone gives
    LoG + peaks         what a matched-scale blob filter gives
    UNet (cached)       what ships

If tophat+peaks is already close to the UNet, then "train a better
UNet" is the wrong next move and the effort belongs elsewhere. If the
gap is large, the UNet is doing real work and improving it is worth GPU.

FAIRNESS, which is the whole design problem. Thresholds are not
comparable across methods -- a LoG response and a sigmoid probability
live on different scales. So every arm is scored at MATCHED DETECTION
COUNT: take the top-N peaks by response, with N set to what the UNet
keeps on the same frame, and report recall there. That removes the
threshold from the comparison entirely.

Everything else is held identical: same volumes, same frames, the same
pool_xy(4) downsampling so the spatial scale matches, the same
peak_local_max(min_distance=1), the same physical NMS at 4.0 um, and
the same `reachable` test against GT at MATCH_UM.

⚠ MATCH_UM here is 7.0, not the 5.0 this project's diagnostics have been
using. The official scorer matches at max_distance 7.0
(`tracking_cellmot/metrics.py: evaluate(..., max_distance=7.0)`), and
every recall number in this log measured at 5.0 is therefore stricter
than the thing being optimised. Both are printed so the two can be
reconciled.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "2026-08-26a-logbase"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
UNET_THRESH = 0.15
NMS_UM = 4.0
MATCH_UM = 7.0          # the OFFICIAL matching distance
MATCH_STRICT = 5.0      # what this project's diagnostics have used
FRAME_STEP = 5          # every 5th frame; reading volumes is the cost

# ---- environment: no support pack. Same reasoning as render6.py --------
# The pack ships a numpy that Kaggle's scipy/matplotlib cannot use, and
# --no-deps then leaves a half-installed dependency tree. This script
# needs numpy, scipy and a zarr reader, all of which Kaggle already has
# and which agree with each other.
sys.path.insert(0, "/kaggle/input")
for p in pathlib.Path("/kaggle/input").glob("*/cell-tracking*"):
    sys.path.insert(0, str(p))
for p in pathlib.Path("/kaggle/input").glob("*/*/cell-tracking*"):
    sys.path.insert(0, str(p))


def _need(mod, pkg=None):
    """Install only if genuinely missing, pinning numpy so pip cannot
    move it. --no-deps is the wrong tool: it left zarr without donfig."""
    try:
        return __import__(mod)
    except ImportError:
        pass
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        pkg or mod, f"numpy=={np.__version__}"],
                       capture_output=True, text=True)
    print(f"  pip install {pkg or mod} -> rc {r.returncode}", flush=True)
    return __import__(mod)


from scipy.ndimage import grey_opening, gaussian_laplace    # noqa: E402
from scipy.spatial import cKDTree                           # noqa: E402
from skimage.feature import peak_local_max                  # noqa: E402

_need("zarr")
try:
    _need("geff")
except Exception as _e:                                     # noqa: BLE001
    print(f"  geff unavailable ({type(_e).__name__}); raw zattrs path",
          flush=True)

from viz_cells import (find_root, geff_path, zarr_path,      # noqa: E402
                       read_geff)

root = find_root(None)
print("data root", root, flush=True)

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

# ---- the UNet's own detections, from the cached kernel output ---------
BASE = pathlib.Path("/kaggle/input")
CACHES = {}
stack = [BASE]
while stack:
    d = stack.pop()
    if d.name.endswith((".zarr", ".geff")) \
            or "cell-tracking-during-development" in d.name:
        continue
    for f in d.glob("detcache_*.npz"):
        CACHES[f.stem[len("detcache_"):]] = f
    if len(d.parts) - len(BASE.parts) < 4:
        stack += [p for p in d.iterdir() if p.is_dir()]
print(f"detection caches found: {len(CACHES)}", flush=True)
assert len(CACHES) >= 12, "attach biohub-edge-ranker as a kernel source"


def _frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


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
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(
            meta["shape"][1:])


def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def norm(p):
    """The pipeline's own normalisation, verbatim from pool_norm."""
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def physical_nms(coords, scores, radius_um, scale=SCALE):
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


def peaks_of(field, topn):
    """Local maxima of `field` on the POOLED grid, suppressed at
    NMS_UM, then the top `topn` by response. Coordinates come back in
    FULL-RESOLUTION voxels so they are comparable with the UNet's."""
    pk = peak_local_max(field, min_distance=1, exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3))
    sc = field[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    c = pk.astype(float)
    c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
    c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
    c, sc = physical_nms(c, sc, NMS_UM)
    if topn is not None and len(c) > topn:
        idx = np.argsort(-sc)[:topn]
        c = c[idx]
    return c


def reachable(pts, gt_um, r):
    if len(pts) == 0 or len(gt_um) == 0:
        return 0
    d, _ = cKDTree(pts * SCALE[None, :]).query(gt_um)
    return int((d <= r).sum())


# LoG sigmas in POOLED voxels. A bright blob of radius r peaks in the
# scale-normalised LoG at sigma = r/sqrt(3); the pooled voxel is
# 1.625 um isotropic and cells are 7-9 um across, so r is 2.2-2.8 voxels
# and the matched sigma is about 1.25-1.6.
SIGMAS = (1.0, 1.3, 1.6, 2.0)
ARMS = ["raw", "tophat"] + [f"log{s:.1f}" for s in SIGMAS] \
    + [f"tophat_log{s:.1f}" for s in SIGMAS]

hit = {a: {MATCH_UM: 0, MATCH_STRICT: 0} for a in ARMS + ["unet"]}
n_gt = 0
n_kept = 0
t0 = time.time()

for name in HOLDOUT:
    if name not in CACHES:
        print(f"  WARN {name}: no detection cache -> skipped", flush=True)
        continue
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in _frame_keys(z.files, "c")) + 1
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_zyx = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    gt_by_t = defaultdict(list)
    for r_ in range(len(g_t)):
        gt_by_t[int(g_t[r_])].append(r_)
    zp = zarr_path(root, "train", name)
    meta = read_array_meta(zp)
    tstart = time.time()
    nf = 0
    for t in range(0, min(T, meta["shape"][0]), FRAME_STEP):
        gi = gt_by_t.get(t, [])
        if not gi:
            continue
        gu = g_zyx[gi] * SCALE[None, :]

        # the UNet's own kept detections, and the count every arm matches
        cc = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        ss = np.asarray(z[f"s{t}"], float).reshape(-1)
        unet = cc[ss >= UNET_THRESH]
        topn = len(unet)
        if topn == 0:
            continue

        vol = load_volume(zp, t, meta)
        p = pool_xy(vol)
        raw = norm(p)
        th = norm(np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None))

        fields = {"raw": raw, "tophat": th}
        for s in SIGMAS:
            # scale-normalised LoG, negated: bright blobs give a maximum
            fields[f"log{s:.1f}"] = -(s ** 2) * gaussian_laplace(raw, s)
            fields[f"tophat_log{s:.1f}"] = -(s ** 2) * gaussian_laplace(th, s)

        n_gt += len(gi)
        n_kept += topn
        for r in (MATCH_UM, MATCH_STRICT):
            hit["unet"][r] += reachable(unet, gu, r)
        for a in ARMS:
            c = peaks_of(fields[a], topn)
            for r in (MATCH_UM, MATCH_STRICT):
                hit[a][r] += reachable(c, gu, r)
        nf += 1
        del vol, p, raw, th, fields
    print(f"  {name}: {nf} frames ({time.time() - tstart:.0f}s)",
          flush=True)

print(f"\n=== RECALL AT MATCHED DETECTION COUNT ===", flush=True)
print(f"  {n_gt:,} annotated GT nodes, {n_kept:,} detections kept per arm",
      flush=True)
print(f"  every arm takes the TOP-N peaks with N = what the UNet keeps on "
      f"that frame, so no threshold enters the comparison", flush=True)
print(f"\n  {'arm':<18} {'recall@7.0':>11} {'recall@5.0':>11} "
      f"{'vs UNet@7.0':>12}", flush=True)
base = hit["unet"][MATCH_UM] / max(1, n_gt)
for a in ["unet"] + ARMS:
    r7 = hit[a][MATCH_UM] / max(1, n_gt)
    r5 = hit[a][MATCH_STRICT] / max(1, n_gt)
    tag = "" if a == "unet" else f"{r7 - base:>+12.4f}"
    print(f"  {a:<18} {r7:>11.4f} {r5:>11.4f} {tag}", flush=True)

print("\n=== READING IT ===", flush=True)
print("  The user's prior, recorded before the run: classical will not "
      "match the UNet. The number that matters is not whether it does, "
      "but where the gap sits.", flush=True)
print("  tophat close to UNet  -> most of the work is the PREPROCESSING, "
      "and 'train a better UNet' is the wrong next move.", flush=True)
print("  tophat far below      -> the network is doing real work and "
      "improving it is worth GPU.", flush=True)
print("  raw far below tophat  -> confirms the preprocessing is load "
      "bearing, which the recorded val_recall (0.573 no-tophat against "
      "0.659 tophat) already suggests.", flush=True)
print("\n  ⚠ recall@5.0 is printed only to reconcile with this project's "
      "earlier numbers. The official scorer matches at 7.0, so every "
      "recall figure measured at 5.0 understates what the metric sees.",
      flush=True)
print(f"\n  {time.time() - t0:.0f}s total", flush=True)
print("\nLOGBASE DONE", flush=True)
