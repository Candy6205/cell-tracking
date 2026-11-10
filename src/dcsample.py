"""dcsample.py -- ep05's RAW heatmap response at every xl detection.

SCRIPT_VERSION 2026-08-30a-dcsample

THE QUESTION. vetoscore proved that the PEAK-LIST proxy for "DeepCenter
is silent here" is broken: dcdet's 4 um NMS erases the peaks of real
cells that sit in a stronger neighbour's shadow, so the proxy vetoed
12-18% of all points (4-6x the local prediction) and lost 0.05-0.07
adj_J in all three domains. The claim the user actually proposed -- veto
where the ep05 HEATMAP response is ~zero -- was measured locally at ~90%
veto precision with 0.2-0.9% collateral and has never been run through
the pipeline, because no cache holds the heatmap value at xl's points.

THIS KERNEL builds that cache. For the 12+12 competition eval sets and
the 8 zh eval crops: run ep05 inference per frame (same pooled
normalisation, same 4-flip TTA as dccache3), then record for EVERY
cached xl detection the max heatmap value in a +-1 pooled-voxel window
around it (the same window fuserows' evidence() used). Output
hdval_<name>.npz with key h{t} aligned row-for-row with the detcache's
c{t}. No peaks, no NMS, no refine -- the map is read, not summarised.

vetoscore2 (CPU) then reruns the three-domain harness with the veto
rule the user designed: drop xl points whose hd < tau, tau in
{0.02, 0.05, 0.10}.

Attach: competition, faustine649/cell-tracking,
faustine649/biohub-dctrain-175-weights (v2), kernel outputs
biohub-cache-train, biohub-edge-ranker, biohub-zh-detect,
biohub-zebrahub-extract. T4 pinned.
"""
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_VERSION = "2026-08-30a-dcsample"
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
from viz_cells import find_root  # noqa: E402

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


_FILES = _walk_files(lambda fn: fn in ("best_ep05.pt", "split_manifest.json")
                     or fn.startswith(("detcache_", "zhdet_")))

# ---- checkpoint -------------------------------------------------------
WEIGHT_DIR = None
for p in _FILES:
    if p.name == "best_ep05.pt":
        WEIGHT_DIR = p.parent
        break
assert WEIGHT_DIR is not None, (
    "best_ep05.pt not found -- attach biohub-dctrain-175-weights v2")

MANIFEST = WEIGHT_DIR / "split_manifest.json"
assert MANIFEST.exists(), f"{MANIFEST} missing"
seen = set(json.loads(MANIFEST.read_text())["all"])
names = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
assert len(names) == 199
EXCLUDED = sorted(n for n in names if n not in seen)
EX_6BBA = [n for n in EXCLUDED if n.startswith("6bba_")]
EX_44B6 = [n for n in EXCLUDED if n.startswith("44b6_")]
assert len(EX_6BBA) == 12 and len(EX_44B6) == 12


# ---- model (verbatim from dccache3) -----------------------------------
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


ck = torch.load(WEIGHT_DIR / "best_ep05.pt", map_location=DEVICE,
                weights_only=False)
got = int(ck.get("epoch", -1))
assert got == 5, f"best_ep05.pt is epoch {got}, expected 5"
cfg = dict(ck.get("config", {}))
NORM = (float(cfg.get("norm_lo_pct", 50.0)),
        float(cfg.get("norm_hi_pct", 99.5)),
        float(cfg.get("norm_clip_lo", -0.5)),
        float(cfg.get("norm_clip_hi", 6.0)))
MODEL = _DCDeepCenterUNet3D(base_channels=int(cfg.get("base_channels", 24)))
MODEL.load_state_dict(ck["model_state"])
MODEL = MODEL.to(DEVICE).eval()
print(f"ep05: epoch {got}, val best_score {ck.get('best_score')}",
      flush=True)
del ck

# ---- shared pieces (verbatim from dccache3) ---------------------------
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


def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x


def heatmap_of(vol):
    """ep05's TTA-averaged sigmoid heatmap on the pooled grid --
    dc_detect_multi minus everything after the sigmoid."""
    x = _dc_normalize(_dc_pool_xy(vol))
    acc = None
    for i in (0, 1, 2, 3):
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = MODEL(torch.from_numpy(xv)[None, None].to(DEVICE)
                       )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    return 1.0 / (1.0 + np.exp(-np.clip(acc / 4.0, -60.0, 60.0)))


def sample_h(h, c):
    """Max heatmap value in a +-1 pooled-voxel window around each
    full-res xl detection -- the same window fuserows' evidence() read.
    c is (N,3) full-res voxel coords (z, y, x)."""
    Zp, Yp, Xp = h.shape
    out = np.zeros(len(c), np.float32)
    for k in range(len(c)):
        zc = int(round(float(c[k, 0])))
        yc = int(round((float(c[k, 1]) - (POOL - 1) / 2) / POOL))
        xc = int(round((float(c[k, 2]) - (POOL - 1) / 2) / POOL))
        zc = min(max(zc, 0), Zp - 1)
        yc = min(max(yc, 0), Yp - 1)
        xc = min(max(xc, 0), Xp - 1)
        out[k] = h[max(0, zc - 1):zc + 2,
                   max(0, yc - 1):yc + 2,
                   max(0, xc - 1):xc + 2].max()
    return out


# ---- caches -----------------------------------------------------------
XL_COMP, XL_ZH = {}, {}
for p in _FILES:
    nm = p.name
    if nm.startswith("detcache_") and nm.endswith(".npz"):
        XL_COMP.setdefault(nm[len("detcache_"):-len(".npz")], p)
    elif nm.startswith("zhdet_") and nm.endswith(".npz"):
        XL_ZH.setdefault(nm[len("zhdet_"):-len(".npz")], p)
print(f"caches: {len(XL_COMP)} detcache, {len(XL_ZH)} zhdet", flush=True)
_miss = [n for n in EX_6BBA + EX_44B6 if n not in XL_COMP]
assert not _miss, f"detcache missing for {_miss[:4]}"

CROPS = None
for c_ in ["/kaggle/input/biohub-zebrahub-extract/crops",
           "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(c_).exists():
        CROPS = pathlib.Path(c_)
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
_mzh = [f.stem for f in ZH_FILES if f.stem not in XL_ZH]
assert not _mzh, f"zhdet missing for {_mzh}"


def _at(npz, t):
    kc = f"c{t}"
    if kc in npz.files:
        return np.asarray(npz[kc], float).reshape(-1, 3)
    return np.zeros((0, 3))


# ---- run --------------------------------------------------------------
OUT = pathlib.Path("/kaggle/working")
t0 = time.time()
done = 0
tot_pts = 0
HIST = np.zeros(6, np.int64)   # <0.02, <0.05, <0.10, <0.15, <0.50, rest
EDGES = [0.02, 0.05, 0.10, 0.15, 0.50]

for nm in EX_6BBA + EX_44B6:
    zp = TRAIN_DIR / f"{nm}.zarr"
    meta = read_array_meta(zp)
    T = int(meta["shape"][0])
    xl = np.load(XL_COMP[nm])
    outs = {}
    for t in range(T):
        c = _at(xl, t)
        if len(c) == 0:
            outs[f"h{t}"] = np.zeros(0, np.float32)
            continue
        vol = np.asarray(load_volume(zp, t, meta))
        hv = sample_h(heatmap_of(vol), c)
        outs[f"h{t}"] = hv
        tot_pts += len(hv)
        HIST += np.histogram(hv, bins=[0] + EDGES + [1.01])[0]
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"hdval_{nm}.npz", **outs)
    done += 1
    print(f"  {done}/40 {nm} ({time.time() - t0:.0f}s)", flush=True)
    gc.collect()

for f in ZH_FILES:
    z = np.load(f)
    img = z["image"]
    T = img.shape[0]
    xl = np.load(XL_ZH[f.stem])
    outs = {}
    for t in range(T):
        c = _at(xl, t)
        if len(c) == 0:
            outs[f"h{t}"] = np.zeros(0, np.float32)
            continue
        hv = sample_h(heatmap_of(np.asarray(img[t])), c)
        outs[f"h{t}"] = hv
        tot_pts += len(hv)
        HIST += np.histogram(hv, bins=[0] + EDGES + [1.01])[0]
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"hdval_{f.stem}.npz", **outs)
    done += 1
    print(f"  {done}/40 {f.stem} ({time.time() - t0:.0f}s)", flush=True)
    del img, z
    gc.collect()

print(f"\n=== hd distribution over ALL {tot_pts:,} xl detections ===",
      flush=True)
lo = 0.0
for e, n in zip(EDGES + [1.0], HIST):
    print(f"  [{lo:.2f}, {e:.2f}): {n:>10,} ({n / max(1, tot_pts):.2%})",
          flush=True)
    lo = e
print("  The [0, 0.02) and [0, 0.05) mass is what a veto at those taus "
      "would delete -- compare with vetoscore's 12-18%% before trusting "
      "the rerun.", flush=True)
print(f"\nDCSAMPLE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
