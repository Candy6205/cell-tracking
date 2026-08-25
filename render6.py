"""Render two 6bba datasets as video, so the data can be LOOKED AT.

SCRIPT_VERSION 2026-08-25a-render6

WHY. `biohub-nms4` measured, in 6bba only, corr(reachable before any
suppression, true cells per frame) = **-0.5580, p=0.059** -- denser 6bba
datasets have a smaller fraction of annotated cells with a detection
peak nearby, before suppression runs at all. 44b6 (+0.031) and Zebrahub
(-0.027) show nothing.

Two reasons that number cannot be trusted as it stands, both worth
knowing before rendering anything:

1. **p = 0.059 on twelve points**, and six correlations were computed,
   so the effective bar is higher than 0.05, not lower.
2. **Leverage.** The twelve true densities are
   50, 51, 58, 61, 64, 70, 96, 101, 120, 209, 246, **521**.
   Eleven sit under 250 and one is more than double the next. A Pearson
   correlation on that shape is close to a two-group comparison, and the
   single point at 521 largely decides it.

That point is `6bba_3abfe10a`. So render IT, and render the sparsest
dataset as a control, and look.

THE HYPOTHESIS BEING LOOKED AT. Cell count in an embryo rises with
developmental time, so "a dense dataset" may simply be "a later one".
Later cells have divided more, so they are smaller and individually
dimmer, and the detector's three branches all read brightness. If that
is the mechanism, misses should concentrate in the LATE frames and in
the crowded regions, and the sparse control should stay clean
throughout.

If instead the misses are spread evenly over time, the density story is
wrong and -0.5580 is one dataset being unusual for some other reason.

WHAT IS DRAWN, per frame, on a maximum-intensity projection along z:

    green  annotated GT cell
    blue   detection kept after NMS (what the pipeline actually has)
    RED    GT cell with NO kept detection within MATCH_UM

Red is the point of the whole thing. It is judged against the POST-NMS
detections, i.e. what the linker downstream actually receives.

⚠ The GT here is SPARSE -- 9.71% of 6bba's cells are annotated. Green
and red mark only annotated cells. Most cells on screen have no marker
and that is not an error. Nothing about unannotated cells can be read
off this video.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "2026-08-25a-render6"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 5.0
UNET_THRESH = 0.15

# the dense outlier that may be carrying -0.5580, and the sparse control
DENSE = "6bba_3abfe10a"     # true density 521, NMS loss 1.3%
SPARSE = "6bba_aeee7805"    # true density  50, NMS loss 0.0%
TARGETS = [DENSE, SPARSE]

TARGET_W = 900              # output width in pixels; height follows aspect
FPS = 10
CRF = 23                    # H.264 quality; lower is bigger and better

# ---- environment -------------------------------------------------------
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")

r = subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/royerlab/"
                    "kaggle-cell-tracking-competition.git",
                    "/kaggle/working/repo"], capture_output=True, text=True)
print("git clone exit", r.returncode, flush=True)

WH = pathlib.Path("/kaggle/working/wheelhouse")
WH.mkdir(parents=True, exist_ok=True)
for w in sorted(PACK.rglob("*.whl")):
    (WH / w.name).write_bytes(w.read_bytes())
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--no-deps"]
                   + [str(w) for w in sorted(WH.glob("*.whl"))],
                   capture_output=True, text=True)
print("pip wheels exit", r.returncode, flush=True)

sys.path.insert(0, "/kaggle/working/repo/src")
sys.path.insert(0, "/kaggle/working/repo/scripts")
sys.path.insert(0, "/kaggle/input")
for p in pathlib.Path("/kaggle/input").glob("*/cell-tracking*"):
    sys.path.insert(0, str(p))
for p in pathlib.Path("/kaggle/input").glob("*/*/cell-tracking*"):
    sys.path.insert(0, str(p))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402
from viz_cells import (find_root, geff_path, zarr_path,          # noqa: E402
                       read_geff)

root = find_root(None)
print("data root", root, flush=True)

# ---- detection caches --------------------------------------------------
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
for n in TARGETS:
    assert n in CACHES, f"{n} has no detection cache -- attach biohub-edge-ranker"


def _frame_keys(files, prefix):
    """prefix + an integer. Guarding with a digit test rather than
    startswith alone, because a GT array named parent_track_id killed
    nms3 v1 that way."""
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


def load_dets(name):
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in _frame_keys(z.files, "c")) + 1
    return [(np.asarray(z[f"c{t}"], float).reshape(-1, 3),
             np.asarray(z[f"s{t}"], float).reshape(-1)) for t in range(T)]


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


def render(name):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    zp = zarr_path(root, "train", name)
    meta = read_array_meta(zp)
    T_img, Z, Y, X = meta["shape"]
    print(f"  volume {Z} x {Y} x {X} per frame, {T_img} frames, "
          f"dtype {meta['dtype']}", flush=True)

    gt = read_geff(geff_path(root, "train", name))
    est = gt.get("extra", {}).get("estimated_number_of_nodes")
    g_t = np.asarray(gt["t"], int)
    g_zyx = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    dets = load_dets(name)
    T = min(T_img, len(dets))
    print(f"  GT nodes {len(g_t)}, annotated frames "
          f"{len(set(g_t.tolist()))}, true cells/frame "
          f"{(float(est) / max(1, T)) if est else float('nan'):.0f}",
          flush=True)

    step = max(1, X // TARGET_W)
    print(f"  downsampling y/x by {step} -> {Y // step} x {X // step}",
          flush=True)

    frames_dir = pathlib.Path("/kaggle/working/frames")
    frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    n_gt_tot = n_miss_tot = 0
    per_frame = []
    for t in range(T):
        vol = load_volume(zp, t, meta)
        mip = vol.max(axis=0)[::step, ::step].astype(np.float32)
        lo, hi = np.percentile(mip, (1.0, 99.5))
        mip = np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)

        gi = np.where(g_t == t)[0]
        cc, ss = dets[t]
        keep = cc[ss >= UNET_THRESH]

        miss = np.zeros(len(gi), bool)
        if len(gi):
            gu = g_zyx[gi] * SCALE[None, :]
            if len(keep):
                d, _ = cKDTree(keep * SCALE[None, :]).query(gu)
                miss = d > MATCH_UM
            else:
                miss[:] = True
        n_gt_tot += len(gi)
        n_miss_tot += int(miss.sum())
        per_frame.append((t, len(gi), int(miss.sum()), len(keep)))

        h, w = mip.shape
        fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mip, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if len(keep):
            ax.scatter(keep[:, 2] / step, keep[:, 1] / step, s=6,
                       c="#3da5ff", marker=".", linewidths=0, alpha=0.75)
        if len(gi):
            gz = g_zyx[gi]
            ok = ~miss
            if ok.any():
                ax.scatter(gz[ok, 2] / step, gz[ok, 1] / step, s=110,
                           facecolors="none", edgecolors="#28d17c",
                           linewidths=1.4)
            if miss.any():
                ax.scatter(gz[miss, 2] / step, gz[miss, 1] / step, s=150,
                           facecolors="none", edgecolors="#ff3b30",
                           linewidths=2.2)
        ax.text(0.012, 0.977,
                f"{name}   t={t:3d}/{T - 1}   detections {len(keep)}   "
                f"annotated {len(gi)}   MISSED {int(miss.sum())}",
                transform=ax.transAxes, va="top", ha="left",
                color="#ffffff", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.55, pad=3,
                          edgecolor="none"))
        ax.text(0.012, 0.020,
                "green = annotated cell found   |   red = annotated cell "
                "with NO detection within 5 um   |   blue = detection "
                "kept after NMS",
                transform=ax.transAxes, va="bottom", ha="left",
                color="#dddddd", fontsize=7.5,
                bbox=dict(facecolor="black", alpha=0.55, pad=3,
                          edgecolor="none"))
        fig.savefig(frames_dir / f"f{t:04d}.png", dpi=100)
        plt.close(fig)
        if t % 20 == 0:
            print(f"    t={t:3d}  detections {len(keep):5d}  "
                  f"annotated {len(gi):3d}  missed {int(miss.sum()):3d}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del vol, mip

    out = pathlib.Path(f"/kaggle/working/{name}.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-framerate", str(FPS), "-i", str(frames_dir / "f%04d.png"),
         "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        # Do not lose a run to a missing binary. imageio bundles its own
        # ffmpeg build and can write the same H.264 file.
        print("  ffmpeg binary failed:", (r.stdout + r.stderr)[-300:],
              flush=True)
        print("  falling back to imageio's bundled encoder", flush=True)
        try:
            names = sorted(frames_dir.glob("*.png"))
            with imageio.get_writer(str(out), fps=FPS, codec="libx264",
                                    quality=None,
                                    output_params=["-crf", str(CRF),
                                                   "-pix_fmt", "yuv420p"]
                                    ) as wr:
                for f in names:
                    wr.append_data(imageio.imread(f))
        except Exception as exc:
            print(f"  fallback FAILED too: {type(exc).__name__}: {exc}",
                  flush=True)
            return None
    if not out.exists():
        print("  no video was written", flush=True)
        return None
    mb = out.stat().st_size / 1e6
    print(f"  wrote {out.name}  {mb:.2f} MB  ({time.time() - t0:.0f}s)",
          flush=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    # ---- does the miss rate rise with TIME inside this one dataset? ----
    # This is the question the 12-point correlation could not answer:
    # here every ANNOTATED frame is a point, and "which dataset" is held
    # fixed, so a trend can only be about time.
    pts = [(t, g, m) for (t, g, m, _) in per_frame if g > 0]
    print(f"  annotated frames: {len(pts)}, GT {n_gt_tot}, "
          f"missed {n_miss_tot} ({100.0 * n_miss_tot / max(1, n_gt_tot):.1f}%)",
          flush=True)
    if len(pts) > 4:
        tt = np.array([p[0] for p in pts], float)
        rate = np.array([p[2] / p[1] for p in pts], float)
        r_pear = float(np.corrcoef(tt, rate)[0, 1])
        # Spearman: ranks only, so one extreme frame cannot decide it.
        # The leverage problem in nms4's 12-point correlation is exactly
        # what this guards against.
        ra = np.argsort(np.argsort(tt)).astype(float)
        rb = np.argsort(np.argsort(rate)).astype(float)
        r_spear = float(np.corrcoef(ra, rb)[0, 1])
        print(f"  corr(miss rate, frame number)  pearson {r_pear:+.4f}  "
              f"spearman {r_spear:+.4f}   over {len(pts)} frames",
              flush=True)
        half = len(pts) // 2
        e = sum(p[2] for p in pts[:half]) / max(1, sum(p[1] for p in pts[:half]))
        l = sum(p[2] for p in pts[half:]) / max(1, sum(p[1] for p in pts[half:]))
        print(f"  miss rate, first half {100 * e:.1f}%  "
              f"second half {100 * l:.1f}%", flush=True)
    return out


made = []
for n in TARGETS:
    try:
        p = render(n)
        if p:
            made.append(p)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  {n} FAILED: {type(exc).__name__}: {exc}", flush=True)

print("\n=== OUTPUT ===", flush=True)
tot = 0.0
for p in made:
    mb = p.stat().st_size / 1e6
    tot += mb
    print(f"  {p.name}  {mb:.2f} MB", flush=True)
print(f"  {tot:.2f} MB total", flush=True)
print("\n  READ IT: if the detector is failing on LATE frames because "
      "cells have divided into smaller dimmer ones, the dense dataset's "
      "miss rate should rise with frame number and the sparse control's "
      "should not. If both are flat, nms4's -0.5580 is not about "
      "developmental time and one dataset is simply unusual.", flush=True)
print("\n  ⚠ The GT is SPARSE (9.71% of 6bba cells annotated). Green and "
      "red mark annotated cells only; most cells on screen carry no "
      "marker and nothing about them can be read off this video.",
      flush=True)
print("\nRENDER6 DONE", flush=True)
