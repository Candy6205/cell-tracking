"""shoulderviz: render the four movies that carry most of the shoulder misses,
with the missed cells marked, so a person can look at them.

WHY. gtpeak + gtsplit found 313 annotated cells that the F4 heatmap RESPONDS
to (window max p50 0.15) but that produce no local maximum of their own -- the
nearest real peak is ~6 um away, on a neighbour. Two further facts came out of
the CSVs offline on 2026-09-08:

  * those cells are 3-4x DIMMER than the cells F4 finds (h at the GT voxel
    p50 0.096 against 0.366 on 6bba); the distributions barely overlap;
  * the rate is a MOVIE-level property, 0.00% to 6.56% on 6bba and 0.00% to
    10.58% on 44b6, and three movies carry two thirds of it. It is NOT
    explained by detection density (corr +0.54 on 12 6bba movies, +0.29 on
    44b6, with hard counterexamples: 268e1230 has the second-worst rate at
    the LOWEST density).

So the next question is what those movies actually look like, and the cheapest
instrument for that is a person's eyes -- which is how the flicker finding
started (2026-08-25, the user watching 6bba_3abfe10a).

WHAT IT RENDERS, CPU only, no detector. MP4 where ffmpeg is available, GIF
otherwise, and it PRINTS which:

  <name>_overview     thin Z-slab MIP of the whole frame over all T
  <name>_zoom<k>      a ~65 um crop around one shoulder miss, +-5 frames
  <name>_zsweep<k>    the same crop at the miss frame, ONE z plane at a time
                      from the top of the stack to the bottom

Only cells inside the displayed slab are circled -- v1 circled every GT node
of the frame onto a slab fixed at Z//2, so cells at other depths appeared as
rings on black and the user could not tell where anything was. That was a
rendering bug, not a fact about the data.

The z sweep is v3 and it exists because of shoulderfeat: in 17 of 18 movies
the shoulder cells sit nearer a z EDGE than that movie's own cells (sign test
p = 1.5e-5). A time clip cannot show whether a cell is cut off by the end of
the stack; a sweep can. Contrast is taken from the CROP, not the frame, since
the whole-frame percentiles are set by the bright core of the embryo and these
cells are by definition the dim ones.

No detections are drawn. Running the detector would need the GPU stack and
would not change what the images show about the CELLS.
"""
import os

SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
PTS = open(SC + "shoulder_pts.py", encoding="utf-8").read()
assert PTS.count("\n    (") == 194, PTS.count("\n    (")

BODY = '''"""shoulderviz -- see the module docstring of make_shoulderviz.py."""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import imageio.v2 as imageio                                 # noqa: E402

SCRIPT_VERSION = "2026-09-08c-shoulderviz"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

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
from viz_cells import (find_root, geff_path, read_geff, open_image,   # noqa: E402
                       zarr_path, read_scale)

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SLAB = 9              # z voxels in the MIP, viz_cells' default
ZOOM_UM = 65.0        # crop half-width around a shoulder cell
ZOOM_DT = 5           # +- frames in a zoom clip
N_ZOOM = 4            # zoom clips per movie (each also gets a z sweep)
FPS = 6

''' + PTS + '''

BY_MOVIE = defaultdict(list)
for _m, _t, _n, _hg, _hw in SHOULDER:
    BY_MOVIE[_m].append((_t, _n, _hg, _hw))
print(f"{len(SHOULDER)} shoulder points over {len(BY_MOVIE)} movies: "
      + ", ".join(f"{k} {len(v)}" for k, v in sorted(BY_MOVIE.items())),
      flush=True)

t0 = time.time()


def _slab(Z, zc, slab):
    z0 = max(0, int(zc) - slab // 2)
    z1 = min(Z, z0 + slab)
    return max(0, z1 - slab), z1


def _mip(img, t, zc, slab=SLAB, roi=None):
    z0, z1 = _slab(img.shape[1], zc, slab)
    sub = np.asarray(img[t, z0:z1])
    if roi is not None:
        y0, y1, x0, x1 = roi
        sub = sub[:, y0:y1, x0:x1]
    return sub.max(axis=0)


def _save(path_noext, frames, fps):
    """MP4 if ffmpeg is here, GIF if it is not -- and SAY WHICH.

    The user asked for mp4 because a GIF's palette and a single still both
    lose the dim cells this run is about. A silent fall back to GIF would
    hand back exactly the file that was rejected, so the fallback prints.
    """
    try:
        imageio.mimsave(f"{path_noext}.mp4", frames, fps=fps, codec="libx264",
                        quality=9, macro_block_size=1)
        return f"{path_noext}.mp4"
    except Exception as e:                                   # noqa: BLE001
        print(f"  [degraded] mp4 failed ({type(e).__name__}: {e}) -> GIF. "
              f"The frames are identical; only the container changed.",
              flush=True)
        imageio.mimsave(f"{path_noext}.gif", frames, fps=fps, loop=0)
        return f"{path_noext}.gif"


def _lohi(a):
    """Contrast from the CROP, not the whole frame.

    The whole-frame percentiles are set by the bright core of the embryo, so
    a dim cell at the edge of the stack renders as near-black. These cells
    ARE the dim ones; normalising on what is actually in view is the only way
    to see them, at the price that brightness is no longer comparable between
    clips. That price is worth it here and is stated in the title.
    """
    v = np.asarray(a, np.float32)
    return float(np.percentile(v, 2.0)), float(np.percentile(v, 99.7))


def _frame(ax, m, lo, hi, pts, marks, roi, title):
    """pts: (n,2) yx of GT in ROI pixels. marks: same, drawn thick red."""
    ax.clear()
    ax.imshow(m, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=90, facecolors="none",
                   edgecolors="#40c0ff", linewidths=0.8)
    if len(marks):
        ax.scatter(marks[:, 1], marks[:, 0], s=260, facecolors="none",
                   edgecolors="#ff3030", linewidths=2.0)
    ax.set_title(title, fontsize=9, color="w")
    ax.set_axis_off()


def render(name, hits):
    zp = zarr_path(ROOT, "train", name)
    img = open_image(zp)
    sc = read_scale(zp)                    # (z, y, x) um per voxel
    gt = read_geff(geff_path(ROOT, "train", name))
    T, Z = img.shape[0], img.shape[1]
    gbt = defaultdict(list)
    for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
        gbt[int(tt)].append((float(z_), float(y_), float(x_)))
    nid_xyz = {int(n): (int(tt), float(z_), float(y_), float(x_))
               for n, tt, z_, y_, x_ in zip(gt["node_ids"], gt["t"], gt["z"],
                                            gt["y"], gt["x"])}
    miss_bt = defaultdict(list)
    for tt, nn, hg, hw in hits:
        v = nid_xyz.get(nn)
        if v is None:
            print(f"  ! {name}: node {nn} not in the geff", flush=True)
            continue
        miss_bt[int(tt)].append((v[1], v[2], v[3], hg, hw))
    samp = np.asarray(img[0]).astype(np.float32)
    lo, hi = (float(np.percentile(samp, 1.0)),
              float(np.percentile(samp, 99.8)))
    zc = Z // 2

    # ---- overview -------------------------------------------------------
    # A GT cell OUTSIDE the slab would otherwise be drawn as a circle on
    # black, which reads as "the detector missed a cell that is not there".
    # It is not there IN THIS SLICE. So only in-slab cells are circled, and
    # the title says how many of the frame's GT that is.
    zs0, zs1 = _slab(Z, zc, SLAB)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="k")
    fr = []
    for t in range(T):
        m = _mip(img, t, zc)
        allg = gbt.get(t, [])
        g = np.asarray([[y_, x_] for z_, y_, x_ in allg
                        if zs0 <= z_ < zs1], float)
        k = np.asarray([[y_, x_] for z_, y_, x_, _a, _b in miss_bt.get(t, [])
                        if zs0 <= z_ < zs1], float)
        _frame(ax, m, lo, hi, g.reshape(-1, 2), k.reshape(-1, 2), None,
               f"{name}  t={t}   z {zs0}-{zs1}   GT in slab {len(g)}/{len(allg)}"
               f"   shoulder miss {len(k)}")
        fig.canvas.draw()
        fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    _save(str(OUT / f"{name}_overview"), fr, FPS)
    plt.close(fig)
    print(f"  {name}_overview  {T} frames  ({time.time() - t0:.0f}s)",
          flush=True)

    # ---- zooms, dimmest first: those are the hardest cases --------------
    flat = sorted(((t, z_, y_, x_, hg, hw)
                   for t, lst in miss_bt.items()
                   for z_, y_, x_, hg, hw in lst), key=lambda r: r[4])
    step = max(1, len(flat) // N_ZOOM)
    for k, (t, z_, y_, x_, hg, hw) in enumerate(flat[::step][:N_ZOOM]):
        ry = int(round(ZOOM_UM / sc[1]))
        rx = int(round(ZOOM_UM / sc[2]))
        y0, y1 = max(0, int(y_) - ry), min(img.shape[2], int(y_) + ry)
        x0, x1 = max(0, int(x_) - rx), min(img.shape[3], int(x_) + rx)
        roi = (y0, y1, x0, x1)
        zz0, zz1 = _slab(Z, z_, 5)
        clo, chi = _lohi(np.asarray(img[t, :, y0:y1, x0:x1]))
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="k")
        fr = []
        for tt in range(max(0, t - ZOOM_DT), min(T, t + ZOOM_DT + 1)):
            m = _mip(img, tt, z_, slab=5, roi=roi)
            g = np.asarray([[yy - y0, xx - x0] for zq, yy, xx in gbt.get(tt, [])
                            if y0 <= yy < y1 and x0 <= xx < x1
                            and zz0 <= zq < zz1], float)
            mk = np.asarray([[yy - y0, xx - x0]
                             for zq, yy, xx, _a, _b in miss_bt.get(tt, [])
                             if y0 <= yy < y1 and x0 <= xx < x1
                             and zz0 <= zq < zz1], float)
            _frame(ax, m, clo, chi, g.reshape(-1, 2), mk.reshape(-1, 2), roi,
                   f"{name} zoom{k}  t={tt}  z {zz0}-{zz1} of {Z}"
                   + (f"   <- the miss (h={hg:.3f}, win={hw:.3f})"
                      if tt == t else ""))
            fig.canvas.draw()
            fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        _save(str(OUT / f"{name}_zoom{k}"), fr, 3)
        plt.close(fig)

        # ---- z sweep: ONE plane at a time through the whole stack, at the
        # miss frame. shoulderfeat found these cells sit at a z EDGE in 17 of
        # 18 movies, so the question a time clip cannot answer is whether the
        # cell is cut off by the end of the stack. This answers it directly:
        # watch where the marker's cell appears and whether it is still
        # growing when the planes run out.
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="k")
        fr = []
        for zq in range(Z):
            m = np.asarray(img[t, zq, y0:y1, x0:x1])
            g = np.asarray([[yy - y0, xx - x0] for zc2, yy, xx in gbt.get(t, [])
                            if y0 <= yy < y1 and x0 <= xx < x1
                            and abs(zc2 - zq) <= 1], float)
            mk = (np.asarray([[y_ - y0, x_ - x0]], float)
                  if abs(z_ - zq) <= 1 else np.zeros((0, 2)))
            _frame(ax, m, clo, chi, g.reshape(-1, 2), mk.reshape(-1, 2), roi,
                   f"{name} zsweep{k}  t={t}  z={zq}/{Z - 1}"
                   + (f"   <- the missed cell is HERE (z={z_:.1f})"
                      if abs(z_ - zq) <= 1 else ""))
            fig.canvas.draw()
            fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        _save(str(OUT / f"{name}_zsweep{k}"), fr, 5)
        plt.close(fig)
    print(f"  {name}: {min(N_ZOOM, len(flat))} zooms, dimmest h_gt "
          f"{flat[0][4]:.3f} .. {flat[::step][:N_ZOOM][-1][4]:.3f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)
    del img


for _name in sorted(BY_MOVIE):
    print(f"\\n=== {_name} ({len(BY_MOVIE[_name])} shoulder misses) ===",
          flush=True)
    render(_name, BY_MOVIE[_name])

print(f"\\nSHOULDERVIZ DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("SHOULDERVIZ OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "shoulderviz.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(BODY)
print("written", len(BODY))
