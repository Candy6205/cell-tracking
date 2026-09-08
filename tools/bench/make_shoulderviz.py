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

MARKERS ARE NEVER DROPPED, and that took three tries. v1 circled every GT node
of the frame onto a slab pinned at Z//2, so cells at other depths drew rings
on empty space -- "I cannot tell where the cells are". v3 fixed that by hiding
anything more than one voxel off-plane, which is the same bug in a different
coat: a marker that vanishes cannot tell you whether the CELL left or the
RENDERER did. v4 draws every marker and puts the distance from the displayed
plane into alpha, line width and dash, so a cell can be followed continuously
as it leaves the slab. The user, 2026-09-08: "连贯一点其实我也能看出来细胞下去
了又或者暗掉了".

A second reason the blue circles look sparse is NOT a rendering choice and
cannot be fixed: GT annotates a median 9.71% of 6bba's cells and 0.77% of
44b6's, so a 44b6 crop holding fifty visible nuclei may carry one circle.
In TIME the annotation is dense -- 23 of the 24 measured movies have GT on
every frame of their range, the exception being 44b6_87bba6c4 which is
missing t=2.

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

SCRIPT_VERSION = "2026-09-08e-shoulderviz"
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


def _depth_style(dz, rgb, s_in, lw_in):
    """(size, linewidth, rgba, is_near) for markers |dz| voxels off-plane.

    v3 DROPPED every marker more than one voxel off the displayed plane, so
    circles blinked in and out and the user could not follow a cell at all --
    which is the same complaint as v1's, in a different disguise: a marker
    that vanishes tells you nothing about whether the CELL went away or the
    RENDERER did. So nothing is dropped any more. Distance from the plane is
    drawn instead: bright, thick and solid in-plane; fading, thinner and
    dashed as it goes away, with a floor so it never disappears entirely.
    The user, 2026-09-08: "连贯一点其实我也能看出来细胞下去了又或者暗掉了".

    v4a NOTE: v4's first cut faded to alpha 0.16 and hit that floor 8 planes
    out, which on a 64-plane stack meant every distant marker was invisible
    over bright tissue -- the drop-out bug AGAIN, this time achieved with
    opacity instead of an if-statement. The fade is now shallow and floored
    at 0.5, and the depth cue is carried mainly by RADIUS and dash density,
    which stay readable at any contrast. A dark backing ring (drawn by
    `_scat`) keeps the marker legible over a bright field.
    """
    d = np.abs(np.asarray(dz, float))
    a = np.clip(1.0 - d / 40.0, 0.50, 1.0)
    near = d <= 1.5
    size = s_in * np.clip(1.0 - d / 16.0, 0.30, 1.0)
    lw = np.where(near, lw_in, np.where(d <= 4.0, lw_in * 0.7, lw_in * 0.5))
    rgba = np.tile(np.asarray(rgb + (1.0,), float), (len(d), 1))
    rgba[:, 3] = a
    return size, lw, rgba, near


def _scat(ax, pts, dz, rgb, s_in, lw_in):
    """Two draws: solid for what is near this plane, dashed for what is not.

    matplotlib carries linestyle per collection, not per point, so the split
    is the only way to say 'this one is off-plane' with the line itself.
    """
    if not len(pts):
        return
    size, lw, rgba, near = _depth_style(dz, rgb, s_in, lw_in)
    # a dark backing ring first: half these crops are bright tissue, and a
    # thin cyan line on a white nucleus is as invisible as one that was never
    # drawn
    ax.scatter(pts[:, 1], pts[:, 0], s=size, facecolors="none",
               edgecolors=(0, 0, 0, 0.55), linewidths=lw * 2.6)
    for msk, ls in ((near, "solid"), (~near, (0, (2, 2)))):
        if msk.any():
            ax.scatter(pts[msk, 1], pts[msk, 0], s=size[msk],
                       facecolors="none", edgecolors=rgba[msk],
                       linewidths=lw[msk], linestyle=ls)


def _frame(ax, m, lo, hi, pts, pdz, marks, mdz, roi, title):
    """pts/marks: (n,2) yx in ROI pixels. pdz/mdz: voxels off the plane."""
    ax.clear()
    ax.imshow(m, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    _scat(ax, pts, pdz, (0.25, 0.75, 1.0), 90.0, 1.1)
    _scat(ax, marks, mdz, (1.0, 0.19, 0.19), 260.0, 2.2)
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
    zmid = (zs0 + zs1 - 1) / 2.0
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="k")
    fr = []
    for t in range(T):
        m = _mip(img, t, zc)
        allg = gbt.get(t, [])
        g = np.asarray([[y_, x_] for _z, y_, x_ in allg], float).reshape(-1, 2)
        gd = np.asarray([z_ - zmid for z_, _y, _x in allg], float)
        km = miss_bt.get(t, [])
        k = np.asarray([[y_, x_] for _z, y_, x_, _a, _b in km],
                       float).reshape(-1, 2)
        kd = np.asarray([z_ - zmid for z_, _y, _x, _a, _b in km], float)
        n_in = int((np.abs(gd) <= SLAB / 2.0).sum())
        _frame(ax, m, lo, hi, g, gd, k, kd, None,
               f"{name}  t={t}   slab z {zs0}-{zs1} of {Z}   GT {len(g)} "
               f"({n_in} in slab)   shoulder miss {len(k)}"
               f"\\nsolid = on this slab, dashed+faint = that many z away")
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
        zmid2 = (zz0 + zz1 - 1) / 2.0
        for tt in range(max(0, t - ZOOM_DT), min(T, t + ZOOM_DT + 1)):
            m = _mip(img, tt, z_, slab=5, roi=roi)
            vis = [(zq, yy, xx) for zq, yy, xx in gbt.get(tt, [])
                   if y0 <= yy < y1 and x0 <= xx < x1]
            g = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vis],
                           float).reshape(-1, 2)
            gd = np.asarray([zq - zmid2 for zq, _y, _x in vis], float)
            vm = [(zq, yy, xx) for zq, yy, xx, _a, _b in miss_bt.get(tt, [])
                  if y0 <= yy < y1 and x0 <= xx < x1]
            mk = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vm],
                            float).reshape(-1, 2)
            md = np.asarray([zq - zmid2 for zq, _y, _x in vm], float)
            _frame(ax, m, clo, chi, g, gd, mk, md, roi,
                   f"{name} zoom{k}  t={tt}  slab z {zz0}-{zz1} of {Z}"
                   + (f"   <- the miss (h={hg:.3f}, win={hw:.3f})"
                      if tt == t else "")
                   + "\\nsolid = on this slab, dashed+faint = that many z away")
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
        vis = [(zc2, yy, xx) for zc2, yy, xx in gbt.get(t, [])
               if y0 <= yy < y1 and x0 <= xx < x1]
        gxy = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vis],
                         float).reshape(-1, 2)
        gz = np.asarray([zc2 for zc2, _y, _x in vis], float)
        mxy = np.asarray([[y_ - y0, x_ - x0]], float)
        for zq in range(Z):
            m = np.asarray(img[t, zq, y0:y1, x0:x1])
            _frame(ax, m, clo, chi, gxy, gz - zq, mxy,
                   np.asarray([z_ - zq], float), roi,
                   f"{name} zsweep{k}  t={t}  z={zq}/{Z - 1}   "
                   f"the missed cell is at z={z_:.1f}"
                   + ("   <- HERE" if abs(z_ - zq) <= 1.5 else
                      f" ({z_ - zq:+.0f})")
                   + f"\\n{len(gxy)} annotated cells here; solid = this "
                     f"plane, faint dashed = that many z off")
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
