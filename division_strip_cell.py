"""
Static filmstrip of one ground-truth division. Defines functions only -- nothing
runs on exec, so you can `exec(open(...).read())` once and then call `strip(...)`
with different settings without touching the Kaggle Dataset again.

    exec(open(f"{CODE}/division_strip_cell.py").read())
    divisions("6bba_afb141ff")          # how many, and at which frames
    strip("6bba_afb141ff", pick=0)      # render one
    strip("6bba_afb141ff", pick=1, pad_yx=18, slab=1, ncols=4, panel_in=4.0)

A GIF of this is unreadable: the frames flick past faster than the eye can
compare them. Side by side you can look at one and then the next.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_cells import (find_root, geff_path, zarr_path, open_image, read_geff,
                       read_scale, list_datasets)

SCRIPT_VERSION = "2026-08-19a"


def _kids(gt):
    row = {int(v): i for i, v in enumerate(gt["node_ids"])}
    k = {}
    for a, b in gt["edges"]:
        ra, rb = row.get(int(a)), row.get(int(b))
        if ra is not None and rb is not None:
            k.setdefault(ra, []).append(rb)
    return k


def divisions(dataset, split="train", root=None, show=True):
    """Return [(index, t, z, y, x), ...] for every GT division in a dataset."""
    r = find_root(root)
    gt = read_geff(geff_path(r, split, dataset))
    kids = _kids(gt)
    ps = sorted((p for p, c in kids.items() if len(c) >= 2),
                key=lambda p: gt["t"][p])
    out = [(i, int(gt["t"][p]), int(gt["z"][p]), int(gt["y"][p]), int(gt["x"][p]))
           for i, p in enumerate(ps)]
    if show:
        print(f"{dataset}: {len(out)} divisions")
        for i, t, z, y, x in out:
            print(f"  pick={i}  t={t:3d}  z={z:3d} y={y:3d} x={x:3d}")
    return out


def find_divisions(candidates=None, split="train", root=None, limit=12):
    """Scan datasets and report which ones contain divisions."""
    r = find_root(root)
    names = candidates or list_datasets(r, split)[:limit]
    hits = []
    for n in names:
        gp = geff_path(r, split, n)
        if not gp.exists():
            continue
        k = _kids(read_geff(gp))
        c = sum(1 for v in k.values() if len(v) >= 2)
        if c:
            hits.append((n, c))
        print(f"  {n}: {c} divisions")
    return hits


def strip(dataset, pick=0, pad_yx=26, slab=3, before=3, after=4, gamma=0.7,
          ncols=4, panel_in=3.6, dpi=190, circle_r=9, split="train", root=None,
          out="division_strip.png", sep_out="division_separation.png"):
    """Render one division as a filmstrip. Every knob is a keyword argument.

    pad_yx    voxels of crop on each side  (26 ~ 10.6 um)
    slab      z planes max-projected       (1 = a single plane, sharpest)
    ncols     panels per row -- fewer columns means BIGGER panels in a notebook
    panel_in  inches per panel
    gamma     <1 lifts mid-tones; microscopy is very dark
    """
    r = find_root(root)
    gt = read_geff(geff_path(r, split, dataset))
    kids = _kids(gt)
    ps = sorted((p for p, c in kids.items() if len(c) >= 2),
                key=lambda p: gt["t"][p])
    if not ps:
        raise SystemExit(f"{dataset} has no GT divisions")
    if pick >= len(ps):
        print(f"[degraded] pick={pick} but {dataset} has only {len(ps)} "
              f"division(s) -- using the last one instead")
        pick = len(ps) - 1
    p = ps[pick]
    c1, c2 = kids[p][:2]
    t0 = int(gt["t"][p])
    zc, yc, xc = int(gt["z"][p]), int(gt["y"][p]), int(gt["x"][p])

    zp = zarr_path(r, split, dataset)
    img = open_image(zp)
    sc = np.asarray(read_scale(zp))
    T, Z, Y, X = img.shape[-4:]
    y0, y1 = max(0, yc - pad_yx), min(Y, yc + pad_yx)
    x0, x1 = max(0, xc - pad_yx), min(X, xc + pad_yx)
    z0 = max(0, min(Z - slab, zc - slab // 2)); z1 = z0 + slab
    if (y1 - y0) < 2 * pad_yx or (x1 - x0) < 2 * pad_yx:
        print(f"[degraded] the crop hit the volume edge, so it is "
              f"{y1-y0}x{x1-x0} voxels instead of {2*pad_yx}x{2*pad_yx} -- "
              f"the panels will not be square")

    def chain(node, steps):
        path, cur = [node], node
        for _ in range(steps):
            ch = kids.get(cur, [])
            if len(ch) != 1:
                break
            cur = ch[0]; path.append(cur)
        return path

    A, B = chain(c1, after + 2), chain(c2, after + 2)
    print(f"{dataset}  division {pick+1}/{len(ps)}  t={t0}  z={zc} y={yc} x={xc}")
    print(f"crop {y1-y0}x{x1-x0} voxels = {(y1-y0)*sc[1]:.0f}x{(x1-x0)*sc[2]:.0f} um"
          f" | z planes {z0}..{z1-1} ({slab*sc[0]:.1f} um)")

    print("sister separation:")
    sep_t, sep_d = [], []
    for k in range(min(len(A), len(B))):
        a, b = A[k], B[k]
        d = float(np.linalg.norm((np.array([gt["z"][a], gt["y"][a], gt["x"][a]])
                                  - np.array([gt["z"][b], gt["y"][b], gt["x"][b]])) * sc))
        sep_t.append(t0 + 1 + k); sep_d.append(d)
        za, zb = int(gt["z"][a]), int(gt["z"][b])
        vis = ("both in slab" if z0 <= za < z1 and z0 <= zb < z1 else
               f"OUT OF SLAB (z={za},{zb} vs {z0}..{z1-1})")
        print(f"  t={t0+1+k:3d}  {d:5.2f} um   {vis}")

    lo, hi = max(0, t0 - before), min(T, t0 + after + 1)
    ts = list(range(lo, hi))
    crops = [np.asarray(img[t, z0:z1, y0:y1, x0:x1]).max(axis=0).astype(float)
             for t in ts]
    vmin, vmax = np.percentile(np.stack(crops), [2, 99.8])
    disp = [np.clip((c - vmin) / (vmax - vmin + 1e-9), 0, 1) ** gamma for c in crops]

    ncols = max(1, min(int(ncols), len(ts)))
    nrows = int(np.ceil(len(ts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols,
                                                    panel_in * nrows * 1.12), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ts):]:
        ax.axis("off")
    for ax, t, im in zip(axes, ts, disp):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for i in np.nonzero(gt["t"].astype(int) == t)[0]:
            zz, yy, xx = gt["z"][i], gt["y"][i], gt["x"][i]
            if not (z0 <= zz < z1 and y0 <= yy < y1 and x0 <= xx < x1):
                continue
            col, lw = "#ff3b30", 1.0
            if i == p:
                col, lw = "#ffcc00", 2.4
            elif i in A:
                col, lw = "#00e5ff", 2.4
            elif i in B:
                col, lw = "#ff2fd0", 2.4
            ax.add_patch(plt.Circle((xx - x0, yy - y0), circle_r, fill=False,
                                    edgecolor=col, linewidth=lw))
        ax.set_title(f"t={t}" + ("   DIVISION" if t == t0 else ""), fontsize=13,
                     color=("#c79000" if t == t0 else "black"),
                     fontweight=("bold" if t == t0 else "normal"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ffcc00" if t == t0 else "#cccccc")
            s.set_linewidth(2.5 if t == t0 else 0.5)

    fig.suptitle(f"{dataset}  |  yellow = parent, cyan / magenta = daughters, "
                 f"red = other annotated cells  |  {slab} z plane(s), "
                 f"{slab*sc[0]:.1f} um", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    if len(sep_d) >= 2 and sep_out:
        f2, a2 = plt.subplots(figsize=(4.6, 2.8), dpi=dpi)
        a2.plot(sep_t, sep_d, "o-", color="#333333")
        a2.axhline(5.0, ls="--", lw=1, color="#ff3b30")
        a2.text(sep_t[0], 5.2, "5 um NMS kernel", color="#ff3b30", fontsize=8)
        a2.set_xlabel("frame"); a2.set_ylabel("sister separation (um)")
        a2.set_ylim(0, max(12, max(sep_d) * 1.15)); a2.grid(alpha=0.3)
        f2.tight_layout(); f2.savefig(sep_out); plt.close(f2)
        print(f"[saved] {sep_out}")
    return out
