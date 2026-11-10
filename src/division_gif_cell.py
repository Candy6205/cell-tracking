# ---- one-off cell: look at a real division. Changes no existing file. ----
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, imageio.v2 as imageio
from viz_cells import find_root, geff_path, zarr_path, open_image, read_geff, read_scale

CANDIDATES = ["6bba_afb141ff", "6bba_2312ac41", "6bba_aeee7805", "6bba_268e1230"]
PAD_YX   = 40     # voxels each side  (~16 um)
SLAB     = 5      # z planes          (~8 um, about one nucleus)
BEFORE, AFTER = 4, 10
PICK     = 0      # which division to render, if a dataset has several

r = find_root(None)

def divisions(gt):
    row = {int(v): i for i, v in enumerate(gt["node_ids"])}
    kids = {}
    for a, b in gt["edges"]:
        ra, rb = row.get(int(a)), row.get(int(b))
        if ra is not None and rb is not None:
            kids.setdefault(ra, []).append(rb)
    return {p: c for p, c in kids.items() if len(c) >= 2}

name = None
for cand in CANDIDATES:
    gp = geff_path(r, "train", cand)
    if not gp.exists():
        continue
    gt = read_geff(gp)
    d = divisions(gt)
    print(f"{cand}: {len(d)} divisions")
    if d and name is None:
        name, GT, DIV = cand, gt, d
if name is None:
    raise SystemExit("no divisions found in the candidates")

parents = sorted(DIV, key=lambda p: GT["t"][p])
p = parents[min(PICK, len(parents) - 1)]
c1, c2 = DIV[p][:2]
t0 = int(GT["t"][p])
zc, yc, xc = int(GT["z"][p]), int(GT["y"][p]), int(GT["x"][p])
print(f"\n{name}: rendering the division at t={t0}, z={zc}, y={yc}, x={xc}")

zp  = zarr_path(r, "train", name)
img = open_image(zp)
sc  = np.asarray(read_scale(zp))
T, Z, Y, X = img.shape[-4:]
y0, y1 = max(0, yc - PAD_YX), min(Y, yc + PAD_YX)
x0, x1 = max(0, xc - PAD_YX), min(X, xc + PAD_YX)
z0 = max(0, min(Z - SLAB, zc - SLAB // 2)); z1 = z0 + SLAB

# how fast do the daughters actually separate?
def follow(node, steps):
    row = {int(v): i for i, v in enumerate(GT["node_ids"])}
    nxt = {}
    for a, b in GT["edges"]:
        ra, rb = row.get(int(a)), row.get(int(b))
        if ra is not None and rb is not None:
            nxt.setdefault(ra, []).append(rb)
    path, cur = [node], node
    for _ in range(steps):
        ch = nxt.get(cur, [])
        if len(ch) != 1:
            break
        cur = ch[0]; path.append(cur)
    return path

pa, pb = follow(c1, AFTER), follow(c2, AFTER)
print("\nsister separation after the split:")
for k in range(min(len(pa), len(pb))):
    a, b = pa[k], pb[k]
    d_um = float(np.linalg.norm(
        (np.array([GT["z"][a], GT["y"][a], GT["x"][a]], float)
         - np.array([GT["z"][b], GT["y"][b], GT["x"][b]], float)) * sc))
    print(f"  t={t0+1+k:3d}  {d_um:5.2f} um"
          + ("   <- below the 5 um NMS kernel, one peak" if d_um < 5 else ""))

probe = np.asarray(img[t0, z0:z1, y0:y1, x0:x1]).max(axis=0)
vmin, vmax = np.percentile(probe, [1, 99.7])
lo, hi = max(0, t0 - BEFORE), min(T, t0 + AFTER + 1)
special = {p: ("yellow", "parent"), c1: ("cyan", "d1"), c2: ("magenta", "d2")}

frames = []
for t in range(lo, hi):
    mip = np.asarray(img[t, z0:z1, y0:y1, x0:x1]).max(axis=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=140)
    ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    m = GT["t"].astype(int) == t
    idx = np.nonzero(m)[0]
    for i in idx:
        zz, yy, xx = GT["z"][i], GT["y"][i], GT["x"][i]
        if not (z0 <= zz < z1 and y0 <= yy < y1 and x0 <= xx < x1):
            continue
        col = "red"
        for chain, (c, _lab) in ((pa, special[c1]), (pb, special[c2])):
            if i in chain:
                col = c
        if i == p:
            col = special[p][0]
        ax.scatter(xx - x0, yy - y0, s=140, facecolors="none",
                   edgecolors=col, linewidths=1.4)
    ax.set_title(f"{name}  t={t}" + ("   <-- GT division" if t == t0 else ""),
                 fontsize=9)
    ax.axis("off"); fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

out = f"division_{name}_t{t0}.gif"
imageio.mimsave(out, frames, fps=2, loop=0)
print(f"\n[saved] {out}   yellow = parent, cyan/magenta = the two daughters, "
      f"red = other annotated cells")
