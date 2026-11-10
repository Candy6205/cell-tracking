"""make_swapvid.py -> swapvid.py: bake the raw image around swap sites into GIFs and contact sheets for the eye.

WHY (2026-09-23). swapz left the swap-site nucleus as a mixture: 44% of the sites have the z maxima through
a and through b on the same or adjacent plane (one object?), 38% three or more planes apart (two stacked?).
No pipeline-side measure separates "one big nucleus with two shoulder peaks" from "two nuclei". A human eye
on the raw frames can. This picks the sites a human can judge in xy -- z-maximum difference <= 1 plane,
a-b distance >= 8 um -- at most 2 per movie, mixes in baseline sites (a true-positive target A with its
detection a and the nearest OTHER primary b, same geometry filter) as known two-cell references, shuffles
them, and renders each as 7 frames (t-3 .. t+3) of a 32 um field, max-projected over the annotated plane
+-1. The sheets carry an index only; the answer key (kind) is in swapvid_key.csv.
Marks on the site frame: yellow cross = annotated centroid A; green circle = the detection matched to A (a);
red circle = the other detection (b: the linker's false target at a swap site, the nearest other primary at
a baseline site). Small yellow dots on every frame = every annotated centroid within 2 planes of the field.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py). CPU, no internet.
"""
import base64
import gzip
import io
import re

import numpy as np
import pandas as pd

N_SWAP, N_BASE, PER_MOVIE_SWAP, SEED = 24, 8, 2, 0

ln = open("swapimg.py", encoding="utf-8").read().splitlines()[69]
blob = re.search(r'"([^"]+)"', ln).group(1)
emb = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(blob)).decode()))
meas = pd.read_csv("bench_csv/swapz/swapz_sites.csv")
for d in (emb, meas):
    for c in ("gz", "gy", "gx"):
        d[c] = d[c].round(1)
key = ["kind", "dataset", "t", "gz", "gy", "gx"]
meas = meas.drop_duplicates(key)
j = emb.merge(meas[key + ["d_ab", "dz_ab", "n_max6", "dip"]], on=key, how="left")
assert len(j) == len(emb), (len(j), len(emb))
# the swapsrc row that follows each swap row is its source (frame t-1): GT parent + source detection
src = emb.shift(-1)
j["pz"], j["py"], j["px"] = src["gz"], src["gy"], src["gx"]
j["sz"], j["sy"], j["sx"] = src["az"], src["ay"], src["ax"]

rng = np.random.default_rng(SEED)
sw = j[(j.kind == "swap") & (j.dz_ab <= 1) & (j.d_ab >= 8)].copy()
sw = sw.sample(frac=1.0, random_state=SEED)
sw = sw.groupby("dataset").head(PER_MOVIE_SWAP).head(N_SWAP)
ba = j[(j.kind == "base") & (j.dz_ab <= 1) & (j.d_ab >= 8) & (j.d_ab <= 12)].copy()
ba = ba.sample(frac=1.0, random_state=SEED).groupby("dataset").head(1).head(N_BASE)
sel = pd.concat([sw, ba]).sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
sel.insert(0, "idx", np.arange(1, len(sel) + 1))
cols = ["idx", "kind", "dataset", "t", "gz", "gy", "gx", "az", "ay", "ax", "bz", "by", "bx",
        "pz", "py", "px", "sz", "sy", "sx", "d_ab", "dz_ab", "n_max6", "dip"]
sel.loc[sel.kind != "swap", ["pz", "py", "px", "sz", "sy", "sx"]] = np.nan
sites_csv = sel[cols].to_csv(index=False, float_format="%.3f", lineterminator="\n")
print(f"selected {len(sw)} swap + {len(ba)} base sites from {sw.dataset.nunique()} + {ba.dataset.nunique()} movies")
print(sel[["idx", "kind", "dataset", "t", "d_ab", "dz_ab"]].to_string(index=False))

TEMPLATE = r'''"""swapvid: the raw image around swap sites, baked into GIFs and contact sheets for a human eye.

Sites = swap sites (frame t: GT child A, detection a matched to A, the linker's false target b) with the z
maxima through a and b <= 1 plane apart and a-b >= 8 um, at most 2 per movie, mixed with baseline sites (a
true-positive target A, its detection a, the nearest other primary b; same geometry) and shuffled. Each site
renders frames t-3 .. t+3 of a 32 um field around A, max-projected over A's plane +-1, with every annotated
centroid within 2 planes as a small yellow dot; on the site frame A = yellow cross, a = green circle, b = red
circle. Outputs: swapvid_NN.png (contact sheet), swapvid_NN.gif (loop), swapvid_key.csv (idx -> kind and
the site). Attach the competition data + faustine649/cell-tracking (viz_cells.py). CPU, no internet.
"""
import io
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRIPT_VERSION = "2026-09-23a-swapvid"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, open_image, read_geff, zarr_path   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SCALE = np.array([1.625, 0.40625, 0.40625])
HY = 40                  # half field in voxels (16.25 um)
HZ_MIP = 1               # planes above/below A's plane in the projection
HZ_GT = 2                # annotated centroids within this many planes are drawn
FRAMES = list(range(-3, 4))
S = 5                    # upscale factor
MS = 600                 # ms per GIF frame

SITES_CSV = """__SITES__"""


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_SMALL = font(18), font(14)


def render_frame(crop2d, lo, hi):
    g = np.clip((crop2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (255 * g ** 0.8).astype(np.uint8)
    im = Image.fromarray(g).resize((g.shape[1] * S, g.shape[0] * S), Image.NEAREST).convert("RGB")
    return im


def to_px(p, y0, x0):
    """volume (z, y, x) -> image (x_px, y_px) of the pixel centre"""
    return ((float(p[2]) - x0 + 0.5) * S, (float(p[1]) - y0 + 0.5) * S)


def cross(d, c, r, col, w=2):
    x, y = c
    d.line([(x - r, y), (x + r, y)], fill=col, width=w)
    d.line([(x, y - r), (x, y + r)], fill=col, width=w)


def circle(d, c, r, col, w=2):
    x, y = c
    d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=w)


if __name__ == "__main__":
    t_all = time.time()
    sites = pd.read_csv(io.StringIO(SITES_CSV))
    print(f"{len(sites)} sites over {sites.dataset.nunique()} movies", flush=True)
    keyrows = []
    for name, grp in sites.groupby("dataset", sort=True):
        t0 = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        T, Z, Y, X = img.shape
        gt = read_geff(geff_path(ROOT, "train", name))
        gt_t = np.asarray(gt["t"], float).round().astype(int)
        gt_zyx = np.stack([np.asarray(gt["z"], float), np.asarray(gt["y"], float), np.asarray(gt["x"], float)], 1)
        for _, r in grp.iterrows():
            t = int(r.t)
            A = np.array([r.gz, r.gy, r.gx], float)
            zc = int(round(A[0]))
            z0, z1 = max(0, zc - HZ_MIP), min(Z, zc + HZ_MIP + 1)
            y0, y1 = max(0, int(round(A[1])) - HY), min(Y, int(round(A[1])) + HY + 1)
            x0, x1 = max(0, int(round(A[2])) - HY), min(X, int(round(A[2])) + HY + 1)
            frames = [f for f in (t + o for o in FRAMES) if 0 <= f < T]
            crops = {f: np.asarray(img[f, z0:z1, y0:y1, x0:x1], dtype=np.float32).max(axis=0) for f in frames}
            allv = np.concatenate([c.ravel() for c in crops.values()])
            lo, hi = np.percentile(allv, 1.0), np.percentile(allv, 99.8)
            ims = []
            for f in frames:
                im = render_frame(crops[f], lo, hi)
                d = ImageDraw.Draw(im)
                m = (gt_t == f) & (np.abs(gt_zyx[:, 0] - A[0]) <= HZ_GT) \
                    & (gt_zyx[:, 1] >= y0) & (gt_zyx[:, 1] < y1) & (gt_zyx[:, 2] >= x0) & (gt_zyx[:, 2] < x1)
                for p in gt_zyx[m]:
                    x, y = to_px(p, y0, x0)
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 0))
                if f == t:
                    cross(d, to_px(A, y0, x0), 12, (255, 220, 0), 3)
                    circle(d, to_px((r.az, r.ay, r.ax), y0, x0), 10, (0, 255, 80), 3)
                    if np.isfinite(r.bz):
                        circle(d, to_px((r.bz, r.by, r.bx), y0, x0), 10, (255, 60, 60), 3)
                lab = f"t{f - t:+d}" if f != t else "t (site)"
                d.rectangle([0, 0, 90, 22], fill=(0, 0, 0))
                d.text((4, 3), lab, fill=(255, 255, 255), font=F_SMALL)
                # scale bar 10 um
                px10 = 10.0 / SCALE[2] * S
                d.line([(im.width - px10 - 8, im.height - 10), (im.width - 8, im.height - 10)], fill=(255, 255, 255), width=3)
                ims.append(im)
            w, h = ims[0].size
            hdr = 30
            sheet = Image.new("RGB", (w * len(ims) + 4 * (len(ims) - 1), h + hdr), (20, 20, 20))
            d = ImageDraw.Draw(sheet)
            d.text((6, 6), f"#{int(r.idx):02d}   {name}  frame {t}   a-b {r.d_ab:.1f} um, z-max diff {r.dz_ab:.0f} planes   "
                           f"field 32 um, planes {z0}-{z1 - 1} max-projected   yellow x = annotated centroid, "
                           f"green o = detection matched to it, red o = the other detection, dots = all annotations",
                   fill=(255, 255, 255), font=F_BIG)
            for k, im in enumerate(ims):
                sheet.paste(im, (k * (w + 4), hdr))
            tag = f"swapvid_{int(r.idx):02d}"
            sheet.save(OUT / f"{tag}.png")
            ims[0].save(OUT / f"{tag}.gif", save_all=True, append_images=ims[1:], duration=MS, loop=0)
            keyrows.append(dict(idx=int(r.idx), kind=r.kind, dataset=name, t=t, gz=r.gz, gy=r.gy, gx=r.gx,
                                d_ab=r.d_ab, dz_ab=r.dz_ab, n_max6=r.n_max6, dip=r.dip,
                                planes=f"{z0}-{z1 - 1}", png=f"{tag}.png", gif=f"{tag}.gif"))
        print(f"  {name}: {len(grp)} sites, shape {tuple(img.shape)}, {time.time() - t0:.0f}s", flush=True)
        del img
    pd.DataFrame(keyrows).sort_values("idx").to_csv(OUT / "swapvid_key.csv", index=False)
    print(f"\nSWAPVID DONE in {(time.time() - t_all) / 60:.1f} min: {len(keyrows)} sites", flush=True)
    print("SWAPVID OK", flush=True)
'''

src = TEMPLATE.replace("__SITES__", sites_csv)
open("swapvid.py", "w", encoding="utf-8", newline="\n").write(src)
print("written swapvid.py", src.count("\n"), "lines")
