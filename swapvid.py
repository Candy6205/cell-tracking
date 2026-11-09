"""swapvid: the raw image around swap sites, baked into GIFs and contact sheets for a human eye.

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

SITES_CSV = """idx,kind,dataset,t,gz,gy,gx,az,ay,ax,bz,by,bx,pz,py,px,sz,sy,sx,d_ab,dz_ab,n_max6,dip
1,base,6bba_fbc898dc,16,16.000,78.000,145.000,16.008,77.057,145.088,16.107,75.857,117.060,,,,,,,11.400,0.000,1,0.486
2,swap,6bba_07e24132,3,46.000,246.000,38.000,44.075,245.170,34.358,46.043,245.963,53.675,46.000,247.000,39.000,46.828,247.217,50.422,8.480,1.000,2,0.141
3,swap,6bba_57b7cc1e,82,18.000,92.000,139.000,15.657,91.870,141.684,20.687,91.113,136.000,17.000,91.000,139.000,18.861,84.637,139.995,8.499,0.000,1,0.020
4,swap,6bba_74686d6a,32,35.000,148.000,106.000,33.524,147.096,104.055,30.885,125.047,90.191,34.000,130.000,95.000,33.623,129.370,92.963,11.420,0.000,0,0.530
5,swap,6bba_9a41d029,23,35.000,31.000,143.000,35.796,21.614,144.570,34.885,43.741,144.292,35.000,34.000,143.000,33.940,44.035,142.159,9.111,0.000,1,0.062
6,swap,6bba_3c5691b6,80,52.000,17.000,101.000,51.859,16.158,100.750,51.047,39.970,98.676,52.000,25.000,99.000,52.055,34.965,96.718,9.799,1.000,1,0.223
7,swap,6bba_0c7fa718,56,40.000,142.000,72.000,39.726,153.181,70.763,40.030,133.375,67.752,40.000,142.000,72.000,40.030,133.375,67.752,8.154,0.000,1,0.323
8,base,6bba_cdcfe533,48,8.000,244.000,206.000,7.970,243.922,204.728,7.771,252.793,222.828,,,,,,,8.195,1.000,1,0.529
9,swap,6bba_1f58c2f6,90,58.000,124.000,147.000,57.558,125.948,143.148,50.875,124.776,140.707,55.000,123.000,144.000,52.106,124.097,141.451,10.920,0.000,1,0.000
10,swap,6bba_a90a0b9c,16,8.000,23.000,209.000,6.863,28.091,201.075,9.571,17.984,215.193,9.000,23.000,209.000,10.517,17.804,215.975,8.314,0.000,1,0.049
11,base,6bba_fc83837d,46,57.000,84.000,209.000,56.768,84.289,208.949,54.981,93.313,230.522,,,,,,,9.934,0.000,1,0.186
12,swap,6bba_55b7eebe,97,52.000,152.000,117.000,52.154,141.550,115.054,53.334,160.296,122.684,52.000,153.000,117.000,53.234,160.080,122.677,8.443,0.000,0,0.000
13,swap,6bba_1ebfb80d,16,6.000,233.000,26.000,4.452,239.252,18.640,5.728,217.675,25.847,4.000,236.000,20.000,3.611,224.697,18.946,9.471,0.000,0,0.046
14,base,6bba_5b28472a,43,28.000,242.000,207.000,27.710,243.129,206.092,34.803,242.993,202.935,,,,,,,11.600,0.000,1,0.000
15,swap,6bba_eebc57a5,50,61.000,40.000,95.000,61.371,29.872,96.865,61.225,52.756,89.715,62.000,42.000,93.000,61.328,49.785,88.803,9.743,0.000,1,0.000
16,swap,6bba_312f0dc3,25,52.000,199.000,139.000,51.710,193.245,131.752,50.895,205.836,154.035,53.000,205.000,142.000,51.972,208.572,149.009,10.480,1.000,2,0.024
17,base,6bba_57b7cc1e,90,6.000,131.000,227.000,5.159,129.964,226.160,5.135,126.928,246.018,,,,,,,8.161,1.000,1,0.408
18,swap,6bba_bb9f20c3,53,48.000,53.000,33.000,47.823,43.653,32.120,48.122,63.614,29.081,48.000,55.000,32.000,47.909,65.941,28.960,8.217,1.000,1,0.103
19,swap,6bba_eebc57a5,59,55.000,191.000,67.000,54.697,182.006,69.825,54.959,203.901,61.820,55.000,189.000,67.000,55.051,196.963,63.849,9.480,1.000,2,0.112
20,swap,6bba_c27cba08,32,3.000,158.000,45.000,2.441,153.974,42.129,1.146,170.286,56.676,3.000,157.000,43.000,1.462,164.669,49.902,9.125,0.000,1,0.000
21,base,6bba_6ca87370,68,31.000,190.000,123.000,29.710,191.508,122.605,24.210,204.589,133.630,,,,,,,11.320,1.000,0,0.000
22,swap,6bba_78a7bd97,90,60.000,23.000,172.000,61.147,28.094,162.207,61.274,15.917,180.016,60.000,23.000,170.000,59.984,16.054,179.919,8.767,0.000,1,0.075
23,swap,6bba_fc83837d,90,27.000,77.000,95.000,27.744,73.404,105.346,25.675,79.773,88.292,28.000,76.000,98.000,25.638,78.542,92.320,8.124,0.000,2,0.003
24,swap,6bba_1d0d8384,30,49.000,140.000,34.000,47.782,137.902,28.835,47.892,140.052,49.934,48.000,137.000,38.000,47.900,136.976,49.717,8.618,1.000,1,0.598
25,swap,6bba_825bd1c6,54,28.000,66.000,49.000,28.571,55.331,49.298,26.913,77.423,49.049,27.000,69.000,49.000,25.944,76.150,47.937,9.371,1.000,2,0.000
26,base,6bba_786893ac,46,50.000,33.000,64.000,49.812,33.472,63.105,48.932,31.402,83.083,,,,,,,8.284,1.000,1,0.353
27,base,6bba_f1fde7e0,63,15.000,133.000,76.000,15.327,132.990,76.299,10.674,134.802,89.527,,,,,,,9.305,0.000,0,0.000
28,swap,6bba_ebff6e76,29,12.000,149.000,243.000,12.113,141.531,238.156,13.210,158.716,249.162,12.000,149.000,242.000,12.210,160.023,247.348,8.480,0.000,0,0.424
29,swap,6bba_907271db,77,29.000,110.000,154.000,27.764,118.349,156.159,27.126,96.048,150.730,29.000,107.000,150.000,27.099,100.070,149.013,9.382,1.000,1,0.366
30,swap,6bba_57b7cc1e,75,15.000,77.000,143.000,16.972,75.312,142.100,11.225,66.357,135.218,15.000,76.000,143.000,13.138,72.319,135.430,10.410,1.000,0,0.000
31,swap,6bba_ebff6e76,30,13.000,150.000,243.000,13.256,157.719,249.393,13.197,140.616,239.043,12.000,149.000,243.000,12.113,141.531,238.156,8.122,0.000,0,0.298
32,swap,6bba_7af54fde,78,60.000,134.000,23.000,61.465,141.650,23.409,55.357,148.279,27.155,59.000,135.000,22.000,56.291,140.192,23.156,10.400,0.000,1,0.000
"""


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
