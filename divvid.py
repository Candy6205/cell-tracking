"""divvid: the raw image around the 22 'timing' divisions (divtime: the second daughter's predicted track exists
two or more frames before the annotated division) plus 4 on-time controls, baked into GIFs and contact sheets
for a human eye.

Question for the eye: before the annotated division frame, is there really a second nucleus next to the mother
(an early, real split the annotator recorded later), one elongated nucleus that the detector split into two
points, or a neighbouring cell whose track the linker confused with the daughter?

Each case renders frames t0-8 .. t0+3 (t0 = the mother's last annotated frame; t0+1 = the annotated daughters)
of a 32 um field centred on the mother at t0, max-projected over the planes spanned by the mother and both
daughters +-1. Marks: yellow cross = the mother's annotated chain (8 frames back); cyan / magenta crosses = the
two daughters' annotated chains (3 frames on); small yellow dots = every other annotated centroid within 2
planes; green circles = every F4HP detection (score >= 0.15) within the projected planes +-1, i.e. what the
pipeline had to link; RED circle = the detection the matcher took as the SECOND daughter (the farther one) at t0+1 and,
on earlier frames, the nearest detection within 5 um per frame walking back from it (what a linker could have chained);
ORANGE circle = the same for the first daughter. Outputs: divvid_NN.gif (loop), divvid_NN.png (sheet), divvid_key.csv.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py) + biohub-hparacache (fdet_F4hp_).
CPU, no internet.
"""
import io
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRIPT_VERSION = "2026-09-24c-divvid"
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
INPUT_ROOT = "/kaggle/input"
SCALE = np.array([1.625, 0.40625, 0.40625])
HY = 40                  # half field in voxels (16.25 um)
HZ_PAD = 1               # planes beyond the lineage's z span in the projection
HZ_GT = 2                # annotated centroids within this many planes of the projection are drawn
HZ_DET = 1               # detections within this many planes of the projection are drawn
BACK, FWD = 8, 3         # frames before t0 and after t0
S = 6                    # upscale factor (bicubic: the raw voxel is 0.41 um, a nucleus ~20 voxels wide)
MS = 600                 # ms per GIF frame
SCORE_MIN = 0.15
STEP_UM = 5.0            # back-walk step on detections
MAX_BACK = 60

CASES_CSV = """idx,kind,movie,t_div,d2_start_off,sep_at_div,sep_min_before,d2_pred_is_mother_track
1,timing,6bba_05db0fb1,62,-6.00,7.84,7.64,0
2,timing,6bba_09961292,48,-49.00,10.56,7.37,0
3,timing,6bba_0e7c0d07,57,-13.00,12.91,13.24,1
4,timing,6bba_12665c0e,39,-15.00,12.19,10.77,1
5,timing,6bba_1d0d8384,14,-12.00,8.45,8.62,0
6,timing,6bba_32db13fc,5,-6.00,14.01,9.71,0
7,ontime,6bba_48816121,16,0.00,23.56,,0
8,timing,6bba_57b7cc1e,12,-4.00,9.80,5.31,1
9,timing,6bba_57b7cc1e,23,-11.00,14.66,14.45,1
10,timing,6bba_57b7cc1e,77,-26.00,15.23,7.05,1
11,timing,6bba_6321a359,8,-9.00,11.94,12.36,0
12,timing,6bba_705ec2c9,42,-3.00,11.39,5.18,1
13,timing,6bba_74686d6a,24,-16.00,15.10,12.78,1
14,timing,6bba_786893ac,63,-2.00,11.01,13.28,0
15,timing,6bba_78a7bd97,56,-2.00,13.25,8.85,0
16,timing,6bba_7af54fde,52,-2.00,13.12,10.97,0
17,timing,6bba_7af54fde,55,-5.00,12.39,7.61,0
18,timing,6bba_969618f6,71,-2.00,14.92,15.06,0
19,ontime,6bba_9e23430b,89,0.00,8.11,,0
20,timing,6bba_ab78413d,67,-2.00,13.47,12.98,0
21,timing,6bba_afb141ff,48,-6.00,10.48,12.55,0
22,ontime,6bba_bb9f20c3,31,0.00,9.37,,0
23,ontime,6bba_cdcfe533,52,0.00,7.20,,0
24,timing,6bba_debd7bfa,12,-4.00,11.44,11.16,0
25,timing,6bba_ebdf3b34,13,-4.00,9.97,8.98,1
26,timing,6bba_fc5f39dc,54,-55.00,16.98,9.93,0
"""


def _walk_files(match):
    hits = {}
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: fn[len(prefix):-len(".npz")] if fn.startswith(prefix) and fn.endswith(".npz") else None


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_SMALL = font(18), font(14)


def render_frame(crop2d, lo, hi):
    g = np.clip((crop2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (255 * g ** 0.8).astype(np.uint8)
    return Image.fromarray(g).resize((g.shape[1] * S, g.shape[0] * S), Image.BICUBIC).convert("RGB")


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


def nearest_det(det, t, p_vox):
    """nearest detection (score >= SCORE_MIN) to a voxel position at frame t: (voxel coords, um distance)"""
    C, Sc = det[f"c{t}"], det[f"s{t}"]
    C = C[Sc >= SCORE_MIN]
    if not len(C):
        return None, np.inf
    d = np.linalg.norm((C - p_vox) * SCALE, axis=1)
    j = int(np.argmin(d))
    return C[j], float(d[j])


def backtrack(det, t_start, p_vox):
    """frame -> voxel position, walking back from p_vox through the nearest detection within STEP_UM per frame"""
    path = {t_start: p_vox}
    p = p_vox
    for t in range(t_start - 1, max(-1, t_start - MAX_BACK - 1), -1):
        q, d = nearest_det(det, t, p)
        if q is None or d > STEP_UM:
            break
        p = q
        path[t] = p
    return path


def lineage(gt, t0):
    """The GT division whose mother sits at frame t0: (mother chain rows back BACK frames incl. the mother,
    daughter-1 chain rows, daughter-2 chain rows, each FWD frames on incl. the daughter)."""
    g_t = np.asarray(gt["t"], float).round().astype(int)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gs, gp = defaultdict(list), {}
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            gs[ra].append(rb)
            gp[rb] = ra
    divs = [m for m, ss in gs.items() if len(ss) >= 2 and g_t[m] == t0]
    assert len(divs) == 1, (t0, len(divs))
    m = divs[0]
    chain_m = [m]
    cur = m
    for _ in range(BACK):
        cur = gp.get(cur)
        if cur is None:
            break
        chain_m.append(cur)
    chains_d = []
    for d in gs[m][:2]:
        ch, cur = [d], d
        for _ in range(FWD - 1):
            nxt = gs.get(cur, [])
            if len(nxt) != 1:
                break
            cur = nxt[0]
            ch.append(cur)
        chains_d.append(ch)
    return chain_m, chains_d[0], chains_d[1]


if __name__ == "__main__":
    t_all = time.time()
    FDET = _walk_files(_npz_prefix("fdet_F4hp_"))
    cases = pd.read_csv(io.StringIO(CASES_CSV))
    missing = sorted(set(cases.movie) - set(FDET))
    assert not missing, f"attach biohub-hparacache: no fdet_F4hp_ cache for {missing}"
    print(f"{len(cases)} cases over {cases.movie.nunique()} movies; F4hp caches {len(FDET)}", flush=True)
    keyrows = []
    for name, grp in cases.groupby("movie", sort=True):
        t_m = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        T, Z, Y, X = img.shape
        gt = read_geff(geff_path(ROOT, "train", name))
        gt_t = np.asarray(gt["t"], float).round().astype(int)
        gt_zyx = np.stack([np.asarray(gt["z"], float), np.asarray(gt["y"], float), np.asarray(gt["x"], float)], 1)
        det = np.load(FDET[name])
        for _, r in grp.iterrows():
            t0 = int(r.t_div)
            cm, c1, c2 = lineage(gt, t0)
            M = gt_zyx[cm[0]]
            zs = [gt_zyx[i][0] for i in (cm[0], c1[0], c2[0])]
            z0, z1 = max(0, int(round(min(zs))) - HZ_PAD), min(Z, int(round(max(zs))) + HZ_PAD + 1)
            y0, y1 = max(0, int(round(M[1])) - HY), min(Y, int(round(M[1])) + HY + 1)
            x0, x1 = max(0, int(round(M[2])) - HY), min(X, int(round(M[2])) + HY + 1)
            frames = [f for f in range(t0 - BACK, t0 + FWD + 1) if 0 <= f < T]
            crops = {f: np.asarray(img[f, z0:z1, y0:y1, x0:x1], dtype=np.float32).max(axis=0) for f in frames}
            allv = np.concatenate([c.ravel() for c in crops.values()])
            lo, hi = np.percentile(allv, 1.0), np.percentile(allv, 99.8)
            in_field = lambda p: (y0 <= p[1] < y1) and (x0 <= p[2] < x1)        # noqa: E731
            # the matcher's view: nearest detection to each GT daughter at t0+1, then the back-walk on detections
            D1, D2 = gt_zyx[c1[0]], gt_zyx[c2[0]]
            if np.linalg.norm((D2 - M) * SCALE) < np.linalg.norm((D1 - M) * SCALE):
                D1, D2 = D2, D1
            walks = {}
            for lab, D in (("d1", D1), ("d2", D2)):
                q, dq = nearest_det(det, t0 + 1, D)
                walks[lab] = backtrack(det, t0 + 1, q) if q is not None else {}
            marks = {}
            for rows, col in ((cm, (255, 220, 0)), (c1, (0, 230, 255)), (c2, (255, 80, 255))):
                for i in rows:
                    marks.setdefault(int(gt_t[i]), []).append((gt_zyx[i], col))
            n_det = {}
            ims = []
            for f in frames:
                im = render_frame(crops[f], lo, hi)
                d = ImageDraw.Draw(im)
                m_gt = (gt_t == f) & (gt_zyx[:, 0] >= z0 - HZ_GT) & (gt_zyx[:, 0] < z1 + HZ_GT) \
                    & (gt_zyx[:, 1] >= y0) & (gt_zyx[:, 1] < y1) & (gt_zyx[:, 2] >= x0) & (gt_zyx[:, 2] < x1)
                for p in gt_zyx[m_gt]:
                    x, y = to_px(p, y0, x0)
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 0))
                C, Sc = det[f"c{f}"], det[f"s{f}"]
                m_d = (Sc >= SCORE_MIN) & (C[:, 0] >= z0 - HZ_DET) & (C[:, 0] < z1 + HZ_DET) \
                    & (C[:, 1] >= y0) & (C[:, 1] < y1) & (C[:, 2] >= x0) & (C[:, 2] < x1)
                n_det[f] = int(m_d.sum())
                for p in C[m_d]:
                    circle(d, to_px(p, y0, x0), 7, (0, 255, 80), 2)
                for lab, col in (("d1", (255, 150, 0)), ("d2", (255, 40, 40))):
                    p = walks[lab].get(f)
                    if p is not None and in_field(p):
                        circle(d, to_px(p, y0, x0), 10, col, 3)
                for p, col in marks.get(f, []):
                    if in_field(p):
                        cross(d, to_px(p, y0, x0), 11, col, 3)
                lab = f"t0{f - t0:+d}" if f != t0 else "t0 (mother)"
                if f == t0 + 1:
                    lab = "t0+1 DIVISION"
                d.rectangle([0, 0, 118, 22], fill=(0, 0, 0))
                d.text((4, 3), lab, fill=(255, 255, 255), font=F_SMALL)
                px10 = 10.0 / SCALE[2] * S
                d.line([(im.width - px10 - 8, im.height - 10), (im.width - 8, im.height - 10)], fill=(255, 255, 255), width=3)
                ims.append(im)
            w, h = ims[0].size
            hdr = 30
            sheet = Image.new("RGB", (w * len(ims) + 4 * (len(ims) - 1), h + hdr), (20, 20, 20))
            d = ImageDraw.Draw(sheet)
            d.text((6, 6), f"#{int(r.idx):02d}  {name}  mother frame {t0}, daughters {t0 + 1}   planes {z0}-{z1 - 1} max-projected, field 32 um   "
                           f"yellow x = mother chain, cyan / magenta x = daughters, dots = other annotations, green o = F4HP detections, red / orange o = the detection matched as 2nd / 1st daughter and its back-walk",
                   fill=(255, 255, 255), font=F_BIG)
            for k, im in enumerate(ims):
                sheet.paste(im, (k * (w + 4), hdr))
            tag = f"divvid_{int(r.idx):02d}"
            sheet.save(OUT / f"{tag}.png")
            ims[0].save(OUT / f"{tag}.gif", save_all=True, append_images=ims[1:], duration=MS, loop=0)
            keyrows.append(dict(idx=int(r.idx), kind=r.kind, movie=name, t0=t0, d2_start_off=r.d2_start_off,
                                sep_at_div=r.sep_at_div, sep_min_before=r.sep_min_before,
                                d2_pred_is_mother_track=int(r.d2_pred_is_mother_track),
                                mz=float(M[0]), my=float(M[1]), mx=float(M[2]), planes=f"{z0}-{z1 - 1}",
                                frames=f"{frames[0]}-{frames[-1]}", n_det_t0=n_det.get(t0, -1), n_det_t0p1=n_det.get(t0 + 1, -1),
                                gif=f"{tag}.gif", png=f"{tag}.png"))
        print(f"  {name}: {len(grp)} cases, shape {tuple(img.shape)}, {time.time() - t_m:.0f}s", flush=True)
        del img
    pd.DataFrame(keyrows).sort_values("idx").to_csv(OUT / "divvid_key.csv", index=False)
    print(f"\nDIVVID DONE in {(time.time() - t_all) / 60:.1f} min: {len(keyrows)} cases", flush=True)
    print("DIVVID OK", flush=True)
