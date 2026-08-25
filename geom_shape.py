"""Are the 12 6bba holdout datasets the same KIND of thing?

The user watched the two rendered movies and said they look completely
different: 6bba_aeee7805's cells sit on a curved surface, 6bba_3abfe10a's
fill the whole field. Both are 64 x 256 x 256 volumes.

That matters for today's nms4 result. `corr(reachable before NMS, true
cells per frame) = -0.5580` was computed ACROSS those 12 datasets,
treating "true cells per frame" as the variable and everything else as
comparable. If the datasets are crops of anatomically different regions,
then the correlation may be about WHICH PART OF THE EMBRYO was cropped,
not about crowding or developmental time.

Measured here, from the detection caches already on disk:

  z spread        how much of the 64-plane depth the cells occupy
  surface fit     fit z = a*y + b*x + c and report R^2. A monolayer on a
                  gently curved surface has most of its z variance
                  explained by position; a filled volume has almost none.
  residual std    z scatter left after the plane fit, in micrometres
  occupancy       fraction of 16x16x16-voxel blocks containing a detection

No ground truth needed and no Kaggle: this is about the shape of the
detected point cloud, which is what the linker actually sees.
"""
import pathlib

import numpy as np

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CACHE = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
# true cells/frame, from nms4's run (estimated_number_of_nodes / T)
TRUE_DENS = {
    "6bba_268e1230": 51, "6bba_2312ac41": 120, "6bba_283bf9f1": 209,
    "6bba_3a1849c2": 70, "6bba_3abfe10a": 521, "6bba_5c824876": 246,
    "6bba_61dd1e0d": 101, "6bba_7af54fde": 96, "6bba_7b5d3b2c": 58,
    "6bba_afb141ff": 61, "6bba_aeee7805": 50, "6bba_c27cba08": 64,
}


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


rows = []
for name in HOLDOUT:
    f = CACHE / f"detcache_{name}.npz"
    if not f.exists():
        print(f"  {name}: no cache locally, skipped")
        continue
    z = np.load(f)
    T = max(int(k[1:]) for k in frame_keys(z.files, "c")) + 1
    pts = []
    for t in range(0, T, 5):            # every 5th frame is plenty
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        c = c[s >= UNET_THRESH]
        if len(c):
            pts.append(c)
    if not pts:
        print(f"  {name}: no detections above threshold")
        continue
    P = np.concatenate(pts)
    n_per_frame = len(P) / max(1, len(pts))

    zz = P[:, 0] * SCALE[0]
    yy = P[:, 1] * SCALE[1]
    xx = P[:, 2] * SCALE[2]

    # plane fit z = a*y + b*x + c
    A = np.stack([yy, xx, np.ones_like(yy)], axis=1)
    coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
    resid = zz - A @ coef
    ss_tot = float(((zz - zz.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / max(1e-9, ss_tot)

    # occupancy of 16-voxel blocks
    blocks = set()
    for p in P:
        blocks.add((int(p[0]) // 16, int(p[1]) // 16, int(p[2]) // 16))
    occ = len(blocks) / (4 * 16 * 16)   # 64/16 x 256/16 x 256/16

    rows.append((name, TRUE_DENS.get(name, 0), n_per_frame,
                 float(zz.std()), r2, float(resid.std()), occ))

print(f"  {'dataset':<18} {'true/fr':>8} {'det/fr':>8} {'z std':>7} "
      f"{'planeR2':>8} {'resid':>7} {'occup':>7}")
for r in sorted(rows, key=lambda r: r[1]):
    print(f"  {r[0]:<18} {r[1]:>8} {r[2]:>8.0f} {r[3]:>7.2f} "
          f"{r[4]:>8.3f} {r[5]:>7.2f} {r[6]:>7.3f}")

if len(rows) > 2:
    d = np.array([r[1] for r in rows], float)
    for lab, col in (("z std", 3), ("plane R2", 4), ("residual", 5),
                     ("occupancy", 6)):
        v = np.array([r[col] for r in rows], float)
        print(f"  corr(true density, {lab:<10}) = "
              f"{float(np.corrcoef(d, v)[0, 1]):+.4f}")

print("\n  READ IT: a high plane R^2 with a small residual means the "
      "cells lie on ONE gently curved surface -- a monolayer. A low R^2 "
      "with a large residual and high occupancy means they fill the "
      "volume. If those two shapes both appear among the 12, then the "
      "datasets are not the same KIND of crop, and 'true cells per "
      "frame' is standing in for anatomy as much as for crowding.")
