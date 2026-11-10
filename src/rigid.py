"""Is the whole field moving? The user watched 6bba_aeee7805 and said it
looks like the viewpoint slowly rotating, "like the Earth spinning".

That matters far more than the shape taxonomy it interrupted, because
the linker is calibrated on DISPLACEMENT: reference.md records pooled
p50 1.82 um / p99 6.71 um, and GATE_UM, MAX_LINK_UM and TIGHT_UM are all
set from numbers of that kind. If part of that displacement is the
sample or the stage moving as a whole, then those gates are sized for a
motion the cells are not making, and the per-cell motion is smaller than
believed.

Method, per consecutive frame pair, on the detections the pipeline
actually keeps:

1. match points between t and t+1 by nearest neighbour, keeping only
   mutual nearest pairs under 12 um -- crude, but a global transform
   does not need a perfect matching, and the alternative (the pipeline's
   own linker) would import the very gate being questioned;
2. fit the best RIGID transform to those pairs (Kabsch: centre both
   clouds, SVD of the cross-covariance, reflection-corrected);
3. report translation magnitude, rotation angle, and -- the number that
   decides it -- the residual left AFTER removing the rigid part.

Reading it:

  bulk >> residual   the field moves as one body. The gates are sized
                     for that, not for cell motion, and per-cell motion
                     is much smaller than the pooled statistics say.
  bulk << residual   cells move independently and the pooled numbers
                     mean what they have been taken to mean.

Cumulative rotation is printed too: a slow steady turn shows as a total
angle far larger than any single step, which is exactly what "rotating
like the Earth" would look like.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 12.0
CACHE = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")

SETS = ["6bba_aeee7805", "6bba_afb141ff", "6bba_268e1230",
        "6bba_2312ac41", "6bba_5c824876", "6bba_3abfe10a"]


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


def kabsch(A, B):
    """Rigid transform taking A onto B. Returns (R, t)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


print(f"  {'dataset':<18} {'pairs':>7} {'bulk um':>9} {'rot deg':>9} "
      f"{'resid um':>9} {'raw um':>8} {'cum rot':>9}  reading")
for name in SETS:
    f = CACHE / f"detcache_{name}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    T = max(int(k[1:]) for k in frame_keys(z.files, "c")) + 1

    def pts(t):
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        return c[s >= UNET_THRESH] * SCALE[None, :]

    bulk, rot, resid, raw, npair = [], [], [], [], []
    R_cum = np.eye(3)
    for t in range(T - 1):
        A, B = pts(t), pts(t + 1)
        if len(A) < 8 or len(B) < 8:
            continue
        ta, tb = cKDTree(A), cKDTree(B)
        da, ia = tb.query(A)
        db, ib = ta.query(B)
        pa = [i for i in range(len(A))
              if da[i] <= MATCH_UM and ib[ia[i]] == i]
        if len(pa) < 8:
            continue
        A2 = A[pa]
        B2 = B[[ia[i] for i in pa]]
        raw.append(float(np.linalg.norm(B2 - A2, axis=1).mean()))
        R, tv = kabsch(A2, B2)
        pred = (R @ A2.T).T + tv
        resid.append(float(np.linalg.norm(B2 - pred, axis=1).mean()))
        bulk.append(float(np.linalg.norm(B2 - A2, axis=1).mean()
                          - np.linalg.norm(B2 - pred, axis=1).mean()))
        ang = np.degrees(np.arccos(
            np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        rot.append(float(ang))
        R_cum = R @ R_cum
        npair.append(len(pa))
    if not raw:
        print(f"  {name:<18}  too few matched pairs")
        continue
    cum = np.degrees(np.arccos(
        np.clip((np.trace(R_cum) - 1.0) / 2.0, -1.0, 1.0)))
    mb, mr = float(np.mean(bulk)), float(np.mean(resid))
    tag = ("BULK DOMINATES" if mb > mr else
           "independent motion dominates" if mr > 3 * mb else "mixed")
    print(f"  {name:<18} {int(np.mean(npair)):>7} {mb:>9.3f} "
          f"{float(np.mean(rot)):>9.4f} {mr:>9.3f} "
          f"{float(np.mean(raw)):>8.3f} {cum:>9.2f}  {tag}")

print("\n  bulk um   = how much of the mean step the rigid fit explains")
print("  resid um  = what is left after removing it: the per-cell motion")
print("  raw um    = the mean step before any correction, i.e. what the")
print("              pooled displacement statistics are made of")
print("  cum rot   = rotation accumulated over the whole movie. A slow")
print("              steady turn shows here and nowhere else.")
print("\n  ⚠ Nearest-neighbour matching is crude and BREAKS DOWN as")
print("  density rises: in a crowded field the true partner is often not")
print("  the nearest point, which inflates both raw and resid. Compare")
print("  bulk against resid WITHIN a row; do not compare rows.")
