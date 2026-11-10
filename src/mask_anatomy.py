"""Falsify the fringe hypothesis. Labels only, no detector, no images.

SCRIPT_VERSION 2026-08-25a-mask-anatomy

WHERE THIS COMES FROM. `biohub-div-mass` measured (Va+Vb)/Vp = 1.772 at
a true division where mass conservation predicts 1.0, and 1.984 at a
fake fork where it predicts 2.0. I read that as "the watershed measures
the TERRITORY between peaks, not the cell", which closes the line. The
user proposed the opposite reading -- the masks are inflated by a fuzzy
FRINGE, everything real is buried under it, and stripping the fringe
brings mass conservation back.

BOTH FIT EVERY NUMBER div_mass PRODUCED. Writing mask = core M + fringe
H, a division reads (M/2+H + M/2+H)/(M+H) = (M+2H)/(M+H); setting that
to the measured 1.772 gives H = 3.39 M, i.e. the cell is 22.8% of the
mask. The fake fork reads (M+H + M+H)/(M+H) = 2.0 either way, and a
continuation reads 1.0 either way. So div_mass cannot separate them, and
its conclusion was over-claimed.

The two readings disagree about what to do next, which is why this
matters: territory means the line is dead and no threshold helps; fringe
means strip the fringe and the signal returns.

THE USER'S METHOD, which is the right one here: assume the fringe
hypothesis, derive something it FORCES to be true, and check that.

  A fringe ends where the heatmap falls below MASK_FLOOR -- that is, at
  BACKGROUND. A territory boundary is a watershed line drawn halfway to
  a neighbouring peak -- cell on both sides, no background anywhere near.

  So: what does a mask's surface actually touch?

PREDICTIONS, WRITTEN DOWN BEFORE THE RUN.

  * mostly BACKGROUND  -> boundaries are set by the threshold. The
    fringe reading is right, div_mass's conclusion is wrong, and an
    intensity-weighted mass (which discounts a dim fringe automatically)
    is the retry.
  * mostly OTHER LABELS -> boundaries are set by neighbours. The
    territory reading is right, a voxel count can never be a mass here,
    and no threshold change rescues it.
  * genuinely mixed -> both mechanisms are live and the answer depends
    on local density, which the second test below resolves.

SECOND, INDEPENDENT FALSIFIER. The 41 crops span 44 to 815 cells per
frame -- an 18x range in density, from the peak counts in zh_masks' log.

  * territory: a cell's share is the available volume divided by the
    number of cells, so median mask volume should fall roughly as 1/N.
    Slope of log(volume) on log(N) near **-1**.
  * fringe: a cell is a cell and a fringe is a fringe, neither of which
    knows how crowded it is, so volume should be roughly FLAT in N.
    Slope near **0**.

The two tests share no arithmetic, so agreeing is evidence and
disagreeing is a real finding about density dependence.

SCALE WARNING, because it governs how much any of this can be trusted:
the median mask is **6.62 voxels** across. The fringe hypothesis needs a
shell only ~1.3 voxels thick, because volume goes as the cube. Every
quantity here is quantised very coarsely, and a one-voxel effect is a
large effect.
"""

import pathlib
import time
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-08-25a-mask-anatomy"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

POOLED = np.array([1.625, 1.625, 1.625])
VOXEL_UM3 = float(np.prod(POOLED))
FRAME_STRIDE = 10               # 10 frames per crop is plenty for a ratio

MASKS = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/zhmask_*.npz"):
    MASKS[f.stem[len("zhmask_"):]] = f
print(f"  zhmask crops found: {len(MASKS)}", flush=True)
assert MASKS, "biohub-zh-masks is not attached"

t0 = time.time()


def faces(L):
    """Count boundary FACES of labelled voxels, split by what is on the
    other side. Six-connected; each face counted once per direction.

    A face where L>0 meets 0 is a THRESHOLD boundary -- the mask stopped
    because the heatmap did. A face where L>0 meets a different L>0 is a
    WATERSHED boundary -- the mask stopped because a neighbour claimed
    the next voxel."""
    bg = nb = 0
    for ax in (0, 1, 2):
        a = np.moveaxis(L, ax, 0)
        u, v = a[:-1], a[1:]
        diff = u != v
        both = (u > 0) & (v > 0) & diff
        nb += int(both.sum())
        bg += int(((u > 0) & (v == 0)).sum())
        bg += int(((v > 0) & (u == 0)).sum())
    return bg, nb


print("\n=== TEST 1: what does a mask's surface touch? ===", flush=True)
tot_bg = tot_nb = 0
rows = []
for name in sorted(MASKS):
    z = np.load(MASKS[name])
    L = z["labels"]
    T = L.shape[0]
    cbg = cnb = 0
    vols, ncells = [], []
    for t in range(0, T, FRAME_STRIDE):
        Lt = L[t]
        b, n = faces(Lt)
        cbg += b
        cnb += n
        cnt = np.bincount(Lt.ravel())
        if len(cnt) > 1:
            v = cnt[1:]
            v = v[v > 0]
            if len(v):
                vols.append(float(np.median(v)) * VOXEL_UM3)
                ncells.append(int(len(v)))
    tot_bg += cbg
    tot_nb += cnb
    if vols:
        rows.append((name, float(np.mean(ncells)), float(np.median(vols)),
                     cbg / max(1, cbg + cnb)))
    del z, L

    if len(rows) % 10 == 0 and rows:
        print(f"  ...{len(rows)}/{len(MASKS)} crops ({time.time()-t0:.0f}s)",
              flush=True)

frac_bg = tot_bg / max(1, tot_bg + tot_nb)
print(f"\n  boundary faces against BACKGROUND:   {tot_bg:>12,}  "
      f"{100*frac_bg:5.1f}%", flush=True)
print(f"  boundary faces against ANOTHER MASK: {tot_nb:>12,}  "
      f"{100*(1-frac_bg):5.1f}%", flush=True)

if frac_bg >= 0.70:
    verdict1 = ("BACKGROUND-DOMINATED. Boundaries are set by the "
                "threshold, so the FRINGE reading is right and "
                "div_mass's conclusion was wrong.")
elif frac_bg <= 0.30:
    verdict1 = ("NEIGHBOUR-DOMINATED. Boundaries are watershed lines, so "
                "the TERRITORY reading is right and a voxel count can "
                "never be a mass on these masks.")
else:
    verdict1 = ("MIXED -- neither mechanism dominates. Read test 2, "
                "which resolves it by density.")
print(f"  VERDICT: {verdict1}", flush=True)

print("\n=== TEST 2: does a cell's mask shrink when the field is crowded? "
      "===", flush=True)
rows.sort(key=lambda r: r[1])
print(f"  {'crop':<26} {'cells/frame':>11} {'median vol um^3':>16} "
      f"{'bg face %':>10}", flush=True)
for nm, nc, vv, fb in rows[:5]:
    print(f"  {nm:<26} {nc:>11.0f} {vv:>16.1f} {100*fb:>9.1f}%", flush=True)
print(f"  {'...':<26}", flush=True)
for nm, nc, vv, fb in rows[-5:]:
    print(f"  {nm:<26} {nc:>11.0f} {vv:>16.1f} {100*fb:>9.1f}%", flush=True)

nc = np.array([r[1] for r in rows], float)
vv = np.array([r[2] for r in rows], float)
ok = (nc > 0) & (vv > 0)
slope, intercept = np.polyfit(np.log(nc[ok]), np.log(vv[ok]), 1)
r = float(np.corrcoef(np.log(nc[ok]), np.log(vv[ok]))[0, 1])
print(f"\n  log(volume) on log(cells/frame): slope {slope:+.3f}, "
      f"r {r:+.3f}, n={int(ok.sum())} crops", flush=True)
print(f"  density range {nc.min():.0f} to {nc.max():.0f} cells/frame "
      f"({nc.max()/max(1,nc.min()):.1f}x); volume range {vv.min():.0f} to "
      f"{vv.max():.0f} um^3 ({vv.max()/max(1,vv.min()):.1f}x)", flush=True)
print("  territory predicts slope near -1 (a cell's share is the volume "
      "divided by the cells); fringe predicts near 0 (a cell does not "
      "know how crowded it is).", flush=True)
if slope <= -0.6:
    verdict2 = "TERRITORY"
elif slope >= -0.25:
    verdict2 = "FRINGE"
else:
    verdict2 = "INTERMEDIATE -- both mechanisms contribute"
print(f"  VERDICT: {verdict2}", flush=True)

print("\n=== what to do with this ===", flush=True)
print("  FRINGE on both tests -> retry mass conservation with an "
      "INTENSITY-WEIGHTED mass. A fringe sits at heatmap 0.05-0.1 and a "
      "core near 0.9, so weighting by brightness discounts it for free. "
      "Needs the Zebrahub images mounted next to these masks; still CPU.",
      flush=True)
print("  TERRITORY on both -> the size channel is dead as a MASS "
      "measure. It may still work as a DENSITY measure, which is a "
      "different feature and a different claim.", flush=True)
print("  DISAGREEING -> the mechanism switches with density, and only "
      "the sparse crops can carry a mass measurement. That is a real "
      "finding and it names its own follow-up: rerun div_mass on the "
      "background-dominated crops alone.", flush=True)

print(f"\n  {time.time()-t0:.0f}s total", flush=True)
print("\nMASK ANATOMY DONE", flush=True)
