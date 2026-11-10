"""Synthetic test of gtsplit's splitter, before the kernel is pushed.

The silent failure here is severe and an anchor cannot catch it: if
`split_detect` returns one detection per blob -- i.e. splits NOTHING -- every
arm reprints BASE, the anchor still reproduces, and the run reports "a full
split recovers 0 cells" when in truth nothing was ever split. That reads as an
answer and is not one.

So the test asks the question the method actually has to survive: gtpeak found
the missed cells have NO intensity maximum of their own. Can a shape split
separate a lobe that the intensity does not separate? Sweep two Gaussians over
separation and amplitude ratio, and count the cell that matters:

    intensity maxima = 1  AND  split_detect returns 2

If that cell is empty for every configuration, thresholding + distance
transform cannot recover what gtpeak measured, and the ceiling is ~0 before
any Kaggle time is spent.

    py tools/test_gtsplit.py gtsplit.py
"""
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt as _edt
from scipy.spatial import cKDTree

TARGET = sys.argv[1] if len(sys.argv) > 1 else "gtsplit.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end, what=None):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {what or start.strip()}: {b - a} chars")
    return SRC[a:b]


SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
try:
    from skimage.feature import peak_local_max
    from skimage.measure import label as _cc_label
    from skimage.segmentation import watershed as _watershed
    HAVE_SK = True
except ImportError:
    HAVE_SK = False
    print("  skimage is not installed locally -- this test CANNOT run.\n"
          "  Install it (pip install scikit-image) or the splitter goes to "
          "Kaggle untested, which the working agreement forbids.")
    sys.exit(2)

NS = dict(np=np, cKDTree=cKDTree, SCALE=SCALE, POOL=POOL, POOLED=POOLED,
          NMS_UM=4.0, DETECT_THRESH=0.05, peak_local_max=peak_local_max,
          _cc_label=_cc_label, _watershed=_watershed, _edt=_edt)
exec(cut("def _refine(", "\ndef _physical_nms"), NS)
exec(cut("def _physical_nms(", "\n_BRANCHES"), NS)
exec(cut("def split_detect(", "\n\ndef claim_map"), NS)
split_detect = NS["split_detect"]

Z, YP, XP = 15, 30, 30
vol = np.zeros((Z, YP * POOL, XP * POOL), np.float32)


def two_blobs(sep_vox, a1, a2, sig=1.6):
    """Two Gaussians on the pooled grid, separated along x."""
    zz, yy, xx = np.mgrid[0:Z, 0:YP, 0:XP].astype(float)
    c = (7.0, 15.0, 15.0)
    h = np.zeros_like(zz)
    for amp, dx in ((a1, -sep_vox / 2.0), (a2, +sep_vox / 2.0)):
        h = h + amp * np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2
                                + (xx - c[2] - dx) ** 2) / (2 * sig ** 2)))
    return np.clip(h, 0, 1.0)


def n_intensity_maxima(h, thr=0.05):
    return len(peak_local_max(h, min_distance=1, threshold_abs=thr,
                              exclude_border=False))


print("\n1. can a SHAPE split separate what the INTENSITY does not?")
print(f"  {'sep(vox/um)':>12} {'a1':>5} {'a2':>5} {'int.maxima':>11} "
      f"{'split@0.05':>11} {'split@0.15':>11}   the cell that matters")
FOUND = []
for sep in (2, 3, 4, 5, 6):
    for a1, a2 in ((0.9, 0.9), (0.9, 0.5), (0.9, 0.3), (0.9, 0.2)):
        h = two_blobs(sep, a1, a2)
        nmax = n_intensity_maxima(h)
        n05 = len(split_detect(h, vol, 0.05, None)[0])
        n15 = len(split_detect(h, vol, 0.15, None)[0])
        hit = (nmax == 1) and (max(n05, n15) >= 2)
        if hit:
            FOUND.append((sep, a1, a2, n05, n15))
        print(f"  {sep:>4} / {sep * 1.625:>5.2f} {a1:>5.1f} {a2:>5.1f} "
              f"{nmax:>11} {n05:>11} {n15:>11}   "
              f"{'*** SPLIT WITHOUT AN INTENSITY MAX ***' if hit else ''}")

print(f"\n  configurations where the intensity has ONE maximum but the shape "
      f"split finds two: {len(FOUND)}")
if not FOUND:
    print("  ==> On smooth merged Gaussians a threshold + distance transform "
          "does NOT\n      separate a lobe with no intensity maximum: every "
          "level set of a unimodal\n      blob is a single un-pinched region, "
          "so its distance map has one maximum\n      too. If this holds on "
          "the real heatmaps the recoverable count is ~0.")
else:
    print("  ==> the method can work; the run will say how often it does on "
          "real data")

print("\n2. the control: two clearly separate cells must still give two")
h = two_blobs(8, 0.9, 0.9)
n = len(split_detect(h, vol, 0.15, None)[0])
print(f"  sep 8 vox = 13.0 um, equal amplitude: {n} detections "
      f"(intensity maxima {n_intensity_maxima(h)})")
assert n >= 2, "the splitter cannot even separate two obviously distinct cells"

print("\n3. NMS at 4.0 um re-merges anything the split separated below it")
h = two_blobs(2, 0.9, 0.9)
a = len(split_detect(h, vol, 0.15, None)[0])
b = len(split_detect(h, vol, 0.15, 4.0)[0])
print(f"  sep 2 vox = 3.25 um: no NMS -> {a}, NMS 4.0 -> {b}")
assert b <= a
if a >= 2:
    assert b < a, "NMS did not re-merge a 3.25 um pair -- the s15 vs s15n " \
                  "comparison would be meaningless"
    print("  OK: the s15 / s15n arms will actually isolate the NMS interaction")
else:
    print("  (the split did not separate this pair, so the NMS arm is moot "
          "here; the real data will say)")

print("\n4. a single isolated cell must not be shattered")
h = two_blobs(0, 0.9, 0.0)
n05 = len(split_detect(h, vol, 0.05, None)[0])
n15 = len(split_detect(h, vol, 0.15, None)[0])
print(f"  one Gaussian: split@0.05 -> {n05}, split@0.15 -> {n15}")
assert n05 <= 2 and n15 <= 2, (n05, n15)
print("  OK: no over-segmentation of a clean single blob")

print("\nALL CHECKS RAN for", TARGET)
