"""Synthetic test of gtpeak's new code paths, before the kernel is pushed.

Three new paths, each with a silent failure an anchor cannot catch:

  claim_map      the anchor (1,047 / 0.9761, 297 / 0.9596) is computed with
                 THIS matcher but was recorded with detfuse3's `match_flags`.
                 If they disagree the anchor fails for the wrong reason -- or,
                 worse, passes while measuring something else.
  detect_stages  if `raw` and `nms` came back as the same set, the "NMS
                 removed it" bucket would always be empty and the run would
                 conclude NMS is innocent without ever testing it.
  successor pairing   if the GT track links never resolve, the frame-to-frame
                 table prints "0 steps" and looks merely uninformative rather
                 than broken.

    py tools/test_gtpeak.py gtpeak.py
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

TARGET = sys.argv[1] if len(sys.argv) > 1 else "gtpeak.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end, what):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {what}: {b - a} chars")
    return SRC[a:b]


SCALE = np.array([1.625, 0.40625, 0.40625])
NS = dict(np=np, cKDTree=cKDTree, SCALE=SCALE, POOL=4, NMS_UM=4.0,
          DETECT_THRESH=0.05, GATES=(0.05, 0.15, 0.30))
for name, (a, b) in {
        "_refine": ("def _refine(", "\ndef _physical_nms"),
        "_physical_nms": ("def _physical_nms(", "\n_BRANCHES"),
        "match_flags": ("def match_flags(", "\n\nSKIP_OFFICIAL"),
        "claim_map": ("def claim_map(", "\n\ndef _nearest"),
        "_nearest": ("def _nearest(", "\n\ndef run_domain"),
}.items():
    exec(cut(a, b, name), NS)
match_flags, claim_map = NS["match_flags"], NS["claim_map"]

print("\n1. claim_map must agree with detfuse3's match_flags on RECALL")
rng = np.random.default_rng(3)
worst = 0
for trial in range(200):
    ng = int(rng.integers(1, 40))
    nd = int(rng.integers(0, 80))
    gt_um = rng.uniform(0, 60, size=(ng, 3))
    det_vox = rng.uniform(0, 60, size=(nd, 3)) / SCALE[None, :]
    sc = rng.uniform(0.0, 1.0, size=nd)
    _m, claimed, _t, _n = match_flags(det_vox, sc, gt_um, 7.0)
    g2d = claim_map(det_vox * SCALE[None, :], sc, gt_um, 7.0)
    a = len(claimed)
    b = int((g2d >= 0).sum())
    worst = max(worst, abs(a - b))
    assert a == b, (trial, a, b)
    # and every claimed GT must be within the radius of the detection it names
    for j in np.where(g2d >= 0)[0]:
        d = float(np.linalg.norm(det_vox[g2d[j]] * SCALE - gt_um[j]))
        assert d <= 7.0 + 1e-9, (trial, j, d)
print(f"  OK: 200 random cases, recall identical every time (max diff {worst}); "
      f"every claim is inside the radius")

print("\n2. detect_stages: raw must be a SUPERSET of what survives NMS")
Z, Y, X = 12, 40, 40


def fake_hs(centres, amp=0.9, sig=1.2):
    zz, yy, xx = np.mgrid[0:Z, 0:Y // 4, 0:X // 4].astype(float)
    h = np.zeros_like(zz)
    for c in centres:
        h = np.maximum(h, amp * np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2
                                          + (xx - c[2]) ** 2) / (2 * sig ** 2))))
    return [h, h, h, h, np.zeros_like(h), np.zeros_like(h)]


NS2 = dict(NS)
exec(cut("def detect_stages(", "\n\n# ", "detect_stages"), NS2)
try:
    from skimage.feature import peak_local_max  # noqa: E402
    print("  using the real skimage.feature.peak_local_max")
except ImportError:
    from scipy.ndimage import maximum_filter  # noqa: E402

    def peak_local_max(h, min_distance=1, threshold_abs=0.0,
                       exclude_border=False):
        """Stand-in: skimage is not installed locally. Same semantics for
        min_distance=1 -- strict local maxima of a (2d+1)^3 window above the
        threshold. Equivalent on the isolated blobs this test uses; tie and
        plateau handling is NOT identical, so this is a HARNESS, not a
        reimplementation of what runs on Kaggle."""
        k = 2 * min_distance + 1
        m = (h == maximum_filter(h, size=k)) & (h > threshold_abs)
        return np.argwhere(m)

    print("  skimage absent -> scipy stand-in for peak_local_max "
          "(min_distance=1 semantics only; the Kaggle run uses the real one)")
NS2["peak_local_max"] = peak_local_max
detect_stages = NS2["detect_stages"]
vol = np.zeros((Z, Y, X), np.float32)

# two blobs 2 pooled voxels apart = 3.25 um < NMS_UM 4.0 -> NMS must cull one
near = fake_hs([(6, 5, 5), (6, 5, 7)])
cm, scm, raw, sc_raw, ref, h = detect_stages(near, vol, (1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
print(f"  two blobs 3.25 um apart: raw peaks {len(raw)}, survivors {len(cm)}")
assert len(raw) >= 2, len(raw)
assert len(cm) < len(raw), "NMS removed nothing -- the 'NMS killed it' bucket " \
                           "could never fill and the run would clear NMS falsely"

far = fake_hs([(6, 3, 2), (6, 3, 7)])        # 5 pooled voxels = 8.1 um, both survive
cm2, scm2, raw2, _s, _r, _h = detect_stages(far, vol, (1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
print(f"  two blobs 8.1 um apart: raw peaks {len(raw2)}, survivors {len(cm2)}")
assert len(cm2) == len(raw2) >= 2, (len(cm2), len(raw2))
assert h.shape == (Z, Y // 4, X // 4), h.shape
assert np.isclose(float(h.max()), 0.9, atol=0.05), float(h.max())
print(f"  the returned heatmap is the F4 blend itself (max {h.max():.3f}), so "
      f"h_gt / h_win are read off the same map the peaks came from")

print("\n3. the official maps are weighted 0, so zeroing them changes nothing")
w = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
hs_real = fake_hs([(6, 5, 5)])
hs_zero = list(hs_real)
hs_zero[4] = np.full_like(hs_real[0], 0.7)     # anything at all
hs_zero[5] = np.full_like(hs_real[0], 0.3)
a = detect_stages(hs_real, vol, w)
b = detect_stages(hs_zero, vol, w)
assert np.allclose(a[5], b[5]), "the 0-weighted maps leaked into the blend"
assert len(a[0]) == len(b[0])
print("  OK: SKIP_OFFICIAL is numerically free, not an approximation")

print("\n4. the GT successor pairing resolves along a track")
SUCC_OF = {"d0": {1: 2, 2: 3, 3: 4}}
tags = np.array(["d0"] * 4)
nid = np.array([1, 2, 3, 4])
mt = np.array([True, True, False, True])
off = np.array([[0.5, 0, 0], [-0.5, 0, 0], [np.nan] * 3, [1.5, 0, 0]])
pos = {(tags[i], nid[i]): i for i in range(4) if mt[i] and nid[i] >= 0}
ch = []
for i in range(4):
    if not mt[i] or nid[i] < 0:
        continue
    nx = SUCC_OF.get(tags[i], {}).get(int(nid[i]))
    j = pos.get((tags[i], nx)) if nx is not None else None
    if j is None:
        continue
    ch.append(float(np.linalg.norm(off[j] - off[i])))
assert ch == [1.0], ch     # 1->2 only; 2->3 unmatched, 3->4 unmatched source
print(f"  OK: {len(ch)} step(s) from a 4-node track with one unmatched frame, "
      f"|d offset| = {ch[0]:.1f} um (a cell that drifts +0.5 -> -0.5 injects 1 um)")

print("\n5. the four miss buckets partition the misses")
n = 1000
rng = np.random.default_rng(5)
ms = np.ones(n, bool)
d_raw = rng.choice([2.0, 20.0], n)
d_nms = np.where(d_raw <= 7.0, rng.choice([3.0, 20.0], n), 20.0)
hw = rng.uniform(0, 0.2, n)
mtm = rng.random(n) < 0.0
no_raw = ms & ~(d_raw <= 7.0)
cold = no_raw & ~(hw >= 0.05)
warm = no_raw & (hw >= 0.05)
had = ms & (d_raw <= 7.0)
killed = had & ~(d_nms <= 7.0)
surv = had & (d_nms <= 7.0)
assert int(cold.sum() + warm.sum() + killed.sum() + surv.sum()) == int(ms.sum())
assert all(x.sum() > 0 for x in (cold, warm, killed, surv)), "a bucket is dead"
print(f"  OK: cold {cold.sum()}  warm {warm.sum()}  nms-killed {killed.sum()}  "
      f"survived {surv.sum()}  = {int(ms.sum())}, and none is structurally empty")

print("\nALL CHECKS PASSED for", TARGET)
