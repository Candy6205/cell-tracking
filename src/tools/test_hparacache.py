"""Check hparacache's two arms are actually different, before spending a GPU.

THE SILENT FAILURE. The kernel's anchor is that the `refine` half reproduces
the shipped `fdet_F4_` cache exactly. That anchor is satisfied by a kernel in
which `_hpara_shift` is a no-op -- both arms would then be byte-identical, the
bench downstream would report "hpara changes nothing", and every check would
pass. The anchor CANNOT catch this, because the anchor is the control arm.

So: put a Gaussian at a known sub-voxel position on the pooled grid and ask
whether the shift finds it, and confirm `_nms_keep` reproduces the suppression
`_physical_nms` has always done.

    py tools/test_hparacache.py hparacache.py
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

TARGET = sys.argv[1] if len(sys.argv) > 1 else "hparacache.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {start.strip()!r}: {b - a} chars")
    return SRC[a:b]


SCALE = np.array([1.625, 0.40625, 0.40625])
NS = {"np": np, "cKDTree": cKDTree, "SCALE": SCALE}
exec(cut("def _nms_keep(", "\n_BRANCHES"), NS)
exec(cut("def _para1(", "\ndef _hpara_shift"), NS)
exec(cut("def _hpara_shift(", "\ndef detect_both"), NS)
_nms_keep = NS["_nms_keep"]
_physical_nms = NS["_physical_nms"]
_para1 = NS["_para1"]
_hpara_shift = NS["_hpara_shift"]

print("\n1. _nms_keep reproduces _physical_nms exactly")
rng = np.random.default_rng(7)
for trial in range(200):
    n = int(rng.integers(0, 40))
    c = rng.random((n, 3)) * np.array([20.0, 200.0, 200.0])
    s = rng.random(n)
    keep = _nms_keep(c, s, 4.0)
    kc, ks = _physical_nms(c, s, 4.0)
    if n <= 1:
        assert np.array_equal(kc, c) and np.array_equal(ks, s)
        continue
    assert np.array_equal(c[keep], kc), trial
    assert np.array_equal(s[keep], ks), trial
print("  200 random frames, 0-40 detections: identical every time   OK")
kept = _nms_keep(np.array([[0., 0., 0.], [0., 1., 0.]]), np.array([0.9, 0.5]),
                 4.0)
assert list(kept) == [0], kept
print("  and it does suppress: two points 0.41 um apart -> the brighter   OK")

print("\n2. _para1 recovers a known vertex, and refuses a minimum")
for true_d in (-0.45, -0.2, 0.0, 0.3, 0.45):
    a, b, c = (-(x - true_d) ** 2 for x in (-1.0, 0.0, 1.0))
    assert abs(_para1(a, b, c) - true_d) < 1e-9, (true_d, _para1(a, b, c))
print("  exact for offsets in [-0.45, +0.45]   OK")
assert _para1(0.0, 0.0, 0.0) == 0.0
assert _para1(1.0, 0.0, 1.0) == 0.0, "an upward cusp must not be shifted into"
print("  a flat window and an upward cusp both return 0   OK")

print("\n3. THE POINT: the shift is NOT a no-op, and it points the right way")
Zp, Yp, Xp = 9, 21, 21


def blob(cz, cy, cx, sig=1.25):
    zz, yy, xx = np.mgrid[0:Zp, 0:Yp, 0:Xp].astype(float)
    return np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
                    / (2 * sig ** 2)))


print(f"  {'true (dz,dy,dx)':<26} {'peak':<14} {'shifted':<24} {'|err| vox'}")
errs_raw, errs_hp = [], []
for dz, dy, dx in ((0.0, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, -0.4),
                   (0.3, 0.3, 0.0), (0.0, -0.45, 0.45), (0.25, -0.3, 0.35)):
    cz, cy, cx = 4 + dz, 10 + dy, 10 + dx
    h = blob(cz, cy, cx)
    pk = np.array([np.unravel_index(int(np.argmax(h)), h.shape)], int)
    got = _hpara_shift(h, pk)[0]
    truth = np.array([cz, cy, cx])
    e_raw = float(np.linalg.norm(pk[0].astype(float) - truth))
    e_hp = float(np.linalg.norm(got - truth))
    errs_raw.append(e_raw)
    errs_hp.append(e_hp)
    print(f"  {str((dz, dy, dx)):<26} {str(list(pk[0])):<14} "
          f"{np.array2string(got, precision=3):<24} "
          f"{e_raw:.3f} -> {e_hp:.3f}")

assert np.mean(errs_hp) < np.mean(errs_raw) - 0.05, (
    f"FATAL: the shift does not improve on the bare peak "
    f"({np.mean(errs_hp):.4f} vs {np.mean(errs_raw):.4f}). If _hpara_shift is "
    f"a no-op, BOTH arms of the kernel are identical, the bench reports "
    f"'hpara changes nothing', and the anchor still passes -- because the "
    f"anchor IS the control arm.")
print(f"\n  mean |error| {np.mean(errs_raw):.3f} -> {np.mean(errs_hp):.3f} "
      f"pooled voxels ({1.625 * np.mean(errs_raw):.3f} -> "
      f"{1.625 * np.mean(errs_hp):.3f} um)   OK")

print("\n4. the shift is bounded by half a voxel on every axis")
big = _hpara_shift(blob(4.0, 10.0, 10.0), np.array([[4, 10, 10]], int))[0]
assert np.allclose(big, [4.0, 10.0, 10.0], atol=1e-9), big
for _ in range(200):
    h = rng.random((Zp, Yp, Xp))
    pk = np.array([[4, 10, 10]], int)
    d = _hpara_shift(h, pk)[0] - pk[0]
    assert np.all(np.abs(d) <= 0.5 + 1e-9), d
print("  200 random heatmaps: |shift| <= 0.5 voxel on all three axes   OK")

print("\n5. a peak on the volume border is left alone, not read out of bounds")
for pk in ([[0, 10, 10]], [[Zp - 1, 10, 10]], [[4, 0, 10]], [[4, 10, Xp - 1]]):
    h = blob(4.2, 10.2, 10.2)
    got = _hpara_shift(h, np.array(pk, int))[0]
    ax = [i for i in range(3) if pk[0][i] in (0, h.shape[i] - 1)]
    for i in ax:
        assert got[i] == pk[0][i], (pk, got, i)
print("  the border axis is not shifted; the others still are   OK")

print("\nALL CHECKS PASSED for", TARGET)
