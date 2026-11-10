"""Synthetic test of refinecmp's placement rules, before the kernel is pushed.

The silent failure: if `place` returned the same coordinates for every rule,
all five rows would be identical, the run would report "nothing beats the
shipped rule", and **the anchor would still pass** -- because the shipped rule
IS `refine`, so the anchor is the control. That is exactly the shape the
working agreement's 2026-09-07 rule was written for.

So this puts a Gaussian at a KNOWN SUB-VOXEL position and asks which rule
finds it. A rule that cannot beat `nore` on a synthetic blob with no noise
will not beat it on real data either.

    py tools/test_refinecmp.py refinecmp.py
"""
import sys

import numpy as np

TARGET = sys.argv[1] if len(sys.argv) > 1 else "refinecmp.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end, what=None):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {what or start.strip()}: {b - a} chars")
    return SRC[a:b]


SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
NS = dict(np=np, POOL=POOL, SCALE=SCALE)
exec(cut("def _refine(", "\ndef _physical_nms"), NS)
exec(cut('RULES = ("nore"', "\n\ndef claim_map"), NS)
place, RULES = NS["place"], NS["RULES"]
assert RULES == ("nore", "refine", "refsm", "hcent", "hpara"), RULES

Zp, Yp, Xp = 9, 21, 21
Z, Y, X = Zp, Yp * POOL, Xp * POOL


def blob(cz, cy, cx, sig=1.25):
    """A Gaussian on the POOLED grid at a (possibly sub-voxel) centre."""
    zz, yy, xx = np.mgrid[0:Zp, 0:Yp, 0:Xp].astype(float)
    return np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
                    / (2 * sig ** 2)))


def raw_from(cz, cy, cx, sig_yx=4.0, sig_z=1.25):
    """The same cell on the RAW grid, which is what `_refine` reads."""
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(float)
    ry = cy * POOL + (POOL - 1) / 2.0
    rx = cx * POOL + (POOL - 1) / 2.0
    v = np.exp(-(((zz - cz) ** 2) / (2 * sig_z ** 2)
                 + ((yy - ry) ** 2 + (xx - rx) ** 2) / (2 * sig_yx ** 2)))
    return (v * 1000.0).astype(np.float32)


print("\n1. _para1 on a clean parabola recovers the vertex")
for true_d in (-0.4, -0.2, 0.0, 0.25, 0.45):
    a, b, c = (-(x - true_d) ** 2 for x in (-1.0, 0.0, 1.0))
    got = NS["_para1"](a, b, c)
    assert abs(got - true_d) < 1e-9, (true_d, got)
print("  OK: exact for offsets in [-0.45, +0.45]")
assert NS["_para1"](0.0, 0.0, 0.0) == 0.0
assert NS["_para1"](1.0, 0.0, 1.0) == 0.0, "a MINIMUM must not be shifted into"
print("  OK: a flat window and an upward cusp both return 0")

print("\n2. the pooled-index round trip the kernel does on `raw`")
pk = np.array([[4, 10, 10], [3, 7, 15]], int)
coords = pk.astype(float)
coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
back = np.rint(np.stack([coords[:, 0], (coords[:, 1] - (POOL - 1) / 2) / POOL,
                         (coords[:, 2] - (POOL - 1) / 2) / POOL], 1)).astype(int)
assert np.array_equal(back, pk), (back, pk)
print("  OK: raw -> pooled index is exact, so hcent / hpara start from the "
      "right voxel")

print("\n3. THE POINT: a cell at a known SUB-VOXEL position -- who finds it?")
print(f"  {'true (dz,dy,dx) in pooled voxels':<34} "
      + "  ".join(f"{r:>7}" for r in RULES))
ERR = {r: [] for r in RULES}
for dz, dy, dx in ((0.0, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4),
                   (0.3, 0.3, 0.0), (0.0, -0.45, 0.45), (0.25, -0.3, 0.35)):
    cz, cy, cx = 4 + dz, 10 + dy, 10 + dx
    h = blob(cz, cy, cx)
    vol = raw_from(cz, cy, cx)
    pk1 = np.array([np.unravel_index(int(np.argmax(h)), h.shape)], int)
    truth_um = np.array([cz, cy * POOL + (POOL - 1) / 2,
                         cx * POOL + (POOL - 1) / 2]) * SCALE
    row = []
    for r in RULES:
        got = place(r, h, vol, pk1)[0] * SCALE
        e = float(np.linalg.norm(got - truth_um))
        ERR[r].append(e)
        row.append(e)
    print(f"  {str((dz, dy, dx)):<34} " + "  ".join(f"{v:>7.3f}" for v in row))

print(f"\n  mean |error| um over {len(ERR['nore'])} sub-voxel placements:")
base = float(np.mean(ERR["nore"]))
for r in RULES:
    m = float(np.mean(ERR[r]))
    print(f"    {r:<7} {m:>7.3f}   {m - base:>+7.3f} vs nore")

assert not np.allclose(ERR["nore"], ERR["hpara"]), \
    "FATAL: hpara returned the same coordinates as nore -- the rules are not " \
    "distinct and the run would report 'nothing helps' having tested nothing"
assert not np.allclose(ERR["refine"], ERR["hcent"]), \
    "FATAL: hcent and refine are identical"
n_distinct = len({tuple(np.round(ERR[r], 6)) for r in RULES})
assert n_distinct == len(RULES), \
    f"only {n_distinct} of {len(RULES)} rules produce distinct answers"
print(f"  OK: all {len(RULES)} rules produce distinct answers")

print("\n4. does the heatmap actually know better than the grid?")
best = min(RULES, key=lambda r: float(np.mean(ERR[r])))
print(f"  best on synthetic data: {best} "
      f"({float(np.mean(ERR[best])):.3f} um mean error against nore's "
      f"{base:.3f})")
if float(np.mean(ERR["hpara"])) < base - 0.05:
    print("  hpara beats the bare grid by more than 0.05 um -- sub-voxel work "
          "is\n  possible in principle; the run says whether it survives real "
          "heatmaps")
else:
    print("  ⚠ hpara does NOT beat the bare grid here. On noiseless synthetic "
          "blobs\n  that is a bad sign for the whole idea; read the real "
          "numbers sceptically")

print("\nALL CHECKS PASSED for", TARGET)
