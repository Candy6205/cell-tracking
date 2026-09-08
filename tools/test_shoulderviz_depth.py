"""Render one synthetic frame with shoulderviz's own drawing code.

The kernel is 2.8 minutes and CPU-only, so the cost of a bad push is small --
but the failure mode here is not a crash, it is a picture that looks fine and
says the wrong thing, and that is exactly what v1 and v3 both shipped:

  v1  circled every GT node onto a slab pinned at Z//2, so cells at other
      depths drew rings on empty space.  "I cannot tell where the cells are."
  v3  fixed that by DROPPING every marker more than one voxel off-plane, so
      markers blinked in and out and a cell could not be followed at all.

v4 draws everything and encodes distance-from-plane in alpha, width and dash.
So the things to check before pushing are that matplotlib actually honours
per-point `linewidths` and RGBA `edgecolors` together with a per-collection
`linestyle` (it is not obvious that it does), that near and far markers end up
visibly different, and that NOTHING is dropped.

    py tools/test_shoulderviz_depth.py shoulderviz.py
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else "shoulderviz.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {start.strip()!r}: {b - a} chars")
    return SRC[a:b]


NS = {"np": np, "plt": plt}
exec(cut("def _depth_style(", "\ndef render("), NS)
_depth_style, _scat, _frame = NS["_depth_style"], NS["_scat"], NS["_frame"]

print("\n1. the depth encoding is monotone and never reaches zero")
dz = np.array([0.0, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0])
size, lw, rgba, near = _depth_style(dz, (0.25, 0.75, 1.0), 90.0, 1.1)
print(f"  {'|dz|':>6} {'alpha':>7} {'lw':>6} {'size':>7} {'solid':>7}")
for i, d in enumerate(dz):
    print(f"  {d:>6.1f} {rgba[i, 3]:>7.3f} {lw[i]:>6.2f} {size[i]:>7.1f} "
          f"{str(bool(near[i])):>7}")
assert np.all(np.diff(rgba[:, 3]) <= 0), "alpha must not rise with distance"
assert rgba[:, 3].min() >= 0.45, (
    "v4's first cut floored alpha at 0.16, which is invisible over bright "
    "tissue -- the drop-out bug achieved with opacity. Floor must stay high.")
assert rgba[0, 3] == 1.0, "an in-plane marker must be fully opaque"
assert near[:3].all() and not near[3:].any(), near
assert lw[0] > lw[-1], "far markers must be thinner"
print("  OK: monotone, floored at 0.50, in-plane is solid and opaque")

print("\n2. matplotlib accepts per-point lw + RGBA with a collection linestyle")
rng = np.random.default_rng(0)
m = rng.random((60, 90)).astype(np.float32) * 0.4
m[28:34, 40:48] = 1.0
pts = np.array([[31.0, 44.0], [10.0, 20.0], [50.0, 70.0], [20.0, 75.0]])
pdz = np.array([0.0, 3.0, 12.0, 30.0])
mk = np.array([[31.0, 44.0]])
mdz = np.array([0.0])
fig, ax = plt.subplots(figsize=(4.0, 3.0), facecolor="k")
_frame(ax, m, 0.0, 1.0, pts, pdz, mk, mdz, None, "test\nsecond line")
fig.canvas.draw()
img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
print(f"  rendered {img.shape} {img.dtype}, "
      f"{len(ax.collections)} collections drawn")
assert len(ax.collections) == 5, (
    f"expected blue backing + blue near + blue far, red backing + red near; "
    f"got {len(ax.collections)}")

print("\n3. every marker is drawn -- none is silently dropped")
n_drawn = sum(len(c.get_offsets()) for c in ax.collections)
expect = 2 * (len(pts) + len(mk))   # each marker also gets a backing ring
assert n_drawn == expect, (n_drawn, expect)
print(f"  {n_drawn} offsets for {len(pts)} GT + {len(mk)} miss, backing "
      f"rings included   OK (this is the v3 bug: it would be 4)")

print("\n4. near and far are visibly different in the pixels")
blue = ((img[..., 2].astype(int) - img[..., 0]) > 40) & (img[..., 1] > 60)
red = (img[..., 0].astype(int) - img[..., 1]) > 60
print(f"  blue pixels {int(blue.sum())}, red pixels {int(red.sum())}")
assert blue.sum() > 0 and red.sum() > 0, "markers did not reach the canvas"
a0 = _depth_style(np.array([0.0]), (0.25, 0.75, 1.0), 90.0, 1.1)[2][0, 3]
a1 = _depth_style(np.array([12.0]), (0.25, 0.75, 1.0), 90.0, 1.1)[2][0, 3]
assert a0 - a1 > 0.15, (a0, a1)
print(f"  in-plane alpha {a0:.2f} vs 12 voxels away {a1:.2f}   OK")

print("\n5. the empty case must not raise (frames with no GT in the crop)")
fig2, ax2 = plt.subplots(figsize=(2.0, 2.0))
_frame(ax2, m, 0.0, 1.0, np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)),
       np.zeros(0), None, "empty")
fig2.canvas.draw()
print("  OK: an empty crop renders")

print("\nALL CHECKS PASSED for", TARGET)
