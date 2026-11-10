"""nanwhere: WHICH axis is out of bounds for the 48 cells gtpeak read as NaN.

No theory. The user, 2026-09-08: "我不行了，到底是什么情况，你去量一下吧".

BACKGROUND, short. gtpeak records `h_gt`, the heatmap value at the annotated
voxel, as

    zc = int(round(z));  yc = int(round(y / POOL));  xc = int(round(x / POOL))
    hg = h[zc, yc, xc] if all three are in range else NaN

48 of 13,646 GT cells came back NaN. This session first read that as "the
annotation is outside the imaged field" -- which the user refuted in one line
(a human cannot annotate a cell that is not in the picture), and which was
already contradicted by the fact that **30 of the 48 were matched by the
detector**, and the detector only runs inside the volume.

The remaining candidate is our own index arithmetic. `pool_xy` TRUNCATES:
`Y2 = (Y // POOL) * POOL`, and the pooled array has `Y // POOL` columns. So a
cell can fall out of range two different ways on y (and the same on x):

  * Y is not divisible by POOL, and the cell sits in the discarded remainder;
  * y rounds up: y / 4 >= (Y // 4) - 0.5 gives index Y // 4, one past the end.

and one way on z, which is NOT pooled: z outside [-0.5, Z - 0.5], i.e. an
annotation genuinely outside the stack.

THE OUTPUT IS THE ANSWER: for each of the 48, the raw (z, y, x), the volume
shape, the pooled shape, the three computed indices, and a label saying which
bound failed and by how much. If it is z, the annotation really is outside the
stack and the earlier reading was right after all. If it is y or x by ONE
index at the top end, it is our off-by-one and these cells are ordinary cells
sitting against the border of the frame.

It also prints, for the whole GT set, how many cells sit in the truncated
remainder -- because if `Y % POOL != 0` then a strip of every frame is thrown
away before the detector ever sees it, and that would be a real (if small)
pipeline fact rather than a measurement artifact.

CPU, geffs + zarr metadata only, about a minute.
"""
import os

SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
PTS = open(SC + "nanpts.py", encoding="utf-8").read()
assert PTS.count("\n    (") == 48, PTS.count("\n    (")
PTS += "\n" + open(SC + "measured_movies.py", encoding="utf-8").read()
assert PTS.count("\n    '") == 24, PTS.count("\n    '")

BODY = '''"""nanwhere -- see the module docstring of make_nanwhere.py."""
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

SCRIPT_VERSION = "2026-09-08a-nanwhere"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets, zarr_path)

POOL = 4        # gtpeak's pooling factor in y and x; z is NOT pooled
ROOT = find_root(None)


def read_shape(zp):
    with open(pathlib.Path(zp) / "0" / "zarr.json") as f:
        return tuple(json.load(f)["shape"])


''' + PTS + '''

BY_MOVIE = {}
for m, t, n, mt in NAN_PTS:
    BY_MOVIE.setdefault(m, []).append((t, n, mt))
print(f"{len(NAN_PTS)} NaN cells over {len(BY_MOVIE)} movies", flush=True)

t0 = time.time()
MOVIES = [d for d in list_datasets(ROOT, "train") if d in set(MEASURED)]
assert len(MOVIES) == 24, len(MOVIES)

WHY = Counter()
REM = Counter()
SHAPES = {}
rows = []
for name in MOVIES:
    T, Z, Y, X = read_shape(zarr_path(ROOT, "train", name))
    Yp, Xp = Y // POOL, X // POOL
    SHAPES[name] = (T, Z, Y, X, Yp, Xp)
    g = read_geff(geff_path(ROOT, "train", name))
    nid = np.asarray(g["node_ids"], np.int64)
    zz = np.asarray(g["z"], float)
    yy = np.asarray(g["y"], float)
    xx = np.asarray(g["x"], float)
    pos = {int(n): (zz[i], yy[i], xx[i]) for i, n in enumerate(nid)}

    # how many of ALL this movie's GT cells sit in the strip pool_xy discards
    rem_y = int((yy >= (Y // POOL) * POOL).sum())
    rem_x = int((xx >= (X // POOL) * POOL).sum())
    REM["y"] += rem_y
    REM["x"] += rem_x
    REM["n"] += len(nid)

    for t, n, mt in BY_MOVIE.get(name, []):
        v = pos.get(n)
        if v is None:
            print(f"  ! {name}: node {n} not in the geff", flush=True)
            continue
        z, y, x = v
        zc = int(round(z))
        yc = int(round(y / POOL))
        xc = int(round(x / POOL))
        why = []
        if not (0 <= zc < Z):
            why.append(f"z {zc} vs [0,{Z})")
        if not (0 <= yc < Yp):
            why.append(f"y {yc} vs [0,{Yp})")
        if not (0 <= xc < Xp):
            why.append(f"x {xc} vs [0,{Xp})")
        key = ("z" if any(w[0] == "z" for w in why) else "") \\
            + ("y" if any(w[0] == "y" for w in why) else "") \\
            + ("x" if any(w[0] == "x" for w in why) else "")
        WHY[key or "IN RANGE (?!)"] += 1
        rows.append((name, t, n, mt, z, y, x, zc, yc, xc, Z, Yp, Xp,
                     " + ".join(why) or "none"))

print(f"\\n=== volume shapes ({len(set(v[1:] for v in SHAPES.values()))} "
      f"distinct) ===", flush=True)
for shp, cnt in Counter(v[1:] for v in SHAPES.values()).most_common():
    Z, Y, X, Yp, Xp = shp
    print(f"  Z={Z} Y={Y} X={X} -> pooled {Yp} x {Xp}   "
          f"Y%{POOL}={Y % POOL} X%{POOL}={X % POOL}   ({cnt} movies)",
          flush=True)

print(f"\\n=== THE ANSWER: which bound failed, for each of the "
      f"{len(rows)} NaN cells ===", flush=True)
print(f"  {'movie':<16} {'t':>3} {'z':>7} {'y':>8} {'x':>8}  "
      f"{'zc':>4} {'yc':>5} {'xc':>5}  {'matched':>7}  why", flush=True)
for r in sorted(rows):
    (nm, t, n, mt, z, y, x, zc, yc, xc, Z, Yp, Xp, why) = r
    print(f"  {nm:<16} {t:>3} {z:>7.2f} {y:>8.2f} {x:>8.2f}  "
          f"{zc:>4} {yc:>5} {xc:>5}  {str(mt):>7}  {why}", flush=True)

print(f"\\n=== the verdict ===", flush=True)
for k, v in WHY.most_common():
    print(f"  {v:>3} of {len(rows)} fail on: {k}", flush=True)
print(f"\\n  If it says y or x: this is OUR index arithmetic at the border of\\n"
      f"  the frame -- `pool_xy` keeps only (Y//{POOL})*{POOL} rows and "
      f"`round(y/{POOL})`\\n  can reach Y//{POOL}, one past the end. The cells "
      f"are in the image.\\n  If it says z: the annotation really is outside "
      f"the stack, and the\\n  reading this run was written to test was right "
      f"after all.", flush=True)

print(f"\\n=== the OTHER thing worth knowing: the discarded strip ===",
      flush=True)
print(f"  pool_xy keeps rows [0, (Y//{POOL})*{POOL}) and columns "
      f"[0, (X//{POOL})*{POOL}).", flush=True)
print(f"  Of all {REM['n']:,} GT cells in these 24 movies, {REM['y']} sit at "
      f"y beyond that\\n  cut and {REM['x']} sit at x beyond it. Anything in "
      f"the discarded strip is invisible\\n  to the detector -- not a "
      f"measurement artifact but a real, if small, hole.", flush=True)

print(f"\\nNANWHERE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("NANWHERE OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "nanwhere.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(BODY)
print("written", len(BODY))
