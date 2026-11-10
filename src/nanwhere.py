"""nanwhere -- see the module docstring of make_nanwhere.py."""
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


# every GT cell whose rounded index fell outside the pooled
# heatmap array in gtpeak, i.e. h_gt came back NaN.
NAN_PTS = [
    ('6bba_3abfe10a', 64, 65000736, True),
    ('44b6_0b24845f', 39, 118000000027, True),
    ('44b6_0b24845f', 48, 127000000029, True),
    ('44b6_0b24845f', 49, 128000000029, True),
    ('44b6_0db75fae', 88, 129000000033, True),
    ('44b6_33b596bf', 21, 65000000083, True),
    ('44b6_33b596bf', 22, 66000000083, True),
    ('44b6_8f5ab931', 33, 159000000035, True),
    ('44b6_8f5ab931', 60, 186000000011, False),
    ('44b6_8f5ab931', 62, 188000000011, True),
    ('44b6_8f5ab931', 63, 189000000011, True),
    ('44b6_c15fded2', 22, 57000000019, True),
    ('44b6_c15fded2', 24, 59000000019, True),
    ('44b6_c96cfa10', 26, 109000000046, True),
    ('44b6_c96cfa10', 60, 143000000034, False),
    ('44b6_c96cfa10', 62, 145000000034, True),
    ('44b6_c96cfa10', 64, 147000000034, False),
    ('44b6_c96cfa10', 65, 148000000034, False),
    ('44b6_c96cfa10', 71, 154000000034, False),
    ('44b6_c96cfa10', 74, 157000000034, True),
    ('44b6_c96cfa10', 76, 159000000034, False),
    ('44b6_c96cfa10', 77, 160000000034, False),
    ('44b6_c96cfa10', 78, 161000000034, True),
    ('44b6_c96cfa10', 79, 162000000034, True),
    ('44b6_c96cfa10', 80, 163000000034, False),
    ('44b6_c96cfa10', 81, 164000000019, True),
    ('44b6_c96cfa10', 81, 164000000034, False),
    ('44b6_c96cfa10', 82, 165000000019, True),
    ('44b6_c96cfa10', 82, 165000000034, False),
    ('44b6_c96cfa10', 83, 166000000034, False),
    ('44b6_c96cfa10', 84, 167000000019, True),
    ('44b6_c96cfa10', 84, 167000000034, False),
    ('44b6_c96cfa10', 85, 168000000019, True),
    ('44b6_c96cfa10', 85, 168000000034, False),
    ('44b6_c96cfa10', 87, 170000000019, True),
    ('44b6_c96cfa10', 88, 171000000019, True),
    ('44b6_c96cfa10', 89, 172000000019, True),
    ('44b6_c96cfa10', 89, 172000000032, True),
    ('44b6_c96cfa10', 90, 173000000019, True),
    ('44b6_c96cfa10', 90, 173000000032, True),
    ('44b6_c96cfa10', 91, 174000000020, True),
    ('44b6_c96cfa10', 91, 174000000033, True),
    ('44b6_c96cfa10', 92, 175000000020, False),
    ('44b6_c96cfa10', 92, 175000000033, False),
    ('44b6_c96cfa10', 93, 176000000020, False),
    ('44b6_c96cfa10', 94, 177000000020, False),
    ('44b6_c96cfa10', 94, 177000000033, False),
    ('44b6_e29f0176', 37, 58000000014, True),
]

# the movies gtpeak/gtsplit actually measured; every OTHER train
# movie has no shoulder labels at all, so including it would put
# unlabelled cells in the denominator of every rate below.
MEASURED = [
    '6bba_2312ac41',
    '6bba_268e1230',
    '6bba_283bf9f1',
    '6bba_3a1849c2',
    '6bba_3abfe10a',
    '6bba_5c824876',
    '6bba_61dd1e0d',
    '6bba_7af54fde',
    '6bba_7b5d3b2c',
    '6bba_aeee7805',
    '6bba_afb141ff',
    '6bba_c27cba08',
    '44b6_0b24845f',
    '44b6_0db75fae',
    '44b6_33b596bf',
    '44b6_5740d24b',
    '44b6_87bba6c4',
    '44b6_8f5ab931',
    '44b6_9bfa6a0a',
    '44b6_a2bb48bb',
    '44b6_c15fded2',
    '44b6_c96cfa10',
    '44b6_d2f34f90',
    '44b6_e29f0176',
]


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
        key = ("z" if any(w[0] == "z" for w in why) else "") \
            + ("y" if any(w[0] == "y" for w in why) else "") \
            + ("x" if any(w[0] == "x" for w in why) else "")
        WHY[key or "IN RANGE (?!)"] += 1
        rows.append((name, t, n, mt, z, y, x, zc, yc, xc, Z, Yp, Xp,
                     " + ".join(why) or "none"))

print(f"\n=== volume shapes ({len(set(v[1:] for v in SHAPES.values()))} "
      f"distinct) ===", flush=True)
for shp, cnt in Counter(v[1:] for v in SHAPES.values()).most_common():
    Z, Y, X, Yp, Xp = shp
    print(f"  Z={Z} Y={Y} X={X} -> pooled {Yp} x {Xp}   "
          f"Y%{POOL}={Y % POOL} X%{POOL}={X % POOL}   ({cnt} movies)",
          flush=True)

print(f"\n=== THE ANSWER: which bound failed, for each of the "
      f"{len(rows)} NaN cells ===", flush=True)
print(f"  {'movie':<16} {'t':>3} {'z':>7} {'y':>8} {'x':>8}  "
      f"{'zc':>4} {'yc':>5} {'xc':>5}  {'matched':>7}  why", flush=True)
for r in sorted(rows):
    (nm, t, n, mt, z, y, x, zc, yc, xc, Z, Yp, Xp, why) = r
    print(f"  {nm:<16} {t:>3} {z:>7.2f} {y:>8.2f} {x:>8.2f}  "
          f"{zc:>4} {yc:>5} {xc:>5}  {str(mt):>7}  {why}", flush=True)

print(f"\n=== the verdict ===", flush=True)
for k, v in WHY.most_common():
    print(f"  {v:>3} of {len(rows)} fail on: {k}", flush=True)
print(f"\n  If it says y or x: this is OUR index arithmetic at the border of\n"
      f"  the frame -- `pool_xy` keeps only (Y//{POOL})*{POOL} rows and "
      f"`round(y/{POOL})`\n  can reach Y//{POOL}, one past the end. The cells "
      f"are in the image.\n  If it says z: the annotation really is outside "
      f"the stack, and the\n  reading this run was written to test was right "
      f"after all.", flush=True)

print(f"\n=== the OTHER thing worth knowing: the discarded strip ===",
      flush=True)
print(f"  pool_xy keeps rows [0, (Y//{POOL})*{POOL}) and columns "
      f"[0, (X//{POOL})*{POOL}).", flush=True)
print(f"  Of all {REM['n']:,} GT cells in these 24 movies, {REM['y']} sit at "
      f"y beyond that\n  cut and {REM['x']} sit at x beyond it. Anything in "
      f"the discarded strip is invisible\n  to the detector -- not a "
      f"measurement artifact but a real, if small, hole.", flush=True)

print(f"\nNANWHERE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("NANWHERE OK", flush=True)
