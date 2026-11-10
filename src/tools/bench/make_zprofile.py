"""shoulderfeat.py -> zprofile.py: does the heatmap fall off at the z edges
for EVERY cell, or only for the ones that were missed?

THE FORK THIS SPLITS. shoulderfeat found the 313 shoulder cells sit nearer a
z edge of the stack than their own movie's cells in 17 of 18 movies (sign test
p = 1.5e-5). Two mechanisms produce that identical table:

  (a) the cell is physically half outside the imaged volume, so only part of
      it was ever recorded -- NOTHING recovers it, at any architecture;
  (b) the detector degrades at the z border of its own input. The three UNet
      branches are `nn.Conv3d(..., padding=1)`, i.e. PyTorch's default ZERO
      padding, and the whole volume goes through in one forward pass, so the
      only boundary is the real one. Accumulating the receptive field --
      2 convs per block, MaxPool3d(2) pooling z as well, 64 -> 32 -> 16 -> 8 --
      the encoder alone reaches about 30 planes and the decoder about as much
      again, against a stack that is only 64 planes deep. In z NO voxel is
      far from a fabricated neighbourhood. In y/x, which are hundreds of
      voxels wide, only a thin rim is. That asymmetry is exactly the shape of
      the finding. And (b) is FIXABLE -- reflect padding, a z-calibration
      curve, a depth-dependent threshold, partial convolutions, or training
      with z-crop augmentation.

The user, 2026-09-08: "这个问题没有其他的解法吗，按照你的说法应该是卷积网络
特有的问题". Yes -- and the cheapest of them is also the DIAGNOSTIC, which is
what this run is.

THE MEASUREMENT. gtsplit's CSVs already carry `h_gt`, the F4 heatmap value at
the annotated voxel, for all 13,646 GT cells -- not just the 313. Join those
to each cell's z from the geff and read the profile:

  * if h_gt sags at BOTH z edges for cells the detector FOUND as well, the
    attenuation is systematic and output-level -- branch (b), and a
    z-calibration is the cheap fix;
  * if h_gt is flat in z among matched cells and only the missed ones are
    weak, the attenuation story is wrong and (a) is the better reading.

The matched cells are the control, and they are the point: 97.6% / 96.0% of
GT is matched, so their profile is a clean readout of the DETECTOR's response
against depth, uncontaminated by the question being asked.

NO GPU, no images -- geffs and an embedded table. About a minute.

NO ANCHOR against a previous run; the internal check is that the 313 shoulder
node_ids must reappear as the low-h_gt tail, and shoulderfeat's per-movie
counts must reproduce.
"""
import os

SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
PTS = open(SC + "shoulder_all.py", encoding="utf-8").read()
assert PTS.count("\n    (") == 313, PTS.count("\n    (")
PTS += "\n" + open(SC + "measured_movies.py", encoding="utf-8").read()
assert PTS.count("\n    '") == 24, PTS.count("\n    '")
HGT = open(SC + "hgt_all.py", encoding="utf-8").read()
assert HGT.count("\n    (") == 13646, HGT.count("\n    (")

BODY = '''"""zprofile -- see the module docstring of make_zprofile.py."""
import pathlib
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-09-08a-zprofile"
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


def read_array_meta(zp):
    """cache_train.py:137 verbatim -- viz_cells does not export it, and the
    stack DEPTH is what makes 'distance to the edge' mean the edge of the
    IMAGED volume rather than the edge of the annotated cells."""
    import json
    with open(pathlib.Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]))

ROOT = find_root(None)
SCALE = np.array([1.625, 0.40625, 0.40625])
NDEC = 12

''' + PTS + "\n" + HGT + '''

SH_IDS = {(m, n) for m, _t, n, _a, _b in SHOULDER}
H_OF = {(m, n): (h, mt) for m, n, h, mt in HGT}
print(f"{len(HGT)} GT cells carry h_gt; {len(SH_IDS)} of them are shoulder",
      flush=True)
assert len(H_OF) == len(HGT), "duplicate (dataset, node_id) in HGT"

t0 = time.time()
MOVIES = [d for d in list_datasets(ROOT, "train") if d in set(MEASURED)]
assert len(MOVIES) == 24, len(MOVIES)

ROWS = []
for name in MOVIES:
    g = read_geff(geff_path(ROOT, "train", name))
    sc = np.asarray(g.get("scale") or SCALE, float)
    nid = np.asarray(g["node_ids"], np.int64)
    zz = np.asarray(g["z"], float)               # VOXELS, not um
    # the true stack depth, so "distance to the edge" is measured against the
    # IMAGED volume and not merely against the annotated cells
    try:
        Z = int(read_array_meta(zarr_path(ROOT, "train", name))["shape"][1])
    except Exception as e:                                    # noqa: BLE001
        print(f"  [degraded] {name}: cannot read the array shape ({e}); "
              f"falling back to the GT z range, which UNDERSTATES the "
              f"distance to the edge", flush=True)
        Z = int(zz.max()) + 1
    hit = miss = 0
    for i, n in enumerate(nid):
        v = H_OF.get((name, int(n)))
        if v is None:
            miss += 1
            continue
        hit += 1
        h, mt = v
        z = float(zz[i])
        ROWS.append((name, name[:4], z, min(z, (Z - 1) - z), float(h),
                     bool(mt), (name, int(n)) in SH_IDS, Z))
    print(f"  {name:<16} Z={Z:>3}  GT {len(nid):>5}  joined {hit:>5}"
          + (f"  !! {miss} GT cells have no h_gt row" if miss else ""),
          flush=True)

DOM = np.array([r[1] for r in ROWS])
ZV = np.array([r[2] for r in ROWS], float)
ZEDGE = np.array([r[3] for r in ROWS], float)
HG = np.array([r[4] for r in ROWS], float)
MT = np.array([r[5] for r in ROWS], bool)
SH = np.array([r[6] for r in ROWS], bool)
print(f"\\n{len(ROWS)} cells joined; {int(MT.sum())} matched "
      f"({100.0 * MT.mean():.2f}%), {int(SH.sum())} shoulder", flush=True)
assert int(SH.sum()) == len(SH_IDS), (int(SH.sum()), len(SH_IDS))

print("\\n=== THE MEASUREMENT: h_gt against distance to the nearer z edge ===",
      flush=True)
print("  MATCHED cells only -- the detector FOUND these, so their profile is "
      "a\\n  clean readout of its response against depth.", flush=True)
print(f"  {'z from edge':<14} " + "  ".join(
    f"{d} n / h_gt p50 / p25" for d in ("6bba", "44b6")), flush=True)
ok = np.isfinite(HG)
edges = np.percentile(ZEDGE[ok], np.linspace(0, 100, NDEC + 1))
edges[-1] += 1e-9
for k in range(NDEC):
    m = ok & (ZEDGE >= edges[k]) & (ZEDGE < edges[k + 1])
    cells = []
    for dm in ("6bba", "44b6"):
        q = m & (DOM == dm) & MT
        if q.sum() < 5:
            cells.append(f"{int(q.sum()):>6}      --      --")
            continue
        cells.append(f"{int(q.sum()):>6}  {np.median(HG[q]):>6.3f}  "
                     f"{np.percentile(HG[q], 25):>6.3f}")
    print(f"  {edges[k]:>5.1f}-{edges[k + 1]:>5.1f} vox " + "   ".join(cells),
          flush=True)

print("\\n  the same rows, but the MISSED cells, for contrast:", flush=True)
for k in range(NDEC):
    m = ok & (ZEDGE >= edges[k]) & (ZEDGE < edges[k + 1])
    cells = []
    for dm in ("6bba", "44b6"):
        q = m & (DOM == dm) & ~MT
        cells.append(f"{int(q.sum()):>6}  "
                     + (f"{np.median(HG[q]):>6.3f}" if q.sum() >= 5 else
                        "    --"))
    print(f"  {edges[k]:>5.1f}-{edges[k + 1]:>5.1f} vox " + "   ".join(cells),
          flush=True)

print("\\n=== the same thing in one line per domain ===", flush=True)
for dm in ("6bba", "44b6"):
    q = ok & (DOM == dm) & MT
    e0 = q & (ZEDGE <= 4)
    e1 = q & (ZEDGE >= 12)
    if e0.sum() < 5 or e1.sum() < 5:
        continue
    a, b = float(np.median(HG[e0])), float(np.median(HG[e1]))
    print(f"  {dm}: matched cells within 4 planes of an edge have h_gt p50 "
          f"{a:.3f} (n={int(e0.sum())});\\n        12+ planes in, "
          f"{b:.3f} (n={int(e1.sum())}).  ratio {a / b:.3f}", flush=True)

print("\\n=== raw z, to separate 'edge' from 'deep' ===", flush=True)
print(f"  {'z (voxels)':<14} {'6bba n':>7} {'p50':>7}   {'44b6 n':>7} {'p50':>7}",
      flush=True)
zed = np.percentile(ZV[ok], np.linspace(0, 100, NDEC + 1))
zed[-1] += 1e-9
for k in range(NDEC):
    m = ok & (ZV >= zed[k]) & (ZV < zed[k + 1])
    cells = []
    for dm in ("6bba", "44b6"):
        q = m & (DOM == dm) & MT
        cells.append(f"{int(q.sum()):>7}  "
                     + (f"{np.median(HG[q]):>6.3f}" if q.sum() >= 5 else
                        "    --"))
    print(f"  {zed[k]:>5.1f}-{zed[k + 1]:>5.1f}    " + "   ".join(cells),
          flush=True)

print("\\nHOW TO READ IT.", flush=True)
print("  A SAG at both ends of the FIRST table -- among cells the detector "
      "found --\\n  means the response is systematically weaker near the "
      "border of the volume,\\n  which is what zero padding in a 3D conv net "
      "produces and which a\\n  z-calibration or reflect padding can undo. "
      "That is branch (b): fixable.", flush=True)
print("  A FLAT first table with only the missed cells weak means the "
      "attenuation\\n  story is wrong: the cells that fail are not failing "
      "because of where they\\n  are in the stack, and branch (a) -- "
      "physically half outside the imaged\\n  volume -- is the better "
      "reading. Nothing recovers those.", flush=True)
print("  CAVEAT that this run CANNOT settle: a real biological gradient "
      "(the embryo\\n  surface at one end of the stack, denser tissue at the "
      "other) would also\\n  bend this curve. A SYMMETRIC sag at BOTH ends is "
      "the signature that points\\n  at the network rather than at the "
      "sample, because tissue has no reason to\\n  be symmetric about the "
      "middle of the imaging volume.", flush=True)

print(f"\\nZPROFILE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("ZPROFILE OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "zprofile.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(BODY)
print("written", len(BODY))
