"""
zebrahub_extract.py -- cut competition-shaped training crops out of Zebrahub.

Provenance
----------
CZ Biohub's own public bucket, under `royerlab`:

    https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/

Host Thibaut Goldsborough, 2026-08-13 on discussion /734330, asked directly
about the imaging data AND the `*_tracks.csv` annotations: "Yes you are free to
use the data and all resources in Zebrahub for this competition! There is no
overlap with the test set."

This is the source, not a competitor's re-upload. `kkunizaw/biohub-zh001r` is a
mirror whose labels could not be reconciled with its own node lists -- only
16.6% of its nodes landed on a non-zero voxel of its own target volume -- so it
is not used here.

Scale
-----
    ZSNS001.ome.zarr   (791, 1, 448, 2174, 2423) uint16
                       voxel z=1.24, y=x=0.439 um   ->  556 x 954 x 1064 um
    ZSNS001_tracks.csv 890 MB, ~18M rows
                       track_id, t, z, y, x, parent_track_id

Track coordinates are in MICROMETRES, not voxels: z reaches 509 while the
volume is only 448 voxels deep, and 448 x 1.24 = 556.

For comparison the whole competition training set holds 133,318 annotated nodes
across 199 datasets, and one dataset is (100, 64, 256, 256) at voxel
(1.625, 0.40625, 0.40625) -- a 104 um cube.

What this does
--------------
Crops in PHYSICAL units and resamples onto the competition's exact grid, so a
model trained here sees the same voxel geometry it will be asked about:

    104 um cube  ->  z 104/1.24 = 84 voxels, y=x 104/0.439 = 237 voxels
                 ->  resampled to (64, 256, 256)

`parent_track_id` is carried through, so lineage -- and therefore divisions --
survives the crop. Two track_ids sharing a parent is a division.

Needs internet. Run it as its own data-prep kernel, never as the submission.
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-21a"

BUCKET = ("https://public.czbiohub.org/royerlab/zebrahub/imaging/"
          "single-objective")
MOVIE = "ZSNS001"

# The competition's geometry, which the output must match exactly.
COMP_SHAPE = (64, 256, 256)                     # z, y, x
COMP_VOXEL = (1.625, 0.40625, 0.40625)          # um
BOX_UM = tuple(s * v for s, v in zip(COMP_SHAPE, COMP_VOXEL))   # 104 um cube

ZH_VOXEL = (1.24, 0.439, 0.439)                 # from the zarr's .zattrs

# Read ONE block per frame and cut a GRID x GRID tile of crops out of it.
#
# The naive version read each crop separately and cost 8.75 s/frame: the zarr
# chunks are (1, 1, 128, 362, 362) and a 104 um crop is 84 x 237 x 237, so
# every read pulled up to eight 33 MB chunks to use a sliver of each. Twenty
# crops of a hundred frames would have taken 4.9 h.
#
# A 3x3 tile spans 252 x 711 x 711 voxels, which touches roughly the same eight
# chunks -- so nine crops now cost one read instead of nine. About 18x less
# downloaded.
GRID = 3
SMOKE = False
N_FRAMES = 20 if SMOKE else 100
T0 = 200                    # mid-development; t runs 0..790

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "zarr<3", "fsspec", "aiohttp", "requests"],
                   capture_output=True, text=True)
print(f"pip exit {r.returncode}", flush=True)
if r.returncode:
    print(r.stderr[-1200:], flush=True)

import pandas as pd                                            # noqa: E402
import zarr                                                    # noqa: E402
from scipy.ndimage import zoom                                 # noqa: E402

# ---------------------------------------------------------------- tracks

OUT = pathlib.Path("/kaggle/working")
csv_path = OUT / f"{MOVIE}_tracks.csv"
if not csv_path.exists():
    t0 = time.time()
    r = subprocess.run(["curl", "-sS", "-o", str(csv_path),
                        f"{BUCKET}/{MOVIE}_tracks.csv"], capture_output=True,
                       text=True)
    if r.returncode:
        raise SystemExit(f"[degraded] track download failed: {r.stderr[-500:]}")
    print(f"tracks downloaded {csv_path.stat().st_size/2**20:.0f} MB in "
          f"{time.time()-t0:.0f}s", flush=True)

tr = pd.read_csv(csv_path)
print(f"tracks: {len(tr):,} rows, columns {list(tr.columns)}", flush=True)
print(f"  t     {tr.t.min():.0f} .. {tr.t.max():.0f}", flush=True)
for c in ("z", "y", "x"):
    print(f"  {c} um  {tr[c].min():.1f} .. {tr[c].max():.1f}", flush=True)
n_div = int((tr.parent_track_id != -1).groupby(tr.parent_track_id).size()
            .gt(1).sum())
print(f"  parents with >1 child (divisions): {n_div:,}", flush=True)
print(f"  competition train set, for scale: 133,318 nodes / 151 divisions",
      flush=True)

# ---- does a frame here mean the same thing as a frame there? ------------
# The competition's pooled displacement is p50 1.82 um between consecutive
# frames. If Zebrahub's interval differs, the motion statistics differ and the
# LINKER would be trained on the wrong thing -- detection would still be fine.
sub = tr[(tr.t >= T0) & (tr.t < T0 + 3)].sort_values(["track_id", "t"])
d = (sub.groupby("track_id")[["z", "y", "x"]].diff()
     .pow(2).sum(axis=1).pow(0.5).dropna())
if len(d):
    print(f"\nframe-to-frame displacement here: p50 {d.median():.2f} um, "
          f"p99 {d.quantile(0.99):.2f}", flush=True)
    print(f"  competition: p50 1.82, p99 6.71", flush=True)
    if not (0.5 < d.median() / 1.82 < 2.0):
        print("  [degraded] the time interval does not match the competition. "
              "Detection crops are still valid; anything trained on MOTION "
              "would be learning a different world.", flush=True)

# ---------------------------------------------------------------- image

store = f"{BUCKET}/{MOVIE}.ome.zarr"
root = zarr.open(store, mode="r")
arr = root["0"]
print(f"\nzarr {arr.shape} {arr.dtype} chunks={arr.chunks}", flush=True)

zb = [int(round(b / v)) for b, v in zip(BOX_UM, ZH_VOXEL)]      # 84, 237, 237
print(f"crop {BOX_UM[0]:.0f} um cube = {zb} voxels here -> resample to "
      f"{COMP_SHAPE}", flush=True)

# Place the tile where the cells are: the densest GRID x GRID window at t=T0.
f0 = tr[tr.t == T0]
if f0.empty:
    raise SystemExit(f"[degraded] no track rows at t={T0}")
gy = (f0.y // BOX_UM[1]).astype(int)
gx = (f0.x // BOX_UM[2]).astype(int)
cell = f0.groupby([gy, gx]).size()
best, best_n = None, -1
for iy in range(int(gy.min()), int(gy.max()) - GRID + 2):
    for ix in range(int(gx.min()), int(gx.max()) - GRID + 2):
        n = sum(int(cell.get((iy + a, ix + b), 0))
                for a in range(GRID) for b in range(GRID))
        if n > best_n:
            best, best_n = (iy, ix), n
iy0, ix0 = best
print(f"\ndensest {GRID}x{GRID} tile at t={T0}: y{iy0} x{ix0}, "
      f"{best_n} cells across it", flush=True)

CROPS = OUT / "crops"
CROPS.mkdir(exist_ok=True)
zc = int(f0.z.median() // BOX_UM[0])

# one block covering the whole tile
blk_um = (zc * BOX_UM[0], iy0 * BOX_UM[1], ix0 * BOX_UM[2])
blk_vx = [int(round(o / v)) for o, v in zip(blk_um, ZH_VOXEL)]
blk_sz = [zb[0], zb[1] * GRID, zb[2] * GRID]
print(f"block {blk_sz} voxels at {blk_vx}, "
      f"{np.prod(blk_sz) * 2 / 2**20:.0f} MB per frame", flush=True)

zoom_f = [c / b for c, b in zip(COMP_SHAPE, zb)]

# GRID^2 crops x N_FRAMES x 64x256x256 uint16 is 7.5 GB at GRID=3, N_FRAMES=100,
# on top of ~1 GB of tracks in pandas. Write through to disk instead of holding
# it: open_memmap gives a real .npy that np.load can mmap straight back.
from numpy.lib.format import open_memmap                       # noqa: E402

TMP = OUT / "tmp"
TMP.mkdir(exist_ok=True)
vols = {(a, b): open_memmap(TMP / f"v{a}{b}.npy", mode="w+",
                            dtype=np.uint16, shape=(N_FRAMES,) + COMP_SHAPE)
        for a in range(GRID) for b in range(GRID)}

t_start = time.time()
for i, t in enumerate(range(T0, T0 + N_FRAMES)):
    sl = tuple(slice(o, o + s) for o, s in zip(blk_vx, blk_sz))
    block = np.asarray(arr[t, 0][sl])
    if block.shape != tuple(blk_sz):
        pad = np.zeros(tuple(blk_sz), dtype=block.dtype)
        pad[:block.shape[0], :block.shape[1], :block.shape[2]] = block
        block = pad
        if i == 0:
            print(f"[degraded] the block runs off the volume edge and is "
                  f"zero-padded; crops on that side are partly empty",
                  flush=True)
    for a in range(GRID):
        for b in range(GRID):
            sub_ = block[:, a * zb[1]:(a + 1) * zb[1],
                         b * zb[2]:(b + 1) * zb[2]]
            vols[(a, b)][i] = zoom(sub_, zoom_f, order=1).astype(np.uint16)
    if (i + 1) % 10 == 0 or i == 0:
        el = time.time() - t_start
        print(f"  frame {i+1}/{N_FRAMES}  {el:.0f}s  "
              f"eta {el / (i + 1) * (N_FRAMES - i - 1):.0f}s", flush=True)

for (a, b), vol in vols.items():
    vol.flush()
    iy, ix = iy0 + a, ix0 + b
    o_um = (zc * BOX_UM[0], iy * BOX_UM[1], ix * BOX_UM[2])
    m = tr[(tr.t >= T0) & (tr.t < T0 + N_FRAMES)
           & (tr.z >= o_um[0]) & (tr.z < o_um[0] + BOX_UM[0])
           & (tr.y >= o_um[1]) & (tr.y < o_um[1] + BOX_UM[1])
           & (tr.x >= o_um[2]) & (tr.x < o_um[2] + BOX_UM[2])].copy()
    # onto the competition grid: physical offset removed, then its voxel size
    nodes = np.stack([
        (m.t - T0).to_numpy(),
        ((m.z - o_um[0]) / COMP_VOXEL[0]).to_numpy(),
        ((m.y - o_um[1]) / COMP_VOXEL[1]).to_numpy(),
        ((m.x - o_um[2]) / COMP_VOXEL[2]).to_numpy(),
    ], axis=1).astype(np.float32)

    name = f"{MOVIE}_z{zc}y{iy}x{ix}"
    np.savez_compressed(
        CROPS / f"{name}.npz", image=vol, nodes=nodes,
        track_id=m.track_id.to_numpy().astype(np.int64),
        parent_track_id=m.parent_track_id.to_numpy().astype(np.int64),
        origin_um=np.asarray(o_um, dtype=np.float64),
        voxel_um=np.asarray(COMP_VOXEL, dtype=np.float64))
    mb = (CROPS / f"{name}.npz").stat().st_size / 2 ** 20
    per_frame = len(nodes) / max(1, N_FRAMES)
    print(f"  {name}: {len(nodes):,} nodes ({per_frame:.0f}/frame), "
          f"{mb:.0f} MB", flush=True)

for p in TMP.glob("*.npy"):
    p.unlink()
TMP.rmdir()

tot = sum(p.stat().st_size for p in CROPS.glob("*.npz")) / 2 ** 20
print(f"\n{len(list(CROPS.glob('*.npz')))} crops, {tot:.0f} MB", flush=True)
print("Competition annotation density, for comparison: about 6.7 annotated "
      "nodes per frame.", flush=True)
csv_path.unlink(missing_ok=True)      # 890 MB, do not ship it in the output
print("EXTRACT DONE", flush=True)
