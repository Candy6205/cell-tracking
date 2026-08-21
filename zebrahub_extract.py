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

SMOKE = True                # 2 crops, 20 frames -- flip off once it looks right
N_CROPS = 2 if SMOKE else 20
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

# Place crops where the cells are: the densest columns of the t=T0 slice.
f0 = tr[tr.t == T0]
if f0.empty:
    raise SystemExit(f"[degraded] no track rows at t={T0}")
gy = (f0.y // BOX_UM[1]).astype(int)
gx = (f0.x // BOX_UM[2]).astype(int)
dens = f0.groupby([gy, gx]).size().sort_values(ascending=False)
print(f"\ndensest boxes at t={T0} (cells per 104 um column):", flush=True)
for (iy, ix), n in dens.head(N_CROPS).items():
    print(f"  y{iy} x{ix}: {n}", flush=True)

CROPS = OUT / "crops"
CROPS.mkdir(exist_ok=True)
zc = int(f0.z.median() // BOX_UM[0])

for k, ((iy, ix), _n) in enumerate(dens.head(N_CROPS).items(), 1):
    t_start = time.time()
    o_um = (zc * BOX_UM[0], iy * BOX_UM[1], ix * BOX_UM[2])
    o_vx = [int(round(o / v)) for o, v in zip(o_um, ZH_VOXEL)]
    sl = tuple(slice(o, o + b) for o, b in zip(o_vx, zb))

    vols = np.empty((N_FRAMES,) + COMP_SHAPE, dtype=np.uint16)
    for i, t in enumerate(range(T0, T0 + N_FRAMES)):
        raw = np.asarray(arr[t, 0][sl])
        if raw.shape != tuple(zb):
            print(f"[degraded] crop {k} frame {t} came back {raw.shape}, "
                  f"expected {tuple(zb)} -- it runs off the volume edge and "
                  f"is padded with zeros", flush=True)
            pad = np.zeros(tuple(zb), dtype=raw.dtype)
            pad[:raw.shape[0], :raw.shape[1], :raw.shape[2]] = raw
            raw = pad
        f = [c / b for c, b in zip(COMP_SHAPE, zb)]
        vols[i] = zoom(raw, f, order=1).astype(np.uint16)

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
        CROPS / f"{name}.npz", image=vols, nodes=nodes,
        track_id=m.track_id.to_numpy().astype(np.int64),
        parent_track_id=m.parent_track_id.to_numpy().astype(np.int64),
        origin_um=np.asarray(o_um, dtype=np.float64),
        voxel_um=np.asarray(COMP_VOXEL, dtype=np.float64))
    mb = (CROPS / f"{name}.npz").stat().st_size / 2 ** 20
    per_frame = len(nodes) / max(1, N_FRAMES)
    print(f"[{k}/{N_CROPS}] {name}: {len(nodes):,} nodes "
          f"({per_frame:.0f}/frame), {mb:.0f} MB, {time.time()-t_start:.0f}s",
          flush=True)

tot = sum(p.stat().st_size for p in CROPS.glob("*.npz")) / 2 ** 20
print(f"\n{len(list(CROPS.glob('*.npz')))} crops, {tot:.0f} MB", flush=True)
print("Competition annotation density, for comparison: about 6.7 annotated "
      "nodes per frame.", flush=True)
csv_path.unlink(missing_ok=True)      # 890 MB, do not ship it in the output
print("EXTRACT DONE", flush=True)
