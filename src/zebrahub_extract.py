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
is not used here. The source itself passed an alignment spot-check on all five
movies on 2026-08-21 (`faustine649/zebrahub-align-spotcheck`, and progress.md
"Zebrahub verified as training data").

The movies, and their two traps (both verified 2026-08-21)
----------------------------------------------------------
    ZSNS001       (791, 1, 448, 2174, 2423)  tracks in UM,    21.7M rows
    ZSNS001_tail  (791, 1, 420, 1217, 1091)  tracks in UM,    cols renamed:
                                             ParentTrackID, NodeID, area
    ZSNS002       no tracks.csv -- unusable
    ZSNS003       (515, 1, 258, 1929, 2038)  tracks in VOXELS (integers)
    ZSNS004       (600, 1, 294, 1926, 2581)  tracks in VOXELS
    ZSNS005       (600, 1, 291, 1910, 2750)  tracks in VOXELS

"In voxels" means array indices, not lengths: convert per axis by the voxel
size before doing any geometry, or z is wrong by ~3x. Units are DETECTED here
(coordinate max vs the volume's physical extent), never assumed.

All five share voxel (1.24, 0.439, 0.439) um -- still read per movie from the
zarr attrs. `parent_track_id` is carried through, so lineage -- and therefore
divisions -- survives the crop. Two track_ids sharing a parent is a division.

Crop geometry
-------------
Crops in PHYSICAL units, resampled onto the competition's exact grid, so a
model trained here sees the same voxel geometry it will be asked about:

    104 um cube  ->  z 104/1.24 = 84 voxels, y=x 104/0.439 = 237 voxels
                 ->  resampled to (64, 256, 256)

Crop placement: cover the competition's density range
-----------------------------------------------------
The first run (2026-08-21a) maximised density and paid for it: 3 of 9 crops
were empty (tile hung off the embryo edge) and 2 were denser than anything in
the competition (1041 and 1211 cells/frame vs the competition's true 50-519).
Now each movie contributes up to 2 crops per density bin

    [30,80) [80,150) [150,250) [250,400) [400,650)   cells/frame at T0

chosen inside one z-slab. For movies whose chunks are small (ZSNS001, _tail:
362x362) the candidates are confined to the best WINDOW x WINDOW bin window so
one block read per frame stays cheap; the whole-plane-chunk movies (003/004/
005) pay for the full plane on any read, so their candidates are unconstrained.

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

SCRIPT_VERSION = "2026-08-21d"

BUCKET = ("https://public.czbiohub.org/royerlab/zebrahub/imaging/"
          "single-objective")
MOVIES = ["ZSNS001", "ZSNS001_tail", "ZSNS003", "ZSNS004", "ZSNS005"]

# The competition's geometry, which the output must match exactly.
COMP_SHAPE = (64, 256, 256)                     # z, y, x
COMP_VOXEL = (1.625, 0.40625, 0.40625)          # um
BOX_UM = tuple(s * v for s, v in zip(COMP_SHAPE, COMP_VOXEL))   # 104 um cube

FALLBACK_VOXEL = (1.24, 0.439, 0.439)           # every movie's, per spot-check

# Density bins to cover, cells/frame at T0. Competition true density is
# 50-519/frame; the low bin reaches under it on purpose -- sparse crops are
# the regime the first run never produced.
BINS = [(30, 80), (80, 150), (150, 250), (250, 400), (400, 650)]
K_PER_BIN = 2
WINDOW = 4                  # bin window for small-chunk movies

SMOKE = False
N_FRAMES = 20 if SMOKE else 100
T0_FRAC = 0.3               # T0 = this fraction into the movie
OUT_BUDGET_MB = 14000       # stop adding movies past this much output

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "zarr<3", "fsspec", "aiohttp", "requests"],
                   capture_output=True, text=True)
print(f"pip exit {r.returncode}", flush=True)
if r.returncode:
    print(r.stderr[-1200:], flush=True)

import pandas as pd                                            # noqa: E402
import zarr                                                    # noqa: E402
from numpy.lib.format import open_memmap                       # noqa: E402
from scipy.ndimage import zoom                                 # noqa: E402

OUT = pathlib.Path("/kaggle/working")
CROPS = OUT / "crops"
CROPS.mkdir(exist_ok=True)


def out_total_mb() -> float:
    return sum(p.stat().st_size for p in CROPS.glob("*.npz")) / 2 ** 20


def read_voxel(root) -> tuple:
    try:
        scale = root.attrs["multiscales"][0]["datasets"][0][
            "coordinateTransformations"][0]["scale"]
        return tuple(float(s) for s in scale[-3:])
    except Exception as e:
        print(f"[degraded] no voxel scale in zarr attrs ({e}); "
              f"assuming {FALLBACK_VOXEL}", flush=True)
        return FALLBACK_VOXEL


def pick_crops(cands: dict) -> list:
    """cands: {(iy, ix): cells}. Up to K_PER_BIN per density bin."""
    chosen = []
    for lo, hi in BINS:
        pool = sorted((p for p, n in cands.items()
                       if lo <= n < hi and p not in chosen),
                      key=lambda p: abs(cands[p] - (lo + hi) / 2))
        chosen += pool[:K_PER_BIN]
    return chosen


def coverage(cands: dict, chosen: list) -> tuple:
    """Score = (density bins covered, crops chosen, total cells)."""
    hit = sum(any(lo <= cands[p] < hi for p in chosen) for lo, hi in BINS)
    return (hit, len(chosen), sum(cands[p] for p in chosen))


def extract(movie: str) -> None:
    print(f"\n{'='*66}\n{movie}\n{'='*66}", flush=True)

    # ---------------------------------------------------------- tracks
    csv_path = OUT / f"{movie}_tracks.csv"
    if not csv_path.exists():
        t0 = time.time()
        r = subprocess.run(["curl", "-sS", "-o", str(csv_path),
                            f"{BUCKET}/{movie}_tracks.csv"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"track download failed: {r.stderr[-500:]}")
        print(f"tracks downloaded {csv_path.stat().st_size/2**20:.0f} MB "
              f"in {time.time()-t0:.0f}s", flush=True)

    tr = pd.read_csv(csv_path)
    tr = tr.rename(columns={"ParentTrackID": "parent_track_id",
                            "TrackID": "track_id"})
    # _tail marks "no parent" with NaN, not -1; astype(int64) turns NaN
    # into garbage (caught as a RuntimeWarning in the 2026-08-21d run).
    tr["parent_track_id"] = tr["parent_track_id"].fillna(-1)
    need = {"track_id", "t", "z", "y", "x", "parent_track_id"}
    missing = need - set(tr.columns)
    if missing:
        raise RuntimeError(f"columns missing after rename: {missing} "
                           f"(has {list(tr.columns)})")
    print(f"tracks: {len(tr):,} rows", flush=True)
    n_div = int((tr.parent_track_id != -1).groupby(tr.parent_track_id).size()
                .gt(1).sum())
    print(f"  parents with >1 child (divisions): {n_div:,}", flush=True)

    # ---------------------------------------------------------- image
    root = zarr.open(f"{BUCKET}/{movie}.ome.zarr", mode="r")
    arr = root["0"]
    voxel = read_voxel(root)
    shape = arr.shape[-3:]
    ext = tuple(s * v for s, v in zip(shape, voxel))
    print(f"zarr {arr.shape} {arr.dtype} chunks={arr.chunks} "
          f"voxel={voxel} um extent={tuple(round(e) for e in ext)} um",
          flush=True)

    # ------------------------------------------------- units, detected
    over_um = sum(tr[c].max() > ext[ax] * 1.05
                  for c, ax in zip(("z", "y", "x"), range(3)))
    units = "voxel" if over_um else "um"
    print(f"detected coordinate units: {units.upper()}", flush=True)
    for c, ax in zip(("z", "y", "x"), range(3)):
        tr[c + "_um"] = tr[c] * voxel[ax] if units == "voxel" else tr[c]

    T0 = int(T0_FRAC * (int(tr.t.max()) + 1))
    if T0 + N_FRAMES > int(tr.t.max()) + 1:
        T0 = max(0, int(tr.t.max()) + 1 - N_FRAMES)
        print(f"[degraded] movie shorter than expected, T0 moved to {T0}",
              flush=True)

    # ---- does a frame here mean the same thing as a frame there? -----
    # Competition pooled displacement is p50 1.82 um between consecutive
    # frames. dt == 1 only -- diffing across gaps once faked p50 = 0.
    sub = (tr[(tr.t >= T0) & (tr.t < T0 + 3)]
           .drop_duplicates(subset=["track_id", "t"])
           .sort_values(["track_id", "t"]))
    dt = sub.groupby("track_id").t.diff()
    dd = sub[["z_um", "y_um", "x_um"]].diff()
    d = np.sqrt((dd[dt == 1] ** 2).sum(axis=1))
    if len(d):
        print(f"frame-to-frame displacement: p50 {d.median():.2f} um, "
              f"p99 {d.quantile(0.99):.2f}  (competition: 1.82 / 6.71)",
              flush=True)
        if not (0.5 < d.median() / 1.82 < 2.0):
            print("  [degraded] time interval does not match the "
                  "competition. Detection crops stay valid; anything "
                  "trained on MOTION would learn a different world.",
                  flush=True)

    # ------------------------------- placement: cover the density bins
    f0 = tr[tr.t == T0]
    if f0.empty:
        raise RuntimeError(f"no track rows at t={T0}")
    gz = (f0.z_um // BOX_UM[0]).astype(int)
    gy = (f0.y_um // BOX_UM[1]).astype(int)
    gx = (f0.x_um // BOX_UM[2]).astype(int)
    cell3 = f0.groupby([gz, gy, gx]).size()

    zc = int(cell3.groupby(level=0).sum().idxmax())
    grid = {(iy, ix): int(n)
            for (iz, iy, ix), n in cell3.items() if iz == zc}
    print(f"z-slab {zc} ({zc*BOX_UM[0]:.0f}..{(zc+1)*BOX_UM[0]:.0f} um): "
          f"{sum(grid.values())} cells in {len(grid)} occupied bins at "
          f"t={T0}", flush=True)

    # whole-plane chunks make any read cost the plane -- no constraint.
    plane_frac = (min(arr.chunks[-2], shape[1]) *
                  min(arr.chunks[-1], shape[2])) / (shape[1] * shape[2])
    if plane_frac >= 0.5:
        chosen = pick_crops(grid)
    else:
        best, best_score = [], (-1,)
        ys = sorted({p[0] for p in grid})
        xs = sorted({p[1] for p in grid})
        for wy in range(min(ys), max(min(ys) + 1, max(ys) - WINDOW + 2)):
            for wx in range(min(xs), max(min(xs) + 1, max(xs) - WINDOW + 2)):
                sub_g = {p: n for p, n in grid.items()
                         if wy <= p[0] < wy + WINDOW
                         and wx <= p[1] < wx + WINDOW}
                ch = pick_crops(sub_g)
                sc = coverage(sub_g, ch)
                if sc > best_score:
                    best, best_score = ch, sc
        chosen = best
    if not chosen:
        print("[degraded] no bin-covering crops found; falling back to the "
              "densest bins", flush=True)
        chosen = sorted(grid, key=grid.get, reverse=True)[:len(BINS)]

    hit = [any(lo <= grid[p] < hi for p in chosen) for lo, hi in BINS]
    print(f"chosen {len(chosen)} crops, expected cells at t={T0}: "
          f"{sorted(grid[p] for p in chosen)}", flush=True)
    for (lo, hi), h in zip(BINS, hit):
        if not h:
            print(f"  [degraded] density bin [{lo},{hi}) not available in "
                  f"this movie's slab", flush=True)

    # ------------------------------------------ one bbox read per frame
    zb = [int(round(b / v)) for b, v in zip(BOX_UM, voxel)]   # 84,237,237
    zoom_f = [c / b for c, b in zip(COMP_SHAPE, zb)]
    zv0 = int(round(zc * BOX_UM[0] / voxel[0]))
    oy = {p: int(round(p[0] * BOX_UM[1] / voxel[1])) for p in chosen}
    ox = {p: int(round(p[1] * BOX_UM[2] / voxel[2])) for p in chosen}
    y0, x0 = min(oy.values()), min(ox.values())
    y1 = max(oy.values()) + zb[1]
    x1 = max(ox.values()) + zb[2]
    bbox_mb = zb[0] * (y1 - y0) * (x1 - x0) * 2 / 2 ** 20
    print(f"read bbox per frame: z[{zv0}:{zv0+zb[0]}] y[{y0}:{y1}] "
          f"x[{x0}:{x1}] = {bbox_mb:.0f} MB raw", flush=True)

    TMP = OUT / "tmp"
    TMP.mkdir(exist_ok=True)
    vols = {p: open_memmap(TMP / f"v{p[0]}_{p[1]}.npy", mode="w+",
                           dtype=np.uint16, shape=(N_FRAMES,) + COMP_SHAPE)
            for p in chosen}

    t_start = time.time()
    warned_pad = False
    for i, t in enumerate(range(T0, T0 + N_FRAMES)):
        # One indexing call so zarr fetches only the touched chunks.
        # arr[t, 0][...] would materialise the WHOLE 4-7 GB frame first --
        # that is what the 2026-08-21a run unknowingly did every frame.
        block = np.asarray(arr[t, 0, zv0:zv0 + zb[0], y0:y1, x0:x1])
        want = (zb[0], y1 - y0, x1 - x0)
        if block.shape != want:
            pad = np.zeros(want, dtype=block.dtype)
            pad[:block.shape[0], :block.shape[1], :block.shape[2]] = block
            block = pad
            if not warned_pad:
                warned_pad = True
                print("[degraded] the bbox runs off the volume edge and is "
                      "zero-padded; crops on that side are partly empty",
                      flush=True)
        for p in chosen:
            sub_ = block[:, oy[p] - y0:oy[p] - y0 + zb[1],
                         ox[p] - x0:ox[p] - x0 + zb[2]]
            vols[p][i] = zoom(sub_, zoom_f, order=1).astype(np.uint16)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            print(f"  frame {i+1}/{N_FRAMES}  {el:.0f}s  "
                  f"eta {el / (i + 1) * (N_FRAMES - i - 1):.0f}s", flush=True)

    # --------------------------------------------------- labels + write
    win = tr[(tr.t >= T0) & (tr.t < T0 + N_FRAMES)]
    for p in chosen:
        vols[p].flush()
        o_um = (zc * BOX_UM[0], p[0] * BOX_UM[1], p[1] * BOX_UM[2])
        m = win[(win.z_um >= o_um[0]) & (win.z_um < o_um[0] + BOX_UM[0])
                & (win.y_um >= o_um[1]) & (win.y_um < o_um[1] + BOX_UM[1])
                & (win.x_um >= o_um[2])
                & (win.x_um < o_um[2] + BOX_UM[2])].copy()
        nodes = np.stack([
            (m.t - T0).to_numpy(),
            ((m.z_um - o_um[0]) / COMP_VOXEL[0]).to_numpy(),
            ((m.y_um - o_um[1]) / COMP_VOXEL[1]).to_numpy(),
            ((m.x_um - o_um[2]) / COMP_VOXEL[2]).to_numpy(),
        ], axis=1).astype(np.float32)

        name = f"{movie}_z{zc}y{p[0]}x{p[1]}"
        np.savez_compressed(
            CROPS / f"{name}.npz", image=vols[p], nodes=nodes,
            track_id=m.track_id.to_numpy().astype(np.int64),
            parent_track_id=m.parent_track_id.to_numpy().astype(np.int64),
            origin_um=np.asarray(o_um, dtype=np.float64),
            voxel_um=np.asarray(COMP_VOXEL, dtype=np.float64),
            t0=np.int64(T0))
        mb = (CROPS / f"{name}.npz").stat().st_size / 2 ** 20
        print(f"  {name}: {len(nodes):,} nodes "
              f"({len(nodes)/max(1, N_FRAMES):.0f}/frame), {mb:.0f} MB",
              flush=True)

    for f in TMP.glob("*.npy"):
        f.unlink()
    TMP.rmdir()
    csv_path.unlink(missing_ok=True)      # do not ship 200-850 MB of CSV


failed = []
for movie in MOVIES:
    if out_total_mb() > OUT_BUDGET_MB:
        print(f"\n[degraded] output budget {OUT_BUDGET_MB} MB reached; "
              f"skipping {movie} and later movies", flush=True)
        break
    try:
        extract(movie)
    except Exception as e:
        failed.append(movie)
        print(f"[degraded] {movie} failed: {type(e).__name__}: {e}",
              flush=True)

n = len(list(CROPS.glob("*.npz")))
print(f"\n{n} crops, {out_total_mb():.0f} MB total", flush=True)
if failed:
    print(f"[degraded] failed movies: {failed}", flush=True)
print("Competition annotation density, for comparison: about 6.7 annotated "
      "nodes per frame (sparse GT); true density 50-519 cells/frame.",
      flush=True)
print("EXTRACT DONE", flush=True)
