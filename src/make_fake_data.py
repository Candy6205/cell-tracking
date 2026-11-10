"""
Generate fake data with the same structure as the competition data, so the
scripts can be smoke-tested without Kaggle credentials.

The numbers it produces are meaningless -- this only checks that the plumbing
(zarr pyramid, geff layout, metadata fields) is read correctly.
"""

import shutil
from pathlib import Path

import numpy as np
import zarr

ROOT = Path("./fake_data")


def make_one(split, name, T=8, Z=16, Y=64, X=64, n_cells=40, n_annot=6,
             bright=1.0, noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    d = ROOT / split
    d.mkdir(parents=True, exist_ok=True)

    # ---- image: OME-Zarr pyramid, full resolution at "0"
    img = r.normal(0.1, noise, size=(T, Z, Y, X)).astype(np.float32)
    pos = np.stack([r.uniform(0, Z, n_cells), r.uniform(0, Y, n_cells),
                    r.uniform(0, X, n_cells)], axis=1)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    for t in range(T):
        pos = pos + r.normal(0, 0.6, pos.shape)
        pos[:, 0] = np.clip(pos[:, 0], 0, Z - 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, Y - 1)
        pos[:, 2] = np.clip(pos[:, 2], 0, X - 1)
        for (cz, cy, cx) in pos:
            g = np.exp(-(((zz - cz) / 1.2) ** 2 + ((yy - cy) / 3.0) ** 2
                         + ((xx - cx) / 3.0) ** 2) / 2)
            img[t] += (bright * g).astype(np.float32)

    zg = zarr.open_group(str(d / f"{name}.zarr"), mode="w")
    # Use blosc + uint16 + one chunk per frame, matching the competition data,
    # so the zarr-free RawZarrImage reader is exercised on a realistic layout.
    a = zg.create_array("0", shape=img.shape, dtype="f4", chunks=(1, Z, Y, X),
                        compressors=zarr.codecs.BloscCodec(cname="zstd"))
    a[:] = img
    zg.attrs["multiscales"] = [{
        "version": "0.5",
        "axes": [{"name": k} for k in "TZYX"],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
            ],
        }],
    }]
    zg.attrs["image_statistics"] = {
        "quantiles": {"0.0": float(img.min()), "0.99": float(np.percentile(img, 99)),
                      "1.0": float(img.max())}
    }

    if split == "test":
        return

    # ---- GEFF: a sparsely annotated tracking graph
    tracks = min(n_annot, n_cells)
    ids, ts, zs, ys, xs, edges = [], [], [], [], [], []
    nid = 0
    p0 = pos[:tracks].copy()
    prev = None
    for t in range(T):
        cur = []
        p0 = p0 + r.normal(0, 0.6, p0.shape)
        for k in range(tracks):
            ids.append(nid)
            ts.append(t)
            zs.append(float(np.clip(p0[k, 0], 0, Z - 1)))
            ys.append(float(np.clip(p0[k, 1], 0, Y - 1)))
            xs.append(float(np.clip(p0[k, 2], 0, X - 1)))
            cur.append(nid)
            nid += 1
        if prev is not None:
            edges.extend([a_, b_] for a_, b_ in zip(prev, cur))
        prev = cur
    if len(prev) >= 2 and len(edges) > 2:   # inject one division
        edges.append([edges[0][0], prev[1]])

    gg = zarr.open_group(str(d / f"{name}.geff"), mode="w")
    ng = gg.create_group("nodes")
    arr = ng.create_array("ids", shape=(len(ids),), dtype="i8")
    arr[:] = np.array(ids)
    pg = ng.create_group("props")
    for key, vals in (("t", ts), ("z", zs), ("y", ys), ("x", xs)):
        v = pg.create_group(key).create_array("values", shape=(len(vals),), dtype="f8")
        v[:] = np.array(vals, dtype=float)
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    ea = gg.create_group("edges").create_array("ids", shape=e.shape, dtype="i8")
    ea[:] = e

    gg.attrs["geff"] = {
        "geff_version": "1.1",
        "directed": True,
        "axes": [{"name": "t", "scale": 1.0}, {"name": "z", "scale": 1.625},
                 {"name": "y", "scale": 0.40625}, {"name": "x", "scale": 0.40625}],
        "extra": {"estimated_number_of_nodes": int(n_cells * T)},
    }


if __name__ == "__main__":
    if ROOT.exists():
        shutil.rmtree(ROOT)
    # Two "imaging batches" differing in brightness, noise and cell count,
    # so LOGO validation has something to leave out.
    for i in range(10):
        make_one("train", f"44b6_{i:04x}aa", n_cells=30 + 6 * i, n_annot=5,
                 bright=1.0, noise=0.05, seed=i)
    for i in range(10):
        make_one("train", f"6bba_{i:04x}bb", n_cells=15 + 4 * i, n_annot=8,
                 bright=2.2, noise=0.12, seed=100 + i)
    for i in range(4):
        make_one("test", f"9c1d_{i:04x}cc", n_cells=25 + 9 * i,
                 bright=1.6, noise=0.08, seed=200 + i)
    print("fake data ->", ROOT.resolve())
