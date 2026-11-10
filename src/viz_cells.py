#!/usr/bin/env python
"""
viz_cells.py -- Biohub Cell Tracking During Development

Three modes:

  1) Structure probe (run this FIRST -- verifies the real zarr/geff layout
     matches what the rest of the code assumes)
       python viz_cells.py inspect
       python viz_cells.py inspect --dataset 44b6_0113de3b

  2) Visualization (thin Z-slab MIP + GT node overlay, saved as PNG)
       python viz_cells.py viz --dataset 44b6_0113de3b --frames 0,25,50,75

  3) Full survey (all train datasets -> CSV + pooled summary)
       python viz_cells.py survey --out survey_train.csv
       python viz_cells.py survey --limit 12          # smoke test first

In an interactive Kaggle notebook, install the optional extras:
       !pip install -q zarr geff

Neither is required. Submission notebooks have no internet and zarr is not in
the Kaggle image, so both readers fall back to reading zarr v3 chunk files
directly with blosc2 (which IS preinstalled), verified byte-identical:
  - open_image() -> RawZarrImage
  - read_geff()  -> _read_geff_from_files
Note that a .geff is ITSELF a zarr store: not needing the `geff` library is not
the same as not needing `zarr`. Conflating those broke a verify run once.

Call the functions directly in a notebook (avoids argparse eating -f kernel.json):
       import viz_cells
       viz_cells.run_inspect()
       viz_cells.run_survey(out="survey_train.csv", limit=12)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

# Bump this on every edit. Print it right after import on Kaggle: a stale
# Dataset version fails silently and looks like a code bug.
SCRIPT_VERSION = "2026-08-18f"

# Fallback voxel size (z, y, x) in micrometers -- mirrors DEFAULT_SCALE in the
# official io.py.
# WARNING: this is only a fallback. The real scale is stored PER DATASET in the
# OME-NGFF metadata:
#     multiscales[0].datasets[0].coordinateTransformations[0].scale[-3:]
# which is exactly what the official evaluate.py reads. Always go through
# read_scale(); never use this constant directly for distance math.
DEFAULT_SCALE = (1.625, 0.40625, 0.40625)

# Official node-matching threshold, in physical distance (micrometers).
MATCH_RADIUS_UM = 7.0

CANDIDATE_ROOTS = [
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    "/kaggle/input/biohub-cell-tracking-during-development",
    "./data",
]


# ---------------------------------------------------------------- paths

def find_root(root: str | None = None) -> Path:
    if root:
        p = Path(root)
        if not p.exists():
            raise FileNotFoundError(f"root does not exist: {p}")
        return p
    for c in CANDIDATE_ROOTS:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Competition data directory not found. Pass --root, or make sure the "
        f"competition data is attached on Kaggle. Tried: {CANDIDATE_ROOTS}"
    )


def list_datasets(root: Path, split: str = "train") -> list[str]:
    d = root / split
    if not d.exists():
        raise FileNotFoundError(f"{d} does not exist")
    return sorted(p.name[: -len(".zarr")] for p in d.glob("*.zarr"))


def zarr_path(root: Path, split: str, name: str) -> Path:
    return root / split / f"{name}.zarr"


def geff_path(root: Path, split: str, name: str) -> Path:
    return root / split / f"{name}.geff"


def group_of(name: str) -> str:
    """Dataset name prefix = one imaging batch. e.g. 44b6_0113de3b -> 44b6"""
    return name.split("_")[0]


# ------------------------------------------------------- warnings / fallbacks
#
# PROJECT RULE: degradation is never silent.
#
# Every fallback path -- a missing package, a missing metadata field, a default
# substituted for a real value -- must go through warn_once(). A silent
# fallback produces a plausible wrong number, which costs far more time to
# diagnose than a crash. warn_once prints the first occurrence and counts the
# rest, so this stays compatible with the "do not flood a Kaggle cell" rule.

_WARNED: dict = {}


def warn_once(key: str, msg: str):
    """Print a degradation warning the first time; count repeats silently."""
    n = _WARNED.get(key, 0)
    _WARNED[key] = n + 1
    if n == 0:
        print(f"[degraded] {msg}", flush=True)


def warning_summary(reset: bool = True):
    """Print how many times each degradation fired. Call at the end of a run."""
    if _WARNED:
        print("\n=== degradations this run ===")
        for k, n in sorted(_WARNED.items(), key=lambda kv: -kv[1]):
            print(f"  {n:6d}x  {k}")
        print("  ^ each of these silently substituted a fallback value; "
              "confirm none of them affects the numbers above.")
    if reset:
        _WARNED.clear()


def _progress(i: int, total: int, every: int, msg: str = ""):
    """
    Throttled progress logging.

    Kaggle notebook cells can die when a cell emits thousands of output lines,
    so never print one line per dataset. every=0 silences progress entirely;
    the final summary is always printed.
    """
    if every and (i == 1 or i == total or i % every == 0):
        print(f"  [{i}/{total}] {msg}", flush=True)


# ---------------------------------------------------------------- image I/O
#
# Kaggle submission notebooks run with INTERNET DISABLED, and zarr is not in
# the Kaggle image -- so `!pip install zarr` works while you are developing and
# then fails silently at submission time. The official getting-started notebook
# sidesteps this by reading the zarr v3 chunk files directly with blosc2 (which
# IS preinstalled). Everything below does the same: zarr is used when present,
# and a zarr-free reader takes over when it is not.

def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def store_attrs(path: Path) -> dict:
    """
    Read a zarr group's user attributes without requiring the zarr package.
    v3 keeps them under "attributes" in zarr.json; v2 used a separate .zattrs.
    """
    meta = _read_json(Path(path) / "zarr.json")
    if isinstance(meta, dict) and "attributes" in meta:
        return dict(meta["attributes"] or {})
    z2 = _read_json(Path(path) / ".zattrs")
    if isinstance(z2, dict):
        return z2
    try:
        import zarr

        return dict(zarr.open(str(path), mode="r").attrs)
    except Exception:
        return {}


def _decompress_chunk(raw: bytes, nbytes: int, codecs: list, where) -> bytes:
    """
    Decompress one zarr v3 chunk, dispatching on the codec declared in
    zarr.json rather than guessing. The competition data uses blosc, and blosc2
    is preinstalled on Kaggle even when zarr is not -- that combination is what
    makes an offline submission possible at all.
    """
    if len(raw) == nbytes and not any(
        c in codecs for c in ("blosc", "zstd", "gzip", "zlib", "lz4")
    ):
        return raw                                        # stored uncompressed

    if "blosc" in codecs:
        import blosc2

        return blosc2.decompress(raw)
    if "zstd" in codecs:
        try:
            import zstandard

            return zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=nbytes)
        except ImportError:
            pass
        try:
            import pyzstd

            return pyzstd.decompress(raw)
        except ImportError:
            pass
        try:
            from numcodecs import Zstd

            return Zstd().decode(raw)
        except ImportError:
            pass
        raise RuntimeError(
            f"{where}: chunk uses the zstd codec and none of zstandard / "
            f"pyzstd / numcodecs is available. This reader is only needed "
            f"OFFLINE; the submission path never touches it. For local "
            f"scoring, run with internet on and `pip install zarr`."
        )
    if "gzip" in codecs or "zlib" in codecs:
        import zlib

        return zlib.decompress(raw)
    if len(raw) == nbytes:
        warn_once("codec_guess",
                  f"{where}: codec chain {codecs} not recognised, but the chunk "
                  f"size matches the uncompressed size -- reading it as raw "
                  f"bytes. VERIFY THIS.")
        return raw
    raise RuntimeError(
        f"{where}: unhandled codec chain {codecs}. Install zarr, or add this "
        f"codec to _decompress_chunk."
    )


def read_zarr_array(path: Path) -> np.ndarray:
    """
    Read a whole zarr v3 array from disk without the zarr package.

    Intended for the SMALL arrays inside a .geff (node ids, coordinates, edge
    lists) -- it materializes everything, so do not point it at an image.
    Handles multi-chunk arrays of any rank; missing chunks become fill_value.
    """
    import itertools

    path = Path(path)
    meta = _read_json(path / "zarr.json")
    if meta is None:
        raise FileNotFoundError(f"no zarr.json in {path}")

    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    try:
        chunks = tuple(int(v) for v in
                       meta["chunk_grid"]["configuration"]["chunk_shape"])
    except Exception:
        chunks = shape
    codecs = [c.get("name") for c in (meta.get("codecs") or [])
              if isinstance(c, dict)]
    enc = meta.get("chunk_key_encoding", {}) or {}
    sep = (enc.get("configuration", {}) or {}).get("separator", "/")
    prefix = "c" if enc.get("name", "default") == "default" else ""

    fill = meta.get("fill_value", 0)
    out = np.full(shape, fill if fill is not None else 0, dtype=dtype)
    if 0 in shape:
        return out

    n_per_chunk = int(np.prod(chunks))
    grid = [-(-s // c) for s, c in zip(shape, chunks)]
    for idx in itertools.product(*[range(g) for g in grid]):
        key = sep.join(str(i) for i in idx)
        cpath = (path / prefix / key) if prefix else (path / key)
        if not cpath.exists():
            continue                       # unwritten chunk -> stays fill_value
        with open(cpath, "rb") as f:
            raw = f.read()
        buf = _decompress_chunk(raw, n_per_chunk * dtype.itemsize, codecs, cpath)
        block = np.frombuffer(buf, dtype=dtype).reshape(chunks)
        sl = tuple(slice(i * c, min((i + 1) * c, s))
                   for i, c, s in zip(idx, chunks, shape))
        trim = tuple(slice(0, x.stop - x.start) for x in sl)
        out[sl] = block[trim]
    return out


class RawZarrImage:
    """
    Minimal read-only view of a zarr v3 array, no zarr package required.

    Only supports the layout this competition actually uses: one chunk per
    timepoint spanning the full (Z, Y, X) volume, i.e. chunk shape
    (1, Z, Y, X), stored at <array>/c/<t>/0/0/0. Anything else raises, rather
    than silently returning wrong data.

    Supports img.shape, img.dtype, img[t] and img[t, a:b].
    """

    def __init__(self, array_path: Path):
        self.path = Path(array_path)
        meta = _read_json(self.path / "zarr.json")
        if meta is None:
            raise FileNotFoundError(f"no zarr.json in {self.path}")
        self.shape = tuple(int(v) for v in meta["shape"])
        self.dtype = np.dtype(meta["data_type"])
        try:
            self.chunks = tuple(
                int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]
            )
        except Exception:
            self.chunks = self.shape
        if len(self.shape) != 4 or self.chunks[0] != 1 or \
                tuple(self.chunks[1:]) != tuple(self.shape[1:]):
            raise ValueError(
                f"{self.path}: unsupported chunking {self.chunks} for shape "
                f"{self.shape}; the zarr-free reader assumes one chunk per frame"
            )
        self._n_spatial = int(np.prod(self.shape[1:]))
        self.codecs = [c.get("name") for c in (meta.get("codecs") or [])
                       if isinstance(c, dict)]

    def _decompress(self, raw: bytes) -> bytes:
        return _decompress_chunk(
            raw, self._n_spatial * self.dtype.itemsize, self.codecs, self.path)

    def _frame(self, t: int) -> np.ndarray:
        p = self.path / "c" / str(int(t)) / "0" / "0" / "0"
        with open(p, "rb") as f:
            raw = f.read()
        buf = self._decompress(raw)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            t, rest = key[0], key[1:]
        else:
            t, rest = key, ()
        if not np.isscalar(t):
            raise TypeError("RawZarrImage only supports a scalar time index")
        vol = self._frame(int(t))
        return vol[rest] if rest else vol


def read_scale(path: Path) -> tuple:
    """
    Read this dataset's voxel size (z, y, x) in um from the OME-NGFF metadata.
    Mirrors the official io.py::_parse_scale:
        multiscales[0].datasets[0].coordinateTransformations[0].scale[-3:]
    Real metadata looks like: "scale": [1.0, 1.625, 0.40625, 0.40625]
    (the leading 1.0 is the T axis, hence [-3:]).
    Falls back to DEFAULT_SCALE if unavailable.
    """
    try:
        attrs = store_attrs(path)
        tf = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tf.get("type") == "scale":
            return tuple(float(v) for v in tf["scale"][-3:])
    except Exception:
        pass
    warn_once("scale_default",
              f"no voxel scale in {Path(path).name} metadata -> falling back to "
              f"DEFAULT_SCALE {DEFAULT_SCALE}; every physical distance below "
              f"assumes that value")
    return DEFAULT_SCALE


def read_quantiles(path: Path) -> dict:
    """
    Precomputed intensity quantiles stored in the zarr attrs by the organizers.
    The official io.py uses these for normalization instead of scanning the
    whole volume -- use them and skip a full pass over every dataset.
    """
    try:
        attrs = store_attrs(path)
        return {
            float(k): float(v)
            for k, v in attrs.get("image_statistics", {}).get("quantiles", {}).items()
        }
    except Exception:
        warn_once("quantiles_missing",
                  f"no precomputed quantiles in {Path(path).name}; anything "
                  f"relying on them will compute its own")
        return {}


def _multiscale_level(path: Path) -> str:
    """Name of the full-resolution level in the OME-NGFF pyramid."""
    try:
        return str(store_attrs(path)["multiscales"][0]["datasets"][0]["path"])
    except Exception:
        warn_once("multiscale_default",
                  f"no multiscales metadata in {Path(path).name} -> assuming "
                  f'the full-resolution level is "0"')
        return "0"


def open_image(path: Path):
    """
    Open an OME-Zarr and return the highest-resolution array, shape (T, Z, Y, X).

    Returns a lazy array -- do NOT materialize it. Falls back to the zarr-free
    RawZarrImage when the zarr package is unavailable, which is the case in a
    Kaggle submission run (no internet, so pip install cannot help).
    """
    path = Path(path)
    try:
        import zarr
    except ImportError:
        warn_once("no_zarr",
                  "zarr not installed -> using the zarr-free RawZarrImage "
                  "reader (expected inside a Kaggle submission run; verified "
                  "byte-identical, but only for one-chunk-per-frame layouts)")
        return RawZarrImage(path / _multiscale_level(path))

    z = zarr.open(str(path), mode="r")

    # OME-Zarr is a multiscale pyramid: zarr.open gives a group, not an array.
    if hasattr(z, "shape"):
        return z

    try:
        ms = z.attrs.get("multiscales", None)
        if ms:
            return z[ms[0]["datasets"][0]["path"]]
    except Exception:
        pass

    for key in ("0", "s0", "scale0"):
        try:
            arr = z[key]
            if hasattr(arr, "shape"):
                return arr
        except Exception:
            continue

    for k in list(z.array_keys()):
        return z[k]

    raise ValueError(f"no image array found in {path}, keys={list(z.group_keys())}")


# ---------------------------------------------------------------- GEFF I/O

def _walk_find_geff_group(g, depth: int = 0):
    """Recursively locate the group holding nodes/ and edges/."""
    try:
        keys = set(g.group_keys())
    except Exception:
        return None
    if "nodes" in keys:
        return g
    if depth >= 3:
        return None
    for k in sorted(keys):
        found = _walk_find_geff_group(g[k], depth + 1)
        if found is not None:
            return found
    return None


def _read_prop(props_group, name: str):
    """A GEFF prop may be props/<name>/values or just props/<name>."""
    try:
        node = props_group[name]
    except Exception:
        return None
    if hasattr(node, "shape"):
        return np.asarray(node[:])
    try:
        return np.asarray(node["values"][:])
    except Exception:
        return None


def _geff_props_dir(nodes_dir: Path) -> Path:
    return nodes_dir / "props"


def _read_geff_from_files(path: Path) -> dict:
    """
    Read a .geff without the zarr package, straight off the filesystem.

    A .geff IS a zarr store, so "no geff library needed" was never the same as
    "no zarr needed" -- that distinction cost a failed verify run. Group
    structure is just directories; each leaf array is read by read_zarr_array.
    """
    path = Path(path)
    out: dict = {"extra": {}, "axes": None, "scale": None}

    attrs = store_attrs(path)
    gm = attrs.get("geff", attrs)
    out["extra"] = dict(gm.get("extra", {}) or {})
    if "axes" in gm:
        names = [a.get("name") for a in gm["axes"]]
        out["axes"] = names
        m = {a.get("name"): a.get("scale") for a in gm["axes"]
             if a.get("scale") is not None}
        if all(k in m for k in ("z", "y", "x")):
            out["scale"] = (float(m["z"]), float(m["y"]), float(m["x"]))

    # locate the group holding nodes/ -- top level, or one level down
    root = path
    if not (root / "nodes").is_dir():
        for child in sorted(p for p in path.iterdir() if p.is_dir()):
            if (child / "nodes").is_dir():
                root = child
                break
    nodes_dir = root / "nodes"
    if not nodes_dir.is_dir():
        raise FileNotFoundError(f"no nodes/ directory under {path}")

    ids_dir = nodes_dir / "ids"
    node_ids = read_zarr_array(ids_dir) if ids_dir.is_dir() else None

    props = _geff_props_dir(nodes_dir)
    coords = {}
    for k in ("t", "z", "y", "x"):
        d = props / k
        if (d / "values").is_dir():
            coords[k] = np.asarray(read_zarr_array(d / "values"), dtype=float)
        elif (d / "zarr.json").exists():
            coords[k] = np.asarray(read_zarr_array(d), dtype=float)

    missing = [k for k in ("t", "z", "y", "x") if k not in coords]
    if missing:
        raise ValueError(f"{path}: missing node props {missing} (zarr-free reader)")

    n = len(coords["t"])
    out["node_ids"] = np.asarray(node_ids if node_ids is not None else np.arange(n))
    out.update({k: coords[k] for k in ("t", "z", "y", "x")})

    edges = np.zeros((0, 2), dtype=np.int64)
    eids = root / "edges" / "ids"
    if eids.is_dir():
        e = np.asarray(read_zarr_array(eids))
        edges = e if (e.ndim == 2 and e.shape[1] == 2) else e.reshape(-1, 2)
    out["edges"] = edges.astype(np.int64)
    return out


def read_geff(path: Path) -> dict:
    """
    Read a GEFF tracking graph.

    Returns:
        node_ids : (N,) int
        t,z,y,x  : (N,) float  -- VOXEL coordinates (stored as int64 in the data)
        edges    : (E, 2) int  -- (source_id, target_id)
        extra    : dict        -- geff metadata "extra", holds
                                  estimated_number_of_nodes
        axes     : list        -- axis names, to sanity-check ordering
        scale    : (z, y, x)   -- per-axis scale from the geff metadata, or None
    """
    try:
        import zarr
    except ImportError:
        warn_once("no_zarr_geff",
                  "zarr not installed -> reading the .geff with the zarr-free "
                  "filesystem reader. A .geff is itself a zarr store, so this "
                  "path is needed even though the geff LIBRARY is optional.")
        return _read_geff_from_files(path)

    out: dict = {"extra": {}, "axes": None, "scale": None}

    def _axes_scale(names, scales):
        m = {n: s for n, s in zip(names, scales) if n and s is not None}
        if all(k in m for k in ("z", "y", "x")):
            return (float(m["z"]), float(m["y"]), float(m["x"]))
        return None

    # Metadata: prefer the official geff library, fall back to raw zattrs.
    try:
        import geff  # noqa

        md = geff.GeffMetadata.read(str(path))
        out["extra"] = dict(getattr(md, "extra", {}) or {})
        axes = getattr(md, "axes", None)
        if axes is not None:
            out["axes"] = [getattr(a, "name", str(a)) for a in axes]
            out["scale"] = _axes_scale(
                out["axes"], [getattr(a, "scale", None) for a in axes]
            )
    except Exception:
        pass

    root = zarr.open(str(path), mode="r")
    try:
        attrs = dict(root.attrs)
        gm = attrs.get("geff", attrs)
        if not out["extra"]:
            out["extra"] = dict(gm.get("extra", {}) or {})
        if "axes" in gm:
            names = [a.get("name") for a in gm["axes"]]
            if out["axes"] is None:
                out["axes"] = names
            if out["scale"] is None:
                out["scale"] = _axes_scale(names, [a.get("scale") for a in gm["axes"]])
    except Exception:
        pass

    g = _walk_find_geff_group(root)
    if g is None:
        raise ValueError(f"no nodes/edges group found in {path}")

    nodes = g["nodes"]
    keys = list(nodes.array_keys()) + list(nodes.group_keys())
    node_ids = np.asarray(nodes["ids"][:]) if "ids" in keys else None

    nprops = nodes["props"]
    coords = {}
    for k in ("t", "z", "y", "x"):
        v = _read_prop(nprops, k)
        if v is not None:
            coords[k] = np.asarray(v, dtype=float)

    if len(coords) < 4:
        # Positions might be packed into a single (N, 4) prop.
        for cand in ("position", "pos", "coords", "location"):
            v = _read_prop(nprops, cand)
            if v is not None and v.ndim == 2 and v.shape[1] >= 4:
                order = out["axes"] or ["t", "z", "y", "x"]
                for i, nm in enumerate(order[:4]):
                    coords[nm] = np.asarray(v[:, i], dtype=float)
                break

    missing = [k for k in ("t", "z", "y", "x") if k not in coords]
    if missing:
        raise ValueError(
            f"{path}: missing node props {missing}. Found: "
            f"{list(nprops.group_keys()) + list(nprops.array_keys())}"
        )

    n = len(coords["t"])
    out["node_ids"] = np.asarray(node_ids if node_ids is not None else np.arange(n))
    out.update({k: coords[k] for k in ("t", "z", "y", "x")})

    edges = np.zeros((0, 2), dtype=np.int64)
    try:
        e = np.asarray(g["edges"]["ids"][:])
        if e.ndim == 2 and e.shape[1] == 2:
            edges = e
        elif e.ndim == 1:
            edges = e.reshape(-1, 2)
    except Exception:
        pass
    out["edges"] = edges.astype(np.int64)

    return out


# ---------------------------------------------------------------- stats

def graph_stats(gt: dict, scale: tuple | None = None) -> dict:
    """
    Compute the statistics we care about from one tracking graph.
    scale: (z, y, x) um/voxel. Defaults to the geff's own scale, then
    DEFAULT_SCALE.
    """
    if scale is None:
        scale = gt.get("scale")
        if scale is None:
            warn_once("graph_scale_default",
                      "graph has no voxel scale and none was passed -> using "
                      f"DEFAULT_SCALE {DEFAULT_SCALE} for displacement stats")
            scale = DEFAULT_SCALE
    t = gt["t"]
    ids = gt["node_ids"]
    edges = gt["edges"]
    n_nodes = len(ids)

    idx = {int(v): i for i, v in enumerate(ids)}

    st: dict = {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edges)),
        "t_min": float(t.min()) if n_nodes else np.nan,
        "t_max": float(t.max()) if n_nodes else np.nan,
    }

    # A node with out-degree >= 2 is a division.
    if len(edges):
        _, cnt = np.unique(edges[:, 0], return_counts=True)
        st["n_divisions"] = int((cnt >= 2).sum())
    else:
        st["n_divisions"] = 0

    # Frame-to-frame displacement in physical distance (um).
    disp = np.array([], dtype=float)
    if len(edges):
        pz, py, px = gt["z"], gt["y"], gt["x"]
        rows_s = np.array([idx.get(int(a), -1) for a in edges[:, 0]])
        rows_t = np.array([idx.get(int(b), -1) for b in edges[:, 1]])
        ok = (rows_s >= 0) & (rows_t >= 0)
        rs, rt = rows_s[ok], rows_t[ok]
        dz = (pz[rt] - pz[rs]) * scale[0]
        dy = (py[rt] - py[rs]) * scale[1]
        dx = (px[rt] - px[rs]) * scale[2]
        disp = np.sqrt(dz**2 + dy**2 + dx**2)

    st["_disp"] = disp
    for p in (50, 90, 95, 99, 100):
        st[f"disp_p{p}"] = float(np.percentile(disp, p)) if len(disp) else np.nan

    span = (st["t_max"] - st["t_min"] + 1) if n_nodes else np.nan
    st["annot_per_frame"] = float(n_nodes / span) if span and span > 0 else np.nan
    return st


# ---------------------------------------------------------------- inspect

def run_inspect(root=None, split="train", dataset=None, attrs_chars=400):
    r = find_root(root)
    names = list_datasets(r, split)
    print(f"[root] {r}")
    print(f"[{split}] {len(names)} datasets")
    print(f"  first 5    : {names[:5]}")
    groups: dict = {}
    for n in names:
        groups[group_of(n)] = groups.get(group_of(n), 0) + 1
    print(f"  by prefix  : {groups}")

    name = dataset or names[0]
    print(f"\n=== inspecting {name} ===")

    zp = zarr_path(r, split, name)
    import zarr

    z = zarr.open(str(zp), mode="r")
    print(f"[zarr] {zp}")
    try:
        print(f"  group_keys : {list(z.group_keys())}")
        print(f"  array_keys : {list(z.array_keys())}")
        print(f"  attrs      : "
              f"{json.dumps(dict(z.attrs), default=str)[:attrs_chars]}")
    except Exception as e:
        print(f"  (top level is not a group: {e})")

    img = open_image(zp)
    print(f"  image shape: {img.shape}  dtype={img.dtype}")
    try:
        print(f"  chunks     : {img.chunks}")
    except Exception:
        pass
    sc = read_scale(zp)
    T, Z, Y, X = img.shape[-4:]
    same = tuple(np.round(sc, 6)) == DEFAULT_SCALE
    print(f"  voxel scale: {sc} um  (default {DEFAULT_SCALE}, "
          f"{'match' if same else 'MISMATCH -- per-dataset scale matters here'})")
    print(f"  physical   : Z {Z*sc[0]:.0f} um x Y {Y*sc[1]:.0f} um x "
          f"X {X*sc[2]:.0f} um, T={T} frames")
    q = read_quantiles(zp)
    if q:
        print(f"  quantiles  : {q}")
        print("               ^ precomputed by the organizers; use these for "
              "normalization instead of scanning the volume")

    gp = geff_path(r, split, name)
    if not gp.exists():
        print(f"\n[geff] absent (expected for the {split} split)")
        return

    print(f"\n[geff] {gp}")
    g0 = zarr.open(str(gp), mode="r")
    print(f"  group_keys : {list(g0.group_keys())}")
    print(f"  attrs      : {json.dumps(dict(g0.attrs), default=str)[:attrs_chars]}")
    gt = read_geff(gp)
    st = graph_stats(gt, sc)
    print(f"  axes       : {gt['axes']}   geff scale: {gt['scale']}")
    print(f"  nodes/edges: {st['n_nodes']} / {st['n_edges']}")
    print(f"  t range    : {st['t_min']:.0f} .. {st['t_max']:.0f}")
    print(f"  divisions  : {st['n_divisions']}")
    print(f"  annot dens : {st['annot_per_frame']:.2f} cells/frame")
    print(f"  displacement: p50={st['disp_p50']:.2f}  p99={st['disp_p99']:.2f}  "
          f"p100={st['disp_p100']:.2f} um   (match threshold {MATCH_RADIUS_UM} um)")
    print(f"  extra      : {gt['extra']}")
    est = gt["extra"].get("estimated_number_of_nodes")
    if est:
        print(f"  >> N_true={est}, GT annotates only "
              f"{100*st['n_nodes']/est:.2f}% of it; "
              f"true density ~ {est/max(T,1):.0f} cells/frame")


# ---------------------------------------------------------------- viz

def _slab_mip(img, t: int, z_center: int | None = None, slab: int = 9,
              roi: tuple | None = None):
    """Max-intensity projection over a thin Z slab centered on z_center."""
    Z = img.shape[1]
    zc = Z // 2 if z_center is None else int(z_center)
    z0 = max(0, zc - slab // 2)
    z1 = min(Z, z0 + slab)
    z0 = max(0, z1 - slab)
    sub = np.asarray(img[t, z0:z1])
    if roi:
        y0, y1, x0, x1 = roi
        sub = sub[:, y0:y1, x0:x1]
    return sub.max(axis=0), (z0, z1)


def run_viz(root=None, split="train", dataset=None, frames="0,25,50,75",
            slab=9, roi=None, out="viz_cells.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = find_root(root)
    names = list_datasets(r, split)
    name = dataset or names[0]
    zp = zarr_path(r, split, name)
    img = open_image(zp)
    sc = read_scale(zp)
    T = img.shape[0]

    fr = ([int(v) for v in frames.split(",") if v.strip()]
          if isinstance(frames, str) else list(frames))
    fr = [f for f in fr if 0 <= f < T] or list(np.linspace(0, T - 1, 4).astype(int))

    gt = None
    gp = geff_path(r, split, name)
    if gp.exists():
        try:
            gt = read_geff(gp)
        except Exception as e:
            warnings.warn(f"failed to read geff: {e}")

    roi_t = None
    if roi:
        roi_t = tuple(int(v) for v in (roi.split(",") if isinstance(roi, str) else roi))

    n = len(fr)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), squeeze=False)
    for ax, t in zip(axes[0], fr):
        mip, (z0, z1) = _slab_mip(img, t, slab=slab, roi=roi_t)
        vmin, vmax = np.percentile(mip, [1, 99.7])
        # Microscopy images MUST use nearest: default smoothing turns nuclei
        # into out-of-focus blobs at these pixel sizes.
        ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        title = f"t={t}  z[{z0}:{z1}]"
        if gt is not None:
            m = gt["t"].astype(int) == t
            zz, yy, xx = gt["z"][m], gt["y"][m], gt["x"][m]
            keep = (zz >= z0) & (zz < z1)
            yv, xv = yy[keep], xx[keep]
            if roi_t:
                y0, y1, x0, x1 = roi_t
                inb = (yv >= y0) & (yv < y1) & (xv >= x0) & (xv < x1)
                yv, xv = yv[inb] - y0, xv[inb] - x0
            ax.scatter(xv, yv, s=90, facecolors="none", edgecolors="red",
                       linewidths=1.2)
            title += f"  GT in slab: {len(yv)}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{name}   slab={slab} planes = {slab*sc[0]:.0f} um", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")

    if gt is not None:
        st = graph_stats(gt, sc)
        print(f"[{name}] nodes={st['n_nodes']} edges={st['n_edges']} "
              f"div={st['n_divisions']} annot={st['annot_per_frame']:.2f}/frame")
        print(f"  displacement p50={st['disp_p50']:.2f} p99={st['disp_p99']:.2f} "
              f"p100={st['disp_p100']:.2f} um -> set the gating radius above p99")
    return out


def run_gif(root=None, split="train", dataset=None, out=None, slab=9,
            roi=None, step=1, fps=8, max_frames=None, overlay=True):
    """
    Write an animated GIF of a movie: thin Z-slab MIP per frame, GT nodes
    circled.

    Static frames are fine for checking focus or brightness, but motion is what
    tells you whether cells are crowded, whether they jitter, and where the
    tracker would struggle. Use --roi on dense datasets; a full 256x256 view of
    700 cells/frame is unreadable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    r = find_root(root)
    name = dataset or list_datasets(r, split)[0]
    zp = zarr_path(r, split, name)
    img = open_image(zp)
    sc = read_scale(zp)
    T = img.shape[0] if max_frames is None else min(img.shape[0], int(max_frames))
    out = out or f"{name}.gif"

    gt = None
    gp = geff_path(r, split, name)
    if overlay and gp.exists():
        try:
            gt = read_geff(gp)
        except Exception as e:
            warn_once("gif_no_gt", f"could not read GT for the overlay: {e}")

    roi_t = None
    if roi:
        roi_t = tuple(int(v) for v in (roi.split(",") if isinstance(roi, str) else roi))

    # One global intensity range, otherwise per-frame autoscaling makes the
    # movie flicker and hides real brightness drift (photobleaching).
    probe, _ = _slab_mip(img, 0, slab=slab, roi=roi_t)
    vmin, vmax = np.percentile(probe, [1, 99.7])

    frames = []
    for t in range(0, T, max(1, int(step))):
        mip, (z0, z1) = _slab_mip(img, t, slab=slab, roi=roi_t)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if gt is not None:
            m = gt["t"].astype(int) == t
            zz, yy, xx = gt["z"][m], gt["y"][m], gt["x"][m]
            keep = (zz >= z0) & (zz < z1)
            yv, xv = yy[keep], xx[keep]
            if roi_t:
                y0, y1, x0, x1 = roi_t
                inb = (yv >= y0) & (yv < y1) & (xv >= x0) & (xv < x1)
                yv, xv = yv[inb] - y0, xv[inb] - x0
            ax.scatter(xv, yv, s=70, facecolors="none", edgecolors="red",
                       linewidths=1.0)
        ax.set_title(f"{name}  t={t}", fontsize=9)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
        plt.close(fig)

    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"[saved] {out}  ({len(frames)} frames, {slab} planes "
          f"= {slab*sc[0]:.0f} um slab)")
    return out


# ---------------------------------------------------------------- survey

def run_survey(root=None, split="train", out="survey_train.csv", limit=None,
               log_every=25):
    """
    Walk every dataset, write a stats table, print a pooled summary.
    log_every: print a progress line every N datasets (0 = silent).
    Errors are always printed -- they are rare and you need to see them.
    """
    import csv

    r = find_root(root)
    names = list_datasets(r, split)
    if limit:
        names = names[: int(limit)]
        print(f"[warning] --limit {limit}: only the FIRST {limit} datasets "
              f"(of {len(list_datasets(r, split))}) are surveyed. Because names "
              f"are sorted, this covers only the earliest prefix group and the "
              f"pooled numbers below are NOT representative.")

    rows = []
    all_disp = []
    for i, name in enumerate(names, 1):
        row = {"dataset": name, "group": group_of(name)}
        sc = DEFAULT_SCALE
        T = np.nan
        try:
            zp = zarr_path(r, split, name)
            img = open_image(zp)
            sc = read_scale(zp)
            T, Z, Y, X = img.shape[-4:]
            row.update({"T": T, "Z": Z, "Y": Y, "X": X,
                        "scale_z": sc[0], "scale_y": sc[1], "scale_x": sc[2]})
        except Exception as e:
            row["img_error"] = str(e)[:120]
            print(f"  [image error] {name}: {e}", flush=True)

        gp = geff_path(r, split, name)
        if gp.exists():
            try:
                gt = read_geff(gp)
                st = graph_stats(gt, sc)
                all_disp.append(st.pop("_disp"))
                row.update({k: v for k, v in st.items() if not k.startswith("_")})
                est = gt["extra"].get("estimated_number_of_nodes")
                row["est_n_nodes"] = est
                if est and st["n_nodes"]:
                    row["annot_ratio"] = st["n_nodes"] / float(est)
                if est and T == T and T:
                    row["true_per_frame"] = float(est) / T
            except Exception as e:
                row["gt_error"] = str(e)[:120]
                print(f"  [geff error] {name}: {e}", flush=True)

        _progress(i, len(names), log_every, name)
        rows.append(row)

    cols = [
        "dataset", "group", "T", "Z", "Y", "X",
        "scale_z", "scale_y", "scale_x",
        "n_nodes", "n_edges", "n_divisions", "t_min", "t_max",
        "annot_per_frame", "est_n_nodes", "annot_ratio", "true_per_frame",
        "disp_p50", "disp_p90", "disp_p95", "disp_p99", "disp_p100",
        "img_error", "gt_error",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[saved] {out}  ({len(rows)} rows)")

    def col(key):
        v = [r_[key] for r_ in rows
             if isinstance(r_.get(key), (int, float))
             and not (isinstance(r_[key], float) and np.isnan(r_[key]))]
        return np.array(v, dtype=float)

    print("\n=== pooled summary ===")
    print(f"datasets          : {len(rows)}")
    scales = sorted({(r_.get("scale_z"), r_.get("scale_y"), r_.get("scale_x"))
                     for r_ in rows if r_.get("scale_z")})
    if len(scales) == 1:
        print(f"voxel scale       : {scales[0]} um (identical across datasets)")
    else:
        print(f"voxel scale       : NOT uniform, {len(scales)} variants "
              f"{scales[:5]} -> convert per dataset for any physical quantity")
    print(f"GT nodes total    : {int(col('n_nodes').sum())}")
    nedge = int(col("n_edges").sum())
    print(f"GT edges total    : {nedge}   <- denominator of the edge Jaccard")
    ndiv = int(col("n_divisions").sum())
    print(f"divisions total   : {ndiv}"
          + (f"  (1 per {nedge//max(ndiv,1)} edges)" if ndiv else ""))
    est = col("est_n_nodes")
    if len(est):
        print(f"N_true total      : {int(est.sum())}")
        print(f"overall annot rate: {100*col('n_nodes').sum()/est.sum():.2f}%")
    if all_disp:
        d = np.concatenate([x for x in all_disp if len(x)])
        if len(d):
            print(f"pooled displacement: p50={np.percentile(d,50):.2f} "
                  f"p99={np.percentile(d,99):.2f} p100={d.max():.2f} um "
                  f"(n={len(d)})  <- pick the gating radius between p99 and p100")

    print("\n=== by imaging group (name prefix) ===")
    tot_w = sum(r_.get("n_edges", 0) or 0 for r_ in rows)
    for g in sorted({r_["group"] for r_ in rows}):
        sub = [r_ for r_ in rows if r_["group"] == g]
        e = sum(r_.get("n_edges", 0) or 0 for r_ in sub)
        tpf = [r_["true_per_frame"] for r_ in sub if r_.get("true_per_frame")]
        ar = [r_["annot_ratio"] for r_ in sub if r_.get("annot_ratio")]
        parts = [f"  {g:8s} n={len(sub):4d}"]
        if tpf:
            parts.append(f"median true cells/frame={np.median(tpf):7.0f}")
        if ar:
            parts.append(f"median annot rate={100*np.median(ar):5.2f}%")
        parts.append(f"share of score weight={100*e/max(tot_w,1):5.1f}%")
        print("  ".join(parts))

    print("\nNote: the score weight w_i ~ number of GT edges. Optimize for the "
          "group that carries the weight, not the one with the most real cells.")
    warning_summary()
    return out


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description="Biohub cell tracking data probe / viz")
    p.add_argument("mode", nargs="?", default="inspect",
                   choices=["inspect", "viz", "survey", "gif"])
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--dataset", default=None)
    p.add_argument("--frames", default="0,25,50,75")
    p.add_argument("--slab", type=int, default=9,
                   help="number of Z planes to project; do not use full-Z")
    p.add_argument("--roi", default=None, help="y0,y1,x0,x1")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--step", type=int, default=1, help="gif: frame stride")
    p.add_argument("--fps", type=int, default=8, help="gif: frames per second")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25,
                   help="progress line every N datasets; 0 = silent. Keep this "
                        "coarse -- a Kaggle cell can die on thousands of lines.")

    # parse_known_args: Jupyter injects "-f kernel.json" into sys.argv, and
    # parse_args would SystemExit(2) on it.
    a, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[ignoring unknown args] {unknown}", file=sys.stderr)

    if a.mode == "gif":
        run_gif(a.root, a.split, a.dataset, a.out, a.slab, a.roi,
                a.step, a.fps, a.max_frames)
    elif a.mode == "inspect":
        run_inspect(a.root, a.split, a.dataset)
    elif a.mode == "viz":
        run_viz(a.root, a.split, a.dataset, a.frames, a.slab, a.roi,
                a.out or "viz_cells.png")
    else:
        run_survey(a.root, a.split, a.out or f"survey_{a.split}.csv", a.limit,
                   a.log_every)


if __name__ == "__main__":
    main()
