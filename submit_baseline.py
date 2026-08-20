#!/usr/bin/env python
"""
submit_baseline.py -- classical tracking-by-detection baseline.

Pipeline:
    zarr -> Difference-of-Gaussians blob detection (per frame)
         -> keep the top-K responses so N_pred matches a target count
         -> Hungarian matching between consecutive frames (physical distance)
         -> optional gap closing
         -> submission CSV

Everything runs on CPU. No GPU, no training.

Typical use
-----------
Predict on test and write a submission:
    python submit_baseline.py predict --split test --out submission.csv

Score locally on train (this is the part that lets you iterate without
burning submissions):
    python submit_baseline.py score --limit 20

From a notebook:
    import submit_baseline as sb
    sb.run_score(limit=20, max_frames=15, oracle_target=True)
    sb.run_sweep(param="per_frame", values=[100, 300, 600], limit=20,
                 max_frames=15)

Fast iteration while tuning -- 20 datasets, 15 frames each takes ~1 min:
    python submit_baseline.py score --limit 20 --max-frames 15

Feed per-dataset cell-count predictions from cell_count.py:
    python submit_baseline.py predict --split test --out submission.csv \
        --target-csv n_true_pred.csv

Notes on the scoring rules this code is built around
----------------------------------------------------
* Score is ~90% edge Jaccard, plus 0.1 * division Jaccard.
* Annotation is sparse: a predicted NODE that matches no GT node is NOT a false
  positive. A predicted EDGE, however, counts toward the denominator as soon as
  ONE of its endpoints matches a GT node of the right degree -- see
  `score_one` for the exact rule, taken from the official metrics.py.
* The node-count penalty is max(0, J * (1 - 0.1 * (N_pred - N_true)/N_true)),
  coefficient 0.1, no upper clamp.

  !! --ratio 2.0 IS UNVERIFIED AS OF 2026-08-18 !!
  It was measured with a scorer whose FP rule was too lenient (it required
  BOTH endpoints of an edge to match GT before an edge could be an FP). Under
  the correct rule, over-detection creates real FP, so the measured optimum may
  move. Re-run `sweep --param ratio` before trusting this default.
  (Two earlier positions on this have now both been wrong: first "stay below
  1.0" from reasoning without measuring, then "2.0" from measuring with a
  broken scorer. Measure again, with the fixed scorer.)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from viz_cells import (
    DEFAULT_SCALE,
    MATCH_RADIUS_UM,
    find_root,
    geff_path,
    group_of,
    list_datasets,
    open_image,
    read_geff,
    read_scale,
    zarr_path,
)

SCRIPT_VERSION = "2026-08-18f"

# Imported defensively: a Kaggle Dataset copy of viz_cells.py can be a version
# behind, and a hard import of a helper it lacks fails at module load. Per the
# project rule, even THAT degradation is announced rather than silent.
try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    _WARNED: dict = {}

    def warn_once(key, msg):
        n = _WARNED.get(key, 0)
        _WARNED[key] = n + 1
        if n == 0:
            print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        if _WARNED:
            print("\n=== degradations this run ===")
            for k, n in sorted(_WARNED.items(), key=lambda kv: -kv[1]):
                print(f"  {n:6d}x  {k}")
        if reset:
            _WARNED.clear()

    warn_once("old_viz_cells",
              "viz_cells.py is missing warn_once/warning_summary -- it is an "
              "older version than submit_baseline.py. Update the Kaggle "
              "Dataset; SCRIPT_VERSIONs should match.")


def _progress(i: int, total: int, every: int, msg: str = ""):
    """
    Throttled progress logging. Defined locally rather than imported from
    viz_cells: a Kaggle Dataset copy of these files can easily be a version
    behind, and a cross-file import of a private helper turns that into an
    ImportError at module load.

    Kaggle cells can die when a cell emits thousands of lines, so never print
    one line per dataset. every=0 silences progress; summaries always print.
    """
    if every and (i == 1 or i == total or i % every == 0):
        print(f"  [{i}/{total}] {msg}", flush=True)

SUBMISSION_COLUMNS = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
)


# ---------------------------------------------------------------- detection

def _sigma_voxels(sigma_um: float, scale: tuple) -> tuple:
    """Convert a physical sigma (um) into per-axis voxel sigmas."""
    return tuple(max(0.5, sigma_um / s) for s in scale)


def _odd_size(radius_um: float, scale: tuple) -> tuple:
    """Odd-sized neighbourhood (in voxels) spanning +-radius_um physically."""
    return tuple(max(1, int(round(radius_um / s)) * 2 + 1) for s in scale)


def detect_frame(vol: np.ndarray, scale: tuple, sigma_um: float = 2.0,
                 sigma_ratio: float = 1.6, nms_um: float = 3.5,
                 cap: int = 2000, truncate: float = 3.0) -> tuple:
    """
    Difference-of-Gaussians blob detection on one 3D frame.

    Returns (coords (n,3) as z,y,x voxel indices, response (n,), capped_flag).

    `truncate` is the Gaussian kernel cutoff in sigmas. scipy defaults to 4.0;
    3.0 is ~25% faster and changes the response by ~0.1% of the image standard
    deviation, which is far below anything that moves a detection.

    The response is normalized by this frame's own 99.9th percentile, so
    responses are comparable ACROSS frames despite photobleaching -- without
    that, a global threshold silently loses all detections in late frames.

    `cap` bounds how many maxima are kept per frame, purely to bound memory.
    """
    from scipy import ndimage

    v = vol.astype(np.float32)
    s1 = _sigma_voxels(sigma_um, scale)
    s2 = tuple(s * sigma_ratio for s in s1)
    resp = (ndimage.gaussian_filter(v, s1, truncate=truncate)
            - ndimage.gaussian_filter(v, s2, truncate=truncate))

    scale_ref = float(np.percentile(resp, 99.9))
    if scale_ref <= 0:
        warn_once("flat_frame",
                  "a frame has a non-positive 99.9th percentile DoG response "
                  "-> returning zero detections for it")
        return np.zeros((0, 3), np.int32), np.zeros(0, np.float32), False
    resp /= scale_ref

    mx = ndimage.maximum_filter(resp, size=_odd_size(nms_um, scale), mode="nearest")
    zz, yy, xx = np.nonzero((resp == mx) & (resp > 0))
    if len(zz) == 0:
        return np.zeros((0, 3), np.int32), np.zeros(0, np.float32), False

    vals = resp[zz, yy, xx]
    capped = len(vals) > cap
    if capped:
        keep = np.argpartition(-vals, cap)[:cap]
        zz, yy, xx, vals = zz[keep], yy[keep], xx[keep], vals[keep]
    return (np.stack([zz, yy, xx], axis=1).astype(np.int32),
            vals.astype(np.float32), capped)


# Detection is by far the most expensive step (two 3D Gaussian filters per
# frame) and it does NOT depend on per_frame / ratio / gate_um / max_gap. So it
# is computed once per (dataset, detector settings) and reused across a sweep.
_DETECT_CACHE: dict = {}
_CACHE_ROW_LIMIT = 20_000_000        # ~400 MB of candidates; then stop caching


def clear_detect_cache():
    """Drop cached detections. Call when memory gets tight."""
    n = len(_DETECT_CACHE)
    _DETECT_CACHE.clear()
    print(f"[cache] cleared {n} cached detections")


def detect_candidates(img, scale: tuple, sigma_um: float = 2.0,
                      sigma_ratio: float = 1.6, nms_um: float = 3.5,
                      max_frames: int | None = None,
                      cand_per_frame: int = 2000, truncate: float = 3.0,
                      cache_key=None) -> dict:
    """
    All candidate maxima for a movie, before any target-count selection.

    This is the cacheable half of detection. `cand_per_frame` bounds memory; if
    a frame actually hits that bound the surplus is dropped, which is a
    degradation and gets a warning.
    """
    if cache_key is not None and cache_key in _DETECT_CACHE:
        return _DETECT_CACHE[cache_key]

    T_full = int(img.shape[0])
    T = T_full if max_frames is None else min(T_full, int(max_frames))

    ts, cs, rs = [], [], []
    for t in range(T):
        vol = np.asarray(img[t])
        if vol.ndim > 3:
            vol = vol.reshape(vol.shape[-3:])
        coords, resp, capped = detect_frame(vol, scale, sigma_um, sigma_ratio,
                                            nms_um, cap=cand_per_frame,
                                            truncate=truncate)
        if capped:
            warn_once("cand_cap",
                      f"a frame produced more than cand_per_frame="
                      f"{cand_per_frame} maxima; the weakest were dropped "
                      f"before target selection. Raise cand_per_frame if you "
                      f"are targeting counts near that bound.")
        if len(coords):
            ts.append(np.full(len(coords), t, np.int32))
            cs.append(coords)
            rs.append(resp)

    if ts:
        out = {"t": np.concatenate(ts), "zyx": np.concatenate(cs),
               "resp": np.concatenate(rs), "T": T, "T_full": T_full}
    else:
        out = {"t": np.zeros(0, np.int32), "zyx": np.zeros((0, 3), np.int32),
               "resp": np.zeros(0, np.float32), "T": T, "T_full": T_full}

    if cache_key is not None:
        cached_rows = sum(len(v["t"]) for v in _DETECT_CACHE.values())
        if cached_rows < _CACHE_ROW_LIMIT:
            _DETECT_CACHE[cache_key] = out
        else:
            warn_once("cache_full",
                      f"detection cache passed {_CACHE_ROW_LIMIT} rows -> no "
                      f"longer caching, so sweeps will recompute detection. "
                      f"Call clear_detect_cache() or lower --limit.")
    return out


def select_top_k(cands: dict, target_n: int) -> dict:
    """
    Keep the globally strongest `target_n` candidates. Cheap; this is the half
    that varies during a per_frame / ratio sweep.

    A global top-K rather than a fixed intensity threshold is what makes the
    detector adapt per dataset -- true density spans 20x across the train
    split. The per-frame response normalization in detect_frame is what makes
    ranking across frames meaningful.
    """
    t_all, c_all, r_all = cands["t"], cands["zyx"], cands["resp"]
    if target_n and len(r_all) > target_n:
        keep = np.argpartition(-r_all, target_n)[:target_n]
        keep.sort()
        t_all, c_all, r_all = t_all[keep], c_all[keep], r_all[keep]
    order = np.lexsort((c_all[:, 0], t_all))
    return {"t": t_all[order], "zyx": c_all[order], "resp": r_all[order],
            "T": cands["T"], "T_full": cands["T_full"]}


# ---------------------------------------------------------------- linking

def _match_two_frames(pa: np.ndarray, pb: np.ndarray, gate: float) -> list:
    """
    Optimal one-to-one assignment between two point sets under a distance gate.

    pa, pb are (n,3) physical coordinates in um. Returns [(ia, ib), ...].

    Greedy nearest-neighbour is the obvious alternative and it is wrong here:
    nuclei sit roughly one diameter apart while typical motion is ~0.4
    diameters, so a cell can genuinely be closer to a neighbour than to its own
    previous position. Only a globally optimal assignment sorts that out.
    """
    from scipy.optimize import linear_sum_assignment

    if len(pa) == 0 or len(pb) == 0:
        return []

    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    big = gate * 10.0
    cost = np.where(d <= gate, d, big)
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if d[i, j] <= gate]


def link(nodes: dict, scale: tuple, gate_um: float = 10.0,
         max_gap: int = 1) -> np.ndarray:
    """
    Link detections across time. Returns an (E, 2) array of row indices into
    `nodes` (source, target).

    max_gap=1 links only consecutive frames. max_gap=2 additionally tries to
    bridge one missed frame for detections that stayed unlinked ("gap
    closing"), with a proportionally wider gate.

    Note: only t -> t+1 edges score. A t -> t+2 edge is dropped by the official
    metric, so gap closing here inserts an interpolated node in the skipped
    frame and emits two single-step edges instead.
    """
    t = nodes["t"]
    zyx = nodes["zyx"].astype(np.float64)
    phys = zyx * np.asarray(scale)[None, :]

    T = int(nodes["T"])
    by_t = {tt: np.nonzero(t == tt)[0] for tt in range(T)}

    edges = []
    for tt in range(T - 1):
        ia, ib = by_t.get(tt, []), by_t.get(tt + 1, [])
        if len(ia) == 0 or len(ib) == 0:
            continue
        for a, b in _match_two_frames(phys[ia], phys[ib], gate_um):
            edges.append((int(ia[a]), int(ib[b])))

    if max_gap >= 2 and len(edges):
        e = np.array(edges)
        has_out = set(e[:, 0].tolist())
        has_in = set(e[:, 1].tolist())
        extra = []
        for tt in range(T - 2):
            ia = np.array([i for i in by_t.get(tt, []) if i not in has_out], int)
            ib = np.array([i for i in by_t.get(tt + 2, []) if i not in has_in], int)
            if len(ia) == 0 or len(ib) == 0:
                continue
            for a, b in _match_two_frames(phys[ia], phys[ib], gate_um * 1.8):
                extra.append((int(ia[a]), int(ib[b])))
                has_out.add(int(ia[a]))
                has_in.add(int(ib[b]))
        if extra:
            edges = _bridge_gaps(nodes, edges, extra)

    return np.array(edges, dtype=np.int64).reshape(-1, 2)


def _bridge_gaps(nodes: dict, edges: list, gap_pairs: list) -> list:
    """
    Turn each t -> t+2 pair into a real node at t+1 plus two single-step edges,
    since the metric only credits consecutive-frame edges.

    Mutates `nodes` in place (appends the interpolated detections).
    """
    t = list(nodes["t"])
    zyx = list(nodes["zyx"])
    resp = list(nodes["resp"])

    out = list(edges)
    for a, b in gap_pairs:
        mid = ((np.asarray(zyx[a], float) + np.asarray(zyx[b], float)) / 2)
        new_i = len(t)
        t.append(int(nodes["t"][a]) + 1)
        zyx.append(np.round(mid).astype(np.int32))
        resp.append(0.0)
        out.append((a, new_i))
        out.append((new_i, b))

    nodes["t"] = np.asarray(t, np.int32)
    nodes["zyx"] = np.asarray(zyx, np.int32)
    nodes["resp"] = np.asarray(resp, np.float32)
    return out


def prune_isolated_nodes(nodes: dict, edges: np.ndarray) -> tuple:
    """
    Drop predicted nodes that ended up with no edge, and reindex the edges.

    This is free score. The node-count penalty counts EVERY predicted node,
    but only nodes that carry an edge can ever produce a TP. An unlinked node
    is pure cost: it inflates N_pred, and because node matching is one-to-one
    it can also occupy a GT node that a linked prediction would rather have.

    Returns (nodes, edges) with node indices remapped.
    """
    n = len(nodes["t"])
    if n == 0 or len(edges) == 0:
        return nodes, edges

    used = np.zeros(n, dtype=bool)
    used[edges[:, 0]] = True
    used[edges[:, 1]] = True
    keep = np.nonzero(used)[0]
    if len(keep) == n:
        return nodes, edges

    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v)
           for k, v in nodes.items()}
    out["pruned"] = int(n - len(keep))
    out["pruned_frac"] = float((n - len(keep)) / n)
    return out, remap[edges]


# ---------------------------------------------------------------- targets

def load_targets(path: str | None) -> dict:
    """Read {dataset: predicted N_true} from a cell_count.py predict CSV."""
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[r["dataset"]] = float(r["n_true_pred"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def target_for(name: str, T: int, targets: dict, per_frame: float,
               ratio: float, base: float | None = None) -> int:
    """Pick this dataset's target node count and apply the safety ratio."""
    if base is None:
        if targets and name not in targets:
            warn_once("target_csv_miss",
                      f"{name} is not in the --target-csv -> using the flat "
                      f"{per_frame}/frame default. On the hidden test split the "
                      f"dataset names differ from train, so a precomputed CSV "
                      f"matches NOTHING: compute the counts in the same run.")
        base = targets.get(name, per_frame * T)
    return max(1, int(round(base * ratio)))


def oracle_target(root: Path, split: str, name: str, T: int,
                  T_full: int) -> float | None:
    """
    The dataset's TRUE node count, straight from the GT metadata.

    Only usable on train, and only as a diagnostic: it answers "how good would
    the detector be if the count estimator were perfect?". The gap between the
    oracle run and a --per-frame run is exactly how much cell_count.py is worth
    -- run it before investing more effort there.
    """
    gp = geff_path(root, split, name)
    if not gp.exists():
        warn_once("oracle_no_geff",
                  f"--oracle-target asked for, but {name} has no geff -> "
                  f"falling back to the --per-frame default for it")
        return None
    try:
        est = read_geff(gp)["extra"].get("estimated_number_of_nodes")
    except Exception as e:
        warn_once("oracle_geff_unreadable",
                  f"--oracle-target: cannot read {name} geff ({e}) -> falling "
                  f"back to the --per-frame default")
        return None
    if not est:
        warn_once("oracle_no_estimate",
                  f"--oracle-target: {name} geff has no "
                  f"estimated_number_of_nodes -> using the --per-frame default")
        return None
    return float(est) * (T / float(T_full or 1))


# ---------------------------------------------------------------- one dataset

def track_one(root: Path, split: str, name: str, cfg: dict) -> tuple:
    """Detect + link one dataset. Returns (nodes, edges, scale)."""
    zp = zarr_path(root, split, name)
    img = open_image(zp)
    scale = read_scale(zp)
    T_full = int(img.shape[0])
    T = T_full if cfg["max_frames"] is None else min(T_full, cfg["max_frames"])

    base = oracle_target(root, split, name, T, T_full) if cfg["oracle"] else None
    target = target_for(name, T, cfg["targets"], cfg["per_frame"], cfg["ratio"],
                        base)

    key = None
    if cfg["cache"]:
        key = (str(root), split, name, cfg["sigma_um"], cfg["sigma_ratio"],
               cfg["nms_um"], cfg["max_frames"], cfg["cand_per_frame"],
               cfg["truncate"])
    cands = detect_candidates(img, scale, cfg["sigma_um"], cfg["sigma_ratio"],
                              cfg["nms_um"], cfg["max_frames"],
                              cfg["cand_per_frame"], cfg["truncate"], key)
    nodes = select_top_k(cands, target)
    edges = link(nodes, scale, cfg["gate_um"], cfg["max_gap"])
    if cfg["prune_isolated"]:
        nodes, edges = prune_isolated_nodes(nodes, edges)
    return nodes, edges, scale


# ---------------------------------------------------------------- submission

def run_predict(root=None, split="test", out="submission.csv", limit=None,
                log_every=10, **cfg_kw):
    cfg = _config(**cfg_kw)
    r = find_root(root)
    names = list_datasets(r, split)
    if limit:
        names = names[: int(limit)]

    n_rows = 0
    tot_nodes = tot_edges = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUBMISSION_COLUMNS)
        for i, name in enumerate(names, 1):
            try:
                nodes, edges, _ = track_one(r, split, name, cfg)
            except Exception as e:
                print(f"  [error] {name}: {e}", flush=True)
                continue

            # Node rows first, then edge rows -- same order as the official
            # geffs_to_csv.py. node_id only has to be unique within a dataset.
            t, zyx = nodes["t"], nodes["zyx"]
            for k in range(len(t)):
                w.writerow([n_rows, name, "node", k, int(t[k]),
                            int(zyx[k, 0]), int(zyx[k, 1]), int(zyx[k, 2]), -1, -1])
                n_rows += 1
            for s, d in edges:
                w.writerow([n_rows, name, "edge", -1, -1, -1, -1, -1,
                            int(s), int(d)])
                n_rows += 1

            tot_nodes += len(t)
            tot_edges += len(edges)
            _progress(i, len(names), log_every,
                      f"{name} nodes={len(t)} edges={len(edges)}")

    print(f"\n[saved] {out}")
    print(f"  {len(names)} datasets, {tot_nodes} nodes, {tot_edges} edges, "
          f"{n_rows} rows")
    warning_summary()
    return out


# ---------------------------------------------------------------- local score

def _match_nodes_per_frame(pred_t, pred_phys, gt_t, gt_phys,
                           max_distance=MATCH_RADIUS_UM) -> dict:
    """
    Optimal one-to-one node matching within each frame, gated at max_distance.
    Returns {pred_row: gt_row}.

    One-to-one is what the official matcher does, which conveniently means the
    "merge collapse" special case in metrics.py cannot arise here.
    """
    from scipy.optimize import linear_sum_assignment

    out = {}
    for tt in np.unique(gt_t):
        gi = np.nonzero(gt_t == tt)[0]
        pi = np.nonzero(pred_t == tt)[0]
        if len(gi) == 0 or len(pi) == 0:
            continue
        d = np.linalg.norm(pred_phys[pi][:, None, :] - gt_phys[gi][None, :, :],
                           axis=2)
        cost = np.where(d <= max_distance, d, max_distance * 10.0)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if d[a, b] <= max_distance:
                out[int(pi[a])] = int(gi[b])
    return out


def score_one(nodes, edges, scale, gt, max_t=None) -> dict:
    """
    Reimplementation of the official edge metric for one dataset.

    Rewritten 2026-08-18 after reading the real
    `tracking_cellmot/metrics.py::_evaluate_matched_graph`. The previous
    version was WRONG in a way that flattered us -- see "the FP rule" below.

    Node matching: optimal one-to-one assignment within each frame, gated at
    7 um physical distance. Same as the official DistanceMatching.

    Then the predicted edge list is cleaned, in this order:
      1. drop duplicate (source, target) pairs
      2. drop any edge that is not exactly one frame forward
         (t_target == t_source + 1). Backward edges and gap-closing edges
         spanning 2+ frames are DISCARDED, not penalised.
      3. collapse merges: if several predicted edges map onto the same
         (matched_source, matched_target) GT edge, keep the first only
      4. cap out-degree at 2 -- a cell has at most two daughters; extra
         outgoing edges are dropped, keeping the first two

    THE FP RULE (this is the part we had wrong):

        out_valid = the edge's SOURCE matched a GT node with out_degree > 0
        in_valid  = the edge's TARGET matched a GT node with in_degree  > 0
        an edge counts toward the denominator iff (out_valid OR in_valid)

    ONE matched endpoint is enough. Our old rule required BOTH endpoints to
    match before an edge could be an FP, so every "linked a real cell to the
    wrong thing" error was silently ignored instead of penalised. That made
    every local score optimistic, and it is the most likely single cause of
    the local-vs-leaderboard gap. `edge_fp_lenient` below reports the old
    count so the size of the correction stays visible.

    FN = GT edges that are not TP.
    """
    gt_ids = gt["node_ids"]
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gt_phys = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * np.asarray(scale)
    gt_t = gt["t"].astype(int)

    pred_phys = nodes["zyx"].astype(float) * np.asarray(scale)
    pred_t = nodes["t"].astype(int)

    m = _match_nodes_per_frame(pred_t, pred_phys, gt_t, gt_phys)
    div = score_divisions_simple(nodes, edges, gt, m)

    # In --max-frames mode only part of the movie is processed, so GT edges
    # outside that window would all be counted as FN and the Jaccard would be
    # meaningless. Restrict the GT to the same window instead.
    gt_edge_set = set()
    gt_out_deg: dict = {}
    gt_in_deg: dict = {}
    for a, b in gt["edges"]:
        ra, rb = gt_row.get(int(a)), gt_row.get(int(b))
        if ra is None or rb is None:
            continue
        if max_t is not None and (gt_t[ra] >= max_t or gt_t[rb] >= max_t):
            continue
        gt_edge_set.add((ra, rb))
        gt_out_deg[ra] = gt_out_deg.get(ra, 0) + 1
        gt_in_deg[rb] = gt_in_deg.get(rb, 0) + 1

    # ---- clean the predicted edge list, in the official order ----------
    seen_pairs = set()
    kept = []                                     # (source, target)
    n_backward = n_gap = n_dup = 0
    for s_, d_ in edges:
        s_, d_ = int(s_), int(d_)
        if (s_, d_) in seen_pairs:
            n_dup += 1
            continue
        seen_pairs.add((s_, d_))
        dt = int(pred_t[d_]) - int(pred_t[s_])
        if dt != 1:
            if dt <= 0:
                n_backward += 1
            else:
                n_gap += 1
            continue
        kept.append((s_, d_))
    if n_gap:
        warn_once("edge_gap_dropped",
                  f"{n_gap} predicted edge(s) span more than one frame; the "
                  f"official metric DISCARDS these, so gap-closing links earn "
                  f"nothing unless intermediate nodes are also emitted")
    if n_backward:
        warn_once("edge_backward_dropped",
                  f"{n_backward} predicted edge(s) go backward or within a "
                  f"frame; dropped, as the official metric does")

    # collapse merges: one predicted edge per matched GT edge pair
    seen_gt_pair = set()
    kept2 = []
    n_merge = 0
    for s_, d_ in kept:
        gs, gd = m.get(s_), m.get(d_)
        if gs is not None and gd is not None:
            if (gs, gd) in seen_gt_pair:
                n_merge += 1
                continue
            seen_gt_pair.add((gs, gd))
        kept2.append((s_, d_))
    if n_merge:
        warn_once("edge_merge_collapsed",
                  f"{n_merge} predicted edge(s) collapsed onto an already-used "
                  f"GT edge; dropped, as the official metric does")

    # cap out-degree at 2, keeping the first two per source
    outdeg: dict = {}
    kept3 = []
    n_outdeg = 0
    for s_, d_ in kept2:
        c = outdeg.get(s_, 0)
        if c >= 2:
            n_outdeg += 1
            continue
        outdeg[s_] = c + 1
        kept3.append((s_, d_))
    if n_outdeg:
        warn_once("edge_outdeg_capped",
                  f"{n_outdeg} outgoing edge(s) dropped from nodes with more "
                  f"than two children, as the official metric does")

    # ---- classify -------------------------------------------------------
    tp = fp = 0
    fp_lenient = 0
    for s_, d_ in kept3:
        gs, gd = m.get(s_), m.get(d_)
        is_tp = gs is not None and gd is not None and (gs, gd) in gt_edge_set
        out_valid = gs is not None and gt_out_deg.get(gs, 0) > 0
        in_valid = gd is not None and gt_in_deg.get(gd, 0) > 0
        if is_tp:
            tp += 1
        elif out_valid or in_valid:
            fp += 1
            if gs is not None and gd is not None:
                fp_lenient += 1          # what the old, too-kind rule counted
    fn = len(gt_edge_set) - tp

    # Decompose the loss. FP being small does NOT mean linking is fine: a wrong
    # link whose endpoints do not both match GT still surfaces mostly as FN.
    #   reachable = GT edges whose BOTH endpoints were matched by a prediction
    #               -> the ceiling detection allows
    #   link_rate = TP / reachable
    #               -> how much of that ceiling the linker actually collects
    matched_gt = set(m.values())
    reachable = sum(1 for (a, b) in gt_edge_set
                    if a in matched_gt and b in matched_gt)

    in_window = (gt_t < max_t).sum() if max_t is not None else len(gt_ids)
    return {**div,
            "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "edge_fp_lenient": fp_lenient,
            "num_pred_nodes": int(len(pred_t)),
            "node_recall": len(matched_gt) / max(int(in_window), 1),
            "edge_reachable": int(reachable),
            "n_gt_edges": int(len(gt_edge_set))}


def score_divisions_simple(nodes, edges, gt, m: dict) -> dict:
    """
    Approximate division scoring, so the local number can be put on the same
    scale as the leaderboard.

    The official metric is `adj_edge_jaccard + 0.1 * division_jaccard`; this
    module scored only the first term, which makes any local-vs-LB comparison
    invalid. This fills the gap.

    Definitions used here:
      * a division = a node with >= 2 outgoing edges
      * TP  = a GT division whose divider is matched by a predicted divider,
              allowing +-1 frame (the official metric allows this because the
              exact frame of a division is subjective, so the predicted divider
              may map to the GT divider's own parent or child on its track)
      * FP  = a predicted division whose divider matches a GT node that is NOT
              a GT divider. As with edges, a division on an UNANNOTATED cell is
              ignored, not penalised -- annotation is sparse
      * FN  = GT divisions with no match

    KNOWN DIFFERENCES from `tracking_cellmot/division_metrics.py`, all of which
    make this version MORE LENIENT (so treat it as an upper bound):
      * no grandchildren check -- the official builds a
        parent -> divider -> children -> grandchildren subgraph and requires the
        local directed topology to be correct
      * no strongly-connected / branch-merge validation
      * greedy pairing instead of bipartite maximum matching
      * daughters are not required to match individually

    Also note the sample size: divisions occur roughly once per 854 GT edges,
    so a handful of datasets contains only a handful of events and the
    resulting Jaccard is extremely noisy. Only trust it over many datasets.
    """
    gt_ids = gt["node_ids"]
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}

    # GT divisions, and each GT node's track neighbours (for the +-1 tolerance)
    gt_out: dict = {}
    gt_in: dict = {}
    for a, b in gt["edges"]:
        ra, rb = gt_row.get(int(a)), gt_row.get(int(b))
        if ra is None or rb is None:
            continue
        gt_out.setdefault(ra, []).append(rb)
        gt_in.setdefault(rb, []).append(ra)
    gt_div = {r for r, kids in gt_out.items() if len(kids) >= 2}

    pred_out: dict = {}
    for s_, d_ in edges:
        pred_out.setdefault(int(s_), []).append(int(d_))
    pred_div = [r for r, kids in pred_out.items() if len(kids) >= 2]

    matched_gt: set = set()
    tp = fp = 0
    for pr in pred_div:
        g = m.get(pr)
        if g is None:
            continue                    # divider on an unannotated cell: ignored
        # +-1 frame tolerance, expressed along the GT track
        cands = [g] + gt_in.get(g, []) + gt_out.get(g, [])
        hit = next((c for c in cands if c in gt_div and c not in matched_gt), None)
        if hit is not None:
            matched_gt.add(hit)
            tp += 1
        elif not any(c in gt_div for c in cands):
            fp += 1                     # matched a GT node that never divides
    fn = len(gt_div) - len(matched_gt)
    return {"div_tp": tp, "div_fp": fp, "div_fn": fn, "n_gt_div": len(gt_div),
            "n_pred_div": len(pred_div)}


def summarise(rows: list) -> dict:
    """
    Aggregate per-dataset counts the way the official summarise() does.

    edge_jaccard is micro-averaged, but adj_edge_jaccard is a PER-DATASET
    adjusted Jaccard averaged with weights w_i = TP_i + FP_i + FN_i. So you
    cannot offset over-detection in one dataset with under-detection in
    another -- each has to be calibrated on its own.
    """
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    micro = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    num = den = 0.0
    for r in rows:
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        if w <= 0 or not np.isfinite(r.get("n_true", np.nan)):
            continue
        j = r["edge_tp"] / w
        ratio = (r["num_pred_nodes"] - r["n_true"]) / r["n_true"]
        num += w * max(0.0, j * (1 - 0.1 * ratio))
        den += w
    adj = num / den if den else float("nan")

    dtp = sum(r.get("div_tp", 0) for r in rows)
    dfp = sum(r.get("div_fp", 0) for r in rows)
    dfn = sum(r.get("div_fn", 0) for r in rows)
    ddenom = dtp + dfp + dfn
    div_j = dtp / ddenom if ddenom else float("nan")

    reach = sum(r.get("edge_reachable", 0) for r in rows)
    n_gt = sum(r.get("n_gt_edges", 0) for r in rows)
    fp_len = sum(r.get("edge_fp_lenient", 0) for r in rows)
    return {"n": len(rows), "edge_jaccard": micro, "adj_edge_jaccard": adj,
            "edge_fp_lenient": fp_len,
            "edge_reachable": reach, "n_gt_edges": n_gt,
            "detect_ceiling": reach / n_gt if n_gt else float("nan"),
            "link_rate": tp / reach if reach else float("nan"),
            "div_tp": dtp, "div_fp": dfp, "div_fn": dfn,
            "division_jaccard": div_j,
            # The official run-level score. Comparable to a leaderboard number;
            # `adj_edge_jaccard` alone is NOT.
            "score_full": adj + (0.1 * div_j if div_j == div_j else 0.0),
            "score": adj,   # divisions ignored in this baseline
            "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "node_recall": float(np.mean([r["node_recall"] for r in rows]))
            if rows else float("nan")}


def run_score(root=None, split="train", limit=None, log_every=10,
              out=None, _quiet=False, **cfg_kw):
    cfg = _config(**cfg_kw)
    r = find_root(root)
    names = [n for n in list_datasets(r, split) if geff_path(r, split, n).exists()]
    if limit:
        # Stride instead of truncating: names are sorted, so the first N are all
        # from one imaging batch and would tell you nothing about the other.
        step = max(1, len(names) // int(limit))
        names = names[::step][: int(limit)]

    rows = []
    for i, name in enumerate(names, 1):
        try:
            nodes, edges, scale = track_one(r, split, name, cfg)
            gt = read_geff(geff_path(r, split, name))
            row = score_one(nodes, edges, scale, gt, max_t=cfg["max_frames"])
        except Exception as e:
            print(f"  [error] {name}: {e}", flush=True)
            continue

        row["dataset"] = name
        row["group"] = group_of(name)
        row["pruned_frac"] = float(nodes.get("pruned_frac", 0.0))
        est = gt["extra"].get("estimated_number_of_nodes")
        if cfg["max_frames"] is not None and est:
            # With --max-frames only part of the movie is covered, so compare
            # against the matching fraction of N_true (N_true counts the WHOLE
            # movie, hence T_full in the denominator).
            est = float(est) * min(1.0, nodes["T"] / float(nodes["T_full"] or 1))
        row["n_true"] = float(est) if est else float("nan")
        w = row["edge_tp"] + row["edge_fp"] + row["edge_fn"]
        row["edge_jaccard"] = row["edge_tp"] / w if w else float("nan")
        rows.append(row)
        _progress(i, len(names), log_every,
                  f"{name} J={row['edge_jaccard']:.3f}")

    if not rows:
        print("no datasets scored")
        return None

    s = summarise(rows)
    ratios = [r_["num_pred_nodes"] / r_["n_true"] for r_ in rows
              if np.isfinite(r_.get("n_true", np.nan)) and r_["n_true"] > 0]
    s["ratio_median"] = float(np.median(ratios)) if ratios else float("nan")

    if _quiet:
        return s

    print("\n=== local score ===")
    print(f"datasets           : {s['n']}")
    print(f"edge TP/FP/FN      : {s['edge_tp']} / {s['edge_fp']} / {s['edge_fn']}")
    _new_fp = s['edge_fp'] - s.get('edge_fp_lenient', s['edge_fp'])
    if _new_fp:
        print(f"                     ({_new_fp} of those FP were IGNORED by the "
              f"pre-2026-08-18 scorer, which needed BOTH endpoints matched -- "
              f"that bias is why old local scores read high)")
    print(f"edge_jaccard       : {s['edge_jaccard']:.4f}  (micro)")
    print(f"adj_edge_jaccard   : {s['adj_edge_jaccard']:.4f}")
    print(f"division_jaccard   : {s['division_jaccard']:.4f}  "
          f"(TP/FP/FN {s['div_tp']}/{s['div_fp']}/{s['div_fn']} -- APPROXIMATE, "
          f"and tiny samples are meaningless)")
    print(f"score_full         : {s['score_full']:.4f}  <- adj_J + 0.1*div_J, "
          f"the leaderboard-comparable number")
    print(f"node_recall        : {s['node_recall']:.4f}  "
          f"(fraction of GT cells our detector found)")
    print(f"detect_ceiling     : {s['detect_ceiling']:.4f}  "
          f"(GT edges with BOTH endpoints detected -- the best micro_J "
          f"detection allows)")
    print(f"link_rate          : {s['link_rate']:.4f}  "
          f"(of those reachable edges, the fraction the linker actually got)")
    print(f"                     micro_J = detect_ceiling x link_rate = "
          f"{s['detect_ceiling']*s['link_rate']:.4f}")

    for g in sorted({r_["group"] for r_ in rows}):
        sub = [r_ for r_ in rows if r_["group"] == g]
        sg = summarise(sub)
        wg = sum(r_["edge_tp"] + r_["edge_fp"] + r_["edge_fn"] for r_ in sub)
        print(f"  {g:8s} n={len(sub):3d}  adj_J={sg['adj_edge_jaccard']:.4f}  "
              f"recall={sg['node_recall']:.3f}  weight={wg}")

    pf = [r_.get("pruned_frac", 0.0) for r_ in rows]
    if any(pf):
        print(f"isolated pruned    : {100*float(np.mean(pf)):.1f}% of detections "
              f"had no edge and were dropped (free: they cannot make a TP)")
    if ratios:
        print(f"N_pred/N_true      : median {s['ratio_median']:.2f}  "
              f"(over-detecting is nearly free here; the detector saturates "
              f"around 1.5)")

    if len(rows) >= 8:
        b = bootstrap_lb_noise(rows)
        print(f"\nLB noise estimate (resampling {b['subsample']}/{b['n_scored']} "
              f"datasets = the public 29%):")
        print(f"  per-dataset J spread : SD {b['per_dataset_sd']:.3f}")
        print(f"  score SD             : {b['boot_sd']:.4f}   "
              f"90% range {b['boot_p05']:.3f}..{b['boot_p95']:.3f}")
        print(f"  -> a public-LB delta smaller than ~{2*b['boot_sd']:.3f} "
              f"cannot be distinguished from which movies got sampled")

    print(f"\nconfig: sigma={cfg['sigma_um']} nms={cfg['nms_um']} "
          f"gate={cfg['gate_um']} ratio={cfg['ratio']} "
          f"per_frame={cfg['per_frame']} max_gap={cfg['max_gap']}"
          + (f" max_frames={cfg['max_frames']}" if cfg["max_frames"] else "")
          + (" ORACLE-TARGET" if cfg["oracle"] else ""))

    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            cols = ["dataset", "group", "edge_tp", "edge_fp", "edge_fn",
                    "edge_jaccard", "num_pred_nodes", "n_true", "node_recall"]
            w_ = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w_.writeheader()
            for r_ in rows:
                w_.writerow(r_)
        print(f"[saved] {out}")
    warning_summary()
    return s


def bootstrap_lb_noise(rows: list, public_frac: float = 0.29,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Estimate how much a leaderboard score would bounce purely from which
    datasets landed in the public split.

    Repeatedly draw a `public_frac` subsample of the scored datasets, recompute
    the weighted adjusted Jaccard, and report the spread. This turns "is a
    +0.02 leaderboard delta real?" from an argument into a measurement -- and
    it uses data we already have, since the metric's unit is a whole movie and
    we hold ground truth for 199 of them.

    Caveat: the estimate is only as good as the sample. Run it on as many
    datasets as you can afford, and note that a well-calibrated pipeline has
    far less per-dataset spread than a badly calibrated one -- so re-measure
    after any big change rather than reusing an old number.
    """
    usable = [r for r in rows
              if np.isfinite(r.get("n_true", np.nan)) and r["n_true"] > 0
              and (r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) > 0]
    n = len(usable)
    k = max(2, int(round(public_frac * n)))
    if n < 8:
        warn_once("bootstrap_small",
                  f"only {n} scored datasets -> the bootstrap noise estimate is "
                  f"itself very noisy; treat it as an order of magnitude")

    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(n)
    for _ in range(n_boot):
        pick = rng.choice(idx, size=k, replace=False)
        scores.append(summarise([usable[i] for i in pick])["adj_edge_jaccard"])
    scores = np.array([v for v in scores if np.isfinite(v)])

    per = np.array([r["edge_tp"] / (r["edge_tp"] + r["edge_fp"] + r["edge_fn"])
                    for r in usable])
    return {"n_scored": n, "subsample": k,
            "per_dataset_sd": float(per.std()),
            "boot_sd": float(scores.std()),
            "boot_p05": float(np.percentile(scores, 5)),
            "boot_p95": float(np.percentile(scores, 95))}


# ---------------------------------------------------------------- verify

def run_verify(csv_path="submission.csv", root=None, gt_split="train"):
    """
    Score an already-written submission CSV against ground truth.

    Why this exists: the visible test/ split duplicates a handful of train
    datasets, so their GT is available under train/. That makes it possible to
    check the EXACT file you are about to submit -- parsing, id remapping,
    coordinate rounding and all -- rather than only the in-memory arrays.

    Any dataset in the CSV with no matching train geff is skipped.
    """
    r = find_root(root)
    per: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = per.setdefault(row["dataset"], {"nodes": {}, "edges": []})
            if row["row_type"] == "node":
                d["nodes"][int(row["node_id"])] = (
                    int(row["t"]), int(row["z"]), int(row["y"]), int(row["x"])
                )
            else:
                d["edges"].append((int(row["source_id"]), int(row["target_id"])))

    print(f"[{csv_path}] {len(per)} datasets")
    rows = []
    for name, d in sorted(per.items()):
        gp = geff_path(r, gt_split, name)
        if not gp.exists():
            print(f"  {name}: no GT under {gt_split}/, skipped")
            continue

        ids = sorted(d["nodes"])
        remap = {nid: i for i, nid in enumerate(ids)}
        arr = np.array([d["nodes"][nid] for nid in ids], dtype=np.int64)
        nodes = {"t": arr[:, 0].astype(np.int32), "zyx": arr[:, 1:].astype(np.int32)}
        edges = np.array([[remap[s], remap[t]] for s, t in d["edges"]
                          if s in remap and t in remap], dtype=np.int64).reshape(-1, 2)

        scale = read_scale(zarr_path(r, gt_split, name))
        gt = read_geff(gp)
        row = score_one(nodes, edges, scale, gt)
        row["dataset"] = name
        row["group"] = group_of(name)
        row["pruned_frac"] = float(nodes.get("pruned_frac", 0.0))
        est = gt["extra"].get("estimated_number_of_nodes")
        row["n_true"] = float(est) if est else float("nan")
        w = row["edge_tp"] + row["edge_fp"] + row["edge_fn"]
        j = row["edge_tp"] / w if w else float("nan")
        line = (f"  {name:22s} J={j:.4f}  nodes={len(ids)} "
                f"edges={len(edges)} recall={row['node_recall']:.3f}")
        if np.isfinite(row["n_true"]) and row["n_true"] > 0:
            line += f"  N/N_true={len(ids)/row['n_true']:.2f}"
        print(line)
        rows.append(row)

    if not rows:
        print("nothing scored")
        return None
    s = summarise(rows)
    print(f"\nedge_jaccard     : {s['edge_jaccard']:.4f}")
    print(f"adj_edge_jaccard : {s['adj_edge_jaccard']:.4f}")
    print(f"division_jaccard : {s['division_jaccard']:.4f}  "
          f"(TP/FP/FN {s['div_tp']}/{s['div_fp']}/{s['div_fn']}, approximate)")
    print(f"score_full       : {s['score_full']:.4f}  <- compare THIS to the LB")
    print(f"node_recall      : {s['node_recall']:.4f}")
    return s


# ---------------------------------------------------------------- config

def _config(sigma_um=2.0, sigma_ratio=1.6, nms_um=3.5, gate_um=10.0,
            per_frame=100.0, ratio=2.0, target_csv=None, max_gap=1,
            max_frames=None, oracle=False, oracle_target=None,
            cand_per_frame=2000, truncate=3.0, cache=True,
            prune_isolated=True):
    """
    Defaults come from the train-set survey:

    sigma_um   2.0  -- nuclei are ~7 um across; the DoG peaks near r/sqrt(3)
    nms_um     3.5  -- half a nucleus; two maxima closer than this are one cell
    gate_um   10.0  -- the 85%-weight group (6bba) has a median per-dataset
                       displacement p99 of 7.8 um, already above the 7 um match
                       radius, so a 7 um gate would cut real links
    per_frame  100  -- an ESTIMATE of true cells per frame, used only when no
                       per-dataset count is supplied. 100 is the median true
                       density of the 85%-weight group (97 cells/frame)
    ratio      2.0  -- how far to over-detect relative to that estimate.
                       RE-MEASURED 2026-08-18 with the corrected FP rule, on 24
                       datasets x 15 frames:
                         ratio  0.8  1.0  1.4  2.0  2.8
                         adj_J .328 .350 .413 .447 .457
                       Still rising at the top of the range, so 2.0 is not the
                       optimum -- it is just where the old, broken sweep stopped
                       being trusted. Extend the sweep before changing this
                       default. Note N_pred/N_true only reaches 0.94 at ratio
                       2.8: the DoG detector runs out of candidates, so beyond
                       that the knob does nothing and nms_um is the real lever.
                       (The old note here quoted adj_J 0.673 at 2.0 with a peak
                       at 2.0; that came from the scorer that could not see
                       most false positives. Ignore it.)

    `oracle_target` is accepted as an alias for `oracle` so the keyword matches
    the --oracle-target CLI flag either way.
    """
    if oracle_target is not None:
        oracle = bool(oracle_target)
    return dict(sigma_um=sigma_um, sigma_ratio=sigma_ratio, nms_um=nms_um,
                gate_um=gate_um, per_frame=per_frame, ratio=ratio,
                targets=load_targets(target_csv), max_gap=max_gap,
                max_frames=max_frames, oracle=oracle,
                cand_per_frame=cand_per_frame, truncate=truncate, cache=cache,
                prune_isolated=bool(prune_isolated))


# ---------------------------------------------------------------- sweep

SWEEPABLE = ("gate_um", "per_frame", "ratio", "nms_um", "sigma_um",
             "sigma_ratio", "max_gap", "prune_isolated")


def run_sweep(root=None, split="train", param="gate_um", values=None,
              limit=20, out=None, **cfg_kw):
    """
    Re-run the local score across one parameter's values and print a table.

    Always sweep with the SAME --limit and --max-frames, so the rows are
    comparable. Change one knob at a time: these interact (a wider gate only
    helps if the detector actually found the cells).
    """
    if param not in SWEEPABLE:
        raise ValueError(f"{param} not sweepable; pick from {SWEEPABLE}")
    values = values or []
    print(f"sweeping {param} over {values}   (limit={limit}, "
          f"max_frames={cfg_kw.get('max_frames')})\n")
    print(f"{param:>12}  {'adj_J':>7}  {'micro_J':>7}  {'recall':>7}  "
          f"{'N/N_true':>9}  {'TP':>6}  {'FP':>5}  {'FN':>6}")
    print("-" * 72)

    results = []
    for v in values:
        kw = dict(cfg_kw)
        kw[param] = v
        s = run_score(root, split, limit, log_every=0, out=None,
                      _quiet=True, **kw)
        if not s:
            continue
        results.append((v, s))
        print(f"{v:>12}  {s['adj_edge_jaccard']:>7.4f}  {s['edge_jaccard']:>7.4f}  "
              f"{s['node_recall']:>7.3f}  {s['detect_ceiling']:>6.3f}  "
              f"{s['link_rate']:>6.3f}  {s['ratio_median']:>9.2f}  "
              f"{s['edge_tp']:>6}  {s['edge_fn']:>6}", flush=True)

    if results:
        best = max(results, key=lambda kv: kv[1]["adj_edge_jaccard"])
        print(f"\nbest {param} = {best[0]}  ->  adj_J = "
              f"{best[1]['adj_edge_jaccard']:.4f}")
        # Only meaningful for an ordered numeric sweep of 3+ values; a 2-value
        # on/off comparison is always "at the edge".
        if len(values) >= 3 and best[0] in (values[0], values[-1]):
            print("  [!] the best value is at the edge of the swept range -- "
                  "extend the range, the optimum is probably outside it.")
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([param, "adj_edge_jaccard", "edge_jaccard", "node_recall",
                        "ratio_median", "edge_tp", "edge_fp", "edge_fn"])
            for v, s in results:
                w.writerow([v, s["adj_edge_jaccard"], s["edge_jaccard"],
                            s["node_recall"], s["ratio_median"],
                            s["edge_tp"], s["edge_fp"], s["edge_fn"]])
        print(f"[saved] {out}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Classical detection + linking baseline")
    p.add_argument("mode", nargs="?", default="score",
                   choices=["predict", "score", "sweep", "verify"])
    p.add_argument("--csv", default="submission.csv",
                   help="verify mode: the submission CSV to score")
    p.add_argument("--root", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=None,
                   help="only process the first N frames -- small-data mode for "
                        "fast tuning; do NOT use for a real submission")
    p.add_argument("--sigma-um", type=float, default=2.0)
    p.add_argument("--sigma-ratio", type=float, default=1.6)
    p.add_argument("--nms-um", type=float, default=3.5)
    p.add_argument("--gate-um", type=float, default=10.0)
    p.add_argument("--per-frame", type=float, default=100.0)
    p.add_argument("--ratio", type=float, default=2.0,
                   help="over-detection factor relative to the count estimate; "
                        "2.0 is the measured optimum, the curve is flat +-25%%")
    p.add_argument("--target-csv", default=None,
                   help="cell_count.py predict output, per-dataset N_true")
    p.add_argument("--max-gap", type=int, default=1, choices=[1, 2])
    p.add_argument("--cand-per-frame", type=int, default=2000,
                   help="max candidate maxima kept per frame before target "
                        "selection; bounds memory and the detection cache")
    p.add_argument("--truncate", type=float, default=3.0,
                   help="Gaussian kernel cutoff in sigmas (scipy default 4.0)")
    p.add_argument("--keep-isolated", action="store_true",
                   help="do NOT drop unlinked nodes (they cost node-count "
                        "penalty and can never produce a TP)")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the detection cache (sweeps get much slower)")
    p.add_argument("--oracle-target", action="store_true",
                   help="score mode only: use the GT's true node count as the "
                        "detection target. Diagnostic -- shows the ceiling if "
                        "the count estimator were perfect.")
    p.add_argument("--param", default="gate_um", choices=SWEEPABLE,
                   help="sweep mode: which knob to vary")
    p.add_argument("--values", default=None,
                   help="sweep mode: comma-separated values, e.g. 6,8,10,12")

    a, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[ignoring unknown args] {unknown}", file=sys.stderr)

    kw = dict(sigma_um=a.sigma_um, sigma_ratio=a.sigma_ratio, nms_um=a.nms_um,
              gate_um=a.gate_um, per_frame=a.per_frame, ratio=a.ratio,
              target_csv=a.target_csv, max_gap=a.max_gap,
              max_frames=a.max_frames, oracle=a.oracle_target,
              cand_per_frame=a.cand_per_frame, truncate=a.truncate,
              cache=not a.no_cache, prune_isolated=not a.keep_isolated)

    if a.mode == "verify":
        run_verify(a.csv, a.root)
    elif a.mode == "predict":
        kw.pop("oracle")
        run_predict(a.root, a.split or "test", a.out or "submission.csv",
                    a.limit, a.log_every, **kw)
    elif a.mode == "sweep":
        vals = [float(v) for v in (a.values or "").split(",") if v.strip()]
        if a.param == "max_gap":
            vals = [int(v) for v in vals]
        kw.pop(a.param, None)
        run_sweep(a.root, a.split or "train", a.param, vals, a.limit or 20,
                  a.out, **kw)
    else:
        run_score(a.root, a.split or "train", a.limit, a.log_every, a.out, **kw)


if __name__ == "__main__":
    main()
