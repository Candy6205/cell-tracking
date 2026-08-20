"""
division_recovery.py -- put divisions back into an ILP graph, as post-processing.

The problem
-----------
With `--ilp-division-weight 1.0 --ilp-appearance-weight 0.0` the solver takes a
fork only when `edge_prob > division_weight - appearance_weight`, i.e. never.
So every real division is currently expressed as: the parent continues into one
daughter, and the other daughter starts a brand-new track. The information is
still in the graph -- it just has the wrong shape. Recovering it means adding
ONE edge.

Why not just lower division_weight
----------------------------------
Measured (`division_features.py`, 33 divisions):

    division edges   p50 5.72 um   p90 8.66   p99 10.42
    ordinary edges   p50 1.82 um   p90 4.16   p99  8.48      AUC 0.86

A division edge is ~3x the median ordinary edge. Lowering `division_weight`
relaxes the bar for EVERY second edge, so what floods in is short spurious ones,
while the genuinely long division edges stay unlikely. Measured: 2 real forks
against 61 spurious. The knob pushes in the wrong direction; a LENGTH BAND does
not.

Features used, and what each was measured to be worth
-----------------------------------------------------
    edge length in the division band        AUC 0.86   (strongest)
    parent peak intensity vs same frame     AUC 0.71   (chromatin condenses)
    elongation                              AUC 0.54   -- DROPPED, useless
    sister separation, track length,
    distance from the volume boundary                  structural filters

The length feature is worthless on its own -- applied to all 129k GT edges it
would admit ~6400 ordinary edges for ~151 division ones, 2.3% precision. It only
works because the candidate set here is small: appearance nodes sitting next to
a track that continues.

Usage (CPU):
    import division_recovery as dr
    dr.candidates(PRED_DIR)                  # look at the pool first
    dr.sweep(PRED_DIR, caps=[0, 5, 15, 40])  # official score per cap

OUTPUT DISCIPLINE (rules 2 and 10): fixed-size tables, per-candidate rows go to
a CSV, `sweep` returns a small dict.
"""

from __future__ import annotations

import csv
import os
import shutil

import numpy as np

from viz_cells import (find_root, geff_path, open_image, read_scale, zarr_path)

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

SCRIPT_VERSION = "2026-08-19a"

# The division-edge band, from the measured p10..p99 of real division edges.
MIN_UM, MAX_UM = 3.5, 11.0
MIN_SISTER_UM = 4.0      # the two daughters must actually be apart
MIN_TRACK = 3            # a 1-2 frame "track" is a detection blip, not a cell
MIN_BOUNDARY_UM = 6.0    # cells drifting in from outside the volume
RZ, RXY = 2, 8           # intensity window, matching division_features.py
N_CONTROL = 12


# ------------------------------------------------------------------ graph i/o

def _load(path):
    import tracksdata as td
    res = td.graph.IndexedRXGraph.from_geff(str(path))
    g = res[0] if isinstance(res, tuple) else res
    na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = [int(v) for v in na["node_id"].to_list()]
    remap = {v: i for i, v in enumerate(ids)}
    nodes = np.stack([np.asarray(na[c].to_list(), float)
                      for c in ("t", "z", "y", "x")], axis=1)
    ea = g.edge_attrs(attr_keys=["source_id", "target_id"])
    if len(ea):
        edges = np.array([[remap[int(s)], remap[int(t)]]
                          for s, t in zip(ea["source_id"].to_list(),
                                          ea["target_id"].to_list())], dtype=int)
    else:
        edges = np.zeros((0, 2), int)
    return nodes, edges


def _save(path, nodes, edges):
    import polars as pl
    import tracksdata as td
    from tracking_cellmot.io import save_graph

    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y),
                             "x": float(x)} for t, z, y, x in nodes])
    if len(edges):
        g.bulk_add_edges([{"source_id": ids[int(s)], "target_id": ids[int(d)]}
                          for s, d in edges])
    save_graph(g, str(path))


# ------------------------------------------------------------------ features

def _peak_ratio(vol, p, controls, scale):
    """Peak intensity at p, over the median peak of controls in the same frame."""
    def peak(c):
        z, y, x = int(c[0]), int(c[1]), int(c[2])
        Z, Y, X = vol.shape
        w = vol[max(0, z - RZ):min(Z, z + RZ + 1),
                max(0, y - RXY):min(Y, y + RXY + 1),
                max(0, x - RXY):min(X, x + RXY + 1)]
        if w.size < 8:
            return np.nan
        return float(w.max() - np.percentile(w, 10))

    base = np.nanmedian([peak(c) for c in controls]) if len(controls) else np.nan
    v = peak(p)
    if not np.isfinite(base) or base <= 0 or not np.isfinite(v):
        return np.nan
    return v / base


def _candidates_one(nodes, edges, scale, shape, vol_getter=None, rng=None):
    """Every (parent, orphan) pair that could be a missed division, with features."""
    from scipy.spatial import cKDTree

    n = len(nodes)
    t = nodes[:, 0].astype(int)
    phys = nodes[:, 1:] * np.asarray(scale)
    out_deg = np.zeros(n, int)
    in_deg = np.zeros(n, int)
    child = {}
    nxt = {}
    for s, d in edges:
        out_deg[s] += 1; in_deg[d] += 1
        child.setdefault(s, []).append(d)
        nxt.setdefault(s, []).append(d)

    # track length forward from a node
    def track_len(i, cap=12):
        L, cur = 1, i
        while L < cap:
            ch = nxt.get(cur, [])
            if len(ch) != 1:
                break
            cur = ch[0]; L += 1
        return L

    Z, Y, X = shape[-3:]
    ext = np.array([Z, Y, X]) * np.asarray(scale)
    by_t = {}
    for i in range(n):
        by_t.setdefault(t[i], []).append(i)

    rows = []
    frames_needed = {}
    for tt in sorted(by_t):
        if tt == 0:
            continue
        prev = [i for i in by_t.get(tt - 1, []) if out_deg[i] == 1]
        orph = [i for i in by_t[tt] if in_deg[i] == 0]
        if not prev or not orph:
            continue
        tree = cKDTree(phys[prev])
        for b in orph:
            # structural filters that need no image
            if track_len(b) < MIN_TRACK:
                continue
            edge_dist = np.minimum(nodes[b, 1:] * np.asarray(scale),
                                   ext - nodes[b, 1:] * np.asarray(scale)).min()
            if edge_dist < MIN_BOUNDARY_UM:
                continue
            for j in tree.query_ball_point(phys[b], MAX_UM):
                a = prev[j]
                d_ab = float(np.linalg.norm(phys[b] - phys[a]))
                if d_ab < MIN_UM:
                    continue
                c = child[a][0]
                sister = float(np.linalg.norm(phys[b] - phys[c]))
                if sister < MIN_SISTER_UM:
                    continue
                rows.append({"parent": a, "orphan": b, "sibling": c, "t": int(tt),
                             "edge_um": d_ab, "sister_um": sister,
                             "track_len": track_len(b),
                             "boundary_um": float(edge_dist),
                             "peak_ratio": np.nan})
                frames_needed.setdefault(int(tt) - 1, []).append(len(rows) - 1)

    # image feature, one frame load per frame that has candidates
    if vol_getter is not None and rows:
        for tt, idxs in frames_needed.items():
            try:
                vol = vol_getter(tt)
            except Exception as e:
                warn_once("no_image",
                          f"could not read frame {tt} ({type(e).__name__}: {e}) "
                          f"-> peak_ratio stays NaN for those candidates and the "
                          f"ranking falls back to edge length alone")
                continue
            pool = [i for i in by_t.get(tt, []) if out_deg[i] == 1]
            if rng is not None and len(pool) > N_CONTROL:
                pool = list(rng.choice(pool, N_CONTROL, replace=False))
            ctrl = nodes[pool, 1:] if len(pool) else np.zeros((0, 3))
            for k in idxs:
                rows[k]["peak_ratio"] = _peak_ratio(
                    vol, nodes[rows[k]["parent"], 1:], ctrl, scale)
    return rows


def _score(rows):
    """Rank candidates. Length is the measured driver; intensity breaks ties."""
    MID = 5.72        # measured median division-edge length
    for r in rows:
        length_fit = 1.0 / (1.0 + abs(r["edge_um"] - MID) / MID)
        pr = r["peak_ratio"]
        bright = 1.0 if not np.isfinite(pr) else float(np.clip(pr / 2.0, 0.3, 1.5))
        r["score"] = length_fit * bright
    return sorted(rows, key=lambda r: -r["score"])


# ------------------------------------------------------------------- public

def _iter(pred_dir, root, gt_split, use_image, seed):
    from pathlib import Path
    r = find_root(root)
    rng = np.random.default_rng(seed)
    for gp in sorted(Path(pred_dir).glob("*.geff")):
        name = gp.stem
        try:
            scale = read_scale(zarr_path(r, gt_split, name))
        except Exception as e:
            warn_once("no_scale", f"{name}: no zarr scale ({e}) -> skipped "
                                  f"entirely, it keeps its original graph")
            continue
        nodes, edges = _load(gp)
        img = None
        getter = None
        if use_image:
            try:
                img = open_image(zarr_path(r, gt_split, name))
                cache = {}

                def getter(tt, _img=img, _c=cache):
                    if tt not in _c:
                        _c[tt] = np.asarray(_img[tt])
                    return _c[tt]
            except Exception as e:
                warn_once("no_img", f"{name}: image unreadable ({e}) -> ranking "
                                    f"uses edge length only for this dataset")
        shape = img.shape if img is not None else (0, 64, 256, 256)
        if img is None:
            warn_once("shape_guess",
                      "no image, so the volume shape falls back to "
                      "(64,256,256) for the boundary filter")
        yield gp, name, nodes, edges, np.asarray(scale), shape, getter, rng


def candidates(pred_dir, root=None, gt_split="train", use_image=True,
               out_dir="/kaggle/working", seed=0):
    """Build the candidate pool and describe it. Adds nothing."""
    allrows, per = [], {}
    for gp, name, nodes, edges, scale, shape, getter, rng in _iter(
            pred_dir, root, gt_split, use_image, seed):
        rows = _candidates_one(nodes, edges, scale, shape, getter, rng)
        per[name] = len(rows)
        for r in rows:
            r["dataset"] = name
        allrows += rows
        print(f"  {name}: {len(nodes)} nodes, {len(rows)} candidates", flush=True)

    if not allrows:
        print("no candidates -- check that the graphs really contain "
              "appearance nodes")
        return None
    rows = _score(allrows)
    e = np.array([r["edge_um"] for r in rows])
    pr = np.array([r["peak_ratio"] for r in rows], float)
    print(f"\n{len(rows)} candidates over {len(per)} datasets "
          f"(median {int(np.median(list(per.values())))} per dataset)")
    print(f"  edge length um   p10 {np.percentile(e,10):.2f}  "
          f"p50 {np.percentile(e,50):.2f}  p90 {np.percentile(e,90):.2f}")
    ok = np.isfinite(pr)
    if ok.any():
        print(f"  parent peak ratio p10 {np.percentile(pr[ok],10):.2f}  "
              f"p50 {np.percentile(pr[ok],50):.2f}  p90 {np.percentile(pr[ok],90):.2f}"
              f"   ({ok.sum()}/{len(pr)} computed)")
    else:
        print("  parent peak ratio: not computed (no image)")

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "division_candidates.csv")
    cols = ["dataset", "t", "parent", "orphan", "sibling", "edge_um",
            "sister_um", "track_len", "boundary_um", "peak_ratio", "score"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})
    print(f"wrote {p} -- send this file, do not print it")
    warning_summary()
    return {"n_candidates": len(rows), "n_datasets": len(per)}


def apply_cap(pred_dir, out_dir, cap, root=None, gt_split="train",
              use_image=True, seed=0):
    """Write modified geffs with at most `cap` recovered forks per dataset."""
    os.makedirs(out_dir, exist_ok=True)
    added_total = 0
    for gp, name, nodes, edges, scale, shape, getter, rng in _iter(
            pred_dir, root, gt_split, use_image, seed):
        rows = _score(_candidates_one(nodes, edges, scale, shape, getter, rng))
        used_p, used_o, new = set(), set(), []
        for r in rows[: max(0, int(cap)) * 4]:
            if len(new) >= cap:
                break
            if r["parent"] in used_p or r["orphan"] in used_o:
                continue
            used_p.add(r["parent"]); used_o.add(r["orphan"])
            new.append([r["parent"], r["orphan"]])
        e2 = np.vstack([edges, np.array(new, int)]) if new else edges
        _save(os.path.join(out_dir, f"{name}.geff"), nodes, e2)
        added_total += len(new)
    return added_total


def sweep(pred_dir, caps=(0, 5, 15, 40), root=None, gt_split="train",
          use_image=True, work="/kaggle/working/dr_sweep", seed=0):
    """Score each cap with the OFFICIAL metric, divisions included."""
    import sys
    sys.path.insert(0, "/kaggle/working/repo/scripts")
    from evaluate import evaluate_pairs
    from tracking_cellmot.metrics import summarise

    r = find_root(root)
    gt_dir = r / gt_split
    out = []
    for cap in caps:
        d = os.path.join(work, f"cap{cap}")
        shutil.rmtree(d, ignore_errors=True)
        n = apply_cap(pred_dir, d, cap, root, gt_split, use_image, seed)
        rows, _ = evaluate_pairs(d, gt_dir)
        s = summarise(rows)
        out.append((cap, n, s))
        print(f"cap={cap:3d}  added {n:4d} forks  score={s['score']:.4f}  "
              f"adj_J={s['adj_edge_jaccard']:.4f}  "
              f"div_J={s['division_jaccard']:.4f} "
              f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})",
              flush=True)

    print(f"\n{'cap':>5}{'added':>7}{'score':>9}{'adj_J':>9}{'div_J':>8}"
          f"{'divTP':>7}{'divFP':>7}")
    for cap, n, s in out:
        print(f"{cap:5d}{n:7d}{s['score']:9.4f}{s['adj_edge_jaccard']:9.4f}"
              f"{s['division_jaccard']:8.4f}{s['division_tp']:7d}"
              f"{s['division_fp']:7d}")
    best = max(out, key=lambda o: o[2]["score"])
    base = [o for o in out if o[0] == 0]
    msg = ""
    if base:
        msg = f"  ({best[2]['score'] - base[0][2]['score']:+.4f} vs cap=0)"
    print(f"\nbest cap={best[0]} -> score {best[2]['score']:.4f}{msg}")
    warning_summary()
    return {"best_cap": best[0], "best_score": round(float(best[2]["score"]), 6),
            "baseline_score": (round(float(base[0][2]["score"]), 6) if base else None)}
