"""
link_replay.py -- replay the linker on CPU from a cached candidate set.

What this is for
----------------
`predict_video` is the only GPU stage. `build_graph` and `ILPSolver.solve` are
pure CPU. `cache_candidates.py` dumps the raw candidate edges once; everything
after that -- edge threshold, appearance/disappearance weights, top-k per
target, distance gates -- is a CPU replay costing no quota.

The question it exists to answer
--------------------------------
With `use_ilp=True` the official `PredictConfig.__post_init__` leaves
`max_parents_per_node` and `max_children_per_node` at None, confirmed in a real
run (`caps=None/None`). So the ONLY thing limiting the linker is
`threshold`, which defaults to 0.5 and our 0.876 submission never set.

The edge head applies `softmax(dim=0)`, normalising over sources: for each
target, `sum_i probs[i, j] = 1`. A target therefore keeps a candidate only if
its best parent holds more than half the column mass. Two plausible parents ->
~0.5/0.5 -> both fall below the gate -> that target emits no edge at all. That
is the `DROPPED` bucket, 60% of all edge failures, and it is worst exactly
where the score is worst (dense movies).

Sweeping `threshold` alone is not enough. `appearance_weight = 0.0` makes
starting a track free while `disappearance_weight = 1.5` makes ending one
expensive, so a lower gate hands the solver cheap ways to dodge that 1.5 and it
will link more aggressively. Rule 15: parameters interact. Sweep the pair.

Usage (CPU, no GPU quota):
    import link_replay as lr
    lr.inventory(CACHE)                       # what is cached, how wide
    lr.sweep(CACHE, thresholds=[0.50, 0.25], disappearances=[1.5])

Rule 11: the ILP and the graph construction are the OFFICIAL implementations,
imported from the repo. Only the sweep harness is ours.

OUTPUT DISCIPLINE (rules 2 and 10): `sweep` prints one fixed-size table, writes
per-dataset rows to a CSV, and returns a small dict. Never per-edge output.
"""

from __future__ import annotations

import csv
import os
import pathlib
import time

import numpy as np

from viz_cells import find_root, geff_path, read_geff, read_scale, zarr_path

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

from submit_baseline import score_one, summarise

SCRIPT_VERSION = "2026-08-20a"

# The submitted 0.876 configuration, as the sweep's origin.
BASE = dict(threshold=0.50, appearance=0.0, disappearance=1.5, division=1.0,
            edge_weight=-1.0)


# ---------------------------------------------------------------- cache I/O

def load_cache(cache_dir) -> dict:
    """Read every .npz written by cache_candidates.py. Returns {name: arrays}."""
    d = pathlib.Path(cache_dir)
    if not d.exists():
        raise FileNotFoundError(f"no cache at {d}")
    out = {}
    for p in sorted(d.glob("*.npz")):
        z = np.load(p)
        out[p.stem] = {k: z[k] for k in ("coords", "src", "tgt", "prob", "dist")}
    if not out:
        raise FileNotFoundError(f"{d} holds no .npz")
    return out


def inventory(cache_dir) -> dict:
    """
    How much wider the replay can search than the submitted configuration.

    `above_0.5` is the entire candidate set the 0.876 submission ever saw.
    `kept / above_0.5` is the headroom ratio: if it is ~1.0 the softmax
    mass-splitting story is wrong and this whole line of work is dead; if it is
    2 or more, the submitted configuration is discarding candidates in bulk.
    """
    cache = load_cache(cache_dir)
    rows = []
    print(f"{'dataset':<18}{'nodes':>8}{'cand':>10}{'>0.5':>9}"
          f"{'ratio':>7}{'targets':>9}{'maxpar':>8}")
    for name, c in sorted(cache.items()):
        n_hi = int((c["prob"] > 0.5).sum())
        n_all = len(c["prob"])
        # candidates per target, to see how far the mass actually spreads
        _, counts = np.unique(c["tgt"], return_counts=True)
        ratio = n_all / n_hi if n_hi else float("inf")
        rows.append(dict(dataset=name, nodes=len(c["coords"]), cand=n_all,
                         above_half=n_hi, ratio=ratio, targets=len(counts),
                         max_parents=int(counts.max())))
        print(f"{name:<18}{len(c['coords']):>8}{n_all:>10}{n_hi:>9}"
              f"{ratio:>7.2f}{len(counts):>9}{int(counts.max()):>8}")

    tot_all = sum(r["cand"] for r in rows)
    tot_hi = sum(r["above_half"] for r in rows)
    overall = tot_all / tot_hi if tot_hi else float("nan")
    print(f"\npooled candidates {tot_all} vs {tot_hi} above 0.5 "
          f"-> {overall:.2f}x")
    if overall < 1.15:
        print("[degraded] the candidate set barely grows below 0.5. The "
              "softmax mass-splitting hypothesis does NOT hold on this data "
              "-- do not spend a sweep on it.")
    return {"n_datasets": len(rows), "candidates": tot_all,
            "above_half": tot_hi, "ratio": overall}


# ---------------------------------------------------------------- selection

def select(c: dict, threshold: float, topk: int | None = None,
           gate_um: float | None = None) -> list:
    """
    Filter cached candidates down to the edge list one ILP run should see.

    Returns the (src, tgt, prob, dist) tuples `build_graph` expects.
    """
    keep = c["prob"] > threshold
    if gate_um is not None:
        keep &= c["dist"] <= gate_um
    src, tgt = c["src"][keep], c["tgt"][keep]
    prob, dist = c["prob"][keep], c["dist"][keep]

    if topk is not None and len(tgt):
        order = np.lexsort((-prob, tgt))
        tgt_sorted = tgt[order]
        first = np.r_[0, np.flatnonzero(np.diff(tgt_sorted)) + 1]
        sizes = np.diff(np.r_[first, len(tgt_sorted)])
        rank = np.arange(len(tgt_sorted)) - np.repeat(first, sizes)
        sel = order[rank < topk]
        src, tgt, prob, dist = src[sel], tgt[sel], prob[sel], dist[sel]

    return [(int(a), int(b), float(p), float(d))
            for a, b, p, d in zip(src, tgt, prob, dist)]


# ---------------------------------------------------------------- the solver

def _repo_modules():
    """Import the official predict module and tracksdata, or say why not."""
    try:
        import predict_unet_transformer as P
        import tracksdata as td
    except ImportError as exc:                             # pragma: no cover
        raise ImportError(
            f"{exc}. link_replay needs the official repo on sys.path and the "
            f"support pack's wheels installed -- it reuses build_graph and "
            f"ILPSolver rather than reimplementing them (rule 11)."
        ) from exc
    return P, td


def relink(coords: np.ndarray, edges: list, appearance: float,
           disappearance: float, division: float = 1.0,
           edge_weight: float = -1.0) -> tuple:
    """
    Run the official ILP over one candidate set. Pure CPU.

    Returns (nodes, edge_pairs) in the shape `score_one` wants: `nodes` is a
    dict of arrays, `edge_pairs` are row indices into it. The solver DROPS
    nodes, so the returned node set is smaller than `coords` -- that is the
    node-count penalty saving, not a bug.
    """
    P, td = _repo_modules()

    graph = P.build_graph(coords, edges)
    if graph.num_edges() == 0:
        warn_once("relink_no_edges",
                  "a candidate set produced zero edges; the ILP was skipped "
                  "and this dataset scores as pure false negatives")
        return None, None

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance,
        disappearance_weight=disappearance,
        division_weight=division,
    )
    with P.suppress_output():
        graph = solver.solve(graph)

    na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nid = np.asarray(na["node_id"].to_list(), dtype=np.int64)
    t = np.asarray(na["t"].to_list(), dtype=np.int64)
    zyx = np.stack([np.asarray(na[k].to_list(), dtype=np.float64)
                    for k in ("z", "y", "x")], axis=1)

    row_of = {int(v): i for i, v in enumerate(nid)}
    pairs = []
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for a, b in zip(ea["source_id"].to_list(), ea["target_id"].to_list()):
            ra, rb = row_of.get(int(a)), row_of.get(int(b))
            if ra is None or rb is None:
                warn_once("edge_endpoint_missing",
                          "an ILP edge referenced a node absent from "
                          "node_attrs; those edges are dropped, so the score "
                          "below is a lower bound")
                continue
            pairs.append((ra, rb))

    T = int(t.max()) + 1 if len(t) else 0
    nodes = {"t": t, "zyx": zyx, "T": T, "T_full": T}
    return nodes, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


# ---------------------------------------------------------------- scoring

def score_config(cache: dict, root, threshold: float, appearance: float,
                 disappearance: float, division: float = 1.0,
                 topk: int | None = None, gate_um: float | None = None,
                 split: str = "train") -> tuple:
    """Score one configuration across every cached dataset. Returns (summary, rows)."""
    r = find_root(root)
    rows = []
    for name, c in sorted(cache.items()):
        edges = select(c, threshold, topk=topk, gate_um=gate_um)
        nodes, pairs = relink(c["coords"], edges, appearance, disappearance,
                              division)
        if nodes is None:
            continue
        scale = read_scale(zarr_path(r, split, name))
        gt = read_geff(geff_path(r, split, name))
        row = score_one(nodes, pairs, scale, gt)
        row["dataset"] = name
        est = gt["extra"].get("estimated_number_of_nodes")
        row["n_true"] = float(est) if est else float("nan")
        if not est:
            warn_once("no_estimated_nodes",
                      f"{name} has no estimated_number_of_nodes in its geff -> "
                      f"it is EXCLUDED from adj_edge_jaccard, which is the "
                      f"scored quantity; only edge_jaccard covers it")
        rows.append(row)
    return summarise(rows), rows


def sweep(cache_dir, root=None, thresholds=(0.50, 0.35, 0.25, 0.15),
          disappearances=(1.0, 1.5, 2.5), appearance: float = 0.0,
          division: float = 1.0, topk: int | None = None,
          gate_um: float | None = None, out: str = "replay_sweep.csv") -> dict:
    """
    The 2-D sweep. Everything here is CPU; no GPU quota is consumed.

    The row at threshold 0.50 / disappearance 1.5 is the submitted 0.876
    configuration and MUST reproduce the recorded local 0.9269 on the full
    12-dataset cache. If it does not, the cache does not represent the
    submitted pipeline and nothing else in the table means anything.
    """
    print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
    cache = load_cache(cache_dir)
    print(f"{len(cache)} dataset(s) cached", flush=True)

    results = []
    print(f"\n{'thr':>6}{'disapp':>8}{'adj_J':>9}{'edge_J':>9}{'div_J':>8}"
          f"{'score':>9}{'TP':>8}{'FP':>7}{'FN':>7}{'nodes':>9}{'s':>6}")
    for thr in thresholds:
        for da in disappearances:
            t0 = time.time()
            s, rows = score_config(cache, root, thr, appearance, da, division,
                                   topk=topk, gate_um=gate_um)
            n_nodes = sum(r_["num_pred_nodes"] for r_ in rows)
            rec = dict(threshold=thr, disappearance=da, appearance=appearance,
                       division=division, topk=topk or 0,
                       gate_um=gate_um or 0.0, n_datasets=s["n"],
                       adj_edge_jaccard=s["adj_edge_jaccard"],
                       edge_jaccard=s["edge_jaccard"],
                       division_jaccard=s["division_jaccard"],
                       score_full=s["score_full"], edge_tp=s["edge_tp"],
                       edge_fp=s["edge_fp"], edge_fn=s["edge_fn"],
                       num_pred_nodes=n_nodes,
                       detect_ceiling=s["detect_ceiling"],
                       link_rate=s["link_rate"], seconds=time.time() - t0)
            results.append(rec)
            print(f"{thr:>6.2f}{da:>8.1f}{s['adj_edge_jaccard']:>9.4f}"
                  f"{s['edge_jaccard']:>9.4f}{s['division_jaccard']:>8.4f}"
                  f"{s['score_full']:>9.4f}{s['edge_tp']:>8}{s['edge_fp']:>7}"
                  f"{s['edge_fn']:>7}{n_nodes:>9}{time.time() - t0:>6.0f}",
                  flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {out} -- send this file, do not print it", flush=True)

    best = max(results, key=lambda r_: r_["score_full"])
    base = next((r_ for r_ in results
                 if r_["threshold"] == BASE["threshold"]
                 and r_["disappearance"] == BASE["disappearance"]), None)
    if base is None:
        warn_once("no_baseline_row",
                  f"the sweep did not include threshold "
                  f"{BASE['threshold']} / disappearance "
                  f"{BASE['disappearance']}, so there is no anchor to the "
                  f"submitted configuration and the deltas below are "
                  f"unanchored")
        delta = float("nan")
    else:
        delta = best["score_full"] - base["score_full"]
        print(f"baseline (thr {BASE['threshold']}, disapp "
              f"{BASE['disappearance']}) score_full {base['score_full']:.4f}")
    print(f"best     (thr {best['threshold']}, disapp "
          f"{best['disappearance']}) score_full {best['score_full']:.4f}  "
          f"delta {delta:+.4f}")
    print("\nlocal absolute numbers are NOT leaderboard estimates; only the "
          "PAIRED delta above transfers.", flush=True)
    warning_summary()

    return {"n_configs": len(results), "n_datasets": results[0]["n_datasets"],
            "best_threshold": best["threshold"],
            "best_disappearance": best["disappearance"],
            "best_score_full": best["score_full"], "delta_vs_baseline": delta,
            "csv": os.path.abspath(out)}
