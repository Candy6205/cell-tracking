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
    # `orig` maps each solved row back to its row in the cached `coords`, so a
    # caller can look up the model's edge probability for a pair the solver
    # dropped. `build_graph` uses the coords index as node_id, which is what
    # makes this a straight lookup rather than a re-match.
    nodes = {"t": t, "zyx": zyx, "T": T, "T_full": T, "orig": nid}
    return nodes, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


# ---------------------------------------------------------------- scoring

def velocity_adjust(coords: np.ndarray, edges: list, nodes: dict,
                    pairs: np.ndarray, scale, weight: float = 0.5,
                    radius_um: float = 5.0) -> list:
    """
    Fold a velocity-consistency penalty into the candidate probabilities.

    Why this is the model's blind spot and not ours to retrain
    ----------------------------------------------------------
    `window_size = 2`: the edge model only ever sees the two frames it is
    linking, so it **structurally cannot use velocity**. It cannot know that a
    cell was already moving in a particular direction. That is a property of
    the architecture, not of how long it trained -- the weights in use are at
    `epoch 402, best_score 0.9835`, i.e. converged.

    The public tier compensates in post-processing rather than by retraining;
    pilkwang's constants carry a velocity weight of 0.5. We never have.

    How
    ---
    Solve once. Every node that received an edge now has an incoming
    displacement. For a candidate `s -> t`, `s + v(s)` is where `s` was headed;
    the residual to `t` is how much that edge asks the cell to change course.
    Penalise the probability by it, then solve again.

    Nodes with no incoming edge have no velocity and are left untouched --
    a track start is not evidence of anything.

    This should compose with a lower `threshold`: opening the gate only helps
    if something can discriminate among the extra candidates, and velocity is
    exactly the discriminator the model lacks. The 2026-08-20 sweep found 0.25
    and 0.15 worse than 0.35, which is what running out of discrimination looks
    like.
    """
    orig = np.asarray(nodes["orig"], dtype=np.int64)
    phys = coords[:, 1:].astype(float) * np.asarray(scale, dtype=float)

    vel: dict[int, np.ndarray] = {}
    for a, b in (pairs if len(pairs) else []):
        oa, ob = int(orig[int(a)]), int(orig[int(b)])
        vel[ob] = phys[ob] - phys[oa]

    if not vel:
        warn_once("velocity_no_pairs",
                  "the first pass produced no edges, so no node has a "
                  "velocity and the second pass is a no-op -- the row below is "
                  "the first-pass result, not a velocity result")
        return edges

    out = []
    for s, t, p, d in edges:
        v = vel.get(int(s))
        if v is None:
            out.append((s, t, p, d))
            continue
        resid = float(np.linalg.norm(phys[int(t)] - (phys[int(s)] + v)))
        pen = weight * min(resid / radius_um, 1.0)
        out.append((s, t, max(0.0, float(p) - pen), d))
    return out


def score_config(cache: dict, root, threshold: float, appearance: float,
                 disappearance: float, division: float = 1.0,
                 topk: int | None = None, gate_um: float | None = None,
                 velocity_weight: float = 0.0,
                 velocity_radius_um: float = 5.0,
                 split: str = "train") -> tuple:
    """Score one configuration across every cached dataset. Returns (summary, rows)."""
    r = find_root(root)
    rows = []
    for name, c in sorted(cache.items()):
        edges = select(c, threshold, topk=topk, gate_um=gate_um)
        scale = read_scale(zarr_path(r, split, name))
        nodes, pairs = relink(c["coords"], edges, appearance, disappearance,
                              division)
        if nodes is None:
            continue
        if velocity_weight:
            # Second pass. The first solve exists only to supply velocities.
            edges = velocity_adjust(c["coords"], edges, nodes, pairs, scale,
                                    weight=velocity_weight,
                                    radius_um=velocity_radius_um)
            nodes, pairs = relink(c["coords"], edges, appearance,
                                  disappearance, division)
            if nodes is None:
                continue
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
          disappearances=(1.0, 1.5, 2.5), velocities=(0.0,),
          appearance: float = 0.0, division: float = 1.0,
          topk: int | None = None, gate_um: float | None = None,
          velocity_radius_um: float = 5.0,
          out: str = "replay_sweep.csv") -> dict:
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
    # Divisions are OFF: `division_weight 1.0` with `appearance 0.0` requires
    # edge_prob > 1.0 for a fork, so none can form. The division term is not
    # printed -- it is 0 by construction, its weight is 0.1, and on a
    # 12-dataset holdout holding ~12 division events it could not be read
    # anyway. Edges carry 90% of the score and 128,883 annotations against
    # 151. It stays in the CSV; it does not get a column here.
    print(f"\n{'thr':>6}{'disapp':>8}{'vel':>6}{'adj_J':>9}{'edge_J':>9}"
          f"{'TP':>8}{'FP':>7}{'FN':>7}{'nodes':>9}{'s':>6}")
    for thr in thresholds:
        for da in disappearances:
            for vw in velocities:
                t0 = time.time()
                s, rows = score_config(cache, root, thr, appearance, da,
                                       division, topk=topk, gate_um=gate_um,
                                       velocity_weight=vw,
                                       velocity_radius_um=velocity_radius_um)
                n_nodes = sum(r_["num_pred_nodes"] for r_ in rows)
                rec = dict(threshold=thr, disappearance=da,
                           appearance=appearance, velocity_weight=vw,
                           velocity_radius_um=velocity_radius_um,
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
                print(f"{thr:>6.2f}{da:>8.1f}{vw:>6.2f}"
                      f"{s['adj_edge_jaccard']:>9.4f}{s['edge_jaccard']:>9.4f}"
                      f"{s['edge_tp']:>8}{s['edge_fp']:>7}{s['edge_fn']:>7}"
                      f"{n_nodes:>9}{time.time() - t0:>6.0f}", flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {out} -- send this file, do not print it", flush=True)

    # Optimise adj_edge_jaccard, not score_full. With forks impossible the two
    # are equal, but naming the edge term keeps the objective explicit.
    key = "adj_edge_jaccard"
    best = max(results, key=lambda r_: r_[key])
    base = next((r_ for r_ in results
                 if r_["threshold"] == BASE["threshold"]
                 and r_["disappearance"] == BASE["disappearance"]
                 and r_["velocity_weight"] == 0.0), None)
    if base is None:
        warn_once("no_baseline_row",
                  f"the sweep did not include threshold "
                  f"{BASE['threshold']} / disappearance "
                  f"{BASE['disappearance']} / velocity 0, so there is no "
                  f"anchor to the submitted configuration and the delta below "
                  f"is unanchored")
        delta = float("nan")
    else:
        delta = best[key] - base[key]
        print(f"\nbaseline  thr {BASE['threshold']}  disapp "
              f"{BASE['disappearance']}  vel 0.00   adj_J {base[key]:.4f}"
              f"   <- the submitted configuration, must read 0.9269")
    print(f"best      thr {best['threshold']}  disapp "
          f"{best['disappearance']}  vel {best['velocity_weight']:.2f}"
          f"   adj_J {best[key]:.4f}   delta {delta:+.4f}")
    print("\nlocal absolute numbers are NOT leaderboard estimates; only the "
          "PAIRED delta above transfers.", flush=True)
    warning_summary()

    return {"n_configs": len(results), "n_datasets": results[0]["n_datasets"],
            "best_threshold": best["threshold"],
            "best_disappearance": best["disappearance"],
            "best_velocity_weight": best["velocity_weight"],
            "best_adj_edge_jaccard": best[key], "delta_vs_baseline": delta,
            "csv": os.path.abspath(out)}
