"""
sweep_edge_threshold.py -- sweep the linker's edge threshold WITHOUT re-running
inference.

Why this is free
----------------
In `predict_unet_transformer.py` the greedy assignment is:

    candidates = sorted([(p, i, j) for all pairs if p > cfg.threshold], reverse=True)
    for p, i, j in candidates:
        if children[i] >= 2 or parents[j] >= 1: continue
        accept

Lowering `cfg.threshold` only APPENDS lower-probability candidates to the tail
of a list that is already sorted by probability. Every earlier decision, and the
cap state at each step, is unchanged. Therefore

    accepted(T_high)  ==  { e in accepted(T_low) : e.prob > T_high }

So one inference run at a very low threshold contains the result for every
higher threshold. Run inference ONCE at 0.05 (~5 min/dataset on a T4), then
sweep here on CPU for free.

Built-in check: the row at T = 0.5 must reproduce the baseline exactly
(adj_edge_jaccard 0.8820 on the 12 held-out datasets). If it does not, the
monotonicity argument above is wrong for this run and nothing else here is
trustworthy.

Usage:
    import sweep_edge_threshold as st
    st.run("/kaggle/working/repo/predictions/unknown/unet_transformer/split_0")
"""

from __future__ import annotations

import csv
import os

import numpy as np

from viz_cells import find_root, geff_path, read_geff, read_scale, zarr_path
from submit_baseline import score_one, summarise

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

SCRIPT_VERSION = "2026-08-18f"

DEFAULT_THRESHOLDS = (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


def _best_child_only(edges, probs):
    """Keep only the highest-probability outgoing edge per source.

    `max_children_per_node = 2` lets a source emit a second edge. Measured at
    threshold 0.05 that second edge is almost always wrong -- and it also turns
    the source into a fork, i.e. a spurious division. This filter prices it.

    NOTE this is a post-hoc filter, not a re-run of the greedy assignment with
    max_children=1: freed target capacity is not redistributed. So it is a
    LOWER bound on what the real setting would score.
    """
    best: dict = {}
    for k, (s_, _d) in enumerate(edges):
        s_ = int(s_)
        if s_ not in best or probs[k] > probs[best[s_]]:
            best[s_] = k
    keep = np.zeros(len(edges), dtype=bool)
    keep[list(best.values())] = True
    return keep


def _load_pred_geff(path):
    """Return (nodes, edges, probs) from a predicted .geff. Needs tracksdata."""
    import tracksdata as td

    res = td.graph.IndexedRXGraph.from_geff(path)
    g = res[0] if isinstance(res, tuple) else res

    na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    node_ids = na["node_id"].to_list()
    remap = {int(v): i for i, v in enumerate(node_ids)}
    nodes = {
        "t": np.asarray(na["t"].to_list(), dtype=np.int32),
        "zyx": np.stack([np.asarray(na[k].to_list(), dtype=np.float64)
                         for k in ("z", "y", "x")], axis=1),
    }

    keys = ["source_id", "target_id"]
    have_prob = "edge_prob" in g.edge_attr_keys()
    if have_prob:
        keys.append("edge_prob")
    else:
        warn_once("no_edge_prob",
                  f"{os.path.basename(path)} has no 'edge_prob' edge attribute "
                  f"-> every edge is treated as probability 1.0, so the sweep "
                  f"below is FLAT and meaningless. Re-run inference with a "
                  f"build that stores edge_prob.")
    ea = g.edge_attrs(attr_keys=keys)
    src = [remap[int(v)] for v in ea["source_id"].to_list()]
    tgt = [remap[int(v)] for v in ea["target_id"].to_list()]
    edges = np.array(list(zip(src, tgt)), dtype=np.int64).reshape(-1, 2)
    probs = (np.asarray(ea["edge_prob"].to_list(), dtype=np.float64)
             if have_prob else np.ones(len(edges)))
    return nodes, edges, probs


def _announce(names, expect=None):
    """Say out loud WHICH datasets were scored, and shout if the set surprises.

    `predict()` wipes every .geff in its output dir on each call, so reading
    that dir while a run is in progress silently scores a partial set. A
    comparison against a different number of datasets is meaningless, and the
    old one-word "3 datasets" line was far too easy to miss.
    """
    print(f"scoring {len(names)} datasets: {', '.join(names)}", flush=True)
    if expect is None:
        return
    if isinstance(expect, int):
        if len(names) != expect:
            warn_once("dataset_count",
                      f"expected {expect} datasets, found {len(names)} -- "
                      f"predict() wipes its output dir on every call, so a run "
                      f"may still be in progress. Numbers below are NOT "
                      f"comparable to any other run.")
        return
    missing, extra = set(expect) - set(names), set(names) - set(expect)
    if missing or extra:
        warn_once("dataset_set",
                  f"dataset set differs from expected: missing {sorted(missing)}, "
                  f"unexpected {sorted(extra)}. Numbers below are NOT comparable "
                  f"to a run over a different set.")


def _prob_sanity(prob_lists, edge_lists):
    """A sweep over a knob that cannot move anything is not a result."""
    allp = np.concatenate([p for p in prob_lists if len(p)]) if prob_lists else np.zeros(0)
    if len(allp) and float(allp.min()) == float(allp.max()):
        warn_once("flat_probs",
                  f"every edge has probability {float(allp.min()):.4f} -- the "
                  f"threshold sweep below CANNOT change anything and every row "
                  f"will be identical. Either edge_prob was not stored, or the "
                  f"graph came from a solver that dropped it.")
    forks = 0
    for e in edge_lists:
        if len(e):
            _, cnt = np.unique(e[:, 0], return_counts=True)
            forks += int((cnt >= 2).sum())
    print(f"predicted forks (nodes with 2+ outgoing edges): {forks}", flush=True)
    if forks == 0:
        warn_once("no_forks",
                  "the predicted graph contains NO forks at all, so the "
                  "second-child threshold has nothing to act on and every "
                  "column of the grid will be identical. division_jaccard "
                  "is structurally 0 for this graph.")


def run(pred_dir, root=None, gt_split="train", thresholds=DEFAULT_THRESHOLDS,
        out_dir="/kaggle/working", expect=None):
    """Score every threshold on the same predicted graphs. Returns a small dict.

    Detection is untouched, so the predicted NODE set -- and therefore the
    node-count penalty -- is identical in every row. Only the edges change.
    """
    from pathlib import Path

    r = find_root(root)
    pred_dir = Path(pred_dir)
    geffs = sorted(pred_dir.glob("*.geff"))
    if not geffs:
        raise SystemExit(f"no .geff files under {pred_dir}")
    print(f"loading {len(geffs)} predicted graphs from {pred_dir}", flush=True)

    loaded = []
    for gp in geffs:
        name = gp.stem
        gtp = geff_path(r, gt_split, name)
        if not gtp.exists():
            warn_once("no_gt", f"{name}: no GT under {gt_split}/ -- SKIPPED, so "
                               f"the totals cover fewer datasets than expected")
            continue
        nodes, edges, probs = _load_pred_geff(gp)
        loaded.append((name, nodes, edges, probs,
                       read_scale(zarr_path(r, gt_split, name)),
                       read_geff(gtp)))
    if not loaded:
        raise SystemExit("nothing to score")
    _announce([x[0] for x in loaded], expect)
    _prob_sanity([x[3] for x in loaded], [x[2] for x in loaded])
    print(f"sweeping {len(thresholds)} thresholds", flush=True)

    table, table1 = [], []
    for T in thresholds:
        rows, rows1 = [], []
        for name, nodes, edges, probs, scale, gt in loaded:
            sel = probs > T
            e, pr = edges[sel], probs[sel]
            est = gt["extra"].get("estimated_number_of_nodes")
            n_true = float(est) if est else float("nan")

            row = score_one(nodes, e, scale, gt)
            row["n_true"] = n_true
            rows.append(row)

            row1 = score_one(nodes, e[_best_child_only(e, pr)], scale, gt)
            row1["n_true"] = n_true
            rows1.append(row1)
        s, s1 = summarise(rows), summarise(rows1)
        table.append((T, s))
        table1.append((T, s1))
        print(f"  T={T:.2f}  adj_J={s['adj_edge_jaccard']:.4f}  "
              f"(1 child only: {s1['adj_edge_jaccard']:.4f})", flush=True)

    print(f"\n{'thresh':>7} {'adj_J':>8} {'micro_J':>8} {'TP':>7} {'FP':>6} "
          f"{'FN':>6} {'link_rate':>10} {'adj_J 1child':>13}")
    for (T, s), (_, s1) in zip(table, table1):
        print(f"{T:7.2f} {s['adj_edge_jaccard']:8.4f} {s['edge_jaccard']:8.4f} "
              f"{s['edge_tp']:7d} {s['edge_fp']:6d} {s['edge_fn']:6d} "
              f"{s['link_rate']:10.4f} {s1['adj_edge_jaccard']:13.4f}")

    b1T, b1 = max(table1, key=lambda kv: kv[1]["adj_edge_jaccard"])
    print(f"\nbest with ONE child per source: T={b1T:.2f} -> "
          f"adj_J {b1['adj_edge_jaccard']:.4f}  "
          f"(TP={b1['edge_tp']} FP={b1['edge_fp']} FN={b1['edge_fn']})")
    print("  that variant emits no forks at all, so division_jaccard would be 0 "
          "-- worth 0.0027 of score at the current 0.0265, so a gain above "
          "+0.003 in adj_J still wins overall")

    best_T, best = max(table, key=lambda kv: kv[1]["adj_edge_jaccard"])
    base = dict(table)[0.50]["adj_edge_jaccard"] if 0.50 in dict(table) else float("nan")
    print(f"\nbest threshold {best_T:.2f} -> adj_J {best['adj_edge_jaccard']:.4f}"
          + (f"  ({best['adj_edge_jaccard'] - base:+.4f} vs the 0.50 default)"
             if base == base else ""))
    if best_T == min(thresholds):
        print("[!] the optimum is at the edge of the swept range -- extend it")

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "edge_threshold_sweep.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "adj_edge_jaccard", "edge_jaccard",
                    "edge_tp", "edge_fp", "edge_fn", "link_rate",
                    "detect_ceiling", "n_datasets"])
        for T, s in table:
            w.writerow([T, f"{s['adj_edge_jaccard']:.6f}",
                        f"{s['edge_jaccard']:.6f}", s["edge_tp"], s["edge_fp"],
                        s["edge_fn"], f"{s['link_rate']:.6f}",
                        f"{s['detect_ceiling']:.6f}", s["n"]])
    print(f"wrote {p}  -- send this file, do not print it")
    warning_summary()
    return {"best_threshold": best_T,
            "best_adj_J": round(best["adj_edge_jaccard"], 6),
            "adj_J_at_0.50": round(base, 6) if base == base else None,
            "best_one_child": (round(b1T, 2), round(b1["adj_edge_jaccard"], 6)),
            "n_datasets": best["n"]}


# --------------------------------------------------------------- second child

def _fork_filter(edges, probs, t2):
    """Keep every first child; keep a source's SECOND child only if p > t2.

    The greedy assignment uses ONE threshold for both outgoing edges, so a
    source that clears 0.5 twice becomes a fork. Measured on 12 datasets the
    model emits 220 forks where 12 divisions exist -- 214 spurious. A separate,
    higher bar for the second edge keeps real divisions while dropping junk.

    Post-hoc filter: freed target capacity is not redistributed, so results are
    a LOWER bound on implementing the same rule inside the greedy loop.
    """
    order = np.argsort(-probs, kind="stable")
    seen: dict = {}
    keep = np.zeros(len(edges), dtype=bool)
    for k in order:
        s_ = int(edges[k][0])
        n = seen.get(s_, 0)
        if n == 0 or (n == 1 and probs[k] > t2):
            keep[k] = True
            seen[s_] = n + 1
    return keep


DEFAULT_T2 = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01)


def run_fork(pred_dir, root=None, gt_split="train",
             thresholds=(0.50, 0.40, 0.30), t2_values=DEFAULT_T2,
             out_dir="/kaggle/working", expect=None):
    """Grid over (edge threshold T, second-child threshold T2).

    `t2 = 1.01` means "never emit a second child" -- the one-child variant.
    Reports `score_full = adj_edge_jaccard + 0.1 * division_jaccard`, because
    this knob trades the two terms against each other and adj_J alone would
    mislead.

    The division term here is `score_divisions_simple`, a LENIENT approximation
    of the official 574-line metric. Use this grid to pick a direction, then
    confirm the winner with the official `evaluate.py`.
    """
    from pathlib import Path

    r = find_root(root)
    geffs = sorted(Path(pred_dir).glob("*.geff"))
    if not geffs:
        raise SystemExit(f"no .geff files under {pred_dir}")

    loaded, names = [], []
    for gp in geffs:
        gtp = geff_path(r, gt_split, gp.stem)
        if not gtp.exists():
            warn_once("no_gt", f"{gp.stem}: no GT under {gt_split}/ -- SKIPPED")
            continue
        nodes, edges, probs = _load_pred_geff(gp)
        loaded.append((nodes, edges, probs,
                       read_scale(zarr_path(r, gt_split, gp.stem)),
                       read_geff(gtp)))
        names.append(gp.stem)
    if not loaded:
        raise SystemExit("nothing to score")
    _announce(names, expect)
    _prob_sanity([x[2] for x in loaded], [x[1] for x in loaded])
    print(f"grid {len(thresholds)}x{len(t2_values)}", flush=True)

    grid = {}
    for T in thresholds:
        for t2 in t2_values:
            rows = []
            for nodes, edges, probs, scale, gt in loaded:
                sel = probs > T
                e, pr = edges[sel], probs[sel]
                row = score_one(nodes, e[_fork_filter(e, pr, t2)], scale, gt)
                est = gt["extra"].get("estimated_number_of_nodes")
                row["n_true"] = float(est) if est else float("nan")
                rows.append(row)
            grid[(T, t2)] = summarise(rows)
        print(f"  T={T:.2f} done", flush=True)

    hdr = "".join(f"{('never' if t2 > 1 else f'{t2:.2f}'):>9}" for t2 in t2_values)
    print(f"\nscore_full = adj_J + 0.1*div_J   (rows: edge threshold, "
          f"cols: second-child threshold)")
    print(f"{'T':>6}{hdr}")
    for T in thresholds:
        print(f"{T:6.2f}" + "".join(f"{grid[(T, t2)]['score_full']:9.4f}"
                                    for t2 in t2_values))

    print(f"\n{'T':>6} {'T2':>6} {'score_full':>11} {'adj_J':>8} {'div_J':>7} "
          f"{'divTP':>6} {'divFP':>6} {'TP':>7} {'FP':>6} {'FN':>6}")
    best = max(grid.items(), key=lambda kv: kv[1]["score_full"])
    for key in (best[0],) + tuple(k for k in grid if k != best[0]):
        s = grid[key]
        tag = "never" if key[1] > 1 else f"{key[1]:.2f}"
        print(f"{key[0]:6.2f} {tag:>6} {s['score_full']:11.4f} "
              f"{s['adj_edge_jaccard']:8.4f} {s['division_jaccard']:7.4f} "
              f"{s['div_tp']:6d} {s['div_fp']:6d} {s['edge_tp']:7d} "
              f"{s['edge_fp']:6d} {s['edge_fn']:6d}")
        if key == best[0]:
            print("  ^ best")

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "fork_threshold_grid.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_threshold", "second_child_threshold", "score_full",
                    "adj_edge_jaccard", "division_jaccard", "div_tp", "div_fp",
                    "div_fn", "edge_tp", "edge_fp", "edge_fn", "n_datasets"])
        for (T, t2), s in sorted(grid.items()):
            w.writerow([T, t2, f"{s['score_full']:.6f}",
                        f"{s['adj_edge_jaccard']:.6f}",
                        f"{s['division_jaccard']:.6f}", s["div_tp"],
                        s["div_fp"], s["div_fn"], s["edge_tp"], s["edge_fp"],
                        s["edge_fn"], s["n"]])
    print(f"\nwrote {p}  -- send this file, do not print it")
    warning_summary()
    return {"best_edge_threshold": best[0][0],
            "best_second_child_threshold": best[0][1],
            "best_score_full": round(best[1]["score_full"], 6),
            "n_datasets": best[1]["n"]}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--root", default=None)
    a, _ = p.parse_known_args()
    run(a.pred_dir, root=a.root)
