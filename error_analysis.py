"""
error_analysis.py -- classify EVERY edge error, instead of correlating
dataset-level scores.

The 12-dataset official baseline loses 530 GT edges (FN) and adds 606 wrong
edges (FP). Correlating twelve aggregate scores against density gives a hint;
counting what actually happened at each of the 1136 errors gives the cause.

For every GT edge we ask, in order:
    1. were BOTH endpoints detected?              -> if not: DETECTION
    2. did we link the source to the right node?  -> if yes: TP
    3. what did we link it to instead?            -> CONFUSION / SPURIOUS
    4. did we link it to nothing at all?          -> DROPPED

and for each bucket we record the GT displacement and the local crowding, so
the buckets can be compared against the TP control distribution.

Usage (CPU, ~1 min for 12 datasets):
    import error_analysis as ea
    ea.run("/kaggle/working/official_pred.csv")

OUTPUT DISCIPLINE (engineering-rules.md rule 2): fixed-size tables only,
never one line per edge. It also NEVER returns the per-edge list -- a notebook
auto-displays the last expression, so returning 10313 records dumps all of them.
Per-edge data goes to a CSV on disk instead; `run()` returns a small dict.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter

import numpy as np

from viz_cells import (
    MATCH_RADIUS_UM, find_root, geff_path, read_geff, read_scale, zarr_path,
)
from submit_baseline import _match_nodes_per_frame

try:
    from viz_cells import warn_once
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

SCRIPT_VERSION = "2026-08-18f"

BUCKETS = (
    "TP",
    "DETECTION",   # an endpoint of the GT edge was never detected
    "DROPPED",     # source detected, but we emitted no forward edge from it
    "CONFUSION",   # we linked the source to a DIFFERENT annotated cell
    "SPURIOUS",    # we linked the source to a node matching no GT cell
)


def _read_csv(csv_path):
    per: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = per.setdefault(row["dataset"], {"nodes": {}, "edges": []})
            if row["row_type"] == "node":
                d["nodes"][int(row["node_id"])] = (
                    int(row["t"]), int(row["z"]), int(row["y"]), int(row["x"]))
            else:
                d["edges"].append((int(row["source_id"]), int(row["target_id"])))
    return per


def _quant(a, qs=(10, 25, 50, 75, 90, 99)):
    if len(a) == 0:
        return "  (empty)"
    v = np.percentile(np.asarray(a, float), qs)
    return "  ".join(f"p{q}={x:5.2f}" for q, x in zip(qs, v))


def analyse_one(nodes, edges, scale, gt):
    """Return (records, fp_records) for one dataset. Coordinates in um."""
    gt_ids = gt["node_ids"]
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gt_phys = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * np.asarray(scale)
    gt_t = gt["t"].astype(int)

    pred_phys = nodes["zyx"].astype(float) * np.asarray(scale)
    pred_t = nodes["t"].astype(int)

    m = _match_nodes_per_frame(pred_t, pred_phys, gt_t, gt_phys)   # pred -> gt
    inv = {g: p for p, g in m.items()}                             # gt   -> pred

    # predicted forward edges, indexed by source
    out: dict = {}
    for s_, d_ in edges:
        s_, d_ = int(s_), int(d_)
        if int(pred_t[d_]) - int(pred_t[s_]) != 1:
            continue
        out.setdefault(s_, []).append(d_)

    # GT edges, and GT out-degree (a divider has 2)
    gt_edges = []
    gt_out_deg = Counter()
    for a, b in gt["edges"]:
        ra, rb = gt_row.get(int(a)), gt_row.get(int(b))
        if ra is None or rb is None:
            continue
        gt_edges.append((ra, rb))
        gt_out_deg[ra] += 1

    # crowding: how many PREDICTED nodes sit within the match radius of a point
    by_frame: dict = {}
    for i, t in enumerate(pred_t):
        by_frame.setdefault(int(t), []).append(i)

    def crowd(point, t):
        idx = by_frame.get(int(t))
        if not idx:
            return 0
        d = np.linalg.norm(pred_phys[idx] - point, axis=1)
        return int((d <= MATCH_RADIUS_UM).sum())

    recs = []
    for ga, gb in gt_edges:
        move = float(np.linalg.norm(gt_phys[gb] - gt_phys[ga]))
        c = crowd(gt_phys[gb], gt_t[gb])
        pa, pb = inv.get(ga), inv.get(gb)
        is_div = gt_out_deg[ga] >= 2

        if pa is None or pb is None:
            recs.append(("DETECTION", move, c, is_div, np.nan))
            continue
        targets = out.get(pa, [])
        if pb in targets:
            recs.append(("TP", move, c, is_div, move))
            continue
        if not targets:
            recs.append(("DROPPED", move, c, is_div, np.nan))
            continue
        # we linked somewhere else -- measure where
        best = min(targets, key=lambda q: np.linalg.norm(pred_phys[q] - gt_phys[gb]))
        chosen = float(np.linalg.norm(pred_phys[best] - pred_phys[pa]))
        kind = "CONFUSION" if m.get(best) is not None else "SPURIOUS"
        recs.append((kind, move, c, is_div, chosen))
    return recs


def run(csv_path="/kaggle/working/official_pred.csv", root=None,
        gt_split="train", limit=None, out_dir="/kaggle/working"):
    """Analyse, print a fixed-size report, ALSO write it to disk.

    Writes:
      <out_dir>/error_summary.txt   the report below, verbatim
      <out_dir>/error_edges.csv     one row per GT edge, for deeper digging

    Returns a small dict of counts. Never returns the per-edge records --
    a notebook would print all 10313 of them.
    """
    import io, os, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        recs = _run_inner(csv_path, root, gt_split, limit)
    if recs is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    sp = os.path.join(out_dir, "error_summary.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    ep = os.path.join(out_dir, "error_edges.csv")
    with open(ep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "gt_move_um", "pred_nodes_within_7um",
                    "source_is_divider", "linked_dist_um"])
        for r in recs:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], int(r[3]),
                        "" if r[4] != r[4] else f"{r[4]:.4f}"])
    print(f"\nwrote {sp}")
    print(f"wrote {ep}  ({len(recs)} rows) -- send this file, do not print it")
    c = Counter(x[0] for x in recs)
    return {"n_gt_edges": len(recs), **{b: c.get(b, 0) for b in BUCKETS}}


class _Tee:
    """Write to both the real stdout and a buffer, so the report is captured."""

    def __init__(self, *streams):
        self._s = streams

    def write(self, data):
        for s in self._s:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._s:
            s.flush()


def _run_inner(csv_path, root, gt_split, limit):
    r = find_root(root)
    per = _read_csv(csv_path)
    names = sorted(per)[:limit] if limit else sorted(per)
    print(f"[{csv_path}] analysing {len(names)} datasets", flush=True)

    allrecs = []
    for name in names:
        gp = geff_path(r, gt_split, name)
        if not gp.exists():
            warn_once("no_gt", f"{name}: no GT under {gt_split}/ -- SKIPPED, "
                               f"so the totals below cover fewer datasets")
            continue
        d = per[name]
        ids = sorted(d["nodes"])
        remap = {nid: i for i, nid in enumerate(ids)}
        arr = np.array([d["nodes"][nid] for nid in ids], dtype=np.int64)
        nodes = {"t": arr[:, 0].astype(np.int32), "zyx": arr[:, 1:].astype(np.int32)}
        edges = np.array([[remap[s], remap[t]] for s, t in d["edges"]
                          if s in remap and t in remap],
                         dtype=np.int64).reshape(-1, 2)
        gt = read_geff(gp)
        allrecs += analyse_one(nodes, edges, read_scale(zarr_path(r, gt_split, name)), gt)

    if not allrecs:
        print("nothing analysed")
        return None

    kinds = np.array([x[0] for x in allrecs])
    move = np.array([x[1] for x in allrecs], float)
    crowd = np.array([x[2] for x in allrecs], float)
    isdiv = np.array([x[3] for x in allrecs], bool)
    n = len(allrecs)

    print(f"\n=== {n} ground-truth edges, by outcome ===")
    print(f"{'bucket':11s} {'count':>7} {'share':>7}   "
          f"{'GT move um':>26}   {'pred nodes within 7um':>22}")
    for b in BUCKETS:
        sel = kinds == b
        if not sel.any():
            print(f"{b:11s} {0:7d} {0.0:6.1%}")
            continue
        print(f"{b:11s} {sel.sum():7d} {sel.mean():6.1%}   "
              f"med {np.median(move[sel]):5.2f}  p90 {np.percentile(move[sel], 90):5.2f}"
              f"  p99 {np.percentile(move[sel], 99):5.2f}   "
              f"med {np.median(crowd[sel]):4.1f}  p90 {np.percentile(crowd[sel], 90):4.1f}")

    print("\n=== is the error explained by MOTION? ===")
    print("GT displacement of the edge, split by whether we got it right:")
    ok = kinds == "TP"
    print(f"  linked correctly : {_quant(move[ok])}")
    print(f"  any failure      : {_quant(move[~ok])}")
    edges_hi = np.percentile(move[ok], 99) if ok.any() else np.nan
    if np.isfinite(edges_hi):
        share_fast = (move[~ok] > edges_hi).mean()
        print(f"  {share_fast:.1%} of failures move further than the 99th percentile "
              f"of correctly-linked edges ({edges_hi:.2f} um)")

    print("\n=== is the error explained by CROWDING? ===")
    print("predicted nodes within 7 um of the correct target:")
    print(f"  linked correctly : {_quant(crowd[ok])}")
    print(f"  any failure      : {_quant(crowd[~ok])}")

    print("\n=== failure rate by crowding bucket ===")
    print(f"{'nodes within 7um':>18} {'GT edges':>9} {'failure rate':>13}")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 10**6)):
        sel = (crowd >= lo) & (crowd <= hi)
        if sel.sum() < 20:
            continue
        lbl = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")
        print(f"{lbl:>18} {sel.sum():9d} {(~ok)[sel].mean():12.1%}")

    print("\n=== failure rate by GT displacement bucket ===")
    print(f"{'GT move (um)':>18} {'GT edges':>9} {'failure rate':>13}")
    bnd = [0, 1, 2, 3, 5, 7, 10, 1e9]
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        sel = (move >= lo) & (move < hi)
        if sel.sum() < 20:
            continue
        lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        print(f"{lbl:>18} {sel.sum():9d} {(~ok)[sel].mean():12.1%}")

    print("\n=== divisions ===")
    if isdiv.any():
        print(f"  GT edges leaving a dividing cell : {isdiv.sum()}  "
              f"failure rate {(~ok)[isdiv].mean():.1%}")
        print(f"  GT edges leaving a normal cell   : {(~isdiv).sum()}  "
              f"failure rate {(~ok)[~isdiv].mean():.1%}")
    else:
        warn_once("no_div", "no GT divisions in this sample -- the division row "
                            "below is absent, not zero")
    return allrecs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/kaggle/working/official_pred.csv")
    p.add_argument("--root", default=None)
    p.add_argument("--limit", type=int, default=None)
    a, _ = p.parse_known_args()
    run(a.csv, root=a.root, limit=a.limit)
