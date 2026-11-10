"""
stranded_audit.py -- are the stranded targets real cells, or spurious detections?

The question this settles
------------------------
Dumping candidates at threshold 0.05 showed that **11.8% of all targets have no
candidate parent above 0.5** — the gate the 0.876 submission ran at. Those are
structural false negatives: no linker could have connected them.

That looks like free headroom, and it is not, because stranding correlates with
over-detection at r = +0.79 and not with density at all (r = -0.07). A stranded
target that is a **spurious detection has no true parent and should stay
unlinked**; forcing an edge into it manufactures a false positive. So the 2.02x
candidate expansion could be mostly upside or mostly noise, and the difference
decides whether lowering the gate is worth a submission.

Estimating this is unnecessary. It can be counted.

What it counts
--------------
1. **Match rate.** Of the predicted nodes that are stranded, what fraction match
   a ground-truth node within the 7 um radius, against the same fraction for
   targets that do have a confident parent? Annotation is sparse — only ~2.8%
   of cells are labelled — so both rates are low in absolute terms, but they
   are depressed *equally*, so the ratio is the signal. Stranded nodes matching
   GT far less often means they are disproportionately invented.

2. **The recoverable count, exactly.** For every GT edge, find the predicted
   node its target matched. If that node is stranded, the edge **cannot** be a
   true positive at threshold 0.5, and lowering the gate is the only thing that
   could ever recover it. Counting those gives the exact number of the 450 FN
   that the gate is responsible for — not an upper bound, the real figure.

Reuses the official-equivalent matcher from `submit_baseline` (rule 11); the
node matching here is the same one the scorer uses, gated at the same 7 um.

Usage (CPU, no quota):
    import stranded_audit as sa
    sa.run(CACHE, COMP)

OUTPUT DISCIPLINE (rules 2 and 10): one row per dataset, one pooled summary,
returns a small dict. Per-node detail goes to a CSV.
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

from viz_cells import find_root, geff_path, read_geff, read_scale, zarr_path

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

from submit_baseline import _match_nodes_per_frame

SCRIPT_VERSION = "2026-08-20a"

GATE = 0.5          # the threshold the 0.876 submission ran at


def audit_one(cache_npz, root, name, split="train", gate: float = GATE) -> dict:
    z = np.load(cache_npz)
    coords = z["coords"]
    tgt, prob = z["tgt"], z["prob"]

    r = find_root(root)
    scale = np.asarray(read_scale(zarr_path(r, split, name)), dtype=float)
    gt = read_geff(geff_path(r, split, name))

    pred_t = coords[:, 0].astype(int)
    pred_phys = coords[:, 1:].astype(float) * scale
    gt_t = gt["t"].astype(int)
    gt_phys = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * scale

    matched = _match_nodes_per_frame(pred_t, pred_phys, gt_t, gt_phys)

    # A node is a "target" if it has any candidate parent at all in the cache.
    # Nodes at t = 0 never do, by construction, and are excluded throughout.
    has_any = np.zeros(len(coords), dtype=bool)
    has_conf = np.zeros(len(coords), dtype=bool)
    has_any[tgt] = True
    has_conf[tgt[prob > gate]] = True
    stranded = has_any & ~has_conf

    n_str = int(stranded.sum())
    n_conf = int(has_conf.sum())
    m_str = int(sum(1 for p in np.nonzero(stranded)[0] if int(p) in matched))
    m_conf = int(sum(1 for p in np.nonzero(has_conf)[0] if int(p) in matched))

    # ---- the exact recoverable count -------------------------------------
    # Invert the matching, then walk every GT edge and ask whether its target
    # landed on a stranded predicted node. If it did, that edge is unreachable
    # at this gate no matter how good the linker is.
    gt_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
    pred_of_gt = {g: p for p, g in matched.items()}

    n_gt_edges = gated = unmatched_target = 0
    for a, b in gt["edges"]:
        rb = gt_row.get(int(b))
        if rb is None:
            continue
        n_gt_edges += 1
        pb = pred_of_gt.get(rb)
        if pb is None:
            unmatched_target += 1        # target cell was never detected
            continue
        if stranded[pb]:
            gated += 1

    return {
        "dataset": name,
        "nodes": int(len(coords)),
        "targets": int(has_any.sum()),
        "stranded": n_str,
        "stranded_frac": n_str / max(1, int(has_any.sum())),
        "match_stranded": m_str,
        "rate_stranded": m_str / n_str if n_str else float("nan"),
        "match_confident": m_conf,
        "rate_confident": m_conf / n_conf if n_conf else float("nan"),
        "n_gt_edges": n_gt_edges,
        "gt_edges_gated": gated,
        "gt_edges_target_undetected": unmatched_target,
    }


def run(cache_dir, root, split="train", gate: float = GATE,
        out: str = "stranded_audit.csv") -> dict:
    print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
    files = sorted(pathlib.Path(cache_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {cache_dir}")

    rows = []
    print(f"\n{'dataset':<18}{'stranded':>9}{'rate_str':>10}{'rate_conf':>11}"
          f"{'ratio':>7}{'gt_edges':>10}{'gated':>7}")
    for p in files:
        row = audit_one(p, root, p.stem, split=split, gate=gate)
        rows.append(row)
        ratio = (row["rate_stranded"] / row["rate_confident"]
                 if row["rate_confident"] else float("nan"))
        print(f"{row['dataset']:<18}{row['stranded']:>9}"
              f"{row['rate_stranded']:>10.3f}{row['rate_confident']:>11.3f}"
              f"{ratio:>7.2f}{row['n_gt_edges']:>10}{row['gt_edges_gated']:>7}",
              flush=True)

    tot_str = sum(r["stranded"] for r in rows)
    tot_ms = sum(r["match_stranded"] for r in rows)
    tot_mc = sum(r["match_confident"] for r in rows)
    tot_conf = sum(r["targets"] - r["stranded"] for r in rows)
    tot_edges = sum(r["n_gt_edges"] for r in rows)
    tot_gated = sum(r["gt_edges_gated"] for r in rows)
    tot_undet = sum(r["gt_edges_target_undetected"] for r in rows)

    rate_s = tot_ms / tot_str if tot_str else float("nan")
    rate_c = tot_mc / tot_conf if tot_conf else float("nan")
    ratio = rate_s / rate_c if rate_c else float("nan")

    print(f"\n=== pooled over {len(rows)} datasets ===")
    print(f"stranded targets            {tot_str}")
    print(f"  of those, match a GT node {tot_ms}  ({rate_s:.3f})")
    print(f"targets with a confident parent {tot_conf}")
    print(f"  of those, match a GT node {tot_mc}  ({rate_c:.3f})")
    print(f"\nmatch-rate ratio stranded/confident: {ratio:.2f}")
    if ratio < 0.5:
        print("  -> stranded nodes are matched far less often. They are "
              "disproportionately SPURIOUS detections, which should stay "
              "unlinked. Lowering the gate mostly buys false positives.")
    elif ratio > 0.8:
        print("  -> stranded nodes match GT about as often as confident ones. "
              "They are largely REAL cells the gate excluded. Lowering the "
              "gate is the right lever.")
    else:
        print("  -> mixed. Some of both; the ILP's appearance/disappearance "
              "weights have to sort them, so read the 2-D sweep, not this.")

    print(f"\nGT edges                          {tot_edges}")
    print(f"  target cell never detected      {tot_undet}")
    print(f"  target detected but STRANDED    {tot_gated}   <- unreachable at "
          f"threshold {gate}, recoverable only by lowering it")
    print(f"\nRecorded FN for this configuration is 450. The gate accounts for "
          f"{tot_gated} of them.")
    if tot_gated > 0:
        print(f"That is the exact ceiling on what lowering the threshold can "
              f"recover -- {tot_gated / 450:.0%} of the FN, before any of it "
              f"is lost again to new false positives.")

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} -- send this file, do not print it", flush=True)
    warning_summary()

    return {"n_datasets": len(rows), "stranded": tot_str,
            "rate_stranded": rate_s, "rate_confident": rate_c,
            "match_ratio": ratio, "gt_edges": tot_edges,
            "gt_edges_gated": tot_gated}
