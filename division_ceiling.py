"""
division_ceiling.py -- how much can orphan-node forks actually recover, HERE?

The borrowed number is not ours
-------------------------------
`docs/public-solutions-recon.md` records **+0.0129** for the orphan-fork pass,
measured by Kunal Desale with the official scorer. That number came off *his*
pipeline: his own `UNet3D(base=24)`, no official repo, no ILP. Its orphan pool
is not ours.

`stranded_audit.py` measured ours. Of 22,474 stranded targets only 1.5% match a
GT node, against 6.1% for targets that have a confident parent. The orphan pool
is a superset of the stranded pool -- a node with no candidate above 0.5 has
in-degree 0 after the solve by construction -- so **our orphan pool is at least
98% spurious**, and quite likely dirtier.

Worse, the fork edges the rule would add are, by definition, edges the model
scored **below 0.5**. That is why the daughter is an orphan in the first place.
The rule asks the least reliable edges in the graph to do the most precise job
in the graph.

`division_recovery.py`'s filters -- a 3.5-11 um length band, a 4 um sister
separation, a 3-frame minimum track, a 6 um boundary margin -- exist precisely
to cut that junk. Whether they are enough at our contamination level is an
empirical question, and it is cheap to answer.

What this measures
------------------
A funnel over every GT division, so the loss is attributable rather than
aggregate:

    GT divisions
      parent detected?
      both daughters detected?
      is at least one daughter stranded, i.e. is there a fork target at all?
      does that daughter sit in the 3.5-11 um band from the parent?
      are the sisters at least 4 um apart?
      is it at least 6 um from the volume boundary?

Whatever survives is the **ceiling** for this pipeline: forks the rule could
recover if it never made a mistake.

And against it, the cost side: roughly how many forks the same filters would
admit in total. A rule that admits 5,000 forks to catch 8 divisions is not a
rule, it is noise, and no weighting of the 0.1 division term rescues it.

Sample-size warning, stated in the output as well: the 12-dataset holdout
contains about 12 divisions. That is enough to see a funnel collapse and not
nearly enough for a `division_jaccard`. Treat a survivor count as a direction.
To size the effect properly the cache has to reach ~66 datasets, which is where
~50 divisions live.

Usage (CPU, no quota):
    import division_ceiling as dc
    dc.run(CACHE, COMP)
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

from viz_cells import geff_path, read_geff, read_scale, zarr_path
from submit_baseline import _match_nodes_per_frame

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

# Single source of truth for the thresholds: they belong to the rule, not here.
try:
    from division_recovery import (MIN_UM, MAX_UM, MIN_SISTER_UM,
                                   MIN_BOUNDARY_UM)
except ImportError:                                        # pragma: no cover
    MIN_UM, MAX_UM, MIN_SISTER_UM, MIN_BOUNDARY_UM = 3.5, 11.0, 4.0, 6.0
    warn_once("division_recovery_import",
              "could not import the band constants from division_recovery -- "
              "using hardcoded copies 3.5/11.0/4.0/6.0. If that file has been "
              "retuned, every number below is against the OLD rule")

SCRIPT_VERSION = "2026-08-20a"

GATE = 0.5

STAGES = ["divisions", "parent_detected", "both_daughters_detected",
          "fork_target_exists", "in_length_band", "sisters_apart",
          "off_boundary"]


def _analyse(npz, root, name, split="train", gate: float = GATE) -> dict:
    """
    CORRECTED 2026-08-20. The first version defined an orphan as a node with no
    incoming candidate above the gate. That is NOT what `division_recovery.py`
    means: its rule selects nodes with **in-degree 0 after the ILP solve**.

    The two sets differ exactly in the interesting case -- a daughter whose
    edge scores above 0.5 but which the solver refused because
    `ilp_division_weight 1.0` with `appearance_weight 0.0` makes a fork require
    `edge_prob > 1.0`, i.e. forbids it outright. The wrong definition threw
    those away and reported a ceiling of 1 of 12; they are 9 of the 12.

    So this now runs the actual ILP and reads in-degree off the solved graph.
    It costs a solve per dataset, which is why this is a CPU kernel and not an
    inline check.
    """
    import link_replay as lr

    z = np.load(npz)
    coords, tgt, prob = z["coords"], z["tgt"], z["prob"]
    src = z["src"]

    scale = np.asarray(read_scale(zarr_path(root, split, name)), dtype=float)
    gt = read_geff(geff_path(root, split, name))

    # Solve exactly the submitted configuration, then work on ITS node set.
    edges = lr.select({"src": src, "tgt": tgt, "prob": prob,
                       "dist": z["dist"]}, gate)
    solved, pairs = lr.relink(coords, edges, lr.BASE["appearance"],
                              lr.BASE["disappearance"], lr.BASE["division"])
    if solved is None:
        warn_once("solve_empty",
                  f"{name} produced no solved graph; it contributes zeros to "
                  f"the funnel below and the totals are NOT over all datasets")
        return dict.fromkeys(STAGES + ["orphans", "fork_pool_upper"], 0) | \
            {"dataset": name}

    pred_t = solved["t"].astype(int)
    # `relink` returns VOXEL coordinates -- `score_one` scales them itself, so
    # the graph never carries micrometres. Forgetting this once already cost a
    # run: gt_phys was in um, pred was in voxels, and the 7 um gate matched
    # nothing, which printed as "0 of 12 parents detected" and looked like a
    # finding rather than a bug.
    pred_phys = solved["zyx"].astype(float) * scale
    gt_t = gt["t"].astype(int)
    gt_phys = np.stack([gt["z"], gt["y"], gt["x"]], axis=1) * scale

    matched = _match_nodes_per_frame(pred_t, pred_phys, gt_t, gt_phys)
    pred_of_gt = {g: p for p, g in matched.items()}

    # Rule 1: a matcher that silently matches almost nothing produces a
    # plausible-looking funnel of zeros. Sparse annotation puts the real rate
    # near 6%, so anything under 1% means the coordinate systems disagree.
    rate = len(matched) / max(1, len(pred_t))
    if rate < 0.01:
        warn_once("match_rate_collapsed",
                  f"{name}: only {len(matched)} of {len(pred_t)} solved nodes "
                  f"matched a GT node ({rate:.4f}). Expected ~0.06. The funnel "
                  f"below is measuring a coordinate mismatch, not the data")

    # in-degree / out-degree ON THE SOLVED GRAPH -- the rule's actual domain
    n = len(pred_t)
    has_conf_in = np.zeros(n, dtype=bool)
    has_conf_out = np.zeros(n, dtype=bool)
    if len(pairs):
        has_conf_out[pairs[:, 0]] = True
        has_conf_in[pairs[:, 1]] = True

    # volume extent in physical units, for the boundary filter
    hi = pred_phys.max(axis=0) if len(pred_phys) else np.zeros(3)
    lo = pred_phys.min(axis=0) if len(pred_phys) else np.zeros(3)

    def off_boundary(p):
        return bool(np.all(p - lo >= MIN_BOUNDARY_UM) and
                    np.all(hi - p >= MIN_BOUNDARY_UM))

    gt_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
    children: dict[int, list[int]] = {}
    for a, b in gt["edges"]:
        ra, rb = gt_row.get(int(a)), gt_row.get(int(b))
        if ra is None or rb is None:
            continue
        children.setdefault(ra, []).append(rb)

    c = dict.fromkeys(STAGES, 0)
    for parent, kids in children.items():
        if len(kids) < 2:
            continue
        c["divisions"] += 1

        pp = pred_of_gt.get(parent)
        if pp is None:
            continue
        c["parent_detected"] += 1

        pk = [pred_of_gt.get(k) for k in kids[:2]]
        if any(x is None for x in pk):
            continue
        c["both_daughters_detected"] += 1

        # The fork target is a daughter the linker could not reach: no
        # confident incoming candidate, so in-degree 0 after the solve.
        orphans = [x for x in pk if not has_conf_in[x]]
        if not orphans:
            continue
        c["fork_target_exists"] += 1

        o = orphans[0]
        d = float(np.linalg.norm(pred_phys[o] - pred_phys[pp]))
        if not (MIN_UM <= d <= MAX_UM):
            continue
        c["in_length_band"] += 1

        other = pk[0] if pk[1] == o else pk[1]
        if float(np.linalg.norm(pred_phys[o] - pred_phys[other])) < MIN_SISTER_UM:
            continue
        c["sisters_apart"] += 1

        if not off_boundary(pred_phys[o]):
            continue
        c["off_boundary"] += 1

    # ---- the cost side, an UPPER bound on how many forks the filters admit --
    # A real orphan-fork pass also requires the parent to already have exactly
    # one child and applies a per-movie cap, so this over-counts. It is here to
    # answer "is the pool 10x the prize or 1000x", not to predict the rule.
    # A node at t = 0 has in-degree 0 by construction and is a track start, not
    # a division daughter. Counting it would inflate the pool for free.
    orphan_rows = np.nonzero(~has_conf_in & (pred_t > 0))[0]
    pool = 0
    for t in np.unique(pred_t):
        prev = np.nonzero((pred_t == t - 1) & has_conf_out)[0]
        if not len(prev):
            continue
        here = orphan_rows[pred_t[orphan_rows] == t]
        if not len(here):
            continue
        dmat = np.linalg.norm(pred_phys[here][:, None, :]
                              - pred_phys[prev][None, :, :], axis=2)
        ok = ((dmat >= MIN_UM) & (dmat <= MAX_UM)).any(axis=1)
        pool += int(ok.sum())

    c["dataset"] = name
    c["orphans"] = int(len(orphan_rows))
    c["fork_pool_upper"] = pool
    return c


def run(cache_dir, root, split="train", out="division_ceiling.csv") -> dict:
    print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
    print(f"band {MIN_UM}-{MAX_UM} um, sisters >= {MIN_SISTER_UM} um, "
          f"boundary >= {MIN_BOUNDARY_UM} um", flush=True)

    files = sorted(pathlib.Path(cache_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {cache_dir}")

    rows = [_analyse(p, root, p.stem, split=split) for p in files]
    tot = {k: sum(r[k] for r in rows) for k in STAGES}
    pool = sum(r["fork_pool_upper"] for r in rows)
    orph = sum(r["orphans"] for r in rows)

    print(f"\n=== funnel over {len(rows)} datasets ===")
    prev = None
    for k in STAGES:
        drop = "" if prev is None else f"   -{prev - tot[k]}"
        print(f"{k:<26}{tot[k]:>6}{drop}")
        prev = tot[k]

    ceiling = tot["off_boundary"]
    print(f"\nceiling: {ceiling} of {tot['divisions']} GT divisions are "
          f"recoverable by an orphan fork on this pipeline")
    print(f"orphan nodes                {orph}")
    print(f"forks the filters admit     {pool}   (upper bound)")
    if ceiling:
        print(f"pool per real division      {pool / ceiling:.0f}x")
        if pool / ceiling > 100:
            print("  -> the filters are not selective enough here. Each real "
                  "fork comes with ~%d spurious ones, every one of which is a "
                  "division FP and an edge FP. Kunal's +0.0129 does not "
                  "transfer to this orphan pool." % (pool / ceiling))
        elif pool / ceiling > 20:
            print("  -> marginal. Worth trying only with a hard per-movie cap, "
                  "and measure the EDGE term as well as the division term.")
        else:
            print("  -> selective enough to be worth running.")
    else:
        print("  -> nothing survives the funnel. The orphan-fork line is dead "
              "on this pipeline; drop it rather than tuning it.")

    print(f"\n[degraded] {tot['divisions']} divisions is a DIRECTION, not a "
          f"number. division_jaccard is meaningless below ~50 datasets, and "
          f"~50 divisions live in ~66 datasets. Do not quote a score from "
          f"this.", flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} -- send this file, do not print it", flush=True)
    warning_summary()

    return {"datasets": len(rows), "divisions": tot["divisions"],
            "ceiling": ceiling, "orphans": orph, "fork_pool_upper": pool}
