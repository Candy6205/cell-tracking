"""
fork_pass.py -- put the divisions the SOLVER refused back into the graph.

The reframe this file exists for
--------------------------------
`division_ceiling.py` (commit eb407fb) corrected the definition of an orphan
and, with it, the whole picture:

    9 of the 12 GT divisions in the holdout have BOTH daughters carrying a
    confident parent edge, above 0.5, in the cached candidates.

So the division information is **not missing from the model**, and the 0.5 gate
is **not** what hides it. The solver is:

    a fork is taken iff  edge_prob > division_weight - appearance_weight

and we submit `division_weight 1.0 / appearance_weight 0.0`, i.e. it requires
`edge_prob > 1.0`. Forking is forbidden outright. The parent keeps one daughter,
the other daughter's confident incoming edge is discarded, and that daughter
ends the solve with in-degree 0.

This pass adds exactly those edges back, after the solve.

Why not `division_recovery.py`
------------------------------
That file ranks fork candidates by **geometry** -- a 3.5-11 um length band and a
sister separation -- because it was designed for a pipeline where the daughter
had no usable score. Ours does. `division_ceiling` already computes both
rankings side by side (`ranks_geom` vs `ranks_prob`) for exactly this reason.
Geometry is a proxy for the thing the cache stores directly.

Why not lower `--ilp-division-weight`
-------------------------------------
Measured 8:1 against: `division_jaccard` 0 -> 0.0308 buys +0.0031 while
`adj_edge_jaccard` loses 0.0245. The weight is global, so relaxing it relaxes
the bar for every short spurious second edge at the same time. This pass is
local: it only ever considers a pair the model already scored, and it cannot
disturb any edge the solver chose.

The accounting, which is what makes this cheap
----------------------------------------------
An added fork **never removes an existing edge**. The orphan had in-degree 0,
and the parent goes from one child to two, which is exactly the out-degree cap
the official metric applies. So per added fork the edge term moves by:

    correct fork               +1 TP, -1 FN        strictly good
    wrong, annotated endpoint  +1 FP
    wrong, unannotated cells   nothing at all

The third row is the important one. `stranded_audit.py` measured 1.5-6.1% of
predicted nodes matching a GT node at all, so most mistakes this pass can make
are invisible to the edge metric -- and the official `division_metrics.py` also
ignores a fork whose parent matches nothing. That is leniency, not an exploit:
it is the same rule that stops sparse annotation from punishing every correct
link into an unannotated cell.

What that does NOT excuse: `progress.md` records this as "the same genre as the
hub-node exploit" if it turns into fork-spamming. The cap and the probability
floor exist so the rule stays a rule.

Read-out discipline
-------------------
The 12-dataset holdout holds ~12 division events, so `division_jaccard` off it
is uninterpretable and `sweep()` says so in its own output. What IS readable on
12 datasets is the **edge term**: if `adj_edge_jaccard` does not fall, the pass
is free and the only remaining question is how many divisions it catches, which
needs a wider cache (~66 datasets for ~50 events).

Usage (CPU, no GPU quota):
    import fork_pass as fp
    fp.selftest()                                  # no data needed
    fp.sweep(CACHE, root=COMP, floors=[0.9, 0.7, 0.5, 0.3])

OUTPUT DISCIPLINE (rules 2 and 10): `sweep` prints one fixed-size table, writes
per-dataset rows to a CSV, and returns a small dict.
"""

from __future__ import annotations

import csv
import time

import numpy as np

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

    warn_once("viz_cells_import",
              "could not import warn_once/warning_summary from viz_cells -- "
              "degradations below print every time instead of being counted, "
              "and no end-of-run summary table is produced")

SCRIPT_VERSION = "2026-08-21a"

# A fork is only proposed for a pair the model scored at least this high. The
# default is the same 0.5 the linker uses, so by construction this pass only
# ever adds edges the submitted pipeline ALREADY considered confident and threw
# away for a structural reason.
DEFAULT_FLOOR = 0.5

# Per-dataset ceiling on added forks. 100 frames at ~1 division per 854 edges
# and ~10k edges per movie puts the true count in the low tens; 40 leaves room
# without letting a bad floor flood the graph.
DEFAULT_CAP = 40


def _progress(i: int, total: int, every: int, msg: str = ""):
    """
    Throttled progress logging. Defined here rather than imported from
    submit_baseline: a Kaggle Dataset copy can be a version behind, and a
    cross-file import of a PRIVATE helper turns that into an ImportError at
    module load (traps 11 and 12).
    """
    if every and (i == 1 or i == total or i % every == 0):
        print(f"  [{i}/{total}] {msg}", flush=True)


# ------------------------------------------------------------------ selection

def _degrees(n: int, pairs) -> tuple:
    """in-degree, out-degree and first-child over a solved edge list."""
    in_deg = np.zeros(n, dtype=np.int64)
    out_deg = np.zeros(n, dtype=np.int64)
    first_child: dict = {}
    for a, b in (pairs if pairs is not None and len(pairs) else []):
        a, b = int(a), int(b)
        out_deg[a] += 1
        in_deg[b] += 1
        first_child.setdefault(a, b)
    return in_deg, out_deg, first_child


def select_forks(solved: dict, pairs, cache: dict,
                 floor: float = DEFAULT_FLOOR, cap: int | None = DEFAULT_CAP,
                 parent_out_degree: int = 1) -> list:
    """
    Choose the fork edges to add to one solved graph.

    Pure function over arrays -- no ILP, no file I/O, no ground truth -- so it
    is unit-testable, and `selftest()` does exactly that.

    Parameters
    ----------
    solved, pairs
        as returned by `link_replay.relink`. `solved["orig"]` maps a solved row
        back to its row in the cached `coords`, which is how a solved pair is
        looked up in the model's own candidate probabilities. Rows that failed
        to map carry -1 and are skipped.
    cache
        one dataset's dict from `link_replay.load_cache` -- needs `src`, `tgt`,
        `prob`.
    floor
        minimum candidate probability for a pair to be proposed at all.
    cap
        at most this many forks per dataset, taken in descending probability.
        None removes the cap.
    parent_out_degree
        1 (default) means only a node that already continues into exactly one
        daughter may fork -- that is what a division is. 0 selects nodes whose
        track ENDS instead, which is a repaired ordinary link and not a
        division; it is offered only so the two can be measured apart.

    Returns
    -------
    list of (prob, parent_row, orphan_row), highest probability first.
    """
    t = np.asarray(solved["t"], dtype=np.int64)
    n = len(t)
    in_deg, out_deg, _ = _degrees(n, pairs)

    orig = np.asarray(solved.get("orig", np.full(n, -1)), dtype=np.int64)
    n_unmapped = int((orig < 0).sum())
    if n_unmapped:
        warn_once("fork_orig_unmapped",
                  f"{n_unmapped} of {n} solved nodes have no row in the cached "
                  f"coords, so no candidate probability can be looked up for "
                  f"them; they are skipped as parent and as orphan, and the "
                  f"fork count below is a lower bound")

    # Index the cache BY TARGET, and invert `orig`, so each orphan looks up
    # only its own column instead of scanning every node in the frame before
    # it. The naive loop is ~18M dict probes per dataset; this is ~30k.
    src = np.asarray(cache["src"], dtype=np.int64)
    tgt = np.asarray(cache["tgt"], dtype=np.int64)
    prob = np.asarray(cache["prob"], dtype=np.float64)
    keep = prob >= floor
    if not keep.any():
        warn_once("fork_no_candidates",
                  f"no cached candidate reaches the floor of {floor}; this "
                  f"dataset contributes zero forks, which is a property of the "
                  f"floor and not of the data")
        return []
    by_tgt: dict = {}
    for a, b, p_ in zip(src[keep], tgt[keep], prob[keep]):
        by_tgt.setdefault(int(b), []).append((float(p_), int(a)))

    solved_of_orig = {int(v): i for i, v in enumerate(orig) if v >= 0}
    if len(solved_of_orig) != n - n_unmapped:
        warn_once("fork_orig_not_injective",
                  f"two solved nodes claim the same cached coords row "
                  f"({len(solved_of_orig)} distinct for {n - n_unmapped} "
                  f"mapped); the later one wins, so a fork may be attributed "
                  f"to the wrong node")

    # eligible parents and orphans, on the solved graph
    parent_ok = (out_deg == parent_out_degree) & (orig >= 0)
    orphan_ok = (in_deg == 0) & (t > 0) & (orig >= 0)

    proposals = []
    for o in np.nonzero(orphan_ok)[0]:
        o = int(o)
        best = None
        for pr, a_orig in by_tgt.get(int(orig[o]), ()):
            p = solved_of_orig.get(a_orig)
            if p is None or not parent_ok[p]:
                continue
            if int(t[p]) != int(t[o]) - 1:
                # cache_candidates only ever emits consecutive frames, so this
                # would mean the coordinate mapping has gone wrong.
                warn_once("fork_non_consecutive",
                          f"a cached candidate joins t={int(t[p])} to "
                          f"t={int(t[o])}; skipped, and the orig mapping in "
                          f"link_replay should be re-checked before trusting "
                          f"any fork count here")
                continue
            if best is None or pr > best[0]:
                best = (pr, p, o)
        if best is not None:
            proposals.append(best)

    # Greedy, highest probability first: one new parent per orphan (in-degree
    # stays 1) and one new child per parent (out-degree stays 2, the cap the
    # official metric applies anyway).
    proposals.sort(key=lambda r: -r[0])
    used_p: set = set()
    chosen = []
    for pr, p, o in proposals:
        if p in used_p:
            continue
        used_p.add(p)
        chosen.append((pr, p, o))
        if cap is not None and len(chosen) >= cap:
            break
    if cap is not None and len(proposals) > len(chosen) and len(chosen) >= cap:
        warn_once("fork_cap_binding",
                  f"the per-dataset cap of {cap} is binding ({len(proposals)} "
                  f"proposals cleared the floor); the forks reported are the "
                  f"{cap} most confident and the rest are dropped, so a higher "
                  f"cap is a different experiment, not a longer one")
    return chosen


def apply_forks(pairs, forks) -> np.ndarray:
    """Append the chosen fork edges to a solved edge list. Never removes one."""
    base = (np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            if pairs is not None else np.zeros((0, 2), dtype=np.int64))
    if not forks:
        return base
    add = np.array([[p, o] for _, p, o in forks], dtype=np.int64)
    return np.concatenate([base, add], axis=0)


# -------------------------------------------------------------------- scoring

def score_config(cache: dict, root, threshold: float, appearance: float,
                 disappearance: float, division: float = 1.0,
                 floor: float = DEFAULT_FLOOR, cap: int | None = DEFAULT_CAP,
                 parent_out_degree: int = 1, split: str = "train",
                 every: int = 25) -> tuple:
    """
    Solve every cached dataset, add forks, score. Returns (summary, rows).

    Deliberately re-uses `link_replay.select` / `.relink` and
    `submit_baseline.score_one` rather than reimplementing either (rule 11).
    """
    import link_replay as lr
    from viz_cells import find_root, geff_path, read_geff, read_scale, zarr_path
    from submit_baseline import score_one, summarise

    if lr.SCRIPT_VERSION < "2026-08-20a":                  # pragma: no cover
        warn_once("link_replay_stale",
                  f"link_replay is {lr.SCRIPT_VERSION}, older than the "
                  f"2026-08-20a that introduced solved['orig']; without it "
                  f"every fork lookup fails and this pass adds nothing")

    r = find_root(root)
    rows = []
    items = sorted(cache.items())
    for i, (name, c) in enumerate(items, 1):
        _progress(i, len(items), every, "fork_pass")
        edges = lr.select(c, threshold)
        scale = read_scale(zarr_path(r, split, name))
        solved, pairs = lr.relink(c["coords"], edges, appearance,
                                  disappearance, division)
        if solved is None:
            continue
        gt = read_geff(geff_path(r, split, name))

        base_row = score_one(solved, pairs, scale, gt)
        forks = select_forks(solved, pairs, c, floor=floor, cap=cap,
                             parent_out_degree=parent_out_degree)
        row = score_one(solved, apply_forks(pairs, forks), scale, gt)

        row["dataset"] = name
        row["n_forks"] = len(forks)
        # The paired deltas are the point: same solve, one switch flipped.
        row["d_edge_tp"] = row["edge_tp"] - base_row["edge_tp"]
        row["d_edge_fp"] = row["edge_fp"] - base_row["edge_fp"]
        row["d_div_tp"] = row["div_tp"] - base_row["div_tp"]
        row["d_div_fp"] = row["div_fp"] - base_row["div_fp"]
        est = gt["extra"].get("estimated_number_of_nodes")
        row["n_true"] = float(est) if est else float("nan")
        if not est:
            warn_once("no_estimated_nodes",
                      f"{name} has no estimated_number_of_nodes in its geff -> "
                      f"it is EXCLUDED from adj_edge_jaccard, which is the "
                      f"scored quantity; only edge_jaccard covers it")
        rows.append(row)
    return summarise(rows), rows


def sweep(cache_dir, root=None, floors=(1.01, 0.9, 0.7, 0.5, 0.3),
          cap: int | None = DEFAULT_CAP, threshold: float = 0.40,
          appearance: float = 0.0, disappearance: float = 3.5,
          parent_out_degree: int = 1, split: str = "train",
          out: str = "fork_sweep.csv") -> dict:
    """
    Sweep the probability floor. CPU only.

    `floors` starts at 1.01, which no probability can reach, so that row is the
    NO-FORK control and must reproduce the plain replay for the same
    (threshold, disappearance). If it does not, the harness is wrong and the
    rest of the table means nothing.

    The linking defaults are the converged optimum from the 2-D replay sweep
    (threshold 0.40, disappearance 3.5), not the submitted 0.50/1.5 -- forks
    are worth measuring on top of the best linker, not the old one.
    """
    import link_replay as lr

    print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
    print("link_replay", lr.SCRIPT_VERSION, flush=True)
    cache = lr.load_cache(cache_dir)
    print(f"{len(cache)} dataset(s) cached", flush=True)

    n_div_expected = len(cache) * 151 / 199
    if n_div_expected < 50:
        print(f"[read-out warning] this cache holds about "
              f"{n_div_expected:.0f} GT division events. division_jaccard "
              f"needs ~50 to mean anything, so read the EDGE columns here and "
              f"treat the division columns as a direction only.", flush=True)

    results = []
    print(f"\n{'floor':>7}{'forks':>7}{'adj_J':>9}{'edge_J':>9}{'TP':>8}"
          f"{'FP':>7}{'FN':>7}{'dTP':>6}{'dFP':>6}{'divTP':>7}{'divFP':>7}"
          f"{'s':>6}")
    for fl in floors:
        t0 = time.time()
        s, rows = score_config(cache, root, threshold, appearance,
                               disappearance, floor=fl, cap=cap,
                               parent_out_degree=parent_out_degree,
                               split=split)
        n_forks = sum(r["n_forks"] for r in rows)
        rec = dict(floor=fl, cap=cap or 0, threshold=threshold,
                   appearance=appearance, disappearance=disappearance,
                   parent_out_degree=parent_out_degree, n_datasets=s["n"],
                   n_forks=n_forks,
                   adj_edge_jaccard=s["adj_edge_jaccard"],
                   edge_jaccard=s["edge_jaccard"],
                   division_jaccard=s["division_jaccard"],
                   score_full=s["score_full"], edge_tp=s["edge_tp"],
                   edge_fp=s["edge_fp"], edge_fn=s["edge_fn"],
                   div_tp=s["div_tp"], div_fp=s["div_fp"], div_fn=s["div_fn"],
                   d_edge_tp=sum(r["d_edge_tp"] for r in rows),
                   d_edge_fp=sum(r["d_edge_fp"] for r in rows),
                   seconds=time.time() - t0)
        results.append(rec)
        print(f"{fl:>7.2f}{n_forks:>7}{s['adj_edge_jaccard']:>9.4f}"
              f"{s['edge_jaccard']:>9.4f}{s['edge_tp']:>8}{s['edge_fp']:>7}"
              f"{s['edge_fn']:>7}{rec['d_edge_tp']:>6}{rec['d_edge_fp']:>6}"
              f"{s['div_tp']:>7}{s['div_fp']:>7}"
              f"{time.time() - t0:>6.0f}", flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {out} -- send this file, do not print it", flush=True)

    control = next((r for r in results if r["n_forks"] == 0), None)
    if control is None:
        print("[degraded] no floor in this sweep produced zero forks, so the "
              "table has no control row and the deltas below are against the "
              "weakest floor, not against no-forks.", flush=True)
        control = max(results, key=lambda r: r["floor"])
    best = max(results, key=lambda r: r["score_full"])
    print(f"\ncontrol  floor {control['floor']:.2f}  "
          f"adj_J {control['adj_edge_jaccard']:.4f}  "
          f"score {control['score_full']:.4f}")
    print(f"best     floor {best['floor']:.2f}  "
          f"adj_J {best['adj_edge_jaccard']:.4f}  "
          f"score {best['score_full']:.4f}  "
          f"delta {best['score_full'] - control['score_full']:+.4f}")
    print("\nThe edge delta is the trustworthy half: if adj_edge_jaccard does "
          "not fall, this pass is free and only the division count is in "
          "question.", flush=True)
    warning_summary()
    return {"n_rows": len(results), "control": control, "best": best,
            "csv": out}


# ------------------------------------------------------------------- selftest

def selftest(verbose: bool = True) -> dict:
    """
    Exercise the whole accounting on a hand-built graph. No data, no ILP.

    The fixture is one movie of three frames:

        track A   a0 -> a1 -> a2          a1 divides in the GT
        daughter        a1    b2          detected at t=2, in-degree 0
        junk            j1                orphan, one WEAK cached candidate
        junk            k1                orphan, one much weaker still

    The solver's edge list holds a0->a1 and a1->a2 -- and NOT a1->b2, because
    forking is forbidden. The cache still holds a1->b2 at 0.82, and that is the
    edge this pass exists to restore.
    """
    from submit_baseline import score_one

    # rows:      0=a0  1=a1  2=a2  3=b2  4=j1  5=k1
    t = np.array([0, 1, 2, 2, 1, 1], dtype=np.int64)
    zyx = np.array([[10, 10, 10], [10, 10, 12], [10, 10, 14], [10, 16, 14],
                    [10, 40, 40], [10, 50, 50]], dtype=np.float64)
    solved = {"t": t, "zyx": zyx, "T": 3, "T_full": 3,
              "orig": np.arange(6, dtype=np.int64)}
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)

    cache = {"src": np.array([1, 0, 0], dtype=np.int64),
             "tgt": np.array([3, 4, 5], dtype=np.int64),
             "prob": np.array([0.82, 0.44, 0.11])}

    checks = {}

    forks = select_forks(solved, pairs, cache, floor=0.5, cap=40)
    checks["one_fork_chosen"] = (len(forks) == 1)
    checks["fork_is_a1_to_b2"] = bool(forks and forks[0][1] == 1
                                      and forks[0][2] == 3)
    # j1 has an eligible parent (a0 continues into a1) but its candidate sits
    # at 0.44, below the floor, so it must not be proposed.
    checks["weak_candidate_rejected"] = all(o != 4 for _, _, o in forks)
    checks["weakest_rejected"] = all(o != 5 for _, _, o in forks)

    lowered = select_forks(solved, pairs, cache, floor=0.4, cap=40)
    checks["floor_admits_more"] = (len(lowered) == 2)
    checks["one_new_child_per_parent"] = (
        len({p for _, p, _ in lowered}) == len(lowered))

    capped = select_forks(solved, pairs, cache, floor=0.4, cap=1)
    checks["cap_binds"] = (len(capped) == 1 and capped[0][2] == 3)

    # parent_out_degree=0 selects track ENDS as parents. a0 continues, so it
    # drops out, and nothing else has a cached candidate: a different set.
    terminal = select_forks(solved, pairs, cache, floor=0.4, cap=40,
                            parent_out_degree=0)
    checks["terminal_parents_are_a_different_set"] = (len(terminal) == 0)

    with_forks = apply_forks(pairs, forks)
    checks["apply_is_additive"] = (
        len(with_forks) == len(pairs) + len(forks)
        and np.array_equal(with_forks[:len(pairs)], pairs))

    # ---- and now the metric, on a GT that says a1 really does divide -------
    gt = {"node_ids": np.array([100, 101, 102, 103], dtype=np.int64),
          "t": np.array([0, 1, 2, 2], dtype=np.int64),
          "z": np.array([10., 10., 10., 10.]),
          "y": np.array([10., 10., 10., 16.]),
          "x": np.array([10., 12., 14., 14.]),
          "edges": [(100, 101), (101, 102), (101, 103)],
          "extra": {"estimated_number_of_nodes": 4}}
    scale = (1.0, 1.0, 1.0)

    before = score_one(solved, pairs, scale, gt)
    after = score_one(solved, with_forks, scale, gt)

    checks["edge_tp_up_by_one"] = (after["edge_tp"] == before["edge_tp"] + 1)
    checks["edge_fn_down_by_one"] = (after["edge_fn"] == before["edge_fn"] - 1)
    checks["edge_fp_unchanged"] = (after["edge_fp"] == before["edge_fp"])
    checks["division_recovered"] = (after["div_tp"] == 1
                                    and before["div_tp"] == 0)
    checks["node_count_unchanged"] = (
        after["num_pred_nodes"] == before["num_pred_nodes"])

    # ---- the other direction: the metric must PUNISH a wrong fork ---------
    # Same graph, same added edge, but a GT in which a1 does not divide and
    # b2 is an annotated cell belonging to something else. If this case came
    # out free, "mostly free" would be a property of the harness, not of the
    # data.
    gt_nodiv = dict(gt, edges=[(100, 101), (101, 102)])
    bad_before = score_one(solved, pairs, scale, gt_nodiv)
    bad_after = score_one(solved, with_forks, scale, gt_nodiv)
    checks["wrong_fork_costs_an_edge_fp"] = (
        bad_after["edge_fp"] == bad_before["edge_fp"] + 1)
    checks["wrong_fork_costs_a_division_fp"] = (
        bad_after["div_fp"] == bad_before["div_fp"] + 1)

    n_bad = sum(1 for v in checks.values() if not v)
    if verbose:
        print("SCRIPT_VERSION", SCRIPT_VERSION)
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"\nbefore  edge TP/FP/FN {before['edge_tp']}/"
              f"{before['edge_fp']}/{before['edge_fn']}  "
              f"div TP/FP/FN {before['div_tp']}/{before['div_fp']}/"
              f"{before['div_fn']}")
        print(f"after   edge TP/FP/FN {after['edge_tp']}/"
              f"{after['edge_fp']}/{after['edge_fn']}  "
              f"div TP/FP/FN {after['div_tp']}/{after['div_fp']}/"
              f"{after['div_fn']}")
        print(f"wrong   edge TP/FP/FN {bad_after['edge_tp']}/"
              f"{bad_after['edge_fp']}/{bad_after['edge_fn']}  "
              f"div TP/FP/FN {bad_after['div_tp']}/{bad_after['div_fp']}/"
              f"{bad_after['div_fn']}   (same fork, non-dividing GT)")
        print(f"\n{len(checks) - n_bad}/{len(checks)} checks passed")
    if n_bad:
        raise AssertionError(
            f"{n_bad} selftest check(s) failed: "
            f"{[k for k, v in checks.items() if not v]}")
    return {"checks": len(checks), "failed": n_bad,
            "before": {k: before[k] for k in
                       ("edge_tp", "edge_fp", "edge_fn", "div_tp")},
            "after": {k: after[k] for k in
                      ("edge_tp", "edge_fp", "edge_fn", "div_tp")}}


if __name__ == "__main__":                                 # pragma: no cover
    selftest()
