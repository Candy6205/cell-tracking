"""
survey_match.py -- pick a DENSITY-MATCHED `44b6` holdout.

Why this is not just "take some 44b6 datasets"
----------------------------------------------
Every local number this project owns comes from 12 `6bba_` datasets, and the
plan calls for a `44b6`-only holdout to probe the 0.051 local-vs-LB offset. Run
naively it cannot answer the question it is asked, because the two batches
differ in more than identity:

    44b6   median 327 true cells/frame,  0.77% of cells annotated
    6bba   median  97 true cells/frame,  9.71% of cells annotated

and `corr(edge J, true density) = -0.68`. An unmatched `44b6` holdout will
score lower **for reasons that have nothing to do with which data the weights
saw**, and the result would be read as evidence of a distribution gap.

The fix is to hold density fixed: pair each `6bba` holdout dataset with the
`44b6` dataset closest to it in true cells/frame, and compare within pairs.
Whatever difference survives that is about the batch, not the crowding.

Reuses `viz_cells.run_survey` rather than reimplementing it (rule 11); the
per-dataset CSV it already writes carries `est_n_nodes` and `T`, and true
density is their ratio.

OUTPUT DISCIPLINE (rules 2 and 10): prints one fixed-size pairing table,
writes the full survey CSV, returns nothing large.
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

SCRIPT_VERSION = "2026-08-20a"

# The 12 held-out `6bba_` datasets every recorded local number comes from.
HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]


def _density(row) -> float:
    """True cells per frame = estimated_number_of_nodes / frames."""
    try:
        est = float(row["est_n_nodes"])
        T = float(row["T"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return est / T if T else float("nan")


def load_survey(path="survey_train.csv") -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = _density(row)
            if np.isfinite(d):
                out[row["dataset"]] = d
    return out


def match(survey: dict, holdout=None, max_rel_gap: float = 0.25) -> list:
    """
    Greedy nearest-density pairing, closest pair first.

    `max_rel_gap` is the largest |d_44b6 - d_6bba| / d_6bba a pair may carry
    and still be reported as matched. A pair beyond it is printed but flagged:
    matching on a 40% density gap would reintroduce exactly the confound this
    file exists to remove.
    """
    holdout = holdout or HOLDOUT_6BBA
    src = [(n, survey[n]) for n in holdout if n in survey]
    missing = [n for n in holdout if n not in survey]
    if missing:
        print(f"[degraded] {len(missing)} holdout dataset(s) absent from the "
              f"survey and silently excluded from the pairing: {missing}",
              flush=True)

    pool = {n: d for n, d in survey.items() if n.startswith("44b6_")}
    if not pool:
        print("[degraded] no 44b6_ datasets in the survey -- nothing to pair "
              "against", flush=True)
        return []

    # All candidate pairs, closest first, then take greedily without reuse.
    cand = sorted(
        ((abs(d6 - d4) / d6, n6, d6, n4, d4)
         for n6, d6 in src for n4, d4 in pool.items()),
        key=lambda t: t[0])

    used6, used4, pairs = set(), set(), []
    for gap, n6, d6, n4, d4 in cand:
        if n6 in used6 or n4 in used4:
            continue
        used6.add(n6)
        used4.add(n4)
        pairs.append((n6, d6, n4, d4, gap))
    return sorted(pairs, key=lambda p: -p[1])


def report(survey: dict, pairs: list, max_rel_gap: float = 0.25) -> list:
    print(f"\n{'6bba holdout':<18}{'cells/f':>9}   {'44b6 match':<18}"
          f"{'cells/f':>9}{'gap':>8}")
    good = []
    for n6, d6, n4, d4, gap in pairs:
        flag = "" if gap <= max_rel_gap else "   <- too far"
        print(f"{n6:<18}{d6:>9.0f}   {n4:<18}{d4:>9.0f}{gap:>7.0%}{flag}")
        if gap <= max_rel_gap:
            good.append(n4)

    d6_all = [d for n, d in survey.items() if n.startswith("6bba_")]
    d4_all = [d for n, d in survey.items() if n.startswith("44b6_")]
    print(f"\nbatch medians: 6bba {np.median(d6_all):.0f} cells/frame "
          f"({len(d6_all)} datasets), 44b6 {np.median(d4_all):.0f} "
          f"({len(d4_all)})")
    lo = min(d for _, d, _, _, _ in pairs)
    hi = max(d for _, d, _, _, _ in pairs)
    inside = sum(1 for d in d4_all if lo <= d <= hi)
    print(f"holdout density range {lo:.0f}-{hi:.0f}; {inside} of "
          f"{len(d4_all)} 44b6 datasets fall inside it")

    print(f"\n{len(good)} of {len(pairs)} pairs are within {max_rel_gap:.0%}. "
          f"Paste into cache_candidates.HOLDOUT_44B6:")
    print("HOLDOUT_44B6 = [")
    for i in range(0, len(good), 4):
        print("    " + ", ".join(f'"{n}"' for n in good[i:i + 4]) + ",")
    print("]")

    if len(good) < 8:
        print(f"\n[degraded] only {len(good)} usable pairs. Below about 8 the "
              f"comparison is too noisy to separate batch from density, which "
              f"is the whole point -- widen max_rel_gap only if you are "
              f"willing to say so out loud when reporting the result.")
    return good


def spanning(survey: dict, holdout=None, n: int = 10) -> list:
    """
    Pick `n` 44b6 datasets spread across the density range the holdout spans.

    Strict 1:1 pairing turned out to discard the low-density half: the 6bba
    holdout reaches down to ~50 cells/frame and 44b6 has almost nothing that
    sparse, so five of twelve had no partner within 25%. Throwing those away
    leaves too few pairs to conclude anything.

    Comparing CURVES instead of pairs uses every dataset. Score both batches,
    fit edge J against log density within each, and read off the vertical
    offset between the fits inside the overlap region. Density is then a
    covariate rather than a confound, and nothing is discarded.

    This picks the 44b6 side: `n` datasets spaced evenly in log density across
    the overlap, so the fit is not dominated by one crowded end.
    """
    holdout = holdout or HOLDOUT_6BBA
    d6 = [survey[x] for x in holdout if x in survey]
    if not d6:
        print("[degraded] no holdout densities available -- cannot define an "
              "overlap region", flush=True)
        return []
    lo, hi = min(d6), max(d6)

    pool = sorted(((d, x) for x, d in survey.items()
                   if x.startswith("44b6_") and lo <= d <= hi))
    if len(pool) < n:
        print(f"[degraded] only {len(pool)} 44b6 datasets fall inside the "
              f"holdout's {lo:.0f}-{hi:.0f} cells/frame range; asking for {n} "
              f"means taking all of them and the spread will be uneven",
              flush=True)
        n = len(pool)
    if not pool:
        return []

    # even spacing in log density, nearest available dataset to each target
    targets = np.exp(np.linspace(np.log(pool[0][0]), np.log(pool[-1][0]), n))
    chosen: list = []
    for t in targets:
        best = min((p for p in pool if p[1] not in chosen),
                   key=lambda p: abs(np.log(p[0]) - np.log(t)), default=None)
        if best:
            chosen.append(best[1])

    print(f"\n{len(chosen)} 44b6 datasets spanning {lo:.0f}-{hi:.0f} "
          f"cells/frame, evenly in log density:")
    for x in chosen:
        print(f"  {x:<18}{survey[x]:>9.0f}")
    print("\nHOLDOUT_44B6 = [")
    for i in range(0, len(chosen), 4):
        print("    " + ", ".join(f'"{x}"' for x in chosen[i:i + 4]) + ",")
    print("]")
    return chosen


def run(root=None, out="survey_train.csv", max_rel_gap: float = 0.25,
        n_spanning: int = 10):
    print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
    if not pathlib.Path(out).exists():
        import viz_cells
        print(f"{out} absent -- surveying all 199 train datasets, this walks "
              f"every geff and takes a few minutes", flush=True)
        viz_cells.run_survey(root=root, split="train", out=out, log_every=25)
    else:
        print(f"reusing existing {out}", flush=True)

    survey = load_survey(out)
    print(f"{len(survey)} datasets with a usable density", flush=True)
    pairs = match(survey, max_rel_gap=max_rel_gap)
    if pairs:
        report(survey, pairs, max_rel_gap=max_rel_gap)

    # The pairing is diagnostic -- it shows how much of the range actually
    # overlaps. The spanning selection is what to cache and score.
    return spanning(survey, n=n_spanning)
