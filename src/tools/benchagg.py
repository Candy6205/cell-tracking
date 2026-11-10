"""benchagg -- the official scorer's aggregation, written once.

`summarise` in `tracking_cellmot.metrics` does NOT average the per-movie
numbers. It micro-averages `edge_jaccard` and `division_jaccard` (sum TP/FP/FN
over movies, divide once) and weight-averages `adj_edge_jaccard` per sample by

    w_i = edge_tp_i + edge_fp_i + edge_fn_i

`tools/hparapair.py` reimplemented that as a plain mean on 2026-09-08 and read
adj_J 0.9126 where the record says 0.9133 -- **four times the effect it was
measuring**. Every paired reader since has re-derived the same twenty lines,
which is three chances to make that mistake again. This is the one copy.

Anything comparing two bench arms should use `load`, `agg`, `check_anchor` and
`paired_bootstrap` from here rather than its own. What stays in the caller is
what SHOULD differ between comparisons: which anchors apply, what the arms
mean, and what the reading is.
"""
import csv

import numpy as np

KEYS = ["edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"]


def load(path):
    """{dataset: {column: float}} from a bench per-movie CSV."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = {}
        for k, v in r.items():
            if k in ("dataset", "fold", "crop", "arm"):
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = float("nan")
        out[r["dataset"]] = d
    return out


def agg(rows):
    """summarise()'s aggregation, verbatim. See the module docstring."""
    etp = sum(r["edge_tp"] for r in rows)
    efp = sum(r["edge_fp"] for r in rows)
    efn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    ej = etp / max(1e-9, etp + efp + efn)
    dj = dtp / max(1e-9, dtp + dfp + dfn)
    ok = [r for r in rows if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in ok]
    tw = sum(w)
    aj = (sum(wi * r["adj_edge_jaccard"] for wi, r in zip(w, ok)) / tw
          if tw else float("nan"))
    return dict(edge_jaccard=ej, division_jaccard=dj, adj_edge_jaccard=aj,
                score=aj + 0.1 * dj)


def div_cells(rows):
    return (int(sum(r["division_tp"] for r in rows)),
            int(sum(r["division_fp"] for r in rows)),
            int(sum(r["division_fn"] for r in rows)))


def check_anchor(tag, rows, want, cells=None, tol=0.0001):
    """Assert an arm reproduces its own log. Returns the aggregate.

    A comparison whose CONTROL does not reproduce the record is not a
    comparison; this raises rather than printing a warning, because a warning
    scrolls past.
    """
    g = agg(rows)
    fail = []
    for k, w in want.items():
        ok = abs(g[k] - w) <= tol
        print(f"  {tag:<6} {k:<20} got {g[k]:.4f}  want {w:.4f}  "
              f"{'OK' if ok else 'MISS'}")
        if not ok:
            fail.append((k, g[k], w))
    if cells is not None:
        got = div_cells(rows)
        ok = got == tuple(cells)
        print(f"  {tag:<6} {'division tp/fp/fn':<20} got {got}  "
              f"want {tuple(cells)}  {'OK' if ok else 'MISS'}")
        if not ok:
            fail.append(("division cells", got, tuple(cells)))
    assert not fail, f"{tag} does not reproduce the record: {fail}"
    return g


def paired_bootstrap(A, H, names, draws=4000, seed=0):
    """Resample the MOVIES once per draw and apply the same draw to both arms,
    so the movie draw cancels and what is left is the treatment.

    Returns {metric: (mean, lo5, hi95, P(>0))}.
    """
    rng = np.random.default_rng(seed)
    D = {k: [] for k in KEYS}
    for _ in range(draws):
        idx = rng.integers(0, len(names), len(names))
        nm = [names[i] for i in idx]
        a, h = agg([A[n] for n in nm]), agg([H[n] for n in nm])
        for k in KEYS:
            D[k].append(h[k] - a[k])
    out = {}
    for k in KEYS:
        d = np.asarray(D[k])
        lo, hi = np.percentile(d, [5, 95])
        out[k] = (float(np.mean(d)), float(lo), float(hi), float(np.mean(d > 0)))
    return out


def print_bootstrap(res):
    print(f"  {'metric':<22}{'mean d':>10}{'90% interval':>26}{'P(>0)':>8}")
    for k in KEYS:
        m, lo, hi, p = res[k]
        print(f"  {k:<22}{m:>+10.5f}{f'[{lo:+.5f}, {hi:+.5f}]':>26}{p:>8.3f}")


def print_per_movie(A, H, names, cols=("adj_edge_jaccard", "edge_jaccard")):
    print(f"  {'metric':<22}{'up':>12}{'flat':>7}{'median':>11}"
          f"{'p10':>11}{'p90':>11}")
    for k in cols:
        d = np.array([H[n][k] - A[n][k] for n in names if k in A[n]])
        d = d[d == d]
        if not len(d):
            continue
        print(f"  {k:<22}{f'{int((d > 0).sum())}/{len(d)}':>12}"
              f"{int((d == 0).sum()):>7}{np.median(d):>+11.5f}"
              f"{np.percentile(d, 10):>+11.5f}{np.percentile(d, 90):>+11.5f}")


def print_concentration(A, H, names, k="adj_edge_jaccard", top=8):
    """An aggregate cannot show whether one movie carries the whole delta."""
    d = np.array([H[n][k] - A[n][k] for n in names])
    w = np.array([A[n]["edge_tp"] + A[n]["edge_fp"] + A[n]["edge_fn"]
                  for n in names])
    o = np.argsort(-np.abs(d * w))
    print(f"  {'movie':<20}{'weight':>9}{'d':>10}{'contribution':>14}")
    for i in o[:top]:
        print(f"  {names[i]:<20}{w[i]:>9.0f}{d[i]:>+10.5f}"
              f"{d[i] * w[i] / w.sum():>+14.6f}")
    share = abs(d[o[:top]] * w[o[:top]]).sum() / max(1e-9, abs(d * w).sum())
    print(f"  top {top} carry {share * 100:.0f}% of the total absolute "
          f"movement, over {len(names)} movies")
    tot = float(np.sum(d * w))
    wo = float(np.sum(np.delete(d * w, o[0])))
    print(f"  dropping the single largest mover flips the sign: "
          f"{'YES' if np.sign(wo) != np.sign(tot) else 'no'}")
