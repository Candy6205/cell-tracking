"""vtstatic -- do the vote candidates REPEAT in place from frame to frame, and does that
separate the texture from the cells?

WHY (2026-09-19). vttrace: 144k of the 224k vote candidates (mapvote: >= 2 single maps agree
on a floor-0.05 maximum, no shipped primary within 7 um) survive the linker as chains of
candidate-to-candidate edges; the linker accepts them because a fixed image structure sits
at the same place in consecutive frames and looks like a cell that did not move. Cells move
(1.7 um per frame at the median of the labelled steps) and the missed ones flicker;
lnpersist read the same fact from the other side on lnrescue's peaks (a peak with a peak
within 4 um in an adjacent frame was annotated at 0.0053 against 0.0088 unfiltered). So the
displacement to the nearest candidate in the adjacent frames, and the length of the run of
near-static repeats, should sort the candidates: near-zero displacement = texture.

WHAT IT DOES (CPU, minutes). Per candidate: votes k, the mean map's value f, the distance
to the nearest candidate at t-1 and t+1 (d_prev, d_next; the smaller is d_adj), the run of
consecutive frames with a candidate within STATIC_UM of the same spot (backwards +
forwards + 1), the distance to the nearest shipped primary at t-1 / t+1 (does it sit where
a tracked cell was or will be), and whether a GT node lies within 7 um at t (annotated).
Tables: annotated share and count by d_adj bin, by run length, by votes; the same for the
subset that fills a hole (a shipped primary within HOLE_UM at t-1 or t+1); the missed
annotated cells (no shipped primary within 7 um) recovered within 7 um under each rule.
Rules (kept candidates): ALL; NOSTATIC (no candidate within STATIC_UM at t-1 or t+1);
NORUN3 (run < 3); MOVING (nearest adjacent candidate between STATIC_UM and MOVE_UM); ALONE (none within MOVE_UM);
HOLE (a shipped primary within HOLE_UM at t-1 or t+1); NOSTATIC&HOLE.
Output: vtstatic_movies.csv (per movie counts) and the printed tables.

Attach: faustine649/cell-tracking (viz_cells), competition, kernel source biohub-mapvote.
"""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-19a-vtstatic"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 7.0
STATIC_UM = 1.0          # a candidate this close in an adjacent frame = the same spot
MOVE_UM = 6.0            # a cell's one-frame step stays under this (labelled steps: p99 6.9 um)
HOLE_UM = 4.0            # a shipped primary this close at t-1 or t+1 = the candidate fills a hole in a track
D_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, np.inf)]
RUN_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 6)]
RULES = ["ALL", "NOSTATIC", "NORUN3", "MOVING", "ALONE", "HOLE", "NOSTATIC&HOLE"]

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]


def nearest(tree, pts):
    if tree is None or len(pts) == 0:
        return np.full(len(pts), np.inf)
    return tree.query(pts)[0]


def static_features(cand_um, ship_um):
    """cand_um / ship_um: per frame (n, 3) arrays in um. Returns per frame dict of arrays:
    d_prev, d_next, d_adj, run (consecutive frames with a candidate within STATIC_UM, incl. t),
    d_ship_prev, d_ship_next."""
    T = len(cand_um)
    ctree = [cKDTree(c) if len(c) else None for c in cand_um]
    stree = [cKDTree(s) if len(s) else None for s in ship_um]
    out = []
    # static links: for each frame, which candidates have a same-spot candidate at t-1 / t+1
    back = [None] * T
    fwd = [None] * T
    for t in range(T):
        c = cand_um[t]
        dp = nearest(ctree[t - 1], c) if t > 0 else np.full(len(c), np.inf)
        dn = nearest(ctree[t + 1], c) if t < T - 1 else np.full(len(c), np.inf)
        back[t] = dp <= STATIC_UM
        fwd[t] = dn <= STATIC_UM
        out.append(dict(d_prev=dp, d_next=dn, d_adj=np.minimum(dp, dn),
                        d_ship_prev=nearest(stree[t - 1], c) if t > 0 else np.full(len(c), np.inf),
                        d_ship_next=nearest(stree[t + 1], c) if t < T - 1 else np.full(len(c), np.inf)))
    # run length: walk backwards and forwards through same-spot neighbours
    for t in range(T):
        c = cand_um[t]
        run = np.ones(len(c), int)
        for direction in (-1, 1):
            cur = c.copy()
            alive = np.ones(len(c), bool)
            tt = t
            while alive.any() and 0 <= tt + direction < T:
                tt += direction
                tr = ctree[tt]
                if tr is None:
                    break
                d, j = tr.query(cur[alive])
                ok = d <= STATIC_UM
                idx = np.where(alive)[0]
                run[idx[ok]] += 1
                cur[idx[ok]] = cand_um[tt][j[ok]]
                alive[idx[~ok]] = False
        out[t]["run"] = run
    return out


def rule_mask(fe, rule):
    d_adj, run = fe["d_adj"], fe["run"]
    hole = (fe["d_ship_prev"] <= HOLE_UM) | (fe["d_ship_next"] <= HOLE_UM)
    nostatic = d_adj > STATIC_UM
    if rule == "ALL":
        return np.ones(len(d_adj), bool)
    if rule == "NOSTATIC":
        return nostatic
    if rule == "NORUN3":
        return run < 3
    if rule == "MOVING":                     # an adjacent candidate one cell-step away: a moving persistent structure
        return (d_adj > STATIC_UM) & (d_adj <= MOVE_UM)
    if rule == "ALONE":                      # no adjacent candidate within a cell step: a flicker
        return d_adj > MOVE_UM
    if rule == "HOLE":
        return hole
    if rule == "NOSTATIC&HOLE":
        return nostatic & hole
    raise ValueError(rule)


if __name__ == "__main__":
    code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
    assert code is not None, "attach faustine649/cell-tracking"
    sys.path.insert(0, code)
    from viz_cells import find_root, geff_path, read_geff  # noqa: E402

    ROOT = find_root(None)
    FDET = {}
    for d_ in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4vt_*.npz"):
        FDET[d_.name[len("fdet_F4vt_"):-len(".npz")]] = d_
    SIX = sorted(n for n in FDET if n.startswith("6bba_"))
    assert len(SIX) >= 128, f"{len(SIX)} fdet_F4vt caches (attach biohub-mapvote)"
    print(f"movies {len(SIX)}", flush=True)
    OUT = pathlib.Path("/kaggle/working")
    fh = open(OUT / "vtstatic_movies.csv", "w", encoding="utf-8", newline="\n")
    fh.write("dataset,shipped,shipped_ann,cands,cands_ann,gt,gt_miss," + ",".join(f"kept_{r}" for r in RULES) + ","
             + ",".join(f"ann_{r}" for r in RULES) + "," + ",".join(f"miss_hit_{r}" for r in RULES) + "\n")
    TOT = defaultdict(float)
    BY_D = defaultdict(lambda: np.zeros(2))         # (bin, k) -> [count, annotated]
    BY_RUN = defaultdict(lambda: np.zeros(2))
    BY_D_HOLE = defaultdict(lambda: np.zeros(2))
    t0 = time.time()
    for k_, name in enumerate(SIX):
        z = np.load(FDET[name])
        T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
        cand_um, ship_um, kv, fv = [], [], [], []
        for t in range(T):
            c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
            s = np.asarray(z[f"s{t}"], float).reshape(-1)
            nn = int(z[f"nnew{t}"]) if f"nnew{t}" in z.files else 0
            n0 = len(c) - nn
            ship_um.append(c[:n0][s[:n0] >= UNET_THRESH] * SCALE)
            cand_um.append(c[n0:] * SCALE)
            kv.append(np.asarray(z[f"k{t}"], int).reshape(-1) if f"k{t}" in z.files else np.full(nn, 2))
            fv.append(np.asarray(z[f"f{t}"], float).reshape(-1) if f"f{t}" in z.files else np.full(nn, np.nan))
        gtg = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gtg["t"], int)
        g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
        fe = static_features(cand_um, ship_um)
        st = defaultdict(float)
        for t in range(T):
            gt_t = g_pos[g_t == t]
            gtree = cKDTree(gt_t) if len(gt_t) else None
            c = cand_um[t]
            st["shipped"] += len(ship_um[t])
            if gtree is not None and len(ship_um[t]):
                st["shipped_ann"] += int((gtree.query(ship_um[t])[0] <= MATCH_UM).sum())
            ann = (gtree.query(c)[0] <= MATCH_UM) if (gtree is not None and len(c)) else np.zeros(len(c), bool)
            st["cands"] += len(c)
            st["cands_ann"] += int(ann.sum())
            hole = (fe[t]["d_ship_prev"] <= HOLE_UM) | (fe[t]["d_ship_next"] <= HOLE_UM)
            for i in range(len(c)):
                kk = int(min(kv[t][i], 4))
                for lo, hi in D_BINS:
                    if lo <= fe[t]["d_adj"][i] < hi:
                        BY_D[(lo, kk)] += (1, ann[i])
                        if hole[i]:
                            BY_D_HOLE[(lo, kk)] += (1, ann[i])
                        break
                for lo, hi in RUN_BINS:
                    if lo <= fe[t]["run"][i] <= hi:
                        BY_RUN[(lo, kk)] += (1, ann[i])
                        break
            masks = {r: rule_mask(fe[t], r) for r in RULES}
            for r in RULES:
                st[f"kept_{r}"] += int(masks[r].sum())
                st[f"ann_{r}"] += int((masks[r] & ann).sum())
            if gtree is not None:
                d_gs = cKDTree(ship_um[t]).query(gt_t)[0] if len(ship_um[t]) else np.full(len(gt_t), np.inf)
                miss = d_gs > MATCH_UM
                st["gt"] += len(gt_t)
                st["gt_miss"] += int(miss.sum())
                if miss.any() and len(c):
                    for r in RULES:
                        ck = c[masks[r]]
                        if len(ck):
                            st[f"miss_hit_{r}"] += int((cKDTree(ck).query(gt_t[miss])[0] <= MATCH_UM).sum())
        for k2, v2 in st.items():
            TOT[k2] += v2
        fh.write(f"{name},{st['shipped']:.0f},{st['shipped_ann']:.0f},{st['cands']:.0f},{st['cands_ann']:.0f},{st['gt']:.0f},{st['gt_miss']:.0f},"
                 + ",".join(f"{st[f'kept_{r}']:.0f}" for r in RULES) + "," + ",".join(f"{st[f'ann_{r}']:.0f}" for r in RULES) + ","
                 + ",".join(f"{st[f'miss_hit_{r}']:.0f}" for r in RULES) + "\n")
        if (k_ + 1) % 16 == 0 or k_ == 0:
            print(f"  [{k_ + 1}/{len(SIX)}] {name}: candidates {st['cands']:,.0f}, annotated {st['cands_ann']:.0f}; NOSTATIC keeps "
                  f"{st['kept_NOSTATIC'] / max(1, st['cands']):.2f} with {st['ann_NOSTATIC']:.0f} annotated  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fh.close()
    base = TOT["shipped_ann"] / max(1, TOT["shipped"])
    print(f"\n=== VTSTATIC over {len(SIX)} movies: shipped primaries {TOT['shipped']:,.0f} (annotated share {base:.4f}); vote candidates "
          f"{TOT['cands']:,.0f}, annotated {TOT['cands_ann']:,.0f} ({TOT['cands_ann'] / max(1, TOT['cands']):.4f}); annotated GT nodes {TOT['gt']:,.0f}, "
          f"missed at 7 um {TOT['gt_miss']:,.0f} ===", flush=True)
    print(f"\n  --- by the displacement to the nearest candidate in an adjacent frame (d_adj, um), per vote count: count / annotated share / x shipped ---", flush=True)
    print(f"  {'d_adj':<12}" + "".join(f"{'k=' + str(kk):>28}" for kk in (2, 3, 4)) + f"{'all k':>28}", flush=True)
    for lo, hi in D_BINS:
        lab = f"[{lo:.1f}, {hi:.1f})" if np.isfinite(hi) else f">= {lo:.1f} / none"
        cells = []
        tot = np.zeros(2)
        for kk in (2, 3, 4):
            v = BY_D[(lo, kk)]
            tot += v
            cells.append(f"{v[0]:>10,.0f} {v[1] / max(1, v[0]):>7.4f} {(v[1] / max(1, v[0])) / max(base, 1e-9):>6.2f}x")
        cells.append(f"{tot[0]:>10,.0f} {tot[1] / max(1, tot[0]):>7.4f} {(tot[1] / max(1, tot[0])) / max(base, 1e-9):>6.2f}x")
        print(f"  {lab:<12}" + "".join(f"{c_:>28}" for c_ in cells), flush=True)
    print(f"\n  --- the same, candidates that FILL A HOLE (a shipped primary within {HOLE_UM} um at t-1 or t+1) ---", flush=True)
    for lo, hi in D_BINS:
        lab = f"[{lo:.1f}, {hi:.1f})" if np.isfinite(hi) else f">= {lo:.1f} / none"
        tot = sum((BY_D_HOLE[(lo, kk)] for kk in (2, 3, 4)), np.zeros(2))
        print(f"  {lab:<12}{tot[0]:>10,.0f} {tot[1] / max(1, tot[0]):>7.4f} {(tot[1] / max(1, tot[0])) / max(base, 1e-9):>6.2f}x", flush=True)
    print(f"\n  --- by the run of same-spot repeats (consecutive frames with a candidate within {STATIC_UM} um), all votes ---", flush=True)
    for lo, hi in RUN_BINS:
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 5 else f">= {lo}")
        tot = sum((BY_RUN[(lo, kk)] for kk in (2, 3, 4)), np.zeros(2))
        print(f"  run {lab:<8}{tot[0]:>10,.0f} {tot[1] / max(1, tot[0]):>7.4f} {(tot[1] / max(1, tot[0])) / max(base, 1e-9):>6.2f}x", flush=True)
    print(f"\n  --- admission rules: kept / annotated / share / x shipped / missed annotated cells recovered within 7 um (of {TOT['gt_miss']:,.0f}) ---", flush=True)
    for r in RULES:
        n, a, h = TOT[f"kept_{r}"], TOT[f"ann_{r}"], TOT[f"miss_hit_{r}"]
        print(f"  {r:<14}{n:>10,.0f}{a:>8,.0f}{a / max(1, n):>9.4f}{(a / max(1, n)) / max(base, 1e-9):>7.2f}x{h:>8,.0f} ({h / max(1, TOT['gt_miss']):.1%})", flush=True)
    print("  reading: mapvote admitted everything at 0.13x the shipped share and recovered 31% of the missed cells for +10% nodes. A rule that "
          "keeps most of the recovery while cutting the count several-fold is worth an attach-style bench arm; a rule whose kept share stays "
          "near 0.13x says the texture moves like the cells do.", flush=True)
    print(f"\nVTSTATIC DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("VTSTATIC OK", flush=True)
