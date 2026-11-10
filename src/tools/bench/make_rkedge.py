"""rkhp.py -> rkedge.py: the SAME two rankers, edge by edge, on hpara points.

THE QUESTION (2026-09-10). On hpara coordinates the un-retrained ranker
(cell C, trained on BASE rows: 0.9298) beats the retrained one (cell D,
trained on hpara rows: 0.9282). Summed over the 128 movies D has 73 fewer
true edges and 13 more false ones than C. The fork tables locate the
retrain's effect on the DETECTION-SCORE columns (second candidates whose
target scores below ~0.36 lose ~0.05 of probability, in both coordinate
systems), but the 73 edges are not in any table: they are ordinary edges,
not forks. This kernel puts both rankers on every candidate pair of every
scored movie and writes the rows where the two arms DIFFER, plus every
labelled row, so the lost true edges can be characterised offline.

WHAT RUNS. rkhp's features (BASE rows for all 199, hpara rows for all 199),
then BOTH fold-ranker sets in one kernel:

    MODELS_BOTH["BASE"][k]  fit on LAB  (BASE primaries' rows)  = cell C's
    MODELS_BOTH["F4"][k]    fit on LAB4 (hpara  primaries' rows) = cell D's

Stage B solves each of the 128 6bba movies twice at the production setting
(DIV_W 0.55, disappearance 1.5, floor 0.48), runs the full post-link patch,
scores each graph with the official scorer, and maps each arm's ILP edges
and FINAL edges back to candidate pairs (t, i, j) through solve_one's
seed_of_node. One row per candidate pair that is labelled (source matched a
GT node with children, 5 um) OR is in either arm's ILP / final edge set
where the two arms disagree. Columns: the 18 harness features, the head's
three, p_C, p_D, label (-1 unlabelled), in_C_ilp, in_C, in_D_ilp, in_D.

ANCHORS, all hard, all on this run's own coordinates:

    ANCHOR_D187  12 holdout  adj_J 0.9204  div_J 0.0667   (round 2 / rkhp)
    +BIDIR pkl   12 holdout  adj_J 0.9252  div_J 0.1389   (round 2 / rkhp)
    C over 128   55/237/70  div_J 0.1519  adj_J 0.9146  score 0.9298 (hparabench)
    D over 128   50/222/75  div_J 0.1441  adj_J 0.9138  score 0.9282 (rkhp)

C and D are the recorded cells and must reproduce to four decimals; if either
misses, the edge table is not the recorded comparison and must not be read.

Sources: rkhp's real metadata, verbatim (working-agreement.md rule 4).
Expected ~3 h CPU: features ~70 min, stage B 2 solves x 128 movies.
"""
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "rkhp.py"
src = open(SRC, encoding="utf-8").read()


def cut(a, b, new, inclusive_b=True):
    """Replace the block from the line containing `a` to the line containing
    `b` (both unique) with `new`."""
    global src
    assert src.count(a) == 1, (a, src.count(a))
    assert src.count(b) == 1, (b, src.count(b))
    i = src.index(a)
    j = src.index(b)
    assert j > i, (a, b)
    if inclusive_b:
        j = src.index("\n", j) + 1
    src = src[:i] + new + src[j:]


# ---- 1. version -------------------------------------------------------------
assert src.count('SCRIPT_VERSION = "2026-09-09a-rkhp"') == 1
src = src.replace('SCRIPT_VERSION = "2026-09-09a-rkhp"',
                  'SCRIPT_VERSION = "2026-09-10a-rkedge"')
assert src.count('DIVW_ARMS = [("dw35", 0.35)]') == 1
src = src.replace("DIVW_ARMS = [(\"dw35\", 0.35)]   # divbench6's winner: +0.0015, both folds up",
                  "DIVW_ARMS = []   # rkedge: no re-solve arms; two rankers instead")

# ---- 2. stage A: BOTH ranker sets --------------------------------------------
STAGE_A = '''# ---- STAGE A: BOTH fold-ranker sets, C's and D's, in one kernel ------------
print("\\n=== STAGE A: two fold-ranker SETS (+BIDIR columns, poolsize "
      "hyperparameters). BASE = fit on BASE primaries' rows (cell C, "
      "hparabench's rankers); F4 = fit on hpara primaries' rows (cell D, "
      "rkhp's rankers). Same names, same folds, same seed ===", flush=True)
COLS = ARMS["+BIDIR"]
MODELS_BOTH = {"BASE": {}, "F4": {}}
for src_tag, LABS in (("BASE", LAB), ("F4", LAB4)):
    for k in (0, 1):
        t0 = time.time()
        train_names = FOUR + [n for n in SIX if FOLD_OF[n] != k]
        held = [n for n in SIX if FOLD_OF[n] == k]
        _need = [n for n in train_names if n not in LABS]
        assert not _need, (src_tag, k, _need[:4])
        Xa = np.concatenate([LABS[n][0] for n in train_names]).astype(np.float32)[:, COLS]
        ya = np.concatenate([LABS[n][1] for n in train_names])
        Xh = np.concatenate([LAB4[n][0] for n in held]).astype(np.float32)[:, COLS]
        yh = np.concatenate([LAB4[n][1] for n in held])
        perm = np.random.default_rng(0).permutation(len(ya))
        n_val = max(1000, len(ya) // 10)
        vi, ti = perm[:n_val], perm[n_val:]
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        m.fit(Xa[ti], ya[ti])
        val_auc = _auc(ya[vi], m.predict_proba(Xa[vi])[:, 1])
        m.fit(Xa, ya)
        held_auc = _auc(yh, m.predict_proba(Xh)[:, 1])
        MODELS_BOTH[src_tag][k] = m
        print(f"  {src_tag:<4} fold{k}: trained on {len(train_names)} movies / "
              f"{len(ya)} rows ({ya.mean():.3f} pos), val AUC {val_auc:.4f}; "
              f"held-out fold on hpara rows ({len(held)} movies, {len(yh)} rows) "
              f"AUC {held_auc:.4f}  {time.time() - t0:.0f}s", flush=True)
        del Xa, ya, Xh, yh
        gc.collect()
print("  expected from the records: BASE 0.9910/0.9929 val, F4 0.9916/0.9931 val; "
      "held-out on hpara rows BASE 0.9933/0.9911, F4 0.9933/0.9917", flush=True)
del LAB
gc.collect()
MODELS_FOLD = MODELS_BOTH["F4"]      # keeps rkhp's AUC block below meaningful
'''
cut("# ---- STAGE A: two out-of-fold +BIDIR rankers", "del LAB, LAB_TRAIN", STAGE_A,
    inclusive_b=True)
# the pooled-AUC block that follows still references LAB4 and MODELS_FOLD; keep it.
assert src.count("MODELS = {}                # arm_model() is kept but unused here") == 1

# ---- 3. stage B: two solves per movie, edges mapped to pairs -----------------
STAGE_B = '''import csv  # noqa: E402


def label_map(name, feats):
    """label_rows' rule, keyed: {(t, i, j): 0/1} for every labelled pair."""
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    child_of = {}
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            child_of.setdefault(ra, []).append(rb)
    gt_by_t = defaultdict(list)
    for i in range(len(g_t)):
        gt_by_t[int(g_t[i])].append(i)
    out = {}
    for t in range(len(feats)):
        pairs, rows, P, C = feats[t]
        if not pairs:
            continue
        Pu, Cu = P * SCALE[None, :], C * SCALE[None, :]

        def match(pos_u, idx_list):
            if not idx_list:
                return {}
            gtree = cKDTree(g_pos[idx_list])
            m = {}
            for di, x in enumerate(pos_u):
                dd, gi = gtree.query(x)
                if dd <= MATCH_UM:
                    m[di] = idx_list[int(gi)]
            return m

        m_p = match(Pu, gt_by_t.get(t, []))
        m_c = match(Cu, gt_by_t.get(t + 1, []))
        det_of_gt = {g: d for d, g in m_c.items()}
        for (i, j) in pairs:
            if i not in m_p:
                continue
            kids = child_of.get(m_p[i], [])
            if not kids:
                continue
            j_true = {det_of_gt.get(b) for b in kids}
            j_true.discard(None)
            out[(t, int(i), int(j))] = 1 if j in j_true else 0
    return out


def edge_keys(edges, seed_of_node):
    """Solution edges -> {(t, i, j)}; edges whose endpoints do not map to
    consecutive-frame seeds (bridges, patch-added nodes) are counted."""
    keys, unmapped = set(), 0
    for a_, b_ in edges:
        sa, sb = seed_of_node.get(a_), seed_of_node.get(b_)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            unmapped += 1
            continue
        keys.add((sa[0], sa[1], sb[1]))
    return keys, unmapped


EDGE_COLS = (["dataset", "fold", "t", "i", "j", "label", "p_C", "p_D",
              "in_C_ilp", "in_C", "in_D_ilp", "in_D"] + BASE_NAMES
             + ["p_fwd", "p_rev", "h_fused"])
EDGE_FH = open("/kaggle/working/rkedge_edges.csv", "w", encoding="utf-8", newline="\\n")
EDGE_FH.write(",".join(EDGE_COLS) + "\\n")
N_EDGE_ROWS = 0
PER_ARM = {"C": {}, "D": {}}
ROWS_ARM = {"C": [], "D": []}
UNMAPPED = {"C": defaultdict(int), "D": defaultdict(int)}
TALLY = defaultdict(int)     # movie-summed edge accounting, printed at the end

for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    lab = label_map(name, feats)
    arm_p, arm_ilp, arm_fin = {}, {}, {}
    cells = []
    for arm, src_tag in (("C", "BASE"), ("D", "F4")):
        nodes0, edges0, pool, aux0 = solve_one(dets, T, seeds, feats,
                                               MODELS_BOTH[src_tag][k], COLS,
                                               disapp=ILP_DISAPP, floor=PROD_FLOOR)
        pmap, seed_of_node = aux0
        outp = pathlib.Path(f"{TMP}/{arm}") / f"{name}.geff"
        outp.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        _nodes, _edges = patch_final(nodes0, edges0, pool, outp)
        row = score_one(outp, name)
        ROWS_ARM[arm].append(row)
        PER_ARM[arm][name] = row
        ilp_keys, un_i = edge_keys(edges0, seed_of_node)
        fin_keys, un_f = edge_keys(_edges, seed_of_node)
        UNMAPPED[arm]["ilp"] += un_i
        UNMAPPED[arm]["final"] += un_f
        UNMAPPED[arm]["final_total"] += len(_edges)
        arm_p[arm] = pmap
        arm_ilp[arm] = ilp_keys
        arm_fin[arm] = fin_keys
        cells.append(f"{arm} {_adj_of(row):.4f} div {row['division_tp']}/"
                     f"{row['division_fp']}/{row['division_fn']} "
                     f"edges {len(_edges)} (unmapped {un_f})")
        del nodes0, edges0, pool, aux0, _nodes, _edges
    # ---- the edge table ------------------------------------------------------
    n_w = 0
    only_c_true = only_d_true = only_c = only_d = 0
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), row in zip(pairs, rows):
            key = (t, int(i_), int(j_))
            lb = lab.get(key, -1)
            ic_i = key in arm_ilp["C"]
            ic = key in arm_fin["C"]
            id_i = key in arm_ilp["D"]
            idd = key in arm_fin["D"]
            differ = (ic != idd) or (ic_i != id_i)
            if lb < 0 and not differ:
                continue
            pc = arm_p["C"].get(key, (float("nan"),))[0]
            pd = arm_p["D"].get(key, (float("nan"),))[0]
            vals = [name, k, t, int(i_), int(j_), lb, f"{pc:.5f}", f"{pd:.5f}",
                    int(ic_i), int(ic), int(id_i), int(idd)]
            vals += [f"{float(row[c]):.5g}" for c in BASE_COLS]
            vals += [f"{float(row[c]):.5g}" for c in (C_PFWD, C_PREV, C_HFUSED)]
            EDGE_FH.write(",".join(str(v) for v in vals) + "\\n")
            n_w += 1
            if ic and not idd:
                only_c += 1
                only_c_true += int(lb == 1)
            if idd and not ic:
                only_d += 1
                only_d_true += int(lb == 1)
    N_EDGE_ROWS += n_w
    for key, lb in lab.items():
        TALLY["labelled"] += 1
        TALLY["true"] += int(lb == 1)
        TALLY["true_in_C"] += int(lb == 1 and key in arm_fin["C"])
        TALLY["true_in_D"] += int(lb == 1 and key in arm_fin["D"])
        TALLY["false_in_C"] += int(lb == 0 and key in arm_fin["C"])
        TALLY["false_in_D"] += int(lb == 0 and key in arm_fin["D"])
    TALLY["only_C"] += only_c
    TALLY["only_D"] += only_d
    TALLY["only_C_true"] += only_c_true
    TALLY["only_D_true"] += only_d_true
    extra = ""
    if name in HOLDOUT_6BBA:
        n_a, e_a, p_a, _ = solve_one(dets, T, seeds, feats, MODEL_D187, MCOLS_D187,
                                     disapp=1.5, floor=0.10)
        outa = pathlib.Path(f"{TMP}/anchor_d187") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_a, e_a, p_a, outa)
        ANCHOR_ROWS.append(score_one(outa, name))
        n_s, e_s, p_s, _ = solve_one(dets, T, seeds, feats, MODEL_BIDIR, MCOLS_BIDIR,
                                     disapp=ILP_DISAPP, floor=PROD_FLOOR)
        outs = pathlib.Path(f"{TMP}/soft_bidir") / f"{name}.geff"
        outs.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_s, e_s, p_s, outs)
        SOFT_ROWS.append(score_one(outs, name))
        extra = f"  anchor {_adj_of(ANCHOR_ROWS[-1]):.4f} soft {_adj_of(SOFT_ROWS[-1]):.4f}"
        del n_a, e_a, p_a, n_s, e_s, p_s
    print(f"  {name} fold{k} | " + " | ".join(cells)
          + f" | only-C {only_c} ({only_c_true} true) only-D {only_d} "
          f"({only_d_true} true) rows {n_w}{extra}  {time.time() - t0:.0f}s",
          flush=True)
    del dets, seeds, feats, lab, arm_p, arm_ilp, arm_fin
    gc.collect()
EDGE_FH.close()
print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min); edge rows written {N_EDGE_ROWS}",
      flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)
'''
cut("for name in SIX:\n    if time.time() - t_all > BUDGET_S:",
    "INCOMPLETE: {len(INCOMPLETE)} movies not attempted", STAGE_B, inclusive_b=True)

# ---- 4. report ---------------------------------------------------------------
REPORT = '''# ---- report ----------------------------------------------------------------
print("\\n=== ANCHORS / official scorer (all HARD, this run's own coordinates) ===",
      flush=True)
FAILS = []


def _check(tag, got, want, tol=ANCHOR_TOL):
    ok = abs(got - want) <= tol
    if not ok:
        FAILS.append((tag, got, want))
    return f"want {want:.4f} -> {'OK' if ok else 'MISS'} ({got - want:+.4f})"


if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):
    ra = _with_ratio(ANCHOR_ROWS)
    print(f"  ANCHOR_D187  hpara-6bba(12)  adj_J {ra['adj_edge_jaccard']:.4f}  div_J "
          f"{ra['division_jaccard']:.4f}  {_check('D187', ra['adj_edge_jaccard'], 0.9204)}",
          flush=True)
    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   hpara-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  {_check('BIDIR', rs['adj_edge_jaccard'], 0.9252)}",
          flush=True)
else:
    FAILS.append(("D187", float("nan"), 0.9204))
WANT = {"C": ((55, 237, 70), 0.1519, 0.9146, 0.9298, "hparabench round 2"),
        "D": ((50, 222, 75), 0.1441, 0.9138, 0.9282, "rkhp")}
for arm in ("C", "D"):
    rows = ROWS_ARM[arm]
    R = _with_ratio(rows)
    tp = sum(r["division_tp"] for r in rows)
    fp = sum(r["division_fp"] for r in rows)
    fn = sum(r["division_fn"] for r in rows)
    dj = tp / max(1, tp + fp + fn)
    cells, wdj, wadj, wsc, srcname = WANT[arm]
    ok_c = (tp, fp, fn) == cells
    if not ok_c:
        FAILS.append((f"{arm}_forks", tp * 1000 + fp, cells[0] * 1000 + cells[1]))
    print(f"  {arm} over {len(rows)}  div {tp}/{fp}/{fn} div_J {dj:.4f} adj_J "
          f"{R['adj_edge_jaccard']:.4f} score {R['score']:.4f}  want "
          f"{cells[0]}/{cells[1]}/{cells[2]} {wdj:.4f} {wadj:.4f} {wsc:.4f} ({srcname}) "
          f"-> forks {'OK' if ok_c else 'MISS'}, "
          f"{_check(arm + '_adj', R['adj_edge_jaccard'], wadj)}", flush=True)
    etp = sum(r["edge_tp"] for r in rows)
    efp = sum(r["edge_fp"] for r in rows)
    efn = sum(r["edge_fn"] for r in rows)
    print(f"      scorer edges TP/FP/FN {etp}/{efp}/{efn}; unmapped final edges "
          f"{UNMAPPED[arm]['final']} of {UNMAPPED[arm]['final_total']} "
          f"(ILP {UNMAPPED[arm]['ilp']})", flush=True)
print("  all anchors reproduced: the edge table IS the recorded C-vs-D comparison."
      if not FAILS else f"  ANCHOR MISS {FAILS}: the edge table is NOT the recorded "
      f"comparison; read nothing from it before the miss is explained.", flush=True)

print("\\n=== EDGE ACCOUNTING, our 5-um labels (the scorer's TP/FP are above; the "
      "two rules differ and only the DIFFERENCE between arms is read here) ===",
      flush=True)
for k_ in ("labelled", "true", "true_in_C", "true_in_D", "false_in_C", "false_in_D",
           "only_C", "only_C_true", "only_D", "only_D_true"):
    print(f"  {k_:<14} {TALLY[k_]:>9,}", flush=True)
print(f"  true edges D loses vs C (labels): {TALLY['true_in_C'] - TALLY['true_in_D']:+d}; "
      f"false edges D adds vs C: {TALLY['false_in_D'] - TALLY['false_in_C']:+d}; "
      f"scorer said -73 true / +13 false.", flush=True)

# ---- the lost true edges, sketched here; the CSV is the deliverable ----------
try:
    import pandas as pd
    E = pd.read_csv("/kaggle/working/rkedge_edges.csv")
    lostC = E[(E.label == 1) & (E.in_C == 1) & (E.in_D == 0)]
    gainD = E[(E.label == 1) & (E.in_D == 1) & (E.in_C == 0)]
    keptT = E[(E.label == 1) & (E.in_C == 1) & (E.in_D == 1)]
    print(f"\\n=== TRUE EDGES: in C only {len(lostC)}, in D only {len(gainD)}, in both "
          f"{len(keptT)} ===", flush=True)
    for tag, g in (("C only (lost by D)", lostC), ("D only", gainD), ("both", keptT)):
        if not len(g):
            continue
        print(f"  {tag:<20} p_C med {g.p_C.median():.3f}  p_D med {g.p_D.median():.3f}  "
              f"dp med {(g.p_D - g.p_C).median():+.3f} | score_j med {g.score_j.median():.3f} "
              f"score_i {g.score_i.median():.3f} | d_um {g.d_um.median():.2f} md_um "
              f"{g.md_um.median():.2f} | rank {g.rank_of_j_for_i.median():.0f} n_cand_i "
              f"{g.n_cand_i.median():.0f} | h_fused {g.h_fused.median():.3f} | "
              f"in D's ILP {g.in_D_ilp.mean():.2f} in C's ILP {g.in_C_ilp.mean():.2f}",
              flush=True)
    if len(lostC):
        print("  lost-by-D true edges by p_D band:", flush=True)
        for lo, hi in ((0, 0.3), (0.3, 0.48), (0.48, 0.6), (0.6, 0.8), (0.8, 1.01)):
            g = lostC[(lostC.p_D >= lo) & (lostC.p_D < hi)]
            print(f"    p_D [{lo:.2f},{hi:.2f}) n={len(g):4d}  p_C med "
                  f"{g.p_C.median() if len(g) else float('nan'):.3f}  score_j med "
                  f"{g.score_j.median() if len(g) else float('nan'):.3f}  in D's ILP "
                  f"{g.in_D_ilp.mean() if len(g) else float('nan'):.2f}", flush=True)
        print("  lost-by-D true edges by target detection score:", flush=True)
        for lo, hi in ((0, 0.24), (0.24, 0.30), (0.30, 0.36), (0.36, 0.5), (0.5, 1.01)):
            g = lostC[(lostC.score_j >= lo) & (lostC.score_j < hi)]
            a = keptT[(keptT.score_j >= lo) & (keptT.score_j < hi)]
            print(f"    score_j [{lo:.2f},{hi:.2f}) lost {len(g):4d}  kept {len(a):6d}  "
                  f"lost share {len(g) / max(1, len(g) + len(a)):.4f}", flush=True)
except Exception as exc:                                    # noqa: BLE001
    print(f"[degraded] sketch failed ({type(exc).__name__}: {exc}); the CSV stands",
          flush=True)

for arm in ("C", "D"):
    _csv = f"/kaggle/working/rkedge_{arm}.csv"
    with open(_csv, "w", encoding="utf-8", newline="\\n") as _fh:
        keys = sorted(next(iter(PER_ARM[arm].values())).keys())
        _fh.write("dataset,fold," + ",".join(keys) + "\\n")
        for n in SIX:
            if n in PER_ARM[arm]:
                r = PER_ARM[arm][n]
                _fh.write(f"{n},{FOLD_OF[n]}," + ",".join(str(r[k_]) for k_ in keys) + "\\n")
    print(f"  wrote {_csv}", flush=True)
warning_summary()
print(f"\\nRKEDGE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("RKEDGE OK" if not FAILS and not INCOMPLETE else "RKEDGE NOT OK", flush=True)
'''
i = src.index("# ---- report ----")
src = src[:i] + REPORT

out = os.path.join(os.path.dirname(os.path.abspath(SRC)), "rkedge.py")
open(out, "w", encoding="utf-8", newline="\n").write(src)
print("written rkedge.py", len(src))
