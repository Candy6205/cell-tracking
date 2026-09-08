"""divbench.py -> divbench2.py: the same bench (128 6bba movies, out-of-fold
+BIDIR, BASE primaries, production ILP), REF patch only, instrumented:
  (1) every fork the final graph carries, with its features at fork time
      and the official scorer's verdict on it (TP / FP / uncounted);
  (2) every GT division not recovered, walked through the patch's
      conditions in order on the pre-patch graph (divgate2's walk, on 125);
  (3) a 2-fold fork gate trained on the counted forks of the other fold,
      and the pruning curve it yields (div tp / fp / J at each cut).
"""
src = open("divbench.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker, start=0):
    i = src.index(marker, start)
    return src.rfind(NL, 0, i) + 1


rep('SCRIPT_VERSION = "2026-09-05a-divbench"', 'SCRIPT_VERSION = "2026-09-06a-divbench2"')

# ---- solve_one: also hand back the pair probabilities and the node -> seed map
rep('''    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)''',
    '''    pmap = {}
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr, row in zip(pairs, probs[t], rows):
            pmap[(t, int(i_), int(j_))] = (float(pr), float(row[C_PFWD]),
                                            float(row[C_PREV]))
    cd = {}
    for g_, (t_, z_, y_, x_) in enumerate(coords.tolist()):
        cd.setdefault((int(t_), int(z_), int(y_), int(x_)), (int(t_), g_ - offset[int(t_)]))
    seed_of_node = {}
    for i_ in range(len(nodes_arr)):
        key = tuple(int(round(v)) for v in nodes_arr[i_])
        seed_of_node[i_ + 1] = cd.get(key)
    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), (pmap, seed_of_node)''')

# ---- _add_safe_divisions: log what it adds -----------------------------
rep('''            if _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s \\
                    < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q))''',
    '''            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))''')
rep('''    for score, t, p, q in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue
        edges.append((p, q))''',
    '''    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])))
        edges.append((p, q))''')

# ---- patch_final: snapshot the pre-patch graph and the id order ----------
rep('''    _linefit(nodes, edges)
    if divs:
        _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)''',
    '''    _linefit(nodes, edges)
    PRE_STATE["nodes"] = nodes
    PRE_STATE["edges"] = list(edges)
    FORK_LOG.clear()
    if divs:
        _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)
    PRE_STATE["ids"] = ids
    PRE_STATE["final_edges"] = list(edges)''')

# ---- STAGE B + report -------------------------------------------------------
start = line_start('DIV_ARMS = ["REF", "NB", "DIVG1", "DIVG0", "NODIV"]')
stage_b = r'''from tracking_cellmot.division_metrics import score_divisions   # noqa: E402

FORK_LOG = {}
PRE_STATE = {}
DOMAIN = "BASE-6bba-all"
ARM_TAG = {"REF": "ref"}
FEAT_NAMES = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap"]
WALK_ORDER = ["mother_absent", "daughter_absent", "recovered", "mother_deleted_post",
              "1a_mother_no_child", "1b_mother_forked_elsewhere",
              "0_sister_deleted_post", "2_mother_no_single_pred",
              "3_child_wrong_frame", "4_mother_child_over_10um",
              "5a_sister_not_orphan", "5b_no_orphans_in_frame",
              "5c_sister_over_12um_of_mother", "6_sister_gap_over_15um",
              "7_not_the_nearest_orphan", "8a_daughter_not_one_child",
              "8b_diverge_under_2.25um", "9_passed_all_gates_then_capped_or_taken"]


def _gt_struct(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1)
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    mothers = {ids[r]: (r, [k for k in ks]) for r, ks in kids.items() if len(ks) >= 2}
    return g_t, g_pos, ids, mothers


def _chain_len(start, step_map, cap=12):
    n, cur = 0, start
    while n < cap:
        nxt = step_map.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def fork_rows(name, k, seeds, aux, tp_forks, fp_forks):
    """Every fork in the final graph, its features, its verdict."""
    pmap, seed_of_node = aux
    nodes, edges, ids = PRE_STATE["nodes"], PRE_STATE["final_edges"], PRE_STATE["ids"]
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    T = max(v["t"] for v in nodes.values()) + 1
    trees = {}
    for t in range(T):
        c, _s = seeds[t]
        trees[t] = cKDTree(np.asarray(c, float) * SCALE) if len(c) else None
    pos_of = {ids[i]: i for i in range(len(ids))}
    tp_n = {ids[i] for i in tp_forks if i < len(ids)}
    fp_n = {ids[i] for i in fp_forks if i < len(ids)}
    rows = []
    for p, ch in succ.items():
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        pq = [(q, FORK_LOG.get((p, q))) for q in ch]
        patched = [q for q, lg in pq if lg is not None]
        if patched:
            q = patched[0]
            src = 1
            c1 = FORK_LOG[(p, q)]["c1"]
            if c1 not in ch:
                c1 = [x for x in ch if x != q][0]
        else:
            src = 0
            # the lower-ranked child is "q"
            sp = seed_of_node.get(p)
            pr = {}
            for x in ch:
                sx = seed_of_node.get(x)
                pr[x] = pmap.get((sp[0], sp[1], sx[1]), (np.nan,))[0] if sp and sx and sx[0] == sp[0] + 1 else np.nan
            ch_sorted = sorted(ch, key=lambda x: (np.nan_to_num(pr[x], nan=-1.0)))
            q, c1 = ch_sorted[0], ch_sorted[-1]

        def _p(a, b):
            sa, sb = seed_of_node.get(a), seed_of_node.get(b)
            if sa is None or sb is None or sb[0] != sa[0] + 1:
                return (np.nan, np.nan, np.nan)
            return pmap.get((sa[0], sa[1], sb[1]), (np.nan, np.nan, np.nan))

        def _sc(a):
            sa = seed_of_node.get(a)
            return float(seeds[sa[0]][1][sa[1]]) if sa is not None else np.nan
        p_c1, pf_c1, _ = _p(p, c1)
        p_q, pf_q, pr_q = _p(p, q)
        d_pc1 = _dist_um(nodes[p]["xyz"], nodes[c1]["xyz"])
        d_pq = _dist_um(nodes[p]["xyz"], nodes[q]["xyz"])
        d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
        c1n, qn = succ.get(c1, []), succ.get(q, [])
        dv = (_dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
              if len(c1n) == 1 and len(qn) == 1 else np.nan)
        tr = trees.get(t)
        dens10 = (len(tr.query_ball_point(np.asarray(nodes[p]["xyz"], float) * SCALE, r=10.0)) - 1
                  if tr is not None else np.nan)
        lg = FORK_LOG.get((p, q), {})
        rank_gap = (p_c1 - p_q) if (p_c1 == p_c1 and p_q == p_q) else np.nan
        row = dict(src=src, d_pc1=d_pc1, d_pq=d_pq, d_s=d_s, dv=dv, p_c1=p_c1, p_q=p_q,
                   pf_c1=pf_c1, pf_q=pf_q, pr_q=pr_q, sc_p=_sc(p), sc_c1=_sc(c1), sc_q=_sc(q),
                   len_back=_chain_len(p, pred), len_c1=_chain_len(c1, succ),
                   len_q=_chain_len(q, succ), dens10=dens10,
                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap)
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES]))
    return rows


def walk_lost(name, scores):
    """divgate2's walk on the pre-patch graph for every GT division the
    scorer did not pair. Returns {condition: count} and a list of
    (mother_gt_id, condition)."""
    g_t, g_pos, gids, mothers = _gt_struct(name)
    nodes, edges = PRE_STATE["nodes"], PRE_STATE["edges"]
    succ, pred = defaultdict(list), defaultdict(list)
    pre_set = set()
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
        pre_set.add((a, b))
    by_t = defaultdict(list)
    for g, v in nodes.items():
        by_t[int(v["t"])].append(g)
    trees = {}
    for t, ids_ in by_t.items():
        trees[t] = (ids_, cKDTree(np.asarray([nodes[g]["xyz"] for g in ids_], float) * SCALE))
    orph_ids, orph_tree = {}, {}
    for t, ids_ in by_t.items():
        o = [g for g in ids_ if not pred.get(g)]
        orph_ids[t] = o
        if o:
            orph_tree[t] = cKDTree(np.asarray([nodes[g]["xyz"] for g in o], float) * SCALE)

    def match(r):
        t = int(g_t[r])
        if t not in trees:
            return None
        ids_, tr = trees[t]
        d, i = tr.query(g_pos[r] * SCALE)
        return ids_[int(i)] if d <= MATCH_UM else None

    out = defaultdict(int)
    det = []
    for gid, (rm, kid_rows) in mothers.items():
        if scores.get(gid, 0) == 1:
            out["recovered"] += 1
            det.append((gid, "recovered"))
            continue
        ga = match(rm)
        if ga is None:
            out["mother_absent"] += 1
            det.append((gid, "mother_absent"))
            continue
        d1, d2 = match(kid_rows[0]), match(kid_rows[1])
        if d1 is None or d2 is None:
            out["daughter_absent"] += 1
            det.append((gid, "daughter_absent"))
            continue
        kept = [g for g in (d1, d2) if (ga, g) in pre_set]
        ch = succ.get(ga, [])
        cond = None
        if len(ch) == 0:
            cond = "1a_mother_no_child"
        elif len(ch) >= 2:
            cond = "1b_mother_forked_elsewhere"
        if cond is None:
            c1 = ch[0]
            t_ = int(nodes[ga]["t"])
            q = d2 if (kept and kept[0] == d1) else d1
            if len(pred.get(ga, [])) != 1:
                cond = "2_mother_no_single_pred"
            elif int(nodes[c1]["t"]) != t_ + 1:
                cond = "3_child_wrong_frame"
            elif _dist_um(nodes[ga]["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
                cond = "4_mother_child_over_10um"
            elif pred.get(q):
                cond = "5a_sister_not_orphan"
            elif orph_tree.get(t_ + 1) is None:
                cond = "5b_no_orphans_in_frame"
            elif _dist_um(nodes[ga]["xyz"], nodes[q]["xyz"]) > DIV_PARENT_UM:
                cond = "5c_sister_over_12um_of_mother"
            else:
                d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
                if d_s > DIV_SISTER_UM:
                    cond = "6_sister_gap_over_15um"
                else:
                    d_q, q_idx = orph_tree[t_ + 1].query(np.asarray(nodes[c1]["xyz"], float) * SCALE)
                    if orph_ids[t_ + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                        cond = "7_not_the_nearest_orphan"
                    else:
                        c1n, qn = succ.get(c1, []), succ.get(q, [])
                        if len(c1n) != 1 or len(qn) != 1:
                            cond = "8a_daughter_not_one_child"
                        else:
                            dv = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
                            cond = ("8b_diverge_under_2.25um" if dv < DIV_DIVERGE_UM
                                    else "9_passed_all_gates_then_capped_or_taken")
        out[cond] += 1
        det.append((gid, cond))
    return out, det


print(f"\n=== STAGE B: {len(SIX)} 6bba movies, one production solve each "
      f"(out-of-fold +BIDIR, disappearance {ILP_DISAPP}, floor {PROD_FLOOR}), "
      f"REF patch, every fork logged and judged, every lost division walked ===",
      flush=True)
print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J 0.9293 "
      "(BASE-6bba hard). Cells: division tp/fp/fn, forks (patch/ilp), "
      "REF adj_J.", flush=True)

ROWS = []
PER = {}
ANCHOR_ROWS = []
INCOMPLETE = []
FORKS = []             # (name, fold, p, q, label, feats)
WALK = defaultdict(int)
WALK_FOLD = {0: defaultdict(int), 1: defaultdict(int)}
tb = time.time()
for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "BASE")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    nodes0, edges0, pool, aux = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                          disapp=ILP_DISAPP, floor=PROD_FLOOR)
    outp = pdir("REF", DOMAIN) / f"{name}.geff"
    patch_final(nodes0, edges0, pool, outp)
    row = score_one(outp, name)
    ROWS.append(row)
    PER[name] = row
    # the scorer's verdict per fork and per GT division
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(outp))
    g = res_[0] if isinstance(res_, tuple) else res_
    ds = score_divisions(g, dset.tracks, scale=dset.scale)
    n_tp, n_fp = len(ds.tp_forks), len(ds.fp_forks)
    if (n_tp, n_fp) != (row["division_tp"], row["division_fp"]):
        warn_once("fork_count_mismatch",
                  f"{name}: score_divisions tp/fp {n_tp}/{n_fp} vs per-sample "
                  f"{row['division_tp']}/{row['division_fp']}")
    fr = fork_rows(name, k, seeds, aux, ds.tp_forks, ds.fp_forks)
    FORKS.extend(fr)
    scores = {int(kk): int(vv) for kk, vv in ds.scores.items()}
    w, det = walk_lost(name, scores)
    for c_, n_ in w.items():
        WALK[c_] += n_
        WALK_FOLD[k][c_] += n_
    n_gt_walk = sum(w.values())
    if n_gt_walk != row["division_tp"] + row["division_fn"]:
        warn_once("walk_count_mismatch",
                  f"{name}: walked {n_gt_walk} GT divisions vs scorer's "
                  f"{row['division_tp'] + row['division_fn']}")
    extra = ""
    if name in HOLDOUT_6BBA:
        n_a, e_a, p_a, _ = solve_one(dets, T, seeds, feats, MODEL_D187, MCOLS_D187,
                                     disapp=1.5, floor=0.10)
        outa = pathlib.Path("/kaggle/working/anchor_d187") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_a, e_a, p_a, outa)
        ANCHOR_ROWS.append(score_one(outa, name))
        extra = f"  anchor {_adj_of(ANCHOR_ROWS[-1]):.4f}"
        del n_a, e_a, p_a
    n_patch = sum(1 for r_ in fr if r_[5][0] == 1)
    print(f"  {name} fold{k} gt_div={row['division_tp'] + row['division_fn']:>2} "
          f"div {row['division_tp']}/{row['division_fp']}/{row['division_fn']:<3} "
          f"forks {len(fr):>4} ({n_patch} patch) "
          f"lost: {', '.join(f'{c_} {n_}' for c_, n_ in sorted(w.items()) if c_ != 'recovered') or '-'}"
          f"   {_adj_of(row):.4f}{extra}  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool, aux, g, res_, dset, ds
    gc.collect()
print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)

# ---- report ----------------------------------------------------------------
print("\n=== ANCHOR / official scorer ===", flush=True)
ANCHOR_FAIL = []
if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):
    ra = _with_ratio(ANCHOR_ROWS)
    got = float(ra["adj_edge_jaccard"])
    ok = abs(got - 0.9293) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  BASE-6bba(12)  adj_J {got:.4f}  div_J {ra['division_jaccard']:.4f}"
          f"  want 0.9293 HARD -> {'OK' if ok else 'MISS'} ({got - 0.9293:+.4f})", flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "BASE-6bba", got, 0.9293))
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "BASE-6bba", float("nan"), 0.9293))
print("  hard anchor reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  ⚠ HARD ANCHOR MISS", flush=True)

RES = _with_ratio(ROWS)
TP = sum(r["division_tp"] for r in ROWS)
FP = sum(r["division_fp"] for r in ROWS)
FN = sum(r["division_fn"] for r in ROWS)
N_GT = TP + FN
print(f"\n=== REF on {len(ROWS)} movies: div {TP}/{FP}/{FN} div_J {TP / max(1, TP + FP + FN):.4f} "
      f"adj_J {RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f} "
      f"(divbench v1: 54/258/71, 0.1410, 0.9136, 0.9277) ===", flush=True)

print("\n=== WHERE THE LOST DIVISIONS GO: first condition that stops each (pre-patch "
      "graph, divgate2's walk) ===", flush=True)
print(f"  {'condition':<42} {'all':>5} {'fold0':>6} {'fold1':>6}", flush=True)
for c_ in WALK_ORDER:
    if WALK.get(c_, 0) or WALK_FOLD[0].get(c_, 0) or WALK_FOLD[1].get(c_, 0):
        print(f"  {c_:<42} {WALK.get(c_, 0):>5} {WALK_FOLD[0].get(c_, 0):>6} "
              f"{WALK_FOLD[1].get(c_, 0):>6}", flush=True)
print(f"  {'total GT divisions walked':<42} {sum(WALK.values()):>5}", flush=True)

# ---- the forks ---------------------------------------------------------------
X_f = np.array([r_[5] for r_ in FORKS], float) if FORKS else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in FORKS], int)
fold_f = np.array([r_[1] for r_ in FORKS], int)
src_f = X_f[:, 0].astype(int) if len(X_f) else np.zeros(0, int)
print(f"\n=== EVERY FORK THE FINAL GRAPHS CARRY: {len(FORKS)} forks; verdict x source ===",
      flush=True)
print(f"  {'source':<8} {'TP':>6} {'FP':>6} {'uncounted':>10} {'precision(counted)':>19}",
      flush=True)
for s_, nm in ((1, "patch"), (0, "ilp")):
    m_ = src_f == s_
    tp_, fp_, un_ = int((lab[m_] == 2).sum()), int((lab[m_] == 1).sum()), int((lab[m_] == 0).sum())
    print(f"  {nm:<8} {tp_:>6} {fp_:>6} {un_:>10} {tp_ / max(1, tp_ + fp_):>19.3f}", flush=True)
print(f"  (scorer's pooled TP {TP}, FP {FP})", flush=True)

print("\n=== FORK FEATURES by verdict (median; TP vs FP vs uncounted), and the "
      "single-feature AUC TP-vs-FP on counted forks ===", flush=True)
print(f"  {'feature':<9} {'TP med':>8} {'FP med':>8} {'UNC med':>8} {'AUC':>7} {'n_TP':>5} {'n_FP':>5}",
      flush=True)
counted = lab >= 1
for j_, fn_ in enumerate(FEAT_NAMES):
    if fn_ == "src":
        continue
    col = X_f[:, j_]
    meds = []
    for L_ in (2, 1, 0):
        v = col[(lab == L_) & np.isfinite(col)]
        meds.append(np.median(v) if len(v) else np.nan)
    mm = counted & np.isfinite(col)
    auc = _auc((lab[mm] == 2).astype(int), col[mm]) if mm.sum() and (lab[mm] == 2).any() and (lab[mm] == 1).any() else float("nan")
    print(f"  {fn_:<9} {meds[0]:>8.3f} {meds[1]:>8.3f} {meds[2]:>8.3f} {auc:>7.3f} "
          f"{int((mm & (lab == 2)).sum()):>5} {int((mm & (lab == 1)).sum()):>5}", flush=True)

# ---- the gate: 2-fold, trained on the counted forks of the other fold ------
print("\n=== A FORK GATE, 2-fold (trained on the OTHER fold's counted forks, TP=1 "
      "FP=0; HistGB max_iter 100, lr 0.05, min_samples_leaf 10), applied to "
      "every fork; pruning curve ===", flush=True)
p_gate = np.full(len(FORKS), np.nan)
Xg = X_f[:, 1:]                    # drop src? keep it as a feature: include
Xg = X_f
for k in (0, 1):
    tr_ = (fold_f != k) & counted
    te_ = fold_f == k
    if tr_.sum() < 20 or (lab[tr_] == 2).sum() < 5:
        warn_once(f"gate_fold{k}", f"fold{k}: too few counted forks to train ({tr_.sum()})")
        continue
    mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                        min_samples_leaf=10, random_state=0)
    mg.fit(Xg[tr_], (lab[tr_] == 2).astype(int))
    p_gate[te_] = mg.predict_proba(Xg[te_])[:, 1]
    te_c = te_ & counted
    print(f"  fold{k}: trained on {int(tr_.sum())} counted forks "
          f"({int((lab[tr_] == 2).sum())} TP); out-of-fold AUC on fold{k}'s "
          f"{int(te_c.sum())} counted forks: {_auc((lab[te_c] == 2).astype(int), p_gate[te_c]):.4f}",
          flush=True)
ok_g = np.isfinite(p_gate)
print(f"  {'cut (frac of ALL forks pruned)':<32} {'thr':>6} {'TP':>4} {'FP':>5} {'FN':>4} "
      f"{'div_J':>7} {'d div_J':>8} {'~d score':>9} {'fold0 J':>8} {'fold1 J':>8}", flush=True)
base_J = TP / max(1, TP + FP + FN)
for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    if not ok_g.any():
        break
    thr = np.quantile(p_gate[ok_g], frac) if frac > 0 else -1.0
    keep = ok_g & (p_gate > thr) if frac > 0 else ok_g
    tp_k = int((lab[keep] == 2).sum())
    fp_k = int((lab[keep] == 1).sum())
    fn_k = N_GT - tp_k
    J = tp_k / max(1, tp_k + fp_k + fn_k)
    per = []
    for k in (0, 1):
        mk = keep & (fold_f == k)
        tpk = int((lab[mk] == 2).sum())
        fpk = int((lab[mk] == 1).sum())
        ngt = sum(WALK_FOLD[k].values())
        per.append(tpk / max(1, tpk + fpk + (ngt - tpk)))
    print(f"  {frac:<32.1f} {thr:>6.3f} {tp_k:>4} {fp_k:>5} {fn_k:>4} {J:>7.4f} "
          f"{J - base_J:>+8.4f} {0.1 * (J - base_J):>+9.4f} {per[0]:>8.4f} {per[1]:>8.4f}",
          flush=True)
print(f"  oracle (every FP fork gone, every TP kept): div_J {TP / max(1, TP + FN):.4f} "
      f"({0.1 * (TP / max(1, TP + FN) - base_J):+.4f} of score)", flush=True)
print("  (adj_J is not re-simulated here: a pruned fork removes one edge; div-gate "
      "2026-08-24 saw adj_J move by +0.001 over a 90% cut)", flush=True)

print("\n=== reading it ===", flush=True)
print("  Two levers on 125 GT divisions, out-of-fold. (1) The lost 71: the walk says "
      "which patch condition stops each real division on the graph the patch saw; "
      "5a (the ILP already gave the second daughter a parent) is the steal case, "
      "1a/1b are the ILP's, 8b is the divergence constant. (2) The 258 counted false "
      "forks: the feature medians say whether TP and FP forks differ in anything the "
      "patch can see at fork time; the 2-fold gate says whether a learned cut on the "
      "forks the pipeline ACTUALLY emits prunes FP faster than TP (div-gate 2026-08-24: "
      "on 23 removable FPs it did not). A cut is worth reading only if fold0 J and "
      "fold1 J both rise.", flush=True)
warning_summary()
print(f"\nDIVBENCH2 DONE in {(time.time() - t_all) / 60:.1f} min (stage B "
      f"{(time.time() - tb) / 60:.1f} min), {len(ROWS)} of {len(SIX)} movies scored", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
if len(ROWS) == len(SIX) and not INCOMPLETE:
    print("DIVBENCH2 OK", flush=True)
else:
    print(f"[degraded] DIVBENCH2 INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)
'''
src = src[:start] + stage_b
open("divbench2.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
