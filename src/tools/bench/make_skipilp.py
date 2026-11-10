"""dupsup.py -> skipilp.py: span-2 candidate edges INSIDE the ILP.

WHAT IT MEASURES. The one bridging variant never run. Four post-link
versions (gap2, gap3, gap2strict, bridgerank) bridged track ENDS after the
solve and came out zero or negative; bridgerank's residue was that 423 of
the 511 annotated 2/3-frame interior gaps are not (end, start) pairs at all:
the track ran on through a wrong link at the missing frame, and only the
solver can trade that link for a jump. Here every (i at t, j at t+2) pair
within SKIP_GATE_UM is a candidate edge of the SAME ILP the shipped linker
solves, with a weight from its own out-of-fold selector, so a skip competes
with the consecutive edges for i's out-slot and j's in-slot under the
shipped costs (floor 0.48, disappearance 1.5, DIV_W 0.55). A chosen skip
becomes two edges through an inserted node at t+1: the sub-threshold
candidate within SKIP_SNAP_UM of the midpoint when one exists (the missed
peak is NMS-suppressed, not absent), else the midpoint.

STAGES. Features + join and stage A are dupsup's (REF is round 2's graph,
hard anchors unchanged). Stage A2 (new): per 6bba movie, the skip proposals,
23 features, GT labels (both endpoints matched at 5 um to a GT node and its
grandchild; the middle GT node's matched/missed flag), two fold HistGB
models out of fold, `skip_pairs.csv` + `skip_rank.pkl`. Stage B: REF plus
four fresh solves per movie, SK50 / SK80 (skip weight = p, offered at
p >= 0.50 / 0.80) and SK50H / SK30H (weight = 0.5 p, offered at 0.50 /
0.30), each through the shipped post-link patches, the official scorer,
the fork table and the joint gate cross, like dupsup's arms.

Attach: dupsup's 16 sources (hmbench's list minus hmerge / ephmcache /
ep2hmcache). CPU. dupsup with 3 arms took ~3 h; here 4 arms + stage A2.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-14a-skipilp"')

# ---- the arms ---------------------------------------------------------------------
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'SKIP_ARMS = [("SK50", dict(tau=0.50, w=1.0)), ("SK80", dict(tau=0.80, w=1.0)),\n'
    '             ("SK50H", dict(tau=0.50, w=0.5)), ("SK30H", dict(tau=0.30, w=0.5))]\n'
    'DUP_ARMS = SKIP_ARMS      # skipilp: four skip-edge solves, no suppression')

# ---- stash pass-1 velocities (features_ext discards them) -------------------------
rep('    seeds, vels, prev_of = pass1_tracks(dets)\n    feats = pair_features(dets, None, T)',
    '    seeds, vels, prev_of = pass1_tracks(dets)\n    _VELS_STASH[0] = vels\n'
    '    feats = pair_features(dets, None, T)')
rep('FEATS = {}                 # name -> (seeds, feats), 6bba only',
    'FEATS = {}                 # name -> (seeds, feats), 6bba only\nVELS4 = {}                 # name -> pass-1 velocities on the F4 primaries')
rep('        FEATS[name] = (seeds4, feats4)\n',
    '        FEATS[name] = (seeds4, feats4)\n        VELS4[name] = _VELS_STASH[0]\n')

# ---- the skip block: constants, features, labels, insertion -------------------------
SKIP_BLOCK = r'''
# ---- skipilp block ----
SKIP_GATE_UM = 14.0            # total displacement over the two frames (7 um per frame)
SKIP_SNAP_UM = GAP1_SNAP_UM    # 3.2: the inserted node snaps to a sub-threshold candidate this near
SKIP_MIN_CAND = GAP1_MIN_CAND_SCORE   # 0.10
SKIP_NAMES = ["d_um", "d_per_frame", "dz_um", "dxy_um", "s_i", "s_j", "v_in", "err_cv",
              "cos_v_disp", "path_best", "path_n", "p_out_best", "p_in_best",
              "mid_prim_d", "mid_prim_s", "mid_cand_d", "mid_cand_s", "n_out_i", "n_in_j",
              "rank_by_d", "d_second", "dens_i", "t_frac"]
_VELS_STASH = [None]
SKIP_PROPS = {}      # name -> dict(t, i, j, X, lab, mid, p)
SKIP_MODELS = {}
SKIP_CHOSEN = []     # (t, i, j) of the skip edges the last solve kept, filled by _skip_insert


def _skip_gt(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    by_t = defaultdict(list)
    for i in range(len(g_t)):
        by_t[int(g_t[i])].append(i)
    trees = {t: (np.asarray(rows, int), cKDTree(g_pos[rows])) for t, rows in by_t.items()}
    return trees, kids


def _skip_match(trees, t, pos_um):
    """label_rows' rule: the nearest GT node within MATCH_UM per detection, -1 if none."""
    out = np.full(len(pos_um), -1, np.int64)
    if t not in trees or len(pos_um) == 0:
        return out
    rows, tree = trees[t]
    dd, gi = tree.query(pos_um)
    ok = dd <= MATCH_UM
    out[ok] = rows[gi[ok]]
    return out


def skip_movie(name, dets, T, seeds, feats, vels, model, cols):
    """Every (i at t, j at t+2) pair within SKIP_GATE_UM: features, GT label, middle flag.

    lab  1 = both endpoints matched GT nodes two GT edges apart; 0 = i matched a GT
         node with grandchildren and j is not one of them; -1 = unlabelled.
    mid  for lab 1: 1 if no primary at t+1 matched the GT node between them (a real
         hole), 0 if one did (the skip is redundant with a path); -1 otherwise.
    """
    probs = [model.predict_proba(rows[:, cols])[:, 1] if len(pairs) else np.zeros(0)
             for pairs, rows, P, C in feats]
    pdict, out_list, out_best, in_best = [], [], [], []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        d_ = {}
        ol = defaultdict(list)
        ob = defaultdict(float)
        ib = defaultdict(float)
        for (i_, j_), pr in zip(pairs, probs[t]):
            i_, j_, pr = int(i_), int(j_), float(pr)
            d_[(i_, j_)] = pr
            ol[i_].append((j_, pr))
            if pr > ob[i_]:
                ob[i_] = pr
            if pr > ib[j_]:
                ib[j_] = pr
        pdict.append(d_)
        out_list.append(ol)
        out_best.append(ob)
        in_best.append(ib)
    prim_um = [np.asarray(seeds[t][0], float).reshape(-1, 3) * SCALE for t in range(T)]
    prim_tree = [cKDTree(prim_um[t]) if len(prim_um[t]) else None for t in range(T)]
    cand_sc, cand_tree = [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float).reshape(-1, 3)
        s_all = np.asarray(dets[t][1], float).reshape(-1)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand_sc.append(s_all[cm])
        cand_tree.append(cKDTree(c_all[cm] * SCALE) if cm.any() else None)
    trees, kids = _skip_gt(name)
    gmatch = [_skip_match(trees, t, prim_um[t]) for t in range(T)]
    matched_gt = {int(g) for t in range(T) for g in gmatch[t] if g >= 0}
    T_, I_, J_, X_, L_, M_ = [], [], [], [], [], []
    for t in range(T - 2):
        P, sP = seeds[t]
        C, sC = seeds[t + 2]
        if len(P) == 0 or len(C) == 0 or prim_tree[t + 2] is None:
            continue
        Pu, Cu = prim_um[t], prim_um[t + 2]
        nb = prim_tree[t + 2].query_ball_point(Pu, SKIP_GATE_UM)
        n_in = defaultdict(int)
        for js in nb:
            for j_ in js:
                n_in[int(j_)] += 1
        dens = np.asarray([len(x) for x in prim_tree[t].query_ball_point(Pu, GATE_UM)], float) - 1.0
        for i_, js in enumerate(nb):
            if not js:
                continue
            js = np.asarray(sorted(js), int)
            dv = Cu[js] - Pu[i_][None]
            d = np.sqrt((dv ** 2).sum(1))
            order = np.argsort(d)
            rank = np.empty(len(js), int)
            rank[order] = np.arange(len(js))
            d2 = float(d[order[1]]) if len(js) > 1 else 99.0
            v = np.asarray(vels[t][i_], float) if len(vels[t]) else np.zeros(3)
            vn = float(np.linalg.norm(v))
            gi = int(gmatch[t][i_])
            gchild_of = {}
            if gi >= 0:
                for c_ in kids.get(gi, []):
                    for gc_ in kids.get(c_, []):
                        gchild_of[gc_] = c_
            for kk, j_ in enumerate(js):
                j_ = int(j_)
                disp = dv[kk]
                dz = abs(float(disp[0]))
                dxy = float(np.linalg.norm(disp[1:]))
                if vn > 1e-6:
                    err_cv = float(np.linalg.norm(Pu[i_] + 2.0 * v - Cu[j_]))
                    cosv = float(np.dot(v, disp) / (vn * max(float(d[kk]), 1e-6)))
                else:
                    err_cv, cosv = np.nan, np.nan
                pb, pn = 0.0, 0
                for v_, p1 in out_list[t].get(i_, ()):
                    p2 = pdict[t + 1].get((v_, j_))
                    if p2 is not None:
                        pb = max(pb, p1 * p2)
                        if p1 >= PROD_FLOOR and p2 >= PROD_FLOOR:
                            pn += 1
                mid = 0.5 * (Pu[i_] + Cu[j_])
                if prim_tree[t + 1] is not None:
                    dd, ii = prim_tree[t + 1].query(mid)
                    mpd, mps = float(dd), float(seeds[t + 1][1][ii])
                else:
                    mpd, mps = 99.0, 0.0
                if cand_tree[t + 1] is not None:
                    dd, ii = cand_tree[t + 1].query(mid)
                    mcd, mcs = float(dd), float(cand_sc[t + 1][ii])
                else:
                    mcd, mcs = 99.0, 0.0
                X_.append([float(d[kk]), float(d[kk]) / 2.0, dz, dxy, float(sP[i_]), float(sC[j_]),
                           vn, err_cv, cosv, pb, float(pn), out_best[t].get(i_, 0.0),
                           in_best[t + 1].get(j_, 0.0), mpd, mps, mcd, mcs, float(len(js)),
                           float(n_in[j_]), float(rank[kk]), d2, float(dens[i_]),
                           t / max(T - 1, 1)])
                T_.append(t)
                I_.append(i_)
                J_.append(j_)
                if gchild_of:
                    gj = int(gmatch[t + 2][j_])
                    if gj in gchild_of:
                        L_.append(1)
                        M_.append(0 if gchild_of[gj] in matched_gt else 1)
                    else:
                        L_.append(0)
                        M_.append(-1)
                else:
                    L_.append(-1)
                    M_.append(-1)
    X = np.asarray(X_, np.float32).reshape(-1, len(SKIP_NAMES))
    return dict(t=np.asarray(T_, np.int32), i=np.asarray(I_, np.int32), j=np.asarray(J_, np.int32),
                X=X, lab=np.asarray(L_, np.int8), mid=np.asarray(M_, np.int8),
                p=np.full(len(T_), np.nan, np.float32))


def _skip_insert(nodes0, edges0, cand, cand_sc, cand_trees, seed_of_node):
    """Every solved t -> t+2 edge becomes two edges through a node inserted at t+1:
    the sub-threshold candidate within SKIP_SNAP_UM of the midpoint (score >=
    SKIP_MIN_CAND) when one exists, else the midpoint. A skip edge leaving a node
    that also has another out-edge is dropped: a division with children in
    different frames is nothing the scorer knows. Node ids stay contiguous."""
    outdeg = defaultdict(int)
    for a_, _b in edges0:
        outdeg[a_] += 1
    new_edges = []
    nid = (max(nodes0) + 1) if nodes0 else 1
    SKIP_CHOSEN.clear()
    for a_, b_ in edges0:
        ta, tb = nodes0[a_]["t"], nodes0[b_]["t"]
        if tb != ta + 2:
            new_edges.append((a_, b_))
            continue
        DUP_STAT["skip_chosen"] += 1
        sa, sb = seed_of_node.get(a_), seed_of_node.get(b_)
        if sa is not None and sb is not None:
            SKIP_CHOSEN.append((int(sa[0]), int(sa[1]), int(sb[1])))
        if outdeg[a_] > 1:
            DUP_STAT["skip_dropped_fork"] += 1
            continue
        tm = ta + 1
        mid = 0.5 * (np.asarray(nodes0[a_]["xyz"], float) + np.asarray(nodes0[b_]["xyz"], float))
        xyz = mid
        tree = cand_trees[tm] if tm < len(cand_trees) else None
        if tree is not None:
            dd, ii = tree.query(mid * SCALE)
            if dd <= SKIP_SNAP_UM and cand_sc[tm][ii] >= SKIP_MIN_CAND:
                xyz = np.asarray(cand[tm][ii], float)
                DUP_STAT["skip_snapped"] += 1
        nodes0[nid] = {"t": int(tm), "xyz": np.asarray(xyz, float)}
        new_edges.append((a_, nid))
        new_edges.append((nid, b_))
        DUP_STAT["skip_inserted"] += 1
        nid += 1
    return new_edges
# ---- end skipilp block ----
'''
rep('DUP_STAT = defaultdict(int)\n', 'DUP_STAT = defaultdict(int)\n' + SKIP_BLOCK)

# ---- solve_one: skip edges in, inserted nodes out ---------------------------------
rep('def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,\n'
    '              divw=None, keep=None):',
    'def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,\n'
    '              divw=None, keep=None, skip=None):')
rep('    graph = PIPE.build_graph(coords, cand_edges)\n    if graph.num_edges() > 0:\n        solver = td.solvers.ILPSolver(',
    '    if skip:\n'
    '        for (t_, i_, j_, w_) in skip:\n'
    '            if keep is not None and not (keep[t_][i_] and keep[t_ + 2][j_]):\n'
    '                continue\n'
    '            cand_edges.append((offset[t_] + i_, offset[t_ + 2] + j_, float(w_), 0.0))\n'
    '        DUP_STAT["skip_offered"] += len(skip)\n'
    '    graph = PIPE.build_graph(coords, cand_edges)\n    if graph.num_edges() > 0:\n        solver = td.solvers.ILPSolver(')
rep('    if keep is not None:\n        # a suppressed primary never had an edge; drop its node so no',
    '    if skip:\n        edges0 = _skip_insert(nodes0, edges0, cand, cand_sc, cand_trees, seed_of_node)\n'
    '    if keep is not None:\n        # a suppressed primary never had an edge; drop its node so no')

# ---- stage A2: proposals, labels, fold models ---------------------------------------
STAGE_A2 = r'''
# ---- STAGE A2: span-2 skip proposals, labels, out-of-fold selector ------------
print(f"\n=== STAGE A2: span-2 skip proposals on {len(SIX)} 6bba movies (gate "
      f"{SKIP_GATE_UM} um total), 23 features, GT labels, two fold HistGB models ===", flush=True)
ta2 = time.time()
SK_TOT = defaultdict(int)
for k_, name in enumerate(SIX):
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS[name]
    sp = skip_movie(name, dets, T, seeds, feats, VELS4[name], MODELS_FOLD[FOLD_OF[name]], COLS)
    SKIP_PROPS[name] = sp
    lab, mid = sp["lab"], sp["mid"]
    n_lab, n_pos = int((lab >= 0).sum()), int((lab == 1).sum())
    n_hole = int(((lab == 1) & (mid == 1)).sum())
    SK_TOT["proposals"] += len(lab)
    SK_TOT["labelled"] += n_lab
    SK_TOT["pos"] += n_pos
    SK_TOT["pos_hole"] += n_hole
    print(f"[A2 {k_ + 1}/{len(SIX)}] {name} fold{FOLD_OF[name]}: primaries "
          f"{sum(len(seeds[t][0]) for t in range(T))} proposals {len(lab):,} labelled {n_lab:,} "
          f"pos {n_pos:,} (real holes {n_hole}) {time.time() - t0:.0f}s", flush=True)
    del dets, VELS4[name]
    gc.collect()
print(f"  proposals {SK_TOT['proposals']:,}; labelled {SK_TOT['labelled']:,}; positive "
      f"{SK_TOT['pos']:,} of which real holes (middle GT node unmatched) {SK_TOT['pos_hole']:,}; "
      f"{(time.time() - ta2) / 60:.1f} min", flush=True)


def _q(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    return " / ".join(f"{float(np.quantile(v, q)):.2f}" for q in qs)


_Xall = np.concatenate([SKIP_PROPS[n]["X"] for n in SIX])
_Lall = np.concatenate([SKIP_PROPS[n]["lab"] for n in SIX])
_Mall = np.concatenate([SKIP_PROPS[n]["mid"] for n in SIX])
print("  d_um quantiles 10/25/50/75/90:", flush=True)
print(f"    positives           {_q(_Xall[_Lall == 1, 0])}", flush=True)
print(f"    positives, holes    {_q(_Xall[(_Lall == 1) & (_Mall == 1), 0])}", flush=True)
print(f"    negatives           {_q(_Xall[_Lall == 0, 0])}", flush=True)
print(f"    unlabelled          {_q(_Xall[_Lall < 0, 0])}", flush=True)
print(f"  path_best (best two-step path p1*p2) quantiles: positives {_q(_Xall[_Lall == 1, 9])}; "
      f"holes {_q(_Xall[(_Lall == 1) & (_Mall == 1), 9])}; negatives {_q(_Xall[_Lall == 0, 9])}", flush=True)
print(f"  mid_cand_d (nearest sub-threshold candidate to the midpoint, um): holes "
      f"{_q(_Xall[(_Lall == 1) & (_Mall == 1), 15])}; negatives {_q(_Xall[_Lall == 0, 15])}", flush=True)
del _Xall, _Lall, _Mall
gc.collect()

for k in (0, 1):
    t0 = time.time()
    tr = [n for n in SIX if FOLD_OF[n] != k]
    he = [n for n in SIX if FOLD_OF[n] == k]
    Xa = np.concatenate([SKIP_PROPS[n]["X"][SKIP_PROPS[n]["lab"] >= 0] for n in tr])
    ya = np.concatenate([SKIP_PROPS[n]["lab"][SKIP_PROPS[n]["lab"] >= 0] for n in tr]).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa, ya)
    SKIP_MODELS[k] = m
    for n in he:
        sp = SKIP_PROPS[n]
        sp["p"] = m.predict_proba(sp["X"])[:, 1].astype(np.float32) if len(sp["X"]) else sp["p"]
    Xh = np.concatenate([SKIP_PROPS[n]["X"][SKIP_PROPS[n]["lab"] >= 0] for n in he])
    yh = np.concatenate([SKIP_PROPS[n]["lab"][SKIP_PROPS[n]["lab"] >= 0] for n in he]).astype(int)
    ph = m.predict_proba(Xh)[:, 1] if len(Xh) else np.zeros(0)
    print(f"  skip fold{k}: trained on {len(tr)} movies / {len(ya):,} rows ({ya.mean():.4f} pos); "
          f"held-out {len(he)} movies / {len(yh):,} rows AUC {_auc(yh, ph):.4f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    del Xa, ya, Xh, yh, ph
    gc.collect()

_P = np.concatenate([SKIP_PROPS[n]["p"] for n in SIX])
_L = np.concatenate([SKIP_PROPS[n]["lab"] for n in SIX])
_M = np.concatenate([SKIP_PROPS[n]["mid"] for n in SIX])
_D = np.concatenate([SKIP_PROPS[n]["X"][:, 0] for n in SIX])
_lab = _L >= 0
print(f"  POOLED out of fold: {int(_lab.sum()):,} labelled rows, AUC model {_auc(_L[_lab], _P[_lab]):.4f}, "
      f"AUC distance only {_auc(_L[_lab], -_D[_lab]):.4f}", flush=True)
print(f"  {'tau':>5} {'offered':>10} {'per movie med/max':>20} {'labelled':>9} {'pos':>7} {'P':>6} "
      f"{'R':>6} {'R holes':>8} {'unlabelled share':>17}", flush=True)
for tau in (0.20, 0.30, 0.50, 0.80):
    sel = _P >= tau
    per = [int((SKIP_PROPS[n]["p"] >= tau).sum()) for n in SIX]
    n_l = int((sel & _lab).sum())
    n_p = int((sel & (_L == 1)).sum())
    prec = n_p / max(1, n_l)
    rec = n_p / max(1, int((_L == 1).sum()))
    rec_h = int((sel & (_L == 1) & (_M == 1)).sum()) / max(1, int(((_L == 1) & (_M == 1)).sum()))
    print(f"  {tau:>5.2f} {int(sel.sum()):>10,} {int(np.median(per)):>9,}/{max(per):<10,} {n_l:>9,} "
          f"{n_p:>7,} {prec:>6.3f} {rec:>6.3f} {rec_h:>8.3f} {(sel & ~_lab).sum() / max(1, sel.sum()):>17.3f}",
          flush=True)
print("  (P/R on labelled rows only; 'R holes' = recall among positives whose middle GT node no "
      "primary matched, the pairs that can earn edges; the unlabelled share is what the ILP "
      "also gets offered and the scorer's node term sees)", flush=True)
del _P, _L, _M, _D, _lab

with open("/kaggle/working/skip_pairs.csv", "w", encoding="utf-8", newline="\n") as _fh:
    _fh.write("dataset,fold,t,i,j,lab,mid,p," + ",".join(SKIP_NAMES) + "\n")
    for n in SIX:
        sp = SKIP_PROPS[n]
        for q in np.where(sp["lab"] >= 0)[0]:
            _fh.write(f"{n},{FOLD_OF[n]},{int(sp['t'][q])},{int(sp['i'][q])},{int(sp['j'][q])},"
                      f"{int(sp['lab'][q])},{int(sp['mid'][q])},{float(sp['p'][q]):.5f},"
                      + ",".join(f"{float(v):.5g}" for v in sp["X"][q]) + "\n")
with open("/kaggle/working/skip_rank.pkl", "wb") as _fh:
    pickle.dump({"models": SKIP_MODELS, "feature_names": SKIP_NAMES, "gate_um": SKIP_GATE_UM,
                 "version": SCRIPT_VERSION}, _fh)
for n in SIX:            # the features are on disk; stage B needs only (t, i, j, p, lab, mid)
    SKIP_PROPS[n]["X"] = None
gc.collect()
print(f"  skip_pairs.csv + skip_rank.pkl written; stage A2 {(time.time() - ta2) / 60:.1f} min", flush=True)

'''
rep('# ---- STAGE B: the full pipeline, the only verdict that counts ----------',
    STAGE_A2 + '# ---- STAGE B: the full pipeline, the only verdict that counts ----------')

# ---- stage B arm loop: skip solves instead of suppression -----------------------------
rep('    cells = []\n    n_prim = sum(len(seeds[t][0]) for t in range(T))\n'
    '    for arm, cfg in BR_ARMS:\n'
    '        keep = _dup_keep(seeds, cfg["R"])\n'
    '        n_keep = sum(int(k_.sum()) for k_ in keep)\n'
    '        DUP_STAT.clear()\n',
    '    cells = []\n'
    '    sp = SKIP_PROPS.pop(name)\n'
    '    lab_of = {(int(t_), int(i_), int(j_)): (int(l_), int(m_))\n'
    '              for t_, i_, j_, l_, m_ in zip(sp["t"], sp["i"], sp["j"], sp["lab"], sp["mid"])}\n'
    '    for arm, cfg in BR_ARMS:\n'
    '        sel = np.where(sp["p"] >= cfg["tau"])[0]\n'
    '        skip_a = [(int(sp["t"][q]), int(sp["i"][q]), int(sp["j"][q]), cfg["w"] * float(sp["p"][q]))\n'
    '                  for q in sel]\n'
    '        DUP_STAT.clear()\n')
rep('        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n'
    '                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, keep=keep)\n',
    '        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n'
    '                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, skip=skip_a)\n'
    '        for key_ in SKIP_CHOSEN:\n'
    '            lm_ = lab_of.get(key_)\n'
    '            if lm_ is None or lm_[0] < 0:\n'
    '                DUP_STAT["skip_chosen_unlab"] += 1\n'
    '            elif lm_[0] == 1:\n'
    '                DUP_STAT["skip_chosen_pos"] += 1\n'
    '                if lm_[1] == 1:\n'
    '                    DUP_STAT["skip_chosen_pos_hole"] += 1\n'
    '            else:\n'
    '                DUP_STAT["skip_chosen_neg"] += 1\n')
rep('        DUP_BY_ARM[arm]["primaries"] += n_prim\n'
    '        DUP_BY_ARM[arm]["suppressed"] += n_prim - n_keep\n', '')
rep('        cells.append(f"{arm} sup {n_prim - n_keep} nodes {len(n_a)} "',
    '        cells.append(f"{arm} skip {DUP_STAT[\'skip_inserted\']}/{len(skip_a)} nodes {len(n_a)} "')
rep('        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep\n',
    '        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, skip_a\n')

# ---- the per-arm bookkeeping table ---------------------------------------------------
rep('print("\\n=== SUPPRESSION PER ARM, summed over the 128 movies ===", flush=True)\n'
    '_dk = ["primaries", "suppressed", "edges_withheld", "nodes_dropped"]',
    'print("\\n=== SKIP EDGES PER ARM, summed over the 128 movies (offered = candidate edges given to '
    'the ILP; chosen = solved t->t+2 edges; inserted = chosen minus those dropped for sharing a '
    'source with another edge; snapped = inserted node sits on a sub-threshold candidate; '
    'pos/neg/unlab = GT label of the chosen pair, hole = positive whose middle GT node no primary '
    'matched) ===", flush=True)\n'
    '_dk = ["skip_offered", "skip_chosen", "skip_dropped_fork", "skip_inserted", "skip_snapped",\n'
    '       "skip_chosen_pos", "skip_chosen_pos_hole", "skip_chosen_neg", "skip_chosen_unlab"]')
rep('''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _dk)
          + f"   ({d_.get('suppressed', 0) / max(1, d_.get('primaries', 1)):.3%} of primaries)", flush=True)''',
    '''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _dk)
          + f"   (chosen {d_.get('skip_chosen', 0) / max(1, d_.get('skip_offered', 1)):.3%} of offered)", flush=True)''')
rep('print(f"  {\'arm\':<8}" + "".join(f"{k_:>17}" for k_ in _dk), flush=True)',
    'print(f"  {\'arm\':<8}" + "".join(f"{k_[5:]:>17}" for k_ in _dk), flush=True)')

# ---- names ------------------------------------------------------------------------
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"skipilp_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"skipilp_arm_{a_}.csv"')
rep('print("DUPSUP OK", flush=True)', 'print("SKIPILP OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] SKIPILP INCOMPLETE or anchor miss:')

assert src.count("skip=skip_a") == 1 and src.count("_skip_insert(") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skipilp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written skipilp.py", len(src))
