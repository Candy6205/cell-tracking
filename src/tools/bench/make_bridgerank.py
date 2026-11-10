"""hparabench.py -> bridgerank.py: LEARNED selection of 2- and 3-frame bridges.

WHAT IS OPEN (2026-09-11, divmiss + rank5 + span4). Of the 5,696 annotated
cells the detector misses on the 128 6bba movies, 65% sit inside a track
that is detected on both sides: 1,004 one-frame gaps, 351 two-frame, 160
three-frame, 240 longer. A straight line between the flanking detections
lands within the scorer's 7 um of the missed cell in 99% of gaps up to
three frames. The shipped `_gap_close` bridges one-frame gaps only
(GAP_SPAN 1, midpoint or repair-band candidate, cap 2% of nodes). span4
tried two-frame bridges with the same distance rule and lost div_J
(bridges reach across a division and take a daughter as somebody's
continuation). rank5 trained a bridge ranker (AUC 0.97) and used it to
REJECT one-frame bridges: monotone loss, because a wrong one-frame bridge
costs two edges and rescues a whole fragment from SHORT_MIN. Nobody has
used a learned model to SELECT longer bridges.

WHAT THIS IS. hparabench.py (round 2 = cell C: BASE fold rankers, hpara
points, floor 0.48, DIV_W 0.55, the shipped composition) with the dw35
re-solve arm removed and FOUR post-link arms added, all replayed on the
SAME solved graph per movie (the solve is pickled in pass 1):

  REF     shipped: span 1 only                      (round 2's REF, anchor)
  H2      span 2 by the old distance rule            (span4's control)
  L2_50   span 1 shipped, then span 2 chosen by the model at p >= 0.50
  L2_80   ... at p >= 0.80
  L23_80  span 2 and 3, p >= 0.80

THE MODEL. Inside `_gap_close`, after the shipped span-1 pass, every
(track end at t, track start at t+step) pair within GAP1_GATE_UM * span
gets 26 features from the GRAPH (not the detector): straight distance,
per-frame distance, z / xy parts, in- and out-velocity, their cosine, the
extrapolation errors, fragment lengths, competition (how many starts near
this end, how many ends near this start, distance rank, second-nearest
distance), clearance (nearest primary node at each interpolated frame),
repair-band candidates at the interpolated frames, fork context (single-
child nodes within DIV_PARENT_UM of the start at the frame before it, so
"this start could be somebody's second daughter" is a feature), whether
the end is itself a daughter, detection scores of both nodes, local
density, t/T, span. Label from GT at MATCH_UM 5.0: the start's GT node is
a descendant of the end's GT node at exactly `step` generations (a
division in between is allowed: the inserted node then matches the
mother and the bridge edges are true edges). Pairs with an unmatched
endpoint are unlabelled and never trained on. Two fold models by movie,
HistGB(300, seed 0), the ranker's estimator; the model applied to a movie
was fitted on the other fold. In the learned arms the assignment is a
Hungarian on 1 - p over pairs with p >= tau; the span-1 pass, the snap to
repair-band candidates, the cap and everything downstream are the shipped
code.

ANCHORS. REF must reproduce round 2 exactly: 55/237/70, div_J 0.1519,
adj_J 0.9146, score 0.9298; REF + gate6 45/53/80 0.2528 / 0.9155 / 0.9408;
REF + joint 49/51/76 0.2784 / 0.9155 / 0.9433. ANCHOR_D187 0.9204 and
+BIDIR 0.9252 as in round 2. A miss anywhere means the harness moved and
the arms are unreadable.

READ. The cross (each arm + gate6 / joint, joint = what ships) and the
paired bootstrap against REF + joint; adj_J separately from div_J (a bridge
that eats a daughter shows up in div_J, a bridge that rescues a fragment
in adj_J). Per span: proposals, accepted, the out-of-fold AUC, precision
at each tau. Per-movie CSVs for every arm.

Attach: rkedge's 16 sources verbatim. CPU, ~6 h (round 2 was 179 min with
one re-solve arm; here pass 1 is one solve per movie and pass 2 is four
patch-and-judge replays per movie).
"""
import os

src = open("hparabench.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


assert 'DET_SRC = "F4HP"' in src, "hparabench.py must be the F4HP (round 2) harness"
assert "RANK_TRAIN" not in src, "hparabench.py has BASE fold rankers; RANK_TRAIN must not exist here"
rep('SCRIPT_VERSION = "2026-09-09a-hparabench"', 'SCRIPT_VERSION = "2026-09-11b-bridgerank"')

# ---- 1. knobs and state for the learned bridges --------------------------
rep('''DIVW_ARMS = [("dw35", 0.35)]   # divbench6's winner: +0.0015, both folds up
''',
    '''# bridgerank: no re-solve arms. The four arms below are post-link replays on
# the SAME solved graph; DIVW_ARMS carries their names so the report and the
# cross iterate over them unchanged (the dw value is the shipped one).
BR_ARMS = [("H2", dict(span=2, model=False, tau=0.0)),
           ("L2_50", dict(span=2, model=True, tau=0.50)),
           ("L2_80", dict(span=2, model=True, tau=0.80)),
           ("L23_80", dict(span=3, model=True, tau=0.80))]
DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS]
BR_COLLECT = False        # pass 1: record (end, start) pairs + labels after span 1
BR_MODEL = None           # pass 2: the fold model for the movie being patched
BR_TAU = 0.0
BR_ROWS = []              # (movie, fold, span, label, feats)
BR_CTX = {}               # per movie: seeds, seed_of_node, gt
BR_STAT = defaultdict(int)
BR_NAMES = ["span", "d_um", "d_per_frame", "dz_um", "dxy_um", "v_in", "v_out",
            "cos_in_out", "cos_in_delta", "err_in", "err_out", "err_mean",
            "len_end", "len_start", "n_starts_near_end", "n_ends_near_start",
            "rank_by_d", "d_second", "clear_min", "n_cand_hits", "cand_score_max",
            "n_mother_cand", "d_mother_min", "end_is_daughter", "s_end", "s_start",
            "dens_end", "t_frac"]
BR_DIR = pathlib.Path("/tmp/br")
BR_DIR.mkdir(parents=True, exist_ok=True)
''')

# ---- 2. the feature / label / cost functions, defined before _gap_close ----
rep('''def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.
''',
    '''def _br_node_score(g):
    so = BR_CTX.get("seed_of_node")
    if so is None:
        return -1.0
    s = so.get(g)
    if s is None:
        return -1.0          # an inserted node has no detection score
    t_, i_ = s
    return float(BR_CTX["seeds"][t_][1][i_])


def _br_frame_trees(t, by_t, nodes, outdeg):
    """(primary tree, single-child-node tree) at frame t, cached per call."""
    key = ("br", t)
    if key in _PRIM_TREES:
        return _PRIM_TREES[key]
    ids = by_t.get(t, [])
    tr = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float) * SCALE) if ids else None
    m_ids = [g for g in ids if outdeg.get(g, 0) == 1]
    mt = cKDTree(np.asarray([nodes[g]["xyz"] for g in m_ids], float) * SCALE) if m_ids else None
    _PRIM_TREES[key] = (tr, mt)
    return _PRIM_TREES[key]


def _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred, by_t, outdeg,
              cand, cand_sc, cand_trees, T):
    """Feature rows for every (end, start) pair with D[r, c] <= GAP1_GATE_UM
    (D is already divided by span). Returns (pairs [(r, c)], rows)."""
    span = step - 1
    pairs, rows = [], []
    n_starts_near = (D <= GAP1_GATE_UM).sum(1)
    n_ends_near = (D <= GAP1_GATE_UM).sum(0)
    order = np.argsort(D, axis=1)
    for r_, ge in enumerate(ends):
        row_d = D[r_]
        ok = np.where(row_d <= GAP1_GATE_UM)[0]
        if len(ok) == 0:
            continue
        a = P[r_]
        pa = pred.get(ge)
        v_in = a - np.asarray(nodes[pa]["xyz"], float) * SCALE if pa is not None else np.zeros(3)
        nv_in = float(np.linalg.norm(v_in))
        len_end = _frag_len(ge, pred, back=True)
        end_is_daughter = 1.0 if (pa is not None and outdeg.get(pa, 0) >= 2) else 0.0
        tr_t, _ = _br_frame_trees(t, by_t, nodes, outdeg)
        dens = float(len(tr_t.query_ball_point(a, 7.0)) - 1) if tr_t is not None else 0.0
        s_end = _br_node_score(ge)
        srt = order[r_]
        d_second = float(row_d[srt[1]] * span) if len(srt) > 1 else 99.0
        for c_ in ok:
            gs = starts[c_]
            b = C[c_]
            delta = b - a
            d = float(np.linalg.norm(delta))
            sb = succ.get(gs)
            v_out = np.asarray(nodes[sb]["xyz"], float) * SCALE - b if sb is not None else np.zeros(3)
            nv_out = float(np.linalg.norm(v_out))
            cos_io = float(v_in @ v_out) / (nv_in * nv_out) if nv_in > 1e-6 and nv_out > 1e-6 else 0.0
            cos_id = float(v_in @ delta) / (nv_in * d) if nv_in > 1e-6 and d > 1e-6 else 0.0
            err_in = float(np.linalg.norm(delta - v_in * step))
            err_out = float(np.linalg.norm(delta - v_out * step))
            err_mean = float(np.linalg.norm(delta - 0.5 * (v_in + v_out) * step))
            len_start = _frag_len(gs, succ, back=False)
            rank = int(np.where(srt == c_)[0][0])
            clear_min, n_hits, sc_max = 99.0, 0, 0.0
            for k in range(1, step):
                frac = k / float(step)
                m_ = (1.0 - frac) * a + frac * b
                tt = t + k
                trk, _ = _br_frame_trees(tt, by_t, nodes, outdeg)
                if trk is not None:
                    clear_min = min(clear_min, float(trk.query(m_)[0]))
                tree = cand_trees[tt]
                if tree is not None:
                    dist, idx = tree.query(m_)
                    if dist <= GAP1_SNAP_UM and cand_sc[tt][idx] >= GAP1_MIN_CAND_SCORE:
                        n_hits += 1
                        sc_max = max(sc_max, float(cand_sc[tt][idx]))
            _, mt = _br_frame_trees(t + step - 1, by_t, nodes, outdeg)
            n_moth, d_moth = 0, 99.0
            if mt is not None:
                near = mt.query_ball_point(b, DIV_PARENT_UM)
                n_moth = len(near)
                if n_moth:
                    d_moth = float(mt.query(b)[0])
            rows.append([span, d, d / span, abs(float(delta[0])), float(np.linalg.norm(delta[1:])),
                         nv_in, nv_out, cos_io, cos_id, err_in, err_out, err_mean,
                         len_end, len_start, float(n_starts_near[r_]), float(n_ends_near[c_]),
                         rank, d_second, clear_min, n_hits, sc_max, n_moth, d_moth,
                         end_is_daughter, s_end, _br_node_score(gs), dens, t / max(1, T)])
            pairs.append((r_, c_))
    return pairs, np.asarray(rows, np.float32).reshape(-1, len(BR_NAMES))


def _br_label(t, step, ge, gs, nodes):
    """1 if the start's GT node is a descendant of the end's GT node at
    exactly `step` generations, 0 if both match GT and it is not, -1 if
    either endpoint has no GT node within MATCH_UM."""
    gt = BR_CTX.get("gt")
    if gt is None:
        return -1
    trees, kids = gt
    if t not in trees or (t + step) not in trees:
        return -1
    rows_a, tr_a = trees[t]
    rows_b, tr_b = trees[t + step]
    da, ia = tr_a.query(np.asarray(nodes[ge]["xyz"], float) * SCALE)
    db, ib = tr_b.query(np.asarray(nodes[gs]["xyz"], float) * SCALE)
    if da > MATCH_UM or db > MATCH_UM:
        return -1
    gen = [int(rows_a[int(ia)])]
    for _ in range(step):
        gen = [c for p_ in gen for c in kids.get(p_, [])]
        if not gen:
            return 0
    return 1 if int(rows_b[int(ib)]) in gen else 0


def _br_outdeg(edges):
    od = defaultdict(int)
    for a_, _b in edges:
        od[a_] += 1
    return od


def _br_collect(T, nodes, succ, pred, by_t, edges, cand, cand_sc, cand_trees):
    """Pass 1: after the shipped span-1 pass, record every span-2 and span-3
    pair with its features and GT label."""
    outdeg = _br_outdeg(edges)
    for span in (2, 3):
        step = span + 1
        for t in range(T - step):
            ends = [g for g in by_t.get(t, []) if g not in succ and g in pred]
            starts = [g for g in by_t.get(t + step, []) if g not in pred and g in succ]
            if not ends or not starts:
                continue
            P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
            C = np.asarray([nodes[g]["xyz"] for g in starts], float) * SCALE
            D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2)) / float(span)
            pairs, rows = _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred,
                                    by_t, outdeg, cand, cand_sc, cand_trees, T)
            for (r_, c_), row in zip(pairs, rows):
                lab = _br_label(t, step, ends[r_], starts[c_], nodes)
                BR_STAT[f"collect_span{span}_pairs"] += 1
                if lab >= 0:
                    BR_STAT[f"collect_span{span}_labelled"] += 1
                    BR_STAT[f"collect_span{span}_pos"] += lab
                BR_ROWS.append((BR_CTX["name"], BR_CTX["fold"], span, lab, row))


def _br_cost(t, step, ends, starts, P, C, D, cost, nodes, succ, pred, by_t, edges,
             cand, cand_sc, cand_trees, T, BIG):
    """Pass 2, learned arms: replace the distance cost for span >= 2 by
    1 - p over pairs with p >= BR_TAU; everything else is BIG."""
    span = step - 1
    outdeg = _br_outdeg(edges)
    pairs, rows = _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred, by_t,
                            outdeg, cand, cand_sc, cand_trees, T)
    new = np.full(cost.shape, BIG)
    if not pairs:
        return new
    p = BR_MODEL.predict_proba(rows)[:, 1]
    BR_STAT[f"span{span}_proposals"] += len(pairs)
    for (r_, c_), p_ in zip(pairs, p):
        if p_ >= BR_TAU:
            new[r_, c_] = 1.0 - float(p_)
            BR_STAT[f"span{span}_over_tau"] += 1
    return new


def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.
''')

# ---- 3. inside _gap_close: learned cost for span >= 2, collect hook ------
rep('''            D = D / float(span)
            cost = np.where(D > GAP1_GATE_UM, BIG, D)
            ri, ci = linear_sum_assignment(cost)
''',
    '''            D = D / float(span)
            cost = np.where(D > GAP1_GATE_UM, BIG, D)
            if span >= 2 and BR_MODEL is not None:
                cost = _br_cost(t, step, ends, starts, P, C, D, cost, nodes, succ,
                                pred, by_t, edges, cand, cand_sc, cand_trees, T, BIG)
            ri, ci = linear_sum_assignment(cost)
''')
rep('''                n_added += 1
    _r = _GAP_LOG[(GAP_NEED_CAND, GAP_SPAN, round(GAP1_GATE_UM, 2),''',
    '''                n_added += 1
                BR_STAT[f"span{span}_bridged"] += 1
    if BR_COLLECT:
        _br_collect(T, nodes, succ, pred, by_t, edges, cand, cand_sc, cand_trees)
    _r = _GAP_LOG[(GAP_NEED_CAND, GAP_SPAN, round(GAP1_GATE_UM, 2),''')

# ---- 4. pass 1: GT context before the REF patch, collect during it, pickle
rep('''    outp = pathlib.Path(f"{TMP}/ref") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    PATCH_CAPS.clear()
    patch_final(nodes0, edges0, pool, outp)
''',
    '''    outp = pathlib.Path(f"{TMP}/ref") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    PATCH_CAPS.clear()
    # bridgerank pass 1: GT for the bridge labels, then collect inside the
    # shipped patch (GAP_SPAN is 1 here, so nothing longer is bridged and the
    # REF graph is round 2's).
    _gt = read_geff(geff_path(root, "train", name))
    _g_t = np.asarray(_gt["t"], int)
    _g_pos = np.stack([np.asarray(_gt[c] , float) for c in ("z", "y", "x")], axis=1) * SCALE
    _row_of = {int(v): i for i, v in enumerate(_gt["node_ids"])}
    _kids = defaultdict(list)
    for _a, _b in _gt["edges"]:
        _ra, _rb = _row_of.get(int(_a)), _row_of.get(int(_b))
        if _ra is not None and _rb is not None:
            _kids[_ra].append(_rb)
    _trees = {}
    for _tt in np.unique(_g_t):
        _rows = np.where(_g_t == _tt)[0]
        _trees[int(_tt)] = (_rows, cKDTree(_g_pos[_rows]))
    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node,
                  gt=(_trees, _kids))
    BR_STAT.clear()
    assert GAP_SPAN == 1 and BR_MODEL is None
    BR_COLLECT = True
    patch_final(nodes0, edges0, pool, outp)
    BR_COLLECT = False
    _brs = dict(BR_STAT)
    BR_CTX["gt"] = None
    del _gt, _g_t, _g_pos, _row_of, _kids, _trees
''')

# the arm block that RE-SOLVES at dw35 -> pickle the solve for pass 2
rep('''    cells = []
    for arm, dw in DIVW_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, divw=dw)
        pm_a, so_a = aux_a
        sc_a, b2_a = _aux_extra(seeds, pm_a)
        PATCH_CAPS.clear()
        patch_final(n_a, e_a, p_a, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, (pm_a, so_a, sc_a, b2_a))
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        n_ilp = sum(1 for r_ in fr_a if r_[5][0] == 0)
        cells.append(f"{arm} forks {len(fr_a)}({n_ilp} ilp) "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/"
                     f"{row_a['division_fn']} {_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a
''',
    '''    cells = [f"br pairs s2 {_brs.get('collect_span2_pairs', 0)} "
             f"(lab {_brs.get('collect_span2_labelled', 0)}, pos {_brs.get('collect_span2_pos', 0)}) "
             f"s3 {_brs.get('collect_span3_pairs', 0)} "
             f"(lab {_brs.get('collect_span3_labelled', 0)}, pos {_brs.get('collect_span3_pos', 0)})"]
    # the arms are replayed in pass 2 on this exact solve
    with open(BR_DIR / f"{name}.pkl", "wb") as _fh:
        pickle.dump((nodes0, edges0, pool[0], pool[1], seeds, pmap, seed_of_node), _fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
''')

# ---- 5. pass 2: train the bridge model, replay the four arms -------------
rep('''print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)
''',
    '''print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)

# ---- BRIDGERANK PASS 2a: the bridge model, two folds by movie --------------
print(f"\\n=== BRIDGE MODEL: {len(BR_ROWS):,} (end, start) pairs at span 2/3 collected "
      f"after the shipped span-1 pass on {len(PER)} movies ===", flush=True)
_mv = np.array([r_[0] for r_ in BR_ROWS])
_fo = np.array([r_[1] for r_ in BR_ROWS], int)
_sp = np.array([r_[2] for r_ in BR_ROWS], int)
_lb = np.array([r_[3] for r_ in BR_ROWS], int)
_X = np.stack([r_[4] for r_ in BR_ROWS]).astype(np.float32) if BR_ROWS else np.zeros((0, len(BR_NAMES)), np.float32)
_ok = _lb >= 0
for s_ in (2, 3):
    m_ = _sp == s_
    print(f"  span {s_}: pairs {int(m_.sum()):,}, labelled {int((m_ & _ok).sum()):,}, "
          f"positive {int((_lb[m_ & _ok] == 1).sum()):,}", flush=True)
# ---- Zebrahub rows (biohub-zhbridge): dense-GT labels, same 28 features ----
# The 6bba labels are too few to train on (88 positives in 128 movies). The
# zebrahub rows enter BOTH folds' training sets; the 6bba rows of the other
# fold are added on top, and every number read on 6bba is still out of fold.
_zh_csv = sorted(pathlib.Path("/kaggle/input").glob("*/**/bridge_pairs_zh.csv"))
_Xz = np.zeros((0, len(BR_NAMES)), np.float32)
_yz = np.zeros(0, int)
_cz = np.zeros(0, object)
if _zh_csv:
    import csv as _csvmod
    with open(_zh_csv[0], encoding="utf-8") as _fh:
        _rd = _csvmod.reader(_fh)
        _hdr = next(_rd)
        assert _hdr[3:] == BR_NAMES, f"bridge_pairs_zh.csv columns {_hdr[3:6]}... differ from BR_NAMES"
        _rows_z = [(r_[0], int(r_[1]), int(r_[2]), [float(v) for v in r_[3:]]) for r_ in _rd]
    _Xz = np.asarray([r_[3] for r_ in _rows_z], np.float32).reshape(-1, len(BR_NAMES))
    _yz = np.asarray([r_[2] for r_ in _rows_z], int)
    _cz = np.asarray([r_[0] for r_ in _rows_z], object)
    _sz = np.asarray([r_[1] for r_ in _rows_z], int)
    print(f"  zebrahub rows from {_zh_csv[0].name}: {len(_yz):,} labelled over "
          f"{len(set(_cz.tolist()))} crops; span 2 positive {int(((_sz == 2) & (_yz == 1)).sum()):,} "
          f"of {int((_sz == 2).sum()):,}, span 3 positive {int(((_sz == 3) & (_yz == 1)).sum()):,} "
          f"of {int((_sz == 3).sum()):,}", flush=True)
    # zebrahub-internal check: 2 folds by crop, AUC on the held-out crops
    _crops = sorted(set(_cz.tolist()))
    _cf = {c_: i_ % 2 for i_, c_ in enumerate(_crops)}
    _zf = np.asarray([_cf[c_] for c_ in _cz], int)
    for kk in (0, 1):
        _m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        _m.fit(_Xz[_zf != kk], _yz[_zf != kk])
        _pz = _m.predict_proba(_Xz[_zf == kk])[:, 1]
        for s_ in (2, 3):
            mm = (_sz[_zf == kk] == s_)
            yy = _yz[_zf == kk][mm]
            if yy.sum() and (yy == 0).any():
                print(f"    zh fold{kk} span {s_}: n {int(mm.sum()):,} pos {int(yy.sum()):,} "
                      f"AUC {_auc(yy, _pz[mm]):.4f}  P@0.5 {float(yy[_pz[mm] >= 0.5].mean()) if (_pz[mm] >= 0.5).any() else float('nan'):.3f} "
                      f"P@0.8 {float(yy[_pz[mm] >= 0.8].mean()) if (_pz[mm] >= 0.8).any() else float('nan'):.3f} "
                      f"R@0.8 {float(yy[_pz[mm] >= 0.8].sum() / max(1, yy.sum())):.3f}", flush=True)
else:
    print("  no bridge_pairs_zh.csv attached: training on 6bba rows only", flush=True)
BR_FOLD_MODEL = {}
_p_oof = np.full(len(_lb), np.nan)
for kk in (0, 1):
    tr = _ok & (_fo != kk)
    te = _fo == kk
    Xtr_ = np.vstack([_X[tr], _Xz]) if len(_yz) else _X[tr]
    ytr_ = np.concatenate([_lb[tr], _yz]) if len(_yz) else _lb[tr]
    npos = int((ytr_ == 1).sum())
    if len(ytr_) < 200 or npos < 30:
        warn_once("bridge_fold_small", f"bridge fold{kk}: {len(ytr_)} labelled rows, {npos} positive")
        continue
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xtr_, ytr_)
    BR_FOLD_MODEL[kk] = m
    if te.any():
        _p_oof[te] = m.predict_proba(_X[te])[:, 1]
    print(f"  fold{kk} model: trained on {len(ytr_):,} labelled rows ({npos} positive; "
          f"{int(tr.sum())} of them 6bba from the other fold, {len(_yz):,} zebrahub); "
          f"applied to fold{kk}", flush=True)
BR_MODEL_OK = len(BR_FOLD_MODEL) == 2
if BR_MODEL_OK:
    print(f"  {'span':>4} {'n lab':>7} {'pos':>5} {'AUC model':>10} {'AUC -d':>7} "
          + "".join(f"{'P@' + str(t_):>8}{'R@' + str(t_):>8}" for t_ in (0.5, 0.8, 0.9))
          + "   p quantiles 50/90/99% over ALL pairs", flush=True)
    for s_ in (2, 3):
        m_ = (_sp == s_) & _ok & np.isfinite(_p_oof)
        y_ = (_lb[m_] == 1).astype(int)
        p_ = _p_oof[m_]
        line = f"  {s_:>4} {int(m_.sum()):>7,} {int(y_.sum()):>5} "
        line += f"{_auc(y_, p_):>10.4f} {_auc(y_, -_X[m_, 1]):>7.4f} " if y_.sum() and (y_ == 0).any() else f"{'nan':>10} {'nan':>7} "
        for t_ in (0.5, 0.8, 0.9):
            sel = p_ >= t_
            prec = float(y_[sel].mean()) if sel.any() else float("nan")
            recl = float(y_[sel].sum() / max(1, y_.sum()))
            line += f"{prec:>8.3f}{recl:>8.3f}"
        ma = (_sp == s_) & np.isfinite(_p_oof)
        q = np.quantile(_p_oof[ma], [0.5, 0.9, 0.99]) if ma.any() else [np.nan] * 3
        line += "   " + " ".join(f"{v:.3f}" for v in q)
        print(line, flush=True)
    # single-column AUCs on the labelled span-2 rows
    m_ = (_sp == 2) & _ok
    y_ = (_lb[m_] == 1).astype(int)
    print("  single-column AUC, span 2 (below 0.5 = informative with the sign flipped):", flush=True)
    print("   " + "  ".join(f"{n_}:{_auc(y_, _X[m_, j_]):.3f}" for j_, n_ in enumerate(BR_NAMES) if j_ > 0), flush=True)
    m_full = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m_full.fit(np.vstack([_X[_ok], _Xz]) if len(_yz) else _X[_ok],
               np.concatenate([_lb[_ok], _yz]) if len(_yz) else _lb[_ok])
    with open("/kaggle/working/bridge_rank.pkl", "wb") as _fh:
        pickle.dump({"model": m_full, "feature_names": BR_NAMES, "version": SCRIPT_VERSION,
                     "trained_on": "labelled span-2/3 (end, start) pairs after the shipped "
                                   "span-1 pass, 128 6bba train movies, hpara points",
                     "gate_um_per_frame": GAP1_GATE_UM, "match_um": MATCH_UM}, _fh)
    print("  bridge_rank.pkl written (fitted on both folds)", flush=True)
else:
    print("  bridge model NOT trained: too few labelled rows in a fold; the learned arms "
          "are skipped and only H2 runs", flush=True)
with open("/kaggle/working/bridge_pairs.csv", "w", encoding="utf-8", newline="\\n") as _fh:
    _fh.write("movie,fold,label,p_oof," + ",".join(BR_NAMES) + "\\n")
    for i_, r_ in enumerate(BR_ROWS):
        if r_[3] >= 0:
            _fh.write(f"{r_[0]},{r_[1]},{r_[3]},{_p_oof[i_]:.4f}," + ",".join(f"{v:.4g}" for v in r_[4]) + "\\n")
print("  bridge_pairs.csv written (labelled rows only, with out-of-fold p)", flush=True)
del _X, BR_ROWS[:]
gc.collect()

# ---- BRIDGERANK PASS 2b: the four arms, replayed on the pickled solves -----
BR_ARMS_RUN = [(a_, c_) for a_, c_ in BR_ARMS if (not c_["model"]) or BR_MODEL_OK]
DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS_RUN]
BR_STAT_ARM = {a_: defaultdict(int) for a_, _c in BR_ARMS_RUN}
print(f"\\n=== PASS 2: {len(BR_ARMS_RUN)} post-link arms replayed on the pickled solve of "
      f"each movie: " + ", ".join(f"{a_} (span {c_['span']}, "
                                  f"{'model p>=' + str(c_['tau']) if c_['model'] else 'distance rule'})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)
tb2 = time.time()
INCOMPLETE2 = []
for name in [n for n in SIX if n in PER]:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE2.append(name)
        continue
    t0 = time.time()
    with open(BR_DIR / f"{name}.pkl", "rb") as _fh:
        nodes0, edges0, cand_, cand_sc_, seeds, pmap, seed_of_node = pickle.load(_fh)
    cand_trees_ = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in cand_]
    pool = (cand_, cand_sc_, cand_trees_)
    k = FOLD_OF[name]
    scsort, best2 = _aux_extra(seeds, pmap)
    aux = (pmap, seed_of_node, scsort, best2)
    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node, gt=None)
    cells = []
    for arm, cfg in BR_ARMS_RUN:
        GAP_SPAN = cfg["span"]
        BR_MODEL = BR_FOLD_MODEL[k] if cfg["model"] else None
        BR_TAU = cfg["tau"]
        BR_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        for k_, v_ in BR_STAT.items():
            BR_STAT_ARM[arm][k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        cells.append(f"{arm} b2 {BR_STAT.get('span2_bridged', 0)} b3 {BR_STAT.get('span3_bridged', 0)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a
    GAP_SPAN = 1
    BR_MODEL = None
    BR_TAU = 0.0
    print(f"  {name} fold{k} REF {_adj_of(PER[name]):.4f} | " + " | ".join(cells)
          + f"  {time.time() - t0:.0f}s", flush=True)
    del nodes0, edges0, pool, cand_, cand_sc_, cand_trees_, seeds, pmap, seed_of_node, aux
    gc.collect()
print(f"  pass 2: {len(SIX) - len(INCOMPLETE) - len(INCOMPLETE2)} movies "
      f"({(time.time() - tb2) / 60:.1f} min)", flush=True)
if INCOMPLETE2:
    print(f"INCOMPLETE2: {len(INCOMPLETE2)} movies not replayed: {INCOMPLETE2}", flush=True)
    # the cross indexes every arm by SIX_OK; drop the movies no arm has
    for _n in INCOMPLETE2:
        PER.pop(_n, None)
    ROWS = [PER[n] for n in SIX if n in PER]
print("\\n=== BRIDGES PER ARM, summed over the movies replayed ===", flush=True)
_bk = ["span1_bridged", "span2_proposals", "span2_over_tau", "span2_bridged",
       "span3_proposals", "span3_over_tau", "span3_bridged"]
print(f"  {'arm':<8}" + "".join(f"{k_:>17}" for k_ in _bk), flush=True)
for a_, _c in BR_ARMS_RUN:
    print(f"  {a_:<8}" + "".join(f"{BR_STAT_ARM[a_].get(k_, 0):>17,}" for k_ in _bk), flush=True)
''')

# ---- 6. the report: wants for the round-2 anchors, per-arm CSVs --------------
rep('''WANT = {"REF + gate6": "want 42/47/83 0.2442 0.9143 0.9387 (divbench3)",
        "REF + joint": "want 46/39/79 0.2805 0.9143 0.9424 (divbench8)",
        "dw35 + gate6": "want 50/68/75 0.2591 0.9143 0.9402 (divbench6)",
        "dw35 + joint": "<-- THE NEW NUMBER"}''',
    '''WANT = {"REF + gate6": "want 45/53/80 0.2528 0.9155 0.9408 (hparabench round 2)",
        "REF + joint": "want 49/51/76 0.2784 0.9155 0.9433 (round 2, what ships)"}''')
rep('''print("\\n=== reading it ===", flush=True)''',
    '''print(f"\\n  PAIRED BOOTSTRAP of every arm + joint AGAINST REF + joint (what ships), "
      f"score and adj_J separately ===", flush=True)
_BSA = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["adj_edge_jaccard"] for d in _draws])
        for k in XRES}
for a_, _c in BR_ARMS_RUN:
    lab = f"{a_} + joint"
    if lab not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[lab] - bank["REF + joint"]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {lab:<16} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"bridgerank_arm_{a_}.csv"
    _keys_a = sorted({k for r in ARM_PER[a_].values() for k in r})
    with open(_csv_a, "w", encoding="utf-8", newline="\\n") as _fh:
        _fh.write("dataset,fold," + ",".join(_keys_a) + "\\n")
        for _n in sorted(ARM_PER[a_]):
            _r = ARM_PER[a_][_n]
            _fh.write(f"{_n},{FOLD_OF.get(_n, -1)}," + ",".join(repr(_r.get(_k)) for _k in _keys_a) + "\\n")
    print(f"  {_csv_a.name} written (ungated per-movie rows of arm {a_})", flush=True)
print("\\n=== reading it ===", flush=True)''')
rep('''_csv = pathlib.Path("/kaggle/working") / f"hparabench_{DET_SRC}.csv"''',
    '''_csv = pathlib.Path("/kaggle/working") / f"bridgerank_REF_{DET_SRC}.csv"''')

# ---- 7. the anchor that licenses everything: REF == round 2 -----------------
rep('''assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
if len(ROWS) == len(SIX) and not INCOMPLETE:
    print("HPARABENCH OK", flush=True)
else:
    print(f"[degraded] HPARABENCH INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)''',
    '''assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
_r2 = (TP, FP, FN) == (55, 237, 70) and abs(RES["adj_edge_jaccard"] - 0.9146) <= ANCHOR_TOL
print(f"  ROUND-2 ANCHOR: REF {TP}/{FP}/{FN} adj_J {RES['adj_edge_jaccard']:.4f} want 55/237/70 "
      f"0.9146 -> {'OK' if _r2 else 'MISS'}", flush=True)
_r2c = XRES.get("REF + joint", {}).get("rc", {})
print(f"  ROUND-2 ANCHOR: REF + joint score {_r2c.get('score', float('nan')):.4f} want 0.9433 -> "
      f"{'OK' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MISS'}", flush=True)
if len(ROWS) == len(SIX) and not INCOMPLETE and not INCOMPLETE2 and _r2:
    print("BRIDGERANK OK", flush=True)
else:
    print(f"[degraded] BRIDGERANK INCOMPLETE or anchor miss: {len(ROWS)}/{len(SIX)}, "
          f"round-2 anchor {'OK' if _r2 else 'MISS'}", flush=True)''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bridgerank.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written bridgerank.py", len(src))
