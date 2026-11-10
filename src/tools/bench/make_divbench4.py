"""divbench3.py -> divbench4.py: the two open division levers, on the same
128-movie F4 bench, in one CPU pass.

(1) THE FALSE-FORK SIDE has room by construction -- the oracle (every counted
    false fork gone, every true one kept) is div_J 0.4000 against the shipped
    gate's 0.2442. Offline on divbench3's own forks_f4.csv (gatelab, this
    session) the CUT is exhausted: the best threshold on the shipped ranking
    reads 0.2531, +0.0009 of score, and every model / fold / feature-set
    variant is inside a paired bootstrap that straddles zero. So the room is
    the RANKING, and the ranking needs features the table does not have.
    divbench4 adds eight, all computable from the solved graph:
      dv2      divergence two frames out (dv, one frame, is the 2nd best
               feature at AUC 0.787 -- this asks whether the horizon is the
               limit)
      ang      the angle at the mother between the two daughters; a real
               division pushes them apart, a false fork picks up a neighbour
               travelling its own way
      sc_q_r   the orphan daughter's detection score as a PER-MOVIE quantile
      sc_p_r   the mother's, likewise. sc_q is the strongest feature by far
               (AUC 0.842) and it is a raw score: its distribution moves with
               the movie and with the imaging batch, so the rank version is
               what can transfer to 44b6.
      dens5    neighbours within 5 um (dens10 already reads inverse, 0.316)
      p_q_marg the mother's ranker p for the orphan edge MINUS the best other
               parent's -- "is the mother even the best explanation for this
               node", which no current feature asks
      q_ends   the orphan's chain stops before the movie does
      steal    the fork came from the new steal pass (below), not the orphan
               pass
    The 20-feature gate is still trained and still written, so divbench3's
    rows are a hard anchor and the comparison is paired on the same forks.

(2) THE LOST-DIVISION SIDE: 23 of the 75 lost on F4 stop at 5a -- the ILP had
    already given the second daughter a parent, so the orphan pass never sees
    her. This is the largest bucket that is not detection. divbench4 both
    MEASURES it (per case: the ranker's p for the mother's edge against the
    p for the edge the ILP made, both distances, and whether that edge is a
    true GT edge -- divsteal did this on Zebrahub, n=54, and on ONE case in
    each competition domain) and RUNS it: a steal pass in the patch that takes
    the daughter back, gated by that same margin, scored by the official
    scorer at three settings.

    The unconditional arm is in deliberately. It is not a ceiling (2026-08-26:
    it measures the sum of its good and bad cases); it is the control that
    says whether the constraint binds and how big the cancellation is.

Composition: production is patch -> gate. A stolen fork is a fork, so the
gate prunes it out-of-fold like any other; the steal arms are pruned by the
gate trained on REF's forks and the combination is reported. No second solve.

Anchors unchanged: D187 on the 12-movie holdout at floor 0.10 = F4-6bba
adj_J 0.9216 / div_J 0.0606 (HARD); +BIDIR at floor 0.48 on the same 12 =
0.9261 (soft); REF over the 128 = 50 / 249 / 75, div_J 0.1337, adj_J 0.9133,
score 0.9266; the 20-feature 2-fold gate = AUC 0.9283 / 0.8326 and
42 / 47 / 83 at p > 0.03.
"""
src = open("divbench3.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker, start=0):
    i = src.index(marker, start)
    return src.rfind(NL, 0, i) + 1


def cut(from_marker, to_marker, new):
    """Replace the whole span from the line holding from_marker up to (not
    including) the line holding to_marker."""
    global src
    a = line_start(from_marker)
    b = line_start(to_marker, a + 1)
    src = src[:a] + new + src[b:]


rep('SCRIPT_VERSION = "2026-09-06a-divbench3"', 'SCRIPT_VERSION = "2026-09-07a-divbench4"')

# ---------------------------------------------------------------- 1. the patch
STEAL = r'''STEAL_LOG = {}          # (p, q) -> dict, one per stolen daughter
# a defaultdict, not a dict: stage B clears it per arm so the counters can be
# attributed, and a cleared plain dict makes the next += a KeyError
PATCH_CAPS = defaultdict(int)


def _add_safe_divisions(nodes, edges, pfun=None, steal=None):
    """The orphan-fork patch, VERBATIM, plus an optional second pass that
    takes a daughter the ILP already gave to somebody else.

    `steal` is None (REF: exactly divbench3) or a float margin: the mother's
    ranker probability for the edge she would make must beat the current
    parent's by more than that. -inf runs it unconditionally.

    The orphan pass runs first and to completion, so REF behaviour is
    untouched by construction; the steal pass only sees mothers and
    daughters the orphan pass did not use, shares the per-frame cap, and has
    its own global cap of the same size.
    """
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    orphan_tree = {}
    orphan_ids = {}
    for t, ids in by_t.items():
        orph = [g for g in ids if not pred.get(g)]
        orphan_ids[t] = orph
        if orph:
            orphan_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in orph], float) * SCALE)
    proposals = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = orphan_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = orphan_ids[t + 1][idx]
            if q == c1:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            d_q, q_idx = tree.query(nodes[c1]["xyz"] * SCALE)
            if orphan_ids[t + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            PATCH_CAPS["orphan_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])), steal=0)
        edges.append((p, q))
        succ[p].append(q)
        pred[q].append(p)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    PATCH_CAPS["orphan_added"] += added
    if steal is None:
        return added, 0

    # ---- second pass: the daughter the ILP already placed -----------------
    steal_ids, steal_tree, parent_of = {}, {}, {}
    for t, ids in by_t.items():
        cand = []
        for g in ids:
            pr = pred.get(g, [])
            if len(pr) != 1 or g in used_q:
                continue
            r = pr[0]
            if len(succ.get(r, [])) != 1:      # r already forks: leave it alone
                continue
            cand.append(g)
            parent_of[g] = r
        steal_ids[t] = cand
        if cand:
            steal_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in cand], float) * SCALE)
    sprop = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1 or p in used_p:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = steal_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = steal_ids[t + 1][idx]
            if q == c1 or parent_of.get(q) == p:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            # c1 is itself stealable, and it is its own nearest neighbour:
            # ask for the nearest stealable node that is not c1
            kk = min(2, len(steal_ids[t + 1]))
            dd, ii = tree.query(nodes[c1]["xyz"] * SCALE, k=kk)
            near, d_q = None, np.inf
            for d_, i_ in zip(np.atleast_1d(dd), np.atleast_1d(ii)):
                g_ = steal_ids[t + 1][int(i_)]
                if g_ != c1:
                    near, d_q = g_, float(d_)
                    break
            if near != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            sprop.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    sprop.sort()
    PATCH_CAPS["steal_proposals"] += len(sprop)
    stolen = 0
    max_steal = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    for score, t, p, q, c1, d_pq, d_s, dv_ in sprop:
        if stolen >= max_steal:
            PATCH_CAPS["steal_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        r = parent_of.get(q)
        if r is None or len(succ.get(r, [])) != 1 or succ[r][0] != q:
            continue
        p_pq = pfun(p, q) if pfun else float("nan")
        p_rq = pfun(r, q) if pfun else float("nan")
        have = bool(np.isfinite(p_pq) and np.isfinite(p_rq))
        if not have:
            PATCH_CAPS["steal_no_prob"] += 1
        if steal > float("-inf"):
            # a margin rule needs both numbers; the pair gate is GATE_UM 10 um
            # and the patch reaches 12, so some proposals simply have none
            if not have:
                continue
            if p_pq - p_rq <= steal:
                PATCH_CAPS["steal_margin"] += 1
                continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["steal_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(steal_ids.get(t + 1, [])), steal=1)
        STEAL_LOG[(p, q)] = dict(r=r, p_pq=float(p_pq), p_rq=float(p_rq),
                                 d_rq=_dist_um(nodes[r]["xyz"], nodes[q]["xyz"]))
        edges.remove((r, q))
        edges.append((p, q))
        succ[r].remove(q)
        pred[q] = [p]
        succ[p].append(q)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        stolen += 1
    PATCH_CAPS["steal_added"] += stolen
    return added, stolen
'''
cut("def _add_safe_divisions(nodes, edges):", "def _adj_of(row):", STEAL)

rep('''def patch_final(nodes0, edges0, pool, out_path, divs=True):''',
    '''def patch_final(nodes0, edges0, pool, out_path, divs=True, pfun=None, steal=None):''')
rep('''    FORK_LOG.clear()
    if divs:
        _add_safe_divisions(nodes, edges)''',
    '''    FORK_LOG.clear()
    STEAL_LOG.clear()
    if divs:
        _add_safe_divisions(nodes, edges, pfun=pfun, steal=steal)''')

# ---------------------------------------------------------------- 2. features
rep('''FEAT_NAMES = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap"]''',
    '''FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
FEAT_NEW = ["dv2", "ang", "sc_q_r", "sc_p_r", "dens5", "p_q_marg", "q_ends", "steal"]
FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order''')

NEWROWS = r'''def _aux_extra(seeds, pmap):
    """Per-movie lookups the new fork features need, built once per solve.

    scsort: every detection score in the movie, sorted, so a raw score can be
            read as a per-movie quantile (sc_q / sc_p are the two strongest
            features and both are raw scores whose scale moves with the movie
            and with the imaging batch).
    best2 : for each (t, j) the two best ranker probabilities over all parents
            i in frame t, with their i, so "is this mother the best available
            explanation for this node" is a lookup.
    """
    allsc = np.concatenate([np.asarray(s_, float) for _c, s_ in seeds]) if seeds \
        else np.zeros(0)
    scsort = np.sort(allsc)
    best2 = {}
    for (t_, i_, j_), v in pmap.items():
        pr = v[0]
        if not np.isfinite(pr):
            continue
        cur = best2.get((t_, j_))
        if cur is None:
            best2[(t_, j_)] = [(pr, i_), (-1.0, -1)]
        elif pr > cur[0][0]:
            cur[1] = cur[0]
            cur[0] = (pr, i_)
        elif pr > cur[1][0]:
            cur[1] = (pr, i_)
    return scsort, best2


def fork_rows(name, k, seeds, aux, tp_forks, fp_forks):
    """Every fork in the final graph, its features, its verdict."""
    pmap, seed_of_node, scsort, best2 = aux
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

    def _rank(v):
        if not np.isfinite(v) or not len(scsort):
            return np.nan
        return float(np.searchsorted(scsort, v) / len(scsort))

    def _step(g):
        n_ = succ.get(g, [])
        return n_[0] if len(n_) == 1 else None

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
        # --- the eight new ones ------------------------------------------
        c1nn = _step(c1n[0]) if len(c1n) == 1 else None
        qnn = _step(qn[0]) if len(qn) == 1 else None
        dv2 = (_dist_um(nodes[c1nn]["xyz"], nodes[qnn]["xyz"]) - d_s
               if c1nn is not None and qnn is not None else np.nan)
        v1 = (np.asarray(nodes[c1]["xyz"], float) - np.asarray(nodes[p]["xyz"], float)) * SCALE
        v2 = (np.asarray(nodes[q]["xyz"], float) - np.asarray(nodes[p]["xyz"], float)) * SCALE
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        ang = (float(np.degrees(np.arccos(np.clip(float(v1 @ v2) / (n1 * n2), -1.0, 1.0))))
               if n1 > 1e-9 and n2 > 1e-9 else np.nan)
        sc_p_v, sc_q_v = _sc(p), _sc(q)
        tr = trees.get(t)
        pxyz = np.asarray(nodes[p]["xyz"], float) * SCALE
        dens10 = (len(tr.query_ball_point(pxyz, r=10.0)) - 1) if tr is not None else np.nan
        dens5 = (len(tr.query_ball_point(pxyz, r=5.0)) - 1) if tr is not None else np.nan
        sq, sp_ = seed_of_node.get(q), seed_of_node.get(p)
        p_q_marg = np.nan
        if sq is not None and sp_ is not None and np.isfinite(p_q):
            b = best2.get((sq[0] - 1, sq[1]))
            if b is not None:
                other = b[0][0] if b[0][1] != sp_[1] else b[1][0]
                if other >= 0:
                    p_q_marg = float(p_q - other)
        cur, steps = q, 0
        while steps < 200:
            nx = _step(cur)
            if nx is None:
                break
            cur, steps = nx, steps + 1
        q_ends = 1.0 if (len(succ.get(cur, [])) == 0 and nodes[cur]["t"] < T - 1) else 0.0
        lg = FORK_LOG.get((p, q), {})
        rank_gap = (p_c1 - p_q) if (p_c1 == p_c1 and p_q == p_q) else np.nan
        row = dict(src=src, d_pc1=d_pc1, d_pq=d_pq, d_s=d_s, dv=dv, p_c1=p_c1, p_q=p_q,
                   pf_c1=pf_c1, pf_q=pf_q, pr_q=pr_q, sc_p=sc_p_v, sc_c1=_sc(c1), sc_q=sc_q_v,
                   len_back=_chain_len(p, pred), len_c1=_chain_len(c1, succ),
                   len_q=_chain_len(q, succ), dens10=dens10,
                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap,
                   dv2=dv2, ang=ang, sc_q_r=_rank(sc_q_v), sc_p_r=_rank(sc_p_v),
                   dens5=dens5, p_q_marg=p_q_marg, q_ends=q_ends,
                   steal=float(lg.get("steal", 0)))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES]))
    return rows
'''
cut("def fork_rows(name, k, seeds, aux, tp_forks, fp_forks):", "def walk_lost(name, scores):",
    NEWROWS)

# ---- walk_lost also reports the 5a cases in full ---------------------------
rep('''def walk_lost(name, scores):
    """divgate2's walk on the pre-patch graph for every GT division the
    scorer did not pair. Returns {condition: count} and a list of
    (mother_gt_id, condition)."""''',
    '''def walk_lost(name, scores, pfun=None, steal_rows=None):
    """divgate2's walk on the pre-patch graph for every GT division the
    scorer did not pair. Returns {condition: count} and a list of
    (mother_gt_id, condition).

    When `steal_rows` is given, every case that stops at 5a is written into
    it in full: the ranker's p for the edge the patch WOULD make against the
    p for the edge the ILP DID make, both distances, whether the competing
    edge is a true GT edge, and whether the competitor already forks (which
    the steal pass refuses). divsteal measured this on Zebrahub with n=54 and
    on ONE case in each competition domain; this is the competition
    population, on production primaries."""''')
rep('''            elif pred.get(q):
                cond = "5a_sister_not_orphan"''',
    '''            elif pred.get(q):
                cond = "5a_sister_not_orphan"
                if steal_rows is not None:
                    r = pred[q][0]
                    p_pq = pfun(ga, q) if pfun else float("nan")
                    p_rq = pfun(r, q) if pfun else float("nan")
                    gr = gt_row_of(r)
                    gq = gt_row_of(q)
                    true_other = (1 if (gr is not None and gq is not None
                                        and (gr, gq) in gt_edge_rows) else
                                  (0 if (gr is not None and gq is not None) else -1))
                    c1n_, qn_ = succ.get(c1, []), succ.get(q, [])
                    dv_ = (_dist_um(nodes[c1n_[0]]["xyz"], nodes[qn_[0]]["xyz"])
                           - _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
                           if len(c1n_) == 1 and len(qn_) == 1 else float("nan"))
                    steal_rows.append(dict(
                        movie=name, gt_mother=int(gid), p_pq=float(p_pq),
                        p_rq=float(p_rq),
                        d_pq=_dist_um(nodes[ga]["xyz"], nodes[q]["xyz"]),
                        d_rq=_dist_um(nodes[r]["xyz"], nodes[q]["xyz"]),
                        d_s=_dist_um(nodes[c1]["xyz"], nodes[q]["xyz"]),
                        dv=float(dv_), true_other=true_other,
                        r_forks=1 if len(succ.get(r, [])) > 1 else 0,
                        q_multi_pred=1 if len(pred.get(q, [])) > 1 else 0))''')
rep('''    g_t, g_pos, gids, mothers = _gt_struct(name)
    nodes, edges = PRE_STATE["nodes"], PRE_STATE["edges"]''',
    '''    g_t, g_pos, gids, mothers = _gt_struct(name)
    nodes, edges = PRE_STATE["nodes"], PRE_STATE["edges"]
    gt_edge_rows, gt_by_t = _gt_edge_rows(name)

    def gt_row_of(g):
        """graph node -> GT row within MATCH_UM, or None."""
        t_ = int(nodes[g]["t"])
        b = gt_by_t.get(t_)
        if b is None:
            return None
        rows_, tr_ = b
        d_, i_ = tr_.query(np.asarray(nodes[g]["xyz"], float) * SCALE)
        return rows_[int(i_)] if d_ <= MATCH_UM else None''')
rep('''def _chain_len(start, step_map, cap=12):''',
    '''_GTE_CACHE = {}


def _gt_edge_rows(name):
    """(set of GT edges as (row, row), {t: (rows, KDTree)}) for the true-other
    test in the 5a walk. Cached: the geff read is not free."""
    if name in _GTE_CACHE:
        return _GTE_CACHE[name]
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1)
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    es = set()
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            es.add((ra, rb))
    by_t = {}
    for t_ in np.unique(g_t):
        rows_ = np.where(g_t == t_)[0]
        by_t[int(t_)] = (rows_, cKDTree(g_pos[rows_] * SCALE))
    _GTE_CACHE[name] = (es, by_t)
    return _GTE_CACHE[name]


def _chain_len(start, step_map, cap=12):''')

# ---------------------------------------------------------------- 3. stage B
STAGE_B = r'''print(f"\n=== STAGE B: {len(SIX)} 6bba movies on F4 primaries, one production solve "
      f"each (out-of-fold +BIDIR, disappearance {ILP_DISAPP}, floor {PROD_FLOOR}); "
      f"REF patch + {len(STEAL_ARMS)} steal arms, every fork logged and judged, every "
      f"lost division walked and every 5a case written out ===", flush=True)
print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J 0.9216 "
      "(F4-6bba hard, div_J 0.0606). Soft: the production +BIDIR pkl at floor 0.48 "
      "on the same 12 = ilpsweep's F48 row 0.9261. REF over the 128 must read "
      "50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266 (divbench3).", flush=True)

TMP = "/tmp/db4"            # geffs are intermediates: ~900 of them under
pathlib.Path(TMP).mkdir(parents=True, exist_ok=True)   # /kaggle/working make
CAPS_BY_ARM = {}           # the output slow to list and slow to fetch
ROWS = []
PER = {}
ANCHOR_ROWS = []
SOFT_ROWS = []
INCOMPLETE = []
FORKS = []             # REF forks: (name, fold, p, q, label, feats)
ARM_FORKS = {a: [] for a, _ in STEAL_ARMS}
ARM_ROWS = {a: [] for a, _ in STEAL_ARMS}
ARM_PER = {a: {} for a, _ in STEAL_ARMS}
ARM_MOVIE = {a: {} for a, _ in STEAL_ARMS}
STEAL5A = []           # one dict per GT division that stops at 5a
MOVIE = {}             # name -> dict(arr, ids, edges, rm)
WALK = defaultdict(int)
WALK_FOLD = {0: defaultdict(int), 1: defaultdict(int)}
tb = time.time()


def _snapshot():
    """A COMPACT final graph: coordinates as float32, edges already remapped
    to row indices as int32, and (p, q) -> edge row for the forks the pruner
    will want to remove. divbench3 kept python lists and dicts here and one
    arm; this run keeps four, so the same layout would be several GB.
    """
    nodes, edges, ids = PRE_STATE["nodes"], PRE_STATE["final_edges"], PRE_STATE["ids"]
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], np.float32)
    rm = {g: i for i, g in enumerate(ids)}
    e = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], np.int32) if edges \
        else np.zeros((0, 2), np.int32)
    out = defaultdict(int)
    for a_, _b in edges:
        out[a_] += 1
    key = {(a_, b_): i for i, (a_, b_) in enumerate(edges) if out[a_] > 1}
    return dict(arr=arr, e=e, key=key)


def _rescore(name, snap, drop_rows, tag):
    """Write the final graph minus those edge ROWS, score it."""
    if drop_rows:
        keep = np.ones(len(snap["e"]), bool)
        keep[np.fromiter(drop_rows, int, len(drop_rows))] = False
        e3 = snap["e"][keep]
    else:
        e3 = snap["e"]
    outp = pathlib.Path(f"{TMP}/prune_{tag}") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    dr._save(str(outp), np.asarray(snap["arr"], float), np.asarray(e3, int))
    return score_one(outp, name)


def _judge(name, k, outp, seeds, aux):
    """official scorer + score_divisions + the fork table for one graph."""
    row = score_one(outp, name)
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(outp))
    g = res_[0] if isinstance(res_, tuple) else res_
    ds = score_divisions(g, dset.tracks, scale=dset.scale)
    fr = fork_rows(name, k, seeds, aux, ds.tp_forks, ds.fp_forks)
    del g, res_, dset
    return row, ds, fr


for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    nodes0, edges0, pool, aux0 = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                           disapp=ILP_DISAPP, floor=PROD_FLOOR)
    pmap, seed_of_node = aux0
    scsort, best2 = _aux_extra(seeds, pmap)
    aux = (pmap, seed_of_node, scsort, best2)

    def pfun(a, b, _pm=pmap, _so=seed_of_node):
        sa, sb = _so.get(a), _so.get(b)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            return float("nan")
        return _pm.get((sa[0], sa[1], sb[1]), (float("nan"),))[0]

    outp = pathlib.Path(f"{TMP}/ref") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    PATCH_CAPS.clear()
    patch_final(nodes0, edges0, pool, outp)
    for k_, v_ in PATCH_CAPS.items():
        CAPS_BY_ARM.setdefault("REF", defaultdict(int))[k_] += v_
    row, ds, fr = _judge(name, k, outp, seeds, aux)
    if (len(ds.tp_forks), len(ds.fp_forks)) != (row["division_tp"], row["division_fp"]):
        warn_once("fork_count_mismatch",
                  f"{name}: score_divisions tp/fp {len(ds.tp_forks)}/{len(ds.fp_forks)} "
                  f"vs per-sample {row['division_tp']}/{row['division_fp']}")
    ROWS.append(row)
    PER[name] = row
    FORKS.extend(fr)
    MOVIE[name] = _snapshot()
    scores = {int(kk): int(vv) for kk, vv in ds.scores.items()}
    n5a_before = len(STEAL5A)
    w, det = walk_lost(name, scores, pfun=pfun, steal_rows=STEAL5A)
    for c_, n_ in w.items():
        WALK[c_] += n_
        WALK_FOLD[k][c_] += n_
    del ds

    cells = []
    for arm, margin in STEAL_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa, pfun=pfun, steal=margin)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        n_st = len(STEAL_LOG)
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        cells.append(f"{arm} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/"
                     f"{row_a['division_fn']} {_adj_of(row_a):.4f}")
        del ds_a

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
    n_patch = sum(1 for r_ in fr if r_[5][0] == 1)
    print(f"  {name} fold{k} gt_div={row['division_tp'] + row['division_fn']:>2} "
          f"div {row['division_tp']}/{row['division_fp']}/{row['division_fn']:<3} "
          f"forks {len(fr):>4} ({n_patch} patch) 5a={len(STEAL5A) - n5a_before} "
          f"lost: {', '.join(f'{c_} {n_}' for c_, n_ in sorted(w.items()) if c_ != 'recovered') or '-'}"
          f"   {_adj_of(row):.4f} | " + " | ".join(cells)
          + f"{extra}  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool, aux, aux0, pmap, seed_of_node, best2
    gc.collect()
print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)

# ---- report ----------------------------------------------------------------
print("\n=== ANCHORS / official scorer ===", flush=True)
ANCHOR_FAIL = []
if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):
    ra = _with_ratio(ANCHOR_ROWS)
    got = float(ra["adj_edge_jaccard"])
    ok = abs(got - 0.9216) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  F4-6bba(12)  adj_J {got:.4f}  div_J {ra['division_jaccard']:.4f}"
          f"  want 0.9216 HARD (div_J 0.0606) -> {'OK' if ok else 'MISS'} ({got - 0.9216:+.4f})",
          flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", got, 0.9216))
    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.9261 soft (ilpsweep F48) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.9261) <= ANCHOR_TOL else 'MISS'}",
          flush=True)
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", float("nan"), 0.9216))

RES = _with_ratio(ROWS)
TP = sum(r["division_tp"] for r in ROWS)
FP = sum(r["division_fp"] for r in ROWS)
FN = sum(r["division_fn"] for r in ROWS)
N_GT = TP + FN
base_J = TP / max(1, TP + FP + FN)
ref_ok = (TP, FP, FN) == (50, 249, 75)
print(f"  REF   F4-6bba(128) div {TP}/{FP}/{FN} div_J {base_J:.4f} adj_J "
      f"{RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f}  want 50/249/75 "
      f"0.1337 0.9133 0.9266 (divbench3) -> {'OK' if ref_ok else 'MISS'}", flush=True)
if not ref_ok:
    ANCHOR_FAIL.append(("REF_FORKS", "F4-6bba", TP * 1000 + FP, 50 * 1000 + 249))
print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)
for k in (0, 1):
    rows_k = [PER[n] for n in SIX if FOLD_OF[n] == k and n in PER]
    rk = _with_ratio(rows_k)
    print(f"  fold{k}: div {sum(r['division_tp'] for r in rows_k)}/"
          f"{sum(r['division_fp'] for r in rows_k)}/{sum(r['division_fn'] for r in rows_k)} "
          f"div_J {rk['division_jaccard']:.4f} adj_J {rk['adj_edge_jaccard']:.4f} "
          f"score {rk['score']:.4f}", flush=True)

print("\n=== WHERE THE LOST DIVISIONS GO (F4; divbench3 in brackets) ===", flush=True)
BASE_WALK = {"mother_absent": 9, "daughter_absent": 22, "5a_sister_not_orphan": 23,
             "8b_diverge_under_2.25um": 11,
             "9_passed_all_gates_then_capped_or_taken": 3, "recovered": 50}
print(f"  {'condition':<42} {'all':>5} {'fold0':>6} {'fold1':>6} {'db3':>6}", flush=True)
for c_ in WALK_ORDER:
    if WALK.get(c_, 0) or BASE_WALK.get(c_, 0):
        print(f"  {c_:<42} {WALK.get(c_, 0):>5} {WALK_FOLD[0].get(c_, 0):>6} "
              f"{WALK_FOLD[1].get(c_, 0):>6} {BASE_WALK.get(c_, 0):>6}", flush=True)

print("\n=== THE PATCH'S OWN BUDGET, per arm, summed over the 128 movies "
      "(rule: instrument the stage before sweeping it -- a proposal refused by "
      "a CAP is not a proposal refused on QUALITY) ===", flush=True)
_ck = ["orphan_added", "orphan_global", "orphan_frame", "steal_proposals",
       "steal_added", "steal_no_prob", "steal_margin", "steal_global", "steal_frame"]
print(f"  {'arm':<12}" + "".join(f"{k_:>17}" for k_ in _ck), flush=True)
for a_ in ["REF"] + [x[0] for x in STEAL_ARMS]:
    d_ = CAPS_BY_ARM.get(a_, {})
    print(f"  {a_:<12}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _ck), flush=True)

# ---- the 5a cases, in full --------------------------------------------------
print(f"\n=== 5a: EVERY GT DIVISION WHOSE SECOND DAUGHTER THE ILP HAD ALREADY GIVEN "
      f"AWAY ({len(STEAL5A)} cases, F4, 128 6bba movies) ===", flush=True)
with open("/kaggle/working/steal5a.csv", "w") as fh:
    keys = ["movie", "gt_mother", "p_pq", "p_rq", "d_pq", "d_rq", "d_s", "dv",
            "true_other", "r_forks", "q_multi_pred"]
    fh.write(",".join(keys) + "\n")
    for r_ in STEAL5A:
        fh.write(",".join(str(r_[k_]) for k_ in keys) + "\n")
if STEAL5A:
    def _q(vals, f):
        v = np.array([x for x in vals if np.isfinite(x)], float)
        return float(np.quantile(v, f)) if len(v) else float("nan")
    pm = [r_["p_pq"] for r_ in STEAL5A]
    po = [r_["p_rq"] for r_ in STEAL5A]
    both = [(a, b) for a, b in zip(pm, po) if np.isfinite(a) and np.isfinite(b)]
    print(f"  {'quantity':<44} {'value':>28}", flush=True)
    print(f"  {'competing edge annotated at both ends':<44} "
          f"{sum(1 for r_ in STEAL5A if r_['true_other'] >= 0):>28}", flush=True)
    print(f"  {'... of those, a TRUE GT edge':<44} "
          f"{sum(1 for r_ in STEAL5A if r_['true_other'] == 1):>28}", flush=True)
    print(f"  {'competitor already forks (steal refuses)':<44} "
          f"{sum(r_['r_forks'] for r_ in STEAL5A):>28}", flush=True)
    print(f"  {'p_mother median (p25 / p75)':<44} "
          f"{_q(pm, .5):.3f} ({_q(pm, .25):.3f} / {_q(pm, .75):.3f})".rjust(28), flush=True)
    print(f"  {'p_other  median (p25 / p75)':<44} "
          f"{_q(po, .5):.3f} ({_q(po, .25):.3f} / {_q(po, .75):.3f})".rjust(28), flush=True)
    print(f"  {'both probabilities available':<44} {len(both):>28}", flush=True)
    for m_ in (0.0, 0.1, 0.2):
        n_ = sum(1 for a, b in both if a - b > m_)
        print(f"  {'p_mother beats p_other by more than ' + str(m_):<44} {n_:>28}", flush=True)
    print(f"  {'d_mother median':<44} {_q([r_['d_pq'] for r_ in STEAL5A], .5):>27.2f}u",
          flush=True)
    print(f"  {'d_other  median':<44} {_q([r_['d_rq'] for r_ in STEAL5A], .5):>27.2f}u",
          flush=True)
    print("  steal5a.csv written", flush=True)

# ---- the forks ---------------------------------------------------------------
X_f = np.array([r_[5] for r_ in FORKS], float) if FORKS else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in FORKS], int)
fold_f = np.array([r_[1] for r_ in FORKS], int)
src_f = X_f[:, 0].astype(int) if len(X_f) else np.zeros(0, int)
counted = lab >= 1
with open("/kaggle/working/forks_f4_v4.csv", "w") as fh:
    fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\n")
    for r_ in FORKS:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\n")
for arm, _m in STEAL_ARMS:
    with open(f"/kaggle/working/forks_f4_{arm}.csv", "w") as fh:
        fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\n")
        for r_ in ARM_FORKS[arm]:
            fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\n")
print(f"\n=== EVERY FORK THE 128 REF GRAPHS CARRY: {len(FORKS)} (divbench3: 7,004); "
      f"verdict x source; forks_f4_v4.csv + one csv per steal arm written ===", flush=True)
print(f"  {'source':<8} {'TP':>6} {'FP':>6} {'uncounted':>10} {'precision(counted)':>19}", flush=True)
for s_, nm in ((1, "patch"), (0, "ilp")):
    m_ = src_f == s_
    tp_, fp_, un_ = int((lab[m_] == 2).sum()), int((lab[m_] == 1).sum()), int((lab[m_] == 0).sum())
    print(f"  {nm:<8} {tp_:>6} {fp_:>6} {un_:>10} {tp_ / max(1, tp_ + fp_):>19.3f}", flush=True)

print("\n=== FORK FEATURES by verdict (median TP / FP / uncounted; AUC TP-vs-FP on counted). "
      "The eight new ones are the last eight rows ===", flush=True)
print(f"  {'feature':<9} {'TP med':>8} {'FP med':>8} {'UNC med':>8} {'AUC':>7} {'n_TP':>5} {'n_FP':>5}",
      flush=True)
for j_, fn_ in enumerate(FEAT_NAMES):
    if fn_ == "src":
        continue
    col = X_f[:, j_]
    meds = []
    for L_ in (2, 1, 0):
        v = col[(lab == L_) & np.isfinite(col)]
        meds.append(np.median(v) if len(v) else np.nan)
    mm = counted & np.isfinite(col)
    auc = (_auc((lab[mm] == 2).astype(int), col[mm])
           if mm.sum() and (lab[mm] == 2).any() and (lab[mm] == 1).any() else float("nan"))
    print(f"  {fn_:<9} {meds[0]:>8.3f} {meds[1]:>8.3f} {meds[2]:>8.3f} {auc:>7.3f} "
          f"{int((mm & (lab == 2)).sum()):>5} {int((mm & (lab == 1)).sum()):>5}", flush=True)


def _curve_line(keep, label, ngt=None, lab_=None, fold_=None, wf=None):
    lb = lab if lab_ is None else lab_
    fo = fold_f if fold_ is None else fold_
    ng = N_GT if ngt is None else ngt
    tp_k = int((lb[keep] == 2).sum())
    fp_k = int((lb[keep] == 1).sum())
    fn_k = ng - tp_k
    J = tp_k / max(1, tp_k + fp_k + fn_k)
    per = []
    for k in (0, 1):
        mk = keep & (fo == k)
        tpk = int((lb[mk] == 2).sum())
        fpk = int((lb[mk] == 1).sum())
        ngt_k = sum((wf or WALK_FOLD)[k].values())
        per.append(tpk / max(1, tpk + fpk + (ngt_k - tpk)))
    print(f"  {label:<38} {tp_k:>4} {fp_k:>5} {fn_k:>4} {J:>7.4f} {J - base_J:>+8.4f} "
          f"{0.1 * (J - base_J):>+9.4f} {per[0]:>8.4f} {per[1]:>8.4f}", flush=True)
    return J


HDR = (f"  {'cut':<38} {'TP':>4} {'FP':>5} {'FN':>4} {'div_J':>7} {'d div_J':>8} "
       f"{'~d score':>9} {'fold0 J':>8} {'fold1 J':>8}")


def _gate_oof(X_use, folds, n_splits, lab_, fold_):
    """Out-of-fold gate p over every row. n_splits 2 = divbench3's alternating
    movie split (the anchor); 3 = GroupKFold over movies."""
    p = np.full(len(lab_), np.nan)
    cnt = lab_ >= 1
    aucs = []
    for tr_m, te_m in folds:
        tr = tr_m & cnt
        if tr.sum() < 20 or (lab_[tr] == 2).sum() < 5:
            warn_once("gate_fold_small", f"a gate fold had {int(tr.sum())} counted forks")
            continue
        mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                            min_samples_leaf=10, random_state=0)
        mg.fit(X_use[tr], (lab_[tr] == 2).astype(int))
        p[te_m] = mg.predict_proba(X_use[te_m])[:, 1]
        tc = te_m & cnt
        if tc.sum() and (lab_[tc] == 2).any() and (lab_[tc] == 1).any():
            aucs.append((int(tc.sum()), _auc((lab_[tc] == 2).astype(int), p[tc])))
    pooled = (_auc((lab_[cnt] == 2).astype(int), p[cnt])
              if np.isfinite(p[cnt]).all() else float("nan"))
    return p, aucs, pooled


FOLDS2 = [(fold_f != k, fold_f == k) for k in (0, 1)]
_mv = np.array([r_[0] for r_ in FORKS])
_um = sorted(set(_mv))
_g3 = {m: i % 3 for i, m in enumerate(_um)}
g3 = np.array([_g3[m] for m in _mv])
FOLDS3 = [(g3 != k, g3 == k) for k in (0, 1, 2)]

print("\n=== THE FORK GATE (HistGB 100 it, lr 0.05, msl 10; trained on the OTHER fold's "
      "counted forks, TP=1 FP=0). 20 features / 2 folds is divbench3's ANCHOR "
      "(AUC 0.9283 / 0.8326, p>0.03 -> 42/47/83, div_J 0.2442) ===", flush=True)
J20 = list(range(len(FEAT20)))
GATE_P = {}
for tag, cols, folds in (("20 feats, 2 folds (ANCHOR)", J20, FOLDS2),
                         ("20 feats, 3 folds", J20, FOLDS3),
                         ("28 feats, 2 folds", list(range(len(FEAT_NAMES))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT_NAMES))), FOLDS3)):
    p_g, aucs, pooled = _gate_oof(X_f[:, cols], folds, len(folds), lab, fold_f)
    GATE_P[tag] = p_g
    print(f"\n  {tag}: folds " + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + f"   pooled out-of-fold AUC {pooled:.4f}", flush=True)
    print(HDR, flush=True)
    ok_g = np.isfinite(p_g)
    _curve_line(ok_g, "none")
    for thr in (0.01, 0.02, 0.03, 0.05, 0.1):
        _curve_line(ok_g & (p_g > thr), f"fixed p > {thr}")
    for frac in (0.7, 0.8, 0.9):
        t_ = np.quantile(p_g[ok_g], frac)
        _curve_line(ok_g & (p_g > t_), f"quantile {int(frac * 100)}% (p > {t_:.4f})")
print(f"\n  oracle (every counted FP fork gone, every TP kept): div_J "
      f"{TP / max(1, TP + FN):.4f} ({0.1 * (TP / max(1, TP + FN) - base_J):+.4f} of score)",
      flush=True)

# the production gate: every counted fork, 20 features (what ships today) and 28
for tag, cols, names_ in (("f4v4_20", J20, FEAT20),
                          ("f4v4_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):
    mg_full = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)
    mg_full.fit(X_f[counted][:, cols], (lab[counted] == 2).astype(int))
    p_full = mg_full.predict_proba(X_f[:, cols])[:, 1]
    Q_FULL = {f: float(np.quantile(p_full, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
    with open(f"/kaggle/working/fork_gate_{tag}.pkl", "wb") as fh:
        pickle.dump({"model": mg_full, "feature_names": names_, "version": SCRIPT_VERSION,
                     "trained_on": "counted F4 forks, 128 6bba train movies, TP=1 FP=0",
                     "p_quantiles_all_forks": Q_FULL, "detector": "F4",
                     "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}, fh)
    print(f"  fork_gate_{tag}.pkl written; p quantiles over all {len(FORKS)} forks: "
          + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q_FULL.items()), flush=True)

# ---- the steal arms ---------------------------------------------------------
print(f"\n=== THE STEAL ARMS: the patch allowed to take a daughter the ILP had already "
      f"placed, official scorer, all {len(SIX)} movies ===", flush=True)
print(f"  {'arm':<24} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'edge_J':>7} "
      f"{'node_r':>8} {'score':>7} {'d adj_J':>8} {'d div_J':>8} {'d score':>8} "
      f"{'f0 score':>9} {'f1 score':>9}", flush=True)
b_ref = RES
f0_ref = _with_ratio([PER[n] for n in SIX if FOLD_OF[n] == 0 and n in PER])
f1_ref = _with_ratio([PER[n] for n in SIX if FOLD_OF[n] == 1 and n in PER])
print(f"  {'REF (no steal)':<24} {TP:>4}/{FP:>4}/{FN:<4} {base_J:>7.4f} "
      f"{RES['adj_edge_jaccard']:>7.4f} {RES.get('edge_jaccard', float('nan')):>7.4f} "
      f"{RES.get('total_node_ratio', float('nan')):>+8.4f} {RES['score']:>7.4f} "
      f"{0.0:>+8.4f} {0.0:>+8.4f} {0.0:>+8.4f} {f0_ref['score']:>9.4f} "
      f"{f1_ref['score']:>9.4f}", flush=True)
for arm, margin in STEAL_ARMS:
    rc = _with_ratio(ARM_ROWS[arm])
    f0 = _with_ratio([ARM_PER[arm][n] for n in SIX if FOLD_OF[n] == 0 and n in ARM_PER[arm]])
    f1 = _with_ratio([ARM_PER[arm][n] for n in SIX if FOLD_OF[n] == 1 and n in ARM_PER[arm]])
    print(f"  {arm:<24} {sum(r['division_tp'] for r in ARM_ROWS[arm]):>4}/"
          f"{sum(r['division_fp'] for r in ARM_ROWS[arm]):>4}/"
          f"{sum(r['division_fn'] for r in ARM_ROWS[arm]):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} "
          f"{rc.get('edge_jaccard', float('nan')):>7.4f} "
          f"{rc.get('total_node_ratio', float('nan')):>+8.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['division_jaccard'] - base_J:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{f0['score'] - f0_ref['score']:>+9.4f} {f1['score'] - f1_ref['score']:>+9.4f}",
          flush=True)
print("  (the last two columns are the per-fold DELTA against REF, not the level)",
      flush=True)

# ---- steal + gate, the production composition -------------------------------
print(f"\n=== STEAL + GATE (production is patch -> gate; a stolen fork is a fork). "
      f"The gate is trained out-of-fold on REF's counted forks, 20 features, "
      f"and applied to the arm's forks ===", flush=True)
print(f"  {'arm / cut':<38} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'d adj_J':>8} {'d score':>8}", flush=True)
GCOLS = J20
GMODEL = {}
for k in (0, 1):
    tr = (fold_f != k) & counted
    mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                        min_samples_leaf=10, random_state=0)
    mg.fit(X_f[tr][:, GCOLS], (lab[tr] == 2).astype(int))
    GMODEL[k] = mg


def _prune_arm(arm, forks_a, movies_a, per_a, thr, tag):
    by_movie = defaultdict(list)
    Xa = np.array([r_[5] for r_ in forks_a], float) if forks_a else np.zeros((0, len(FEAT_NAMES)))
    pa = np.full(len(forks_a), np.nan)
    for k in (0, 1):
        m_ = np.array([r_[1] == k for r_ in forks_a], bool)
        if m_.any():
            pa[m_] = GMODEL[k].predict_proba(Xa[m_][:, GCOLS])[:, 1]
    for i_, r_ in enumerate(forks_a):
        by_movie[r_[0]].append((i_, r_[2], r_[3]))
    rows_c, pf = [], {0: [], 1: []}
    for name in SIX:
        if name not in movies_a:
            continue
        key = movies_a[name]["key"]
        want = [(p_, q_) for i_, p_, q_ in by_movie[name] if pa[i_] <= thr]
        miss = [e_ for e_ in want if e_ not in key]
        if miss:
            warn_once("prune_edge_missing",
                      f"{name}: {len(miss)} fork edges to prune are not in the "
                      f"snapshot's fork map -> NOT removed, the cut is a LOWER bound")
        drop = {key[e_] for e_ in want if e_ in key}
        row = _rescore(name, movies_a[name], drop, tag) if drop else per_a[name]
        rows_c.append(row)
        pf[FOLD_OF[name]].append(row)
    return _with_ratio(rows_c), rows_c, pf


for thr in (0.03,):
    rc, rows_c, pf = _prune_arm("REF", FORKS, MOVIE, PER, thr, f"ref_p{thr}")
    print(f"  {'REF + gate p>' + str(thr):<38} "
          f"{sum(r['division_tp'] for r in rows_c):>4}/"
          f"{sum(r['division_fp'] for r in rows_c):>4}/"
          f"{sum(r['division_fn'] for r in rows_c):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f}   want 42/47/83 0.2442 (divbench3)",
          flush=True)
    REF_GATED = rc
    for arm, margin in STEAL_ARMS:
        rc2, rows2, pf2 = _prune_arm(arm, ARM_FORKS[arm], ARM_MOVIE[arm], ARM_PER[arm],
                                     thr, f"{arm}_p{thr}")
        f0 = _with_ratio(pf2[0])
        f1 = _with_ratio(pf2[1])
        f0r = _with_ratio(pf[0])
        f1r = _with_ratio(pf[1])
        print(f"  {arm + ' + gate p>' + str(thr):<38} "
              f"{sum(r['division_tp'] for r in rows2):>4}/"
              f"{sum(r['division_fp'] for r in rows2):>4}/"
              f"{sum(r['division_fn'] for r in rows2):<4} {rc2['division_jaccard']:>7.4f} "
              f"{rc2['adj_edge_jaccard']:>7.4f} {rc2['score']:>7.4f} "
              f"{rc2['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
              f"{rc2['score'] - b_ref['score']:>+8.4f}"
              f"   vs REF+gate: {rc2['score'] - REF_GATED['score']:+.4f} "
              f"(fold0 {f0['score'] - f0r['score']:+.4f} fold1 {f1['score'] - f1r['score']:+.4f})",
              flush=True)

print("\n=== reading it ===", flush=True)
print("  (1) The false-fork side: the eight new features are read from their single-"
      "feature AUCs and from the 28-vs-20 gate rows at the same cut, on the same "
      "forks. Offline on divbench3's table the paired bootstrap over movies gives a "
      "90% interval of about +-0.03 on div_J, so a 28-feature gate has to beat 20 by "
      "more than that before it is anything but noise; the pre-registered reading is "
      "that nothing under +0.03 div_J counts, whatever its sign.", flush=True)
print("  (2) The steal: a submission candidate iff its 'd score' is positive in BOTH "
      "folds against REF+gate at the same cut. The unconditional arm is a control, "
      "not a candidate -- if it comes back near zero that is a reason to look for the "
      "cancellation between its good and bad cases, not to close the line.", flush=True)
print("  (3) The 5a table says whether the mechanism can work at all: if the mother's "
      "edge almost never beats the ILP's, the margin rule fires on nothing and the "
      "lever is the LINKER, not the patch.", flush=True)
warning_summary()
print(f"\nDIVBENCH4 DONE in {(time.time() - t_all) / 60:.1f} min (stage B "
      f"{(time.time() - tb) / 60:.1f} min), {len(ROWS)} of {len(SIX)} movies scored", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
if len(ROWS) == len(SIX) and not INCOMPLETE:
    print("DIVBENCH4 OK", flush=True)
else:
    print(f"[degraded] DIVBENCH4 INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)
'''
a = line_start('print(f"\\n=== STAGE B: {len(SIX)} 6bba movies on F4 primaries, one production solve "')
src = src[:a] + STAGE_B

# the steal arms, declared next to the other run constants
rep('PROD_FLOOR = 0.48                                     # ref 56030644, LB 0.933',
    '''PROD_FLOOR = 0.48                                     # ref 56030644, LB 0.933
# (arm name, ranker margin the mother's edge must beat the ILP's by). -inf is
# the unconditional control: it measures the SUM of its good and bad cases,
# which is not a ceiling (2026-08-26), and it says whether the gates bind.
STEAL_ARMS = [("steal_all", float("-inf")), ("steal_m0", 0.0), ("steal_m20", 0.20)]''')

open("divbench4.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
