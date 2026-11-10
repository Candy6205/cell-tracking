"""submit_f48.py -> submit_gate.py: ref 56030644's pipeline (F4, +BIDIR
ranker, ILP floor 0.48) plus the fork gate from divbench3: every fork in
the final graph (the ILP's and the orphan patch's) is scored by
fork_gate_f4.pkl on the same 20 features the bench used, and the
mother->orphan edge of every fork with p <= GATE_THR is removed. Nothing
else changes."""
src = open("submit_f48.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-05a-submit-f48"', 'SCRIPT_VERSION = "2026-09-06a-submit-gate"')

# ---- the gate, loaded next to the ranker ---------------------------------
rep('''print(f"ranker {RANKER_NAME} from {_pk} | pool {len(_pool)} datasets | "
      f"version {_blob.get('version')} | {len(_cols)} features = 18 + "
      f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)
''',
    '''print(f"ranker {RANKER_NAME} from {_pk} | pool {len(_pool)} datasets | "
      f"version {_blob.get('version')} | {len(_cols)} features = 18 + "
      f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)

# ---- the fork gate (divbench3, 2026-09-06) -------------------------------
GATE_NAME = "fork_gate_f4.pkl"
GATE_THR = 0.03            # divbench3: p > 0.03 = +0.0121 local score, both folds up, 44b6 +0.0020
GATE_FEATS = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap"]
GATE = None
_gk = None
import os as _os                                   # noqa: E402
for _root, _dirs, _files in _os.walk("/kaggle/input"):
    if GATE_NAME in _files:
        _gk = _os.path.join(_root, GATE_NAME)
        break
assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-divbench3 as a kernel source")
with open(_gk, "rb") as _f:
    _gb = pickle.load(_f)
assert list(_gb["feature_names"]) == GATE_FEATS, _gb["feature_names"]
assert _gb.get("detector") == "F4", _gb.get("detector")
GATE = _gb["model"]
assert hasattr(GATE, "predict_proba")
print(f"fork gate {GATE_NAME} from {_gk} | version {_gb.get('version')} | "
      f"trained on {_gb.get('trained_on')} | threshold p > {GATE_THR} | "
      f"its p-quantiles over the bench's forks {_gb.get('p_quantiles_all_forks')}", flush=True)
PMAP = {}              # (t, i, j) -> (ranker p, head p_fwd, head p_rev), per movie
SEED_OF_NODE = {}      # ILP node id -> (t, seed index), per movie
FORK_LOG = {}          # (mother, orphan) -> the patch's numbers, per movie
''')

# ---- link_ilp: keep the pair probabilities and the node -> seed map ------
rep('''        extra = _head_columns(head, t, pairs)
        pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
        n_pairs += len(pairs)
        for (i, j), p in zip(pairs, pr):''',
    '''        extra = _head_columns(head, t, pairs)
        pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
        n_pairs += len(pairs)
        for (i, j), p, ex in zip(pairs, pr, extra):
            PMAP[(t, int(i), int(j))] = (float(p), float(ex[0]), float(ex[1]))
        for (i, j), p in zip(pairs, pr):''')
rep('''    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
    coords_rows = []''',
    '''    PMAP.clear()
    SEED_OF_NODE.clear()
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
    coords_rows = []''')
rep('''    nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                     "xyz": nodes_arr[i, 1:].astype(float)}
             for i in range(len(nodes_arr))}
    edges = [(int(a) + 1, int(b) + 1) for a, b in e2]''',
    '''    nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                     "xyz": nodes_arr[i, 1:].astype(float)}
             for i in range(len(nodes_arr))}
    edges = [(int(a) + 1, int(b) + 1) for a, b in e2]
    _cd = {}
    for g_, (t_, z_, y_, x_) in enumerate(coords.tolist()):
        _cd.setdefault((int(t_), int(z_), int(y_), int(x_)), (int(t_), g_ - offset[int(t_)]))
    for i in range(len(nodes_arr)):
        SEED_OF_NODE[i + 1] = _cd.get(tuple(int(round(v)) for v in nodes_arr[i]))''')

# ---- _add_safe_divisions: log what it adds --------------------------------
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

# ---- the gate itself, after the patch ---------------------------------------
rep('''def _emit(ds, nodes, edges):''',
    '''def _chain_len(start, step_map, cap=12):
    n, cur = 0, start
    while n < cap:
        nxt = step_map.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def _gate_prune(nodes, edges, dets):
    """divbench3's fork_rows + the cut: every fork of the final graph gets the
    20 bench features, the gate's p, and loses its mother->orphan edge when
    p <= GATE_THR. Returns the new edge list and (forks, removed)."""
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    T = max(v["t"] for v in nodes.values()) + 1
    seeds = [seeds_of(dets[t]) for t in range(T)]
    trees = {t: (cKDTree(np.asarray(seeds[t][0], float) * SCALE) if len(seeds[t][0]) else None)
             for t in range(T)}

    def _p(a, b):
        sa, sb = SEED_OF_NODE.get(a), SEED_OF_NODE.get(b)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            return (np.nan, np.nan, np.nan)
        return PMAP.get((sa[0], sa[1], sb[1]), (np.nan, np.nan, np.nan))

    def _sc(a):
        sa = SEED_OF_NODE.get(a)
        return float(seeds[sa[0]][1][sa[1]]) if sa is not None else np.nan

    forks = []
    X = []
    for p, ch in succ.items():
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        patched = [q for q in ch if (p, q) in FORK_LOG]
        if patched:
            q = patched[0]
            src = 1
            c1 = FORK_LOG[(p, q)]["c1"]
            if c1 not in ch:
                c1 = [x for x in ch if x != q][0]
        else:
            src = 0
            pr_ = {x: _p(p, x)[0] for x in ch}
            ch_sorted = sorted(ch, key=lambda x: np.nan_to_num(pr_[x], nan=-1.0))
            q, c1 = ch_sorted[0], ch_sorted[-1]
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
        forks.append((p, q))
        X.append([row[f] for f in GATE_FEATS])
    if not forks:
        return edges, (0, 0)
    pg = GATE.predict_proba(np.asarray(X, float))[:, 1]
    drop = {pq for pq, pv in zip(forks, pg) if pv <= GATE_THR}
    kept = [e for e in edges if e not in drop]
    return kept, (len(forks), len(drop))


def _emit(ds, nodes, edges):''')
rep('''    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    _oc = defaultdict(int)
    for a, b in edges:
        _oc[a] += 1
    print(f"    final: nodes={len(nodes)} edges={len(edges)} "
          f"forks={sum(1 for v in _oc.values() if v >= 2)}", flush=True)''',
    '''    _linefit(nodes, edges)
    FORK_LOG.clear()
    _add_safe_divisions(nodes, edges)
    n_before = len(edges)
    try:
        edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets)
    except Exception as _e:                      # noqa: BLE001
        print(f"    WARN fork gate failed ({type(_e).__name__}: {_e}) -- forks kept", flush=True)
        n_forks, n_drop = -1, 0
    _oc = defaultdict(int)
    for a, b in edges:
        _oc[a] += 1
    print(f"    gate: forks={n_forks} removed={n_drop} (edges {n_before} -> {len(edges)})",
          flush=True)
    print(f"    final: nodes={len(nodes)} edges={len(edges)} "
          f"forks={sum(1 for v in _oc.values() if v >= 2)}", flush=True)''')

open("submit_gate.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
