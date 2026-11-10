"""dupsup.py -> nodeledger.py: WHICH of REF's final nodes could be deleted at a profit?

THE TERM. adj_J = edge_J x (1 - 0.1 x (N_pred - N_est) / N_est), clamped at 0 below and
NOT above (metrics.py, reference.md): 10% fewer nodes than the estimate is +1% of J.
Deleting a node earns J x 0.1 / N_est (~4e-8 on 6bba) and, if the node sits on an
annotated cell, loses its TP edges (~1.8e-5 each of J): the deleted set must carry
annotated cells at < ~0.24% against the 5.7% average. Our nodes the public 0.947
pipeline does not have sit on annotated cells at 0.5-1.9% (2026-09-1x reading), so a
low-rate population exists; this kernel finds out WHERE it is.

PER FINAL NODE of REF (after every patch), on the 128 6bba movies: t, detection score
(NaN for an inserted bridge node), in / out degree, component size (nodes), frames from
the component's first frame and to its last, primaries within 10 um in the same frame,
distance to the nearest GT node, the one-to-one match at 7 um (the scorer's radius,
greedy nearest per GT node). PER EDGE: tp (both ends matched, a GT edge), fp_vis (at
least one end matched, not a GT edge), invisible (both ends unmatched: the scorer never
sees it). Written as nl_<movie>.npz with N_est and N_pred.

REPORT. Matched rate and counts by score decile, component size, track position, density;
a set of deletion rules priced exactly per movie from the ledger (TP' / FP' / FN' / N_pred'
with the movie's own N_est), aggregated with the harness's weights, next to the official
REF numbers; the ledger's own REF reconstruction is printed as the sanity row.

Attach: dupsup's 16 sources. CPU: features ~1 h + REF ~70 min + the cross ~1 h; no arms.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-16a-nodeledger"')
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = []             # nodeledger: REF only')

BLOCK = r'''
# ---- nodeledger block ----
NL_SCORE_UM = 7.0
NL_DENS_UM = 10.0
NL_MOVIES = {}                     # name -> dict(est, n_pred, tp, fp, fn, n_gt_edges)
NL_NODES = []                      # per node rows (movie index, ...)
NL_EDGES = []
NL_NCOLS = ["t", "score", "inserted", "in_deg", "out_deg", "comp_len", "from_start", "to_end", "dens10", "d_gt", "matched7"]


def _nl_components(nodes_f, edges_f):
    parent = {nid: nid for nid in nodes_f}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a_, b_ in edges_f:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    comp = {nid: find(nid) for nid in nodes_f}
    return comp


def node_ledger(name, T, seeds, seed_of_node, nodes_f, edges_f, trees, kids, est):
    ids = sorted(nodes_f)
    pos_of = {nid: k for k, nid in enumerate(ids)}
    n = len(ids)
    t_arr = np.array([int(nodes_f[nid]["t"]) for nid in ids], np.int64)
    xyz = np.stack([np.asarray(nodes_f[nid]["xyz"], float) for nid in ids]) * SCALE if n else np.zeros((0, 3))
    score = np.full(n, np.nan, np.float32)
    inserted = np.ones(n, np.int8)
    for k, nid in enumerate(ids):
        s_ = seed_of_node.get(nid)
        if s_ is not None:
            score[k] = float(seeds[s_[0]][1][s_[1]])
            inserted[k] = 0
    in_deg = np.zeros(n, np.int32)
    out_deg = np.zeros(n, np.int32)
    for a_, b_ in edges_f:
        out_deg[pos_of[a_]] += 1
        in_deg[pos_of[b_]] += 1
    comp = _nl_components(nodes_f, edges_f)
    c_size, c_min, c_max = defaultdict(int), {}, {}
    for nid in ids:
        c = comp[nid]
        c_size[c] += 1
        tt = int(nodes_f[nid]["t"])
        c_min[c] = min(c_min.get(c, tt), tt)
        c_max[c] = max(c_max.get(c, tt), tt)
    comp_len = np.array([c_size[comp[nid]] for nid in ids], np.int32)
    from_start = np.array([int(nodes_f[nid]["t"]) - c_min[comp[nid]] for nid in ids], np.int32)
    to_end = np.array([c_max[comp[nid]] - int(nodes_f[nid]["t"]) for nid in ids], np.int32)
    dens = np.zeros(n, np.int32)
    d_gt = np.full(n, np.inf, np.float32)
    g_row = np.full(n, -1, np.int64)
    matched = np.zeros(n, np.int8)
    for t in np.unique(t_arr):
        m = np.where(t_arr == t)[0]
        P = xyz[m]
        tr = cKDTree(P)
        dens[m] = tr.query_ball_point(P, NL_DENS_UM, return_length=True) - 1
        if int(t) in trees:
            rows, gtree = trees[int(t)]
            dd, gi = gtree.query(P)
            d_gt[m] = dd
            g_row[m] = rows[gi]
            # one-to-one at the scorer's radius: each GT node takes its nearest node within 7 um
            best = {}
            for k_, (d_, g_) in zip(m, zip(dd, rows[gi])):
                if d_ <= NL_SCORE_UM and (g_ not in best or d_ < best[g_][0]):
                    best[g_] = (d_, k_)
            for g_, (d_, k_) in best.items():
                matched[k_] = 1
    gmatch = {ids[k]: int(g_row[k]) for k in range(n) if matched[k]}
    kind = np.zeros(len(edges_f), np.int8)
    for e_, (a_, b_) in enumerate(edges_f):
        ga, gb = gmatch.get(a_, -1), gmatch.get(b_, -1)
        if ga < 0 and gb < 0:
            kind[e_] = 0
        elif ga >= 0 and gb >= 0 and gb in kids.get(ga, ()):
            kind[e_] = 1
        else:
            kind[e_] = 2
    n_gt_edges = sum(len(v) for v in kids.values())
    tp = int((kind == 1).sum())
    fp = int((kind == 2).sum())
    fn = n_gt_edges - tp
    NL_MOVIES[name] = dict(est=float(est) if est else float("nan"), n_pred=n, tp=tp, fp=fp, fn=fn, n_gt_edges=n_gt_edges)
    node_arr = np.stack([t_arr, score, inserted, in_deg, out_deg, comp_len, from_start, to_end, dens,
                         np.where(np.isfinite(d_gt), d_gt, 999.0), matched], axis=1).astype(np.float32)
    edge_arr = np.array([[pos_of[a_], pos_of[b_], kind[e_]] for e_, (a_, b_) in enumerate(edges_f)], np.int64).reshape(-1, 3)
    np.savez_compressed(f"/kaggle/working/nl_{name}.npz", nodes=node_arr, edges=edge_arr, cols=np.array(NL_NCOLS),
                        est=NL_MOVIES[name]["est"], n_pred=n, tp=tp, fp=fp, fn=fn)
    NL_NODES.append((name, node_arr))
    NL_EDGES.append((name, edge_arr))
    print(f"  ledger {name}: nodes {n:,} (inserted {int(inserted.sum()):,}), matched at 7 um {int(matched.sum()):,} "
          f"({matched.mean() if n else 0:.3%}), edges tp {tp:,} fp_vis {fp:,} invisible {int((kind == 0).sum()):,}, "
          f"GT edges {n_gt_edges:,}, N_est {NL_MOVIES[name]['est']:.0f}", flush=True)


def nl_price(rule):
    """rule(node_arr) -> bool mask of nodes to delete. Exact per-movie recount from the ledger,
    aggregated with the harness's weights (tp + fp + fn of the movie). Returns (adj_agg, edge_agg,
    ratio, n_deleted, tp_lost, fp_removed, per-movie deltas)."""
    num = den_w = 0.0
    e_num = 0.0
    n_del = tp_lost = fp_rem = 0
    tot_pred = tot_est = 0.0
    per = []
    for (name, nodes), (_n2, edges) in zip(NL_NODES, NL_EDGES):
        mv = NL_MOVIES[name]
        dele = rule(nodes)
        touched = dele[edges[:, 0]] | dele[edges[:, 1]] if len(edges) else np.zeros(0, bool)
        tl = int(((edges[:, 2] == 1) & touched).sum()) if len(edges) else 0
        fr = int(((edges[:, 2] == 2) & touched).sum()) if len(edges) else 0
        tp_ = mv["tp"] - tl
        fp_ = mv["fp"] - fr
        fn_ = mv["fn"] + tl
        denom = tp_ + fp_ + fn_
        j = tp_ / denom if denom else 0.0
        npred = mv["n_pred"] - int(dele.sum())
        ratio = (npred - mv["est"]) / mv["est"] if mv["est"] == mv["est"] and mv["est"] else 0.0
        adj = max(0.0, j * (1 - 0.1 * ratio))
        w = denom
        num += adj * w
        e_num += j * w
        den_w += w
        n_del += int(dele.sum())
        tp_lost += tl
        fp_rem += fr
        tot_pred += npred
        tot_est += mv["est"] if mv["est"] == mv["est"] else 0.0
        per.append(adj)
    return (num / max(den_w, 1e-9), e_num / max(den_w, 1e-9), (tot_pred - tot_est) / max(tot_est, 1e-9),
            n_del, tp_lost, fp_rem, np.array(per))


def nl_summary():
    if not NL_NODES:
        return
    allN = np.concatenate([a for _n, a in NL_NODES])
    c = {k: i for i, k in enumerate(NL_NCOLS)}
    sc, ins, mt = allN[:, c["score"]], allN[:, c["inserted"]] == 1, allN[:, c["matched7"]] == 1
    print(f"\n=== NODELEDGER over {len(NL_NODES)} movies: {len(allN):,} final nodes, {int(ins.sum()):,} inserted, "
          f"matched at 7 um {mt.mean():.3%}; break-even for deletion ~0.24% ===", flush=True)
    base = nl_price(lambda a: np.zeros(len(a), bool))
    print(f"  ledger REF: adj {base[0]:.4f} edge_J {base[1]:.4f} node ratio {base[2]:+.4f}  (official REF: see the arms table)", flush=True)

    def table(title, key, bins, labels):
        print(f"\n  by {title}: bucket / nodes / share / matched at 7 um / tp edges touching", flush=True)
        for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
            m = (key >= lo) & (key < hi)
            if m.sum() == 0:
                continue
            rate = mt[m].mean()
            print(f"    {lab:<22} {int(m.sum()):>9,} {m.mean():>7.3%} {rate:>9.3%}  {'<- below break-even' if rate < 0.0024 else ''}", flush=True)
    prim = ~ins
    q = np.nanquantile(sc[prim], np.linspace(0, 1, 11))
    table("detection score decile (primaries)", np.where(prim, sc, np.inf), list(q[:-1]) + [q[-1] + 1e-6],
          [f"[{a:.3f}, {b:.3f})" for a, b in zip(q[:-1], q[1:])])
    table("component size (nodes)", allN[:, c["comp_len"]], [0, 4, 5, 6, 8, 12, 20, 50, 10 ** 9],
          ["1-3", "4", "5", "6-7", "8-11", "12-19", "20-49", "50+"])
    pos = np.minimum(allN[:, c["from_start"]], allN[:, c["to_end"]])
    table("frames to the nearer track end", pos, [0, 1, 2, 3, 5, 10 ** 9], ["0 (an end)", "1", "2", "3-4", "5+"])
    table("primaries within 10 um, same frame", allN[:, c["dens10"]], [0, 1, 2, 4, 8, 10 ** 9], ["0", "1", "2-3", "4-7", "8+"])
    table("in-degree x out-degree", allN[:, c["in_deg"]] * 10 + allN[:, c["out_deg"]], [0, 1, 2, 10, 11, 12, 20, 21, 22, 10 ** 9],
          ["in0 out0", "in0 out1", "in0 out2+", "in1 out0", "in1 out1", "in1 out2+", "in2+ out0", "in2+ out1", "in2+ out2+"])
    print(f"\n  inserted (bridge) nodes: {int(ins.sum()):,}, matched {mt[ins].mean() if ins.any() else float('nan'):.3%}", flush=True)
    RULES = [
        ("score < 0.20", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.20)),
        ("score < 0.25", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.25)),
        ("score < 0.30", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.30)),
        ("component <= 4 nodes", lambda a: a[:, c['comp_len']] <= 4),
        ("component <= 5 nodes", lambda a: a[:, c['comp_len']] <= 5),
        ("component <= 6 nodes", lambda a: a[:, c['comp_len']] <= 6),
        ("track end & score < 0.25", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.25) & (np.minimum(a[:, c['from_start']], a[:, c['to_end']]) == 0)),
        ("track end & score < 0.30", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.30) & (np.minimum(a[:, c['from_start']], a[:, c['to_end']]) == 0)),
        ("dens10 == 0 & score < 0.30", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.30) & (a[:, c['dens10']] == 0)),
        ("dens10 >= 4 & score < 0.25", lambda a: (a[:, c['inserted']] == 0) & (a[:, c['score']] < 0.25) & (a[:, c['dens10']] >= 4)),
        ("inserted bridge nodes", lambda a: a[:, c['inserted']] == 1),
        ("in0 out0 (isolated)", lambda a: (a[:, c['in_deg']] == 0) & (a[:, c['out_deg']] == 0)),
    ]
    print(f"\n  === DELETION RULES, priced exactly from the ledger (weighted aggregate over the movies, REF ledger adj {base[0]:.4f}) ===", flush=True)
    print(f"  {'rule':<28} {'deleted':>9} {'share':>7} {'matched%':>9} {'tp lost':>8} {'fp gone':>8} {'ratio':>8} {'d edge_J':>9} {'d adj_J':>9} {'movies +/-':>11}", flush=True)
    for nm, r in RULES:
        adj, ej, ratio, nd, tl, fr, per = nl_price(r)
        m_all = np.concatenate([r(a) for _n, a in NL_NODES])
        rate = mt[m_all].mean() if m_all.any() else float("nan")
        d = per - base[6]
        print(f"  {nm:<28} {nd:>9,} {m_all.mean():>7.3%} {rate:>9.3%} {tl:>8,} {fr:>8,} {ratio:>+8.4f} {ej - base[1]:>+9.4f} {adj - base[0]:>+9.4f} "
              f"{int((d > 0).sum()):>5}/{int((d < 0).sum()):<5}", flush=True)
    with open("/kaggle/working/nodeledger_movies.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("dataset,est,n_pred,tp,fp,fn,n_gt_edges\n")
        for nm_, mv in sorted(NL_MOVIES.items()):
            fh.write(f"{nm_},{mv['est']},{mv['n_pred']},{mv['tp']},{mv['fp']},{mv['fn']},{mv['n_gt_edges']}\n")
    print("  nl_<movie>.npz + nodeledger_movies.csv written", flush=True)
# ---- end nodeledger block ----
'''
rep('\ndef _judge(name, k, outp, seeds, aux):\n', BLOCK + '\n\ndef _judge(name, k, outp, seeds, aux):\n')

rep('    patch_final(nodes0, edges0, pool, outp)\n    BR_CTX["gt"] = None\n',
    '    _nl_nodes, _nl_edges = patch_final(nodes0, edges0, pool, outp)\n'
    '    node_ledger(name, T, seeds, seed_of_node, _nl_nodes, _nl_edges, _trees, _kids, _gt["extra"].get("estimated_number_of_nodes"))\n'
    '    del _nl_nodes, _nl_edges\n'
    '    BR_CTX["gt"] = None\n')
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', '\nnl_summary()\n\nBR_ARMS_RUN = list(BR_ARMS)\n')
rep('print("DUPSUP OK", flush=True)', 'print("NODELEDGER OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] NODELEDGER INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"nodeledger_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"nodeledger_arm_{a_}.csv"')

assert src.count("node_ledger(") == 2 and src.count("nl_summary()") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nodeledger.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written nodeledger.py", len(src))
