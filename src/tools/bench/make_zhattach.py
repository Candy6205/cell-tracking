"""zhpara4.py -> zhattach.py: the attach-only vote arms on Zebrahub, scored dense AND on a train-like sparse GT.

WHY (2026-09-20). zhvote: out of sample (Zebrahub, never seen by the four weights) the vote
candidates are 2.5x purer relative to the shipped primaries than on the in-sample 6bba
bench (0.32x against 0.13x), 17% stand on a missed cell, and they recover 40% of the missed
cells (6bba 31%). The 6bba bench cannot show what that is worth to the METRIC because its
detections are in-sample. This kernel runs the attach-only arms of vtattach on the Zebrahub
crops with zhpara4's solver and scores every arm two ways:
  DENSE   the crop's full GT (what zhpara4 scores; N_est = the dense node count)
  SPARSE  the GT thinned to whole tracks at a train-like node share (FRAC 0.055, the
          competition's 113k annotated nodes over ~2.1M cells), N_DRAWS random draws with
          fixed seeds, N_est = the dense node count (the metric's estimated cell count).
          The sparse view is the one the metric takes: unannotated candidates only cost nodes.
The delta arm - REF per crop, averaged over draws, then over crops with a paired bootstrap,
is the out-of-sample sign the in-sample bench could not give. It cannot match the hidden
set exactly (different instrument, dense-vs-machine annotation), only approximate it.

ARMS (each its own solve; the D187 18-column ranker, ILP_FLOOR raised to the production
0.48 as zhbridge did, disappearance 1.5, DIV_W 0.55, the shipped post-link patches):
  REF   the shipped hpara peaks alone (zhvote's c/s minus the appended candidates)
  VT    every candidate with >= 2 votes admitted, free linking (vttrace's VT17)
  XB    the same, candidate-candidate pairs withheld from the ILP, dangling candidates
        (degree <= 1 after the solve) dropped (vtattach's VT17XB)
  XBF   XB with the admission f4 >= 0.10 (zhvote: keeps 62% of the candidates and 83% of
        those on missed cells)
patch_and_save allocates gap-close ids past the LARGEST id (vtattach v2's XB bug).

Attach: zhpara4's sources + faustine649/biohub-zhvote (the zhvote_<crop>.npz caches).
CPU ~4-5 h (41 crops x 4 solves + 4 dense metrics + 4 x N_DRAWS sparse metrics), the
per-crop CSV rewritten after every crop.
"""
import os

src = open("zhpara4.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09c-zhpara4"', 'SCRIPT_VERSION = "2026-09-20a-zhattach"')
rep('ILP_FLOOR = 0.10                            # candidates fed to the ILP',
    'ILP_FLOOR = 0.48                            # PRODUCTION floor (zhbridge did the same); zhpara4 used 0.10')

# 1. solve_one: the attach-only rule and the node -> seed map
rep('def solve_one(dets, T):\n', 'def solve_one(dets, T, n_ship=None, attach_only=False):\n')
rep('''    cand_edges = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= ILP_FLOOR:
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))
''', '''    cand_edges = []
    n_withheld = 0
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= ILP_FLOOR:
                if attach_only and n_ship is not None and i_ >= n_ship[t] and j_ >= n_ship[t + 1]:
                    n_withheld += 1          # candidate -> candidate: never offered
                    continue
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))
    SOLVE_STAT["withheld"] = n_withheld
''')
rep('''    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)
''', '''    cd = {}
    for g_, (t_, z_, y_, x_) in enumerate(coords.tolist()):
        cd.setdefault((int(t_), int(z_), int(y_), int(x_)), (int(t_), g_ - offset[int(t_)]))
    son = {}
    for i_ in range(len(nodes_arr)):
        key = tuple(int(round(v)) for v in nodes_arr[i_])
        son[i_ + 1] = cd.get(key)
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), son
''')

# 2. patch_and_save: gap-close ids past the largest surviving id
rep('''    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
''', '''    edges = list(edges0)
    nid2 = (max(nodes) if nodes else 0) + 1     # past the LARGEST id: after a node drop len+1 collides (vtattach v2)
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
''')

# 3. the crop loop
cut = "NGT = {}\nMOVED = []\nt_zh = time.time()\ndone = []\nfor name in BOTH:"
assert src.count(cut) == 1
head = src[:src.index(cut)]
assert "SOLVE_STAT" in head and "def patch_and_save(" in head
head = head.replace("ARM_SRC = [(\"REF\", ZH_REF), (\"HP\", ZH_HP)]\nZROWS = {tag: {} for tag, _ in ARM_SRC}\n",
                    "ARM_SRC = [(\"REF\", ZH_REF), (\"HP\", ZH_HP)]   # zhpara4's, unused here\nZROWS = {}\n")
assert head.count("SOLVE_STAT[") == 1
head = head.replace("def solve_one(dets, T, n_ship=None, attach_only=False):\n",
                    "SOLVE_STAT = {}\n\n\ndef solve_one(dets, T, n_ship=None, attach_only=False):\n")

LOOP = r'''
# ---------------------------------------------------------------- zhattach
FRAC = 0.055               # train-like annotated node share (113,121 annotated over ~2.1M cells)
N_DRAWS = 6
ADMIT_F4 = 0.10
DENSE_MAX = 20000          # dense scoring only below this GT node count: the official matcher's cost matrix is N_pred x N_gt (zhpara3/4 were OOM-killed there)
ZHV = {f.stem[len("zhvote_"):]: f for f in pathlib.Path("/kaggle/input").glob("*/**/zhvote_*.npz")}
assert ZHV, "attach faustine649/biohub-zhvote (zhvote_<crop>.npz)"
CROPS = sorted(ZHV, key=lambda n: ZHV[n].stat().st_size)      # smallest first: a kill costs the fewest
print(f"zhattach: {len(CROPS)} zhvote caches", flush=True)
ARMS = [("REF", dict(cand=False)), ("VT", dict(cand=True, attach=False, dangle=False, fmin=None)),
        ("XB", dict(cand=True, attach=True, dangle=True, fmin=None)), ("XBF", dict(cand=True, attach=True, dangle=True, fmin=ADMIT_F4))]
ZROWS = {a: {} for a, _ in ARMS}          # dense rows
SROWS = {a: {} for a, _ in ARMS}          # per crop: list of sparse rows (one per draw)
CNT = {a: defaultdict(int) for a, _ in ARMS}
NGT = {}
done = []


def zh_gt_tracks(z, T):
    """(nodes (t, z, y, x) voxels, track ids, edges) in zh_load's order: within-track consecutive frames plus the
    parent link at a track's first frame."""
    nodes = np.asarray(z["nodes"], float).reshape(-1, 4)
    tid = np.asarray(z["track_id"], np.int64)
    pid = np.asarray(z["parent_track_id"], np.int64)
    keep = (nodes[:, 0] >= 0) & (nodes[:, 0] < T)
    nodes, tid, pid = nodes[keep], tid[keep], pid[keep]
    order = np.lexsort((nodes[:, 0], tid))
    nodes, tid, pid = nodes[order], tid[order], pid[order]
    row_of = {(int(tid[i]), int(nodes[i, 0])): i for i in range(len(nodes))}
    edges = []
    for i in range(1, len(nodes)):
        if tid[i] == tid[i - 1] and nodes[i, 0] == nodes[i - 1, 0] + 1:
            edges.append((i - 1, i))
    first, par_of = {}, {}
    for i in range(len(nodes)):
        k = int(tid[i])
        if k not in first or nodes[i, 0] < nodes[first[k], 0]:
            first[k] = i
        if pid[i] > 0:
            par_of[k] = int(pid[i])
    for k, i in first.items():
        p = par_of.get(k)
        if p is None:
            continue
        j = row_of.get((p, int(nodes[i, 0]) - 1))
        if j is not None:
            edges.append((j, i))
    return nodes, tid, np.asarray(edges, int).reshape(-1, 2)


def sparse_draw(nodes, tid, edges, frac, seed):
    """Keep whole tracks, each with probability frac (so the expected node share is frac); edges with both ends kept,
    re-indexed. Returns (nodes, edges)."""
    rng = np.random.RandomState(seed)
    tracks = np.unique(tid)
    keep_t = set(tracks[rng.random_sample(len(tracks)) < frac].tolist())
    keep = np.array([int(v) in keep_t for v in tid], bool)
    new_idx = np.full(len(nodes), -1, int)
    new_idx[keep] = np.arange(int(keep.sum()))
    e = [(new_idx[a], new_idx[b]) for a, b in edges if keep[a] and keep[b]]
    return nodes[keep], np.asarray(e, int).reshape(-1, 2)


def cand_mask(z, t, cfg):
    """admission mask over the appended candidates of frame t."""
    nn = int(z[f"nnew{t}"])
    if not cfg.get("cand", False) or nn == 0:
        return np.zeros(nn, bool)
    m = np.ones(nn, bool)
    if cfg.get("fmin") is not None:
        m &= np.asarray(z[f"f{t}"], float).reshape(-1) >= cfg["fmin"]
    return m


def arm_dets(z, T, cfg):
    """(dets, n_ship_seed) for one arm: the shipped peaks plus the admitted candidates appended per frame;
    n_ship_seed[t] = how many of the frame's SEEDS (score >= UNET_THRESH) are shipped (they come first)."""
    dets, n_ship = [], []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        nn = int(z[f"nnew{t}"])
        n0 = len(c) - nn
        m = cand_mask(z, t, cfg)
        cc = np.concatenate([c[:n0], c[n0:][m]]) if nn else c
        ss = np.concatenate([s[:n0], s[n0:][m]]) if nn else s
        dets.append((cc, ss))
        n_ship.append(int((s[:n0] >= UNET_THRESH).sum()))
    return dets, n_ship


def drop_dangling(nodes, edges, son, n_ship):
    """drop every candidate node (seed index >= n_ship[t]) with at most one edge, with its edge."""
    deg = defaultdict(int)
    for a_, b_ in edges:
        deg[a_] += 1
        deg[b_] += 1
    drop = {g for g in nodes if son.get(g) is not None and son[g][1] >= n_ship[son[g][0]] and deg[g] <= 1}
    return ({g: v for g, v in nodes.items() if g not in drop},
            [(a_, b_) for a_, b_ in edges if a_ not in drop and b_ not in drop], len(drop))


def _score(gp, gtg, n_est):
    rr = td.graph.IndexedRXGraph.from_geff(str(gp))
    g = rr[0] if isinstance(rr, tuple) else rr
    er = compute_metric(g, gtg, scale=tuple(SCALE))
    rec = node_recall(g, gtg) if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0
    row = per_sample_metrics(er, float(n_est), rec)
    n_nodes = int(g.num_nodes())
    del g, rr
    return row, n_nodes


def _checkpoint():
    if not done:
        return
    ks = sorted({k for a, _ in ARMS for n in done for _ng, r_ in SROWS[a][n] for k in r_})
    with open("/kaggle/working/zhattach_dense.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("crop,arm,n_gt,n_pred," + ",".join(ks) + "\n")
        for n in done:
            for a, _ in ARMS:
                r_ = ZROWS[a].get(n) or {}
                fh.write(f"{n},{a},{NGT[n]},{CNT[a][n + ':n_pred']}," + ",".join(repr(r_.get(k)) for k in ks) + "\n")
    with open("/kaggle/working/zhattach_sparse.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("crop,arm,draw,n_gt_draw," + ",".join(ks) + "\n")
        for n in done:
            for a, _ in ARMS:
                for d_, (ng_, r_) in enumerate(SROWS[a][n]):
                    fh.write(f"{n},{a},{d_},{ng_}," + ",".join(repr(r_.get(k)) for k in ks) + "\n")


if __name__ == "__main__":
    t_zh = time.time()
    for name in CROPS:
        if time.time() - t_zh > BUDGET_S:
            print(f"  BUDGET reached after {len(done)} crops; the rest are skipped", flush=True)
            break
        t0 = time.time()
        ZH = ZHV
        apply_config(SHIPPED, None)
        z = np.load(ZHV[name])
        T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
        gt_nodes, gt_tid, gt_e = zh_gt_tracks(z, T)
        gp = GT_DIR / f"{name}.geff"
        dr._save(str(gp), gt_nodes, gt_e)
        _r = td.graph.IndexedRXGraph.from_geff(str(gp))
        gtg = _r[0] if isinstance(_r, tuple) else _r
        n_est = int(len(gt_nodes))
        NGT[name] = n_est
        draws = []
        for d_ in range(N_DRAWS):
            sn, se = sparse_draw(gt_nodes, gt_tid, gt_e, FRAC, seed=1000 + d_)
            if len(sn) == 0 or len(se) == 0:
                continue                                   # a draw that kept no track (tiny crop): nothing to score
            gpd = GT_DIR / f"{name}_d{d_}.geff"
            dr._save(str(gpd), sn, se)
            _rd = td.graph.IndexedRXGraph.from_geff(str(gpd))
            draws.append((len(sn), _rd[0] if isinstance(_rd, tuple) else _rd))
        n_new_total = sum(int(z[f"nnew{t}"]) for t in range(T))
        line = []
        for arm, cfg in ARMS:
            apply_config(SHIPPED, None)
            dets, n_ship = arm_dets(z, T, cfg)
            n_adm = sum(len(c) for c, _s in dets) - sum(len(np.asarray(z[f"c{t}"]).reshape(-1, 3)) - int(z[f"nnew{t}"]) for t in range(T))
            SOLVE_STAT.clear()
            nodes0, edges0, pool, son = solve_one(dets, T, n_ship=n_ship, attach_only=cfg.get("attach", False))
            n_solved_new = sum(1 for g in nodes0 if son.get(g) is not None and son[g][1] >= n_ship[son[g][0]])
            n_dropped = 0
            if cfg.get("dangle", False):
                nodes0, edges0, n_dropped = drop_dangling(nodes0, edges0, son, n_ship)
            outp = pdir(arm, "zh") / f"{name}.geff"
            patch_and_save(nodes0, edges0, pool, outp)
            del nodes0, edges0, pool, son, dets
            gc.collect()
            if n_est <= DENSE_MAX:
                row, n_pred = _score(outp, gtg, n_est)
                ZROWS[arm][name] = row
            else:
                rr_ = td.graph.IndexedRXGraph.from_geff(str(outp))
                g_ = rr_[0] if isinstance(rr_, tuple) else rr_
                n_pred = int(g_.num_nodes())
                del g_, rr_
                row = None
            CNT[arm][name + ":n_pred"] = n_pred
            CNT[arm]["admitted"] += n_adm
            CNT[arm]["solved_new"] += n_solved_new
            CNT[arm]["dropped"] += n_dropped
            CNT[arm]["withheld"] += int(SOLVE_STAT.get("withheld", 0))
            CNT[arm]["n_pred"] += n_pred
            srows = []
            for ng_, gtd in draws:
                r_, _n = _score(outp, gtd, n_est)
                srows.append((ng_, r_))
            SROWS[arm][name] = srows
            sp = float(np.mean([_adj_of(r_) for _n, r_ in srows]))
            line.append(f"{arm} adm {n_adm} solved {n_solved_new} drop {n_dropped} pred {n_pred:,} dense {(_adj_of(row) if row is not None else float('nan')):.4f} sparse {sp:.4f}")
            shutil.rmtree(outp, ignore_errors=True)
            gc.collect()
        done.append(name)
        print(f"  [{len(done)}/{len(CROPS)}] {name}: T={T} gt={n_est:,} (draws ~{int(np.mean([d[0] for d in draws]))} nodes) new {n_new_total:,} | "
              + " | ".join(line) + f"  ({time.time() - t0:.0f}s, total {(time.time() - t_zh) / 60:.1f} min)", flush=True)
        for d_ in range(N_DRAWS):
            shutil.rmtree(GT_DIR / f"{name}_d{d_}.geff", ignore_errors=True)
        shutil.rmtree(gp, ignore_errors=True)
        del gtg, _r, draws, z, gt_nodes, gt_tid, gt_e
        _checkpoint()
        gc.collect()

    print(f"\n  {len(done)} of {len(CROPS)} crops scored in {(time.time() - t_zh) / 60:.1f} min", flush=True)
    print(f"\n=== ZHATTACH, {len(done)} Zebrahub crops (out of sample), arms vs REF; N_est = the dense GT count ===", flush=True)
    print(f"  {'arm':<5}{'admitted':>10}{'withheld':>10}{'solved new':>12}{'dropped':>9}{'pred nodes':>12}   DENSE adj_J  edge_J  node_r   d adj_J  d edge_J | SPARSE (mean of {N_DRAWS} draws) adj_J  edge_J  node_r   d adj_J  d edge_J", flush=True)
    dn = [n for n in done if n in ZROWS["REF"]]
    print(f"  dense view over {len(dn)} crops (GT <= {DENSE_MAX:,} nodes); sparse view over all {len(done)}", flush=True)
    ref_d = _with_ratio([ZROWS["REF"][n] for n in dn])
    ref_s = _with_ratio([r_ for n in done for _ng, r_ in SROWS["REF"][n]])
    for arm, _cfg in ARMS:
        dd = _with_ratio([ZROWS[arm][n] for n in dn])
        ss = _with_ratio([r_ for n in done for _ng, r_ in SROWS[arm][n]])
        print(f"  {arm:<5}{CNT[arm]['admitted']:>10,}{CNT[arm]['withheld']:>10,}{CNT[arm]['solved_new']:>12,}{CNT[arm]['dropped']:>9,}{CNT[arm]['n_pred']:>12,}   "
              f"{dd['adj_edge_jaccard']:.4f}  {dd['edge_jaccard']:.4f}  {dd['total_node_ratio']:+.4f}  {dd['adj_edge_jaccard'] - ref_d['adj_edge_jaccard']:+.4f}  {dd['edge_jaccard'] - ref_d['edge_jaccard']:+.4f} | "
              f"{ss['adj_edge_jaccard']:.4f}  {ss['edge_jaccard']:.4f}  {ss['total_node_ratio']:+.4f}  {ss['adj_edge_jaccard'] - ref_s['adj_edge_jaccard']:+.4f}  {ss['edge_jaccard'] - ref_s['edge_jaccard']:+.4f}", flush=True)
    # paired bootstrap over crops of the per-crop delta (sparse: mean over draws) arm - REF
    rng = np.random.RandomState(7)
    print(f"\n  PAIRED BOOTSTRAP over {len(done)} crops, 2,000 draws, per-crop delta arm - REF (sparse view averaged over the {N_DRAWS} GT draws; unweighted crop mean):", flush=True)
    for arm, _cfg in ARMS[1:]:
        for view, key in (("sparse adj_J", "adj_edge_jaccard"), ("sparse edge_J", "edge_jaccard"), ("dense adj_J", "adj_edge_jaccard")):
            if view.startswith("sparse"):
                d = np.array([np.mean([(_adj_of(ra) if key == "adj_edge_jaccard" else ra[key]) - (_adj_of(rr) if key == "adj_edge_jaccard" else rr[key])
                                       for (_na, ra), (_nr, rr) in zip(SROWS[arm][n], SROWS["REF"][n])]) for n in done])
            else:
                d = np.array([_adj_of(ZROWS[arm][n]) - _adj_of(ZROWS["REF"][n]) for n in dn])
            if len(d) == 0:
                continue
            bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)])
            lo, hi = np.percentile(bs, [5, 95])
            print(f"    {arm:<5}{view:<14} mean {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  crops up {int((d > 0).sum())} / down {int((d < 0).sum())}  {'YES' if lo > 0 or hi < 0 else 'no'}", flush=True)
    print("  reading: the sparse view is the metric's; a positive XB / XBF interval here, where the in-sample 6bba bench read +0.0006 [-0.0001, +0.0013],"
          " is the out-of-sample sign. Zebrahub's domain shift exceeds the hidden set's, so the size is an upper bound.", flush=True)
    print(f"\nZHATTACH DONE in {(time.time() - t_zh) / 60:.1f} min", flush=True)
    print("ZHATTACH OK" if len(done) == len(CROPS) else f"ZHATTACH INCOMPLETE {len(done)}/{len(CROPS)}", flush=True)
'''

out_src = head + LOOP
assert out_src.count('if __name__ == "__main__":') == 1 and out_src.count("def drop_dangling(") == 1
assert out_src.index("def solve_one(") < out_src.index("def patch_and_save(") < out_src.index("def zh_gt_tracks(")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "zhattach.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written zhattach.py", len(out_src.splitlines()), "lines")
