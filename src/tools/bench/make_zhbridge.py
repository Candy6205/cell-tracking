"""zhpara4.py -> zhbridge.py: bridge-pair labels from Zebrahub's dense GT.

WHY. bridgerank (2026-09-11) could not train a selector for 2/3-frame
bridges on competition data: 6bba annotates 9.7% of its cells, so only 88
(end, start) pairs in 128 movies carried a positive label against 186,571
proposals. Zebrahub annotates every cell. This kernel runs the same
collection on the Zebrahub crops and writes the labelled pairs so
bridgerank v2 can train on them (plus the 6bba rows) and still apply the
model to 6bba out of fold.

WHAT IT DOES, per crop with an hpara cache (`ZH_HP`, the placement that
ships): `zh_load` -> `solve_one` (zhpara4's solver: the D187 18-column
ranker, no edge head; disappearance 1.5, DIV_W 0.55, and ILP_FLOOR raised
from zhpara4's 0.10 to the production 0.48 so the fragments the bridges
join are produced the way the shipped 6bba solve produces them) -> the
shipped span-1 `_gap_close` (GAP_NEED_CAND False, cap 0.02) with the
bridgerank collect hook -> every span-2 and span-3 (end, start) pair
within GAP1_GATE_UM per frame, 28 graph features, label from the dense GT
at MATCH_UM 5.0 (start's GT node is a descendant of the end's GT node at
exactly step generations). No short filter, no divisions, no geff, no
compute_metric -- the OOM kill zhpara3/4 hit lives in the metric and this
kernel never calls it.

The feature and label code is taken VERBATIM from the generated
bridgerank.py (the `_br_*` block), so the two kernels cannot drift.

OUTPUT: bridge_pairs_zh.csv (crop, span, label, the 28 features), written
after every crop. Attach: zhpara4's sources verbatim. CPU, ~1.5 h.
"""
import os

src = open("zhpara4.py", encoding="utf-8").read()
br = open("bridgerank.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


# the shared block: from the first _br_ function to _gap_close, plus its knobs
_i = br.index("def _br_node_score(")
_j = br.index("def _gap_close(", _i)
BR_BLOCK = br[_i:_j]
_k0 = br.index("BR_COLLECT = False")
_k1 = br.index("BR_DIR = pathlib.Path", _k0)
BR_KNOBS = br[_k0:_k1]
assert "BR_NAMES = [" in BR_KNOBS and "_br_collect(" in BR_BLOCK and "_br_cost(" in BR_BLOCK

rep('SCRIPT_VERSION = "2026-09-09c-zhpara4"', 'SCRIPT_VERSION = "2026-09-11b-zhbridge"')
rep('ILP_FLOOR = 0.10                            # candidates fed to the ILP',
    'ILP_FLOOR = 0.48    # zhbridge: the PRODUCTION floor (zhpara4 used 0.10), so the '
    'fragments look like the shipped 6bba solve\'s')

rep('''def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.
''',
    "# ---- bridgerank's bridge-pair machinery, verbatim from bridgerank.py ----\n"
    + BR_KNOBS + "\n\n" + BR_BLOCK
    + '''def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.
''')
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

cut = "NGT = {}\nMOVED = []\nt_zh = time.time()\ndone = []\nfor name in BOTH:"
assert src.count(cut) == 1
head = src[:src.index(cut)]
tail = r'''
# ---- ZHBRIDGE: collect labelled bridge pairs on every hpara crop ---------
NGT = {}             # read by zhpara4's _checkpoint(), which this run never calls
# zhpara4 cut BOTH down to its two tripwire crops plus the three v3 never
# scored (five); this run wants EVERY crop that has both caches.
ALLZH = sorted(set(ZH_REF) & set(ZH_HP))
assert len(ALLZH) >= 38, f"only {len(ALLZH)} crops carry both a refine and an hpara cache"
ZH = ZH_HP
apply_config(SHIPPED, None)          # GAP_NEED_CAND False, cap 0.02, SHORT_MIN 4
GAP_SPAN = 1
assert BR_MODEL is None and GAP_SPAN == 1 and GAP1_CAP_FRAC == 0.02 and not GAP_NEED_CAND
print(f"\n=== ZHBRIDGE: {len(ALLZH)} hpara crops; solve at floor {ILP_FLOOR}, disappearance "
      f"{ILP_DISAPP}, DIV_W {DIV_W}; shipped span-1 gap close; collect span-2/3 pairs with "
      f"dense-GT labels ===", flush=True)
CSV = pathlib.Path("/kaggle/working/bridge_pairs_zh.csv")
fh = open(CSV, "w", encoding="utf-8", newline="\n")
fh.write("crop,span,label," + ",".join(BR_NAMES) + "\n")
TOT = defaultdict(int)
done = []
t_zh = time.time()
for name in ALLZH:
    if time.time() - t_all > 9.0 * 3600:
        print(f"  BUDGET reached after {len(done)} crops; the rest are skipped", flush=True)
        break
    t0 = time.time()
    dets, T, gt_arr, gt_e, n_cross = zh_load(name)
    nodes0, edges0, pool = solve_one(dets, T)
    cand, cand_sc, cand_trees = pool
    # seed index of every ILP node, by the rounded coordinate the solve used
    seeds = [seeds_of(d_) for d_ in dets]
    cd = {}
    for t in range(T):
        for i_, xyz in enumerate(np.asarray(seeds[t][0], float).reshape(-1, 3)):
            cd.setdefault((t, int(round(xyz[0])), int(round(xyz[1])), int(round(xyz[2]))), (t, i_))
    seed_of_node = {}
    for g_, n_ in nodes0.items():
        key = (int(n_["t"]),) + tuple(int(round(v)) for v in n_["xyz"])
        seed_of_node[g_] = cd.get(key)
    n_unmapped = sum(1 for v in seed_of_node.values() if v is None)
    # dense GT: rows (t, z, y, x) in voxels, edges as row pairs (parent, child)
    g_t = np.asarray(gt_arr[:, 0], int)
    g_pos = np.asarray(gt_arr[:, 1:4], float) * SCALE
    kids = defaultdict(list)
    for a_, b_ in np.asarray(gt_e, int).reshape(-1, 2):
        kids[int(a_)].append(int(b_))
    trees = {}
    for tt in np.unique(g_t):
        rows_ = np.where(g_t == tt)[0]
        trees[int(tt)] = (rows_, cKDTree(g_pos[rows_]))
    BR_CTX.clear()
    BR_CTX.update(name=name, fold=-1, seeds=seeds, seed_of_node=seed_of_node, gt=(trees, kids))
    BR_STAT.clear()
    del BR_ROWS[:]
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()} for g, v in nodes0.items()}
    edges = list(edges0)
    BR_COLLECT = True
    _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)
    BR_COLLECT = False
    n_lab = 0
    for _nm, _fo, sp, lab, row in BR_ROWS:
        if lab < 0:
            continue
        n_lab += 1
        fh.write(f"{name},{sp},{lab}," + ",".join(f"{v:.5g}" for v in row) + "\n")
    fh.flush()
    for k_, v_ in BR_STAT.items():
        TOT[k_] += v_
    done.append(name)
    print(f"  [{len(done)}/{len(ALLZH)}] {name}: T={T} nodes={len(nodes0):,} gt={len(gt_arr):,} "
          f"unmapped nodes {n_unmapped} span1 bridges {BR_STAT.get('span1_bridged', 0)} | "
          f"s2 pairs {BR_STAT.get('collect_span2_pairs', 0)} lab {BR_STAT.get('collect_span2_labelled', 0)} "
          f"pos {BR_STAT.get('collect_span2_pos', 0)} | s3 pairs {BR_STAT.get('collect_span3_pairs', 0)} "
          f"lab {BR_STAT.get('collect_span3_labelled', 0)} pos {BR_STAT.get('collect_span3_pos', 0)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del dets, nodes0, edges0, pool, cand, cand_sc, cand_trees, seeds, cd, seed_of_node
    del gt_arr, gt_e, g_t, g_pos, kids, trees, nodes, edges
    del BR_ROWS[:]
    gc.collect()
fh.close()
print(f"\n  {len(done)} of {len(ALLZH)} crops in {(time.time() - t_zh) / 60:.1f} min", flush=True)
for sp in (2, 3):
    print(f"  span {sp}: pairs {TOT.get(f'collect_span{sp}_pairs', 0):,}, labelled "
          f"{TOT.get(f'collect_span{sp}_labelled', 0):,}, positive {TOT.get(f'collect_span{sp}_pos', 0):,}",
          flush=True)
print(f"  {CSV.name} written", flush=True)
warning_summary()
if len(done) == len(ALLZH):
    print("ZHBRIDGE OK", flush=True)
else:
    print(f"ZHBRIDGE INCOMPLETE ({len(done)} of {len(ALLZH)} crops)", flush=True)
'''
out_src = head + tail
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "zhbridge.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written zhbridge.py", len(out_src))
