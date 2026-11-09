"""zhpara4.py + forkprop_sz2.py -> zhfork.py: the fork gate's column sets on Zebrahub, where EVERY fork is judged.

WHY (2026-09-25). On 6bba 17,700 of the arm's 18,457 forks are uncounted (the mother sits on an unannotated cell) and the
gate is fitted and read on 725 counted forks, 66 true; every size-column reading so far is inside that resolution.
Zebrahub's crops carry dense GT with parent links (8 crops: 1,081 GT divisions; 41 hpara-cached crops), so a fork's
verdict is known for (nearly) every fork and the gate's true-kept / false-cut can be read on thousands of judged forks.
The user (2026-09-25): a domain shift is expected (size-based scores may not transfer), but with the variables
controlled -- one fork population, gates differing only in their columns -- the image gate can be tested there.

WHAT. zhpara4's Zebrahub pipeline (the D187 18-column ranker, no edge heads -> the head columns are NaN as in a
submission whose head pass failed; ILP floor 0.48, disappearance 1.5, DIV_W 0.55; the shipped gap-close / short filter
/ linefit at SHORT_MIN 4) on the 41 hpara-cached crops, then forkprop's orphan patch as it ships (contrast-ranked,
uncapped, DIV_DIVERGE_UM 1.5; no node-cost solve here), then forkprop_sz2's fork rows: the 28 gate / new features and
the 61 image-side columns (11 image + the half-max size five + sz2's 25) read from the crop's own image, with the fork
nodes' coordinates. Every fork judged against the dense GT at MATCH_UM 5 um: 2 = the mother matches a GT node that
divides and the fork's children cover both GT daughters, 1 = the mother matches a GT node but the fork is not that
division, 0 = the mother matches no GT node. Then, in-kernel: the population's counts, single-column AUCs of the size
columns, and 5-fold-by-crop gates for the column sets (31 / 36 / 33 / one per sz2 measure / ratios / all 61) with TP
kept, FP kept, AUC at p > 0.03. Output zhfork_forks.csv (the table; the offline paired comparisons run from it).
Attach: support pack, cell-tracking, competition, kernels biohub-poolsize (ranker pkl), biohub-zebrahub-extract (crops),
biohub-zhhpcache (hp caches). CPU, ~3 h (41 crops; the 6bba harness takes ~100 s per movie for solve + patch + features).
"""
zp = open("zhpara4.py", encoding="utf-8").read()
fp = open("forkprop_sz2.py", encoding="utf-8").read()


def zslice(a, b):
    i = zp.index(a)
    return zp[i:zp.index(b, i)]


def fslice(a, b=None):
    i = fp.index(a)
    if b is None:                                    # to the next top-level def
        j = fp.index("\ndef ", i + 1) + 1
    else:
        j = fp.index(b, i)
    return fp[i:j]


head = zp[:zp.index("# ---- the three evaluation domains")]
head = head.replace('SCRIPT_VERSION = "2026-09-09c-zhpara4"', 'SCRIPT_VERSION = "2026-09-25a-zhfork"  # zhpara4 was "2026-09-09c-zhpara4"', 1)
head = head.replace('assert len(CACHES) >= 40, "attach biohub-edge-ranker as kernel source"',
                    'print("  (zhfork: the 6bba detection caches are not needed; the competition domains are not run)", flush=True)', 1)
assert head.count("2026-09-25a-zhfork") == 1 and "def solve_one(" not in head and "MODEL = _blob" in head and "assert len(CACHES)" not in head
zh_load = zslice("def zh_load(", "# ---- the sweep")
assert "def zh_load(" in zh_load and "return dets, T, nodes, np.asarray(edges, int), n_cross" in zh_load

FEATS = fslice("FEAT20 = [", "FEAT_NAMES = ") + fslice("FEAT_NAMES = ", "\n") + "\n"
PATCH = fslice("def _add_safe_divisions(")
IMGBLK = fslice("def _fi_crop(", "def fork_rows(")
CHAIN = fslice("def _chain_len(") + fslice("def _aux_extra(") + fslice("def fork_rows(", "def walk_lost(")
assert "def _fi_sizes2(" in IMGBLK and "def fp_contrast(" in IMGBLK and "def fork_img_feats(" in IMGBLK
assert "steal" in PATCH and "fp_contrast" in PATCH and "_aux_extra" in CHAIN and "def fork_rows(" in CHAIN

MAIN = r'''
# ---- zhfork: the shipped composition's constants that the patch and the solve read ---------------------------------
ILP_FLOOR = 0.48                 # zhattach / zhbridge: the production floor
SHORT_MIN = 4
GAP_NEED_CAND = False
GAP1_CAP_FRAC = 0.02
DIV_DIVERGE_UM = 1.5             # forkprop's CR_NOCAP_DV150 arm
FP_RANK = "contrast"
FP_CAP = 100.0
FP_MEMO = {}
FP_STAT = defaultdict(int)
DA_ADMIT = False
DA_MIN_SEP_UM = 3.0
DA_CTX = {}                      # read only when DA_ADMIT (never here)
PATCH_CAPS = defaultdict(int)    # the patch's cap counters
FORK_LOG = {}
STEAL_LOG = {}
PRE_STATE = {}
IMG_CTX = {}
FI_STAT = defaultdict(int)
FI_HALF_UM, FI_PEAK_UM, FI_SIZE_UM = 8.0, 2.0, 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)
BUDGET_S = 10.0 * 3600
MATCH_UM = 5.0
'''

SOLVE = r'''
# ---- the crops ----------------------------------------------------------------------------------------------------
ZH_HP = {f.stem[len("zhdet_hp_"):]: f for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_hp_*.npz")}
assert len(ZH_HP) >= 30, f"attach faustine649/biohub-zhhpcache: {len(ZH_HP)} hpara caches found"
ZH = ZH_HP
CROPS = None
for _c in ["/kaggle/input/biohub-zebrahub-extract/crops", "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(_c).exists():
        CROPS = pathlib.Path(_c)
        break
if CROPS is None:
    _hits = sorted(p_ for p_ in pathlib.Path("/kaggle/input").glob("*/**/crops") if p_.is_dir() and any(p_.glob("*.npz")))
    CROPS = _hits[0] if _hits else None
assert CROPS is not None, "attach faustine649/biohub-zebrahub-extract (crops)"
CROP_OF = {p_.stem: p_ for p_ in CROPS.glob("*.npz")}
NAMES = sorted(n_ for n_ in ZH_HP if n_ in CROP_OF)
print(f"crops with an hpara cache and an image: {len(NAMES)} of {len(ZH_HP)} caches / {len(CROP_OF)} crops", flush=True)
assert len(NAMES) >= 30, NAMES


def solve_zh(dets, T):
    """zhpara4's solve_one with the two lookups forkprop's fork rows need: pmap (t, i, j) -> (p, NaN, NaN) (no edge
    heads here) and seed_of_node (node id -> (t, seed index))."""
    seeds, vels, prev_of = pass1_tracks(dets)
    feats = pair_features(dets, None, T)
    probs = [MODEL.predict_proba(rows[:, MCOLS])[:, 1] if len(pairs) else np.zeros(0) for pairs, rows, P, C in feats]
    coords_rows, offset, n_glob, cd = [], {}, 0, {}
    for t in range(T):
        c, s_ = seeds[t]
        offset[t] = n_glob
        n_glob += len(c)
        for i_, xyz in enumerate(c):
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
            cd[(t,) + tuple(int(round(float(v))) for v in xyz)] = (t, i_)
    coords = np.asarray(np.round(coords_rows), np.int16)
    pmap, cand_edges = {}, []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            pmap[(t, int(i_), int(j_))] = (float(pr), float("nan"), float("nan"))
            if pr >= ILP_FLOOR:
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_, float(pr), 0.0))
    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"), appearance_weight=0.0,
                                      disappearance_weight=ILP_DISAPP, division_weight=DIV_W)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
    na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nid_l = [int(v) for v in na["node_id"].to_list()]
    remap = {v: i for i, v in enumerate(nid_l)}
    nodes_arr = np.stack([np.asarray(na[c_].to_list(), float) for c_ in ("t", "z", "y", "x")], axis=1)
    rows_e = []
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for a_, b_ in zip(ea["source_id"].to_list(), ea["target_id"].to_list()):
            ra, rb = remap.get(int(a_)), remap.get(int(b_))
            if ra is not None and rb is not None:
                rows_e.append([ra, rb])
    e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
    nodes0 = {i_ + 1: {"t": int(nodes_arr[i_, 0]), "xyz": nodes_arr[i_, 1:].astype(float)} for i_ in range(len(nodes_arr))}
    edges0 = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e2]
    seed_of_node = {i_ + 1: cd.get((int(nodes_arr[i_, 0]),) + tuple(int(round(v)) for v in nodes_arr[i_, 1:])) for i_ in range(len(nodes_arr))}
    cand, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1]) else None)
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), pmap, seed_of_node, seeds


def patch_zh(nodes0, edges0, pool):
    """forkprop's patch_final without the geff: gap-close, short filter, linefit, the orphan patch; PRE_STATE filled."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()} for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    PRE_STATE["nodes"] = nodes
    PRE_STATE["edges"] = list(edges)
    FORK_LOG.clear()
    STEAL_LOG.clear()
    n0 = len(edges)
    _add_safe_divisions(nodes, edges)
    PRE_STATE["ids"] = sorted(nodes)
    PRE_STATE["final_edges"] = list(edges)
    return nodes, edges, len(edges) - n0


def judge_forks(nodes, edges, gt_nodes, gt_edges):
    """every fork mother of the final graph against the dense GT: 2 = the mother matches (MATCH_UM) a GT node with two
    children and the fork's children cover both GT daughters; 1 = the mother matches a GT node but not that division;
    0 = no GT node within MATCH_UM of the mother. Returns {mother id: label} and the GT division count."""
    succ = defaultdict(list)
    for a_, b_ in edges:
        succ[a_].append(b_)
    gsucc = defaultdict(list)
    for a_, b_ in gt_edges:
        gsucc[int(a_)].append(int(b_))
    gt_by_t = defaultdict(list)
    for i_, r_ in enumerate(gt_nodes):
        gt_by_t[int(r_[0])].append(i_)
    trees = {t_: (cKDTree(gt_nodes[idx_][:, 1:] * SCALE), np.asarray(idx_)) for t_, idx_ in gt_by_t.items()}
    n_gt_div = sum(1 for g_, ch in gsucc.items() if len(ch) >= 2)
    out = {}
    for p_, ch in succ.items():
        if len(ch) < 2:
            continue
        t_ = int(nodes[p_]["t"])
        tr_ = trees.get(t_)
        if tr_ is None:
            out[p_] = 0
            continue
        d_, j_ = tr_[0].query(np.asarray(nodes[p_]["xyz"], float) * SCALE)
        if d_ > MATCH_UM:
            out[p_] = 0
            continue
        g_ = int(tr_[1][j_])
        gch = gsucc.get(g_, [])
        if len(gch) < 2:
            out[p_] = 1
            continue
        covered = 0
        for gc_ in gch[:2]:
            gx = gt_nodes[gc_][1:] * SCALE
            if any(np.linalg.norm(np.asarray(nodes[c_]["xyz"], float) * SCALE - gx) <= MATCH_UM for c_ in ch):
                covered += 1
        out[p_] = 2 if covered == 2 else 1
    return out, n_gt_div


# ---- the run ------------------------------------------------------------------------------------------------------
t_all = time.time()
OUT = pathlib.Path("/kaggle/working")
COORD_HDR = ",t,z_p,y_p,x_p,z_c1,y_c1,x_c1,z_q,y_q,x_q"
fh = open(OUT / "zhfork_forks.csv", "w", encoding="utf-8", newline="\n")
fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + COORD_HDR + "\n")
ALL = []
TOT = defaultdict(int)
for k_, name in enumerate(NAMES):
    if time.time() - t_all > BUDGET_S:
        print(f"[budget] stopped before {name}: {k_} crops done", flush=True)
        break
    t0 = time.time()
    dets, T, gt_nodes, gt_edges, n_cross = zh_load(name)
    nodes0, edges0, pool, pmap, seed_of_node, seeds = solve_zh(dets, T)
    z_ = np.load(CROP_OF[name])
    IMG_CTX.clear()
    IMG_CTX.update(name=name, img=np.asarray(z_["image"]))
    FP_MEMO.clear()
    FP_STAT.clear()
    nodes, edges, n_added = patch_zh(nodes0, edges0, pool)
    verdict, n_gt_div = judge_forks(nodes, edges, gt_nodes, gt_edges)
    ids = PRE_STATE["ids"]
    pos = {g: i for i, g in enumerate(ids)}
    tp_forks = [pos[g] for g, v in verdict.items() if v == 2]
    fp_forks = [pos[g] for g, v in verdict.items() if v == 1]
    scsort, best2 = _aux_extra(seeds, pmap)
    rows = fork_rows(name, k_ % 5, seeds, (pmap, seed_of_node, scsort, best2), tp_forks, fp_forks)
    for r_ in rows:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5])
                 + ("," + ",".join(f"{v:.6g}" for v in r_[6]) if len(r_) > 6 else "") + "\n")
    fh.flush()
    ALL.extend(rows)
    n2, n1, n0 = sum(1 for r_ in rows if r_[4] == 2), sum(1 for r_ in rows if r_[4] == 1), sum(1 for r_ in rows if r_[4] == 0)
    TOT["gt_div"] += n_gt_div
    TOT["forks"] += len(rows)
    TOT["tp"] += n2
    TOT["fp"] += n1
    TOT["unc"] += n0
    TOT["patch_added"] += n_added
    del z_, dets, nodes0, edges0, pool, pmap, nodes, edges
    IMG_CTX.clear()
    gc.collect()
    print(f"  [{k_ + 1}/{len(NAMES)}] {name}: T={T} GT divisions {n_gt_div} | forks {len(rows)} (patch added {n_added}): "
          f"TP {n2} FP {n1} uncounted {n0} | image {FI_STAT['forks']} forks, contrast crops {FP_STAT['crops']} | {time.time() - t0:.0f}s "
          f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
fh.close()
print(f"\n=== ZEBRAHUB FORK POPULATION: {TOT['forks']:,} forks in {len(NAMES)} crops, judged TP {TOT['tp']:,} / FP {TOT['fp']:,} / "
      f"uncounted {TOT['unc']:,} (6bba: 725 counted of 18,457); GT divisions {TOT['gt_div']:,}, the pipeline forked on "
      f"{TOT['tp']:,} of them ===", flush=True)

# ---- the gate on a judged population: single columns, then the column sets, 5 folds by crop ------------------------
X_f = np.array([r_[5] for r_ in ALL], float) if ALL else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in ALL], int)
fold_f = np.array([r_[1] for r_ in ALL], int)
counted = lab >= 1
if counted.sum() and (lab == 2).sum() >= 20 and (lab == 1).sum() >= 20:
    y_all = (lab == 2).astype(int)
    print("\n  single columns on the counted forks: AUC TP vs FP, TP / FP medians (the size columns; NaN skipped)", flush=True)
    for fn_ in FEAT_IMG:
        col = X_f[:, FEAT_NAMES.index(fn_)]
        m_ = counted & np.isfinite(col)
        if m_.sum() and (lab[m_] == 2).any() and (lab[m_] == 1).any():
            print(f"    {fn_:<11} AUC {roc_auc_score(y_all[m_], col[m_]):.3f}  TP {np.median(col[m_ & (lab == 2)]):.3g}  FP {np.median(col[m_ & (lab == 1)]):.3g}", flush=True)
    J20 = list(range(len(FEAT20)))
    JIMG11 = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG[:11]]
    JSZ = {"hm": [FEAT_NAMES.index(f_) for f_ in ("r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum")]}
    for m_ in SZ2:
        JSZ[m_] = [FEAT_NAMES.index(f"{m_}_{s_}") for s_ in ("p", "c1", "q", "max", "sum")]
    JRAT = JSZ["hm"][3:] + [FEAT_NAMES.index(f"{m_}_{s_}") for m_ in SZ2 for s_ in ("max", "sum")]
    JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]
    SETS = {"img31": J20 + JIMG11, "img36 (size gate)": J20 + JIMG11 + JSZ["hm"], "img33 (ratios)": J20 + JIMG11 + JSZ["hm"][3:],
            **{f"img_{m_}": J20 + JIMG11 + JSZ[m_] for m_ in SZ2}, "img_ratios": J20 + JIMG11 + JRAT, "imgall (61)": J20 + JIMG,
            "gate20": J20}
    print(f"\n=== THE GATE ON A JUDGED POPULATION: {int(counted.sum()):,} counted forks ({int((lab == 2).sum()):,} TP), HistGB "
          f"(100 it, lr 0.05, msl 10) out of fold by crop (5 folds), cut p > 0.03 and at the cut that keeps 90% of the TP ===", flush=True)
    print(f"    {'columns':<20} {'AUC':>6} {'TP kept/TP':>12} {'FP kept/FP':>14} {'FP kept at 90% TP':>18}", flush=True)
    for nm_, cols in SETS.items():
        p_ = np.full(len(lab), np.nan)
        for f_ in sorted(set(fold_f.tolist())):
            tr_ = counted & (fold_f != f_)
            te_ = fold_f == f_
            if tr_.sum() < 50 or (lab[tr_] == 2).sum() < 5:
                continue
            mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
            mg.fit(X_f[tr_][:, cols], (lab[tr_] == 2).astype(int))
            p_[te_] = mg.predict_proba(X_f[te_][:, cols])[:, 1]
        ok_ = counted & np.isfinite(p_)
        if not ok_.sum() or not (lab[ok_] == 2).any() or not (lab[ok_] == 1).any():
            continue
        auc_ = roc_auc_score(y_all[ok_], p_[ok_])
        keep = ok_ & (p_ > 0.03)
        tp_all, fp_all = int((ok_ & (lab == 2)).sum()), int((ok_ & (lab == 1)).sum())
        thr90 = float(np.quantile(p_[ok_ & (lab == 2)], 0.10))
        fp90 = int((ok_ & (lab == 1) & (p_ > thr90)).sum())
        print(f"    {nm_:<20} {auc_:>6.3f} {int((keep & (lab == 2)).sum()):>5}/{tp_all:<6} {int((keep & (lab == 1)).sum()):>6}/{fp_all:<7} {fp90:>10} (p > {thr90:.4f})", flush=True)
    print("    reading: a column set earns a place only if it cuts distinctly more false forks at the same true-fork retention; "
          "the domain shift (zebrafish nuclei, the detector out of sample) makes the absolute size values suspect, the "
          "comparison between column sets on this one population is the controlled read.", flush=True)
else:
    print("  too few judged forks for a gate reading", flush=True)
warning_summary()
print(f"ZHFORK DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("ZHFORK OK", flush=True)
'''
out = head + MAIN + FEATS + PATCH + IMGBLK + CHAIN + zh_load + SOLVE
assert out.count("def _add_safe_divisions(") == 2, out.count("def _add_safe_divisions(")   # zhpara4's, then forkprop's (the later definition wins)
assert out.count("def fork_rows(") == 1 and out.count("def _fi_sizes2(") == 1 and out.count("def zh_load(") == 1
open("zhfork.py", "w", encoding="utf-8", newline="\n").write(out)
print("written zhfork.py", out.count("\n"), "lines")
