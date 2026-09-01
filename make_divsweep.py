"""Generate divsweep.py from f4diag.py by asserted replacement.

THE QUESTION (the user's, 2026-09-02, after divgate2): DIV_DIVERGE_UM is
2.25 and divgate2 measured that ZERO of the real divisions reaching that
test clear it -- zh p90 is 1.94, median 0.80, and the two competition
cases read 0.71 and -0.09. The gate sits above the whole distribution.
So what does the SCORE do when it comes down?

WHY IT NEEDS A RUN RATHER THAN ARITHMETIC. Lowering the gate admits true
forks and false ones together. The patch already emits 456 / 861 / 128
forks per domain with 2 / 0 / 20 of them true, so the marginal fork is
almost certainly false -- but the official division metric only charges
a false fork that lands on annotated cells (reference.md: a fork lying
entirely among unannotated cells enters none of the FP sets), which is
why the record says 26 false divisions on 6bba and not 456. Whether the
net is positive is a scored question, not a countable one.

POST-LINK ONLY, SO ONE SOLVE SERVES EVERY ARM. patch_and_save already
works on a fresh copy of the solved graph, so the ILP runs once per
dataset and each arm replays the four post-link stages. All arms share
one detection set, one ranker and one solution, so the 44b6 in-sample
optimism is identical in every arm and cancels in the delta -- the same
argument that made gap3 a three-domain run and thr5 a two-domain one.

ARMS: DIV_DIVERGE_UM 2.25 (shipped), 1.50, 1.00, 0.50, 0.00.
ANCHOR: the 2.25 arm must reproduce F4-6bba adj_J 0.9216 / div_J 0.0606,
F4-44b6 0.8353 / 0.0000, F4-zh 0.8118 / 0.0215 (divwhy, 2026-09-02).

Reports per arm and domain: score, adj_J, div_J, nodes, edges, and the
number of forks _add_safe_divisions actually added, so the true and
false sides can be read against each other rather than assumed.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "f4diag.py"
DST = pathlib.Path(__file__).parent / "divsweep.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""edgediag -- where do the ~7 missing points of edge_J actually go?',
    '"""divsweep -- what does the score do when the fork divergence gate '
    'comes down?')
rep('SCRIPT_VERSION = "2026-09-01a-f4diag"',
    'SCRIPT_VERSION = "2026-09-02a-divsweep"')

# ---- 1. the arms and a patch that reports how many forks it added -------
rep('''MATCH_R = 7.0   # the scorer's constant, NOT the 5.0 the old diagnostics used''',
    r'''MATCH_R = 7.0   # the scorer's constant, NOT the 5.0 the old diagnostics used

# ---- the sweep ----------------------------------------------------------
DIVERGE_ARMS = [2.25, 1.50, 1.00, 0.50, 0.00]
FORKS = {}          # (arm, domain) -> forks added by _add_safe_divisions


def patch_arm(nodes0, edges0, pool, out_path, diverge):
    """patch_and_save with DIV_DIVERGE_UM set for this arm, returning the
    fork count. Everything upstream of the patch is shared, so the arms
    differ in exactly one constant."""
    global DIV_DIVERGE_UM
    keep = DIV_DIVERGE_UM
    DIV_DIVERGE_UM = diverge
    try:
        cand, cand_sc, cand_trees = pool
        nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
                 for g, v in nodes0.items()}
        edges = list(edges0)
        nid2 = len(nodes) + 1
        nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
        nodes, edges = _short_filter(nodes, edges)
        _linefit(nodes, edges)
        added = _add_safe_divisions(nodes, edges)
        ids = sorted(nodes)
        rm = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
        dr._save(str(out_path), arr, e3)
        return int(added), len(nodes), len(edges)
    finally:
        DIV_DIVERGE_UM = keep''')

# ---- 2. the competition loop, one solve per dataset, five replays -------
rep('''# ---- two arms x the two competition domains -----------------------------
for _arm in ("BASE", "F4"):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    domain = f"{_arm}-{_dm}"
    td0 = time.time()
    outd = pdir("DIAG", domain)
    for name in names:
        dets = load_dets(name, _arm)
        T = len(dets)
        gt = read_geff(geff_path(root, "train", name))
        gt_t = np.asarray(gt["t"], int)
        gt_pos = np.stack([np.asarray(gt[c], float)
                           for c in ("z", "y", "x")], axis=1)
        row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
        gt_e = [(row_of[int(a_)], row_of[int(b_)])
                for a_, b_ in gt["edges"]
                if int(a_) in row_of and int(b_) in row_of]
        diag_one(domain, name, dets, T, gt_t, gt_pos,
                 np.asarray(gt_e, int), outd / f"{name}.geff")
        gc.collect()
    res[domain], PER2[domain] = score_dir(outd)
    print(f"  {domain}: {len(names)} datasets ({time.time() - td0:.0f}s)",
          flush=True)''',
    '''# ---- F4 x the two competition domains, five post-link arms --------------
NODES_N = {}
EDGES_N = {}
for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    domain = f"F4-{_dm}"
    td0 = time.time()
    for name in names:
        dets = load_dets(name, "F4")
        T = len(dets)
        nodes0, edges0, pool = solve_one(dets, T)
        for dv in DIVERGE_ARMS:
            outd = pdir(f"DV{dv:.2f}", domain)
            nf, nn_, ne_ = patch_arm(nodes0, edges0, pool,
                                     outd / f"{name}.geff", dv)
            k = (dv, domain)
            FORKS[k] = FORKS.get(k, 0) + nf
            NODES_N[k] = NODES_N.get(k, 0) + nn_
            EDGES_N[k] = EDGES_N.get(k, 0) + ne_
        del dets, nodes0, edges0, pool
        gc.collect()
    for dv in DIVERGE_ARMS:
        res[(dv, domain)], PER2[(dv, domain)] = \\
            score_dir(pdir(f"DV{dv:.2f}", domain))
    print(f"  {domain}: {len(names)} datasets x {len(DIVERGE_ARMS)} arms "
          f"({time.time() - td0:.0f}s)", flush=True)''')

# ---- 3. the zebrahub loop, same shape -----------------------------------
rep('''# ---- zebrahub: dropped, the question named 6bba and 44b6 ----------------
if False:
    tz = time.time()
    GT_DIR = pathlib.Path("/kaggle/working/ps_zh_gt")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    zrows = []
    outd = pdir("DIAG", "zh")
    for name in SET_ZH:
        dets, T, gt_arr, gt_e, n_cross = zh_load(name)
        diag_one("zh", name, dets, T, gt_arr[:, 0].astype(int),
                 gt_arr[:, 1:], np.asarray(gt_e, int),
                 outd / f"{name}.geff")
        gp = GT_DIR / f"{name}.geff"
        dr._save(str(gp), gt_arr, gt_e)
        r_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        gtg = r_[0] if isinstance(r_, tuple) else r_
        rr = td.graph.IndexedRXGraph.from_geff(str(outd / f"{name}.geff"))
        g = rr[0] if isinstance(rr, tuple) else rr
        er = compute_metric(g, gtg, scale=tuple(SCALE))
        rec = (node_recall(g, gtg)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        zrows.append(per_sample_metrics(er, float(len(gt_arr)), rec))
        del dets
        gc.collect()
    res["zh"] = summarise(zrows)
    PER2["zh"] = None
    _ZROWS = zrows
    print(f"  zebrahub: {len(SET_ZH)} crops ({time.time() - tz:.0f}s)",
          flush=True)
else:
    _ZROWS = []''',
    '''# ---- zebrahub: IN. It is the only dense-GT domain, so it is the only
# arm where a false fork is really false. -------------------------------
_ZROWS = []
if SET_ZH:
    tz = time.time()
    GT_DIR = pathlib.Path("/kaggle/working/ps_zh_gt")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    zrows = {dv: [] for dv in DIVERGE_ARMS}
    for name in SET_ZH:
        _db, T, gt_arr, gt_e, n_cross = zh_load(name)
        _z4 = np.load(FDET_F4[name])
        dets = [(_z4[f"c{t}"], _z4[f"s{t}"]) for t in range(T)]
        del _db
        gp = GT_DIR / f"{name}.geff"
        dr._save(str(gp), gt_arr, gt_e)
        r_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        gtg = r_[0] if isinstance(r_, tuple) else r_
        nodes0, edges0, pool = solve_one(dets, T)
        for dv in DIVERGE_ARMS:
            outd = pdir(f"DV{dv:.2f}", "F4-zh")
            op = outd / f"{name}.geff"
            nf, nn_, ne_ = patch_arm(nodes0, edges0, pool, op, dv)
            k = (dv, "F4-zh")
            FORKS[k] = FORKS.get(k, 0) + nf
            NODES_N[k] = NODES_N.get(k, 0) + nn_
            EDGES_N[k] = EDGES_N.get(k, 0) + ne_
            rr = td.graph.IndexedRXGraph.from_geff(str(op))
            g = rr[0] if isinstance(rr, tuple) else rr
            er = compute_metric(g, gtg, scale=tuple(SCALE))
            rec = (node_recall(g, gtg)
                   if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
            zrows[dv].append(per_sample_metrics(er, float(len(gt_arr)),
                                                rec))
        del dets, nodes0, edges0, pool
        gc.collect()
    for dv in DIVERGE_ARMS:
        res[(dv, "F4-zh")] = summarise(zrows[dv])
        PER2[(dv, "F4-zh")] = None
    _ZROWS = zrows[DIVERGE_ARMS[0]]
    print(f"  zebrahub: {len(SET_ZH)} crops x {len(DIVERGE_ARMS)} arms "
          f"({time.time() - tz:.0f}s)", flush=True)''')

# ---- 4. the report ------------------------------------------------------
_old_report_start = text.index("# ---- report ----")
text = text[:_old_report_start] + r'''# ---- report -------------------------------------------------------------
SWEEP_DOMAINS = ["F4-6bba", "F4-44b6", "F4-zh"]
BASE_DV = DIVERGE_ARMS[0]

print("\n=== ANCHOR / official scorer ===", flush=True)
print(f"  The DIV_DIVERGE_UM {BASE_DV} arm is the shipped pipeline and "
      "must reproduce divwhy's F4 rows: 6bba adj_J 0.9216 div_J 0.0606, "
      "44b6 0.8353 / 0.0000, zh 0.8118 / 0.0215. Every arm shares one "
      "detection set, one ranker and one ILP solution per dataset, so "
      "they differ in exactly one post-link constant.", flush=True)
for d in SWEEP_DOMAINS:
    r = res.get((BASE_DV, d))
    if not r:
        continue
    print(f"  {d:<8} score {r['score']:.4f}  adj_J "
          f"{r['adj_edge_jaccard']:.4f}  div_J "
          f"{r['division_jaccard']:.4f}", flush=True)

print("\n=== THE SWEEP ===", flush=True)
print("  forks = how many second daughters _add_safe_divisions actually "
      "added. d_div_J and d_adj_J are against the shipped arm, so a "
      "positive d_score is the whole case for changing the constant.",
      flush=True)
for d in SWEEP_DOMAINS:
    b = res.get((BASE_DV, d))
    if not b:
        continue
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'DIV_DIVERGE_UM':<15} {'forks':>8} {'nodes':>10} "
          f"{'edges':>10} {'adj_J':>8} {'div_J':>8} {'score':>8} "
          f"{'d_adj_J':>9} {'d_div_J':>9} {'d_score':>9}", flush=True)
    for dv in DIVERGE_ARMS:
        r = res.get((dv, d))
        if not r:
            continue
        print(f"  {dv:<15.2f} {FORKS.get((dv, d), 0):>8,} "
              f"{NODES_N.get((dv, d), 0):>10,} "
              f"{EDGES_N.get((dv, d), 0):>10,} "
              f"{r['adj_edge_jaccard']:>8.4f} "
              f"{r['division_jaccard']:>8.4f} {r['score']:>8.4f} "
              f"{r['adj_edge_jaccard'] - b['adj_edge_jaccard']:>+9.4f} "
              f"{r['division_jaccard'] - b['division_jaccard']:>+9.4f} "
              f"{r['score'] - b['score']:>+9.4f}", flush=True)

print("\n=== THE THREE-DOMAIN READING ===", flush=True)
print(f"  {'DIV_DIVERGE_UM':<15} " + " ".join(
    f"{d:>22}" for d in SWEEP_DOMAINS), flush=True)
for dv in DIVERGE_ARMS:
    cells = []
    for d in SWEEP_DOMAINS:
        r, b = res.get((dv, d)), res.get((BASE_DV, d))
        if not r or not b:
            cells.append(f"{'-':>22}")
            continue
        cells.append(f"{r['score'] - b['score']:>+11.4f} score"
                     f"{'':>4}")
    print(f"  {dv:<15.2f} " + " ".join(cells), flush=True)
print("  A value only passes if all three deltas are >= 0 and the "
      "weakest is above the ~0.002-0.003 holdout noise. 44b6 is D187 "
      "in-sample, but every arm shares that solve so the optimism is "
      "identical across arms and cancels in the delta.", flush=True)

print("\n=== reading it ===", flush=True)
print("  divgate2 measured that the shipped 2.25 rejects 100% of the "
      "real divisions that reach it (zh p90 1.94, median 0.80). If the "
      "sweep is flat, the false forks the lower gate admits cost exactly "
      "what the true ones buy and the constant is not a lever. If div_J "
      "rises while adj_J falls, the two terms are trading and the 0.1 "
      "weight decides. Only a rise in score with no domain dissenting is "
      "a change worth a submission slot.", flush=True)
print(f"\nDIVSWEEP DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)
'''

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
