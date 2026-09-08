"""gap2_rank.py -> ilpsweep.py: the same harness (features, +BIDIR / D187 linkers,
patch_final, official scorer, three domains, hard anchor), with the ILP's
disappearance cost and candidate floor as the swept variable instead of
the 3-frame bridge. One solve per arm per dataset."""
src = open("gap2_rank.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker):
    i = src.index(marker)
    return src.rfind(NL, 0, i) + 1


rep('SCRIPT_VERSION = "2026-09-04a-gap2"', 'SCRIPT_VERSION = "2026-09-05a-ilpsweep"')

rep('def solve_one(dets, T, seeds, feats, model, cols):',
    'def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None):')
rep('            if pr >= ILP_FLOOR:\n                cand_edges.append(',
    '            if pr >= (ILP_FLOOR if floor is None else floor):\n                cand_edges.append(')
rep('''    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=DIV_W)''',
    '''    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=(ILP_DISAPP if disapp is None else disapp),
            division_weight=DIV_W)''')

rep('PATCH_ARMS = ["REF", "GAP2_NB", "GAP2_POOL", "GAP2_NOCTX"]',
    '''PATCH_ARMS = ["REF", "D10", "D20", "F48", "D20F48"]   # (disappearance, ILP floor)
ILP_PARAMS = {"REF": (1.5, 0.10), "D10": (1.0, 0.10), "D20": (2.0, 0.10),
              "F48": (1.5, 0.48), "D20F48": (2.0, 0.48)}   # D20F48 = the 0.948 notebook's ILP
ILP_COUNTS = defaultdict(list)     # (arm, domain) -> [(name, nodes out of ILP, edges out of ILP)]
ILP_RAW = defaultdict(list)        # domain -> [(name, F4 detections fed in)]''')
rep('ARM_TAG = {"REF": "ref", "GAP2_NB": "g2nb", "GAP2_POOL": "g2pool", "GAP2_NOCTX": "g2noctx"}',
    'ARM_TAG = {"REF": "ref", "D10": "d10", "D20": "d20", "F48": "f48", "D20F48": "d20f48"}')

rep('''        nodes0, edges0, pool = solve_one(dets, T, seeds, feats, model, cols)
        cells = []
        for arm in PATCH_ARMS:
            outp = pdir(arm, domain) / f"{name}.geff"
            stat = GAP2_STATS[(arm, domain)]
            patch_final(nodes0, edges0, pool, outp, gap2=GAP2_MODE[arm], stat=stat)
''',
    '''        ILP_RAW[domain].append((name, int(sum(len(c_) for c_, _ in seeds))))
        cells = []
        for arm in PATCH_ARMS:
            dis_, flo_ = ILP_PARAMS[arm]
            nodes0, edges0, pool = solve_one(dets, T, seeds, feats, model, cols,
                                             disapp=dis_, floor=flo_)
            ILP_COUNTS[(arm, domain)].append((name, len(nodes0), len(edges0)))
            outp = pdir(arm, domain) / f"{name}.geff"
            patch_final(nodes0, edges0, pool, outp)
''')

# bookkeeping block -> the ILP funnel + TP/FP/FN
start = line_start('=== GAP2 bookkeeping per variant and domain ===')
end = line_start('=== PAIRED per-dataset adj_J delta vs REF ===')
new_block = '''print()
print("=== THE ILP'S REJECTION per variant and domain: F4 detections in, nodes / edges "
      "out of the ILP (before patches), and the official scorer's edge TP / FP / FN of "
      "the finished graph ===", flush=True)
print(f"  {'variant':<8} {'domain':<9} {'raw_dets':>9} {'ilp_nodes':>10} {'vs_raw':>7} "
      f"{'ilp_edges':>10} {'edge_tp':>8} {'edge_fp':>8} {'edge_fn':>8} "
      f"{'d_tp':>6} {'d_fp':>6} {'d_fn':>6}", flush=True)
for arm in PATCH_ARMS:
    for d in DOMAINS:
        c = ILP_COUNTS.get((arm, d), [])
        if not c:
            continue
        raw = sum(x[1] for x in ILP_RAW[d])
        n_, e_ = sum(x[1] for x in c), sum(x[2] for x in c)
        rows_, ref_ = ROWS[(arm, d)], ROWS[("REF", d)]
        tp = sum(r["edge_tp"] for r in rows_)
        fp = sum(r["edge_fp"] for r in rows_)
        fn = sum(r["edge_fn"] for r in rows_)
        tpr = sum(r["edge_tp"] for r in ref_)
        fpr = sum(r["edge_fp"] for r in ref_)
        fnr = sum(r["edge_fn"] for r in ref_)
        print(f"  {arm:<8} {d:<9} {raw:>9} {n_:>10} {n_ / raw - 1:+7.3f} {e_:>10} "
              f"{tp:>8} {fp:>8} {fn:>8} {tp - tpr:+6} {fp - fpr:+6} {fn - fnr:+6}",
              flush=True)
print("  (the 0.948 notebook's ILP on the visible 4: -15.1% of its raw detections, "
      "-39% on 44b6_0b24845f; ours on official nodes -8.0%)", flush=True)

'''
src = src[:start] + new_block + src[end:]

# reading block
start = line_start('=== reading it ===')
end = src.index('warning_summary()', start)
src = src[:start] + '''print()
print("=== reading it ===", flush=True)
print("  The question (2026-09-05): the 0.948 notebook's ILP (appearance 0, disappearance 2, "
      "candidate floor 0.48) drops 15% of its raw detections on the visible 4 before any "
      "post-processing, and its file is 24% smaller than ours at +0.003 LB. Ours keeps a "
      "chain when its edge probabilities sum above 1.5 (floor 0.10). D20 / F48 / D20F48 "
      "price its rejection on OUR nodes and linker; D10 prices the opposite direction. "
      "Read d_fn (correct edges lost) against d_fp (wrong edges shed) and node_r: the "
      "scorer only counts edges touching an annotated cell, and the node term is signed "
      "with no upper clamp. Positive in 6bba AND 44b6 = a submission candidate; the zh "
      "column is the out-of-sample check.", flush=True)

''' + src[end:]

rep('GAP2 DONE in', 'ILPSWEEP DONE in')
rep('print("GAP2 OK", flush=True)', 'print("ILPSWEEP OK", flush=True)')
rep('[degraded] GAP2 INCOMPLETE:', '[degraded] ILPSWEEP INCOMPLETE:')
open("ilpsweep.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
