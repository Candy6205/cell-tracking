"""edgeprob_rank2.py -> divbench.py: the division bench.

The local division sample is 12 (6bba holdout) + 2 (44b6): one true
positive moves div_J by 0.08. This script scores ALL 128 6bba train
movies (125 of the 151 GT divisions) out-of-fold: two +BIDIR rankers,
each trained on the 71 44b6 movies plus the other 64 6bba movies, BASE
(xl) primaries + BASE edgeprob caches (the only detector with caches
for all 199), production ILP (disappearance 1.5, floor 0.48), then the
fork rules replayed as post-link variants on the same solve.
"""
src = open("edgeprob_rank2.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker, start=0):
    i = src.index(marker, start)
    return src.rfind(NL, 0, i) + 1


rep('SCRIPT_VERSION = "2026-09-03b-edgeprob-rank2"', 'SCRIPT_VERSION = "2026-09-05a-divbench"')

# ---- solve_one: disappearance / floor as parameters ----------------------
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

# ---- patch_final: the division stage switchable ------------------------
rep('def patch_final(nodes0, edges0, pool, out_path):',
    'def patch_final(nodes0, edges0, pool, out_path, divs=True):')
rep('''    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)''',
    '''    _linefit(nodes, edges)
    if divs:
        _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)''')

# ---- FEATURES block --------------------------------------------------------
start = line_start('# ---- FEATURES + JOIN')
end = line_start('def _auc(')
features_block = '''# ---- FEATURES + JOIN: all 199 train movies on BASE -----------------------
t_all = time.time()
SIX = sorted(n for n in ALL_TRAIN if n.startswith("6bba_"))
FOUR = sorted(n for n in ALL_TRAIN if n.startswith("44b6_"))
assert len(SIX) == 128 and len(FOUR) == 71, (len(SIX), len(FOUR))
FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name
PROD_FLOOR = 0.48                                     # ref 56030644, LB 0.933
_jobs = [(n, "44b6") for n in FOUR] + [(n, "6bba") for n in SIX]
print(f"\\n=== FEATURES + JOIN: {len(_jobs)} passes, all 199 train movies on "
      f"BASE (xl) primaries + BASE edgeprob; every 6bba movie keeps its full "
      f"features for the solve ===", flush=True)
FEATS = {}                 # name -> (seeds, feats), 6bba only
LAB = {}                   # name -> (X, y) labelled rows, all 199
for k_, (name, role) in enumerate(_jobs):
    t0 = time.time()
    dets = load_dets(name, "BASE")
    T = len(dets)
    seeds, feats = features_ext(name, dets, T, EP_OF["BASE"][name],
                                EP2_OF["BASE"][name])
    X_, y_ = label_rows(name, feats)
    LAB[name] = (X_, y_)
    n_pairs = int(sum(len(p) for p, _r, _P, _C in feats))
    if role == "6bba":
        FEATS[name] = (seeds, feats)
    print(f"[{k_ + 1}/{len(_jobs)}] {name} {role}"
          f"{' fold' + str(FOLD_OF[name]) if role == '6bba' else ''}: T={T} "
          f"pairs={n_pairs} labelled={len(y_)} pos={int(y_.sum())} "
          f"{time.time() - t0:.0f}s", flush=True)
    del dets, X_, y_
    if role != "6bba":
        del seeds, feats
    gc.collect()
print(f"  join: {JOIN['datasets']} passes, {JOIN['pairs']} pairs, 100% "
      f"matched; NaN p_rev {JOIN['nan_p_rev']}; empty frames "
      f"{JOIN['empty_frames']}", flush=True)
print(f"  features in {(time.time() - t_all) / 60:.1f} min", flush=True)


'''
src = src[:start] + features_block + src[end:]

# ---- STAGE A block ---------------------------------------------------------
start = line_start('# ---- STAGE A: rankers on the D175 pool')
end = line_start('# ---- STAGE B: the full pipeline')
stage_a = '''# ---- STAGE A: two out-of-fold +BIDIR rankers ------------------------------
print("\\n=== STAGE A: two out-of-fold +BIDIR rankers (18 + p_fwd p_rev h_fused; "
      "poolsize hyperparameters), each trained on the 71 44b6 movies + the "
      "other 64 6bba movies ===", flush=True)
COLS = ARMS["+BIDIR"]
MODELS_FOLD = {}
for k in (0, 1):
    t0 = time.time()
    train_names = FOUR + [n for n in SIX if FOLD_OF[n] != k]
    held = [n for n in SIX if FOLD_OF[n] == k]
    Xa = np.concatenate([LAB[n][0] for n in train_names]).astype(np.float32)[:, COLS]
    ya = np.concatenate([LAB[n][1] for n in train_names])
    Xh = np.concatenate([LAB[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([LAB[n][1] for n in held])
    perm = np.random.default_rng(0).permutation(len(ya))
    n_val = max(1000, len(ya) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa[ti], ya[ti])
    val_auc = _auc(ya[vi], m.predict_proba(Xa[vi])[:, 1])
    m.fit(Xa, ya)
    held_auc = _auc(yh, m.predict_proba(Xh)[:, 1])
    MODELS_FOLD[k] = m
    print(f"  fold{k}: trained on {len(train_names)} movies / {len(ya)} rows "
          f"({ya.mean():.3f} pos), val AUC {val_auc:.4f}; held-out fold "
          f"({len(held)} 6bba movies, {len(yh)} rows) AUC {held_auc:.4f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    del Xa, ya, Xh, yh
    gc.collect()
del LAB
gc.collect()
MODELS = {}                # arm_model() is kept but unused here


'''
src = src[:start] + stage_a + src[end:]

# ---- STAGE B + report ---------------------------------------------------
start = line_start('=== STAGE B: full pipeline')
stage_b = '''DIV_ARMS = ["REF", "NB", "DIVG1", "DIVG0", "NODIV"]
# (DIV_PARENT_UM, DIV_SISTER_UM, DIV_DIVERGE_UM, patch on)
DIV_PARAMS = {"REF": (12.0, 15.0, 2.25, True),    # shipped
              "NB": (7.0, 12.0, 2.25, True),      # the 0.948 notebook's geometry
              "DIVG1": (12.0, 15.0, 1.0, True),
              "DIVG0": (12.0, 15.0, 0.0, True),
              "NODIV": (12.0, 15.0, 2.25, False)}  # ILP forks only
DIV_DEFAULTS = dict(DIV_PARENT_UM=DIV_PARENT_UM, DIV_SISTER_UM=DIV_SISTER_UM,
                    DIV_DIVERGE_UM=DIV_DIVERGE_UM)
ARM_TAG = {a: a.lower() for a in DIV_ARMS}
DOMAIN = "BASE-6bba-all"
print(f"\\n=== STAGE B: {len(SIX)} 6bba movies, one production solve each "
      f"(out-of-fold +BIDIR, disappearance {ILP_DISAPP}, floor {PROD_FLOOR}), "
      f"{len(DIV_ARMS)} division variants replayed on it ===", flush=True)
print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J "
      "0.9293 (BASE-6bba hard, rank2 v1). Cells: division tp/fp/fn per arm, "
      "then REF adj_J.", flush=True)
print("  " + " ".join(f"{a:>11}" for a in DIV_ARMS) + "   REF adj_J", flush=True)

ROWS = defaultdict(list)
PER = defaultdict(dict)
ANCHOR_ROWS = []
INCOMPLETE = []
tb = time.time()
for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "BASE")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    nodes0, edges0, pool = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                     disapp=ILP_DISAPP, floor=PROD_FLOOR)
    cells = []
    for arm in DIV_ARMS:
        pu, su, dv, on = DIV_PARAMS[arm]
        globals().update(DIV_PARENT_UM=pu, DIV_SISTER_UM=su, DIV_DIVERGE_UM=dv)
        outp = pdir(arm, DOMAIN) / f"{name}.geff"
        patch_final(nodes0, edges0, pool, outp, divs=on)
        globals().update(DIV_DEFAULTS)
        row = score_one(outp, name)
        ROWS[arm].append(row)
        PER[arm][name] = row
        cells.append(f"{row['division_tp']}/{row['division_fp']}/{row['division_fn']}")
    extra = ""
    if name in HOLDOUT_6BBA:
        n_a, e_a, p_a = solve_one(dets, T, seeds, feats, MODEL_D187, MCOLS_D187,
                                  disapp=1.5, floor=0.10)
        outp = pathlib.Path("/kaggle/working/anchor_d187") / f"{name}.geff"
        outp.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_a, e_a, p_a, outp)
        ANCHOR_ROWS.append(score_one(outp, name))
        extra = f"  anchor {_adj_of(ANCHOR_ROWS[-1]):.4f}"
        del n_a, e_a, p_a
    r0 = PER["REF"][name]
    print(f"  {name} fold{k} gt_div={r0['division_tp'] + r0['division_fn']:>2} "
          + " ".join(f"{c:>11}" for c in cells)
          + f"   {_adj_of(r0):.4f}{extra}  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool
    gc.collect()
print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies x "
      f"{len(DIV_ARMS)} variants ({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}",
          flush=True)

# ---- report ----------------------------------------------------------------
print("\\n=== ANCHOR / official scorer ===", flush=True)
ANCHOR_FAIL = []
if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):
    ra = _with_ratio(ANCHOR_ROWS)
    got = float(ra["adj_edge_jaccard"])
    ok = abs(got - 0.9293) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  BASE-6bba(12)  adj_J {got:.4f}  div_J "
          f"{ra['division_jaccard']:.4f}  want 0.9293 HARD -> "
          f"{'OK' if ok else 'MISS'} ({got - 0.9293:+.4f})", flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "BASE-6bba", got, 0.9293))
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "BASE-6bba", float("nan"), 0.9293))
if ANCHOR_FAIL:
    print("  ⚠ HARD ANCHOR MISS -- the tables below describe a pipeline nobody "
          "has scored before; the run ends in error after printing them.",
          flush=True)
else:
    print("  hard anchor reproduced: the harness is the recorded one.", flush=True)

RES = {a: _with_ratio(ROWS[a]) for a in DIV_ARMS if ROWS[a]}


def _divsum(rows):
    tp = sum(r["division_tp"] for r in rows)
    fp = sum(r["division_fp"] for r in rows)
    fn = sum(r["division_fn"] for r in rows)
    return tp, fp, fn, (tp / (tp + fp + fn) if tp + fp + fn else float("nan"))


print("\\n=== PER VARIANT, all 128 6bba movies (official scorer; div_J = pooled "
      "TP/(TP+FP+FN), as the scorer pools it) ===", flush=True)
print(f"  {'variant':<8} {'div_tp':>6} {'div_fp':>6} {'div_fn':>6} {'div_J':>7} "
      f"{'adj_J':>7} {'edge_J':>7} {'node_r':>8} {'score':>7} {'d div_J':>8} "
      f"{'d adj_J':>8} {'d score':>8}", flush=True)
for arm in DIV_ARMS:
    if arm not in RES:
        continue
    r, b = RES[arm], RES["REF"]
    tp, fp, fn, dj = _divsum(ROWS[arm])
    print(f"  {arm:<8} {tp:>6} {fp:>6} {fn:>6} {dj:7.4f} {r['adj_edge_jaccard']:7.4f} "
          f"{r.get('edge_jaccard', float('nan')):7.4f} "
          f"{r.get('total_node_ratio', float('nan')):+8.4f} {r['score']:7.4f} "
          f"{r['division_jaccard'] - b['division_jaccard']:+8.4f} "
          f"{r['adj_edge_jaccard'] - b['adj_edge_jaccard']:+8.4f} "
          f"{r['score'] - b['score']:+8.4f}", flush=True)

print("\\n=== PER FOLD (each fold's ranker never saw its movies) ===", flush=True)
for k in (0, 1):
    for arm in DIV_ARMS:
        rows_k = [PER[arm][n] for n in SIX if FOLD_OF[n] == k and n in PER[arm]]
        if not rows_k:
            continue
        rk = _with_ratio(rows_k)
        tp, fp, fn, dj = _divsum(rows_k)
        print(f"  fold{k} {arm:<8} movies {len(rows_k):>3} div {tp}/{fp}/{fn} "
              f"div_J {dj:.4f} adj_J {rk['adj_edge_jaccard']:.4f} "
              f"score {rk['score']:.4f}", flush=True)

print("\\n=== THE MOVIES WITH GT DIVISIONS: division tp/fp/fn per variant ===",
      flush=True)
print(f"  {'movie':<14} {'gt':>3} " + " ".join(f"{a:>11}" for a in DIV_ARMS),
      flush=True)
n_gt_total = 0
for name in SIX:
    if name not in PER["REF"]:
        continue
    r0 = PER["REF"][name]
    gt = r0["division_tp"] + r0["division_fn"]
    n_gt_total += gt
    if gt == 0:
        continue
    print(f"  {name:<14} {gt:>3} " + " ".join(
        f"{PER[a][name]['division_tp']}/{PER[a][name]['division_fp']}/"
        f"{PER[a][name]['division_fn']:>2}".rjust(11) for a in DIV_ARMS
        if name in PER[a]), flush=True)
print(f"  GT divisions scored in all: {n_gt_total} (the record: 125 in 6bba)",
      flush=True)
fp_nodiv_movies = sum(1 for n in SIX if n in PER["REF"]
                      and PER["REF"][n]["division_tp"] + PER["REF"][n]["division_fn"] == 0
                      and PER["REF"][n]["division_fp"] > 0)
print(f"  movies with NO GT division but REF false forks counted: "
      f"{fp_nodiv_movies}", flush=True)

print("\\n=== PAIRED per-movie score delta vs REF (score = adj_J + 0.1 div_J) ===",
      flush=True)
for arm in DIV_ARMS[1:]:
    common = [n for n in PER["REF"] if n in PER[arm]]
    if not common:
        continue
    dl = np.array([(PER[arm][n]["score"] - PER["REF"][n]["score"]) for n in common])
    up, dn = int((dl > 1e-9).sum()), int((dl < -1e-9).sum())
    print(f"  {arm:<8} group {RES[arm]['score'] - RES['REF']['score']:+.4f}  "
          f"per-movie mean {dl.mean():+.4f} min {dl.min():+.4f} max {dl.max():+.4f}  "
          f"up {up} down {dn} same {len(dl) - up - dn} of {len(dl)}", flush=True)

print("\\n=== reading it ===", flush=True)
print("  The division sample was 12 (one TP = 0.08 div_J). Here it is ~125, "
      "out-of-fold for the ranker, on BASE primaries (the only detector cached "
      "for all 199; competition GT divisions never lose a daughter at "
      "detection, so the fork rules are what is measured). REF = shipped "
      "rules; NB = the 0.948 notebook's tighter geometry (parent 7 um, "
      "sister 12 um); DIVG1 / DIVG0 = the divergence test at 1.0 / 0 um "
      "(divgate2: 0% of real divisions pass 2.25); NODIV = the ILP's own "
      "forks only, i.e. what the patch buys in total. The scorer's div_J "
      "is pooled, so a variant is read by d div_J AND d adj_J together "
      "(score); 44b6 is not here, and the weight of divisions on LB is "
      "0.1 x div_J.", flush=True)
warning_summary()
n_scored = sum(len(PER[a]) for a in DIV_ARMS)
n_expected = len(DIV_ARMS) * len(SIX)
print(f"\\nDIVBENCH DONE in {(time.time() - t_all) / 60:.1f} min (stage B "
      f"{(time.time() - tb) / 60:.1f} min), {n_scored} of {n_expected} "
      f"variant-movies scored", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
if n_scored == n_expected and not INCOMPLETE:
    print("DIVBENCH OK", flush=True)
else:
    print(f"[degraded] DIVBENCH INCOMPLETE: {n_scored}/{n_expected}", flush=True)
'''
src = src[:start] + stage_b
open("divbench.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
