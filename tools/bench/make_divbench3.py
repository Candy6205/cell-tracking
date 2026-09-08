"""divbench2.py -> divbench3.py: the division bench on F4 (production
primaries), the fork gate trained out-of-fold on F4 forks, the pruning
applied for real (edges removed, geff rewritten, official scorer), the
fork table and the production gate written to disk, and the 12-movie 44b6
draw checked under the production ranker.

Ranker training rows stay BASE (that is how the production +BIDIR ranker
was trained: D175 pool on BASE primaries, applied to F4 nodes); the solve
uses F4 detections + the head's F4 caches (biohub-fusecache for the 12
holdout, biohub-fuse6bba for the other 116; biohub-edgeprob + biohub-ep6bba).
"""
src = open("divbench2.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker, start=0):
    i = src.index(marker, start)
    return src.rfind(NL, 0, i) + 1


rep('SCRIPT_VERSION = "2026-09-06a-divbench2"', 'SCRIPT_VERSION = "2026-09-06a-divbench3"')

# ---- features_ext: a missing head-2 cache -> NaN columns -----------------
rep('''    K, skey, ex1 = _load_head(name, ep_path, T, ours)
    K2, skey2, ex2 = _load_head(name, ep2_path, T, ours)''',
    '''    K, skey, ex1 = _load_head(name, ep_path, T, ours)
    if ep2_path is None:                    # head 2 not cached for this movie
        K2, skey2, ex2 = K, skey, np.full(ex1.shape, np.nan, np.float32)
    else:
        K2, skey2, ex2 = _load_head(name, ep2_path, T, ours)''')

# ---- FEATURES: label rows from BASE (all 199), solve features from F4 ------
rep('''    if role == "6bba":
        FEATS[name] = (seeds, feats)
    print(f"[{k_ + 1}/{len(_jobs)}] {name} {role}"
          f"{' fold' + str(FOLD_OF[name]) if role == '6bba' else ''}: T={T} "
          f"pairs={n_pairs} labelled={len(y_)} pos={int(y_.sum())} "
          f"{time.time() - t0:.0f}s", flush=True)
    del dets, X_, y_
    if role != "6bba":
        del seeds, feats
    gc.collect()''',
    '''    n4 = ""
    if role == "6bba" or name in SET_44B6:
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        FEATS[name] = (seeds4, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4
    print(f"[{k_ + 1}/{len(_jobs)}] {name} {role}"
          f"{' fold' + str(FOLD_OF[name]) if role == '6bba' else ''}: T={T} "
          f"pairs={n_pairs} labelled={len(y_)} pos={int(y_.sum())}{n4} "
          f"{time.time() - t0:.0f}s", flush=True)
    del dets, X_, y_, seeds, feats
    gc.collect()''')
rep('''FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name''',
    '''FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name
_missing = [n for n in SIX if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"
_missing = [n for n in SIX if n not in EDGEPROB_F4]
assert not _missing, f"no edgeprob F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"
MODEL_BIDIR, MCOLS_BIDIR, POOL_BIDIR = _load_pkl("edge_ranker_ep175v3_bidir.pkl", 170, 180)
assert not (set(POOL_BIDIR) & set(SET_44B6)), "the production ranker saw the 44b6 draw"''')

# ---- STAGE B + report, rewritten -------------------------------------------
start = line_start('print(f"\\n=== STAGE B: {len(SIX)} 6bba movies, one production solve each "')
stage_b = r'''print(f"\n=== STAGE B: {len(SIX)} 6bba movies on F4 primaries, one production solve "
      f"each (out-of-fold +BIDIR, disappearance {ILP_DISAPP}, floor {PROD_FLOOR}), "
      f"REF patch, every fork logged and judged, every lost division walked ===",
      flush=True)
print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J 0.9216 "
      "(F4-6bba hard, div_J 0.0606). Soft: the production +BIDIR pkl at floor 0.48 "
      "on the same 12 = ilpsweep's F48 row 0.9261.", flush=True)

ROWS = []
PER = {}
ANCHOR_ROWS = []
SOFT_ROWS = []
INCOMPLETE = []
FORKS = []             # (name, fold, p, q, label, feats)
MOVIE = {}             # name -> dict(arr, ids, edges, rm)
WALK = defaultdict(int)
WALK_FOLD = {0: defaultdict(int), 1: defaultdict(int)}
tb = time.time()


def _snapshot():
    nodes, edges, ids = PRE_STATE["nodes"], PRE_STATE["final_edges"], PRE_STATE["ids"]
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    return dict(arr=arr, ids=ids, edges=list(edges), rm={g: i for i, g in enumerate(ids)})


def _rescore(name, snap, drop, tag):
    """Write the final graph minus the edges in `drop`, score it."""
    rm = snap["rm"]
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in snap["edges"] if (a_, b_) not in drop], int)
    outp = pathlib.Path(f"/kaggle/working/prune_{tag}") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    dr._save(str(outp), snap["arr"], e3)
    return score_one(outp, name)


for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    nodes0, edges0, pool, aux = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                          disapp=ILP_DISAPP, floor=PROD_FLOOR)
    outp = pdir("REF", DOMAIN) / f"{name}.geff"
    patch_final(nodes0, edges0, pool, outp)
    row = score_one(outp, name)
    ROWS.append(row)
    PER[name] = row
    MOVIE[name] = _snapshot()
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(outp))
    g = res_[0] if isinstance(res_, tuple) else res_
    ds = score_divisions(g, dset.tracks, scale=dset.scale)
    if (len(ds.tp_forks), len(ds.fp_forks)) != (row["division_tp"], row["division_fp"]):
        warn_once("fork_count_mismatch",
                  f"{name}: score_divisions tp/fp {len(ds.tp_forks)}/{len(ds.fp_forks)} "
                  f"vs per-sample {row['division_tp']}/{row['division_fp']}")
    fr = fork_rows(name, k, seeds, aux, ds.tp_forks, ds.fp_forks)
    FORKS.extend(fr)
    scores = {int(kk): int(vv) for kk, vv in ds.scores.items()}
    w, det = walk_lost(name, scores)
    for c_, n_ in w.items():
        WALK[c_] += n_
        WALK_FOLD[k][c_] += n_
    extra = ""
    if name in HOLDOUT_6BBA:
        n_a, e_a, p_a, _ = solve_one(dets, T, seeds, feats, MODEL_D187, MCOLS_D187,
                                     disapp=1.5, floor=0.10)
        outa = pathlib.Path("/kaggle/working/anchor_d187") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_a, e_a, p_a, outa)
        ANCHOR_ROWS.append(score_one(outa, name))
        n_s, e_s, p_s, _ = solve_one(dets, T, seeds, feats, MODEL_BIDIR, MCOLS_BIDIR,
                                     disapp=ILP_DISAPP, floor=PROD_FLOOR)
        outs = pathlib.Path("/kaggle/working/soft_bidir") / f"{name}.geff"
        outs.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_s, e_s, p_s, outs)
        SOFT_ROWS.append(score_one(outs, name))
        extra = f"  anchor {_adj_of(ANCHOR_ROWS[-1]):.4f} soft {_adj_of(SOFT_ROWS[-1]):.4f}"
        del n_a, e_a, p_a, n_s, e_s, p_s
    n_patch = sum(1 for r_ in fr if r_[5][0] == 1)
    print(f"  {name} fold{k} gt_div={row['division_tp'] + row['division_fn']:>2} "
          f"div {row['division_tp']}/{row['division_fp']}/{row['division_fn']:<3} "
          f"forks {len(fr):>4} ({n_patch} patch) "
          f"lost: {', '.join(f'{c_} {n_}' for c_, n_ in sorted(w.items()) if c_ != 'recovered') or '-'}"
          f"   {_adj_of(row):.4f}{extra}  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool, aux, g, res_, dset, ds
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
print("  hard anchor reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  ⚠ HARD ANCHOR MISS", flush=True)

RES = _with_ratio(ROWS)
TP = sum(r["division_tp"] for r in ROWS)
FP = sum(r["division_fp"] for r in ROWS)
FN = sum(r["division_fn"] for r in ROWS)
N_GT = TP + FN
base_J = TP / max(1, TP + FP + FN)
print(f"\n=== REF on {len(ROWS)} movies, F4: div {TP}/{FP}/{FN} div_J {base_J:.4f} "
      f"adj_J {RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f} "
      f"(BASE bench: 54/258/71, 0.1410, 0.9136, 0.9277) ===", flush=True)
for k in (0, 1):
    rows_k = [PER[n] for n in SIX if FOLD_OF[n] == k and n in PER]
    rk = _with_ratio(rows_k)
    print(f"  fold{k}: div {sum(r['division_tp'] for r in rows_k)}/"
          f"{sum(r['division_fp'] for r in rows_k)}/{sum(r['division_fn'] for r in rows_k)} "
          f"div_J {rk['division_jaccard']:.4f} adj_J {rk['adj_edge_jaccard']:.4f} "
          f"score {rk['score']:.4f}", flush=True)

print("\n=== WHERE THE LOST DIVISIONS GO (F4; BASE bench in brackets) ===", flush=True)
BASE_WALK = {"mother_absent": 7, "daughter_absent": 26, "1a_mother_no_child": 2,
             "2_mother_no_single_pred": 2, "5a_sister_not_orphan": 19,
             "5c_sister_over_12um_of_mother": 3, "7_not_the_nearest_orphan": 1,
             "8b_diverge_under_2.25um": 10, "9_passed_all_gates_then_capped_or_taken": 1,
             "recovered": 54}
print(f"  {'condition':<42} {'all':>5} {'fold0':>6} {'fold1':>6} {'BASE':>6}", flush=True)
for c_ in WALK_ORDER:
    if WALK.get(c_, 0) or BASE_WALK.get(c_, 0):
        print(f"  {c_:<42} {WALK.get(c_, 0):>5} {WALK_FOLD[0].get(c_, 0):>6} "
              f"{WALK_FOLD[1].get(c_, 0):>6} {BASE_WALK.get(c_, 0):>6}", flush=True)

# ---- the forks ---------------------------------------------------------------
X_f = np.array([r_[5] for r_ in FORKS], float) if FORKS else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in FORKS], int)
fold_f = np.array([r_[1] for r_ in FORKS], int)
name_f = [r_[0] for r_ in FORKS]
src_f = X_f[:, 0].astype(int) if len(X_f) else np.zeros(0, int)
counted = lab >= 1
with open("/kaggle/working/forks_f4.csv", "w") as fh:
    fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\n")
    for r_ in FORKS:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\n")
print(f"\n=== EVERY FORK THE FINAL GRAPHS CARRY (F4): {len(FORKS)} forks; verdict x source "
      f"(forks_f4.csv written) ===", flush=True)
print(f"  {'source':<8} {'TP':>6} {'FP':>6} {'uncounted':>10} {'precision(counted)':>19}", flush=True)
for s_, nm in ((1, "patch"), (0, "ilp")):
    m_ = src_f == s_
    tp_, fp_, un_ = int((lab[m_] == 2).sum()), int((lab[m_] == 1).sum()), int((lab[m_] == 0).sum())
    print(f"  {nm:<8} {tp_:>6} {fp_:>6} {un_:>10} {tp_ / max(1, tp_ + fp_):>19.3f}", flush=True)

print("\n=== FORK FEATURES by verdict (median TP / FP / uncounted; AUC TP-vs-FP on counted) ===",
      flush=True)
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


def _curve_line(keep, label):
    tp_k = int((lab[keep] == 2).sum())
    fp_k = int((lab[keep] == 1).sum())
    fn_k = N_GT - tp_k
    J = tp_k / max(1, tp_k + fp_k + fn_k)
    per = []
    for k in (0, 1):
        mk = keep & (fold_f == k)
        tpk = int((lab[mk] == 2).sum())
        fpk = int((lab[mk] == 1).sum())
        ngt = sum(WALK_FOLD[k].values())
        per.append(tpk / max(1, tpk + fpk + (ngt - tpk)))
    print(f"  {label:<34} {tp_k:>4} {fp_k:>5} {fn_k:>4} {J:>7.4f} {J - base_J:>+8.4f} "
          f"{0.1 * (J - base_J):>+9.4f} {per[0]:>8.4f} {per[1]:>8.4f}", flush=True)
    return J


HDR = (f"  {'cut':<34} {'TP':>4} {'FP':>5} {'FN':>4} {'div_J':>7} {'d div_J':>8} "
       f"{'~d score':>9} {'fold0 J':>8} {'fold1 J':>8}")

print("\n=== ONE-FEATURE RULE: prune forks whose orphan daughter's detection score sc_q "
      "is below a cut (quantiles of sc_q over ALL forks) ===", flush=True)
print(HDR, flush=True)
j_sc = FEAT_NAMES.index("sc_q")
sc = X_f[:, j_sc]
ok_sc = np.isfinite(sc)
_curve_line(ok_sc | ~ok_sc, "none")
for frac in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    thr = np.quantile(sc[ok_sc], frac)
    keep = ~ok_sc | (sc >= thr)
    _curve_line(keep, f"sc_q >= {thr:.3f} ({int(frac * 100)}% of forks)")

# ---- the gate: 2-fold out-of-fold, then the production gate on everything ----
print("\n=== THE FORK GATE (HistGB 100 it, lr 0.05, min_samples_leaf 10; trained on the "
      "OTHER fold's counted forks, TP=1 FP=0), pruning curve by quantile and by fixed "
      "probability ===", flush=True)
p_gate = np.full(len(FORKS), np.nan)
GATES = {}
for k in (0, 1):
    tr_ = (fold_f != k) & counted
    te_ = fold_f == k
    if tr_.sum() < 20 or (lab[tr_] == 2).sum() < 5:
        warn_once(f"gate_fold{k}", f"fold{k}: too few counted forks to train ({tr_.sum()})")
        continue
    mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                        min_samples_leaf=10, random_state=0)
    mg.fit(X_f[tr_], (lab[tr_] == 2).astype(int))
    p_gate[te_] = mg.predict_proba(X_f[te_])[:, 1]
    GATES[k] = mg
    te_c = te_ & counted
    print(f"  fold{k}: trained on {int(tr_.sum())} counted forks ({int((lab[tr_] == 2).sum())} TP); "
          f"out-of-fold AUC on fold{k}'s {int(te_c.sum())} counted forks: "
          f"{_auc((lab[te_c] == 2).astype(int), p_gate[te_c]):.4f}", flush=True)
ok_g = np.isfinite(p_gate)
print(HDR, flush=True)
THR_Q = {}
for frac in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    thr = np.quantile(p_gate[ok_g], frac) if frac > 0 else -1.0
    THR_Q[frac] = thr
    keep = ok_g & (p_gate > thr) if frac > 0 else ok_g
    _curve_line(keep, f"quantile {int(frac * 100)}% (p > {thr:.4f})")
for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2):
    keep = ok_g & (p_gate > thr)
    _curve_line(keep, f"fixed p > {thr}")
print(f"  oracle (every FP fork gone, every TP kept): div_J {TP / max(1, TP + FN):.4f} "
      f"({0.1 * (TP / max(1, TP + FN) - base_J):+.4f} of score)", flush=True)

# the production gate: every counted F4 fork
mg_full = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                         min_samples_leaf=10, random_state=0)
mg_full.fit(X_f[counted], (lab[counted] == 2).astype(int))
p_full = mg_full.predict_proba(X_f)[:, 1]
Q_FULL = {frac: float(np.quantile(p_full, frac)) for frac in (0.5, 0.6, 0.7, 0.8, 0.9)}
print(f"  production gate (all {int(counted.sum())} counted forks, {int((lab == 2).sum())} TP): "
      f"its p over all {len(FORKS)} forks has quantiles "
      + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q_FULL.items()), flush=True)
with open("/kaggle/working/fork_gate_f4.pkl", "wb") as fh:
    pickle.dump({"model": mg_full, "feature_names": FEAT_NAMES, "version": SCRIPT_VERSION,
                 "trained_on": "counted F4 forks, 128 6bba train movies, TP=1 FP=0",
                 "p_quantiles_all_forks": Q_FULL,
                 "oof_thresholds": {str(k_): float(v_) for k_, v_ in THR_Q.items()},
                 "detector": "F4", "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}, fh)
print("  fork_gate_f4.pkl written", flush=True)

# ---- pruning applied for real: edges removed, geff rewritten, official scorer ----
print("\n=== PRUNING APPLIED FOR REAL on the 128 F4 graphs (out-of-fold gate p, edge "
      "mother->orphan removed for every fork under the cut), official scorer ===",
      flush=True)
print(f"  {'cut':<22} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'edge_J':>7} {'node_r':>8} "
      f"{'score':>7} {'d adj_J':>8} {'d div_J':>8} {'d score':>8} {'f0 score':>9} {'f1 score':>9}",
      flush=True)
fork_by_movie = defaultdict(list)
for idx_, r_ in enumerate(FORKS):
    fork_by_movie[r_[0]].append((idx_, r_[2], r_[3]))
CUTS = [("none", None), ("oof q70", THR_Q.get(0.7)), ("oof q80", THR_Q.get(0.8)),
        ("fixed p>0.01", 0.01), ("fixed p>0.03", 0.03)]
PRUNE_RES = {}
for tag, thr in CUTS:
    rows_c = []
    per_fold = {0: [], 1: []}
    for name in SIX:
        if name not in MOVIE:
            continue
        if thr is None:
            row = PER[name]
        else:
            drop = {(p_, q_) for idx_, p_, q_ in fork_by_movie[name]
                    if np.isfinite(p_gate[idx_]) and p_gate[idx_] <= thr}
            row = _rescore(name, MOVIE[name], drop, tag.replace(" ", "_").replace(">", "")) if drop else PER[name]
        rows_c.append(row)
        per_fold[FOLD_OF[name]].append(row)
    rc = _with_ratio(rows_c)
    PRUNE_RES[tag] = rc
    b = PRUNE_RES["none"]
    f0, f1 = _with_ratio(per_fold[0]), _with_ratio(per_fold[1])
    print(f"  {tag:<22} {sum(r['division_tp'] for r in rows_c):>4}/"
          f"{sum(r['division_fp'] for r in rows_c):>4}/{sum(r['division_fn'] for r in rows_c):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} "
          f"{rc.get('edge_jaccard', float('nan')):>7.4f} {rc.get('total_node_ratio', float('nan')):>+8.4f} "
          f"{rc['score']:>7.4f} {rc['adj_edge_jaccard'] - b['adj_edge_jaccard']:>+8.4f} "
          f"{rc['division_jaccard'] - b['division_jaccard']:>+8.4f} {rc['score'] - b['score']:>+8.4f} "
          f"{f0['score']:>9.4f} {f1['score']:>9.4f}", flush=True)

# ---- the 44b6 draw under the production ranker and the production gate ----------
print(f"\n=== THE 12-MOVIE 44b6 DRAW (F4, production +BIDIR pkl, floor {PROD_FLOOR}): "
      f"the production gate applied at its own quantile thresholds ===", flush=True)
ROWS44 = {tag: [] for tag, _ in CUTS}
t44 = time.time()
for name in SET_44B6:
    if time.time() - t_all > BUDGET_S:
        print(f"  budget: {name} skipped", flush=True)
        continue
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    nodes0, edges0, pool, aux = solve_one(dets, T, seeds, feats, MODEL_BIDIR, MCOLS_BIDIR,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR)
    outp = pathlib.Path("/kaggle/working/ep_F4-44b6_ref") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    patch_final(nodes0, edges0, pool, outp)
    row0 = score_one(outp, name)
    snap = _snapshot()
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(outp))
    g = res_[0] if isinstance(res_, tuple) else res_
    ds = score_divisions(g, dset.tracks, scale=dset.scale)
    fr = fork_rows(name, -1, seeds, aux, ds.tp_forks, ds.fp_forks)
    Xf44 = np.array([r_[5] for r_ in fr], float) if fr else np.zeros((0, len(FEAT_NAMES)))
    pg = mg_full.predict_proba(Xf44)[:, 1] if len(fr) else np.zeros(0)
    cells = []
    for tag, thr in CUTS:
        if thr is None:
            row = row0
        else:
            thr_use = Q_FULL[0.7] if tag == "oof q70" else Q_FULL[0.8] if tag == "oof q80" else thr
            drop = {(r_[2], r_[3]) for r_, pv in zip(fr, pg) if pv <= thr_use}
            row = _rescore(name, snap, drop, "44b6_" + tag.replace(" ", "_").replace(">", "")) if drop else row0
        ROWS44[tag].append(row)
        cells.append(f"{row['division_tp']}/{row['division_fp']}/{row['division_fn']} {_adj_of(row):.4f}")
    print(f"  {name} forks {len(fr)} ({int((np.array([r_[4] for r_ in fr]) >= 1).sum()) if fr else 0} counted): "
          + " | ".join(cells) + f"  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool, aux, g, res_, dset, ds, snap
    gc.collect()
print(f"  {'cut':<22} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} {'d adj_J':>8} {'d score':>8}",
      flush=True)
for tag, _ in CUTS:
    if not ROWS44[tag]:
        continue
    rc = _with_ratio(ROWS44[tag])
    b = _with_ratio(ROWS44["none"])
    print(f"  {tag:<22} {sum(r['division_tp'] for r in ROWS44[tag]):>4}/"
          f"{sum(r['division_fp'] for r in ROWS44[tag]):>4}/{sum(r['division_fn'] for r in ROWS44[tag]):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b['adj_edge_jaccard']:>+8.4f} {rc['score'] - b['score']:>+8.4f}",
          flush=True)
print(f"  (44b6 block {(time.time() - t44) / 60:.1f} min; soft reference: +BIDIR F4-44b6 at "
      f"floor 0.48 = ilpsweep F48 0.8384)", flush=True)

print("\n=== reading it ===", flush=True)
print("  The BASE bench said the false forks are separable at fork time (AUC 0.92/0.90, "
      "+0.011 of score at a 70% cut). This is the same measurement on the production "
      "primaries (F4) with the pruning applied for real and scored by the official "
      "scorer, adj_J included; the production gate (all counted F4 forks) is written to "
      "fork_gate_f4.pkl with its p-quantiles over all forks; the 44b6 draw says whether "
      "the 6bba-trained gate hurts the other domain (2 divisions there, so read adj_J and "
      "FP forks, not div_J). A cut is a submission candidate if 'd score' is positive in "
      "both 6bba folds and 44b6's adj_J does not fall.", flush=True)
warning_summary()
print(f"\nDIVBENCH3 DONE in {(time.time() - t_all) / 60:.1f} min (stage B "
      f"{(time.time() - tb) / 60:.1f} min), {len(ROWS)} of {len(SIX)} movies scored", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
if len(ROWS) == len(SIX) and not INCOMPLETE:
    print("DIVBENCH3 OK", flush=True)
else:
    print(f"[degraded] DIVBENCH3 INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)
'''
src = src[:start] + stage_b
open("divbench3.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
