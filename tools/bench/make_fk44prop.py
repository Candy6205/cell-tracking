"""fk44img.py -> fk44prop.py: forkprop's winning arm replayed on the 71 44b6 movies, under the image gate.

WHY (2026-09-17). forkprop (128 6bba movies): the orphan-fork patch with contrast-ranked proposals,
caps x100 (nothing refused by a cap) and DIV_DIVERGE_UM 1.5, pruned by the joint image gate, beats
REF + jointimg by +0.0050 [+0.0012, +0.0089] (61/36/64 against 55/42/70); every capped arm is
indistinguishable from REF. The hidden test carries both domains and the 44b6 half has not been
measured: with no cap the patch admits ~3.7x more forks, and the gate that prunes them has to
hold on 44b6 too (fk44img: 44b6 forks are few, 26 GT divisions, and the image columns read
weaker there). This run answers that and produces the gate a submission would ship.

WHAT. fk44img's harness (71 44b6 movies, F4HP primaries, out-of-fold rankers, the production ILP
and patches, official scorer, the image opened around the judge) plus:
  * forkprop's knobs (FP_RANK / FP_CAP / fp_contrast) in `_add_safe_divisions`, verbatim;
  * one arm, CR_NOCAP_DV150, run as a second patch_final on the same ILP solve of every movie,
    the image open around its patch and its judge (contrast at every proposal, image columns on
    every fork), judged and snapshotted like fk44img's steal arms were;
  * THE CROSS: REF and the arm each under the joint image gate (20 + 11 columns, p > 0.03),
    44b6 folds out of fold; each cell's gate is trained on that cell's own 6bba forks (all 128
    movies: forks_dw55.csv for REF, forks_CR_NOCAP_DV150.csv for the arm, dataset
    biohub-forktables) plus the other 44b6 fold's counted forks of the same cell -- forkprop's
    cross with the domains swapped; paired bootstrap over the 71 movies, score and adj_J,
    gated and ungated;
  * fork_gate_prop.pkl: the arm's counted forks in BOTH domains, all movies (the submission gate
    for a submit_prop variant), with the provenance submit_img asserts on.

Attach: fk44img's sources + the dataset faustine649/biohub-forktables. CPU, ~4-5 h
(fk44img 2.1 h + the arm's patch, crops and judge per movie).
"""
import os

src = open("fk44img.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-fk44img"', 'SCRIPT_VERSION = "2026-09-17b-fk44prop"')

# ---- forkprop's knobs ---------------------------------------------------------------------------
rep('DIV_GLOBAL_CAP = 0.00375\n',
    'DIV_GLOBAL_CAP = 0.00375\n'
    'FP_RANK = "dist"          # shipped: proposals by d_pq + 0.15 d_s; "contrast": by the image contrast, distance as tie-break\n'
    'FP_CAP = 1.0              # multiplier on DIV_FRAME_CAP and DIV_GLOBAL_CAP in the orphan pass\n'
    'FP_MEMO = {}              # (t, z, y, x) -> contrast, per movie\n'
    'FP_STAT = defaultdict(int)\n')

# ---- the orphan pass, forkprop's text ----------------------------------------------------------
rep('''            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()''',
    '''            if FP_RANK == "contrast":
                key_ = (-(fp_contrast(nodes, p) + fp_contrast(nodes, c1) + fp_contrast(nodes, q)), d_pq + DIV_W_SISTER * d_s)
            else:
                key_ = (d_pq + DIV_W_SISTER * d_s,)
            proposals.append((key_, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()''')
rep('''    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0''',
    '''    max_global = max(1, int(DIV_GLOBAL_CAP * FP_CAP * len(nodes)))
    added = 0''')
rep('''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1''',
    '''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * FP_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1''')

# ---- fp_contrast, forkprop's text, at the end of the forkimg block ------------------------------
rep('# ---- end forkimg block ----\n', '''

def fp_contrast(nodes, g):
    """the image contrast of one node (memoised per movie); -1 when no image."""
    img = IMG_CTX.get("img")
    if img is None:
        FP_STAT["no_img"] += 1
        return -1.0
    n_ = nodes[g]
    key = (int(n_["t"]),) + tuple(int(round(float(v))) for v in n_["xyz"])
    if key not in FP_MEMO:
        _pk, con, _rg = _fi_node(img, int(n_["t"]), n_["xyz"])
        FP_MEMO[key] = float(con) if np.isfinite(con) else -1.0
        FP_STAT["crops"] += 1
    return FP_MEMO[key]
# ---- end forkimg block ----
''')

# ---- the arm ------------------------------------------------------------------------------------
rep('''STEAL_ARMS = []       # 44b6 has ~26 GT divisions: it cannot price a steal.
                      # The 5a cases are still written out (steal5a_44b6.csv).
''', '''# (arm name, the orphan pass's knobs). forkprop's winning arm: proposals ranked by the image
# contrast of mother + both daughters, caps x100 (no proposal refused by a cap), diverge 1.5.
# The steal machinery below is untouched but never engaged (steal stays None).
STEAL_ARMS = [("CR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5))]
PROP_DEFAULTS = ("dist", 1.0, 2.25)
''')
rep('''    for arm, margin in STEAL_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa, pfun=pfun, steal=margin)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        n_st = len(STEAL_LOG)
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
''', '''    for arm, cfg in STEAL_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        _tfi = time.time()
        try:
            IMG_CTX.clear()
            IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
        except Exception as _e:                                   # noqa: BLE001
            warn_once("forkprop_open", f"{name}: image not opened ({_e}); contrast -1, image features NaN")
            IMG_CTX.clear()
        FP_MEMO.clear()
        FP_RANK, FP_CAP, DIV_DIVERGE_UM = cfg["rank"], cfg["cap"], cfg["diverge"]
        patch_final(nodes0, edges0, pool, outa)
        FP_RANK, FP_CAP, DIV_DIVERGE_UM = PROP_DEFAULTS
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        n_st = len(STEAL_LOG)
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        IMG_CTX.clear()
        FI_STAT["seconds"] += int(time.time() - _tfi)
''')
rep('f"/kaggle/working/forks_f4_{arm}.csv"', 'f"/kaggle/working/forks_44b6_{arm}.csv"')
rep('print(f"\\n=== THE STEAL ARMS: the patch allowed to take a daughter the ILP had already "\n'
    '      f"placed, official scorer, all {len(BENCH)} movies ===", flush=True)',
    'print(f"\\n=== THE ARM, UNGATED: forkprop\'s orphan patch (contrast-ranked proposals, caps x100, diverge 1.5) "\n'
    '      f"against REF (distance-ranked, shipped caps, diverge 2.25), official scorer, all {len(BENCH)} 44b6 movies; "\n'
    '      f"forkprop read it at -0.0103 ungated on 6bba, the gate is what makes it pay ===", flush=True)')

# ---- the cross under the image gate, the bootstrap, the submission gate ------------------------
CROSS = r'''
# ---- fk44prop: REF and the arm under the JOINT IMAGE GATE, 44b6 out of fold ---------------------
print(f"\n=== FK44PROP CROSS: REF and the arm each under the JOINT IMAGE GATE (20 + 11 columns, HistGB "
      f"100 it, lr 0.05, msl 10, real pruning at p > 0.03, official scorer), 44b6 folds out of fold. Each "
      f"cell's gate is trained on that cell's OWN 6bba forks (all 128 movies, dataset biohub-forktables: "
      f"forks_dw55.csv for REF, forks_<arm>.csv for the arm) plus the other 44b6 fold's counted forks of "
      f"the same cell -- forkprop's cross with the domains swapped ===", flush=True)
import pandas as _pd                                                                  # noqa: E402
JIMG_COLS = J20 + JIMG
_SIX_CSV = {"REF": "forks_dw55.csv", **{a_: f"forks_{a_}.csv" for a_, _c in STEAL_ARMS}}
print(f"  fk44prop: proposal contrast crops {FP_STAT['crops']:,} ({FP_STAT['no_img']:,} without an image)",
      flush=True)


def _six_tab(fn):
    hit = [p_ for p_ in pathlib.Path("/kaggle/input").glob(f"**/{fn}")]
    assert hit, f"attach the dataset faustine649/biohub-forktables ({fn})"
    t_ = _pd.read_csv(hit[0])
    assert list(t_.columns[5:]) == FEAT_NAMES, (fn, list(t_.columns[5:])[:8])
    return t_


def _fit_img(X_, y_):
    m_ = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
    m_.fit(X_[:, JIMG_COLS], y_)
    return m_


def _cell_img(cell, forks_a, movies_a, per_a):
    t6 = _six_tab(_SIX_CSV[cell])
    c6 = t6[t6.label >= 1]
    X6, y6 = c6[FEAT_NAMES].to_numpy(float), (c6.label.to_numpy(int) == 2).astype(int)
    Xa = np.array([r_[5] for r_ in forks_a], float) if forks_a else np.zeros((0, len(FEAT_NAMES)))
    la = np.array([r_[4] for r_ in forks_a], int)
    fa = np.array([r_[1] for r_ in forks_a], int)
    pa = np.full(len(forks_a), np.nan)
    aucs = []
    for k in (0, 1):
        tr = (fa != k) & (la >= 1)
        m_ = _fit_img(np.concatenate([X6, Xa[tr]]), np.concatenate([y6, (la[tr] == 2).astype(int)]))
        held = fa == k
        if held.any():
            pa[held] = m_.predict_proba(Xa[held][:, JIMG_COLS])[:, 1]
        hc = held & (la >= 1)
        if (la[hc] == 2).any() and (la[hc] == 1).any():
            aucs.append((int(hc.sum()), _auc((la[hc] == 2).astype(int), pa[hc])))
    by_movie = defaultdict(list)
    for i_, r_ in enumerate(forks_a):
        by_movie[r_[0]].append((i_, r_[2], r_[3]))
    rows_c, pf, n_cut = [], {0: [], 1: []}, 0
    for name in BENCH:
        if name not in movies_a:
            continue
        key = movies_a[name]["key"]
        want = [(p_, q_) for i_, p_, q_ in by_movie[name] if pa[i_] <= 0.03]
        miss = [e_ for e_ in want if e_ not in key]
        if miss:
            warn_once("prune_edge_missing",
                      f"{name}: {len(miss)} fork edges to prune are not in the snapshot's fork map -> "
                      f"NOT removed, the cut is a LOWER bound")
        drop = {key[e_] for e_ in want if e_ in key}
        n_cut += len(drop)
        row = _rescore(name, movies_a[name], drop, f"img_{cell}") if drop else per_a[name]
        rows_c.append(row)
        pf[FOLD_OF[name]].append(row)
    return dict(rows=rows_c, pf=pf, rc=_with_ratio(rows_c), aucs=aucs, n6=int(len(y6)), tp6=int(y6.sum()),
                n44=int((la >= 1).sum()), tp44=int((la == 2).sum()), forks=len(forks_a), cut=n_cut)


XIMG = {}
for cell, fk_, mv_, pr_ in [("REF", FORKS, MOVIE, PER)] + [(a_, ARM_FORKS[a_], ARM_MOVIE[a_], ARM_PER[a_])
                                                          for a_, _c in STEAL_ARMS]:
    XIMG[cell] = _cell_img(cell, fk_, mv_, pr_)
_rr = XIMG["REF"]["rc"]
_rf = {k: _with_ratio(XIMG["REF"]["pf"][k])["score"] for k in (0, 1)}
print(f"  {'cell':<28} {'forks':>6} {'cut':>6} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'edge_J':>7} "
      f"{'score':>7} {'vs REF+img':>11} {'f0':>8} {'f1':>8}   train rows 6bba (TP) + 44b6 (TP) / AUC per held 44b6 fold",
      flush=True)
for cell, r_ in XIMG.items():
    rc = r_["rc"]
    f0 = _with_ratio(r_["pf"][0])["score"] - _rf[0]
    f1 = _with_ratio(r_["pf"][1])["score"] - _rf[1]
    print(f"  {cell + ' + jointimg':<28} {r_['forks']:>6,} {r_['cut']:>6,} "
          f"{sum(r['division_tp'] for r in r_['rows']):>4}/"
          f"{sum(r['division_fp'] for r in r_['rows']):>4}/"
          f"{sum(r['division_fn'] for r in r_['rows']):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc.get('edge_jaccard', float('nan')):>7.4f} {rc['score']:>7.4f} "
          f"{rc['score'] - _rr['score']:>+11.4f} {f0:>+8.4f} {f1:>+8.4f}   "
          f"{r_['n6']} ({r_['tp6']}) + {r_['n44']} ({r_['tp44']}) / "
          + " ".join(f"{n}:{a:.4f}" for n, a in r_["aucs"]), flush=True)
print(f"  (REF + gate p>0.03 with the 44b6-only 20-column gate read {REF_GATED['score']:.4f} above; forkprop's "
      f"6bba cross: REF + jointimg 0.9484, CR_NOCAP_DV150 + jointimg 0.9534, +0.0050 [+0.0012, +0.0089])", flush=True)

_ok = [n for n in BENCH if n in PER]
for cell in XIMG:
    assert len(XIMG[cell]["rows"]) == len(_ok), (cell, len(XIMG[cell]["rows"]), len(_ok))
UNG = {"REF": [PER[n] for n in _ok]}
for a_, _c in STEAL_ARMS:
    assert all(n in ARM_PER[a_] for n in _ok), a_
    UNG[a_] = [ARM_PER[a_][n] for n in _ok]
print(f"\n  PAIRED BOOTSTRAP over the {len(_ok)} 44b6 movies, 2,000 draws, the same resample for every cell; "
      f"the arm AGAINST REF, gated (what a submission would ship) and ungated ===", flush=True)
_rb = np.random.default_rng(9090)
_draws = [_rb.choice(len(_ok), size=len(_ok), replace=True) for _ in range(2000)]


def _bank(rows_):
    agg = [_with_ratio([rows_[i] for i in d]) for d in _draws]
    return {"score": np.array([a["score"] for a in agg]), "adj_J": np.array([a["adj_edge_jaccard"] for a in agg]),
            "edge_J": np.array([a.get("edge_jaccard", float("nan")) for a in agg])}


BSG = {c_: _bank(XIMG[c_]["rows"]) for c_ in XIMG}
BSU = {c_: _bank(UNG[c_]) for c_ in UNG}
print(f"    {'comparison':<44} {'metric':<7} {'mean d':>9} {'90% interval':>22}  distinguishable", flush=True)
for a_, _c in STEAL_ARMS:
    for tag_, bank in ((f"{a_} + jointimg vs REF + jointimg", BSG), (f"{a_} ungated vs REF ungated", BSU)):
        for nm_ in ("score", "adj_J", "edge_J"):
            d = bank[a_][nm_] - bank["REF"][nm_]
            lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
            print(f"    {tag_:<44} {nm_:<7} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
                  f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: the arm ships only if its gated cell does not lose on 44b6 (interval not below zero on score "
      "AND adj_J); a 44b6 loss the size of the 6bba gain cancels it on a half-and-half test set.", flush=True)

# ---- the submission gate: the arm's counted forks in BOTH domains, all movies -------------------
for a_, cfg_ in STEAL_ARMS:
    t6 = _six_tab(_SIX_CSV[a_])
    c6 = t6[t6.label >= 1]
    X6, y6 = c6[FEAT_NAMES].to_numpy(float), (c6.label.to_numpy(int) == 2).astype(int)
    fk_ = ARM_FORKS[a_]
    Xa = np.array([r_[5] for r_ in fk_], float) if fk_ else np.zeros((0, len(FEAT_NAMES)))
    la = np.array([r_[4] for r_ in fk_], int)
    cnt = la >= 1
    Xall = np.concatenate([X6, Xa[cnt]])
    yall = np.concatenate([y6, (la[cnt] == 2).astype(int)])
    m_full = _fit_img(Xall, yall)
    X_every = np.concatenate([t6[FEAT_NAMES].to_numpy(float), Xa]) if len(Xa) else t6[FEAT_NAMES].to_numpy(float)
    p_every = m_full.predict_proba(X_every[:, JIMG_COLS])[:, 1]
    Q = {f: float(np.quantile(p_every, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
    n_train = {"6bba": int(len(y6)), "44b6": int(cnt.sum()), "tp": int(yall.sum())}
    blob = {"model": m_full, "feature_names": [FEAT_NAMES[j] for j in JIMG_COLS], "version": SCRIPT_VERSION,
            "trained_on": (f"counted F4HP forks under the {a_} orphan patch (proposals ranked by image contrast, "
                           f"caps x{cfg_['cap']:g}, DIV_DIVERGE_UM {cfg_['diverge']}), 128 6bba (forkprop) + "
                           f"{len(ROWS)} 44b6 (fk44prop), 20 gate + 11 image features, TP=1 FP=0"),
            "n_train": n_train, "p_quantiles_all_forks": Q, "detector": "F4",
            "image": {"half_um": FI_HALF_UM, "peak_um": FI_PEAK_UM, "size_um": FI_SIZE_UM,
                      "sigma_vox": list(FI_SIGMA_VOX)},
            "patch": {"arm": a_, "rank": cfg_["rank"], "cap": float(cfg_["cap"]), "diverge": float(cfg_["diverge"]),
                      "DIV_W_SISTER": DIV_W_SISTER, "DIV_FRAME_CAP": DIV_FRAME_CAP, "DIV_GLOBAL_CAP": DIV_GLOBAL_CAP,
                      "DIV_CHILD_UM": DIV_CHILD_UM, "DIV_PARENT_UM": DIV_PARENT_UM, "DIV_SISTER_UM": DIV_SISTER_UM},
            "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}
    with open("/kaggle/working/fork_gate_prop.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    print(f"\n  fork_gate_prop.pkl written: n_train {n_train}; p quantiles over all {len(p_every):,} forks of the arm "
          f"in both domains: " + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q.items()), flush=True)
    _csv_a = pathlib.Path("/kaggle/working") / f"fk44prop_arm_{a_}.csv"
    _keys_a = sorted({k for r in ARM_PER[a_].values() for k in r})
    with open(_csv_a, "w", encoding="utf-8", newline="\n") as _fh:
        _fh.write("dataset,fold," + ",".join(_keys_a) + "\n")
        for _n in sorted(ARM_PER[a_]):
            _r = ARM_PER[a_][_n]
            _fh.write(f"{_n},{FOLD_OF.get(_n, -1)}," + ",".join(repr(_r.get(_k)) for _k in _keys_a) + "\n")
    print(f"  {_csv_a.name} written (ungated per-movie rows of the arm)", flush=True)

'''
rep('print("\\n=== reading it ===", flush=True)', CROSS + 'print("\\n=== reading it ===", flush=True)')

# ---- names -------------------------------------------------------------------------------------
rep('print("FK44IMG OK", flush=True)', 'print("FK44PROP OK", flush=True)')
rep('print(f"[degraded] FK44IMG INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)',
    'print(f"[degraded] FK44PROP INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)')
rep('print(f"\\nFK44IMG DONE in', 'print(f"\\nFK44PROP DONE in')
rep('f"fk44img_{DET_SRC}.csv"', 'f"fk44prop_{DET_SRC}.csv"')

assert src.count("fp_contrast(") == 4 and src.count("FP_CAP *") == 2 and src.count("PROP_DEFAULTS") == 2
assert src.count("fork_gate_prop.pkl") == 2 and src.index("def fp_contrast(") < src.index("\nfor name in BENCH:\n")
assert src.index("def _add_safe_divisions(") < src.index("def fp_contrast(")      # called at run time only: fine
assert "forks_f4_{arm}" not in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fk44prop.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written fk44prop.py", len(src))
