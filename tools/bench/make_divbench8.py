"""divbench7.py -> divbench8.py: the JOINT fork gate, under real pruning and
the official scorer, with the pkl fitted in the image that will load it.

WHY. gatelab7-10 (2026-09-08, offline on both fork tables) found the first
gate variant in this whole line to clear the bench's own noise bar: adding
44b6's 67 counted forks to the training set and evaluating OUT OF FOLD ON
6bba moves div_J 0.2442 -> 0.2805, paired +0.0360, 90% [+0.0163, +0.0597].
Four controls say the 44b6 LABELS carry it -- padding with a bootstrap of
the same 6bba rows is -0.0280, and the same 44b6 rows with labels shuffled
is -0.0186.

WHAT IS STILL MISSING, and is exactly what this run supplies:

  1. That div_J is SIMULATED from the fork table's labels. adj_J is not
     measured at all. The gate's cut removes edges from the graph, so the
     official scorer has to run on the pruned graphs -- every recorded
     number on this line is of that form (divbench3: 42/47/83, div_J
     0.2442, adj_J 0.9143, score 0.9387).
  2. The pkl must be FITTED ON KAGGLE. Local scikit-learn 1.9.0 reproduces
     the RANKING to four decimals but not the CALIBRATION: the production
     gate's quantiles over all 7,004 forks read 80% 0.0238 locally against
     Kaggle's 0.0275, and the cut is a fixed 0.03.

WHAT IT COSTS. The gate is the only thing that changes, so this is ONE
solve and ONE patch per movie -- divbench7's REF pass -- and then four
cheap gate passes over the same snapshots. Cheaper than divbench7.

THE 44b6 ROWS ARE EMBEDDED IN THE SOURCE, not mounted. Only the 67 COUNTED
forks are ever trained on (61 FP + 6 TP), which is 12 KB of literal. This
avoids a dataset upload and, with it, any chance of the MCP connector
wiping this kernel's data sources.

ARMS, all four applied to the SAME REF snapshots with real pruning at
p > 0.03, each trained out of fold on 6bba:

  gate6   6bba's own counted forks only            ANCHOR: 42/47/83, div_J
                                                   0.2442, adj_J 0.9143,
                                                   score 0.9387
  joint   + all 67 counted 44b6 forks              the candidate
  shuf    + the same 67 rows, LABELS SHUFFLED      control (E): if this
                                                   matches joint, the gain
                                                   is regularisation, not
                                                   label information
  pad     + 67 rows bootstrapped from the SAME     control (A): if this
          6bba training fold                       matches joint, the gain
                                                   is row count

ANCHORS: REF ungated must reproduce 50/249/75, div_J 0.1337, adj_J 0.9133,
score 0.9266; gate6 must reproduce 42/47/83, div_J 0.2442, adj_J 0.9143,
score 0.9387.
"""
import os

src = open("divbench7.py", encoding="utf-8").read()
NL = "\n"
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07d-divbench7"', 'SCRIPT_VERSION = "2026-09-08a-divbench8"')

# ---- one solve, one patch: divbench8 changes the GATE, not the patch ------
rep('''CAP_ARMS = [("cap3x", 0.0228, 0.01125), ("cap8x", 0.0608, 0.03),
            ("nocap", 1.0, 1.0)]''',
    '''CAP_ARMS = []         # divbench8 sweeps the GATE, not the patch: one solve, one
                      # patch, then four cheap gate passes over the same snapshots''')

# ---- the 44b6 counted forks, embedded ------------------------------------
BLK = open(SC + "blk44.py", encoding="utf-8").read()
assert BLK.count("\n    (") == 67, BLK.count("\n    (")
rep('''FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order''',
    '''FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order

# ---------------------------------------------------------------------------
# The 67 COUNTED forks from 44b6's 71 movies (61 FP + 6 TP), lifted verbatim
# out of divbench5's forks_44b6.csv. Only counted rows are here because only
# counted rows are ever trained on. Embedded rather than mounted so that this
# kernel keeps its data sources.
# ---------------------------------------------------------------------------
''' + BLK + '''assert FEAT44 == FEAT_NAMES, "the 44b6 fork table's columns are not this run's"
X44 = np.array([r_[1] for r_ in ROWS44], float)
Y44 = np.array([1 if r_[0] == 2 else 0 for r_ in ROWS44], int)
assert X44.shape == (67, len(FEAT_NAMES)), X44.shape
assert int(Y44.sum()) == 6, int(Y44.sum())''')

# ---- the gate arms replace the cap arms -----------------------------------
A_MARK = '''SIX_OK = [n for n in SIX if n in PER]
print(f"\\n=== THE CAP ARMS, UNGATED'''
a = src.index(A_MARK)
b = src.index('print("\\n=== reading it ===", flush=True)')
ARMS_BLOCK = r'''SIX_OK = [n for n in SIX if n in PER]
b_ref = RES
f0_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 0])
f1_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 1])
print(f"\n=== REF, UNGATED (one solve, one patch, official scorer, {len(SIX_OK)} "
      f"movies) ===", flush=True)
print(f"  REF  {len(FORKS):,} forks   "
      f"{sum(r['division_tp'] for r in ROWS)}/{sum(r['division_fp'] for r in ROWS)}/"
      f"{sum(r['division_fn'] for r in ROWS)}   div_J {b_ref['division_jaccard']:.4f}   "
      f"adj_J {b_ref['adj_edge_jaccard']:.4f}   score {b_ref['score']:.4f}"
      f"      want 50/249/75  0.1337  0.9133  0.9266", flush=True)

print("\n=== THE GATE ARMS: the SAME 128 REF graphs, four gates, real pruning at "
      "p > 0.03, official scorer. Every gate is trained OUT OF FOLD on 6bba; the "
      "arms differ only in what is ADDED to each training fold ===", flush=True)

X_ALL = np.array([r_[5] for r_ in FORKS], float)
LAB_ALL = np.array([r_[4] for r_ in FORKS], int)
FLD_ALL = np.array([r_[1] for r_ in FORKS], int)
CNT_ALL = LAB_ALL >= 1
BY_MOVIE = defaultdict(list)
for i_, r_ in enumerate(FORKS):
    BY_MOVIE[r_[0]].append((i_, r_[2], r_[3]))
print(f"  6bba: {len(FORKS):,} forks, {int(CNT_ALL.sum())} counted, "
      f"{int((LAB_ALL == 2).sum())} TP    44b6 rows available to add: {len(X44)} "
      f"counted, {int(Y44.sum())} TP", flush=True)


def _gate_rows(tag, thr, add=None, per_fold_add=None):
    """add: (X, y) appended to BOTH training folds.
    per_fold_add: fn(k, tr_mask) -> (X, y) built from that fold's own rows."""
    pa = np.full(len(FORKS), np.nan)
    aucs = []
    for kk in (0, 1):
        tr = (FLD_ALL != kk) & CNT_ALL
        Xtr, ytr = X_ALL[tr][:, J20], (LAB_ALL[tr] == 2).astype(int)
        ex = add if add is not None else (per_fold_add(kk, tr) if per_fold_add else None)
        if ex is not None and len(ex[0]):
            Xtr = np.vstack([Xtr, np.asarray(ex[0], float)[:, J20]])
            ytr = np.concatenate([ytr, np.asarray(ex[1], int)])
        mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                            min_samples_leaf=10, random_state=0)
        mg.fit(Xtr, ytr)
        te = FLD_ALL == kk
        pa[te] = mg.predict_proba(X_ALL[te][:, J20])[:, 1]
        tc = te & CNT_ALL
        aucs.append((int(tc.sum()), _auc((LAB_ALL[tc] == 2).astype(int), pa[tc])))
    rows_c, pf = [], {0: [], 1: []}
    for name in SIX_OK:
        key = MOVIE[name]["key"]
        want = [(p_, q_) for i_, p_, q_ in BY_MOVIE[name]
                if np.isfinite(pa[i_]) and pa[i_] <= thr]
        miss = [e_ for e_ in want if e_ not in key]
        if miss:
            warn_once("prune_edge_missing",
                      f"{name}: {len(miss)} fork edges to prune are not in the "
                      f"snapshot's fork map -> NOT removed, the cut is a LOWER bound")
        drop = {key[e_] for e_ in want if e_ in key}
        row = _rescore(name, MOVIE[name], drop, f"{tag}_p{thr}") if drop else PER[name]
        rows_c.append(row)
        pf[FOLD_OF[name]].append(row)
    return _with_ratio(rows_c), rows_c, pf, aucs, pa


_rng8 = np.random.default_rng(20260908)
_shuf_y = Y44.copy()
_rng8.shuffle(_shuf_y)


def _pad_same(kk, tr):
    """Control (A): 67 rows BOOTSTRAPPED from this fold's own counted 6bba forks.
    Same row count, same distribution, no information that was not already there."""
    ii = np.where(tr)[0]
    pick = np.random.default_rng(1000 + kk).choice(ii, size=len(X44), replace=True)
    return X_ALL[pick], (LAB_ALL[pick] == 2).astype(int)


THR = 0.03
GATE_ARMS = [
    ("gate6", "6bba only (SHIPPED)", dict()),
    ("joint", "+ 67 counted 44b6", dict(add=(X44, Y44))),
    ("shuf", "+ the same 67, labels SHUFFLED", dict(add=(X44, _shuf_y))),
    ("pad", "+ 67 bootstrapped 6bba rows", dict(per_fold_add=_pad_same)),
]
print(f"\n  {'arm':<8} {'what it trains on':<32} {'div tp/fp/fn':>14} {'div_J':>7} "
      f"{'adj_J':>7} {'score':>7} {'d score vs REF':>15} {'vs gate6':>9} "
      f"{'f0':>8} {'f1':>8}", flush=True)
GRES = {}
for tag, what, kw in GATE_ARMS:
    rc, rows2, pf2, aucs, pa = _gate_rows(tag, THR, **kw)
    GRES[tag] = dict(rc=rc, rows=rows2, pf=pf2, aucs=aucs, p=pa)
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    base = GRES["gate6"]
    d6 = rc["score"] - base["rc"]["score"]
    df0 = f0["score"] - _with_ratio(base["pf"][0])["score"]
    df1 = f1["score"] - _with_ratio(base["pf"][1])["score"]
    print(f"  {tag:<8} {what:<32} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} "
          f"{rc['score'] - b_ref['score']:>+15.4f} {d6:>+9.4f} "
          f"{df0:>+8.4f} {df1:>+8.4f}"
          + "   AUC " + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + ("      want 42/47/83  0.2442  0.9143  0.9387 (divbench3)"
             if tag == "gate6" else ""), flush=True)

print(f"\n  PAIRED BOOTSTRAP over the {len(SIX_OK)} movies, 2,000 draws, on the "
      f"OFFICIAL SCORE of the pruned graphs -- the same resample for every arm, so "
      f"the movie draw cancels ===", flush=True)
_names = np.array(SIX_OK)
_pos = {n: i for i, n in enumerate(SIX_OK)}
_rb = np.random.default_rng(8080)
_draws = [_rb.choice(len(_names), size=len(_names), replace=True) for _ in range(2000)]
BS = {}
for tag, _w, _k in GATE_ARMS:
    rows2 = GRES[tag]["rows"]
    BS[tag] = np.array([_with_ratio([rows2[i] for i in d])["score"] for d in _draws])
print(f"    {'arm':<8} {'mean d score vs gate6':>22} {'90% interval':>24}  "
      f"distinguishable", flush=True)
for tag, _w, _k in GATE_ARMS:
    if tag == "gate6":
        continue
    d = BS[tag] - BS["gate6"]
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"    {tag:<8} {float(np.mean(d)):>+22.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
          f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)

print("\n  Each arm's ranking read at its OWN best threshold on the fork table, so a "
      "shifted probability scale cannot be mistaken for a worse ranking (upper "
      "bounds, chosen on the labels -- they compare RANKINGS, not achievable "
      "scores):", flush=True)
print(f"    {'arm':<8} {'best p':>8} {'TP':>4} {'FP':>5} {'div_J':>8} {'vs gate6':>10}",
      flush=True)
_bestJ = {}
for tag, _w, _k in GATE_ARMS:
    pa = GRES[tag]["p"]
    ok = np.isfinite(pa)
    bJ, bthr, btp, bfp = -1.0, 0.0, 0, 0
    for t_ in np.unique(pa[ok]):
        keep = ok & (pa > t_)
        tp_ = int((LAB_ALL[keep] == 2).sum())
        fp_ = int((LAB_ALL[keep] == 1).sum())
        J_ = tp_ / max(1, tp_ + fp_ + (TP + FN - tp_))
        if J_ > bJ:
            bJ, bthr, btp, bfp = J_, float(t_), tp_, fp_
    _bestJ[tag] = bJ
    print(f"    {tag:<8} {bthr:>8.4f} {btp:>4} {bfp:>5} {bJ:>8.4f} "
          f"{bJ - _bestJ['gate6']:>+10.4f}", flush=True)

# ---- the joint production gate, FITTED HERE -------------------------------
print(f"\n=== THE JOINT PRODUCTION GATE, fitted in this image on all "
      f"{int(CNT_ALL.sum())} counted 6bba forks + all {len(X44)} counted 44b6 forks "
      f"===", flush=True)
_Xj = np.vstack([X_ALL[CNT_ALL][:, J20], X44[:, J20]])
_yj = np.concatenate([(LAB_ALL[CNT_ALL] == 2).astype(int), Y44])
mg_joint = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                          min_samples_leaf=10, random_state=0)
mg_joint.fit(_Xj, _yj)
p_j = mg_joint.predict_proba(X_ALL[:, J20])[:, 1]
Q_J = {f: float(np.quantile(p_j, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
with open("/kaggle/working/fork_gate_joint.pkl", "wb") as fh:
    pickle.dump({"model": mg_joint, "feature_names": FEAT20,
                 "version": SCRIPT_VERSION,
                 "trained_on": "counted F4 forks, 128 6bba + 71 44b6 train movies, "
                               "TP=1 FP=0",
                 "n_train": {"6bba": int(CNT_ALL.sum()), "44b6": int(len(X44)),
                             "tp": int(_yj.sum())},
                 "p_quantiles_all_forks": Q_J, "detector": "F4",
                 "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}, fh)
print(f"  fork_gate_joint.pkl written; {len(_Xj)} training rows, {int(_yj.sum())} TP; "
      f"p quantiles over all {len(FORKS)} 6bba forks: "
      + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q_J.items()), flush=True)
print("  (the 6bba-only gate written above reads "
      + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q_FULL.items())
      + " -- if these two disagree, the 0.03 cut sits at a different place in each "
        "and the submission must re-read it from ITS OWN quantiles)", flush=True)
_kept = int((p_j > 0.03).sum())
print(f"  at a fixed p > 0.03 the joint gate keeps {_kept:,} of {len(FORKS):,} forks "
      f"({100.0 * _kept / len(FORKS):.1f}%); the 6bba-only gate keeps "
      f"{int((p_full > 0.03).sum()):,} ({100.0 * (p_full > 0.03).sum() / len(FORKS):.1f}%)",
      flush=True)

'''
src = src[:a] + ARMS_BLOCK + src[b:]

# ---- how to read it -------------------------------------------------------
rep('''print("  (1) FIRST read the 'frame-refused' column: if it does not fall to ~0 at "
      "cap8x and nocap, the arms did not lift the constraint and nothing else in the "
      "table means anything.", flush=True)
print("  (2) The decision is the GATED table. An arm is a submission candidate iff its "
      "score beats REF+gate in BOTH folds. The ungated table is how this line was read "
      "in August; it is here to show the shape of the trade, not to decide.", flush=True)
print("  (3) Size: the bench resolves about +-0.03 of div_J at 90% (gatelab, paired "
      "bootstrap over movies) = +-0.003 of score, while adj_J is measured far more "
      "precisely. An arm that trades -0.002 of adj_J for +0.03 of div_J is a real "
      "loss. Read both columns, never the sum alone.", flush=True)
print("  (4) nocap is a CONTROL, not a candidate: it measures the sum of its good and "
      "bad cases and a gated version can sit ABOVE it (2026-08-26). If nocap comes "
      "back near zero, look for the cancellation.", flush=True)''',
    '''print("  (1) FIRST check the gate6 anchor: 42/47/83, div_J 0.2442, adj_J 0.9143, "
      "score 0.9387. It is divbench3's shipped gate refit from the same rows. If it "
      "does not land there, the pruning path changed and nothing below it means "
      "anything.", flush=True)
print("  (2) joint is a submission candidate iff it beats gate6 in BOTH folds AND its "
      "paired-bootstrap interval on the official score excludes zero. Offline the "
      "div_J gain was +0.0360 [+0.0163, +0.0597]; a division_jaccard gain of +0.036 "
      "is worth +0.0036 of score, so the interval to look for here is roughly "
      "[+0.0016, +0.0060] IF adj_J does not move against it.", flush=True)
print("  (3) READ adj_J SEPARATELY. div_J enters the score at 0.1 weight and adj_J at "
      "1.0. divbench3 and divbench4 both found adj_J RISES about +0.0010 under the "
      "cut, but that was a different cut keeping different edges. An arm that trades "
      "-0.002 of adj_J for +0.036 of div_J is a LOSS. Read both columns, never the "
      "sum alone.", flush=True)
print("  (4) THE CONTROLS ARE THE POINT, and they are read BEFORE the candidate. shuf "
      "is the same 67 feature rows with their labels destroyed and pad is 67 more "
      "rows drawn from the training fold itself. If either matches joint, then what "
      "44b6 adds is NOT label information -- it is regularisation or row count -- and "
      "joint is not a finding whatever its score does. Offline they read -0.0186 and "
      "-0.0280 of div_J against joint's +0.0360.", flush=True)
print("  (5) fork_gate_joint.pkl is fitted HERE, in the image that will load it, "
      "because local scikit-learn 1.9.0 reproduces the ranking but not the "
      "calibration (80% quantile 0.0238 local against 0.0275 on Kaggle) and the cut "
      "is a fixed 0.03. Compare the two quantile lines before shipping.", flush=True)''')

rep('DIVBENCH7 DONE in', 'DIVBENCH8 DONE in')
rep('print("DIVBENCH7 OK", flush=True)', 'print("DIVBENCH8 OK", flush=True)')
rep('f"[degraded] DIVBENCH7 INCOMPLETE:', 'f"[degraded] DIVBENCH8 INCOMPLETE:')
rep('"/kaggle/working/forks_f4_ref7.csv"', '"/kaggle/working/forks_f4_ref8.csv"')
rep('''for tag, cols, names_ in (("ref7_20", J20, FEAT20),):''',
    '''for tag, cols, names_ in (("ref8_20", J20, FEAT20),):''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "divbench8.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
