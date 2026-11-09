"""mergeveto.py -- lineage context and detection-side vetoes for mergefeat's merge scores.

SCRIPT_VERSION 2026-09-16a-mergeveto

QUESTION. mergefeat scores every hypcache parent (two primaries within 8.5 um) with an
out-of-fold probability p that the two are ONE cell. Its firing table says how many
two-cell pairs (lab 0) and one-end-annotated pairs (lab 2) a threshold tau also merges.
This kernel asks WHERE those wrong firings sit, and whether a rule the pipeline can
evaluate without ground truth removes them:

  ground-truth context of each scored pair (judge only, never a feature):
    sisters      the two matched GT nodes descend from the same division (lab 0 only)
    age_gt       frames since the birth (division or track start) of the matched cell(s)
    to_div_gt    frames until the matched cell divides (inf if it never does)
  detection-side context (usable at inference; from the hypcache children only):
    n_m1, n_m2, n_p1, n_p2   primaries within R_MID um of the pair's midpoint at t-1, t-2, t+1, t+2
    age_det      consecutive earlier frames (up to 5) in which BOTH a and b had distinct counterparts
                 within 4 um (0 = the pair configuration is new this frame)
    fut_det      the same forward
    div_sig      n_m1 == 1 and age_det == 0 (the detections themselves show a split at t)

REPORTS. (1) composition of lab 0 / lab 1 / lab 2 firings at each tau by sisters, age_gt,
to_div_gt; (2) the firing table of mergefeat's p with each candidate veto applied
(recall on lab 1, fires on lab 0, lab 2, unannotated, wrong per fix); (3) a refit of the
classifier with the detection-side context appended to mergefeat's features (from
mergefeat_pairs.csv; same two folds by movie), out of fold, with its own firing table.

OUTPUT. mergeveto_<name>.npz (t, q, p, lab, n_m1, n_m2, n_p1, n_p2, age_det, fut_det, and
p2 = the refit's out-of-fold probability where features exist, else p) for every 6bba
movie; mergeveto_context.csv (labelled rows with the GT context) for offline reading.

Attach: faustine649/cell-tracking (code), biohub-hypcache, biohub-mergefeat, the
competition. CPU, well under an hour.
"""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-16a-mergeveto"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 5.0
UNET_THRESH = 0.15
R_MID = 5.0          # um around the midpoint, for the neighbouring-frame counts
PAIR_UM = 4.0        # a counterpart of a or b in another frame must lie within this
AGE_MAX = 5
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
if code is not None:
    sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff  # noqa: E402

ROOT = find_root(None)


def walk(prefix):
    out = {}
    for d in pathlib.Path("/kaggle/input").glob(f"*/**/{prefix}*.npz"):
        out[d.name[len(prefix):-len(".npz")]] = d
    return out


HYP = walk("fdet_F4hyp_")
MF = walk("mergefeat_")
SIX = sorted(n for n in MF if n.startswith("6bba_"))
assert SIX and all(n in HYP for n in SIX), "a movie lacks a hypcache cache"
ALL6 = sorted(n for n in HYP if n.startswith("6bba_"))
FOLD_OF = {n: k % 2 for k, n in enumerate(ALL6)}          # mergefeat's folds: sorted index over the 128 movies
PAIRS_CSV = next(iter(pathlib.Path("/kaggle/input").glob("*/**/mergefeat_pairs.csv")), None)
print(f"hypcache {len(HYP)}, mergefeat {len(MF)}; movies {len(SIX)}; pairs csv {PAIRS_CSV}", flush=True)
OUT = pathlib.Path("/kaggle/working")


# ---- ground truth with lineage ---------------------------------------------------------------
def gt_lineage(name):
    """Per GT node (row order of the geff arrays): t, position (um), birth row, end row,
    whether the segment ends in a division, the division node the segment descends from
    (-1 if none)."""
    gt = read_geff(geff_path(ROOT, "train", name))
    ids = np.asarray(gt["node_ids"], np.int64)
    g_t = np.asarray(gt["t"], np.int64)
    pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1) * SCALE
    N = len(ids)
    idx = {int(i): k for k, i in enumerate(ids)}
    edges = np.asarray(gt["edges"], np.int64).reshape(-1, 2)
    if len(edges):
        s = np.array([idx[int(a)] for a in edges[:, 0]])
        d = np.array([idx[int(b)] for b in edges[:, 1]])
        if (g_t[s] > g_t[d]).sum() > (g_t[s] < g_t[d]).sum():   # edges stored child -> parent
            s, d = d, s
    else:
        s = d = np.zeros(0, np.int64)
    pred = np.full(N, -1, np.int64)
    nchild = np.zeros(N, np.int64)
    pred[d] = s
    np.add.at(nchild, s, 1)
    child1 = np.full(N, -1, np.int64)
    one = nchild[s] == 1
    child1[s[one]] = d[one]
    order = np.argsort(g_t, kind="stable")
    birth = np.arange(N)
    for k in order:
        p = pred[k]
        if p >= 0 and nchild[p] == 1:
            birth[k] = birth[p]
    end = np.arange(N)
    for k in order[::-1]:
        if nchild[k] == 1:
            end[k] = end[child1[k]]
    div_from = np.where(pred[birth] >= 0, pred[birth], -1)
    ends_div = nchild[end] >= 2
    trees = {}
    for tt in np.unique(g_t):
        rows = np.where(g_t == tt)[0]
        trees[int(tt)] = (rows, cKDTree(pos[rows]))
    return dict(t=g_t, birth=birth, end=end, ends_div=ends_div, div_from=div_from, trees=trees,
                n_div=int((nchild >= 2).sum()))


def gt_match(trees, t, P_um):
    out = np.full(len(P_um), -1, np.int64)
    if t in trees and len(P_um):
        rows, tree = trees[t]
        dd, gi = tree.query(P_um)
        out[dd <= MATCH_UM] = rows[gi[dd <= MATCH_UM]]
    return out


CTX_NAMES = ["n_m1", "n_m2", "n_p1", "n_p2", "age_det", "fut_det"]


def pair_context(t, T, tree, A, B):
    """Detection-side context of the pairs (A, B) at frame t: primaries within R_MID of the
    midpoint at t-1, t-2, t+1, t+2; consecutive earlier / later frames (up to AGE_MAX) in
    which both a and b have distinct counterparts within PAIR_UM."""
    ctx = np.zeros((len(A), len(CTX_NAMES)), np.float32)
    mid = 0.5 * (A + B)
    for col, dt in ((0, -1), (1, -2), (2, 1), (3, 2)):
        if 0 <= t + dt < T and tree[t + dt] is not None:
            ctx[:, col] = tree[t + dt].query_ball_point(mid, R_MID, return_length=True)
    for col, sign in ((4, -1), (5, 1)):
        alive = np.ones(len(A), bool)
        for k in range(1, AGE_MAX + 1):
            tt = t + sign * k
            if not (0 <= tt < T) or tree[tt] is None:
                break
            d1, i1 = tree[tt].query(A)
            d2, i2 = tree[tt].query(B)
            alive &= (d1 <= PAIR_UM) & (d2 <= PAIR_UM) & (i1 != i2)
            ctx[:, col] += alive
    return ctx


# ---- per movie -------------------------------------------------------------------------------
ROWS = {}
t_all = time.time()
for k_, name in enumerate(SIX):
    z = np.load(HYP[name])
    zm = np.load(MF[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    ch, par = [], []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], float)
        s = np.asarray(z[f"s{t}"], float)
        p = np.asarray(z[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in z.files else np.zeros((0, 2), np.int64)
        prim = c[s >= UNET_THRESH]
        ch.append(prim[:len(prim) - len(p)] * SCALE)
        par.append(p)
    tree = [cKDTree(c) if len(c) else None for c in ch]
    L = gt_lineage(name)
    gm = [gt_match(L["trees"], t, ch[t]) for t in range(T)]
    mt = np.asarray(zm["t"], np.int64)
    mq = np.asarray(zm["q"], np.int64)
    mp = np.asarray(zm["p"], np.float32)
    ml = np.asarray(zm["lab"], np.int64)
    n = len(mt)
    ctx = np.zeros((n, len(CTX_NAMES)), np.float32)
    sis = np.zeros(n, np.int8)
    age = np.full(n, -1, np.int32)
    todiv = np.full(n, -1, np.int32)
    lab_chk = np.full(n, -9, np.int64)
    for t in np.unique(mt):
        rows = np.where(mt == t)[0]
        p = par[t]
        ai, bi = p[mq[rows], 0], p[mq[rows], 1]
        A, B = ch[t][ai], ch[t][bi]
        ctx[rows] = pair_context(t, T, tree, A, B)
        ga, gb = gm[t][ai], gm[t][bi]
        lab_chk[rows] = np.where((ga >= 0) & (ga == gb), 1, np.where((ga >= 0) & (gb >= 0), 0, np.where((ga >= 0) | (gb >= 0), 2, -1)))
        ok = (ga >= 0) & (gb >= 0)
        both = ok & (ga != gb)
        sis[rows[both]] = (L["div_from"][ga[both]] >= 0) & (L["div_from"][ga[both]] == L["div_from"][gb[both]])
        g1 = np.where(ga >= 0, ga, gb)          # the annotated end (either for lab 1 / 2, a for lab 0)
        has = g1 >= 0
        a_age = t - L["t"][L["birth"][g1[has]]]
        a_div = np.where(L["ends_div"][g1[has]], L["t"][L["end"][g1[has]]] - t, 9999)
        age[rows[has]] = a_age
        todiv[rows[has]] = a_div
        if both.any():
            # two cells: the younger of the two
            b_age = t - L["t"][L["birth"][gb[both]]]
            age[rows[both]] = np.minimum(age[rows[both]], b_age)
    assert np.array_equal(lab_chk, ml), (name, "labels differ from mergefeat's")
    ROWS[name] = dict(t=mt, q=mq, p=mp, lab=ml, ctx=ctx, sis=sis, age=age, todiv=todiv)
    print(f"[{k_ + 1}/{len(SIX)}] {name}: rows {n:,} (lab 1 {(ml == 1).sum()}, lab 0 {(ml == 0).sum()}, "
          f"sisters {int(sis.sum())}, lab 2 {(ml == 2).sum()}); GT divisions {L['n_div']}  {time.time() - t_all:.0f}s", flush=True)

names = sorted(ROWS)
P = np.concatenate([ROWS[n]["p"] for n in names])
LAB = np.concatenate([ROWS[n]["lab"] for n in names])
CTX = np.concatenate([ROWS[n]["ctx"] for n in names])
SIS = np.concatenate([ROWS[n]["sis"] for n in names])
AGE = np.concatenate([ROWS[n]["age"] for n in names])
TODIV = np.concatenate([ROWS[n]["todiv"] for n in names])
print(f"\nrows {len(P):,}: lab 1 {(LAB == 1).sum():,}, lab 0 {(LAB == 0).sum():,} (sisters {int(SIS.sum()):,}), "
      f"lab 2 {(LAB == 2).sum():,}, unannotated {(LAB == -1).sum():,}", flush=True)


def fire_table(prob, veto, title):
    """veto: bool array, True = never merge. Prints recall on lab 1, fires on lab 0 / 2 / -1, wrong per fix."""
    print(f"\n=== {title} ===", flush=True)
    print(f"  {'tau':>5} {'recall 1cell':>13} {'fires 2cell':>12} {'fires 1-ann':>12} {'fires unann':>12} "
          f"{'n fixes':>8} {'n wrong':>8} {'wrong / fix':>12}", flush=True)
    for tau in TAUS:
        f = (prob >= tau) & ~veto
        r1 = f[LAB == 1].mean()
        f0 = f[LAB == 0].mean()
        f2 = f[LAB == 2].mean()
        fu = f[LAB == -1].mean()
        nfix = int(f[LAB == 1].sum())
        nwr = int(f[LAB == 0].sum() + f[LAB == 2].sum())
        print(f"  {tau:>5.2f} {r1:>13.3f} {f0:>12.3f} {f2:>12.3f} {fu:>12.3f} {nfix:>8} {nwr:>8} {nwr / max(1, nfix):>12.2f}", flush=True)


# (1) where the firings sit
print("\n=== composition of firings by ground-truth context (judge only) ===", flush=True)
for tau in TAUS:
    f = P >= tau
    for lab, nm in ((0, "two-cell"), (1, "one-cell"), (2, "one-annotated")):
        m = f & (LAB == lab)
        if m.sum() == 0:
            print(f"  tau {tau:.2f} {nm:<14} fires 0", flush=True)
            continue
        a = AGE[m]
        d = TODIV[m]
        line = (f"  tau {tau:.2f} {nm:<14} fires {int(m.sum()):>6}: age<=1 {np.mean(a <= 1):.2f} age<=3 {np.mean(a <= 3):.2f} "
                f"age<=6 {np.mean(a <= 6):.2f}; to_div<=1 {np.mean(d <= 1):.2f} to_div<=3 {np.mean(d <= 3):.2f}")
        if lab == 0:
            line += f"; sisters {np.mean(SIS[m] > 0):.2f}"
        base = LAB == lab
        line += (f"   [all {nm}: age<=3 {np.mean(AGE[base] <= 3):.2f}, to_div<=3 {np.mean(TODIV[base] <= 3):.2f}"
                 + (f", sisters {np.mean(SIS[base] > 0):.2f}" if lab == 0 else "") + "]")
        print(line, flush=True)

# (2) the detection-side vetoes on mergefeat's p
n_m1, n_m2, n_p1, n_p2, age_det, fut_det = (CTX[:, k] for k in range(6))
print("\n=== detection-side context by label (medians / rates) ===", flush=True)
for lab, nm in ((1, "one-cell"), (0, "two-cell"), (2, "one-annotated"), (-1, "unannotated")):
    m = LAB == lab
    print(f"  {nm:<14} n_m1==1 {np.mean(n_m1[m] == 1):.2f} n_m1>=2 {np.mean(n_m1[m] >= 2):.2f} n_p1>=2 {np.mean(n_p1[m] >= 2):.2f} "
          f"age_det==0 {np.mean(age_det[m] == 0):.2f} age_det>=3 {np.mean(age_det[m] >= 3):.2f} "
          f"fut_det==0 {np.mean(fut_det[m] == 0):.2f} fut_det>=3 {np.mean(fut_det[m] >= 3):.2f} "
          f"div_sig {np.mean((n_m1[m] == 1) & (age_det[m] == 0)):.2f}", flush=True)
NOVETO = np.zeros(len(P), bool)
fire_table(P, NOVETO, "mergefeat p, no veto (reference)")
VETOES = {
    "div_sig (n_m1==1 & age_det==0)": (n_m1 == 1) & (age_det == 0),
    "age_det==0": age_det == 0,
    "age_det<=1": age_det <= 1,
    "age_det<=1 | fut_det<=1": (age_det <= 1) | (fut_det <= 1),
    "n_m1>=2 (two things there before)": n_m1 >= 2,
    "n_m1>=2 | n_p1>=2": (n_m1 >= 2) | (n_p1 >= 2),
    "age_det>=3 & fut_det>=3 (pair persists both ways)": (age_det >= 3) & (fut_det >= 3),
}
for nm, v in VETOES.items():
    fire_table(P, v, f"veto: {nm}   (vetoes {v.mean():.1%} of all rows, {v[LAB == 1].mean():.1%} of one-cell)")

# (3) the refit with the context appended, from mergefeat_pairs.csv (labelled rows)
P2 = {n: ROWS[n]["p"].copy() for n in names}
if PAIRS_CSV is not None:
    import csv
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    key_of = {}
    for n in names:
        r = ROWS[n]
        for k in range(len(r["t"])):
            key_of[(n, int(r["t"][k]), int(r["q"][k]))] = k
    feats = defaultdict(list)          # name -> rows (index k, feature vector)
    with open(PAIRS_CSV, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        fcols = hdr[6:]
        for row in rd:
            key = (row[0], int(row[2]), int(row[3]))
            if key in key_of:
                feats[row[0]].append((key_of[key], np.array([float(v) for v in row[6:]], np.float32)))
    print(f"\n=== refit: mergefeat's {len(fcols)} features + {CTX_NAMES} on the labelled rows ===", flush=True)
    XF, YF, KF, NF = {}, {}, {}, {}
    for n in names:
        if not feats[n]:
            continue
        ks = np.array([k for k, _ in feats[n]])
        X = np.stack([v for _, v in feats[n]])
        X = np.concatenate([X, ROWS[n]["ctx"][ks]], axis=1)
        XF[n], KF[n], YF[n] = X, ks, ROWS[n]["lab"][ks]
    aucs = {}
    for fold in (0, 1):
        tr = [n for n in XF if FOLD_OF[n] != fold]
        Xt = np.concatenate([XF[n][np.isin(YF[n], (0, 1))] for n in tr])
        yt = np.concatenate([YF[n][np.isin(YF[n], (0, 1))] for n in tr]).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        m.fit(np.where(np.isfinite(Xt), Xt, -1.0), yt)
        for n in XF:
            if FOLD_OF[n] == fold:
                pr = m.predict_proba(np.where(np.isfinite(XF[n]), XF[n], -1.0))[:, 1].astype(np.float32)
                P2[n][KF[n]] = pr
        print(f"  fold{fold}: {len(tr)} movies, {len(yt):,} labelled rows", flush=True)
    Pn = np.concatenate([P2[n] for n in names])
    lab01 = np.isin(LAB, (0, 1))
    print(f"  out-of-fold AUC one-cell vs two-cell: mergefeat {roc_auc_score(LAB[lab01], P[lab01]):.4f} -> "
          f"refit {roc_auc_score(LAB[lab01], Pn[lab01]):.4f}", flush=True)
    # context-only stack, for the record
    for fold in (0, 1):
        tr = [n for n in XF if FOLD_OF[n] != fold]
        Xt = np.concatenate([ROWS[n]["ctx"][KF[n]][np.isin(YF[n], (0, 1))] for n in tr])
        yt = np.concatenate([YF[n][np.isin(YF[n], (0, 1))] for n in tr]).astype(int)
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xt, yt)
        aucs[fold] = [(n, roc_auc_score(YF[n][np.isin(YF[n], (0, 1))], m.predict_proba(ROWS[n]["ctx"][KF[n]][np.isin(YF[n], (0, 1))])[:, 1]))
                      for n in XF if FOLD_OF[n] == fold and len(np.unique(YF[n][np.isin(YF[n], (0, 1))])) == 2]
    ctx_auc = np.mean([a for fold in aucs for _, a in aucs[fold]])
    print(f"  context-only stack, mean per-movie out-of-fold AUC: {ctx_auc:.3f}", flush=True)
    fire_table(Pn, NOVETO, "refit p2 (features + context), no veto; the unannotated column is mergefeat's p there")
    fire_table(Pn, VETOES["div_sig (n_m1==1 & age_det==0)"], "refit p2 + veto div_sig")
else:
    print("\nno mergefeat_pairs.csv attached: refit skipped", flush=True)

# ---- outputs ---------------------------------------------------------------------------------
for n in names:
    r = ROWS[n]
    np.savez_compressed(OUT / f"mergeveto_{n}.npz", t=r["t"], q=r["q"], p=r["p"], p2=P2[n], lab=r["lab"],
                        **{nm: r["ctx"][:, k] for k, nm in enumerate(CTX_NAMES)})
with open(OUT / "mergeveto_context.csv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("dataset,t,q,lab,p,p2,sisters,age_gt,to_div_gt," + ",".join(CTX_NAMES) + "\n")
    for n in names:
        r = ROWS[n]
        for k in np.where(r["lab"] >= 0)[0]:
            fh.write(f"{n},{int(r['t'][k])},{int(r['q'][k])},{int(r['lab'][k])},{float(r['p'][k]):.5f},{float(P2[n][k]):.5f},"
                     f"{int(r['sis'][k])},{int(r['age'][k])},{int(r['todiv'][k])},"
                     + ",".join(f"{float(v):g}" for v in r["ctx"][k]) + "\n")
print(f"\nMERGEVETO DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("MERGEVETO OK", flush=True)
