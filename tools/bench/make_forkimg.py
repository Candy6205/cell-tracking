"""dupsup.py -> forkimg.py: RAW-IMAGE features for the fork gate.

WHY (2026-09-17). What ships is REF + the joint fork gate: divisions 49 TP / 51 FP / 76 FN,
div_J 0.2784 (0.028 of the score); a perfect gate on REF's forks would read 55/0/70 = 0.44,
+0.016 of score. The gate's 20 features are geometry, ranker probabilities, detection scores,
chain lengths and density -- nothing reads the IMAGE. In H2B-labelled embryos a dividing
nucleus condenses (brighter, more compact) in its last frame and the two daughters are twins
(same brightness, same size, both compact) while a false fork joins the mother to an
unrelated neighbour of ordinary brightness and size. Earlier mass tests (div_mass, div intmass,
2026-08-25) read Zebrahub MASK volumes (AUC 0.61); nobody has read the raw intensity at the
mother and the daughters of the competition's own forks.

WHAT. dupsup's harness, REF only (no arms). fork_rows gets FEAT_IMG per fork from the movie's
zarr: a smoothed crop (sigma ~1 um) of half-size HALF_UM at each of the mother p (frame t),
her predecessor (t-1), and the daughters c1 and q (t+1):
  i_p      mother peak (max of the smoothed crop within PEAK_UM of the node)
  c_p      mother contrast (peak - crop median) / crop median
  r_p      mother radius of gyration (um) of the background-subtracted signal within SIZE_UM
  br_p     mother peak / predecessor peak (brightening into the last mother frame)
  c_c1 c_q r_c1 r_q   the same contrast and radius for the two daughters
  sym_i    |peak_c1 - peak_q| / (peak_c1 + peak_q)     twins are 0
  sym_r    |r_c1 - r_q| / (r_c1 + r_q)
  dau_ratio (peak_c1 + peak_q) / (2 peak_p)
Then the harness's own tables: per-feature AUC TP-vs-FP on the counted forks, the out-of-fold
gate curves with and without FEAT_IMG, and THE CROSS with two new cells, REF + gate6img and
REF + jointimg (the 67 44b6 forks enter with NaN image columns; HistGB takes NaN natively),
against REF + gate6 / REF + joint, with the paired bootstrap. Writes fork_gate_img.pkl and
fork_gate_jointimg.pkl (20 + 11 features) for a submission variant if the cross says yes.

Attach: dupsup's 16 sources. CPU: features ~1 h + REF ~70 min + ~27k crops ~70 min + cross ~1 h.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-17a-forkimg"')
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = []             # forkimg: REF only')

# ---- the feature list -----------------------------------------------------------------
rep("FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order",
    'FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]\n'
    "FEAT_NAMES = FEAT20 + FEAT_NEW + FEAT_IMG      # the first 20 are divbench3's, in order; the last 11 read the image")

BLOCK = r'''
# ---- forkimg block ----
from scipy import ndimage as _ndi                                # noqa: E402
from viz_cells import open_image as _open_image                  # noqa: E402
FI_HALF_UM = 8.0
FI_PEAK_UM = 2.0
FI_SIZE_UM = 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)
IMG_CTX = {}                       # name, img (set around REF's _judge), frame cache
FI_STAT = defaultdict(int)


def _fi_crop(img, t, xyz_vox):
    """smoothed crop around a node, its offset; None when the frame is out of range."""
    if t < 0 or t >= img.shape[0]:
        return None, None
    hz = int(np.ceil(FI_HALF_UM / SCALE[0]))
    hy = int(np.ceil(FI_HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    z1, y1, x1 = z + hz + 1, y + hy + 1, x + hy + 1
    try:
        crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1])
    except Exception:                                            # noqa: BLE001
        fr = IMG_CTX.get("frames", {})
        if t not in fr:
            fr.clear()
            fr[t] = np.asarray(img[t])
            IMG_CTX["frames"] = fr
        crop = fr[t][z0:z1, y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    sm = _ndi.gaussian_filter(crop.astype(np.float32), FI_SIGMA_VOX)
    return sm, np.array([z0, y0, x0], float)


def _fi_node(img, t, xyz_vox):
    """(peak, contrast, radius of gyration um) of one node; NaNs when unreadable."""
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return np.nan, np.nan, np.nan
    c = np.asarray(xyz_vox, float) - off
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    d = np.linalg.norm((idx - c) * SCALE, axis=1)
    v = sm.reshape(-1)
    near = d <= FI_PEAK_UM
    if not near.any():
        return np.nan, np.nan, np.nan
    bg = float(np.median(v))
    j = int(np.argmax(np.where(near, v, -np.inf)))
    peak = float(v[j])
    con = (peak - bg) / max(bg, 1e-6)
    dp = np.linalg.norm((idx - idx[j]) * SCALE, axis=1)
    m = dp <= FI_SIZE_UM
    w = np.clip(v[m] - bg, 0, None)
    rg = float(np.sqrt((w * dp[m] ** 2).sum() / w.sum())) if w.sum() > 0 else np.nan
    return peak, float(con), rg


def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}
    img = IMG_CTX.get("img")
    if img is None or IMG_CTX.get("name") != name:
        FI_STAT["no_img"] += 1
        return out
    t = int(nodes[p]["t"])
    i_p, c_p, r_p = _fi_node(img, t, nodes[p]["xyz"])
    pr = pred.get(p, [])
    if len(pr) == 1:
        i_pm, _c, _r = _fi_node(img, t - 1, nodes[pr[0]]["xyz"])
    else:
        i_pm = np.nan
    i_1, c_1, r_1 = _fi_node(img, t + 1, nodes[c1]["xyz"])
    i_q, c_q, r_q = _fi_node(img, t + 1, nodes[q]["xyz"])
    out.update(i_p=i_p, c_p=c_p, r_p=r_p,
               br_p=(i_p / i_pm) if (np.isfinite(i_pm) and i_pm > 0) else np.nan,
               c_c1=c_1, c_q=c_q, r_c1=r_1, r_q=r_q,
               sym_i=(abs(i_1 - i_q) / (i_1 + i_q)) if (np.isfinite(i_1) and np.isfinite(i_q) and (i_1 + i_q) > 0) else np.nan,
               sym_r=(abs(r_1 - r_q) / (r_1 + r_q)) if (np.isfinite(r_1) and np.isfinite(r_q) and (r_1 + r_q) > 0) else np.nan,
               dau_ratio=((i_1 + i_q) / (2 * i_p)) if (np.isfinite(i_1) and np.isfinite(i_q) and np.isfinite(i_p) and i_p > 0) else np.nan)
    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out
# ---- end forkimg block ----
'''
rep('\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n', BLOCK + '\n\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n')
rep('''                   steal=float(lg.get("steal", 0)))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''', '''                   steal=float(lg.get("steal", 0)))
        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''')

# ---- open the movie around REF's judge ---------------------------------------------------
rep('''    row, ds, fr = _judge(name, k, outp, seeds, aux)
''', '''    _tfi = time.time()
    try:
        IMG_CTX.clear()
        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
    except Exception as _e:                                       # noqa: BLE001
        warn_once("forkimg_open", f"{name}: image not opened ({_e}); image features NaN")
        IMG_CTX.clear()
    row, ds, fr = _judge(name, k, outp, seeds, aux)
    IMG_CTX.clear()
    FI_STAT["seconds"] += int(time.time() - _tfi)
''')

# ---- the 44b6 rows: the column check against the 28 harness features, then NaN image columns ----
rep("assert FEAT44 == FEAT_NAMES, \"the 44b6 fork table's columns are not this run's\"",
    "assert FEAT44 == FEAT20 + FEAT_NEW, \"the 44b6 fork table's columns are not this run's\"   # forkimg: image columns padded below")
rep('assert X44.shape == (67, len(FEAT_NAMES)), X44.shape',
    'assert X44.shape == (67, len(FEAT20) + len(FEAT_NEW)), X44.shape\n'
    'X44 = np.hstack([X44, np.full((len(X44), len(FEAT_IMG)), np.nan)])   # forkimg: no image read for the 44b6 forks')

# ---- gate curves with the image features ------------------------------------------------------
rep('''                         ("28 feats, 2 folds", list(range(len(FEAT_NAMES))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT_NAMES))), FOLDS3)):''',
    '''                         ("28 feats, 2 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS3),
                         ("20 feats + 11 image, 2 folds", J20 + JIMG, FOLDS2),
                         ("20 feats + 11 image, 3 folds", J20 + JIMG, FOLDS3),
                         ("11 image alone, 2 folds", JIMG, FOLDS2)):''')
rep('J20 = list(range(len(FEAT20)))\n',
    'J20 = list(range(len(FEAT20)))\n'
    'JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]\n'
    'print(f"\\n  forkimg: image features on {FI_STAT[\'forks\']:,} forks, {FI_STAT[\'nan_rows\']:,} unreadable, "\n'
    '      f"{FI_STAT[\'no_img\']:,} without an image handle, {FI_STAT[\'seconds\'] / 60:.0f} min of crops", flush=True)\n')

# ---- production pickles: 20 + image (6bba) and joint ---------------------------------------
rep('for tag, cols, names_ in (("dw55_20", J20, FEAT20),):',
    'for tag, cols, names_ in (("dw55_20", J20, FEAT20), ("img_20", J20 + JIMG, FEAT20 + FEAT_IMG)):')

# ---- the cross: a cols parameter, two more cells --------------------------------------------
rep('def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None):',
    'def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None, cols=None):\n    cols = J20 if cols is None else cols')
rep('        Xtr_, ytr_ = Xa[tr][:, J20], (la[tr] == 2).astype(int)', '        Xtr_, ytr_ = Xa[tr][:, cols], (la[tr] == 2).astype(int)')
rep('            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, J20]])', '            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, cols]])')
rep('        pa[te] = mg.predict_proba(Xa[te][:, J20])[:, 1]', '        pa[te] = mg.predict_proba(Xa[te][:, cols])[:, 1]')
rep('''WANT = {"REF + gate6": "want 45/53/80 0.2528 0.9155 0.9408 (hparabench round 2)",''',
    '''CELLS.append(("REF + gate6img", "REF", "gate6img", FORKS, MOVIE, PER, None))
CELLS.append(("REF + jointimg", "REF", "jointimg", FORKS, MOVIE, PER, (X44, Y44)))
COLS_OF = {"REF + gate6img": J20 + JIMG, "REF + jointimg": J20 + JIMG}
WANT = {"REF + gate6": "want 45/53/80 0.2528 0.9155 0.9408 (hparabench round 2)",''')
rep('''    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add)''',
    '''    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add, cols=COLS_OF.get(lab))''')

# ---- the bootstrap: image gate against what ships -------------------------------------------
rep('''for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"dupsup_arm_{a_}.csv"''',
    '''print("\\n=== FORKIMG: the image gate against what ships (paired bootstrap, same 2,000 draws) ===", flush=True)
for lab_, ref_ in (("REF + gate6img", "REF + gate6"), ("REF + jointimg", "REF + joint")):
    if lab_ not in BS or ref_ not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA), ("edge_J", _BSE)):
        d = bank[lab_] - bank[ref_]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {lab_:<16} vs {ref_:<12} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: div_J moves 1/176 per fork (+0.0006 of score per true division kept, -0.00016 per false fork "
      "removed); an interval that excludes zero on 'score' is the only thing that carries to a submission.", flush=True)
# the joint image gate on every counted fork of both domains, for a submission variant
_mj = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
_Xj = np.vstack([X_f[counted][:, J20 + JIMG], X44[:, J20 + JIMG]])
_yj = np.concatenate([(lab[counted] == 2).astype(int), Y44])
_mj.fit(_Xj, _yj)
with open("/kaggle/working/fork_gate_jointimg.pkl", "wb") as fh:
    pickle.dump({"model": _mj, "feature_names": FEAT20 + FEAT_IMG, "version": SCRIPT_VERSION,
                 "trained_on": "counted F4 forks, 128 6bba + 67 44b6 (44b6 image columns NaN), TP=1 FP=0",
                 "n_train": {"6bba": int(counted.sum()), "44b6": int(len(Y44)), "tp": int(_yj.sum())},
                 "image": dict(half_um=FI_HALF_UM, peak_um=FI_PEAK_UM, size_um=FI_SIZE_UM, sigma_vox=FI_SIGMA_VOX),
                 "detector": "F4", "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}, fh)
print(f"  fork_gate_jointimg.pkl written (n_train 6bba {int(counted.sum())} / 44b6 {len(Y44)} / tp {int(_yj.sum())})", flush=True)

for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"dupsup_arm_{a_}.csv"''')

# ---- names ---------------------------------------------------------------------------------------
rep('print("DUPSUP OK", flush=True)', 'print("FORKIMG OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] FORKIMG INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"forkimg_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"forkimg_arm_{a_}.csv"')

assert src.count("fork_img_feats(") == 2 and src.count("COLS_OF.get(lab)") == 1 and src.count("JIMG") >= 6
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "forkimg.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written forkimg.py", len(src))
