"""divadmit.py -> forkprop.py: the orphan-fork patch's PROPOSALS ranked by image contrast,
its caps and its divergence gate loosened, all priced under the IMAGE fork gate.

WHY (2026-09-17). divadmit: loosening the divergence gate (DIV_DIVERGE_UM 2.25 -> 1.5 / 0.75 / 0)
lost true divisions ungated (55 -> 54 / 46 / 40) because the extra proposals outrank the true ones
for the same parent or the same frame slot; the frame cap refuses 1,590 of REF's proposals and
3 lost divisions "passed all gates then capped or taken", 11 stop at the divergence gate. The
proposals are ranked by DISTANCE (d_pq + 0.15 d_s). forkimg showed that the raw-image CONTRAST at
the mother and the two daughters separates true forks from false ones (AUC 0.83-0.84 each; a
true fork's cells stand 3x over background, a false fork's 0.9x). So: rank the proposals by
contrast (the sum of the three contrasts, distance as the tie-break), let the bright ones take
the cap slots, loosen the cap and the divergence gate, and let the image gate do the pruning.

ARMS (post-link changes replayed on the pickled REF solve, as divadmit did; no admit pass):
  CR               contrast ranking, shipped caps, diverge 2.25
  CR_CAP2          contrast ranking, frame + global caps x2
  CR_CAP4          ... x4
  CR_CAP4_DV150    ... x4 and diverge 1.5
  DIST_CAP4_DV150  distance ranking, caps x4, diverge 1.5 (is it the ranking or the cap?)
  CR_NOCAP_DV150   contrast ranking, caps x100 (none), diverge 1.5
THE CROSS: REF x (gate6 / joint / gate6img / jointimg) and every arm x (joint / jointimg); jointimg =
the 44b6 forks WITH image columns (forks_44b6img.csv from the dataset biohub-forktables, 88 counted)
added to both training folds. Paired bootstrap of every arm + jointimg against REF + jointimg (the
gate that is being submitted) and of every arm + joint against REF + joint (what shipped).

Attach: divadmit's 16 sources + the dataset faustine649/biohub-forktables. CPU: pass 1 ~65 min +
image crops ~30 min + pass 2 six arms ~2 h + the cross ~1.5 h.
"""
import os

src = open("divadmit.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-divadmit"', 'SCRIPT_VERSION = "2026-09-17a-forkprop"')

# ---- arms --------------------------------------------------------------------------------------
rep('''DA_ARMS = [("DV150", dict(diverge=1.5, admit=False)),
           ("DV075", dict(diverge=0.75, admit=False)),
           ("DV000", dict(diverge=0.0, admit=False)),
           ("ADM", dict(diverge=2.25, admit=True)),
           ("ADM_DV075", dict(diverge=0.75, admit=True))]''',
    '''DA_ARMS = [("CR", dict(diverge=2.25, admit=False, rank="contrast", cap=1.0)),
           ("CR_CAP2", dict(diverge=2.25, admit=False, rank="contrast", cap=2.0)),
           ("CR_CAP4", dict(diverge=2.25, admit=False, rank="contrast", cap=4.0)),
           ("CR_CAP4_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=4.0)),
           ("DIST_CAP4_DV150", dict(diverge=1.5, admit=False, rank="dist", cap=4.0)),
           ("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0))]
FP_RANK = "dist"          # shipped: proposals by d_pq + 0.15 d_s; "contrast": by the image contrast, distance as tie-break
FP_CAP = 1.0              # multiplier on DIV_FRAME_CAP and DIV_GLOBAL_CAP in the orphan pass
FP_MEMO = {}              # (t, z, y, x) -> contrast, per movie
FP_STAT = defaultdict(int)''')

# ---- the feature list and the image block (forkimg's, verbatim) ---------------------------------
rep("FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order",
    'FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]\n'
    "FEAT_NAMES = FEAT20 + FEAT_NEW + FEAT_IMG      # the first 20 are divbench3's, in order; the last 11 read the image")
fi = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_forkimg.py"), encoding="utf-8").read()
BLOCK = fi[fi.index("BLOCK = r'''") + len("BLOCK = r'''"):fi.index("'''\nrep('\\ndef fork_rows(")]
assert "# ---- forkimg block ----" in BLOCK and "def fork_img_feats" in BLOCK
BLOCK = BLOCK.replace("# ---- end forkimg block ----", '''

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
# ---- end forkimg block ----''')
rep('\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n', BLOCK + '\n\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n')
rep('''                   steal=float(lg.get("steal", 0)))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''', '''                   steal=float(lg.get("steal", 0)))
        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''')

# ---- the orphan pass: ranking and caps ------------------------------------------------------------
rep('            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))',
    '''            if FP_RANK == "contrast":
                key_ = (-(fp_contrast(nodes, p) + fp_contrast(nodes, c1) + fp_contrast(nodes, q)), d_pq + DIV_W_SISTER * d_s)
            else:
                key_ = (d_pq + DIV_W_SISTER * d_s,)
            proposals.append((key_, t, p, q, c1, d_pq, d_s, dv_))''')
rep('''    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            PATCH_CAPS["orphan_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue''',
    '''    max_global = max(1, int(DIV_GLOBAL_CAP * FP_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            PATCH_CAPS["orphan_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * FP_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue''')

# ---- pass 1: the movie open around REF's judge (image features for REF's forks) --------------------
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

# ---- pass 2: the movie open per movie, the knobs per arm ----------------------------------------------
rep('''    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node, gt=None)
    cells = []
    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DA_ADMIT = cfg["admit"]
''', '''    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node, gt=None)
    try:
        IMG_CTX.clear()
        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
    except Exception as _e:                                       # noqa: BLE001
        warn_once("forkimg_open2", f"{name}: image not opened in pass 2 ({_e}); contrast -1, image features NaN")
        IMG_CTX.clear()
    FP_MEMO.clear()
    cells = []
    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DA_ADMIT = cfg["admit"]
        FP_RANK = cfg["rank"]
        FP_CAP = cfg["cap"]
''')
rep('''    DIV_DIVERGE_UM = 2.25
    DA_ADMIT = False
    DA_CTX.clear()
    print(f"  {name} fold{k} REF''', '''    DIV_DIVERGE_UM = 2.25
    DA_ADMIT = False
    FP_RANK = "dist"
    FP_CAP = 1.0
    DA_CTX.clear()
    IMG_CTX.clear()
    print(f"  {name} fold{k} REF''')
rep('''      f"each movie: " + ", ".join(f"{a_} (DIV_DIVERGE_UM {c_['diverge']}"
                                  f"{', admit dropped primaries' if c_['admit'] else ''})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)''',
    '''      f"each movie: " + ", ".join(f"{a_} (rank {c_['rank']}, caps x{c_['cap']:g}, DIV_DIVERGE_UM {c_['diverge']})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)''')

# ---- the 44b6 rows: the 28-column literal (joint anchor) and the 39-column table (jointimg) ---------------
rep("assert FEAT44 == FEAT_NAMES, \"the 44b6 fork table's columns are not this run's\"",
    "assert FEAT44 == FEAT20 + FEAT_NEW, \"the 44b6 fork table's columns are not this run's\"   # forkprop: image columns padded below")
rep('assert X44.shape == (67, len(FEAT_NAMES)), X44.shape',
    '''assert X44.shape == (67, len(FEAT20) + len(FEAT_NEW)), X44.shape
X44 = np.hstack([X44, np.full((len(X44), len(FEAT_IMG)), np.nan)])   # the joint anchor: no image columns
_f44 = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/forks_44b6img.csv")]
assert _f44, "attach the dataset faustine649/biohub-forktables (forks_44b6img.csv)"
import pandas as _pd                                              # noqa: E402
_t44 = _pd.read_csv(_f44[0])
assert list(_t44.columns[5:]) == FEAT_NAMES, list(_t44.columns[5:])
_t44 = _t44[_t44.label >= 1]
X44IMG = _t44[FEAT_NAMES].to_numpy(float)
Y44IMG = (_t44.label == 2).astype(int).to_numpy()
print(f"  jointimg: {len(X44IMG)} counted 44b6 forks with image columns from {_f44[0]} (TP {int(Y44IMG.sum())})", flush=True)''')

# ---- the gate curves and pickles --------------------------------------------------------------------
rep('''                         ("28 feats, 2 folds", list(range(len(FEAT_NAMES))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT_NAMES))), FOLDS3)):''',
    '''                         ("28 feats, 2 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS3),
                         ("20 feats + 11 image, 2 folds", J20 + JIMG, FOLDS2)):''')
rep('J20 = list(range(len(FEAT20)))\n',
    'J20 = list(range(len(FEAT20)))\n'
    'JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]\n'
    'print(f"\\n  forkprop: image features on {FI_STAT[\'forks\']:,} forks ({FI_STAT[\'nan_rows\']:,} unreadable, {FI_STAT[\'no_img\']:,} without an image), "\n'
    '      f"proposal contrast crops {FP_STAT[\'crops\']:,} ({FP_STAT[\'no_img\']:,} without an image)", flush=True)\n')

# ---- the cross: a cols parameter, the four gates -----------------------------------------------------
rep('def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None):',
    'def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None, cols=None):\n    cols = J20 if cols is None else cols')
rep('        Xtr_, ytr_ = Xa[tr][:, J20], (la[tr] == 2).astype(int)', '        Xtr_, ytr_ = Xa[tr][:, cols], (la[tr] == 2).astype(int)')
rep('            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, J20]])', '            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, cols]])')
rep('        pa[te] = mg.predict_proba(Xa[te][:, J20])[:, 1]', '        pa[te] = mg.predict_proba(Xa[te][:, cols])[:, 1]')
rep('''    for _gn, _add in (("gate6", None), ("joint", (X44, Y44))):
        CELLS.append((f"{_arm} + {_gn}", _arm, _gn, _fa, _ma, _pa, _add))''',
    '''    _gates = ((("gate6", None), ("joint", (X44, Y44)), ("gate6img", None), ("jointimg", (X44IMG, Y44IMG))) if _arm == "REF"
              else (("joint", (X44, Y44)), ("jointimg", (X44IMG, Y44IMG))))
    for _gn, _add in _gates:
        CELLS.append((f"{_arm} + {_gn}", _arm, _gn, _fa, _ma, _pa, _add))
COLS_OF = {c_[0]: (J20 + JIMG if c_[2].endswith("img") else J20) for c_ in CELLS}''')
rep('''    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add)''',
    '''    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add, cols=COLS_OF[lab])''')

# ---- the bootstrap against the image gate ---------------------------------------------------------------
rep('''for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"divadmit_arm_{a_}.csv"''',
    '''print(f"\\n  PAIRED BOOTSTRAP of every arm + jointimg AGAINST REF + jointimg (the image gate under submission), "
      f"score and adj_J ===", flush=True)
for lab_ in ["REF + jointimg"] + [f"{a_} + jointimg" for a_, _c in BR_ARMS_RUN]:
    if lab_ not in BS or "REF + joint" not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[lab_] - bank["REF + jointimg"] if lab_ != "REF + jointimg" else bank[lab_] - bank["REF + joint"]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        ref_ = "REF + joint" if lab_ == "REF + jointimg" else "REF + jointimg"
        print(f"    {lab_:<26} vs {ref_:<15} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "
      "DIST_CAP4_DV150 control says whether the ranking or the cap did it.", flush=True)

for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"divadmit_arm_{a_}.csv"''')

# ---- names ---------------------------------------------------------------------------------------------------
rep('f"divadmit_arm_{a_}.csv"', 'f"forkprop_arm_{a_}.csv"')
rep('f"divadmit_REF_{DET_SRC}.csv"', 'f"forkprop_REF_{DET_SRC}.csv"')
rep('print("DIVADMIT OK", flush=True)', 'print("FORKPROP OK", flush=True)')

assert src.count("fork_img_feats(") == 2 and src.count("fp_contrast(nodes, p)") == 1 and src.count("COLS_OF[lab]") == 1
assert src.count("FP_CAP * ") == 2 and src.count("X44IMG") >= 3
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "forkprop.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written forkprop.py", len(src))
