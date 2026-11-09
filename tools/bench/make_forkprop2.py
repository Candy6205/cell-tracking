"""forkprop.py -> forkprop2.py: the uncapped orphan patch under the image gate, swept where it can still gain.

WHY (2026-09-18). forkprop: with no cap and DIV_DIVERGE_UM 1.5 the orphan patch under the joint image gate is
+0.0050 [+0.0012, +0.0089] over REF + jointimg (61/36/64 against 55/42/70); the cap, not the ranking, was the
binding gate, and the image gate is what makes the flood of proposals affordable. Two knobs were left at their
first setting: the gate's cut (p > 0.03 is divbench3's, chosen for a gate whose p ran to 0.36 at the 90th
percentile; this gate's 90th percentile is 0.012, so 0.03 may sit anywhere on its curve), and the patch's
geometry (divergence 1.5; parent 12 / sister 15 / child 10 um), each refusing a handful of the 64 remaining
lost divisions (8b_diverge 11 at 2.25, 5c 2, 7 1). fk44prop also produced the arm's 44b6 forks, so the gate's
44b6 half can now be the SAME population as its 6bba half.

WHAT. forkprop's harness (128 6bba movies, REF solved once and pickled, arms replayed post-link, official
scorer, hard anchors) with:
  * arms (all contrast-ranked, caps x100): NOCAP_DV150 (forkprop's winner, the anchor), NOCAP_DV100 (diverge
    1.0), NOCAP_DV000 (diverge 0: only converging pairs refused), NOCAP_DV150_GEO (diverge 1.5, parent 15 /
    sister 20 / child 12 um);
  * gates: jointimg = 6bba arm forks (other fold) + fk44img's 88 REF 44b6 forks (forkprop's gate, the anchor);
    jointimg44 = the same + fk44prop's 207 44b6 forks of the NOCAP_DV150 patch instead (the population the
    shipped fork_gate_prop.pkl was fitted on);
  * cells: REF + joint (want 0.9433), REF + jointimg (want 0.9484), NOCAP_DV150 + jointimg (want 0.9534),
    every arm + jointimg44 at p > 0.03, and NOCAP_DV150 + jointimg44 at p > 0.01 / 0.02 / 0.05;
  * paired bootstrap over the 128 movies (score, adj_J) of every cell against REF + jointimg and against
    NOCAP_DV150 + jointimg44 (the reference arm at the shipped cut).
Attach: forkprop's sources + the kernel biohub-fk44prop. CPU ~6.5 h.
"""
import os
import re

src = open("forkprop.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-forkprop"', 'SCRIPT_VERSION = "2026-09-18a-forkprop2"')

# ---- the arms ---------------------------------------------------------------------------------
rep('''DA_ARMS = [("CR", dict(diverge=2.25, admit=False, rank="contrast", cap=1.0)),
           ("CR_CAP2", dict(diverge=2.25, admit=False, rank="contrast", cap=2.0)),
           ("CR_CAP4", dict(diverge=2.25, admit=False, rank="contrast", cap=4.0)),
           ("CR_CAP4_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=4.0)),
           ("DIST_CAP4_DV150", dict(diverge=1.5, admit=False, rank="dist", cap=4.0)),
           ("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0))]''',
    '''GEO_SHIPPED = (12.0, 15.0, 10.0)      # DIV_PARENT_UM, DIV_SISTER_UM, DIV_CHILD_UM
DA_ARMS = [("NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, geo=GEO_SHIPPED)),
           ("NOCAP_DV100", dict(diverge=1.0, admit=False, rank="contrast", cap=100.0, geo=GEO_SHIPPED)),
           ("NOCAP_DV000", dict(diverge=0.0, admit=False, rank="contrast", cap=100.0, geo=GEO_SHIPPED)),
           ("NOCAP_DV150_GEO", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, geo=(15.0, 20.0, 12.0)))]''')
rep('''      f"each movie: " + ", ".join(f"{a_} (rank {c_['rank']}, caps x{c_['cap']:g}, DIV_DIVERGE_UM {c_['diverge']})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)''',
    '''      f"each movie: " + ", ".join(f"{a_} (rank {c_['rank']}, caps x{c_['cap']:g}, DIV_DIVERGE_UM {c_['diverge']}, "
                                  f"parent/sister/child {c_['geo'][0]:g}/{c_['geo'][1]:g}/{c_['geo'][2]:g} um)"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)''')
rep('''    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DA_ADMIT = cfg["admit"]
''', '''    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DIV_PARENT_UM, DIV_SISTER_UM, DIV_CHILD_UM = cfg["geo"]
        DA_ADMIT = cfg["admit"]
''')
rep('''    DIV_DIVERGE_UM = 2.25
    DA_ADMIT = False
    FP_RANK = "dist"
    FP_CAP = 1.0
    DA_CTX.clear()
    IMG_CTX.clear()
''', '''    DIV_DIVERGE_UM = 2.25
    DIV_PARENT_UM, DIV_SISTER_UM, DIV_CHILD_UM = GEO_SHIPPED
    DA_ADMIT = False
    FP_RANK = "dist"
    FP_CAP = 1.0
    DA_CTX.clear()
    IMG_CTX.clear()
''')

# ---- the 44b6 forks of the NOCAP_DV150 patch (fk44prop) ---------------------------------------
rep('''print(f"  jointimg: {len(X44IMG)} counted 44b6 forks with image columns from {_f44[0]} (TP {int(Y44IMG.sum())})", flush=True)
''', '''print(f"  jointimg: {len(X44IMG)} counted 44b6 forks with image columns from {_f44[0]} (TP {int(Y44IMG.sum())})", flush=True)
_f44p = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/forks_44b6_CR_NOCAP_DV150.csv")]
assert _f44p, "attach the kernel faustine649/biohub-fk44prop (forks_44b6_CR_NOCAP_DV150.csv)"
_t44p = _pd.read_csv(_f44p[0])
assert list(_t44p.columns[5:]) == FEAT_NAMES, list(_t44p.columns[5:])
_t44p = _t44p[_t44p.label >= 1]
X44P = _t44p[FEAT_NAMES].to_numpy(float)
Y44P = (_t44p.label == 2).astype(int).to_numpy()
assert len(Y44P) == 207 and int(Y44P.sum()) == 11, (len(Y44P), int(Y44P.sum()))      # fk44prop's n_train
print(f"  jointimg44: {len(X44P)} counted 44b6 forks of the NOCAP_DV150 patch from {_f44p[0]} (TP {int(Y44P.sum())})", flush=True)
''')

# ---- the cross: cells with their own cut, two references, the bootstrap -------------------------
i = src.index("THR = 0.03\n")
j = src.index('for a_, _c in BR_ARMS_RUN:\n    _csv_a = pathlib.Path("/kaggle/working") / f"forkprop_arm_{a_}.csv"')
assert src.count("THR = 0.03\n") == 1 and i < j
CROSS = r'''THR = 0.03
A0 = BR_ARMS_RUN[0][0]                      # the reference arm: forkprop's winner
CELLS = []                                  # (label, arm, gate, forks, movies, per, add, thr)
CELLS.append(("REF + joint", "REF", "joint", FORKS, MOVIE, PER, (X44, Y44), THR))
CELLS.append(("REF + jointimg", "REF", "jointimg", FORKS, MOVIE, PER, (X44IMG, Y44IMG), THR))
for a_, _c in BR_ARMS_RUN:
    if a_ == A0:
        CELLS.append((f"{a_} + jointimg", a_, "jointimg", ARM_FORKS[a_], ARM_MOVIE[a_], ARM_PER[a_], (X44IMG, Y44IMG), THR))
    CELLS.append((f"{a_} + jointimg44", a_, "jointimg44", ARM_FORKS[a_], ARM_MOVIE[a_], ARM_PER[a_], (X44P, Y44P), THR))
for thr_ in (0.01, 0.02, 0.05):
    CELLS.append((f"{A0} + jointimg44 p>{thr_:g}", A0, "jointimg44", ARM_FORKS[A0], ARM_MOVIE[A0], ARM_PER[A0], (X44P, Y44P), thr_))
COLS_OF = {c_[0]: (J20 + JIMG if "img" in c_[2] else J20) for c_ in CELLS}
WANT = {"REF + joint": "want 49/51/76 0.2784 0.9155 0.9433 (round 2, what ships)",
        "REF + jointimg": "want 55/42/70 0.3293 0.9155 0.9484 (forkprop)",
        f"{A0} + jointimg": "want 61/36/64 0.3789 0.9155 0.9534 (forkprop's CR_NOCAP_DV150)"}
print(f"\n=== THE CROSS: {len(CELLS)} cells, real pruning, official scorer. jointimg = the cell's own 6bba forks (other fold) "
      f"+ fk44img's 88 REF 44b6 forks; jointimg44 = the same 6bba forks + fk44prop's 207 44b6 forks of the NOCAP_DV150 patch "
      f"(the shipped fork_gate_prop.pkl's population). The cut is p > 0.03 unless the label says otherwise ===", flush=True)
print(f"  {'cell':<30} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} {'vs REF+jointimg':>16} {'f0':>8} {'f1':>8}   AUC / anchor",
      flush=True)
XRES = {}
for lab, _arm, _gn, _fa, _ma, _pa, _add, _thr in CELLS:
    rc, rows2, pf2, aucs = _gate_and_prune("".join(ch for ch in lab if ch.isalnum() or ch == "_"), _fa, _ma, _pa, _thr,
                                           add=_add, cols=COLS_OF[lab])
    XRES[lab] = dict(rc=rc, rows=rows2, pf=pf2)
    base = XRES.get("REF + jointimg", XRES[lab])
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    d = rc["score"] - base["rc"]["score"]
    df0 = f0["score"] - _with_ratio(base["pf"][0])["score"]
    df1 = f1["score"] - _with_ratio(base["pf"][1])["score"]
    print(f"  {lab:<30} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} {d:>+16.4f} "
          f"{df0:>+8.4f} {df1:>+8.4f}   "
          + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + "   " + WANT.get(lab, ""), flush=True)

print(f"\n  PAIRED BOOTSTRAP over the {len(SIX_OK)} movies, 2,000 draws, the same resample for every cell: every cell "
      f"against REF + jointimg (what ships, LB 0.951) and against {A0} + jointimg44 (the reference arm at the shipped cut) ===",
      flush=True)
_rb = np.random.default_rng(9090)
_draws = [_rb.choice(len(SIX_OK), size=len(SIX_OK), replace=True) for _ in range(2000)]


def _bank(rows_):
    agg = [_with_ratio([rows_[i] for i in d]) for d in _draws]
    return {"score": np.array([a["score"] for a in agg]), "adj_J": np.array([a["adj_edge_jaccard"] for a in agg])}


BS = {k: _bank(XRES[k]["rows"]) for k in XRES}
REFS = ["REF + jointimg", f"{A0} + jointimg44"]
print(f"    {'cell':<30} {'against':<26} {'metric':<6} {'mean d':>9} {'90% interval':>22}  distinguishable", flush=True)
for lab in XRES:
    for ref_ in REFS:
        if lab == ref_:
            continue
        for nm_ in ("score", "adj_J"):
            d = BS[lab][nm_] - BS[ref_][nm_]
            lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
            print(f"    {lab:<30} {ref_:<26} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
                  f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print(f"    reading: a knob counts only if its cell beats {A0} + jointimg44 with the interval off zero; the 44b6 half of "
      f"any winner is unmeasured here (fk44prop measured the NOCAP_DV150 patch at p > 0.03 only).", flush=True)

'''
src = src[:i] + CROSS + src[j:]

# ---- names -------------------------------------------------------------------------------------
rep('f"forkprop_arm_{a_}.csv"', 'f"forkprop2_arm_{a_}.csv"')
rep('f"forkprop_REF_{DET_SRC}.csv"', 'f"forkprop2_REF_{DET_SRC}.csv"')
rep('print("FORKPROP OK", flush=True)', 'print("FORKPROP2 OK", flush=True)')
src = re.sub(r'FORKPROP INCOMPLETE', 'FORKPROP2 INCOMPLETE', src)

assert src.count("jointimg44") >= 6 and src.count("GEO_SHIPPED") == 5, (src.count("jointimg44"), src.count("GEO_SHIPPED"))
assert src.count("XRES[lab] = dict(rc=rc, rows=rows2, pf=pf2)") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "forkprop2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written forkprop2.py", len(src))
