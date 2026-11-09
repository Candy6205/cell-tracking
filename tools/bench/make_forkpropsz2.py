"""forkprop_sz.py -> forkprop_sz2.py: five more ways of measuring a nucleus' size, side by side in the fork gate cross.

WHY (2026-09-25). The size gate (submit_sz, 36 columns) read LB 0.954 against submit_nc35's 0.956; offline the 31 / 36 /
33-column gates are indistinguishable on the fork tables (every paired interval across zero). The user: the reading
depends on how size is measured ("尺寸这个和尺寸怎么量的关系太大了"). The shipped measure is one choice -- the crop's
median as background, half of (peak - background) as the threshold, the connected component that holds the peak --
and it is sensitive to a raised background in dense regions and to a touching neighbour merging into the component.

WHAT. forkprop_sz.py verbatim, with:
  * five more size measures per node from the same smoothed crop (_fi_sizes2): hm10 (background = the crop's 10th
    percentile), c30 / c70 (thresholds at 0.3 / 0.7 of peak - background, median background), prof (radial-profile
    radius: shell-averaged intensity around the peak drops to half way between the peak and the 6-8 um shells' mean; no
    connectivity), mom (background-subtracted radius of gyration inside the half-max component, no fixed window);
    each as p / c1 / q + max ratio + sum ratio -> 25 columns appended to FEAT_IMG (61 image-side columns in all);
  * the fork tables (forkssz2_*.csv) carry the fork nodes' frame and voxel coordinates after the features, so a later
    kernel can compute detector-heatmap measures at the same nodes (the user's second ask);
  * pass 2 replays ONE arm (CR_NOCAP_DV150, the shipped patch) instead of seven: forkprop_sz's pass 2 took 411 min;
  * the cross: REF + gate6 / joint / jointimg (36, reproduces forkprop_sz) / img31; the arm + joint / jointimg /
    img31 / img33 / one cell per new measure (31 + its 5) / img_ratios (31 + every measure's two ratios) / imgall (61);
    official scorer, real pruning at p > 0.03; paired bootstrap of every image cell of the arm against its img31 and
    its jointimg cells.
Attach: forkprop_sz's sources. CPU ~5 h (pass 1 106 min, one arm ~70 min, 15 cross cells ~2 h). Kernel biohub-forkprop-sz2.
"""
src = open("forkprop_sz.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-24a-forkprop_sz"', 'SCRIPT_VERSION = "2026-09-25a-forkprop_sz2"')

# ---- the columns ----------------------------------------------------------------------------------------------------
rep('''FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",
            "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]   # size variant: raw-image half-max radius
''', '''SZ2 = ["hm10", "c30", "c70", "prof", "mom"]          # sz2: five more size measures (see _fi_sizes2)
FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",
            "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"] + [f"{m_}_{s_}" for m_ in SZ2 for s_ in ("p", "c1", "q", "max", "sum")]
''')

# ---- the measures ---------------------------------------------------------------------------------------------------
SIZES2 = '''def _fi_sizes2(img, t, xyz_vox):
    """five more size measures (um) of one node from the same smoothed crop as _fi_hm; NaN when unreadable:
    hm10  half-max radius with the background = the crop's 10th percentile instead of its median
    c30   the connected set at or above bg + 0.3 (peak - bg) holding the peak (bg = median), as an equal-volume sphere
    c70   the same at 0.7
    prof  radial-profile radius: the shell-averaged intensity around the peak (0.5 um shells) falls below
          far + 0.5 (peak - far), far = the mean of the 6-8 um shells; no connectivity, a touching neighbour cannot merge
    mom   the background-subtracted, intensity-weighted radius of gyration inside the half-max component (no fixed window)"""
    out = {k_: np.nan for k_ in SZ2}
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return out
    c = np.asarray(xyz_vox, float) - off
    zz, yy, xx = np.indices(sm.shape)
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    near = d <= FI_PEAK_UM
    if not near.any():
        return out
    j = np.unravel_index(int(np.argmax(np.where(near, sm, -np.inf))), sm.shape)
    peak = float(sm[j])
    bg = float(np.median(sm))
    bg10 = float(np.percentile(sm, 10))
    vox = float(np.prod(SCALE))
    dp = np.sqrt(((zz - j[0]) * SCALE[0]) ** 2 + ((yy - j[1]) * SCALE[1]) ** 2 + ((xx - j[2]) * SCALE[2]) ** 2)

    def _rad(mask):
        if not mask[j]:
            return np.nan
        lab_, _n = _ndi.label(mask)
        vol = int((lab_ == lab_[j]).sum()) * vox
        return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))

    if peak > bg10:
        out["hm10"] = _rad(sm >= bg10 + 0.5 * (peak - bg10))
    if peak > bg:
        out["c30"] = _rad(sm >= bg + 0.3 * (peak - bg))
        out["c70"] = _rad(sm >= bg + 0.7 * (peak - bg))
        lab_, _n = _ndi.label(sm >= bg + 0.5 * (peak - bg))
        comp = lab_ == lab_[j]
        w = np.clip(sm[comp] - bg, 0, None)
        out["mom"] = float(np.sqrt((w * dp[comp] ** 2).sum() / w.sum())) if w.sum() > 0 else np.nan
    shells = np.arange(0.0, 8.5, 0.5)
    prof = np.full(len(shells) - 1, np.nan)
    for i_, (a_, b_) in enumerate(zip(shells[:-1], shells[1:])):
        m_ = (dp >= a_) & (dp < b_)
        if m_.any():
            prof[i_] = float(sm[m_].mean())
    far_ = prof[12:16]
    far = float(np.nanmean(far_)) if np.isfinite(far_).any() else np.nan
    if np.isfinite(far) and peak > far:
        half = far + 0.5 * (peak - far)
        r_ = np.nan
        for i_ in range(len(prof)):
            if np.isfinite(prof[i_]) and prof[i_] < half:
                if i_ == 0:
                    r_ = 0.25
                else:
                    p0, p1 = prof[i_ - 1], prof[i_]
                    x0, x1 = shells[i_ - 1] + 0.25, shells[i_] + 0.25
                    r_ = (x0 + (p0 - half) / (p0 - p1) * (x1 - x0)) if (np.isfinite(p0) and p0 > p1) else x1
                break
        out["prof"] = float(r_)
    return out


def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}'''
rep('''def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}''', SIZES2)
rep('''               rhm_sum=((r_h1 + r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan)
    FI_STAT["forks"] += 1
''', '''               rhm_sum=((r_h1 + r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan)
    s_p, s_1, s_q = _fi_sizes2(img, t, nodes[p]["xyz"]), _fi_sizes2(img, t + 1, nodes[c1]["xyz"]), _fi_sizes2(img, t + 1, nodes[q]["xyz"])
    for m_ in SZ2:
        a_, b_, c_ = s_p[m_], s_1[m_], s_q[m_]
        ok_ = np.isfinite(a_) and np.isfinite(b_) and np.isfinite(c_) and a_ > 0
        out.update({f"{m_}_p": a_, f"{m_}_c1": b_, f"{m_}_q": c_,
                    f"{m_}_max": (max(b_, c_) / a_) if ok_ else np.nan, f"{m_}_sum": ((b_ + c_) / a_) if ok_ else np.nan})
    FI_STAT["forks"] += 1
''')

# ---- the fork nodes' coordinates in the tables (for a later heatmap kernel) ------------------------------------------
rep("        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES]))\n",
    "        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES],\n"
    "                     (float(t),) + tuple(float(v) for v in nodes[p][\"xyz\"]) + tuple(float(v) for v in nodes[c1][\"xyz\"])\n"
    "                     + tuple(float(v) for v in nodes[q][\"xyz\"])))\n")
COORD_HDR = ',t,z_p,y_p,x_p,z_c1,y_c1,x_c1,z_q,y_q,x_q'
rep('''with open("/kaggle/working/forkssz_dw55.csv", "w") as fh:
    fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\\n")
    for r_ in FORKS:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\\n")
for arm, _m in DIVW_ARMS:
    with open(f"/kaggle/working/forkssz_{arm}.csv", "w") as fh:
        fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\\n")
        for r_ in ARM_FORKS[arm]:
            fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\\n")
''', '''with open("/kaggle/working/forkssz2_dw55.csv", "w") as fh:
    fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "%s\\n")
    for r_ in FORKS:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5])
                 + ("," + ",".join(f"{v:.6g}" for v in r_[6]) if len(r_) > 6 else "") + "\\n")
for arm, _m in DIVW_ARMS:
    with open(f"/kaggle/working/forkssz2_{arm}.csv", "w") as fh:
        fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "%s\\n")
        for r_ in ARM_FORKS[arm]:
            fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5])
                     + ("," + ",".join(f"{v:.6g}" for v in r_[6]) if len(r_) > 6 else "") + "\\n")
''' % (COORD_HDR, COORD_HDR))

# ---- one arm --------------------------------------------------------------------------------------------------------
rep("BR_ARMS = DA_ARMS         # the replay loop below iterates BR_ARMS_RUN\n",
    "BR_ARMS = [a_ for a_ in DA_ARMS if a_[0] == \"CR_NOCAP_DV150\"]   # sz2: the shipped patch only (forkprop_sz's 7 arms took 411 min)\n")
rep('("20 feats + 16 image, 2 folds", J20 + JIMG, FOLDS2)', '("20 feats + 41 image, 2 folds", J20 + JIMG, FOLDS2)')

# ---- the cross cells ------------------------------------------------------------------------------------------------
rep('''    _gates = ((("gate6", None), ("joint", (X44, Y44)), ("gate6img", None), ("jointimg", (X44IMG, Y44IMG))) if _arm == "REF"
              else (("joint", (X44, Y44)), ("jointimg", (X44IMG, Y44IMG))))
''', '''    _gates = ((("gate6", None), ("joint", (X44, Y44)), ("jointimg", (X44IMG, Y44IMG)), ("img31", (X44IMG, Y44IMG))) if _arm == "REF"
              else tuple([("joint", (X44, Y44))] + [(g_, (X44IMG, Y44IMG)) for g_ in IMG_SETS]))
''')
rep('''COLS_OF = {c_[0]: (J20 + JIMG if c_[2].endswith("img") else J20) for c_ in CELLS}
''', '''COLS_OF = {c_[0]: (IMG_SETS[c_[2]] if c_[2] in IMG_SETS else J20) for c_ in CELLS}
''')
rep('''THR = 0.03
CELLS = []          # (label, arm, gate) in the order they are printed
''', '''THR = 0.03
JIMG11 = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG[:11]]
JSZ = {"hm": [FEAT_NAMES.index(f_) for f_ in ("r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum")]}
for m_ in SZ2:
    JSZ[m_] = [FEAT_NAMES.index(f"{m_}_{s_}") for s_ in ("p", "c1", "q", "max", "sum")]
JRAT = JSZ["hm"][3:] + [FEAT_NAMES.index(f"{m_}_{s_}") for m_ in SZ2 for s_ in ("max", "sum")]
IMG_SETS = {"jointimg": J20 + JIMG11 + JSZ["hm"],          # forkprop_sz's 36-column gate (must reproduce it)
            "img31": J20 + JIMG11,                          # submit_nc35's 31
            "img33": J20 + JIMG11 + JSZ["hm"][3:],           # 31 + the two half-max ratios
            **{f"img_{m_}": J20 + JIMG11 + JSZ[m_] for m_ in SZ2},
            "img_ratios": J20 + JIMG11 + JRAT,               # 31 + every measure's two ratios (12)
            "imgall": J20 + JIMG}                            # everything (61)
assert len(IMG_SETS["jointimg"]) == 36 and len(IMG_SETS["img31"]) == 31 and len(IMG_SETS["imgall"]) == 61, [len(v) for v in IMG_SETS.values()]
print("  sz2 column sets: " + ", ".join(f"{k_} {len(v_)}" for k_, v_ in IMG_SETS.items()), flush=True)
CELLS = []          # (label, arm, gate) in the order they are printed
''')
# a paired bootstrap of every image cell of the arm against its img31 and its jointimg cells
rep('''for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"forkprop_arm_{a_}.csv"
''', '''print(f"\\n=== SIZE MEASURES: every image cell of the arm AGAINST its img31 (submit_nc35's columns) and its jointimg "
      f"(the 36-column size gate), paired bootstrap over the {len(SIX_OK)} movies, 2,000 draws, official score; "
      f"the bench resolves about +-0.003 of score (66 true forks) ===", flush=True)
for a_, _c in BR_ARMS_RUN:
    print(f"    {'cell':<26} {'div tp/fp/fn':>14} {'score':>7} {'vs img31':>9} {'90% interval':>22} {'vs jointimg':>12} {'90% interval':>22}", flush=True)
    for g_ in IMG_SETS:
        lab_ = f"{a_} + {g_}"
        if lab_ not in BS or f"{a_} + img31" not in BS or f"{a_} + jointimg" not in BS:
            continue
        rows_ = XRES[lab_]["rows"]
        d1 = BS[lab_] - BS[f"{a_} + img31"]
        d2 = BS[lab_] - BS[f"{a_} + jointimg"]
        print(f"    {lab_:<26} {sum(r['division_tp'] for r in rows_):>4}/{sum(r['division_fp'] for r in rows_):>4}/"
              f"{sum(r['division_fn'] for r in rows_):<4} {XRES[lab_]['rc']['score']:>7.4f} "
              f"{float(np.mean(d1)):>+9.4f} [{float(np.quantile(d1, .05)):>+8.4f}, {float(np.quantile(d1, .95)):>+8.4f}] "
              f"{float(np.mean(d2)):>+12.4f} [{float(np.quantile(d2, .05)):>+8.4f}, {float(np.quantile(d2, .95)):>+8.4f}]"
              + ("  YES" if (np.quantile(d1, .05) > 0 or np.quantile(d1, .95) < 0) else ""), flush=True)
print("    reading: a measure earns a place only if its cell beats img31 with the interval off zero; the 36-column "
      "jointimg cell is forkprop_sz's (0.9546 there).", flush=True)

for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"forkprop_arm_{a_}.csv"
''')
rep('print("FORKPROPSZ OK", flush=True)', 'print("FORKPROPSZ2 OK", flush=True)')

assert src.count("_fi_sizes2(") == 4 and src.count("IMG_SETS") >= 6 and "gate6img" not in src and src.count("forkssz2_") == 2, (
    src.count("_fi_sizes2("), src.count("IMG_SETS"), src.count("forkssz2_"))
open("forkprop_sz2.py", "w", encoding="utf-8", newline="\n").write(src)
print("written forkprop_sz2.py", src.count("\n"), "lines")
