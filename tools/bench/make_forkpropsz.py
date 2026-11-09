"""make_forkpropsz.py -> forkprop_sz.py + fk44prop_sz.py: the shipped gate's two fork harnesses with the raw-image
half-maximum SIZE columns, for a size-gate submission variant.

WHY (2026-09-24). forkvel2 (forkimg's REF population) found the half-max radius of the raw image the strongest new fork
column since contrast: true daughters 3.1 um against 4.9-5.1 for false ones (AUC 0.83), the user's eye in numbers. Its
gate gain on top of the shipped 31 columns was +0.004..+0.043 div_J depending on the fold split; the user asked for the
submission anyway. The shipped gate (fork_gate_prop.pkl) is fitted on the UNCAPPED contrast-ranked patch's fork
population (forkprop's 725 counted 6bba forks + fk44prop's 44b6 ones), not on forkimg's REF population, and the record
has seen a gate re-fit on a changed population move the score by -0.001; so the size columns are computed on that
population, by the harnesses that made it, and the gate is fitted where it will be applied.

WHAT. forkprop.py and fk44prop.py verbatim, with five columns appended to FEAT_IMG (the velocity columns are left out:
forkvel2 killed them in the gate):
  r_hm_p, r_hm_c1, r_hm_q   raw-image half-maximum radius (um) at the mother and both daughters: on the smoothed crop,
           the connected voxel set at or above background + half of (peak - background) containing the peak, as the
           radius of the sphere of equal volume
  rhm_max  max(r_hm_c1, r_hm_q) / r_hm_p;  rhm_sum  (r_hm_c1 + r_hm_q) / r_hm_p
forkprop_sz writes its fork tables as forkssz_*.csv (the old names live in the forktables dataset both kernels attach,
and fk44prop's glob must find the new ones); its joint cell pads the dataset's 44b6 image table with NaN size columns.
fk44prop_sz reads forkssz_*.csv from biohub-forkprop-sz, computes the size columns on its own 44b6 forks, and writes
fork_gate_sz.pkl (20 + 16 columns) plus forkssz_44b6img.csv. Kernel names FORKPROPSZ / FK44PROPSZ.
Attach: forkprop's / fk44prop's sources; fk44prop_sz also biohub-forkprop-sz. CPU: forkprop ~11 h, fk44prop ~2 h.
"""
import os

HM_FUNC = '''def _fi_hm(img, t, xyz_vox):
    """raw-image half-maximum radius (um) of one node: on the smoothed crop, the connected set at or above
    bg + 0.5 (peak - bg) that contains the peak (peak = max within FI_PEAK_UM of the node, bg = crop median),
    reported as the radius of the sphere of equal volume. NaN when unreadable."""
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return np.nan
    c = np.asarray(xyz_vox, float) - off
    zz, yy, xx = np.indices(sm.shape)
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    near = d <= FI_PEAK_UM
    if not near.any():
        return np.nan
    bg = float(np.median(sm))
    j = np.unravel_index(int(np.argmax(np.where(near, sm, -np.inf))), sm.shape)
    peak = float(sm[j])
    if peak <= bg:
        return np.nan
    lab_, _n = _ndi.label(sm >= bg + 0.5 * (peak - bg))
    vol = int((lab_ == lab_[j]).sum()) * float(np.prod(SCALE))
    return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))


def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}'''

HM_UPDATE = '''    r_hp, r_h1, r_hq = _fi_hm(img, t, nodes[p]["xyz"]), _fi_hm(img, t + 1, nodes[c1]["xyz"]), _fi_hm(img, t + 1, nodes[q]["xyz"])
    out.update(r_hm_p=r_hp, r_hm_c1=r_h1, r_hm_q=r_hq,
               rhm_max=(max(r_h1, r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan,
               rhm_sum=((r_h1 + r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan)
    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out'''


def build(src_name, out_name, version, is44):
    src = open(src_name, encoding="utf-8").read()

    def rep(a, b, n=1):
        nonlocal src
        assert src.count(a) == n, (src_name, src.count(a), a[:90])
        src = src.replace(a, b)

    rep('SCRIPT_VERSION = "', f'SCRIPT_VERSION = "{version}"  # {src_name[:-3]} was "')
    rep('FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]',
        'FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",\n'
        '            "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]   # size variant: raw-image half-max radius')
    rep('''def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}''', HM_FUNC)
    rep('''    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out''', HM_UPDATE)
    src = src.replace('"20 feats + 11 image, 2 folds"', '"20 feats + 16 image, 2 folds"')
    src = src.replace('"11 image alone, 2 folds"', '"16 image alone, 2 folds"')
    if not is44:
        # the joint cell reads the dataset's 44b6 image table (31 + 8 columns): pad the five new ones with NaN
        rep('''_t44 = _pd.read_csv(_f44[0])
assert list(_t44.columns[5:]) == FEAT_NAMES, list(_t44.columns[5:])''',
            '''_t44 = _pd.read_csv(_f44[0])
for _c in FEAT_NAMES:
    if _c not in _t44.columns:
        _t44[_c] = np.nan                                       # size variant: the dataset's table predates the size columns
assert all(_c in _t44.columns for _c in FEAT_NAMES), list(_t44.columns[5:])''')
        rep('with open("/kaggle/working/forks_dw55.csv", "w") as fh:', 'with open("/kaggle/working/forkssz_dw55.csv", "w") as fh:')
        rep('    with open(f"/kaggle/working/forks_{arm}.csv", "w") as fh:', '    with open(f"/kaggle/working/forkssz_{arm}.csv", "w") as fh:')
        rep('print("FORKPROP OK", flush=True)', 'print("FORKPROPSZ OK", flush=True)')
        assert src.count("_fi_hm(") == 4
    else:
        rep('_SIX_CSV = {"REF": "forks_dw55.csv", **{a_: f"forks_{a_}.csv" for a_, _c in STEAL_ARMS}}',
            '_SIX_CSV = {"REF": "forkssz_dw55.csv", **{a_: f"forkssz_{a_}.csv" for a_, _c in STEAL_ARMS}}   # from biohub-forkprop-sz')
        rep('    assert hit, f"attach the dataset faustine649/biohub-forktables ({fn})"', '    assert hit, f"attach the kernel faustine649/biohub-forkprop-sz ({fn})"')
        rep('with open("/kaggle/working/forks_44b6img.csv", "w") as fh:', 'with open("/kaggle/working/forkssz_44b6img.csv", "w") as fh:')
        rep('with open("/kaggle/working/fork_gate_prop.pkl", "wb") as fh:', 'with open("/kaggle/working/fork_gate_sz.pkl", "wb") as fh:')
        rep('fork_gate_prop.pkl written', 'fork_gate_sz.pkl written')
        rep('20 gate + 11 image features, TP=1 FP=0"),', '20 gate + 16 image features (5 raw-image half-max size), TP=1 FP=0"),')
        rep('print("FK44PROP OK", flush=True)', 'print("FK44PROPSZ OK", flush=True)')
        assert src.count("_fi_hm(") == 4
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", out_name)
    open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
    print("written", out_name, len(src))


build("forkprop.py", "forkprop_sz.py", "2026-09-24a-forkprop_sz", False)
build("fk44prop.py", "fk44prop_sz.py", "2026-09-24a-fk44prop_sz", True)
