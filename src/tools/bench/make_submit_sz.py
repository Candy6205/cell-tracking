"""submit_nc35.py -> submit_sz.py: the shipped submission with the raw-image half-max SIZE columns in the fork gate.

WHY (2026-09-24). forkvel2 found the raw-image half-max radius the strongest new fork column since contrast (true daughters
3.1 um against 4.9-5.1 for false ones, AUC 0.83; the user's eye: false daughters are mother-sized). Its gain on top of the
shipped 31 gate columns was +0.004..+0.043 div_J by fold split; the user asked for the submission. The gate is refitted on
the shipped composition's own fork population by forkprop_sz + fk44prop_sz (fork_gate_sz.pkl, 20 + 16 columns).

WHAT. submit_nc35.py verbatim (node cost 0.35, XT exemption, the uncapped contrast-ranked orphan patch, the image fork gate
at p > 0.03) with: `_fi_hm(frame, xyz)` (the half-max radius on the same smoothed crop `_fi_node` uses), the five columns
r_hm_p, r_hm_c1, r_hm_q, rhm_max, rhm_sum appended to GATE_FEATS / FEAT_IMG and filled by `_fork_img_feats`, the gate file
fork_gate_sz.pkl (attach biohub-fk44prop-sz instead of biohub-fk44prop), and the training-size assertion relaxed to the new
gate's n_train (the fork population is the same arm, the count may differ by a few). Nothing else changes.
"""
src = open("submit_nc35.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-23a-submit-nc35"', 'SCRIPT_VERSION = "2026-09-24a-submit-sz"')
rep('GATE_NAME = "fork_gate_prop.pkl"', 'GATE_NAME = "fork_gate_sz.pkl"')
rep('''              "i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
FEAT_IMG = GATE_FEATS[20:]''',
    '''              "i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",
              "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]      # submit_sz: raw-image half-max size
FEAT_IMG = GATE_FEATS[20:]''')
rep('''                         f"faustine649/biohub-fk44prop as a kernel source")''',
    '''                         f"faustine649/biohub-fk44prop-sz as a kernel source")''')
rep('''assert _nt.get("6bba") == 725 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 70, (
    f"fork_gate_prop.pkl was trained on {_nt}, not forkprop's 725 counted 6bba forks of the CR_NOCAP_DV150 "
    f"patch plus fk44prop's 44b6 ones -- this is not the gate forkprop / fk44prop measured")''',
    '''assert 600 <= _nt.get("6bba", 0) <= 900 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 70, (
    f"fork_gate_sz.pkl was trained on {_nt}, not forkprop_sz's counted 6bba forks of the CR_NOCAP_DV150 "
    f"patch plus fk44prop_sz's 44b6 ones -- this is not the gate forkprop_sz / fk44prop_sz measured")
assert "r_hm_p" in _gb["feature_names"] and len(_gb["feature_names"]) == 36, _gb["feature_names"]''')
# ---- the half-max radius, on the submission's frame-based crop ---------------------------------------
rep('''def _fork_img_feats(fc, nodes, pred, p, c1, q):
    t = int(nodes[p]["t"])''',
    '''def _fi_hm(frame, xyz_vox):
    """raw-image half-maximum radius (um) of one node in one frame: on the smoothed crop, the connected set at or above
    bg + 0.5 (peak - bg) that contains the peak (peak = max within FI_PEAK_UM of the node, bg = crop median), as the
    radius of the sphere of equal volume. NaN when unreadable. Same crop and smoothing as _fi_node."""
    if frame is None:
        return np.nan
    hz = int(np.ceil(FI_HALF_UM / SCALE[0]))
    hy = int(np.ceil(FI_HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    crop = frame[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return np.nan
    sm = _ndi.gaussian_filter(crop.astype(np.float32), FI_SIGMA_VOX)
    c = np.asarray(xyz_vox, float) - np.array([z0, y0, x0], float)
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


def _fork_img_feats(fc, nodes, pred, p, c1, q):
    t = int(nodes[p]["t"])''')
rep('''    i_q, c_q, r_q = _fi_node(fc.get(t + 1), nodes[q]["xyz"])
    return dict(i_p=i_p, c_p=c_p, r_p=r_p,''',
    '''    i_q, c_q, r_q = _fi_node(fc.get(t + 1), nodes[q]["xyz"])
    r_hp, r_h1, r_hq = _fi_hm(fc.get(t), nodes[p]["xyz"]), _fi_hm(fc.get(t + 1), nodes[c1]["xyz"]), _fi_hm(fc.get(t + 1), nodes[q]["xyz"])
    _ok = np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0
    return dict(r_hm_p=r_hp, r_hm_c1=r_h1, r_hm_q=r_hq,
                rhm_max=(max(r_h1, r_hq) / r_hp) if _ok else np.nan,
                rhm_sum=((r_h1 + r_hq) / r_hp) if _ok else np.nan,
                i_p=i_p, c_p=c_p, r_p=r_p,''')
assert src.count("_fi_hm(") == 4 and src.count('"rhm_sum"') == 1
open("submit_sz.py", "w", encoding="utf-8", newline="\n").write(src)
print("written submit_sz.py", src.count("\n"), "lines")
