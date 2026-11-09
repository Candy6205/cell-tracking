"""make_forkvel2.py -> forkvel2.py: forkimg.py + velocity continuity + a raw-image half-maximum size, for the fork gate.

WHY (2026-09-24). forkvel measured, on detector coordinates, that at a real division the best-continuing daughter
leaves the mother's velocity by 4 um per frame (median; 1.6% under 1.0) while the shipped graph's forks continue the
mother within 1.4 um (35% under 1.0, 56% under 1.5): the user's point that a daughter cannot keep the mother's speed
and direction, only a false fork can, because nothing divided. The gate has no such term (its motion terms are the
angle at the fork and the separation growth). Also: the gate's radius feature is a fixed-window moment (2.5-2.66 um for
every fork); the user's eye says false daughters are mother-sized and real ones smaller, and a heatmap-derived mask
cannot carry size (the heatmap's blobs are fixed-radius by construction), so the size must come from the raw image.

WHAT. forkimg.py verbatim (dupsup's harness, REF only, the 128 6bba train movies, the fork table with the scorer's
verdict per fork, the gate curves, the cross, the bootstrap), with seven columns appended to FEAT_IMG:
  cont     min over the two branches of |(c - p) - (p - m)| um/frame, m = the mother's single predecessor (NaN if none)
  cont2    the same one frame later, min over the branches whose successor is single
  r_hm_p, r_hm_c1, r_hm_q   raw-image half-maximum radius (um): on the smoothed crop, the connected voxel set at or above
           background + half of (peak - background) that contains the peak, as the radius of the sphere of equal volume
  rhm_max  max(r_hm_c1, r_hm_q) / r_hm_p;  rhm_sum  (r_hm_c1 + r_hm_q) / r_hm_p
The 44b6 fork rows carry NaN in all of them (as they do for the image columns; HistGB takes NaN). The harness's
"20 + image" cells therefore become 20 + 18; the 31-column shipped set is re-fitted offline from the table
(scratchpad gatelab on forks_dw55.csv, out of fold by movie), which is where the three sets are compared.
v2 of forkimg died at the joint pickle block (`lab` rebound by a loop); the fork label array is kept as LAB_F here.
Attach: forkimg's 19 sources. CPU, ~4 h.
"""
import os

src = open("forkimg.py", encoding="utf-8").read()


def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (src.count(a), a[:90])
    src = src.replace(a, b)


rep('SCRIPT_VERSION = "', 'SCRIPT_VERSION = "2026-09-24a-forkvel2"  # forkimg was "')
# ---- the new columns ------------------------------------------------------------------------------
rep('FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]',
    'FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",\n'
    '            "cont", "cont2", "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]   # forkvel2: motion + half-max size')
rep('"20 feats + 11 image, 2 folds"', '"20 feats + 18 image/motion, 2 folds"')
rep('"20 feats + 11 image, 3 folds"', '"20 feats + 18 image/motion, 3 folds"')
rep('"11 image alone, 2 folds"', '"18 image/motion alone, 2 folds"')
# ---- half-maximum radius and velocity continuity ---------------------------------------------------
rep('''def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}''',
    '''def _fi_hm(img, t, xyz_vox):
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


def _vel_cont(nodes, pred, succ, p, c1, q):
    """(cont, cont2): how well the best branch continues the mother's velocity, at the fork and one frame later (um/frame)."""
    pr = pred.get(p, [])
    if len(pr) != 1:
        return np.nan, np.nan
    P = np.asarray(nodes[p]["xyz"], float) * SCALE
    v_m = P - np.asarray(nodes[pr[0]]["xyz"], float) * SCALE
    cont = min(float(np.linalg.norm(np.asarray(nodes[c]["xyz"], float) * SCALE - P - v_m)) for c in (c1, q))
    c2s = []
    for c in (c1, q):
        s = succ.get(c, [])
        if len(s) == 1:
            C = np.asarray(nodes[c]["xyz"], float) * SCALE
            c2s.append(float(np.linalg.norm(np.asarray(nodes[s[0]]["xyz"], float) * SCALE - C - v_m)))
    return cont, (min(c2s) if c2s else np.nan)


def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}''')
rep('''    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out''',
    '''    r_hp, r_h1, r_hq = _fi_hm(img, t, nodes[p]["xyz"]), _fi_hm(img, t + 1, nodes[c1]["xyz"]), _fi_hm(img, t + 1, nodes[q]["xyz"])
    out.update(r_hm_p=r_hp, r_hm_c1=r_h1, r_hm_q=r_hq,
               rhm_max=(max(r_h1, r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan,
               rhm_sum=((r_h1 + r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan)
    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out''')
rep('''        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
''', '''        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
        row["cont"], row["cont2"] = _vel_cont(nodes, pred, succ, p, c1, q)
''')
# ---- the v2 crash: the fork label array must survive the report's loops ------------------------------
rep('''lab = np.array([r_[4] for r_ in FORKS], int)
''', '''lab = np.array([r_[4] for r_ in FORKS], int)
LAB_F = lab.copy()                      # forkvel2: `lab` is rebound by a loop below; the joint gate reads this copy
''')
rep('_yj = np.concatenate([(lab[counted] == 2).astype(int), Y44])', '_yj = np.concatenate([(LAB_F[counted] == 2).astype(int), Y44])')
# ---- names --------------------------------------------------------------------------------------------
rep('print("FORKIMG OK", flush=True)', 'print("FORKVEL2 OK", flush=True)')
rep('print(f"[degraded] FORKIMG INCOMPLETE or anchor miss:', 'print(f"[degraded] FORKVEL2 INCOMPLETE or anchor miss:')
rep('=== FORKIMG: the image gate against what ships', '=== FORKVEL2: the image+motion gate against what ships')

assert src.count("_vel_cont(") == 2 and src.count("_fi_hm(") == 4 and src.count("LAB_F") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "forkvel2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written forkvel2.py", len(src))
