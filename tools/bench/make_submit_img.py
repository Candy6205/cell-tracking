"""submit_hp.py -> submit_img.py: the SAME pipeline with the fork gate reading the image.

Nothing else changes -- not the heatmaps, not the placement, not the ranker, not the ILP, not
DIV_W, not the patches, not the threshold (p > 0.03). Only the gate: fork_gate_jointimg.pkl,
20 + 11 features, where the 11 read the raw image at the mother (t), her predecessor (t-1) and
the two daughters (t+1): peak within 2 um of a smoothed crop (sigma (0.6, 1.5, 1.5) voxels,
half-size 8 um), contrast over the crop median, radius of gyration within 3.5 um, brightening,
daughter symmetry, daughters/mother ratio -- forkimg's `fork_img_feats`, byte-for-byte the
same arithmetic on the same voxels (the crop is cut from the full frame, then smoothed).

WHY (forkimg, 2026-09-17, 128 6bba movies, official scorer): true forks' mother and daughters
stand 3x over their local background, false forks' 0.9x (AUC 0.83-0.84 per feature); the joint
gate with the image columns keeps 53/50/72 against the shipped 49/51/76, score 0.9457 against
0.9433, paired bootstrap +0.0025 [+0.0004, +0.0049], the edge term untouched. The 44b6 half of
the gate's training forks carries the same columns (fk44img).

COST. A second read of the frames after linking: forks are processed in frame order with a
three-frame cache, so each frame a fork touches is decoded about once (~15-25 s per movie on
the bench; ~1 h over the hidden test on top of the ~6 h run).

ATTACH what submit_hp attached plus faustine649/biohub-gatefit (the pickle).
"""
import os

src = open("submit_hp.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-10a-submit-hp"', 'SCRIPT_VERSION = "2026-09-17a-submit-img"')

# ---- the gate: pickle, features, provenance check --------------------------------------------
rep('GATE_NAME = "fork_gate_joint.pkl"', 'GATE_NAME = "fork_gate_jointimg.pkl"')
rep('''GATE_FEATS = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap"]''',
    '''GATE_FEATS = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap",
              "i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
FEAT_IMG = GATE_FEATS[20:]
FI_HALF_UM = 8.0
FI_PEAK_UM = 2.0
FI_SIZE_UM = 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)''')
rep('''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-divbench8 as a kernel source")''',
    '''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-gatefit as a kernel source")''')
rep('''assert _nt.get("6bba") == 299 and _nt.get("44b6") == 67 and _nt.get("tp") == 56, (
    f"fork_gate_joint.pkl was trained on {_nt}, not divbench8's 299 6bba + 67 44b6 "
    f"counted forks with 56 TP -- this is not the gate the +0.0037 was measured on")''',
    '''assert _nt.get("6bba") == 292 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 56, (
    f"fork_gate_jointimg.pkl was trained on {_nt}, not forkimg's 292 counted 6bba forks plus "
    f"fk44img's 44b6 ones -- this is not the gate forkimg measured")
_gi = _gb.get("image") or {}
assert (float(_gi.get("half_um", -1)), float(_gi.get("peak_um", -1)), float(_gi.get("size_um", -1))) == (FI_HALF_UM, FI_PEAK_UM, FI_SIZE_UM) \\
    and tuple(float(v) for v in _gi.get("sigma_vox", ())) == FI_SIGMA_VOX, f"image feature constants differ: {_gi}"''')

# ---- the image features, forkimg's arithmetic on frames from load_volume ---------------------
rep('''def _gate_prune(nodes, edges, dets):''', '''from scipy import ndimage as _ndi                    # noqa: E402


class _FrameCache:
    """the last three decoded frames of one movie (forks are visited in frame order)."""

    def __init__(self, zp, meta, T):
        self.zp, self.meta, self.T = zp, meta, T
        self.fr = {}
        self.reads = 0

    def get(self, t):
        if t < 0 or t >= self.T:
            return None
        if t not in self.fr:
            if len(self.fr) >= 3:
                del self.fr[min(self.fr)]
            self.fr[t] = load_volume(self.zp, t, self.meta)
            self.reads += 1
        return self.fr[t]


def _fi_node(frame, xyz_vox):
    """(peak, contrast, radius of gyration um) of one node in one frame; NaNs when unreadable."""
    if frame is None:
        return np.nan, np.nan, np.nan
    hz = int(np.ceil(FI_HALF_UM / SCALE[0]))
    hy = int(np.ceil(FI_HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    crop = frame[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return np.nan, np.nan, np.nan
    sm = _ndi.gaussian_filter(crop.astype(np.float32), FI_SIGMA_VOX)
    c = np.asarray(xyz_vox, float) - np.array([z0, y0, x0], float)
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


def _fork_img_feats(fc, nodes, pred, p, c1, q):
    t = int(nodes[p]["t"])
    i_p, c_p, r_p = _fi_node(fc.get(t), nodes[p]["xyz"])
    pr = pred.get(p, [])
    i_pm = _fi_node(fc.get(t - 1), nodes[pr[0]]["xyz"])[0] if len(pr) == 1 else np.nan
    i_1, c_1, r_1 = _fi_node(fc.get(t + 1), nodes[c1]["xyz"])
    i_q, c_q, r_q = _fi_node(fc.get(t + 1), nodes[q]["xyz"])
    return dict(i_p=i_p, c_p=c_p, r_p=r_p,
                br_p=(i_p / i_pm) if (np.isfinite(i_pm) and i_pm > 0) else np.nan,
                c_c1=c_1, c_q=c_q, r_c1=r_1, r_q=r_q,
                sym_i=(abs(i_1 - i_q) / (i_1 + i_q)) if (np.isfinite(i_1) and np.isfinite(i_q) and (i_1 + i_q) > 0) else np.nan,
                sym_r=(abs(r_1 - r_q) / (r_1 + r_q)) if (np.isfinite(r_1) and np.isfinite(r_q) and (r_1 + r_q) > 0) else np.nan,
                dau_ratio=((i_1 + i_q) / (2 * i_p)) if (np.isfinite(i_1) and np.isfinite(i_q) and np.isfinite(i_p) and i_p > 0) else np.nan)


def _gate_prune(nodes, edges, dets, zp=None, meta=None):''')
rep('''    forks = []
    X = []
    for p, ch in succ.items():
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        patched = [q for q in ch if (p, q) in FORK_LOG]''',
    '''    forks = []
    X = []
    fc = _FrameCache(zp, meta, T) if zp is not None else None
    t_img = time.time()
    for p, ch in sorted(succ.items(), key=lambda kv: (nodes[kv[0]]["t"], kv[0])):
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        patched = [q for q in ch if (p, q) in FORK_LOG]''')
rep('''                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap)
        forks.append((p, q))
        X.append([row[f] for f in GATE_FEATS])
    if not forks:
        return edges, (0, 0)''',
    '''                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap)
        row.update(_fork_img_feats(fc, nodes, pred, p, c1, q) if fc is not None else {f: np.nan for f in FEAT_IMG})
        forks.append((p, q))
        X.append([row[f] for f in GATE_FEATS])
    if fc is not None:
        print(f"    image features: {len(forks)} forks, {fc.reads} frame reads, {time.time() - t_img:.0f}s", flush=True)
    if not forks:
        return edges, (0, 0)''')
rep('''        edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets)''',
    '''        edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets, zp, meta)''')

assert src.count("_fork_img_feats(") == 2 and src.count("_FrameCache(") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "submit_img.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written submit_img.py", len(src))
