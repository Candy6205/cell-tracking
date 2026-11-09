"""hsaddle.py -> hmsize.py: the detector's own heatmap as a size measure at every fork node of forkprop_sz2's tables.

WHY (2026-09-25). The user: the fork gate's size reading depends on how size is measured; the raw-image measures
(forkprop_sz2) are one family, and the F4 heatmap -- the network's output, trained to put a fixed-width blob on each
centre -- is no volume, but it separates cell from background where the raw image is flat ("理论上分得清细胞和背景").
The bench cannot run the detector (CPU); this GPU kernel reads forkprop_sz2's fork tables (node coordinates after the
features), computes the four-branch F4 heatmap per frame (hsaddle's stack, equal weights), measures each fork node on
it, and writes a table the offline gate comparison joins on (movie, p, q).

WHAT. hsaddle.py's model / heatmap head verbatim (up to its embedded site list), then: read forkssz2_CR_NOCAP_DV150.csv
(+ forkssz2_dw55.csv, the REF forks) from biohub-forkprop-sz2; the distinct nodes per (movie, frame); per frame the mean
of branch_maps4; per node on the pooled grid (1.625 um isotropic): hpk (max within 2 um), hr50 / hr20 (equal-volume
radius of the component at 0.5 / 0.2 of hpk holding the peak), hmass3 / hmass5 (sum of heatmap - 0.05 within 3 / 5 um),
hprof (radial-profile radius: shell means fall half way from the peak to the 5-7 um shells' mean), hmom (heatmap-weighted
radius of gyration in the 0.5 component). Output hmsize_nodes.csv and hmsize_forks.csv (movie, p, q, label, the measures at
p / c1 / q and the max / sum daughter-to-mother ratios). Budget 10 h, flushed per movie.
Attach: support pack, dctrain-175 weights, cell-tracking, competition, kernel biohub-forkprop-sz2. T4, ~4 h (about 1 s of
heatmap per frame, most frames of 128 movies).
"""
hs = open("hsaddle.py", encoding="utf-8").read()
head = (hs[:hs.index("SITES_B64 = ")] + "\n" + hs[hs.index("UNPOOL_OFF = "):hs.index("def load_sites(")]
        + hs[hs.index("def to_pooled("):hs.index("def profile_peaks(")])   # the pooled-grid helpers sit after the site list; load_sites is skipped
head = head.replace('SCRIPT_VERSION = "2026-09-12a-hsaddle"', 'SCRIPT_VERSION = "2026-09-25a-hmsize"  # hsaddle was "2026-09-12a-hsaddle"', 1)
assert head.count("2026-09-25a-hmsize") == 1 and "def branch_maps4(" in head and "def to_pooled(" in head and "PSCALE = " in head

MAIN = r'''
# ---- hmsize: the heatmap measures -----------------------------------------------------------------------------------
HM_FLOOR = 0.05
HM_KEYS = ["hpk", "hr50", "hr20", "hmass3", "hmass5", "hprof", "hmom"]
HM_RATIO = ["hr50", "hr20", "hmass5", "hprof", "hmom"]
BUDGET_S = 10.0 * 3600


def hm_measures(h, zyx_full):
    """the heatmap measures of one node; h = the pooled F4 map, zyx_full = full-resolution voxel coords"""
    out = {k_: np.nan for k_ in HM_KEYS}
    p = to_pooled(zyx_full)
    Z, Y, X = h.shape
    r = int(np.ceil(8.0 / PSCALE[0]))
    z0, y0, x0 = (int(round(float(v))) for v in p)
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return out
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = np.asarray(h[zs, ys, xs], np.float32)
    zz, yy, xx = np.mgrid[zs, ys, xs]
    d = np.sqrt(((zz - p[0]) * PSCALE[0]) ** 2 + ((yy - p[1]) * PSCALE[1]) ** 2 + ((xx - p[2]) * PSCALE[2]) ** 2)
    near = d <= 2.0
    if not near.any():
        return out
    j = np.unravel_index(int(np.argmax(np.where(near, sub, -np.inf))), sub.shape)
    pk = float(sub[j])
    out["hpk"] = pk
    if pk <= HM_FLOOR:
        return out
    dp = np.sqrt(((zz - zz[j]) * PSCALE[0]) ** 2 + ((yy - yy[j]) * PSCALE[1]) ** 2 + ((xx - xx[j]) * PSCALE[2]) ** 2)
    vox = float(np.prod(PSCALE))

    def _rad(mask):
        if not mask[j]:
            return np.nan
        lab_, _n = ndi.label(mask)
        return float((3.0 * int((lab_ == lab_[j]).sum()) * vox / (4.0 * np.pi)) ** (1.0 / 3.0))

    out["hr50"] = _rad(sub >= 0.5 * pk)
    out["hr20"] = _rad(sub >= 0.2 * pk)
    w_ = np.clip(sub - HM_FLOOR, 0, None)
    out["hmass3"] = float((w_ * (dp <= 3.0)).sum())
    out["hmass5"] = float((w_ * (dp <= 5.0)).sum())
    lab_, _n = ndi.label(sub >= 0.5 * pk)
    comp = lab_ == lab_[j]
    wc = sub[comp]
    out["hmom"] = float(np.sqrt((wc * dp[comp] ** 2).sum() / wc.sum())) if wc.sum() > 0 else np.nan
    shells = np.arange(0.0, 8.0, 1.0)
    prof = np.full(len(shells) - 1, np.nan)
    for i_, (a_, b_) in enumerate(zip(shells[:-1], shells[1:])):
        m_ = (dp >= a_) & (dp < b_)
        if m_.any():
            prof[i_] = float(sub[m_].mean())
    far_ = prof[5:7]
    far = float(np.nanmean(far_)) if np.isfinite(far_).any() else np.nan
    if np.isfinite(far) and pk > far:
        half = far + 0.5 * (pk - far)
        r_ = np.nan
        for i_ in range(len(prof)):
            if np.isfinite(prof[i_]) and prof[i_] < half:
                if i_ == 0:
                    r_ = 0.5
                else:
                    p0, p1 = prof[i_ - 1], prof[i_]
                    x0_, x1_ = shells[i_ - 1] + 0.5, shells[i_] + 0.5
                    r_ = (x0_ + (p0 - half) / (p0 - p1) * (x1_ - x0_)) if (np.isfinite(p0) and p0 > p1) else x1_
                break
        out["hprof"] = float(r_)
    return out


def fork_tables():
    """forkprop_sz2's fork tables with coordinates: [(table, movie, p, q, label, t, zyx_p, zyx_c1, zyx_q)]"""
    rows = []
    for tab in ("CR_NOCAP_DV150", "dw55"):
        hit = sorted(pathlib.Path("/kaggle/input").glob(f"**/forkssz2_{tab}.csv"))
        assert hit, f"attach the kernel faustine649/biohub-forkprop-sz2 (forkssz2_{tab}.csv)"
        with open(hit[0], encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split(",")
            ci = {c_: i_ for i_, c_ in enumerate(hdr)}
            need = ["movie", "p", "q", "label", "t", "z_p", "y_p", "x_p", "z_c1", "y_c1", "x_c1", "z_q", "y_q", "x_q"]
            assert all(c_ in ci for c_ in need), (tab, [c_ for c_ in need if c_ not in ci])
            for line in fh:
                f_ = line.rstrip("\n").split(",")
                if len(f_) < len(hdr):
                    continue
                g = lambda c_: f_[ci[c_]]                                    # noqa: E731
                rows.append((tab, g("movie"), int(g("p")), int(g("q")), int(g("label")), int(float(g("t"))),
                             tuple(float(g(c_)) for c_ in ("z_p", "y_p", "x_p")),
                             tuple(float(g(c_)) for c_ in ("z_c1", "y_c1", "x_c1")),
                             tuple(float(g(c_)) for c_ in ("z_q", "y_q", "x_q"))))
    return rows


def node_key(t, zyx):
    return (int(t),) + tuple(int(round(float(v) * 4)) for v in zyx)          # quarter-voxel rounding


if __name__ == "__main__":
    t_all = time.time()
    ROWS = fork_tables()
    NODES = defaultdict(set)                                  # movie -> {node_key}
    for (_tab, mv, _p, _q, _l, t, zp, zc, zq) in ROWS:
        NODES[mv].add(node_key(t, zp))
        NODES[mv].add(node_key(t + 1, zc))
        NODES[mv].add(node_key(t + 1, zq))
    n_nodes = sum(len(v) for v in NODES.values())
    print(f"fork rows {len(ROWS):,} ({sum(1 for r in ROWS if r[0] == 'CR_NOCAP_DV150'):,} arm + {sum(1 for r in ROWS if r[0] == 'dw55'):,} REF) "
          f"-> {n_nodes:,} distinct nodes in {len(NODES)} movies", flush=True)
    OUT = pathlib.Path("/kaggle/working")
    MEAS = {}                                                 # (movie, node_key) -> measures
    fn = open(OUT / "hmsize_nodes.csv", "w", encoding="utf-8", newline="\n")
    fn.write("movie,t,z,y,x," + ",".join(HM_KEYS) + "\n")
    t0 = time.time()
    n_frames = 0
    done_movies = 0
    for k_, mv in enumerate(sorted(NODES)):
        if time.time() - t_all > BUDGET_S:
            print(f"[budget] stopped before {mv}: {done_movies} movies done", flush=True)
            break
        img = open_image(TRAIN_DIR / f"{mv}.zarr")
        by_t = defaultdict(list)
        for nk in NODES[mv]:
            by_t[nk[0]].append(nk)
        tm = time.time()
        for t in sorted(by_t):
            if t < 0 or t >= img.shape[0]:
                continue
            vol = np.asarray(img[t])
            hs_ = branch_maps4(vol)
            h = sum(hs_) / 4.0                                # F4: equal weights
            for nk in by_t[t]:
                zyx = np.array(nk[1:], float) / 4.0
                m_ = hm_measures(h, zyx)
                MEAS[(mv, nk)] = m_
                fn.write(f"{mv},{t},{zyx[0]:.3f},{zyx[1]:.3f},{zyx[2]:.3f}," + ",".join(f"{m_[c_]:.5g}" if m_[c_] == m_[c_] else "nan" for c_ in HM_KEYS) + "\n")
            n_frames += 1
            del hs_, h, vol
        fn.flush()
        done_movies += 1
        gc.collect()
        print(f"  [{k_ + 1}/{len(NODES)}] {mv}: {len(by_t)} frames, {len(NODES[mv])} nodes, {time.time() - tm:.0f}s "
              f"(total {(time.time() - t0) / 60:.1f} min)", flush=True)
    fn.close()
    with open(OUT / "hmsize_forks.csv", "w", encoding="utf-8", newline="\n") as ff:
        cols = [f"{k_}_{s_}" for k_ in HM_KEYS for s_ in ("p", "c1", "q")] + [f"{k_}_{s_}" for k_ in HM_RATIO for s_ in ("max", "sum")]
        ff.write("table,movie,p,q,label,t," + ",".join(cols) + "\n")
        n_w = 0
        for (tab, mv, p, q, lab, t, zp, zc, zq) in ROWS:
            mp, mc, mq = MEAS.get((mv, node_key(t, zp))), MEAS.get((mv, node_key(t + 1, zc))), MEAS.get((mv, node_key(t + 1, zq)))
            if mp is None or mc is None or mq is None:
                continue
            vals = []
            for k_ in HM_KEYS:
                vals += [mp[k_], mc[k_], mq[k_]]
            for k_ in HM_RATIO:
                a_, b_, c_ = mp[k_], mc[k_], mq[k_]
                ok_ = np.isfinite(a_) and np.isfinite(b_) and np.isfinite(c_) and a_ > 0
                vals += [(max(b_, c_) / a_) if ok_ else np.nan, ((b_ + c_) / a_) if ok_ else np.nan]
            ff.write(f"{tab},{mv},{p},{q},{lab},{t}," + ",".join(f"{v:.5g}" if v == v else "nan" for v in vals) + "\n")
            n_w += 1
    print(f"\nhmsize_forks.csv: {n_w:,} fork rows of {len(ROWS):,}; hmsize_nodes.csv: {len(MEAS):,} nodes over {n_frames:,} frames", flush=True)
    # a first look: the counted forks of the arm, TP against FP, medians per measure
    arm = [(r, MEAS.get((r[1], node_key(r[5], r[6]))), MEAS.get((r[1], node_key(r[5] + 1, r[7]))), MEAS.get((r[1], node_key(r[5] + 1, r[8]))))
           for r in ROWS if r[0] == "CR_NOCAP_DV150" and r[4] >= 1]
    arm = [a_ for a_ in arm if a_[1] is not None and a_[2] is not None and a_[3] is not None]
    print(f"\n=== THE ARM'S COUNTED FORKS ({len(arm)}; TP {sum(1 for a_ in arm if a_[0][4] == 2)}): median of each measure, "
          f"mother / daughter c1 / daughter q, TP against FP; AUC of the max ratio (TP vs FP) ===", flush=True)
    from sklearn.metrics import roc_auc_score
    for k_ in HM_KEYS:
        line = f"  {k_:<7}"
        for lab_, nm_ in ((2, "TP"), (1, "FP")):
            sel = [a_ for a_ in arm if a_[0][4] == lab_]
            med = [float(np.nanmedian([a_[i_][k_] for a_ in sel])) if sel else float("nan") for i_ in (1, 2, 3)]
            line += f"  {nm_} {med[0]:.3g} / {med[1]:.3g} / {med[2]:.3g}"
        if k_ in HM_RATIO:
            y_, s_ = [], []
            for a_ in arm:
                v0, v1, v2 = a_[1][k_], a_[2][k_], a_[3][k_]
                if np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2) and v0 > 0:
                    y_.append(int(a_[0][4] == 2)); s_.append(max(v1, v2) / v0)
            if y_ and 0 < sum(y_) < len(y_):
                line += f"  | max-ratio AUC {roc_auc_score(y_, s_):.3f} (n {len(y_)})"
        print(line, flush=True)
    print(f"HMSIZE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("HMSIZE OK", flush=True)
'''
out = head + MAIN
assert "import gc" in out or "import gc," in out or "gc" in head, "gc import"
open("hmsize.py", "w", encoding="utf-8", newline="\n").write(out)
print("written hmsize.py", out.count("\n"), "lines")
