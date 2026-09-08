"""gtpeak.py -> refinecmp.py: five ways to place a detection, compared on the
one number that matters -- how much invented motion each injects.

WHY. gtpeak measured, over every annotated GT point in the 12 6bba holdout +
12 excluded 44b6 at every frame:

  * the detections F4 finds sit 1.41 / 1.71 um from the GT cell at the median;
  * the CHANGE in that error between a cell and its own next frame is
    1.29 / 1.51 um, which is 47% / 72% of the real frame-to-frame
    displacement (`d_pc1` p50 2.72 / 2.10 um). That is motion no cell made,
    and the ranker's distance and velocity features see it as motion;
  * `_refine` -- the centre of mass of the RAW volume over +-2 z, +-5 yx that
    every detection passes through -- helps the median a little and hurts the
    tail: 1.65 -> 1.39 p50, but 1.93 -> 2.00 p75 on 6bba and 4.19 -> 4.25 p90
    on 44b6. It has never been validated against any alternative.

Most of 6bba's jitter is consistent with the pooled grid itself: 1.625 um
isotropic, independently quantised at t and t+1, gives about 1.15 um against
the observed 1.29. So this is a SUB-VOXEL LOCALISATION problem, and `_refine`
is the stage that is supposed to solve it.

WHY THIS LEVER AND NOT THE OTHERS. It adds ZERO detections. The metric
counts every predicted node in `N_pred` but only scores edges with at least
one GT-matched endpoint, so anything that ADDS cells (splitting, longer
bridges, a second detector) pays node cost for cells the edge term cannot
see -- gtsplit priced that and it closes. Placement is free of all of it, and
it moves 100% of cells rather than the 2.3% that are missing.

THE FIVE RULES, all applied to the SAME peaks, each then NMS'd and matched
independently because the coordinates are what NMS and the greedy 1:1 see:

  nore     the pooled peak, unpooled, no refinement at all -- the floor
  refine   `_refine(vol, c)`, +-2 z / +-5 yx on the RAW volume   <- SHIPS
  refsm    the same on a smaller window, +-1 z / +-2 yx
  hcent    centre of mass of the HEATMAP in +-1 pooled voxel
  hpara    parabolic sub-voxel fit of the HEATMAP along each axis, then
           `_refine` from that starting point

`hcent` and `hpara` are the user's point (2026-09-08) applied where it is
free: the heatmap knows where the cell is to better than a voxel, and
`_refine` currently ignores it and asks the raw intensity instead.

ANCHOR: the `refine` rule IS the shipped path, so it must reproduce gtpeak
exactly -- 6bba 1,047 GT / recall 0.9761 and 44b6 297 / 0.9596 on the
t %% 10 == 0 subset, |offset| p50 1.41 / 1.71, jitter p50 1.29 / 1.51.
"""
import os

src = open("gtpeak.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-gtpeak"', 'SCRIPT_VERSION = "2026-09-08c-refinecmp"')

rep("""def claim_map(det_um, sc, gt_um, radius):""",
    '''RULES = ("nore", "refine", "refsm", "hcent", "hpara")
SHIPPED_RULE = "refine"


def _para1(a, b, c):
    """Sub-voxel offset of the vertex of the parabola through (-1,a) (0,b)
    (1,c). Zero when the three are colinear or b is not the largest."""
    d = a - 2.0 * b + c
    if d >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def place(rule, h, vol, pk):
    """pk: (n, 3) POOLED integer peak indices. Returns unpooled voxel coords.

    Every rule starts from the same peaks, so the comparison is purely about
    where a detection is PUT, never about which detections exist.
    """
    Zp, Yp, Xp = h.shape
    zyx = pk.astype(float)
    if rule in ("hcent", "hpara"):
        for i in range(len(pk)):
            z0, y0, x0 = (int(v) for v in pk[i])
            sl = (slice(max(0, z0 - 1), min(Zp, z0 + 2)),
                  slice(max(0, y0 - 1), min(Yp, y0 + 2)),
                  slice(max(0, x0 - 1), min(Xp, x0 + 2)))
            w = h[sl].astype(np.float64)
            if rule == "hcent":
                s = float(w.sum())
                if s <= 0:
                    continue
                zz, yy, xx = np.mgrid[sl]
                zyx[i] = [(zz * w).sum() / s, (yy * w).sum() / s,
                          (xx * w).sum() / s]
            else:
                b = float(h[z0, y0, x0])
                for ax, (lo, hi, c0) in enumerate(((0, Zp, z0), (0, Yp, y0),
                                                   (0, Xp, x0))):
                    if c0 - 1 < lo or c0 + 1 >= hi:
                        continue
                    idx_m = [z0, y0, x0]
                    idx_p = [z0, y0, x0]
                    idx_m[ax] -= 1
                    idx_p[ax] += 1
                    zyx[i][ax] += _para1(float(h[tuple(idx_m)]), b,
                                         float(h[tuple(idx_p)]))
    coords = zyx.copy()
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    if rule == "nore":
        return coords
    if rule == "refine":
        return np.array([_refine(vol, c) for c in coords])
    if rule == "refsm":
        return np.array([_refine(vol, c, rz=1, ryx=2) for c in coords])
    if rule == "hcent":
        return coords
    return np.array([_refine(vol, c) for c in coords])   # hpara + _refine


def claim_map(det_um, sc, gt_um, radius):''')

# ---- the per-frame loop measures every rule -------------------------------
rep("""        cm, scm, raw, sc_raw, ref, h = detect_stages(hs, vol, F4_W)""",
    """        cm, scm, raw, sc_raw, ref, h = detect_stages(hs, vol, F4_W)
        pk_i = np.rint(np.stack([raw[:, 0], (raw[:, 1] - (POOL - 1) / 2) / POOL,
                                 (raw[:, 2] - (POOL - 1) / 2) / POOL], 1)
                       ).astype(int) if len(raw) else np.zeros((0, 3), int)
        for rl in RULES:
            rc_ = place(rl, h, vol, pk_i) if len(pk_i) else np.zeros((0, 3))
            rc_, rs_ = _physical_nms(rc_, sc_raw, NMS_UM) if len(rc_) \\
                else (rc_, sc_raw)
            kk_ = rs_ >= GATE_SHIP
            ru_ = (rc_[kk_] * SCALE[None, :]) if kk_.any() else np.zeros((0, 3))
            g_ = claim_map(ru_, rs_[kk_], gt_um, MATCH_R)
            for j_ in range(len(gt_vox)):
                k_ = int(g_[j_])
                o_ = (ru_[k_] - gt_um[j_]) if k_ >= 0 else np.full(3, np.nan)
                RULE_ROWS.setdefault(rl, []).append(
                    (tag, int(t_), int(nid[j_]) if nid is not None else -1,
                     k_ >= 0, float(o_[0]), float(o_[1]), float(o_[2])))""")
rep("""        nid = NID_OF.get((tag, t_))
        for j in range(len(gt_vox)):""",
    """        for j in range(len(gt_vox)):""")
rep("""        hs = branch_maps6(vol, vol_other, idx, q)""",
    """        nid = NID_OF.get((tag, t_))
        hs = branch_maps6(vol, vol_other, idx, q)""")
rep("""    R = []
    NF = 0""",
    """    R = []
    NF = 0
    RULE_ROWS.clear()""")
rep("""def run_domain(domain, frame_iter, dense):""",
    """RULE_ROWS = {}


def run_domain(domain, frame_iter, dense):""")

# ---- the comparison replaces gtpeak's single localisation block -----------
A = src.index("""    # ---- Q3: localisation ---""")
B = src.index("""    with open(f"/kaggle/working/gtpeak_{domain}.csv", "w") as fh:""")
CMP = r'''    # ---- the five placement rules ----------------------------------------
    print(f"\n  --- WHERE TO PUT A DETECTION: five rules, same peaks ---",
          flush=True)
    print(f"    {'rule':<7} {'recall':>8} {'|offset| p50':>13} {'p90':>7} "
          f"{'jitter p50':>11} {'p90':>7} {'>1um':>7} {'>2um':>7}   "
          f"signed z / y / x", flush=True)
    JIT = {}
    for rl in RULES:
        rr = RULE_ROWS.get(rl, [])
        if len(rr) != len(R):
            warn_once("rule_len", f"{rl}: {len(rr)} rows for {len(R)} GT points")
            continue
        rtag = np.array([x[0] for x in rr])
        rnid = np.array([x[2] for x in rr], int)
        rmt = np.array([x[3] for x in rr], bool)
        roff = np.array([[x[4], x[5], x[6]] for x in rr], float)
        rmag = np.linalg.norm(roff, axis=1)
        pos_ = {(rtag[i], rnid[i]): i for i in range(len(rr))
                if rmt[i] and rnid[i] >= 0}
        ch_ = []
        for i in range(len(rr)):
            if not rmt[i] or rnid[i] < 0:
                continue
            nx_ = SUCC_OF.get(rtag[i], {}).get(int(rnid[i]))
            j_ = pos_.get((rtag[i], nx_)) if nx_ is not None else None
            if j_ is None:
                continue
            ch_.append(float(np.linalg.norm(roff[j_] - roff[i])))
        ch_ = np.asarray(ch_, float)
        JIT[rl] = (float(rmt.mean()), float(np.nanpercentile(rmag[rmt], 50)),
                   float(np.nanpercentile(ch_, 50)) if len(ch_) else float("nan"))
        anc = ""
        if rl == SHIPPED_RULE:
            want = (1.41, 1.29) if domain == "6bba" else (1.71, 1.51)
            got = (np.nanpercentile(rmag[rmt], 50),
                   np.nanpercentile(ch_, 50) if len(ch_) else float("nan"))
            ok = abs(got[0] - want[0]) < 0.02 and abs(got[1] - want[1]) < 0.02
            anc = f"   ANCHOR want {want[0]:.2f} / {want[1]:.2f} " \
                  f"{'OK' if ok else '*** MISMATCH ***'}"
            if not ok:
                warn_once("refine_anchor",
                          f"{domain}: the shipped rule reads {got[0]:.2f} / "
                          f"{got[1]:.2f} against gtpeak's {want}")
        print(f"    {rl:<7} {rmt.mean():>8.4f} "
              f"{np.nanpercentile(rmag[rmt], 50):>13.2f} "
              f"{np.nanpercentile(rmag[rmt], 90):>7.2f} "
              + (f"{np.nanpercentile(ch_, 50):>11.2f} "
                 f"{np.nanpercentile(ch_, 90):>7.2f} "
                 f"{100.0 * np.mean(ch_ > 1.0):>6.1f}% "
                 f"{100.0 * np.mean(ch_ > 2.0):>6.1f}%" if len(ch_) else
                 f"{'--':>11} {'--':>7} {'--':>7} {'--':>7}")
              + f"   {np.nanmean(roff[rmt, 0]):+.3f} / "
                f"{np.nanmean(roff[rmt, 1]):+.3f} / "
                f"{np.nanmean(roff[rmt, 2]):+.3f}" + anc, flush=True)
    if SHIPPED_RULE in JIT:
        b_ = JIT[SHIPPED_RULE]
        print(f"\n    against the shipped rule ({SHIPPED_RULE}): "
              f"d recall / d |offset| p50 / d jitter p50", flush=True)
        for rl in RULES:
            if rl == SHIPPED_RULE or rl not in JIT:
                continue
            v = JIT[rl]
            print(f"      {rl:<7} {v[0] - b_[0]:>+8.4f} {v[1] - b_[1]:>+8.2f} um "
                  f"{v[2] - b_[2]:>+8.2f} um", flush=True)
    print(f"    (the pooled grid is 1.625 um isotropic, so a rule that cannot "
          f"beat `nore` by about 0.4 um is not doing sub-voxel work at all. "
          f"The real frame-to-frame displacement is p50 "
          f"{'2.72' if domain == '6bba' else '2.10'} um, so the jitter column "
          f"is a share of THAT, not of the offset column.)", flush=True)

'''
src = src[:A] + CMP + src[B:]

rep('GTPEAK DONE in', 'REFINECMP DONE in')
rep('print("GTPEAK OK", flush=True)', 'print("REFINECMP OK", flush=True)')
rep("gtpeak_{domain}.csv", "refinecmp_{domain}.csv", n=2)
rep('''print("  (3) For question 3 read the SIGNED means against the stds.''',
    '''print("  (3) THE DECISION: the jitter p50 column, because that is the invented "
      "motion the ranker reads as real. `refine` is what ships and carries the "
      "anchor; anything that beats it there, without losing recall, is a free "
      "improvement -- it adds no nodes, so none of gtsplit's node-cost "
      "arithmetic applies to it.", flush=True)
print("  (3b) `nore` is the floor: it is the raw pooled grid with no sub-voxel "
      "work at all. If `refine` barely beats `nore`, the stage that exists to "
      "solve this problem is not solving it, and hcent / hpara say whether the "
      "HEATMAP knows better than the raw intensity does.", flush=True)
print("  (3c) For the signed means read them against the stds.''')


# ---- drop what the replaced localisation block used to consume -----------
rep("""    tag = np.array([r[0] for r in R])
    tt = np.array([r[1] for r in R], int)
    nid = np.array([r[2] for r in R], int)
""", """    tt = np.array([r[1] for r in R], int)
""")
rep("""    d_ref = np.array([r[7] for r in R], float)
""", "")
rep("""    off = np.array([[r[10], r[11], r[12]] for r in R], float)
    mag = np.linalg.norm(off, axis=1)
""", "")
_BS = chr(92)     # the CMP block is a raw string, so this `\n` is two chars
rep('print(f"' + _BS + 'n  --- WHERE TO PUT A DETECTION',
    'print("' + _BS + 'n  --- WHERE TO PUT A DETECTION')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "refinecmp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
