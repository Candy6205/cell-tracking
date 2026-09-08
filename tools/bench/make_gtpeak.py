"""detfuse3.py -> gtpeak.py: where F4's missed cells actually go, on 6bba and
44b6 only, and how far off the ones it FINDS are.

THE USER'S THREE QUESTIONS, 2026-09-08:

  1. is there a peak on the heatmap at all?
  2. is there still a peak AFTER non-maximum suppression?
  3. localisation -- being off by 7 um is unlikely, but "off a bit here, off a
     bit there" could bend the links; measure that too.

WHAT IS ALREADY KNOWN, and why it does not answer this. detfuse3 v3 measured
the window-max of the F4 blend near F4's misses ON ZEBRAHUB: >= 0.05 within
3.25 um for **99.6%** of 16,226 misses. "F4's misses are not cold in the
heatmap: the map responds and the peak / NMS stage loses them." But:

  * that is Zebrahub. On 6bba / 44b6 the same table had n = 8 and n = 4 and
    the entry says outright that it "FAILED to inform".
  * a window MAX is not a peak. It cannot separate "peak_local_max never
    emitted anything there" from "it did and `_physical_nms` killed it",
    which is exactly questions 1 and 2.
  * localisation was never measured at all, at any venue.

WHAT THIS RUN DOES. Every annotated GT point in the 12 6bba holdout + 12
excluded 44b6 datasets, at EVERY frame, gets a row:

  h_gt, h_win   the F4 blend at the GT voxel and its max within +-2 pooled
                voxels (3.25 um) -- comparable to detfuse3's zh column
  d_raw         distance to the nearest RAW peak_local_max output, before
                _refine and before NMS                      <- question 1
  d_ref         the same after _refine (the centre-of-mass step nobody has
                ever validated)
  d_nms         distance to the nearest SURVIVING detection <- question 2
  offset        the signed per-axis error of the detection that claims it,
                and the CHANGE in that error along the GT track  <- question 3

STRIDE 1 IS NOT OPTIONAL. detfuse3 ran every 10th frame, which is why n was
8 and 4. Stride 1 gives ~10x the misses AND -- structurally -- consecutive
frames, without which the frame-to-frame offset change cannot be computed at
all. That is the measurement question 3 actually asks for: a detection that
sits +2 um off in frame t and -2 um off in frame t+1 injects 4 um of motion
that no cell made, and that is what bends a link.

ANCHOR. The stride-10 frames are a SUBSET of stride 1 (both are t = 0, 10,
20, ...), so detfuse3 v4's numbers must reproduce exactly on that subset:
**6bba 1,047 GT points, recall 0.9761; 44b6 297 GT points, recall 0.9596**,
F4 at gate 0.15, 7 um, greedy 1:1. The GT COUNTS are part of the anchor --
if the frame set drifted, they move first.

The official detector is dropped entirely (user, 2026-09-08: we cannot know
what it trained on). Its two maps are still computed as zeros so the F4
weights stay literally unchanged; nothing reads them.
"""
import os

src = open("detfuse3.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-04a-detfuse3-curves"', 'SCRIPT_VERSION = "2026-09-08a-gtpeak"')

# ---- every frame, and only the competition domains ------------------------
rep("COMP_STRIDE = 10", "COMP_STRIDE = 1        # detfuse3 used 10; that is why n was 8 and 4")
rep('run_domain("zh", zh_frames(), dense=True)',
    '# gtpeak drops zebrahub: the question is about 6bba / 44b6.\n'
    '# run_domain("zh", zh_frames(), dense=True)')

# ---- the two official maps are never read; return zeros and skip the work --
rep('''    hs.extend(official_maps(vol, vol_other, idx, q, hs[0].shape))''',
    '''    if SKIP_OFFICIAL:
        # F4's weights on these two are exactly 0.0, so zeros are numerically
        # identical to computing them -- and this is a third of the forward
        # passes at stride 1.
        hs.extend([np.zeros_like(hs[0]), np.zeros_like(hs[0])])
    else:
        hs.extend(official_maps(vol, vol_other, idx, q, hs[0].shape))''')

# ---- detect_weighted also hands back the pre-NMS stages -------------------
rep('''def detect_weighted(hs, vol, weights):
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c) for c in coords])
    return _physical_nms(ref, sc, NMS_UM)''',
    '''def detect_weighted(hs, vol, weights):
    return detect_stages(hs, vol, weights)[:2]


def detect_stages(hs, vol, weights):
    """The same detection, with every intermediate kept. detfuse3 returned
    only the survivors, which cannot answer 'was there a peak before NMS'."""
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        z0 = np.zeros((0, 3))
        return z0, np.zeros(0), z0, np.zeros(0), z0, h
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    raw = coords.copy()
    ref = np.array([_refine(vol, c) for c in coords])
    cm, scm = _physical_nms(ref, sc, NMS_UM)
    return cm, scm, raw, sc, ref, h''')

# ---- GT rows need the frame index and the track, not just coordinates -----
rep('''        for t in range(0, T, COMP_STRIDE):
            gi = gbt.get(t)
            if not gi:
                continue''',
    '''        for t in range(0, T, COMP_STRIDE):
            gi = gbt.get(t)
            if not gi:
                continue
            NID_OF[(name, t)] = np.asarray(nbt[t], np.int64)''')
rep('''        gbt = defaultdict(list)
        for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
            gbt[int(tt)].append([float(z_), float(y_), float(x_)])''',
    '''        gbt = defaultdict(list)
        nbt = defaultdict(list)
        for nid_, tt, z_, y_, x_ in zip(gt["node_ids"], gt["t"], gt["z"],
                                        gt["y"], gt["x"]):
            gbt[int(tt)].append([float(z_), float(y_), float(x_)])
            nbt[int(tt)].append(int(nid_))
        SUCC_OF[name] = {int(a_): int(b_) for a_, b_ in gt["edges"]}''')
rep('''            yield name, vol, np.asarray(gi, float), other, idx, q''',
    '''            yield name, vol, np.asarray(gi, float), other, idx, q, t''')

# ---- the measurement replaces detfuse3's arm sweep ------------------------
A = src.index("def run_domain(domain, frame_iter, dense):")
# end BEFORE the zebrahub mounting block: it defines t0, files and
# FRAME_STRIDE, which the rest of the script still needs. Only the
# run_domain("zh", ...) CALL is dropped, not the scaffolding.
B = src.index("# ======================= zebrahub ")
BODY = r'''SKIP_OFFICIAL = True
F4_W = dict(ARMS)["F4"]
assert F4_W[4] == 0.0 and F4_W[5] == 0.0, F4_W
GATE_SHIP = 0.15            # what ships
MATCH_R = 7.0               # detfuse3's radius for the recall anchor
NID_OF = {}                 # (dataset, t) -> GT node ids, in gt_vox order
SUCC_OF = {}                # dataset -> {node id: next node id}
ANCHOR_RECALL = {"6bba": (1047, 0.9761), "44b6": (297, 0.9596)}


def claim_map(det_um, sc, gt_um, radius):
    """detfuse3's greedy 1:1, but returning WHICH detection took each GT
    point -- the recall is unchanged, the offsets are what is new."""
    g2d = np.full(len(gt_um), -1, np.int64)
    if not len(det_um) or not len(gt_um):
        return g2d
    tree = cKDTree(gt_um)
    taken = np.zeros(len(gt_um), bool)
    for i in np.argsort(-sc):
        near = tree.query_ball_point(det_um[i], radius)
        if not near:
            continue
        near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
        free = [g for g in near if not taken[g]]
        if free:
            taken[free[0]] = True
            g2d[free[0]] = i
    return g2d


def _nearest(pts_um, gt_um):
    if not len(pts_um):
        return np.full(len(gt_um), np.inf)
    return cKDTree(pts_um).query(gt_um, k=1)[0]


def run_domain(domain, frame_iter, dense):
    """One row per annotated GT point: what the heatmap said there, which
    detection stage still had it, and how far off the winner was."""
    R = []
    NF = 0
    for tag, vol, gt_vox, vol_other, idx, q, t_ in frame_iter:
        gt_um = gt_vox * SCALE[None, :]
        NF += 1
        hs = branch_maps6(vol, vol_other, idx, q)
        cm, scm, raw, sc_raw, ref, h = detect_stages(hs, vol, F4_W)
        keep = scm >= GATE_SHIP
        g2d = claim_map(cm[keep] * SCALE[None, :], scm[keep], gt_um, MATCH_R)
        cm_um = (cm[keep] * SCALE[None, :]) if keep.any() else np.zeros((0, 3))
        gk = sc_raw >= GATE_SHIP
        d_raw = _nearest(raw[gk] * SCALE[None, :] if gk.any() else raw[:0], gt_um)
        d_raw0 = _nearest(raw * SCALE[None, :], gt_um)
        d_ref = _nearest(ref[gk] * SCALE[None, :] if gk.any() else ref[:0], gt_um)
        d_nms = _nearest(cm_um, gt_um)
        Zp, Yp, Xp = h.shape
        nid = NID_OF.get((tag, t_))
        for j in range(len(gt_vox)):
            zc = int(round(gt_vox[j, 0]))
            yc = int(round(gt_vox[j, 1] / POOL))
            xc = int(round(gt_vox[j, 2] / POOL))
            hg = (float(h[zc, yc, xc])
                  if 0 <= zc < Zp and 0 <= yc < Yp and 0 <= xc < Xp else np.nan)
            sl = (slice(max(0, zc - 2), min(Zp, zc + 3)),
                  slice(max(0, yc - 2), min(Yp, yc + 3)),
                  slice(max(0, xc - 2), min(Xp, xc + 3)))
            hw = (float(h[sl].max())
                  if sl[0].start < sl[0].stop and sl[1].start < sl[1].stop
                  and sl[2].start < sl[2].stop else np.nan)
            k = int(g2d[j])
            off = (cm_um[k] - gt_um[j]) if k >= 0 else np.full(3, np.nan)
            R.append((tag, int(t_), int(nid[j]) if nid is not None else -1,
                      hg, hw, float(d_raw0[j]), float(d_raw[j]),
                      float(d_ref[j]), float(d_nms[j]), k >= 0,
                      float(off[0]), float(off[1]), float(off[2])))
        del hs, h
    _report(domain, R, NF)


def _q(a, qs=(50, 75, 90, 95, 99)):
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    if not len(a):
        return "  (empty)"
    return "  ".join(f"p{p}={np.percentile(a, p):.2f}" for p in qs)


def _report(domain, R, NF):
    tag = np.array([r[0] for r in R])
    tt = np.array([r[1] for r in R], int)
    nid = np.array([r[2] for r in R], int)
    hg = np.array([r[3] for r in R], float)
    hw = np.array([r[4] for r in R], float)
    d_raw0 = np.array([r[5] for r in R], float)
    d_raw = np.array([r[6] for r in R], float)
    d_ref = np.array([r[7] for r in R], float)
    d_nms = np.array([r[8] for r in R], float)
    mt = np.array([r[9] for r in R], bool)
    off = np.array([[r[10], r[11], r[12]] for r in R], float)
    mag = np.linalg.norm(off, axis=1)

    print(f"\n{'=' * 78}\n=== {domain}: {len(R):,} annotated GT points over "
          f"{NF} frames, EVERY frame ===\n{'=' * 78}", flush=True)

    # ---- the anchor: detfuse3 v4's frames are the t % 10 == 0 subset ------
    sub = (tt % 10) == 0
    want_n, want_r = ANCHOR_RECALL[domain]
    got_n, got_r = int(sub.sum()), float(mt[sub].mean()) if sub.any() else 0.0
    ok = (got_n == want_n) and abs(got_r - want_r) < 5e-4
    print(f"\n  ANCHOR (t % 10 == 0, F4 gate {GATE_SHIP}, {MATCH_R} um, greedy "
          f"1:1): {got_n:,} GT points, recall {got_r:.4f}   "
          f"want {want_n:,} / {want_r:.4f}   {'OK' if ok else '*** MISMATCH ***'}",
          flush=True)
    print(f"  full set: recall {mt.mean():.4f} ({int(mt.sum()):,} of {len(R):,})",
          flush=True)

    # ---- Q1 + Q2: where the missed ones go -------------------------------
    ms = ~mt
    n = int(ms.sum())
    print(f"\n  --- WHERE THE {n:,} MISSES GO (question 1 and 2) ---", flush=True)
    if n:
        no_raw = ms & ~(d_raw <= MATCH_R)
        cold = no_raw & ~(hw >= 0.05)
        warm = no_raw & (hw >= 0.05)
        had_raw = ms & (d_raw <= MATCH_R)
        killed = had_raw & ~(d_nms <= MATCH_R)
        survived = had_raw & (d_nms <= MATCH_R)
        rows = [
            ("no peak, and the map is COLD there (h_win < 0.05)", cold,
             "a better detector is the only fix"),
            ("no peak, but the map RESPONDS (h_win >= 0.05)", warm,
             "peak extraction, not detector capacity"),
            ("had a raw peak, NMS removed it", killed,
             f"NMS_UM {NMS_UM} is the lever"),
            ("a detection survived within 7 um, the 1:1 gave it away", survived,
             "a matching / duplicate problem, not detection"),
        ]
        print(f"    {'bucket':<52} {'n':>6} {'share':>7}   fix", flush=True)
        for name, m, fix in rows:
            print(f"    {name:<52} {int(m.sum()):>6} "
                  f"{100.0 * m.sum() / n:>6.1f}%   {fix}", flush=True)
        assert int(cold.sum() + warm.sum() + killed.sum() + survived.sum()) == n
        print(f"\n    h_win over the misses: {_q(hw[ms])}", flush=True)
        print(f"    h at the GT voxel:     {_q(hg[ms])}", flush=True)
        print(f"    >= 0.05 within 3.25 um: "
              f"{100.0 * np.nanmean(hw[ms] >= 0.05):.1f}%   >= 0.15: "
              f"{100.0 * np.nanmean(hw[ms] >= 0.15):.1f}%"
              f"   (detfuse3 on zebrahub: 99.6% / 85.6%)", flush=True)
        print(f"    nearest raw peak at ANY score: {_q(d_raw0[ms])}", flush=True)

    # ---- Q3: localisation ------------------------------------------------
    print(f"\n  --- LOCALISATION of the {int(mt.sum()):,} that ARE found "
          f"(question 3) ---", flush=True)
    print(f"    |offset| um:            {_q(mag[mt])}", flush=True)
    print(f"    mean SIGNED offset um:  z {np.nanmean(off[mt, 0]):+.3f}  "
          f"y {np.nanmean(off[mt, 1]):+.3f}  x {np.nanmean(off[mt, 2]):+.3f}"
          f"   (a bias bends every link the same way; jitter does not)",
          flush=True)
    print(f"    std per axis um:        z {np.nanstd(off[mt, 0]):.3f}  "
          f"y {np.nanstd(off[mt, 1]):.3f}  x {np.nanstd(off[mt, 2]):.3f}",
          flush=True)
    print(f"    did _refine help? distance to nearest peak, matched GT only:"
          f"\n      before _refine (raw): {_q(d_raw[mt])}"
          f"\n      after  _refine:       {_q(d_ref[mt])}", flush=True)

    # ---- Q3, the form that actually bends links --------------------------
    pos = {(tag[i], nid[i]): i for i in range(len(R)) if mt[i] and nid[i] >= 0}
    ch = []
    for i in range(len(R)):
        if not mt[i] or nid[i] < 0:
            continue
        nx = SUCC_OF.get(tag[i], {}).get(int(nid[i]))
        j = pos.get((tag[i], nx)) if nx is not None else None
        if j is None:
            continue
        ch.append(float(np.linalg.norm(off[j] - off[i])))
    print(f"\n    THE ONE THAT BENDS LINKS: the CHANGE in offset between a "
          f"cell and its own next frame,\n    along {len(ch):,} GT track "
          f"steps. This is motion no cell made.", flush=True)
    if ch:
        print(f"      |offset(t+1) - offset(t)| um: {_q(ch)}", flush=True)
        print(f"      share of steps above 1 um: "
              f"{100.0 * np.mean(np.asarray(ch) > 1.0):.1f}%   above 2 um: "
              f"{100.0 * np.mean(np.asarray(ch) > 2.0):.1f}%", flush=True)

    with open(f"/kaggle/working/gtpeak_{domain}.csv", "w") as fh:
        fh.write("dataset,t,node_id,h_gt,h_win,d_raw_any,d_raw,d_ref,d_nms,"
                 "matched,off_z,off_y,off_x\n")
        for r in R:
            fh.write(",".join(str(v) for v in r) + "\n")
    print(f"\n  gtpeak_{domain}.csv written ({len(R):,} rows)", flush=True)


'''
src = src[:A] + BODY + src[B:]

rep('''print(f"\\nDETFUSE3 DONE in {(time.time() - t0) / 60:.1f} min", flush=True)''',
    '''print("\\n=== reading it ===", flush=True)
print("  (1) THE ANCHOR FIRST. detfuse3 v4's frames are the t % 10 == 0 subset of "
      "this one, so 1,047 / 0.9761 and 297 / 0.9596 must come back exactly. The GT "
      "COUNT is the sharper half: if the frame set drifted it moves before the "
      "recall does.", flush=True)
print("  (2) The four miss buckets are the answer to questions 1 and 2, and they are "
      "ordered by how expensive the fix is. Only the FIRST needs a better detector. "
      "On zebrahub detfuse3 found 99.6% of misses have h_win >= 0.05, i.e. the first "
      "bucket was nearly empty there -- if it is nearly empty here too, then "
      "'the detector cannot see them' is wrong on competition data as well, and "
      "NMS_UM 4.0 (never swept) and peak extraction are the live levers.", flush=True)
print("  (3) For question 3 read the SIGNED means against the stds. A bias moves every "
      "detection the same way and mostly cancels in a link; jitter does not. The "
      "pooled grid is 1.625 um isotropic, so anything under ~0.8 um is sub-voxel and "
      "cannot be fixed by better peak picking.", flush=True)
print("  (4) The last table is the one the question was really about. A cell whose "
      "detection sits +2 um off at t and -2 um off at t+1 has 4 um of invented "
      "motion, and the ranker's distance and velocity features see exactly that. "
      "Compare its p50 to the real frame-to-frame displacement before calling it "
      "large or small.", flush=True)
print(f"\\nGTPEAK DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("GTPEAK OK", flush=True)''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gtpeak.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
