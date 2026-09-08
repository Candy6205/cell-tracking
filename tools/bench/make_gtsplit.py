"""gtpeak.py -> gtsplit.py: split every blob and count what comes back.

THE QUESTION (user, 2026-09-08): "先试试全劈能捞回多少" -- price the ceiling of
blob splitting before anyone builds a splitter, because the obvious worry is
that it floods the pipeline with cells.

WHY IT IS A CEILING AND NOT A PROPOSAL. The user also killed the analogy this
was reaching for: the fork gate is a trained MODEL, and it exists because
every fork the pipeline emits can be labelled TP/FP by the official scorer.
No such labels exist for blobs -- GT annotates 9.71% of 6bba's cells and 0.77%
of 44b6's, so for almost any blob nobody knows whether it holds one cell or
two. A classifier is not available here. What IS available is the ceiling:
split everything, and see how many of gtpeak's 313 recoverable cells come
back and at what cost in detections.

HOW TO SPLIT, and why the obvious method cannot work. gtpeak found the missed
cells have NO local maximum -- their nearest real peak is ~6 um away, on a
neighbour. Anything driven by intensity maxima (h-maxima, prominence,
watershed seeded by peak_local_max) therefore returns the same single peak and
splits nothing. The split has to come from SHAPE: threshold the heatmap into a
mask, distance-transform it with the anisotropic voxel size, seed on maxima of
the DISTANCE map, and watershed that. A dumbbell-shaped mask has two distance
maxima even where the intensity has one.

THE INTERACTION THAT COULD VOID THE WHOLE IDEA. `NMS_UM` is 4.0 um and two
cells sharing a blob are usually closer than that, so ordinary NMS would
re-merge exactly what the split just separated. Arms are therefore reported
both without NMS (the true ceiling) and with it (what the pipeline would
actually see).

ARMS
  BASE          gtpeak's detection, unchanged -- carries the anchor
  s05           mask h >= 0.05, watershed on the distance map, no NMS
  s15           mask h >= 0.15, no NMS
  s15n          mask h >= 0.15, then the pipeline's NMS at 4.0 um

READ IT AS: for each arm, detections per frame (the cost) against recall and
-- the number the question asks for -- how many of BASE's misses it recovers,
broken down by gtpeak's buckets. A split that recovers the 313 while doubling
detections is a real answer to "how many", and still a bad trade; say both.

ANCHOR: BASE must reproduce 6bba 1,047 GT / recall 0.9761 and 44b6 297 /
0.9596 on the t % 10 == 0 subset, and its miss buckets must reproduce
gtpeak's 4 / 224 / 6 / 1 and 1 / 89 / 3 / 11.
"""
import os

src = open("gtpeak.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-gtpeak"', 'SCRIPT_VERSION = "2026-09-08b-gtsplit"')

rep("from skimage.feature import peak_local_max",
    "from skimage.feature import peak_local_max\n"
    "from skimage.measure import label as _cc_label\n"
    "from skimage.segmentation import watershed as _watershed\n"
    "from scipy.ndimage import distance_transform_edt as _edt")

# ---- the splitter, and the arms ------------------------------------------
rep("""def claim_map(det_um, sc, gt_um, radius):""",
    '''POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
# (name, mask threshold, NMS radius or None, minimum separation of watershed
# seeds in pooled voxels). seed 1 is the maximally permissive ceiling and it
# OVER-segments: the local test shattered two equal-brightness cells 6.5 um
# apart into five detections. seed 2 (3.25 um) is the same idea lightly
# regularised, so that the dets/fr column reads as a cost rather than as an
# artifact of the seeding.
SPLIT_ARMS = [("s05", 0.05, None, 1), ("s15", 0.15, None, 1),
              ("s15d", 0.15, None, 2), ("s15n", 0.15, NMS_UM, 2)]


def split_detect(h, vol, thr, nms_um, seed_min=1):
    """One detection per SHAPE-separated lobe of the mask h >= thr.

    Intensity maxima are useless here by construction -- gtpeak showed the
    cells this is meant to recover have none. So: binarise, distance-transform
    with the real voxel size (1.625 x 1.625 x 1.625 um on the pooled grid),
    seed on maxima of the DISTANCE map, watershed. A blob holding two cells is
    dumbbell-shaped and has two distance maxima even with one intensity peak.
    """
    mask = h >= thr
    if not mask.any():
        return np.zeros((0, 3)), np.zeros(0)
    dist = _edt(mask, sampling=POOLED)
    seeds = peak_local_max(dist, min_distance=seed_min, threshold_abs=1e-6,
                           labels=_cc_label(mask), exclude_border=False)
    if len(seeds) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    mk = np.zeros(h.shape, np.int32)
    mk[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    lab = _watershed(-dist, mk, mask=mask)
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    # each lobe is scored and placed at its own brightest voxel, so a lobe
    # that is merely part of a bright cell keeps that cell's score and cannot
    # be mistaken for a new, better detection
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    starts = np.searchsorted(fs, np.arange(1, n + 1), "left")
    ends = np.searchsorted(fs, np.arange(1, n + 1), "right")
    hv = h.ravel()
    idx = []
    for a_, b_ in zip(starts, ends):
        if b_ > a_:
            vox = order[a_:b_]
            idx.append(int(vox[int(np.argmax(hv[vox]))]))
    if not idx:
        return np.zeros((0, 3)), np.zeros(0)
    zyx = np.array(np.unravel_index(np.asarray(idx), h.shape), float).T
    sc = hv[np.asarray(idx)].astype(float)
    coords = zyx.copy()
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c) for c in coords])
    if nms_um is not None:
        ref, sc = _physical_nms(ref, sc, nms_um)
    return ref, sc


def claim_map(det_um, sc, gt_um, radius):''')

# ---- run_domain also runs the split arms ---------------------------------
rep("""        nid = NID_OF.get((tag, t_))""",
    """        for anm, athr, anms, asd in SPLIT_ARMS:
            ac, asc = split_detect(h, vol, athr, anms, asd)
            ak = asc >= GATE_SHIP
            ag = claim_map(ac[ak] * SCALE[None, :] if ak.any() else ac[:0],
                           asc[ak], gt_um, MATCH_R)
            ARM_HIT.setdefault(anm, []).extend((ag >= 0).tolist())
            ARM_ND[anm] = ARM_ND.get(anm, 0) + int(ak.sum())
        ARM_NF[0] += 1
        nid = NID_OF.get((tag, t_))""")
rep("""def run_domain(domain, frame_iter, dense):""",
    """ARM_HIT, ARM_ND, ARM_NF = {}, {}, [0]


def run_domain(domain, frame_iter, dense):""")
rep("""    R = []
    NF = 0""",
    """    R = []
    NF = 0
    ARM_HIT.clear()
    ARM_ND.clear()
    ARM_NF[0] = 0""")

# ---- the arm report, right after the bucket table -------------------------
rep("""    # ---- Q3: localisation ------------------------------------------------""",
    '''    # ---- the split arms --------------------------------------------------
    print("\\n  --- FULL SPLIT: every blob cut at its shape saddles ---", flush=True)
    print(f"    {'arm':<6} {'mask':>5} {'NMS':>5} {'seed':>5} {'dets/fr':>9} "
          f"{'recall':>8} {'d recall':>9} "
          f"{'of the ' + str(int(ms.sum())) + ' misses, recovered':>34}",
          flush=True)
    print(f"    {'BASE':<6} {'--':>5} {NMS_UM:>5.1f} {'--':>5} "
          f"{'(gtpeak)':>9} {mt.mean():>8.4f} {0.0:>+9.4f}", flush=True)
    for anm, athr, anms, asd in SPLIT_ARMS:
        hit = np.asarray(ARM_HIT.get(anm, []), bool)
        if len(hit) != len(R):
            warn_once("arm_len", f"{anm}: {len(hit)} flags for {len(R)} rows")
            continue
        rec = float(hit.mean())
        got = hit & ms
        by = []
        for nmb, mb in (("cold", cold), ("no-peak", warm),
                        ("nms-killed", killed), ("1:1", survived)):
            by.append(f"{nmb} {int((hit & mb).sum())}/{int(mb.sum())}")
        print(f"    {anm:<6} {athr:>5.2f} "
              f"{('--' if anms is None else f'{anms:.1f}'):>5} {asd:>5} "
              f"{ARM_ND.get(anm, 0) / max(1, ARM_NF[0]):>9.1f} {rec:>8.4f} "
              f"{rec - mt.mean():>+9.4f}   {int(got.sum()):>4} "
              f"({100.0 * got.sum() / max(1, ms.sum()):>4.1f}%)  "
              + "  ".join(by), flush=True)
    print(f"    (dets/fr is the COST. BASE emits {'137.9' if domain == '6bba' else '310.1'}"
          f"/frame at this gate, detfuse3 v4. An arm that recovers the "
          f"no-peak column while roughly holding that number is the only kind "
          f"that could ever be affordable; one that doubles it has answered "
          f"'how many' and nothing else.)", flush=True)

    # ---- Q3: localisation ------------------------------------------------''')

# ---- the misses' buckets must exist before the arm report reads them ------
rep("""    ms = ~mt
    n = int(ms.sum())""",
    """    ms = ~mt
    n = int(ms.sum())
    no_raw = ms & ~(d_raw <= MATCH_R)
    cold = no_raw & ~(hw >= 0.05)
    warm = no_raw & (hw >= 0.05)
    had_raw = ms & (d_raw <= MATCH_R)
    killed = had_raw & ~(d_nms <= MATCH_R)
    survived = had_raw & (d_nms <= MATCH_R)""")
rep("""        no_raw = ms & ~(d_raw <= MATCH_R)
        cold = no_raw & ~(hw >= 0.05)
        warm = no_raw & (hw >= 0.05)
        had_raw = ms & (d_raw <= MATCH_R)
        killed = had_raw & ~(d_nms <= MATCH_R)
        survived = had_raw & (d_nms <= MATCH_R)
        rows = [""",
    """        rows = [""")

rep('''print("  (2) The four miss buckets are the answer to questions 1 and 2''',
    '''print("  (2) THE NUMBER THE RUN WAS FOR: the 'no-peak' column of the split table. "
      "That is how many of gtpeak's shoulder cells a full shape split actually "
      "recovers -- the ceiling, with no classifier and no selectivity. Read it "
      "against dets/fr in the same row, which is the whole cost.", flush=True)
print("  (2b) s15 against s15n isolates NMS: two cells sharing a blob are usually "
      "under 4 um apart, so if s15n loses most of what s15 recovered then the "
      "pipeline's own NMS re-merges the split and no splitter can ship without "
      "changing NMS_UM as well.", flush=True)
print("  (2c) The four miss buckets are the answer to questions 1 and 2''')

rep('GTPEAK DONE in', 'GTSPLIT DONE in')
rep('print("GTPEAK OK", flush=True)', 'print("GTSPLIT OK", flush=True)')
rep("gtpeak_{domain}.csv", "gtsplit_{domain}.csv", n=2)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gtsplit.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
