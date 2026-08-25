"""Mass conservation as a division signature -- measured on real masks.

SCRIPT_VERSION 2026-08-25a-div-mass

THE IDEA, which is the user's. A mother that divides does not vanish and
get replaced by two unrelated cells: her material is CONSERVED. So

    V(daughter_a) + V(daughter_b)  ~=  V(mother)

while a spurious fork -- the mother's real successor plus some unrelated
neighbour -- sums to about TWO mothers. Per daughter that is V/2, i.e.
(1/2)^(1/3) = 0.794 of the mother's diameter.

WHY THIS IS NOT ALREADY IN THE MODEL. The orphan filter stack checks
sister distance, mutual-nearest, both-children-continue and divergence.
Not one of them touches SIZE. Neither do any of the division model's 49
columns, nor any of the edge ranker's 18. Every feature this project owns
is a coordinate or a detection score. Size is a new channel and this is
the first time we can read it: `biohub-zh-masks` put watershed labels for
all 41 Zebrahub crops on disk, on an ISOTROPIC 1.625 um grid, so a voxel
count is a real volume with no resampling.

WHY ZEBRAHUB AND NOT THE COMPETITION DATA. The competition's GT is
sparse, so within a few um of a labelled cell the only other labelled
cell is nearly always its own successor -- the negative class is filtered
out by construction, which is what produced `morph_probe` step 2b's THREE
negatives and every HOCT gate's "109 of 109". Zebrahub's GT is dense, so
"mother plus an unrelated real neighbour" is constructible. Also: the
user's standing objection, that measuring on 6bba risks reading a
holdout the public weights may have memorised, does not apply here.

READING IT. Step 1 is a GATE and step 2 is SUPPRESSED unless it passes,
because measuring a NEW quantity a NEW way while asking a NEW question
means a negative result cannot be attributed to anything. Two anchors:

  * physical -- `equivalent_diameter_area` on the competition's masks was
    median 7.08 um [5.41, 8.95] over all detections and 8.83 um [7.08,
    10.22] over GT-matched ones (biohub-hoct-score / biohub-morph-probe).
    Zebrahub is a different embryo, so these are order-of-magnitude
    checks, not equalities. A median at 1 um or 40 um means the pooled
    indexing is wrong and every number below is void.
  * plumbing -- the division mining below is div_train.py's block copied
    line for line, INCLUDING the 2026-08-24 fix. Its known broken
    signature is unmistakable: 26 of 41 crops reporting no division and
    68 positives for all of Zebrahub. A raw event count in the thousands
    means the mining is right; anything near 68 means this run is void
    for the same reason the earlier two were. div-train v3 and
    div-kinematics trained on 2019 and 2001 detection-matched positives
    from these same crops, so the mask-matched count belongs in that
    neighbourhood -- not equal to it, because matching into a watershed
    label is a different criterion from matching to a detection.

THE FAILURE MODE TO WATCH. Two daughters immediately after a division sit
against each other, and a watershed seeded on a detection heatmap may
give them ONE label. Then both map to the same region, each "daughter"
gets the mother's volume, the ratio reads 1.0, and mass conservation
looks disproven when really it was never measured. Every such event is
DROPPED and COUNTED separately -- if that count dominates, the answer is
"the masks cannot resolve a fresh division", which is a fact about the
segmentation and not about the biology.
"""

import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-08-25a-div-mass"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

POOL = 4                        # zh_masks pooled Y/X by this
POOLED = np.array([1.625, 1.625, 1.625])       # um, isotropic by design
VOXEL_UM3 = float(np.prod(POOLED))             # 4.2915 um^3
SCALE = np.array([1.625, 0.40625, 0.40625])    # FULL-res voxel -> um
GATE_UM = 10.0                  # the pipeline's linking gate
DIAM_LO, DIAM_HI = 3.0, 20.0    # gate 1: plausible cell diameter, um
HIT_FLOOR = 0.25                # gate 1: fraction of GT nodes inside a label
BROKEN_MINING = 200             # gate 2: raw events at or below this is the
                                # 68-positive signature -> void


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def describe(name, a, unit=""):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        print(f"  {name:<28} n=0  -- NOTHING TO REPORT", flush=True)
        return
    print(f"  {name:<28} n={len(a):>7}  median {np.median(a):7.3f}{unit}"
          f"  [p10 {pct(a, 10):6.3f}, p90 {pct(a, 90):6.3f}]", flush=True)


def auc(pos, neg):
    """Mann-Whitney AUC, no sklearn dependency."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv), float)
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


# ---- load ---------------------------------------------------------------
MASKS = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/zhmask_*.npz"):
    MASKS[f.stem[len("zhmask_"):]] = f
print(f"\n  zhmask crops found: {len(MASKS)}", flush=True)
assert MASKS, ("biohub-zh-masks is not attached. Without it there is "
               "nothing to measure -- this kernel reads masks and the GT "
               "they carry, and needs no other source.")
if len(MASKS) < 41:
    print(f"  WARN only {len(MASKS)} of 41 crops -- every count below is "
          f"over this subset, so it is NOT comparable to the 2019/2001 "
          f"anchor without scaling", flush=True)

t_start = time.time()

# =========================================================================
# STEP 1 -- GATE: does a voxel count on these masks read as a real cell?
# =========================================================================
print("\n=== STEP 1 (GATE): geometry and GT->mask hit rate ===", flush=True)

g_hit, g_miss = 0, 0
g_diam_px, g_vol = [], []
per_crop_hit = {}

CACHE = {}          # name -> (labels, counts, nodes, tid, pid, T)
GT_CACHE = {}       # name -> (nodes, tid, pid, T); small, keep all of them


def load_gt(name):
    """GT arrays only. `np.load` on an npz is lazy per member, so this
    never decompresses `labels` -- which matters because the mining runs
    over all 41 crops and needs none of it."""
    if name not in GT_CACHE:
        z = np.load(MASKS[name])
        GT_CACHE[name] = (np.asarray(z["nodes"], float),
                          np.asarray(z["track_id"], np.int64),
                          np.asarray(z["parent_track_id"], np.int64),
                          int(z["labels"].shape[0]))
    return GT_CACHE[name]


def load(name):
    """Labels too, and the per-frame voxel counts. One crop resident at a
    time: an uncompressed (T, Z, Y/4, X/4) int32 label stack is hundreds
    of MB."""
    if name in CACHE:
        return CACHE[name]
    z = np.load(MASKS[name])
    lab = z["labels"]
    T = lab.shape[0]
    cnt = [np.bincount(lab[t].ravel()) for t in range(T)]
    nodes, tid, pid, _ = load_gt(name)
    CACHE.clear()
    CACHE[name] = (lab, cnt, nodes, tid, pid, T)
    return CACHE[name]


def label_at(lab, cnt, t, zyx):
    """Watershed label containing a GT node, and its voxel count.
    `zyx` is FULL-resolution voxel index, as `nodes` stores it and as the
    detections do; Y/X divide by POOL to reach the pooled grid."""
    L = lab[t]
    z0 = int(round(zyx[0]))
    y0 = int(round(zyx[1] / POOL))
    x0 = int(round(zyx[2] / POOL))
    if not (0 <= z0 < L.shape[0] and 0 <= y0 < L.shape[1]
            and 0 <= x0 < L.shape[2]):
        return 0, 0
    v = int(L[z0, y0, x0])
    return v, (int(cnt[t][v]) if v and v < len(cnt[t]) else 0)


for name in sorted(MASKS):
    lab, cnt, nodes, tid, pid, T = load(name)
    h = m = 0
    for r in range(len(nodes)):
        t = int(nodes[r, 0])
        if not (0 <= t < T):
            continue
        v, nvox = label_at(lab, cnt, t, nodes[r, 1:])
        if v and nvox:
            h += 1
            g_vol.append(nvox * VOXEL_UM3)
            g_diam_px.append((6.0 * nvox / np.pi) ** (1.0 / 3.0))
        else:
            m += 1
    g_hit += h
    g_miss += m
    per_crop_hit[name] = (h, h + m)

rate = g_hit / max(1, g_hit + g_miss)
diam_px = np.asarray(g_diam_px, float)
diam_um = diam_px * POOLED[0]

print(f"  GT nodes: {g_hit + g_miss}, inside a watershed label {g_hit} "
      f"({100 * rate:.1f}%)", flush=True)
describe("equiv diameter (voxels)", diam_px)
describe("equiv diameter (um)", diam_um, " um")
describe("volume (um^3)", g_vol)
empty = [n for n, (h, tot) in per_crop_hit.items() if h == 0]
if empty:
    print(f"  WARN {len(empty)} crops had NO GT node inside any label: "
          f"{empty[:5]}", flush=True)

med_um = float(np.median(diam_um)) if len(diam_um) else float("nan")
gate_ok = (rate >= HIT_FLOOR and DIAM_LO <= med_um <= DIAM_HI)
print(f"\n  anchor: competition masks read 7.08 um over all detections, "
      f"8.83 um over GT-matched ones.", flush=True)
print(f"  here: {med_um:.2f} um median over GT-matched, hit rate "
      f"{100 * rate:.1f}%", flush=True)
print(f"  GATE {'PASSED' if gate_ok else 'FAILED'}", flush=True)
if not gate_ok:
    print("\n  STEP 2 SUPPRESSED. Either the GT does not land in the "
          "masks (pooled indexing wrong, or the detector missed almost "
          "everything) or a voxel count does not read as a cell. Any "
          "mass-conservation number computed on top of this would be "
          "unattributable, which is the whole reason this gate exists.",
          flush=True)
    print("\nDIV MASS DONE (gate failed)", flush=True)
    sys.exit(0)

# =========================================================================
# STEP 2 -- mine divisions with div_train.py's block, verbatim
# =========================================================================
print("\n=== STEP 2: division mining (div_train.py's block, verbatim) ===",
      flush=True)

raw_events = 0
per_crop_raw = {}
DIVS = defaultdict(list)        # name -> [(t_mother, r_p, r_a, r_b)]
CONT = defaultdict(list)        # name -> [(t, r_now, r_next)]
DIVIDING = defaultdict(set)     # name -> {(t_mother, track_id)} to exclude

pair_hits = 0                   # before the dedupe, for the anchor
for name in sorted(MASKS):
    nodes, tid, pid, T = load_gt(name)
    first, byk = {}, defaultdict(dict)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        byk[int(tid[r])][tt] = r
        kk = int(tid[r])
        # BUG FIXED 2026-08-24 in div_train.py: this compared tt (a TIME)
        # against first[kk] (a ROW INDEX). Carried across verbatim.
        if kk not in first or tt < int(nodes[first[kk], 0]):
            first[kk] = r
    # parent_track_id < 0 is "no parent"; ZSNS001_tail stores NaN, which
    # the int64 cast turns into garbage, so > 0 is the safe test.
    par = {int(tid[r]): int(pid[r]) for r in range(len(nodes)) if pid[r] > 0}
    n_here = 0
    for kk, r in first.items():
        p = par.get(kk)
        if p is None:
            continue
        t1 = int(nodes[r, 0])
        if t1 - 1 < 0 or t1 >= T:
            continue
        rp = byk.get(p, {}).get(t1 - 1)
        sib = [q for q, rr in first.items()
               if par.get(q) == p and q != kk
               and int(nodes[rr, 0]) == t1]
        if rp is None or not sib:
            continue
        rs = first[sib[0]]
        pair_hits += 1
        if kk > int(tid[rs]):
            # DEDUPE. div_train.py does not do this: it keys a set on
            # (t, ip, j1, j2), and swapping the two daughters gives a
            # DIFFERENT tuple, so every division enters its training set
            # twice. That is why the count below should land near HALF
            # the 2019/2001 anchor, not on it.
            continue
        n_here += 1
        DIVS[name].append((t1 - 1, rp, r, rs))
        DIVIDING[name].add((t1 - 1, p))
    per_crop_raw[name] = n_here
    raw_events += n_here

    # continuations: the same track present at t and t+1, not dividing
    for k_, tmap in byk.items():
        ts = sorted(tmap)
        for a, b in zip(ts, ts[1:]):
            if b != a + 1 or (a, k_) in DIVIDING[name]:
                continue
            CONT[name].append((a, tmap[a], tmap[b]))

n_empty = sum(1 for v in per_crop_raw.values() if v == 0)
print(f"  raw GT division events: {raw_events} over {len(MASKS)} crops "
      f"({n_empty} crops with none)", flush=True)
print(f"  same events before the daughter-order dedupe: {pair_hits} "
      f"-- this is the number div_train.py's set actually holds",
      flush=True)
print(f"  tripwire: the known-broken mining gave 68 positives for all of "
      f"Zebrahub with 26 of 41 crops empty.", flush=True)
if raw_events <= BROKEN_MINING or n_empty > len(MASKS) // 2:
    print("  MINING LOOKS BROKEN -- this is the recorded failure "
          "signature, not a measurement. STOPPING.", flush=True)
    print("\nDIV MASS DONE (mining tripwire)", flush=True)
    sys.exit(0)
print(f"  anchor: div-train v3 / div-kinematics trained on 2019 / 2001 "
      f"DETECTION-matched positives from these crops.", flush=True)
print(f"  continuation pairs available: "
      f"{sum(len(v) for v in CONT.values())}", flush=True)

# =========================================================================
# STEP 3 -- the measurement
# =========================================================================
print("\n=== STEP 3: does mass halve at a division? ===", flush=True)

MAX_CONT = 200_000              # continuations outnumber divisions ~1000:1
rng = np.random.default_rng(0)

d_sum, d_each, d_diam, d_asym = [], [], [], []
c_ratio, c_diam = [], []
f_sum = []
n_merged = n_undet = n_used = 0
n_fake = 0

for name in sorted(MASKS):
    lab, cnt, nodes, tid, pid, T = load(name)

    # --- true divisions -------------------------------------------------
    for (t, rp, ra, rb) in DIVS[name]:
        lp, vp = label_at(lab, cnt, t, nodes[rp, 1:])
        la, va = label_at(lab, cnt, t + 1, nodes[ra, 1:])
        lb, vb = label_at(lab, cnt, t + 1, nodes[rb, 1:])
        if not (lp and la and lb):
            n_undet += 1
            continue
        if la == lb:
            # ONE label for both daughters: the watershed did not resolve
            # the split, so this event carries no information about mass.
            n_merged += 1
            continue
        n_used += 1
        d_sum.append((va + vb) / vp)
        d_each.append(va / vp)
        d_each.append(vb / vp)
        d_diam.append(((va / vp) ** (1.0 / 3.0)))
        d_diam.append(((vb / vp) ** (1.0 / 3.0)))
        d_asym.append(min(va, vb) / max(va, vb))

    # --- continuations (the control) ------------------------------------
    cs = CONT[name]
    if len(cs) > MAX_CONT // max(1, len(MASKS)):
        idx = rng.choice(len(cs), MAX_CONT // max(1, len(MASKS)),
                         replace=False)
        cs = [cs[i] for i in idx]
    for (t, r0, r1) in cs:
        l0, v0 = label_at(lab, cnt, t, nodes[r0, 1:])
        l1, v1 = label_at(lab, cnt, t + 1, nodes[r1, 1:])
        if not (l0 and l1):
            continue
        c_ratio.append(v1 / v0)
        c_diam.append((v1 / v0) ** (1.0 / 3.0))

    # --- FAKE forks: mother + her real successor + an unrelated neighbour
    # This is the test the competition data cannot support. A spurious
    # fork should sum to about TWO mothers, not one.
    bt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < T:
            bt[tt].append(r)
    for (t, r0, r1) in CONT[name][:2000]:
        l0, v0 = label_at(lab, cnt, t, nodes[r0, 1:])
        l1, v1 = label_at(lab, cnt, t + 1, nodes[r1, 1:])
        if not (l0 and l1):
            continue
        here = nodes[r0, 1:] * SCALE
        cands = []
        for rq in bt.get(t + 1, []):
            if rq == r1:
                continue
            d = float(np.linalg.norm(nodes[rq, 1:] * SCALE - here))
            if d <= GATE_UM:
                cands.append(rq)
        if not cands:
            continue
        rq = cands[int(rng.integers(len(cands)))]
        lq, vq = label_at(lab, cnt, t + 1, nodes[rq, 1:])
        if not lq or lq == l1:
            continue
        f_sum.append((v1 + vq) / v0)
        n_fake += 1

print(f"  divisions usable {n_used}; DROPPED: {n_merged} because both "
      f"daughters shared one watershed label, {n_undet} because a member "
      f"was not detected", flush=True)
if n_used == 0:
    print("  NOTHING USABLE. If n_merged dominates, the answer is that "
          "this segmentation cannot resolve a fresh division -- a fact "
          "about the masks, not about mass conservation.", flush=True)
    print("\nDIV MASS DONE (no usable divisions)", flush=True)
    sys.exit(0)
frac_merged = n_merged / max(1, n_merged + n_undet + n_used)
print(f"  merged-label fraction {100 * frac_merged:.1f}% -- this is the "
      f"ceiling on how many divisions ANY size feature could see",
      flush=True)

print("\n  --- prediction vs measurement ---", flush=True)
print("  mass conserved => sum ratio 1.0, each daughter 0.5, "
      "diameter 0.794", flush=True)
describe("DIV (Va+Vb)/Vp", d_sum)
describe("DIV Vchild/Vmother", d_each)
describe("DIV diam ratio", d_diam)
describe("DIV asymmetry min/max", d_asym)
describe("CONT Vnext/Vnow", c_ratio)
describe("CONT diam ratio", c_diam)
describe("FAKE FORK (V1+Vq)/V0", f_sum)

print("\n  --- can it separate? ---", flush=True)
print(f"  per-EDGE, volume ratio, division vs continuation: "
      f"AUC {auc(c_ratio, d_each):.4f}  "
      f"(1.0 = daughters always smaller)", flush=True)
print(f"  per-FORK, sum ratio, true division vs fake fork: "
      f"AUC {auc(f_sum, d_sum):.4f}  "
      f"(1.0 = fakes always sum higher)", flush=True)

sep = (np.median(c_ratio) / np.median(d_each)) if len(c_ratio) and \
    len(d_each) else float("nan")
print(f"\n  continuation/daughter median ratio {sep:.2f} "
      f"(mass conservation predicts 2.0)", flush=True)

print("\n  HOW TO READ THIS. The per-FORK AUC is the one that matters. "
      "The orphan patch's problem is that it cannot tell a real division "
      "from a mother plus a passing neighbour, and NOTHING it currently "
      "checks looks at size. A per-fork AUC near 0.5 means mass "
      "conservation is invisible under this segmentation and the line "
      "closes here. Anything well above it is a division test the "
      "pipeline does not have.", flush=True)
print("  The per-EDGE number is the ranker-facing one, and it is the "
      "orphan-starvation risk in reverse: if daughters really are half "
      "size, a size-aware ranker will push mother->daughter edges DOWN "
      "unless it is given the SIGNED ratio and can learn 'near 0.5 with "
      "a comparable sibling = division'.", flush=True)

print(f"\n  {time.time() - t_start:.0f}s total", flush=True)
print("\nDIV MASS DONE", flush=True)
