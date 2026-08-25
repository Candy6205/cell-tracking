"""Mass conservation, weighed by BRIGHTNESS instead of counted in voxels.

SCRIPT_VERSION 2026-08-25a-div-intmass

THE STORY SO FAR. `biohub-div-mass` measured (Va+Vb)/Vp = 1.772 at a real
division where mass conservation predicts 1.0, and 1.984 at a fake fork
where it predicts 2.0 -- nearly the same number, per-fork AUC 0.605. I
concluded the watershed measures TERRITORY between peaks. The user
countered that the masks are simply inflated by a FRINGE, and
`biohub-mask-anatomy` settled it in 19 seconds, against me:

  * 76.8% of all mask boundary faces touch BACKGROUND, only 23.2% touch
    another mask. Masks stop because the heatmap faded, not because a
    neighbour pushed back.
  * across an 18x density range (45 to 809 cells per frame) median mask
    volume is FLAT: log-log slope -0.028, r -0.127. A tessellation
    cannot do that; a cell-plus-fringe can, because neither a cell nor a
    fringe knows how crowded it is.

So mass conservation was never tested. It was drowned. Fitting
mask = core M + fringe H to the measured 1.772 gives H = 3.39 M: the
cell is ~23% of the mask, a shell about 1.3 voxels thick on a 6.6-voxel
mask, which is enough because volume goes as the cube.

THE FIX, and why it is the right shape. A fringe voxel is fringe because
the detector heatmap fell to 0.05-0.1 there; a core voxel sits near 0.9.
In a nuclear-labelled image the same voxels are dim and bright
respectively. So SUMMING BRIGHTNESS instead of COUNTING VOXELS discounts
the fringe automatically and continuously -- no new threshold to tune,
which matters because at 6.6 voxels across there is no room to erode.
`morph_cheap` already measured that crop intensity tracks the mask-based
regionprops at rho 0.965, so intensity is a quantity we can read.

Biologically this is also the right thing to weigh: DNA doubles before
mitosis and each daughter receives half, so total fluorescence should be
conserved even while the mask geometry is not.

THE GATE, and it is not optional. "Weighting by brightness discounts the
fringe" is a HYPOTHESIS, not a fact. If the fringe is bright too, the
weighting does nothing and every number in step 2 is the old number in a
new costume -- unattributable in exactly the way `morph_probe` was
designed to prevent. So step 1 measures the discount directly:

  * VOLUME fraction in the outer shell (voxels having a background
    neighbour). The fringe reading says this should be large.
  * INTENSITY-MASS fraction in that same shell.

If the intensity fraction is not clearly below the volume fraction, the
weighting failed and STEP 2 IS SUPPRESSED. Predictions written down now:
volume fraction well above half, intensity fraction materially lower.
If they come out equal, brightness is flat across the mask, which would
mean the mask is not a cell plus a halo at all and BOTH readings of
div_mass are wrong.

WHAT THIS STILL CANNOT FIX. 44.7% of divisions had both daughters inside
ONE label -- one detector peak served two cells. No weighting touches
that, and it is a hard ceiling on any mask-based size feature here.
"""

import functools
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-08-25a-div-intmass"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

POOL = 4
POOLED = np.array([1.625, 1.625, 1.625])
VOXEL_UM3 = float(np.prod(POOLED))
SCALE = np.array([1.625, 0.40625, 0.40625])
GATE_UM = 10.0
MAX_CROPS = 12                  # images are the cost; divisions are plentiful
MAX_CONT_PER_CROP = 4000
SHELL_MIN_DISCOUNT = 0.08       # gate: intensity fraction must sit at least
                                # this far below the volume fraction


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def describe(name, a, unit=""):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        print(f"  {name:<30} n=0  -- NOTHING TO REPORT", flush=True)
        return
    print(f"  {name:<30} n={len(a):>7}  median {np.median(a):7.3f}{unit}"
          f"  [p10 {pct(a, 10):6.3f}, p90 {pct(a, 90):6.3f}]", flush=True)


def auc(pos, neg):
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
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def _pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm(p):
    """cache_train.py's normalisation, so intensities are on the same
    scale every other measurement in this project used."""
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


MASKS = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/zhmask_*.npz"):
    MASKS[f.stem[len("zhmask_"):]] = f
CROPS = None
for c in pathlib.Path("/kaggle/input").glob("*/**/crops"):
    CROPS = c
    break
print(f"\n  zhmask crops {len(MASKS)}; images at {CROPS}", flush=True)
assert MASKS, "biohub-zh-masks is not attached"
assert CROPS is not None, (
    "biohub-zebrahub-extract is not attached. This kernel needs the raw "
    "images -- weighing by brightness is the entire point and there is "
    "nothing to weigh without them.")

t0 = time.time()

# ---- pick the crops with the most divisions, so 12 crops go furthest ----
GT = {}
for name in sorted(MASKS):
    z = np.load(MASKS[name])
    GT[name] = (np.asarray(z["nodes"], float),
                np.asarray(z["track_id"], np.int64),
                np.asarray(z["parent_track_id"], np.int64),
                int(z["labels"].shape[0]))


def mine(name):
    """div_train.py's division block, carried across verbatim including
    the 2026-08-24 time-vs-row-index fix, plus the daughter-order dedupe
    div_train does not do."""
    nodes, tid, pid, T = GT[name]
    first, byk = {}, defaultdict(dict)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        byk[int(tid[r])][tt] = r
        kk = int(tid[r])
        if kk not in first or tt < int(nodes[first[kk], 0]):
            first[kk] = r
    par = {int(tid[r]): int(pid[r]) for r in range(len(nodes)) if pid[r] > 0}
    divs, dividing = [], set()
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
        if kk > int(tid[rs]):
            continue
        divs.append((t1 - 1, rp, r, rs))
        dividing.add((t1 - 1, p))
    cont = []
    for k_, tmap in byk.items():
        ts = sorted(tmap)
        for a, b in zip(ts, ts[1:]):
            if b == a + 1 and (a, k_) not in dividing:
                cont.append((a, tmap[a], tmap[b]))
    return divs, cont


def lookup(L, vol, ims, nodes, t, r):
    """Watershed label under a GT node, with its voxel count and its
    summed brightness. `nodes` stores FULL-resolution voxel indices, so
    Y and X divide by POOL to reach the pooled grid the masks live on.
    Takes everything explicitly: the caller frees these arrays between
    crops, and a closure over them would read a deleted name."""
    Lt = L[t]
    z0 = int(round(nodes[r, 1]))
    y0 = int(round(nodes[r, 2] / POOL))
    x0 = int(round(nodes[r, 3] / POOL))
    if not (0 <= z0 < Lt.shape[0] and 0 <= y0 < Lt.shape[1]
            and 0 <= x0 < Lt.shape[2]):
        return 0, 0.0, 0.0
    v = int(Lt[z0, y0, x0])
    if not v or v >= len(vol[t]):
        return 0, 0.0, 0.0
    return v, float(vol[t][v]), float(ims[t][v])


PLAN = []
for name in sorted(MASKS):
    if not (CROPS / f"{name}.npz").exists():
        continue
    d, c = mine(name)
    PLAN.append((len(d), name, d, c))
PLAN.sort(reverse=True, key=lambda r: r[0])
PLAN = PLAN[:MAX_CROPS]
print(f"  using {len(PLAN)} crops carrying "
      f"{sum(r[0] for r in PLAN)} division events", flush=True)
assert PLAN, "no crop has both a mask file and an image file"

# =========================================================================
# The pass. Step 1 accumulates the gate; step 2 accumulates the answer.
# Both are computed here because they read the same volumes, but step 2
# is NOT PRINTED unless the gate passes.
# =========================================================================
sh_vol = sh_int = tot_vol = tot_int = 0.0
d_sum_v, d_sum_i = [], []
c_ratio_v, c_ratio_i = [], []
f_sum_v, f_sum_i = [], []
n_used = n_merged = n_undet = 0
rng = np.random.default_rng(0)

for k, (ndiv, name, divs, cont) in enumerate(PLAN):
    z = np.load(MASKS[name])
    L = z["labels"]
    nodes, tid, pid, T = GT[name]
    zc = np.load(CROPS / f"{name}.npz")
    if "image" not in zc.files:
        print(f"  WARN {name}: crop has no 'image' -> SKIPPED, and its "
              f"{ndiv} divisions leave the sample", flush=True)
        continue
    img = zc["image"]
    Tn = min(T, img.shape[0])

    vol = {}    # (t, label) -> voxel count
    ims = {}    # (t, label) -> summed brightness above the frame floor
    for t in range(Tn):
        Lt = L[t]
        P = _norm(_pool_xy(np.asarray(img[t])))
        if P.shape != Lt.shape:
            zz = min(P.shape[0], Lt.shape[0])
            yy = min(P.shape[1], Lt.shape[1])
            xx = min(P.shape[2], Lt.shape[2])
            P, Lt = P[:zz, :yy, :xx], Lt[:zz, :yy, :xx]
        flat = Lt.ravel()
        n = int(flat.max()) + 1
        w = np.clip(P.ravel(), 0.0, None)        # below-floor is not mass
        cv = np.bincount(flat, minlength=n)
        ci = np.bincount(flat, weights=w, minlength=n)
        vol[t] = cv
        ims[t] = ci

        # --- gate: how much of each mask lives in its outermost shell ---
        shell = np.zeros(Lt.shape, bool)
        for ax in (0, 1, 2):
            a = np.moveaxis(Lt, ax, 0)
            s = np.moveaxis(shell, ax, 0)
            s[:-1] |= (a[:-1] > 0) & (a[1:] == 0)
            s[1:] |= (a[1:] > 0) & (a[:-1] == 0)
        m = Lt > 0
        tot_vol += float(m.sum())
        tot_int += float(w.reshape(Lt.shape)[m].sum())
        sh_vol += float(shell.sum())
        sh_int += float(w.reshape(Lt.shape)[shell].sum())

    # partial, not a closure: the names below are freed at the end of
    # this iteration, and a closure would read them after the del.
    at = functools.partial(lookup, L, vol, ims, nodes)

    for (t, rp, ra, rb) in divs:
        if t + 1 >= Tn:
            continue
        lp, vp, ip_ = at(t, rp)
        la, va, ia = at(t + 1, ra)
        lb, vb, ib = at(t + 1, rb)
        if not (lp and la and lb):
            n_undet += 1
            continue
        if la == lb:
            n_merged += 1
            continue
        if vp <= 0 or ip_ <= 0:
            continue
        n_used += 1
        d_sum_v.append((va + vb) / vp)
        d_sum_i.append((ia + ib) / ip_)

    cs = cont if len(cont) <= MAX_CONT_PER_CROP else \
        [cont[i] for i in rng.choice(len(cont), MAX_CONT_PER_CROP,
                                     replace=False)]
    bt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < Tn:
            bt[tt].append(r)
    for (t, r0, r1) in cs:
        if t + 1 >= Tn:
            continue
        l0, v0, i0 = at(t, r0)
        l1, v1, i1 = at(t + 1, r1)
        if not (l0 and l1) or v0 <= 0 or i0 <= 0:
            continue
        c_ratio_v.append(v1 / v0)
        c_ratio_i.append(i1 / i0)
        here = nodes[r0, 1:] * SCALE
        cand = [rq for rq in bt.get(t + 1, [])
                if rq != r1
                and np.linalg.norm(nodes[rq, 1:] * SCALE - here) <= GATE_UM]
        if not cand:
            continue
        lq, vq, iq = at(t + 1, cand[int(rng.integers(len(cand)))])
        if not lq or lq == l1:
            continue
        f_sum_v.append((v1 + vq) / v0)
        f_sum_i.append((i1 + iq) / i0)

    print(f"  {k+1}/{len(PLAN)} {name}: {ndiv} divisions, "
          f"{time.time()-t0:.0f}s", flush=True)
    del z, L, zc, img, vol, ims

# =========================================================================
# STEP 1 -- the gate
# =========================================================================
print("\n=== STEP 1 (GATE): did brightness actually discount the fringe? "
      "===", flush=True)
fv = sh_vol / max(1e-9, tot_vol)
fi = sh_int / max(1e-9, tot_int)
print(f"  outer shell holds {100*fv:5.1f}% of mask VOLUME", flush=True)
print(f"  outer shell holds {100*fi:5.1f}% of mask BRIGHTNESS", flush=True)
print(f"  discount: {100*(fv-fi):+.1f} points", flush=True)
if fv - fi < SHELL_MIN_DISCOUNT:
    print(f"\n  GATE FAILED. Brightness is nearly flat across the mask, so "
          f"weighting by it does not strip anything and step 2 would be "
          f"div_mass's number in a new costume. It also means the mask is "
          f"NOT a bright core plus a dim halo -- which contradicts the "
          f"fringe reading that mask_anatomy supported, and is itself the "
          f"finding. STEP 2 SUPPRESSED.", flush=True)
    print("\nDIV INTMASS DONE (gate failed)", flush=True)
    sys.exit(0)
print("  GATE PASSED -- the outer shell is a large share of the volume "
      "and a materially smaller share of the light, which is what a "
      "core-plus-halo looks like.", flush=True)

# =========================================================================
# STEP 2 -- the answer
# =========================================================================
print("\n=== STEP 2: mass conservation, weighed two ways ===", flush=True)
print(f"  divisions usable {n_used}; dropped {n_merged} merged-label, "
      f"{n_undet} undetected", flush=True)
print("  conserved => division 1.0, continuation 1.0, fake fork 2.0",
      flush=True)
print("\n  by VOXEL COUNT (div_mass's measure, recomputed here):",
      flush=True)
describe("DIV (Va+Vb)/Vp", d_sum_v)
describe("CONT Vnext/Vnow", c_ratio_v)
describe("FAKE FORK", f_sum_v)
print("\n  by BRIGHTNESS:", flush=True)
describe("DIV (Ia+Ib)/Ip", d_sum_i)
describe("CONT Inext/Inow", c_ratio_i)
describe("FAKE FORK", f_sum_i)

a_v = auc(f_sum_v, d_sum_v)
a_i = auc(f_sum_i, d_sum_i)
print(f"\n  per-FORK AUC, true division vs fake fork:", flush=True)
print(f"    voxel count  {a_v:.4f}   (div_mass recorded 0.6051 over all "
      f"41 crops)", flush=True)
print(f"    brightness   {a_i:.4f}", flush=True)
print(f"    change       {a_i - a_v:+.4f}", flush=True)
print("\n  ANCHOR: the voxel-count row is the control. If it does not land "
      "near 0.605 then this 12-crop subset is not comparable to div_mass "
      "and only the brightness-minus-voxel DIFFERENCE may be read.",
      flush=True)
print("  The number that decides the line is the brightness AUC. Near "
      "0.605 means the fringe was not what was hiding the signal and "
      "there is no signal to hide. Well above it means mass conservation "
      "is real, was drowned, and is now a division test the orphan patch "
      "does not have -- aimed straight at the 26 false divisions per 5 "
      "true ones on the 6bba holdout.", flush=True)
print(f"\n  {time.time()-t0:.0f}s total", flush=True)
print("\nDIV INTMASS DONE", flush=True)
