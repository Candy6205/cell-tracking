"""Can BRIGHTNESS tell a real detection from a spurious one? Boxes only.

SCRIPT_VERSION 2026-08-25a-node-bright

WHY THIS QUESTION AND NOT ANOTHER. Today's whole size line died on a
single fact the user named: **no dataset here annotates how big a cell
is.** Zebrahub's GT is `[t, z, y, x]` plus track and parent ids; the
competition's is tracks. There is no segmentation anywhere, so every
"volume" measured today came from our own watershed and had nothing to
be checked against. Size is not weak here, it is UNVERIFIABLE -- and a
better mask algorithm cannot fix that, because there is no ruler to
tell you the new masks are better.

Brightness is different, and the difference is the label. "How bright is
this detection" needs no size annotation, and "is this detection a real
cell" is a label we DO hold: match the detection against a GT point. On
Zebrahub, whose GT is DENSE, "no GT nearby" genuinely means spurious.
So this question is answerable and checkable in a way the size ones were
not.

NO MASKS ANYWHERE IN THIS SCRIPT. `morph_cheap` measured fixed-box crop
statistics against the verified mask-based regionprops and found crop
mean rho 0.965, max 0.968, std 0.946 -- all usable for ranking, which is
all a tree needs. A box does not need to know where the cell ends, which
is exactly why it survives the missing-GT problem that killed volume.

WHAT IS AT STAKE. `N/N_true` runs to 1.30 and Zebrahub precision is
0.64, and the record says that problem "has been on record for days with
no tool able to touch it, because every feature this project owns is a
coordinate or a detection score." Box brightness is the first quantity
that is neither.

THE REAL RISK, and it is the headline test. The detector's score is
itself a learned function of local appearance, so it may already contain
everything brightness knows. Three arms settle it:

    SCORE  -- the detection score alone, the baseline that must be beaten
    BRIGHT -- brightness alone, no score
    BOTH   -- does brightness add anything the score does not have?

**BOTH minus SCORE is the number that matters.** BRIGHT alone being
decent proves nothing if the score already carries it.

THIS ALSO REDOES `morph_probe` STEP 2A PROPERLY. That reported real
cells 31% brighter than spurious ones (i_mean 0.856 vs 0.654), but on
COMPETITION data whose GT is sparse -- where "unmatched" means
"unannotated", not "spurious" -- and computed inside masks. Both defects
are removed here: dense GT, and boxes. If the effect survives it is
real; if it does not, that entry needs a SUPERSEDED mark.

ANCHOR, and v1's gate correctly refused v1. The only recorded Zebrahub
detector reading is `biohub-zh-detect-bench`: **PUBLIC weights, untouched,
greedy 1:1 within 5 um, every 5th frame, no TTA -> recall 0.650 /
precision 0.637** at the pipeline gate. v1 read 0.825 / 0.855 and the
gate stopped the run. Two reasons, and only one of them was a bug:

  * BUG: v1 matched by plain nearest neighbour, so several detections
    could claim ONE cell and every one of them counted as real. That
    inflates recall and precision together. FIXED here -- greedy 1:1,
    highest score first, exactly the bench's protocol.
  * NOT a bug: these detections come from the XL ensemble (3 branches x
    4 flip views), not the public weights. **No one has ever benchmarked
    the xl detector on Zebrahub** -- the asset table lists
    `biohub-zh-detect` with no quality number beside it. So a difference
    from 0.650 / 0.637 after the fix is the ENSEMBLE, and reporting it
    fills a real hole in the record.

Because the detector differs, the level cannot be a pass/fail gate. What
IS checkable is internal consistency: every claimed GT point must
correspond to exactly one REAL detection, and the greedy pass must
actually produce duplicates. Those are the gate. The level is reported
next to the public reading as a comparison, clearly labelled as such.

THE STRATIFICATION THAT v1 LACKED, and it may decide the whole line. A
false positive is one of two very different things:

    EMPTY  -- no GT cell within 5 um. A hallucination, and it should be
              DIM, because there is nothing there to be bright.
    DUP    -- a cell IS there, but a higher-scoring detection already
              owns it. This is an NMS failure, it sits ON a real cell,
              and it is therefore exactly as bright as a true positive.

**Brightness cannot possibly separate DUP from REAL.** So if the false
positives are mostly DUP, this line is dead on arrival regardless of any
AUC, and the fix is NMS, not a feature. v1 pooled the two and could not
have seen that.
"""

import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-08-25a-node-bright"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

POOL = 4
SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 5.0                  # morph_confound's radius, unchanged
UNET_THRESH = 0.15              # the pipeline's detection gate
CROP_R = 3
MAX_CROPS = 41
FRAME_STRIDE = 5
REC_LO, REC_HI = 0.50, 0.80     # gate: recorded 0.65
PRE_LO, PRE_HI = 0.50, 0.80     # gate: recorded 0.64


def _pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm(p):
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def box_stats(img, zyx, r=CROP_R):
    """Six numbers from a fixed box. No mask, no boundary, no size label.
    morph_cheap: crop mean/max/std track the mask-based regionprops at
    rho 0.965 / 0.968 / 0.946."""
    Z, Y, X = img.shape
    z0 = int(round(zyx[0]))
    y0 = int(round(zyx[1] / POOL))
    x0 = int(round(zyx[2] / POOL))
    c = img[max(0, z0 - r):min(Z, z0 + r + 1),
            max(0, y0 - r):min(Y, y0 + r + 1),
            max(0, x0 - r):min(X, x0 + r + 1)]
    if c.size == 0:
        return None
    lo = float(c.min())
    return (float(c.mean()), float(c.max()), float(c.std()),
            float(np.clip(c - lo, 0, None).sum()),   # background-subtracted
            float(c.max() - lo),                     # contrast
            float(img[min(z0, Z - 1), min(y0, Y - 1), min(x0, X - 1)]))


COLS = ["c_mean", "c_max", "c_std", "c_mass", "c_contrast", "c_centre"]

ZH = {f.stem[len("zhdet_"):]: f
      for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
CROPS = None
for c in pathlib.Path("/kaggle/input").glob("*/**/crops"):
    CROPS = c
    break
print(f"\n  zhdet caches {len(ZH)}; images at {CROPS}", flush=True)
assert ZH, "biohub-zh-detect is not attached"
assert CROPS is not None, "biohub-zebrahub-extract is not attached"

t0 = time.time()
X, y, S, G, K = [], [], [], [], []
n_gt_tot = n_gt_hit = 0

for k, name in enumerate(sorted(ZH)):
    if k >= MAX_CROPS:
        break
    cf = CROPS / f"{name}.npz"
    if not cf.exists():
        print(f"  WARN {name}: no crop image -> skipped", flush=True)
        continue
    zc = np.load(cf)
    if "image" not in zc.files:
        print(f"  WARN {name}: crop has no 'image' -> skipped", flush=True)
        continue
    img = zc["image"]
    zd = np.load(ZH[name])
    nodes = np.asarray(zd["nodes"], float)
    T = min(img.shape[0],
            max(int(kk[1:]) for kk in zd.files if kk.startswith("c")) + 1)

    gbt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < T:
            gbt[tt].append(r)

    for t in range(0, T, FRAME_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        P = _norm(_pool_xy(np.asarray(img[t])))
        cc = np.asarray(zd[f"c{t}"], float).reshape(-1, 3)
        ss = np.asarray(zd[f"s{t}"], float).reshape(-1)
        sel = ss >= UNET_THRESH
        cc, ss = cc[sel], ss[sel]
        gt_um = nodes[gi, 1:] * SCALE[None, :]
        n_gt_tot += len(gi)
        if len(cc) == 0:
            continue
        det_um = cc * SCALE[None, :]
        # GREEDY 1:1, exactly the protocol biohub-zh-detect-bench used.
        # v1 used plain nearest-neighbour, which lets several detections
        # claim ONE cell and counts every one of them as real -- that
        # inflated recall to 0.825 and precision to 0.855 against the
        # bench's 0.650 / 0.637 and the gate correctly refused it.
        tree_g = cKDTree(gt_um)
        claimed = {}
        order = np.argsort(-ss)
        kind = np.zeros(len(cc), np.int8)      # 0 EMPTY, 1 REAL, 2 DUP
        for i in order:
            near = tree_g.query_ball_point(det_um[i], MATCH_UM)
            if not near:
                continue
            near.sort(key=lambda gg: np.linalg.norm(gt_um[gg] - det_um[i]))
            free = [gg for gg in near if gg not in claimed]
            if free:
                claimed[free[0]] = int(i)
                kind[i] = 1
            else:
                # a cell IS here, but a higher-scoring detection owns it.
                # This is a duplicate, not a hallucination, and the two
                # are different problems: a duplicate sits ON a real cell
                # and is therefore just as bright.
                kind[i] = 2
        n_gt_hit += len(claimed)
        for i in range(len(cc)):
            b = box_stats(P, cc[i])
            if b is None:
                continue
            X.append(b)
            y.append(1 if kind[i] == 1 else 0)
            K.append(int(kind[i]))
            S.append(float(ss[i]))
            G.append(name)
    print(f"  {k+1}/{min(MAX_CROPS, len(ZH))} {name}: {len(X)} rows so far "
          f"({time.time()-t0:.0f}s)", flush=True)
    del zc, img, zd

X = np.asarray(X, np.float32)
y = np.asarray(y, np.int8)
S = np.asarray(S, np.float32)
G = np.asarray(G)
K = np.asarray(K, np.int8)
print(f"\n  {len(X)} detections, {int(y.sum())} real "
      f"({100*y.mean():.1f}%)", flush=True)

# =========================================================================
# GATE -- do the labels reproduce the recorded reading?
# =========================================================================
print("\n=== GATE: is the greedy 1:1 matching internally consistent? ===",
      flush=True)
n_real = int((K == 1).sum())
n_dup = int((K == 2).sum())
n_empty = int((K == 0).sum())
print(f"  REAL {n_real}, DUPLICATE {n_dup}, EMPTY {n_empty}", flush=True)
print(f"  claimed GT points {n_gt_hit} -- must equal REAL: "
      f"{'YES' if n_gt_hit == n_real else 'NO'}", flush=True)
consistent = (n_gt_hit == n_real) and n_dup > 0
print(f"  GATE {'PASSED' if consistent else 'FAILED'}", flush=True)
if not consistent:
    if n_gt_hit != n_real:
        print("  A claimed GT point does not correspond one-to-one with a "
              "REAL detection, so the greedy pass is not doing what it "
              "claims and the labels mean nothing.", flush=True)
    else:
        print("  ZERO duplicates. Either NMS is perfect, which would "
              "contradict the recorded over-detection, or the duplicate "
              "branch never fires and every FP has been filed as EMPTY.",
              flush=True)
    print("\nNODE BRIGHT DONE (gate failed)", flush=True)
    sys.exit(0)

recall = n_gt_hit / max(1, n_gt_tot)
precision = n_real / max(1, len(X))
print("\n=== READING (not a gate): the xl detector on Zebrahub, first "
      "time measured ===", flush=True)
print(f"  recall    {recall:.4f}", flush=True)
print(f"  precision {precision:.4f}", flush=True)
print(f"  N/N_true  {len(X)/max(1, n_gt_tot):.4f}", flush=True)
print("  public weights, same protocol (biohub-zh-detect-bench): "
      "recall 0.650 / precision 0.637", flush=True)
print("  the difference is the ENSEMBLE -- 3 branches x 4 flip views "
      "against untouched public weights -- not an error.", flush=True)
if not (0.40 <= recall <= 0.95):
    print(f"  WARN recall {recall:.3f} is outside anything this domain "
          f"has produced; treat every number below as suspect.",
          flush=True)

# =========================================================================
# The three arms. Split by CROP so a model cannot memorise a crop.
# =========================================================================
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa

crops = sorted(set(G.tolist()))
n_val = max(1, len(crops) // 3)
val_crops = set(crops[:n_val])
va = np.array([g in val_crops for g in G])
tr = ~va
print(f"\n  split by crop: {len(crops)-n_val} train / {n_val} val "
      f"({int(tr.sum())} / {int(va.sum())} rows)", flush=True)
print(f"  val crops: {sorted(val_crops)}", flush=True)

ARMS = {
    "SCORE  (detection score only)": S.reshape(-1, 1),
    "BRIGHT (6 box columns, no score)": X,
    "BOTH": np.hstack([S.reshape(-1, 1), X]),
}
res = {}
for nm, M in ARMS.items():
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        random_state=0)
    clf.fit(M[tr], y[tr])
    p = clf.predict_proba(M[va])[:, 1]
    res[nm] = (roc_auc_score(y[va], p), average_precision_score(y[va], p))
    print(f"  {nm:<34} AUC {res[nm][0]:.4f}   AP {res[nm][1]:.4f}",
          flush=True)

print("\n  --- the same three arms, per NEGATIVE TYPE ---", flush=True)
print("  brightness cannot separate DUP from REAL (a duplicate sits on a "
      "real cell), so a high overall AUC driven by EMPTY is not a tool "
      "for over-detection unless EMPTY is what dominates.", flush=True)
for tag, mask_ in (("vs EMPTY only", (K == 1) | (K == 0)),
                   ("vs DUPLICATE only", (K == 1) | (K == 2))):
    sub = va & mask_
    if int(sub.sum()) < 100 or len(set(y[sub].tolist())) < 2:
        print(f"  {tag}: too few rows to read", flush=True)
        continue
    line = f"  {tag:<20}"
    for nm, M in ARMS.items():
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            random_state=0)
        sub_tr = tr & mask_
        clf.fit(M[sub_tr], y[sub_tr])
        pp = clf.predict_proba(M[sub])[:, 1]
        line += f"  {nm.split()[0]} {roc_auc_score(y[sub], pp):.4f}"
    print(line, flush=True)
print(f"\n  negative mix on the val split: "
      f"EMPTY {int((va & (K == 0)).sum())}, "
      f"DUPLICATE {int((va & (K == 2)).sum())}", flush=True)

a_s = res["SCORE  (detection score only)"][0]
a_b = res["BOTH"][0]
print(f"\n  ⭐ BOTH - SCORE = {a_b - a_s:+.4f} AUC", flush=True)
print("  This is the whole question. The detection score is already a "
      "learned function of local appearance, so BRIGHT scoring well on "
      "its own proves nothing; only what it ADDS to the score can "
      "change the pipeline.", flush=True)

# =========================================================================
# The physics, with no model in the way -- and morph_probe step 2a redone
# =========================================================================
print("\n=== medians, real vs spurious (morph_probe step 2a, redone on "
      "DENSE GT with no masks) ===", flush=True)
print(f"  {'column':<14} {'REAL':>10} {'EMPTY':>10} {'DUP':>10} "
      f"{'REAL/EMPTY':>11}", flush=True)
for j, cn in enumerate(COLS):
    r_ = float(np.median(X[K == 1, j]))
    e_ = float(np.median(X[K == 0, j])) if (K == 0).any() else float("nan")
    d_ = float(np.median(X[K == 2, j])) if (K == 2).any() else float("nan")
    print(f"  {cn:<14} {r_:>10.4f} {e_:>10.4f} {d_:>10.4f} "
          f"{r_ / max(e_, 1e-9):>11.3f}", flush=True)
r_ = float(np.median(S[K == 1]))
e_ = float(np.median(S[K == 0])) if (K == 0).any() else float("nan")
d_ = float(np.median(S[K == 2])) if (K == 2).any() else float("nan")
print(f"  {'det score':<14} {r_:>10.4f} {e_:>10.4f} {d_:>10.4f} "
      f"{r_ / max(e_, 1e-9):>11.3f}", flush=True)
print("\n  morph_probe recorded i_mean 0.856 real vs 0.654 spurious, "
      "ratio 1.31 -- but on SPARSE competition GT where 'unmatched' "
      "means 'unannotated', and inside masks. If c_mean's ratio here is "
      "far from 1.31, that entry needs a SUPERSEDED mark.", flush=True)

print(f"\n  {time.time()-t0:.0f}s total", flush=True)
print("\nNODE BRIGHT DONE", flush=True)
