"""zhcomplete -- is Zebrahub's GT actually complete, or just denser?

SCRIPT_VERSION 2026-08-30a-zhcomplete

THE QUESTION (user's, 2026-08-30): every "honest denominator" claim,
every zh precision number, and the proposed fusion negative-mining all
assume Zebrahub labels EVERY cell. The record never verified that --
the labels are the output of their automated tracking (reference.md:
"machine-generated, not human GT"), the 2026-08-21 spot-check verified
labels sit ON cells (accuracy), never that all cells HAVE labels
(completeness), and ~10% of ZSNS001 label points sit at near-background
intensity, so their pipeline is visibly imperfect even where it did
fire.

TWO INDEPENDENT MEASUREMENTS, neither trusting our own models' scores:

1. THEIR OWN FLICKER (all 41 crops, no models involved). Their labels
   carry track ids. A track present at t-1 and t+1 but absent at t is
   a dropout of THEIR tracker, measured from their own data. Guarded
   against crop-boundary artifacts: a gap only counts when both
   flanking positions sit >= EDGE_UM inside the crop on every axis.
   This is a LOWER bound on their per-frame incompleteness -- it only
   sees dropouts INSIDE a track; a cell their tracker never picked up
   at all is invisible to this test.

2. UNMATCHED-PEAK FORENSICS (the 8 eval crops, where both detector
   caches exist). Take xl detections >= 0.15, greedy 1:1 to labels at
   7 um. For each UNMATCHED peak ask two label-free questions:
     - brightness: raw image max in a +-1.6 um window, compared to the
       matched peaks' distribution and to random in-crop positions;
     - consensus: does the INDEPENDENTLY trained ep05 DeepCenter also
       have a peak within 3 um (its cache, floor 0.05)?
   An unmatched peak that is bright AND seconded by a second detector
   is most plausibly an UNLABELED CELL. The fraction of such peaks
   bounds the completeness gap from a second direction:
       gap_estimate = cell_like_unmatched / (labels + cell_like_unmatched)

CONSEQUENCES, decided before the run:
  - both small (<~3%): zh keeps "honest denominator" status with a
    footnote; fusion negatives = unmatched AND dark AND no-consensus;
  - moderate: zh precision history gets a systematic-understatement
    footnote in reference.md; negatives only from the filtered subset;
  - large (>~10%): the user's suspicion holds, the fusion plan's
    negative mining is unsound as proposed, and zh's denominator status
    is downgraded in the docs.

Attach (kernel outputs): biohub-zebrahub-extract (crops),
biohub-zh-detect (zhdet_*), biohub-dccache3 (dcdet_ep05_*). CPU.
"""
import gc
import os
import pathlib
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-08-30a-zhcomplete"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0
CONSENSUS_UM = 3.0
EDGE_UM = 5.0            # boundary guard for the flicker test
BRIGHT_WIN = (1, 4, 4)   # +-voxels ~ +-1.6 um per axis
FRAME_STRIDE = 10        # forensics samples every 10th frame


def _walk(prefix):
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


CROPS = None
for c in ("/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
          "crops", "/kaggle/input/biohub-zebrahub-extract/crops"):
    if pathlib.Path(c).exists():
        CROPS = pathlib.Path(c)
        break
if CROPS is None:
    for root_, dirs, _f in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        if "crops" in dirs:
            CROPS = pathlib.Path(root_) / "crops"
            break
assert CROPS is not None, "zebrahub crops not mounted"
CROP_FILES = sorted(CROPS.glob("*.npz"))
assert CROP_FILES, f"{CROPS} is empty"

ZHDET = _walk("zhdet_")
DCDET = _walk("dcdet_ep05_")
print(f"crops {len(CROP_FILES)}, zhdet {len(ZHDET)}, "
      f"dcdet_ep05 {len(DCDET)}", flush=True)
assert ZHDET, "attach biohub-zh-detect"
assert DCDET, "attach biohub-dccache3"

# =====================================================================
# 1. their own flicker, from their own track ids
# =====================================================================
print("\n=== 1. THEIR OWN FLICKER (all crops, their track ids, no "
      "models) ===", flush=True)
print(f"  a gap counts only when both flanking positions are "
      f">= {EDGE_UM} um inside the crop on every axis, so leaving the "
      f"crop does not masquerade as a dropout", flush=True)

tot_mid = 0          # track-frames where the track is mid-flight
tot_gap1 = 0         # single-frame holes
tot_gapN = 0         # holes of any length (counted once per hole)
per_movie = defaultdict(lambda: [0, 0, 0])
t0 = time.time()
for f in CROP_FILES:
    z = np.load(f)
    if "nodes" not in z.files or "track_id" not in z.files:
        print(f"  WARN {f.stem}: nodes/track_id missing -> skipped",
              flush=True)
        continue
    nodes = np.asarray(z["nodes"], float)
    tid = np.asarray(z["track_id"], np.int64)
    shape_vox = np.array([64, 256, 256], float)
    lo_um = EDGE_UM
    hi_um = shape_vox * SCALE - EDGE_UM
    by_track = defaultdict(dict)
    for r in range(len(nodes)):
        by_track[int(tid[r])][int(nodes[r, 0])] = nodes[r, 1:] * SCALE
    movie = f.stem.split("_z")[0]
    for k, frames in by_track.items():
        ts = sorted(frames)
        if len(ts) < 2:
            continue
        for a, b in zip(ts[:-1], ts[1:]):
            if b == a + 1:
                continue
            pa, pb = frames[a], frames[b]
            if np.all(pa >= lo_um) and np.all(pa <= hi_um) \
                    and np.all(pb >= lo_um) and np.all(pb <= hi_um):
                tot_gapN += 1
                per_movie[movie][2] += 1
                if b == a + 2:
                    tot_gap1 += 1
                    per_movie[movie][1] += 1
        # mid-flight frames: how many opportunities for a 1-frame hole
        n_mid = max(0, ts[-1] - ts[0] - 1)
        tot_mid += n_mid
        per_movie[movie][0] += n_mid
print(f"  mid-flight track-frames {tot_mid:,}; holes: 1-frame "
      f"{tot_gap1:,}, any-length {tot_gapN:,}", flush=True)
if tot_mid:
    print(f"  THEIR per-frame dropout rate (1-frame holes / mid-flight "
          f"frames): {100.0 * tot_gap1 / tot_mid:.2f}%   any-length "
          f"holes: {100.0 * tot_gapN / tot_mid:.2f}%", flush=True)
print("  per movie (mid-flight / 1-frame / any):", flush=True)
for m in sorted(per_movie):
    v = per_movie[m]
    print(f"    {m:<14} {v[0]:>9,} {v[1]:>7,} {v[2]:>7,}"
          + (f"   ({100.0 * v[1] / v[0]:.2f}%)" if v[0] else ""),
          flush=True)
print("  ⚠ this sees only dropouts INSIDE a track; a cell their "
      "tracker never found at all is invisible here -> LOWER bound.",
      flush=True)

# =====================================================================
# 2. unmatched-peak forensics on the 8 eval crops
# =====================================================================
print(f"\n=== 2. UNMATCHED-PEAK FORENSICS (crops with both caches, "
      f"every {FRAME_STRIDE}th frame) ===", flush=True)
both = sorted(set(ZHDET) & set(DCDET))
print(f"  crops with both caches: {len(both)}", flush=True)

rows = []          # (matched01, bright, consensus01)
rand_bright = []
n_labels = 0
for stem in both:
    f = CROPS / f"{stem}.npz"
    if not f.exists():
        print(f"  WARN {stem}: crop file missing -> skipped", flush=True)
        continue
    z = np.load(f)
    img = z["image"]
    nodes = np.asarray(z["nodes"], float)
    zx = np.load(ZHDET[stem])
    zd = np.load(DCDET[stem])
    T = img.shape[0]
    gbt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < T:
            gbt[tt].append(nodes[r, 1:])
    rng = np.random.default_rng(20260830)
    for t in range(0, T, FRAME_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        n_labels += len(gi)
        vol = np.asarray(img[t])
        gt_um = np.asarray(gi, float) * SCALE[None, :]

        def bright_at(vox, vol=vol):
            zc, yc, xc = (int(round(v)) for v in vox)
            dz, dy, dx = BRIGHT_WIN
            sl = (slice(max(0, zc - dz), min(vol.shape[0], zc + dz + 1)),
                  slice(max(0, yc - dy), min(vol.shape[1], yc + dy + 1)),
                  slice(max(0, xc - dx), min(vol.shape[2], xc + dx + 1)))
            return float(vol[sl].max()) if vol[sl].size else 0.0

        kc, ks = f"c{t}", f"s{t}"
        if kc not in zx.files:
            continue
        xc_ = np.asarray(zx[kc], float)
        xs_ = np.asarray(zx[ks], float)
        keep = xs_ >= 0.15
        xc_, xs_ = xc_[keep], xs_[keep]
        if len(xc_) == 0:
            continue
        det_um = xc_ * SCALE[None, :]
        tree = cKDTree(gt_um)
        claimed = set()
        matched = np.zeros(len(xc_), bool)
        for i in np.argsort(-xs_):
            near = tree.query_ball_point(det_um[i], MATCH_UM)
            near.sort(key=lambda g: float(np.linalg.norm(gt_um[g]
                                                         - det_um[i])))
            free = [g for g in near if g not in claimed]
            if free:
                claimed.add(free[0])
                matched[i] = True
        dc_c = np.asarray(zd[kc], float) if kc in zd.files \
            else np.zeros((0, 3))
        dc_s = np.asarray(zd[ks], float) if kc in zd.files \
            else np.zeros(0)
        dc_c = dc_c[dc_s >= 0.05]
        dc_tree = cKDTree(dc_c * SCALE[None, :]) if len(dc_c) else None
        for i in range(len(xc_)):
            cons = bool(dc_tree is not None
                        and dc_tree.query(det_um[i])[0] <= CONSENSUS_UM)
            rows.append((int(matched[i]), bright_at(xc_[i]), int(cons)))
        for _ in range(20):
            v = rng.uniform([0, 0, 0], [64, 256, 256])
            rand_bright.append(bright_at(v))
        del vol
    del z, img
    gc.collect()
    print(f"  {stem} done ({time.time() - t0:.0f}s)", flush=True)

R = np.asarray(rows, float)
RB = np.asarray(rand_bright, float)
if len(R):
    m = R[:, 0] > 0.5
    um = ~m
    b_m, b_u = R[m, 1], R[um, 1]
    thr_bright = float(np.percentile(b_m, 25))
    bg_p90 = float(np.percentile(RB, 90)) if len(RB) else float("nan")
    print(f"\n  peaks: {len(R):,} total, matched {int(m.sum()):,}, "
          f"unmatched {int(um.sum()):,} "
          f"({100.0 * um.mean():.1f}% of peaks)", flush=True)
    print(f"  brightness (raw counts): matched q25/50/75 "
          f"{np.percentile(b_m, [25, 50, 75]).round(0).tolist()}, "
          f"unmatched {np.percentile(b_u, [25, 50, 75]).round(0).tolist()}, "
          f"random p50/p90 "
          f"{np.percentile(RB, [50, 90]).round(0).tolist() if len(RB) else 'n/a'}",
          flush=True)
    cell_like = um & (R[:, 1] >= thr_bright) & (R[:, 2] > 0.5)
    dark_alone = um & (R[:, 1] < thr_bright) & (R[:, 2] < 0.5)
    print(f"\n  unmatched peaks, cross-classified (bright = above the "
          f"matched peaks' own q25 = {thr_bright:.0f}; consensus = ep05 "
          f"peak within {CONSENSUS_UM} um):", flush=True)
    for name, mask in (
            ("bright AND consensus  (cell-like)", cell_like),
            ("bright, no consensus", um & (R[:, 1] >= thr_bright)
             & (R[:, 2] < 0.5)),
            ("dim, consensus", um & (R[:, 1] < thr_bright)
             & (R[:, 2] > 0.5)),
            ("dim AND no consensus  (junk-like)", dark_alone)):
        print(f"    {name:<36} {int(mask.sum()):>7,} "
              f"({100.0 * mask.sum() / max(1, um.sum()):.1f}% of "
              f"unmatched)", flush=True)
    n_cell_like = int(cell_like.sum())
    gap = n_cell_like / max(1, n_labels + n_cell_like)
    print(f"\n  labels seen in sampled frames: {n_labels:,}", flush=True)
    print(f"  completeness-gap ESTIMATE from this direction: "
          f"{100.0 * gap:.2f}%  (cell-like unmatched / (labels + "
          f"cell-like unmatched))", flush=True)
    print("  ⚠ both classifiers are imperfect: a dim real cell reads "
          "junk-like (gap UNDERcounted), a bright artifact both "
          "detectors like reads cell-like (gap OVERcounted). Read this "
          "together with test 1, which has the opposite blind spot.",
          flush=True)

print("\n=== reading it ===", flush=True)
print("  Test 1 bounds THEIR tracker's per-frame dropout from their own "
      "tracks; test 2 estimates unlabeled-cell density from label-free "
      "evidence. Small+small -> zh keeps its denominator status with a "
      "footnote and fusion negatives get the dark+no-consensus filter. "
      "Large either way -> the user's challenge stands: zh precision "
      "history gets a systematic footnote and negative mining as "
      "proposed is unsound.", flush=True)
print(f"\nZHCOMPLETE DONE in {(time.time() - t0) / 60:.1f} min",
      flush=True)
