"""ttascore.py -- the readout half of ttaprobe, on CPU, off its saved caches.

WHY THIS EXISTS. `biohub-ttaprobe` ran all 36 passes (12 movies x 3 view
counts, 69 min T4, 23,760 extra U-Net passes) and wrote every cache, then
died on the LAST line of the run:

    NameError: name 'defaultdict' is not defined

`_label_pairs` was cut into a file whose imports do not include it. The
probabilities are on disk and correct, so re-running the GPU work would buy
nothing. This does the labelling and the AUC on CPU instead.

WHAT IT ANSWERS, unchanged from ttaprobe's pre-registered reading: does
averaging the UNet features over D4 views make the EDGE PROBABILITY more
informative? Single-column AUC against poolsize's own labels -- no bench, no
ILP, no thresholds.

  * AUC flat with the probabilities moved -> the mechanism works and buys
    nothing on our features; the ~20 GPU hours for real caches are refused.
  * tta8 > tta4 > noTTA -> it is worth the caches, and the view count
    matters, which is their claim.
  * AUC down -> their +0.005 came from something else in that step, or it
    needs the eight-view DETECTION ensemble we do not run.

The GPU log already shows the probabilities MOVED, and hard: median p_fwd
0.827 -> 0.728 on 6bba_2312ac41 and 0.669 -> 0.399 on 6bba_283bf9f1, with
the `fwd>0.5` counts falling in step. So the wrapper is not a no-op, and a
flat AUC here is a real null rather than a broken run.

Attach: faustine649/biohub-ttaprobe (the 36 probability caches),
faustine649/biohub-fusecache + biohub-fuse6bba (fdet_F4_ for the
primaries), faustine649/cell-tracking (viz_cells' geff reader), and the
competition data. CPU only.
"""
import sys
import pathlib
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

SCRIPT_VERSION = "2026-09-09a-ttascore"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 5.0
ARMS = ["noTTA", "tta4", "tta8"]

BASE = pathlib.Path("/kaggle/input")
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "faustine649/cell-tracking is not mounted"
sys.path.insert(0, code)
from viz_cells import read_geff                                # noqa: E402


def _walk(pred):
    out, stack = {}, [BASE]
    while stack:
        d = stack.pop()
        if d.name.endswith((".zarr", ".geff")):
            continue
        for f in d.glob("*.npz"):
            k = pred(f.name)
            if k is not None:
                out[k] = f
        if len(d.parts) - len(BASE.parts) < 5:
            stack += [p for p in d.iterdir() if p.is_dir()]
    return out


def _fdet_key(n):
    if n.startswith("fdet_F4hp_") or not n.startswith("fdet_F4_"):
        return None
    return n[len("fdet_F4_"):-len(".npz")]


FDET = _walk(_fdet_key)
print(f"fdet_F4 caches: {len(FDET)}", flush=True)

PROB = {}
for a in ARMS:
    pre = "edgeprob_" + a + "_"
    PROB[a] = _walk(lambda n, p=pre: n[len(p):-len(".npz")]
                    if n.startswith(p) else None)
    print(f"{a}: {len(PROB[a])} probability caches", flush=True)

NAMES = sorted(set.intersection(*[set(PROB[a]) for a in ARMS]) & set(FDET))
print(f"movies in ALL arms and carrying an fdet_F4 cache: {len(NAMES)}",
      flush=True)
assert len(NAMES) == 12, (len(NAMES), NAMES)

COMP = next((p for p in (BASE / "biohub-cell-tracking-during-development",
                         BASE / "competitions" /
                         "biohub-cell-tracking-during-development")
             if p.exists()), None)
assert COMP is not None, "competition data not mounted"
TRAIN = COMP / "train"


def primaries(name):
    """[(N_t, 3) full-res float] per frame, seeds_of order -- the same
    primaries edgeprob_cache indexed its (t, i, j) rows against."""
    z = np.load(FDET[name])
    T = 0
    while f"c{T}" in z.files:
        T += 1
    out = []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        out.append(c[s >= UNET_THRESH])
    z.close()
    return out


def label_pairs(name, z, prim):
    """poolsize's labelling rule.

    A row counts only when the SOURCE primary matched a GT node that HAS
    children; the label is 1 iff the TARGET primary matched one of them.
    The same population every other number on this project uses.
    """
    gt = read_geff(TRAIN / f"{name}.geff")
    gpos = np.stack([np.asarray(gt[c], float)
                     for c in ("z", "y", "x")], axis=1) * SCALE
    gtt = np.asarray(gt["t"], int)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    child = {}
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is not None and rb is not None:
            child.setdefault(ra, []).append(rb)
    by_t = defaultdict(list)
    for i_ in range(len(gtt)):
        by_t[int(gtt[i_])].append(i_)

    def match(pos_um, idx):
        if not idx:
            return {}
        tr = cKDTree(gpos[idx])
        out = {}
        for d_, x_ in enumerate(pos_um):
            dd, gi = tr.query(x_)
            if dd <= MATCH_UM:
                out[d_] = idx[int(gi)]
        return out

    et = np.asarray(z["t"], int)
    ei = np.asarray(z["i"], int)
    ej = np.asarray(z["j"], int)
    keep, lab = [], []
    for t_ in sorted(set(et.tolist())):
        mp = match(prim[t_] * SCALE[None, :], by_t.get(t_, []))
        mc = match(prim[t_ + 1] * SCALE[None, :], by_t.get(t_ + 1, []))
        det_of = {g: d for d, g in mc.items()}
        for r_ in np.where(et == t_)[0]:
            i_, j_ = int(ei[r_]), int(ej[r_])
            if i_ not in mp:
                continue
            kids = child.get(mp[i_], [])
            if not kids:
                continue
            tgt = {det_of.get(b_) for b_ in kids}
            tgt.discard(None)
            keep.append(int(r_))
            lab.append(1 if j_ in tgt else 0)
    return np.asarray(keep, int), np.asarray(lab, int)


P = {a: [] for a in ARMS}
Y = {a: [] for a in ARMS}
for n in NAMES:
    prim = primaries(n)
    base_key = None
    for a in ARMS:
        z = np.load(PROB[a][n])
        k, y = label_pairs(n, z, prim)
        key = (np.asarray(z["t"], np.int64), np.asarray(z["i"], np.int64),
               np.asarray(z["j"], np.int64))
        # Every arm ran on the SAME primaries through the same gate, so the
        # (t, i, j) row set must be identical. If it is not, the arms are not
        # scoring the same pairs and nothing below compares anything.
        if base_key is None:
            base_key = key
        else:
            assert all(np.array_equal(base_key[c], key[c]) for c in range(3)), (
                f"{n}/{a}: (t, i, j) rows differ from the noTTA arm")
        P[a].append(np.asarray(z["p_fwd"], float)[k])
        Y[a].append(y)
        z.close()
    print(f"  {n}: {len(Y[ARMS[0]][-1]):,} labelled rows", flush=True)

print(f"\n=== EDGE-FEATURE TTA: does averaging the UNet features over D4 "
      f"views make the edge probability more informative? ===", flush=True)
print(f"  {'arm':<8} {'rows':>10} {'pos':>7} {'AUC p_fwd':>11} "
      f"{'vs noTTA':>10}  {'p_fwd p50':>10}   |dp| p50 / p90 vs noTTA",
      flush=True)
base_auc = None
base_p = None
for a in ARMS:
    p = np.concatenate(P[a])
    y = np.concatenate(Y[a])
    auc = float(roc_auc_score(y, p))
    if base_auc is None:
        base_auc, base_p, d = auc, p, ""
    else:
        dp = np.abs(p - base_p)
        d = f"{np.percentile(dp, 50):.5f} / {np.percentile(dp, 90):.5f}"
    print(f"  {a:<8} {len(y):>10,} {y.mean():>7.4f} {auc:>11.5f} "
          f"{auc - base_auc:>+10.5f}  {np.median(p):>10.4f}   {d}", flush=True)

print("\n  Read against the rule fixed before the GPU run:", flush=True)
print("  * flat AUC with the probabilities moved -> the mechanism works and "
      "buys nothing here; the ~20 GPU hours for real caches are REFUSED.",
      flush=True)
print("  * tta8 > tta4 > noTTA -> worth the caches, and the view count "
      "matters, which is their claim.", flush=True)
print("  * AUC DOWN -> their +0.005 came from something else in that step, "
      "or it needs the eight-view DETECTION ensemble we do not run.",
      flush=True)
print("TTASCORE OK", flush=True)
