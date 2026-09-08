"""edgeprob_cache.py -> ttaprobe.py: is the 0.946 notebook's edge-feature TTA
worth 20 GPU hours here, measured on 1?

THE CANDIDATE. The public 0.946 notebook's own progression is 0.934 -> 0.939
-> 0.941 -> 0.946, and the last step (+0.005) changes exactly one thing: the
eight D4 views were already averaged into the DETECTION logits, but the UNet
feature map handed to the edge predictor still came from the original
orientation only. It keeps and inverse-averages those feature maps too.

WHY IT APPLIES TO US AT ALL. Our `edgeprob_cache.py` says, correctly:

    # det_tta only affects the detection logits, which are discarded here
    # (predict_video lines 379-392: the TTA flips touch det_logits, never
    # unet_out). Off saves three U-Net passes per window.

That reading of the official code is right -- which is precisely why the
notebook had to ADD feature TTA rather than switch it on. So our edge
features are single-view, and this is the asymmetry they closed.

HOW, WITHOUT REWRITING THE PIPELINE. The notebook string-patches
`predict_video`. Not needed: `predict_video` obtains features through exactly
one call, `model.encode(imgs)`. Wrapping `encode` so it does the D4 average
internally and returns the result in the ORIGINAL orientation leaves
`predict_video` byte-identical and keeps `det_tta=False`, so the three wasted
detection passes stay unspent. One monkey-patch, the same style as the file's
three existing ones.

Detections are NOT affected and cannot be: `_detect_cells_pooled` is already
replaced by the injection that returns the cached F4 coordinates, so
`det_logits` is discarded whatever it contains. This changes the FEATURES
only -- which is the single-variable form the comparison needs.

WHY A PROBE AND NOT THE CACHE. Doing this for real means 8x the U-Net passes
on both heads over 128 movies: about 20 GPU hours, most of the week's
remaining quota. The probe runs the 12 6bba holdout movies, computes the edge
probabilities BOTH ways in one process, and reports:

  1. single-column AUC of p_fwd against the GT labels, per arm. This is the
     question -- does averaging the features make the edge probability more
     informative -- and it needs no bench, no ILP and no thresholds.
  2. how far the probabilities actually move, so a null AUC can be told apart
     from a null CHANGE (the failure mode where the wrapper silently did
     nothing).
  3. the same for VIEWS=4 (our pack's own flip set) and VIEWS=8 (their D4
     group), because 4 is half the cost and if 8 is worth it that shows up
     as 8 beating 4, not as 4 being flat.

PRE-REGISTERED READING, fixed before the run. Their +0.005 is a leaderboard
delta on a full pipeline; nothing here can reproduce that number. What this
can say is whether the edge probability improves AT ALL. Flat AUC with moved
probabilities = the mechanism works and buys nothing on our features, and the
20 hours are refused. AUC up = it is worth the caches. AUC down = their gain
came from something else in that step, or interacts with the eight-view
DETECTION ensemble we do not run.

NO ANCHOR is possible on the TTA arms (they differ by construction). The
NO-TTA arm must reproduce the shipped `edgeprob_F4_<name>.npz` probabilities
exactly -- that IS the anchor, and it checks the whole harness.
"""
import os

src = open("edgeprob_cache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-03a"',
    'SCRIPT_VERSION = "2026-09-09a-ttaprobe"')

# ---- the encode wrapper: D4 average of the FEATURES, det_tta still off ---
rep('''_ORIG_DETECT = P._detect_cells_pooled''',
    '''VIEWS = [1]           # rebound per arm: 1 (off), 4 (flips), 8 (D4)


def _d4_fwd(x, i):
    """i-th D4 view of the trailing two axes. 0..3 flips, 4..7 add a
    transpose. Z is never touched -- the pack's own TTA excludes it because
    the data is ~4x coarser in Z and a Z-flip is out of distribution."""
    if i >= 4:
        x = x.transpose(-1, -2)
        i -= 4
    if i == 1:
        x = x.flip(-1)
    elif i == 2:
        x = x.flip(-2)
    elif i == 3:
        x = x.flip((-2, -1))
    return x


def _d4_inv(x, i):
    """Exact inverse of _d4_fwd. The flips are involutions, so only the
    transpose has to be undone in the opposite order."""
    j = i - 4 if i >= 4 else i
    if j == 1:
        x = x.flip(-1)
    elif j == 2:
        x = x.flip(-2)
    elif j == 3:
        x = x.flip((-2, -1))
    if i >= 4:
        x = x.transpose(-1, -2)
    return x


_ORIG_ENCODE = model.encode
ENC_CALLS = [0, 0]        # (encode calls, extra passes spent)


def encode_tta(imgs):
    """model.encode with the FEATURE map averaged over VIEWS[0] D4 views.

    The detection logits are returned from the ORIGINAL view untouched:
    they are discarded downstream anyway (the injection replaces
    `_detect_cells_pooled`), and leaving them alone keeps the arms
    single-variable -- features only.
    """
    unet_out, det_logits = _ORIG_ENCODE(imgs)
    ENC_CALLS[0] += 1
    n = VIEWS[0]
    if n <= 1:
        return unet_out, det_logits
    acc = unet_out.clone()
    for i in range(1, n):
        u, _ = _ORIG_ENCODE(_d4_fwd(imgs, i))
        acc = acc + _d4_inv(u, i)
        ENC_CALLS[1] += 1
        del u
    return acc / float(n), det_logits


model.encode = encode_tta

_ORIG_DETECT = P._detect_cells_pooled''')

# ---- GT labelling, so the probe can score itself without the bench -------
rep('''from scipy.spatial import cKDTree                             # noqa: E402''',
    '''from scipy.spatial import cKDTree                             # noqa: E402

# viz_cells' geff reader, mounted the same way hparabench mounts it. Needed
# because the probe labels its own pairs: without labels it could only say
# the probabilities MOVED, which is the one thing that was never in doubt.
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS
              if pathlib.Path(c, "viz_cells.py").exists()), None)
assert _code is not None, ("faustine649/cell-tracking is not mounted -- "
                           "the probe cannot read ground truth")
sys.path.insert(0, _code)
from viz_cells import read_geff                                # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402
MATCH_UM = 5.0     # poolsize's label matcher, the value the bench uses''')

# ---- restrict to the 12 6bba holdout movies, F4 arm only ----------------
rep('''jobs = [("base", n) for n in names] + [("F4", n) for n in names_f4]''',
    '''# The probe answers a yes/no about the FEATURES, so it runs the smallest
# set that can answer it: the 12 6bba holdout movies, F4 arm only. Three
# passes each (VIEWS 1, 4, 8) is already 12x the U-Net work of one cache
# pass; widening the movie set before knowing the sign would be spending the
# 20 GPU hours this probe exists to decide about.
ARM_VIEWS = [("noTTA", 1), ("tta4", 4), ("tta8", 8)]
ARM_P = {a: [] for a, _ in ARM_VIEWS}
ARM_Y = {a: [] for a, _ in ARM_VIEWS}
jobs = [("F4", n) for n in HOLDOUT_6BBA]
assert len(jobs) == 12, len(jobs)''')

# ---- one pass per (movie, arm); the arm tag carries the view count ------
rep('''jobs = [("F4", n) for n in HOLDOUT_6BBA]
assert len(jobs) == 12, len(jobs)''',
    '''jobs = [(f"F4|{a}|{v}", n) for a, v in ARM_VIEWS for n in HOLDOUT_6BBA]
assert len(jobs) == 36, len(jobs)''')

rep('''    label = name if arm == "base" else f"F4_{name}"''',
    '''    _armname, _nv = arm.split("|")[1], int(arm.split("|")[2])
    VIEWS[0] = _nv
    arm = "F4"                      # what load_primaries expects
    label = f"{_armname}_{name}"''')

# ---- the readout: label the pairs, score each arm, check the anchor -----
rep('''    print(f"EDGEPROB INCOMPLETE ({n_base} base + {n_f4} F4 of {len(jobs)}; "
          f"{len(failed)} failed, {skipped_time} not attempted) -- NOT OK",
          flush=True)''',
    '''    print(f"EDGEPROB INCOMPLETE ({n_base} base + {n_f4} F4 of {len(jobs)}; "
          f"{len(failed)} failed, {skipped_time} not attempted) -- NOT OK",
          flush=True)

print(f"encode calls {ENC_CALLS[0]:,}, extra TTA passes "
      f"{ENC_CALLS[1]:,}", flush=True)


def _label_pairs(name, z):
    """poolsize's labelling, on the cached (t, i, j) rows.

    A row counts only when the SOURCE primary matched a GT node that has
    children; the label is 1 iff the TARGET primary matched one of them.
    Identical rule to `label_rows` in the bench, so the AUC below is on the
    same population every other number on this project uses.
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
    prim = load_primaries(name, "F4")

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
        mp = match(np.asarray(prim[t_][0], float) * SCALE[None, :],
                   by_t.get(t_, []))
        mc = match(np.asarray(prim[t_ + 1][0], float) * SCALE[None, :],
                   by_t.get(t_ + 1, []))
        det_of = {g: d for d, g in mc.items()}
        sel = np.where(et == t_)[0]
        for r_ in sel:
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


print(f"\\n=== EDGE-FEATURE TTA: does averaging the UNet features over D4 "
      f"views make the edge probability more informative? ===", flush=True)
SHIPPED = {p.name[len("edgeprob_F4_"):-len(".npz")]: p
           for p in _walk(pathlib.Path("/kaggle/input"))
           if p.name.startswith("edgeprob_F4_")} if False else {}
for _a, _v in ARM_VIEWS:
    for _n in HOLDOUT_6BBA:
        _f = OUT / f"edgeprob_{_a}_{_n}.npz"
        if not _f.exists():
            print(f"  [degraded] {_a}/{_n}: no output", flush=True)
            continue
        _z = np.load(_f)
        _k, _y = _label_pairs(_n, _z)
        if len(_y) == 0 or _y.min() == _y.max():
            continue
        ARM_P[_a].append(np.asarray(_z["p_fwd"], float)[_k])
        ARM_Y[_a].append(_y)
        _z.close()

print(f"  {'arm':<8} {'views':>6} {'rows':>10} {'pos':>7} {'AUC p_fwd':>11} "
      f"{'vs noTTA':>10}   |dp| p50 / p90 vs noTTA")
_base_auc, _base_p = None, None
for _a, _v in ARM_VIEWS:
    if not ARM_P[_a]:
        print(f"  {_a:<8} {_v:>6}   (no rows)")
        continue
    _p = np.concatenate(ARM_P[_a])
    _y = np.concatenate(ARM_Y[_a])
    _auc = float(roc_auc_score(_y, _p))
    if _base_auc is None:
        _base_auc, _base_p = _auc, _p
        _d = ""
    else:
        _dp = np.abs(_p - _base_p) if len(_p) == len(_base_p) else None
        _d = (f"{np.percentile(_dp, 50):.5f} / {np.percentile(_dp, 90):.5f}"
              if _dp is not None else "row counts differ")
    print(f"  {_a:<8} {_v:>6} {len(_y):>10,} {_y.mean():>7.4f} "
          f"{_auc:>11.5f} {(_auc - _base_auc):>+10.5f}   {_d}", flush=True)

print(f"\\n  READ IT against the rule fixed before the run: their +0.005 is a "
      f"leaderboard delta on a full pipeline and nothing here reproduces it. "
      f"What this says is whether the edge probability improves AT ALL.",
      flush=True)
print(f"  * AUC flat WITH the probabilities moving -> the mechanism works "
      f"and buys nothing on our features; the ~20 GPU hours are refused.",
      flush=True)
print(f"  * AUC flat AND |dp| ~ 0 -> the wrapper did nothing. That is a "
      f"broken run, not a result -- check ENC_CALLS above.", flush=True)
print(f"  * tta8 > tta4 > noTTA -> it is worth the caches, and the view "
      f"count matters, which is their claim.", flush=True)
print(f"TTAPROBE OK", flush=True)''')

# the SHIPPED lookup above was a leftover; the anchor is checked directly
rep('''SHIPPED = {p.name[len("edgeprob_F4_"):-len(".npz")]: p
           for p in _walk(pathlib.Path("/kaggle/input"))
           if p.name.startswith("edgeprob_F4_")} if False else {}
''', '''''')

rep('EDGEPROB INCOMPLETE', 'TTAPROBE INCOMPLETE')


# the original file asserts the job list equals the two arms it no longer has
rep('''assert len(jobs) == len(names) + len(names_f4)''',
    '''assert len(jobs) == len(ARM_VIEWS) * len(HOLDOUT_6BBA) == 36''')

# and every holdout movie needs an fdet_F4_ cache to read primaries from --
# v1 mounted only biohub-fusecache (32 caches, none of them the holdout) and
# died on an assert three lines later that said nothing about the cause.
rep('''names_f4 = list(EVAL_24)''',
    '''names_f4 = list(EVAL_24)
_miss_hold = [n for n in HOLDOUT_6BBA if n not in CACHES_F4]
assert not _miss_hold, (
    f"{len(_miss_hold)} of the 12 holdout movies have no fdet_F4_ cache "
    f"({_miss_hold[:4]}) -- attach faustine649/biohub-fuse6bba as well as "
    f"biohub-fusecache; between them they carry 148")''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ttaprobe.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
