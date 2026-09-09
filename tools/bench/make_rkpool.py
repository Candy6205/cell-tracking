"""hparabench_ref.py -> rkpool.py / rkpool_ref.py: the ranker's OWN
water-and-soil problem -- it is trained on BASE detections and deployed on F4.

THE USER'S POINT, generalised. Told that ttabench's -0.0051 was measured with
a ranker fitted on non-TTA columns: *"哦，排序器水土不服，那先重训也行"*.
Chasing that down produced something bigger than the TTA question.

`edgeprob_rank2.py:1608` builds the production pool as

    _jobs = [(n, "BASE", "pool") for n in POOL175]

so `edge_ranker_ep175v3_bidir.pkl` -- the pkl every submission loads --
learned the map from features to truth on BASE primaries (the xl three-branch
mean) and is deployed on F4 ones (that mean plus ep05 DeepCenter as a fourth
equal-weight map). `hparabench_ref` inherits it: stage A trains from `LAB`,
built with `EP_OF["BASE"]`, and stage B solves F4. **The mismatch ships, and
it has never been measured.**

(It also means ttabench was worse than "90% noTTA". Its override replaced
`EDGEPROB_F4` only; training reads `EDGEPROB`. Not one tta4 row entered
training -- the fold rankers were 100% noTTA. That -0.0051 is water-and-soil,
not TTA, and the matched test needs tta4 in the TRAINING rows.)

WHY IT COSTS NO GPU, and why it could not be run until two days ago. A ranker
trained on F4 needs F4 primaries and F4 head probabilities for the whole pool.
On 2026-09-07 `fuse44b6` (59 of 59, 90.1 min) and `ep44b6` (59 of 59, 45.3
min) landed exactly those for the 59 44b6 train movies that lacked them, so
**all 199 movies now have both**. hparabench simply never attached them -- its
log reads `199 BASE + 140 F4`. Attach biohub-fuse44b6 + biohub-ep44b6 and the
F4 pool is the same 175 movies the BASE pool is. No pool-size confound, no new
GPU.

THE TWO ARMS, identical in every other respect -- same movies, same folds,
same 300-iteration HistGB, same random_state, same stage B, same DET_SRC:

    rkpool_ref.py   RANK_TRAIN = "BASE"   what ships today
    rkpool.py       RANK_TRAIN = "F4"     trained on the coordinates it is
                                          deployed on

ANCHORS. The BASE arm is hparabench_ref's code path with LAB_TRAIN bound back
to LAB, so it must reproduce divbench3 to the digit: REF over the 128 =
50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266, and the gate cross cells
0.2442 / 0.2805 / 0.2591. Those asserts are inherited unchanged and are NOT
restated here -- if attaching two more cache datasets perturbed the BASE path,
they fail on their own, which is the point of running the control arm rather
than reusing round 2's CSV.

⚠ ONE ASYMMETRY IS DELIBERATE. The held-out-fold AUC is computed on F4 rows in
both arms, because F4 is what stage B solves. For the BASE arm that is a
cross-domain read on purpose: it is what the mismatch costs, in AUC, before
any of it reaches the ILP.

⚠ WHAT THIS IS NOT. It does not say the official head likes our nodes better.
`progress.md:104` records that both official edge heads were trained on all
199 train movies, so every local head number is in-sample, and `progress.md:46`
records submit_off (official nodes + head only, no ranker) at +0.004 / +0.057
locally and **0.895 on LB, the worst on record**. This arm changes only which
primaries the RANKER learned on; the head's caches are the same files in both.

Attach: hparabench_ref's sources PLUS faustine649/biohub-fuse44b6 and
faustine649/biohub-ep44b6. CPU only.
"""
import os

src = open("hparabench_ref.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-hparabench"',
    'SCRIPT_VERSION = "2026-09-09a-rkpool"\n'
    '# Which primaries the fold rankers LEARN on. Stage B always solves\n'
    '# DET_SRC (F4), so "BASE" is the shipped mismatch and "F4" removes it.\n'
    'RANK_TRAIN = "BASE"')

# ---- 1. the F4 pool must be complete before it may be used -------------
# Goes in the "Rule 1: a missing dataset is a crash here" block, which is the
# first point where ALL_TRAIN exists (line 1483; the FDET_F4 discovery at 1421
# is too early -- checkgen caught that in one second on the first draft).
rep('''_missing = [n for n in NAMES_EVAL if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for: {_missing} -- attach biohub-fusecache"''',
    '''_missing = [n for n in NAMES_EVAL if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for: {_missing} -- attach biohub-fusecache"
# fuse44b6 + ep44b6 (2026-09-07, 59 of 59 each) completed the F4 side of the
# pool. Without them RANK_TRAIN="F4" would silently train on 140 movies
# against the BASE arm's 175, and the delta would be pool size, not primaries.
if RANK_TRAIN == "F4":
    _need = [n for n in ALL_TRAIN if n not in FDET_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no fdet_F4_ cache "
        f"({_need[:4]}) -- attach faustine649/biohub-fuse44b6. A smaller pool "
        f"than the BASE arm's would make the delta pool size, not primaries.")
    _need = [n for n in ALL_TRAIN if n not in EDGEPROB_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no edgeprob_F4_ cache "
        f"({_need[:4]}) -- attach faustine649/biohub-ep44b6.")
    print(f"  RANK_TRAIN=F4: all {len(ALL_TRAIN)} movies carry F4 primaries "
          f"and F4 head probabilities; the F4 pool IS the BASE pool.",
          flush=True)''')

# ---- 2. labelled F4 rows for every movie the rankers train on ----------
# Today feats4 is built for `role == "6bba" or name in SET_44B6` (140 movies)
# and labelled only for the 128 6bba. RANK_TRAIN="F4" needs labelled F4 rows
# for all 199. FEATS still keeps only what stage B solves, so the 59 pool-only
# 44b6 movies are labelled and freed at once: peak memory holds one extra
# feats4, not 59.
rep('''    if role == "6bba" or name in SET_44B6:
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        FEATS[name] = (seeds4, feats4)
        if role == "6bba":
            LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4''',
    '''    _solved = role == "6bba" or name in SET_44B6
    if _solved or RANK_TRAIN == "F4":
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        if _solved:
            FEATS[name] = (seeds4, feats4)
        if role == "6bba" or RANK_TRAIN == "F4":
            LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4
        if not _solved:            # pool-only 44b6: labelled, then released
            del seeds4, feats4''')

# ---- 3. stage A trains from whichever table RANK_TRAIN names ----------
rep('''COLS = ARMS["+BIDIR"]
MODELS_FOLD = {}''',
    '''COLS = ARMS["+BIDIR"]
# THE ONE DIFFERENCE BETWEEN THE ARMS. Same names, same folds, same columns,
# same estimator, same seed; only which table the rows come from. LAB holds
# BASE-primary rows for all 199 (what ships); LAB4 holds F4-primary rows -- for
# the 128 6bba always, and for all 199 when RANK_TRAIN == "F4".
LAB_TRAIN = LAB if RANK_TRAIN == "BASE" else LAB4
print(f"  RANK_TRAIN={RANK_TRAIN}: the fold rankers learn on "
      f"{'BASE (the xl three-branch mean)' if RANK_TRAIN == 'BASE' else 'F4 (that mean + ep05 DeepCenter)'}"
      f" primaries while stage B solves {DET_SRC}. "
      f"{'This is the mismatch that ships.' if RANK_TRAIN == 'BASE' else 'This removes it.'}",
      flush=True)
_need = [n for n in ALL_TRAIN if n not in LAB_TRAIN]
assert not _need, (f"RANK_TRAIN={RANK_TRAIN} but {len(_need)} movies have no "
                   f"labelled rows: {_need[:4]}")
MODELS_FOLD = {}''')

rep('''    Xa = np.concatenate([LAB[n][0] for n in train_names]).astype(np.float32)[:, COLS]
    ya = np.concatenate([LAB[n][1] for n in train_names])
    Xh = np.concatenate([LAB[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([LAB[n][1] for n in held])''',
    '''    Xa = np.concatenate([LAB_TRAIN[n][0] for n in train_names]).astype(np.float32)[:, COLS]
    ya = np.concatenate([LAB_TRAIN[n][1] for n in train_names])
    # The held-out fold is read on F4 rows in BOTH arms, because F4 is what
    # stage B solves. For RANK_TRAIN="BASE" that is a cross-domain read on
    # purpose: it is what the mismatch costs in AUC, before the ILP sees it.
    Xh = np.concatenate([LAB4[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([LAB4[n][1] for n in held])''')

rep('''          f"({ya.mean():.3f} pos), val AUC {val_auc:.4f}; held-out fold "
          f"({len(held)} 6bba movies, {len(yh)} rows) AUC {held_auc:.4f}  "''',
    '''          f"({ya.mean():.3f} pos), val AUC {val_auc:.4f} (own primaries); "
          f"held-out fold read on F4 rows "
          f"({len(held)} 6bba movies, {len(yh)} rows) AUC {held_auc:.4f}  "''')

# LAB is freed after stage A. In the BASE arm LAB_TRAIN is the same object, so
# name it in the del rather than leaving a live alias behind the gc.collect().
rep('''del LAB
gc.collect()''',
    '''del LAB, LAB_TRAIN
gc.collect()''')

# ---- 4. the CSV must not overwrite the other arm's -------------------
rep('f"hparabench_{DET_SRC}.csv"', 'f"rkpool_{RANK_TRAIN}.csv"')

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for fname, rt in (("rkpool.py", "F4"), ("rkpool_ref.py", "BASE")):
    body = src.replace('RANK_TRAIN = "BASE"', f'RANK_TRAIN = "{rt}"', 1)
    assert body.count(f'RANK_TRAIN = "{rt}"') >= 1, rt
    open(os.path.abspath(os.path.join(base, fname)), "w",
         encoding="utf-8", newline="\n").write(body)
    print("written", fname, len(body), "RANK_TRAIN", rt)
