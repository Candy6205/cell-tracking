"""hp44full_ref.py -> fk44f4.py: the 44b6 forks the retrain's joint gate needs,
and the 44b6 score it has never had on more than 12 movies.

THE GAP. The shipped `fork_gate_joint.pkl` is trained on 128 6bba counted
forks PLUS 71 44b6 ones, and the 44b6 half comes from `divbench5` -- generated
under the SHIPPED BASE-pool ranker. Retrain the ranker and the 6bba forks
change while the 44b6 training forks do not, so the joint gate is fitted on a
mixture of two different fork-generating processes. `rkpool` measured the
consequence without being able to name it: the joint gate's margin over the
6bba-only gate collapses from **+0.0037** on the shipped ranker to **+0.0004**
on the retrained one. This run produces the missing half.

AND IT CLOSES THE OTHER GAP IN THE SAME PASS, which is why it is worth two
hours rather than one. The retrain's 44b6 evidence is currently `rkfloor`'s
**12** movies (+0.0011 at floor 0.48). `hp44full` established that 12 44b6
movies carry 2 of the domain's 26 GT divisions and that the small draw
misreads this domain. This run scores all **71** with the retrained fold
rankers, so `forks_44b6.csv` and the per-movie CSV both fall out of one solve.

THE CHANGE, and it is rkpool's edit ported to divbench5's lineage. Today
stage A trains from `LAB`, built with `EP_OF["BASE"]` for all 199, while
stage B solves F4 -- the same train/deploy split rkpool measured on 6bba.
`RANK_TRAIN = "F4"` builds labelled F4 rows for all 199 and binds stage A to
them. `fuse44b6` + `ep44b6` (2026-09-07) and `fuse6bba` + `ep6bba` mean all
199 movies carry F4 primaries and F4 head probabilities, so the F4 pool is
the same 199 the BASE pool is -- no pool-size confound.

⚠ THE ANCHORS MOVE, BY CONSTRUCTION. `hp44full_ref` reproduces divbench5 v2's
REF over all 71 (6/61/20, div_J 0.0690, adj_J 0.8868, score 0.8937) and
`ANCHOR_D187` 0.8353. `ANCHOR_D187` loads a FIXED pkl and cannot see
RANK_TRAIN, so it stays HARD and is the proof the harness did not move. The
REF row is the measurement and is downgraded to `[expected]`, exactly as
rkpool does -- and for the same reason rkpool's v1 died at the very end
without it.

⚠ AND THE ANCHOR CONSTANTS IN THIS LINEAGE ARE LITERALS, NOT THE
`ANCHOR_HARD` / `ANCHOR_SOFT` DICTS. Those dicts are never read (rkhp,
2026-09-09, working-agreement.md). The downgrade below is written against the
`ANCHOR_FAIL` site, which is the code that actually fires.

WHAT IS READ FROM IT:
  1. `forks_44b6.csv` -- 71 movies' counted forks under the retrained ranker.
     Joined with `rkpool`'s 6bba forks it gives a COHERENT joint gate, priced
     offline the way `gatelab` prices them (the pruning curve is a pure
     function of (label, gate p)). If the joint gate's margin comes back, the
     collapse was the mixture and the retrain becomes a one-constant change.
  2. `fk44f4_F4.csv` -- the retrain's 44b6 score on all 71, paired against
     `hp44full_ref`'s recorded control, 26 GT divisions instead of 2.

Attach: hp44full_ref's sources. CPU only, about 2 h.
"""
import os

src = open("hp44full_ref.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('DET_SRC = "F4"', 'DET_SRC = "F4"\n'
    '# Which primaries the fold rankers LEARN on. Stage B always solves\n'
    '# DET_SRC, so "BASE" is the shipped mismatch and "F4" removes it.\n'
    'RANK_TRAIN = "F4"')

# ---- 1. the F4 pool must be complete before it may be used -------------
rep('''BENCH_SET = set(BENCH)''',
    '''BENCH_SET = set(BENCH)
if RANK_TRAIN == "F4":
    _need = [n for n in ALL_TRAIN if n not in FDET_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no fdet_F4_ cache "
        f"({_need[:4]}) -- attach biohub-fuse44b6 / biohub-fuse6bba. A "
        f"smaller pool than the BASE arm's would make the delta pool size, "
        f"not primaries.")
    _need = [n for n in ALL_TRAIN if n not in EDGEPROB_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no edgeprob_F4_ cache "
        f"({_need[:4]}) -- attach biohub-ep44b6 / biohub-ep6bba.")
    print(f"  RANK_TRAIN=F4: all {len(ALL_TRAIN)} movies carry F4 primaries "
          f"and F4 head probabilities; the F4 pool IS the BASE pool.",
          flush=True)''')

# ---- 2. labelled F4 rows for every movie the rankers train on ----------
# FEATS still keeps only what stage B solves, so the 128 pool-only 6bba
# movies are labelled and released at once: peak memory holds one extra
# feats4, not 128.
rep('''    if name in BENCH_SET:
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        FEATS[name] = (seeds4, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4''',
    '''    _solved = name in BENCH_SET
    if _solved or RANK_TRAIN == "F4":
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        if _solved:
            FEATS[name] = (seeds4, feats4)
        if RANK_TRAIN == "F4":
            LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4
        if not _solved:            # pool-only: labelled, then released
            del seeds4, feats4''')

rep('''LAB = {}                   # name -> (X, y) labelled rows, all 199''',
    '''LAB = {}                   # name -> (X, y) labelled rows, all 199
LAB4 = {}                  # the same rows on F4 primaries, when RANK_TRAIN=F4''')

# ---- 3. stage A trains from whichever table RANK_TRAIN names ----------
rep('''COLS = ARMS["+BIDIR"]
MODELS_FOLD = {}''',
    '''COLS = ARMS["+BIDIR"]
# THE ONE DIFFERENCE FROM hp44full_ref. Same names, same folds, same columns,
# same estimator, same seed; only which table the rows come from.
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
    # The held-out fold is read on the SOLVED primaries in both arms, because
    # that is what stage B uses. For RANK_TRAIN="BASE" that is a cross-domain
    # read on purpose: it is what the mismatch costs in AUC.
    _HELD = LAB4 if LAB4 else LAB
    Xh = np.concatenate([_HELD[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([_HELD[n][1] for n in held])''')

# ---- 4. the outputs must not overwrite hp44full_ref's ------------------
rep('f"hp44full_{DET_SRC}.csv"', 'f"fk44f4_{RANK_TRAIN}.csv"')

# ---- 5. the REF anchor cannot hold in the arm that retrains the ranker -
# rkpool v1 scored all its movies, wrote its CSV, and then died on exactly
# this. ANCHOR_D187 loads a fixed pkl and stays HARD; only the REF row moves.
rep('''assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''',
    '''if RANK_TRAIN != "BASE":
    _moved = [f for f in ANCHOR_FAIL if f[0] != "ANCHOR_D187"]
    ANCHOR_FAIL = [f for f in ANCHOR_FAIL if f not in _moved]
    for _f in _moved:
        print(f"  [expected] {_f[0]} {_f[1]} moved to {_f[2]} from {_f[3]} -- "
              f"this arm retrains the ranker, so the REF graphs are NOT "
              f"divbench5's and this anchor cannot hold; it is the "
              f"measurement. ANCHOR_D187 loads a fixed pkl, is untouched by "
              f"the retrain, and stays hard.", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "fk44f4.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written fk44f4.py", len(src), "RANK_TRAIN F4, DET_SRC F4")
