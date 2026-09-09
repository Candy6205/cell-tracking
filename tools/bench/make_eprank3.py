"""edgeprob_rank2.py -> eprank3.py / eprank3_ref.py: RETRAIN THE PRODUCTION
RANKER ON THE COORDINATES IT IS DEPLOYED ON, and write the pkl.

The user: *"你去把重训落地，这种只提供24部的资料的事情请不要再发生了"*.
`rkpool` measures whether the mismatch costs score, on 128 6bba movies out of
fold. This produces the shippable artefact: a `+BIDIR` pkl whose D175 pool
rows were computed on F4 primaries.

THE MISMATCH, in this file's own two lines. `edgeprob_rank2.py:1608`:

    _jobs = [(n, "BASE", "pool") for n in POOL175]
    for det_arm in DET_ARMS:
        _jobs += [(n, det_arm, ...) for n in NAMES_EVAL]

and :1707, where the arms are fitted:

    Xa, ya = X_pool[:, cols], y_pool

`X_pool` collects only `role == "pool"` rows, and every one of those is
`det_arm == "BASE"`. `NAMES_EVAL` is **24 movies**. So the F4 feature pass
existed on 24 movies and none of them were in the pool: the shipped
`edge_ranker_ep175v3_bidir.pkl` learned on the xl three-branch mean and is
deployed on that mean plus ep05 DeepCenter, and every feature that reads the
detector -- `score_i`, `abs_logit_diff`, `dens_i`, `dens_j`, `n_cand_i`, and
the head's own `p_fwd` / `p_rev` / `h_fused` -- shifts between the two.

**It was not a choice.** On 2026-09-03, when this ranker was trained,
`edgeprob_F4_` existed for 24 movies (`hparabench-ref`'s log:
`199 BASE + 24 F4`). `ep6bba` added 128 on 09-06 and `ep44b6` the last 59 on
09-07. The constraint lifted and nobody went back. That is the case the
working-agreement's new "cache the whole population" rule is written from.

THE TWO ARMS. One switch, `POOL_ARM`; everything else identical -- same 175
names, same 10% seeded validation split, same HistGB(max_iter=300,
random_state=0), same refit-on-all, same stage B, same 24 evaluation movies
on F4 primaries in both.

    eprank3_ref.py   POOL_ARM = "BASE"   what ships today
    eprank3.py       POOL_ARM = "F4"     trained where it is deployed

ANCHORS on the ref arm, from the record's soft anchors for this exact
harness: `+BIDIR` F4-6bba **0.9261**, F4-44b6 **0.8515**. `+BIDIR`'s fit does
not depend on ARM_ORDER (`perm` is a fresh `default_rng(0)` per arm and the
permutation-importance RNG touches no model), so cutting the other arms below
must not move it -- if those two numbers move, this port broke something and
the F4 arm is unreadable.

WHY ONLY `+BIDIR` IS TRAINED HERE. `edgeprob2_F4_` covers **24** movies, not
199 -- the second head was never run over F4 primaries for the pool. So
`+BIDIR2`, `+BOTH` and `+AVG` cannot be trained on an F4 pool at all; their
head-2 columns would be NaN on every pool row and HistGB would happily fit
that and produce a plausible-looking, meaningless model. They are cut in BOTH
arms so the two stay comparable. No loss: the record already killed the
second head ("the two heads agree too much to ensemble", −0.002 to −0.010 on
both groups). The head-only eval arms (`HEAD1`, `HEAD12`) are untouched --
they train nothing.

⚠ This kernel writes a pkl. It does NOT change what any submission loads;
`make_submit_ep2.py` selects by `RANKER_NAME` and still names the v3 pkl.
Switching production over is a separate, deliberate edit, and it should wait
for `rkpool`'s paired interval -- one of whose three pre-registered branches
is that BASE training acts as augmentation and F4 training is WORSE.

Attach: edgeprob_rank2's sources PLUS faustine649/biohub-fuse44b6,
faustine649/biohub-ep44b6, faustine649/biohub-fuse6bba, faustine649/biohub-ep6bba.
CPU only.
"""
import os

src = open("edgeprob_rank2.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-03b-edgeprob-rank2"',
    'SCRIPT_VERSION = "2026-09-09a-eprank3"\n'
    '# Which primaries the D175 POOL rows are computed on. The 24 evaluation\n'
    '# movies are scored on F4 in both arms, because F4 is what ships.\n'
    'POOL_ARM = "BASE"')

# ---- 1. only +BIDIR can be trained on an F4 pool ------------------------
rep('ARM_ORDER = ["+BIDIR", "+BIDIR2", "+BOTH", "+AVG"]      # trained',
    '''# CUT IN BOTH ARMS, so the two stay comparable. edgeprob2_F4_ covers 24
# movies, not 199: the second head was never run over F4 primaries for the
# pool, so +BIDIR2 / +BOTH / +AVG have NaN head-2 columns on every F4 pool row.
# HistGB treats NaN as a first-class split and would fit that silently. The
# record already killed the second head anyway (-0.002 to -0.010, both groups).
ARM_ORDER = ["+BIDIR"]                                  # trained''')

rep('''PKL_NAME = {"+BIDIR": "edge_ranker_ep175v3_bidir.pkl",
            "+BIDIR2": "edge_ranker_ep175v3_bidir2.pkl",
            "+BOTH": "edge_ranker_ep175v3_both.pkl",
            "+AVG": "edge_ranker_ep175v3_avg.pkl"}''',
    '''# v4, and the pool arm is IN THE FILENAME. Two pkls that differ only in which
# detections their training rows came from are exactly the pair a later
# submission must not confuse.
PKL_NAME = {"+BIDIR": f"edge_ranker_ep175v4_bidir_{POOL_ARM.lower()}.pkl"}''')

# ---- 2. the pool is featurised on POOL_ARM ------------------------------
rep('_jobs = [(n, "BASE", "pool") for n in POOL175]',
    '_jobs = [(n, POOL_ARM, "pool") for n in POOL175]')

# ---- 3. head 2 is absent on most F4 movies; do not KeyError on it -------
rep('''    seeds, feats = features_ext(name, dets, T, EP_OF[det_arm][name],
                                EP2_OF[det_arm][name])''',
    '''    # .get, not [], because edgeprob2_F4_ covers 24 movies and the F4 pool
    # needs 175. features_ext already handles ep2_path=None by filling head
    # 2's five columns with NaN; +BIDIR does not read them, and every arm
    # that does is cut from ARM_ORDER above.
    seeds, feats = features_ext(name, dets, T, EP_OF[det_arm][name],
                                EP2_OF[det_arm].get(name))''')

# ---- 4. the F4 pool must be COMPLETE before it may be used --------------
rep('print(f"D175 pool ({len(POOL175)}): {\' \'.join(POOL175)}", flush=True)',
    '''print(f"D175 pool ({len(POOL175)}): {' '.join(POOL175)}", flush=True)
print(f"\\n=== POOL_ARM = {POOL_ARM} ===", flush=True)
if POOL_ARM == "BASE":
    print("  The shipped configuration: the pool's rows come from BASE "
          "primaries (the xl three-branch mean) and the 24 evaluation movies "
          "are scored on F4. THIS IS THE MISMATCH. Its purpose here is to "
          "reproduce the recorded soft anchors (+BIDIR F4-6bba 0.9261, "
          "F4-44b6 0.8515) so the F4 arm is readable against it.", flush=True)
else:
    _need = [n for n in POOL175 if n not in FDET_F4]
    assert not _need, (
        f"POOL_ARM=F4 but {len(_need)} pool movies have no fdet_F4_ cache "
        f"({_need[:4]}) -- attach faustine649/biohub-fuse6bba and "
        f"faustine649/biohub-fuse44b6. A pool smaller than the BASE arm's "
        f"175 would make any delta pool size, not primaries.")
    _need = [n for n in POOL175 if n not in EDGEPROB_F4]
    assert not _need, (
        f"POOL_ARM=F4 but {len(_need)} pool movies have no edgeprob_F4_ "
        f"cache ({_need[:4]}) -- attach faustine649/biohub-ep6bba and "
        f"faustine649/biohub-ep44b6.")
    _h2 = sum(n in EDGEPROB2_F4 for n in POOL175)
    print(f"  All {len(POOL175)} pool movies carry F4 primaries and F4 head-1 "
          f"probabilities, so the F4 pool IS the BASE pool -- 175 names "
          f"either way. Head 2 on F4 covers {_h2} of them, which is why only "
          f"+BIDIR is trained.", flush=True)
    print("  This is the retrain the record has wanted since 2026-09-07 and "
          "could not run before it: on 09-03, when the shipped pkl was fitted, "
          "edgeprob_F4_ existed for 24 movies and none were in the pool.",
          flush=True)''')

# ---- 5. name the anchors where they are printed ------------------------
rep('''print("\\n=== RANKER LAYOUT (what a submission kernel must feed the pkl) ===",
      flush=True)''',
    '''print(f"\\n  ANCHOR (POOL_ARM=BASE only): +BIDIR must read F4-6bba 0.9261 and "
      f"F4-44b6 0.8515 in STAGE B below -- the record's soft anchors for this "
      f"harness. +BIDIR's fit does not depend on ARM_ORDER (fresh "
      f"default_rng(0) per arm; the permutation RNG touches no model), so "
      f"cutting the other three arms must not move them. For POOL_ARM=F4 they "
      f"move by construction and the move IS the measurement.", flush=True)
print("\\n=== RANKER LAYOUT (what a submission kernel must feed the pkl) ===",
      flush=True)''')

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for fname, pa in (("eprank3.py", "F4"), ("eprank3_ref.py", "BASE")):
    body = src.replace('POOL_ARM = "BASE"', f'POOL_ARM = "{pa}"', 1)
    assert body.count(f'POOL_ARM = "{pa}"') >= 1, pa
    open(os.path.abspath(os.path.join(base, fname)), "w",
         encoding="utf-8", newline="\n").write(body)
    print("written", fname, len(body), "POOL_ARM", pa)
