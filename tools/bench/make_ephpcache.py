"""ep6bba.py -> ephpcache.py: the edge head, re-run on the hpara primaries.

WHY THIS EXISTS, and it was not in the plan. `hparabench`'s F4HP arm died on

    44b6_0b24845f t=0: 40 of 990 harness pairs have no edgeprob row

`hparacache` holds the detection SET fixed -- same NMS survivors, same count,
same order -- so the cached `edgeprob_F4_<name>.npz` rows were expected to
stay validly joined by (t, i, j). They do not, and the reason is a step I did
not think about: the harness builds CANDIDATE PAIRS by gating on the distance
between detections (`GATE_UM = 10.0`). Moving every point by a median 0.43 um
moves some pairs across that 10 um line in both directions, so the pair set
is not the cached one. 40 of 990 on the first frame checked.

Two ways out. Gate the pairs on the REFINE coordinates and feed hpara only to
the features -- cheap, but it threads two coordinate systems through a
3,000-line script and adds a third "held fixed" approximation. Or re-run the
head on the hpara primaries: about 25 GPU minutes for 24 movies at ep6bba's
34-54 s each, and it REMOVES an approximation instead of adding one -- the
original design froze the head at its refine-based value and said so.

This is the second option.

THE CUT, four changes to ep6bba:
  * `CACHES_F4` discovers `fdet_F4hp_*` instead of `fdet_F4_`;
  * `names_f4` becomes the 24 evaluation movies, not the 116 pool movies;
  * the output label becomes `F4hp_<name>`, so the file is
    `edgeprob_F4hp_<name>.npz` and cannot be confused with the shipped one;
  * the count asserts move from 116 to 24.

Everything that computes a probability is untouched, which matters: the arm
must differ from the control in COORDINATES only, never in the head.

NO ANCHOR is possible -- by construction the outputs differ from the cached
ones. What IS checked: the per-frame primary counts `n_per_frame` must equal
the shipped `edgeprob_F4_<name>.npz`'s, because hparacache did not add or
remove a single detection. If those differ, the two arms are not scoring the
same cells and everything downstream is void.

Attach: ep6bba's sources, plus faustine649/biohub-hparacache.
"""
import os

src = open("ep6bba.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-06a-ep6bba"',
    'SCRIPT_VERSION = "2026-09-08a-ephpcache"')

rep('''CACHES_F4 = _walk_files(lambda fn: fn[len("fdet_F4_"):-len(".npz")]
                        if fn.startswith("fdet_F4_") and fn.endswith(".npz")''',
    '''CACHES_F4 = _walk_files(lambda fn: fn[len("fdet_F4hp_"):-len(".npz")]
                        if fn.startswith("fdet_F4hp_") and fn.endswith(".npz")''')

rep('''assert len(CACHES_F4) >= 116, ("fewer than 24 fdet_F4_*.npz under ''',
    '''assert len(CACHES_F4) >= 24, ("fewer than 24 fdet_F4hp_*.npz under ''')

rep('''names_f4 = [n for n in ALL_TRAIN
            if n.startswith("6bba_") and n not in set(HOLDOUT_6BBA)]
assert len(names_f4) == 116, len(names_f4)''',
    '''# the 24 EVALUATION movies -- the only ones hparacache re-cached. The
# 175-movie pool stays on refine coordinates and keeps its shipped
# edgeprob_F4_ rows; see make_hparabench.py for why.
names_f4 = [n for n in ALL_TRAIN if n in set(CACHES_F4)]
assert len(names_f4) == 24, (len(names_f4), sorted(names_f4)[:5])
assert sorted(names_f4) == sorted(list(HOLDOUT_6BBA) + list(SET_44B6)), (
    "the hpara caches are not the 12 + 12 evaluation movies: "
    f"{sorted(set(names_f4) ^ set(list(HOLDOUT_6BBA) + list(SET_44B6)))}")''')

rep('''    label = name if arm == "base" else f"F4_{name}"''',
    '''    label = name if arm == "base" else f"F4hp_{name}"''')

# ---- the one check that is possible: the primary counts must not have moved
rep('''        n_per_frame=fwd.n)''',
    '''        n_per_frame=fwd.n)
    _ship = SHIPPED_EP.get(name)
    if _ship is None:
        NPF_BAD.append(f"{name}(no shipped cache)")
        print(f"  !! {name}: no shipped edgeprob_F4_ to compare n_per_frame "
              f"against -- the only available check on this run is off",
              flush=True)
    else:
        _z = np.load(_ship)
        _old = np.asarray(_z["n_per_frame"], np.int64)
        _new = np.asarray(fwd.n, np.int64)
        if _old.shape != _new.shape or not np.array_equal(_old, _new):
            NPF_BAD.append(name)
            print(f"  !! {name}: n_per_frame DIFFERS from the shipped F4 "
                  f"cache ({_old.shape} vs {_new.shape}) -- hparacache was "
                  f"supposed to keep every detection, so the two arms are "
                  f"NOT scoring the same cells", flush=True)
        _z.close()''')

rep('''jobs = [("F4", n) for n in names_f4]          # the base pass is in biohub-edgeprob''',
    '''SHIPPED_EP = _walk_files(lambda fn: fn[len("edgeprob_F4_"):-len(".npz")]
                         if fn.startswith("edgeprob_F4_")
                         and fn.endswith(".npz") else None)
print(f"shipped edgeprob_F4_ caches for the n_per_frame check: "
      f"{len(SHIPPED_EP)}", flush=True)
NPF_BAD = []

jobs = [("F4", n) for n in names_f4]          # the base pass is in biohub-edgeprob''')

rep('''print(f"jobs: {len(names)} base + {len(names_f4)} F4 = {len(jobs)}",
      flush=True)''',
    '''print(f"jobs: {len(names_f4)} F4hp datasets (the 12 + 12 evaluation "
      f"movies)", flush=True)''')

# ---- the tail: report the one check this run can make
rep('''if complete:
    print(f"EDGEPROB OK ({n_base} base + {n_f4} F4 of {len(jobs)})",
          flush=True)''',
    '''print(f"n_per_frame check: {len(names_f4) - len(NPF_BAD)} of "
      f"{len(names_f4)} datasets reproduce the shipped edgeprob_F4_ counts"
      + (f" -- MISMATCHES: {NPF_BAD}" if NPF_BAD else " exactly"), flush=True)
if complete and not NPF_BAD:
    print(f"EPHPCACHE OK ({n_f4} F4hp of {len(jobs)})",
          flush=True)''')
# The label is `F4hp_`, so the file is `edgeprob_F4hp_<name>.npz`, which does
# NOT start with `edgeprob_F4_`. Left alone, n_f4 comes out 0, all 24 outputs
# are miscounted as base, and `complete` is False -- so the run of
# 2026-09-08, which cached all 24 and passed n_per_frame 24/24 exactly,
# signed off with
#
#     24 cached (24 base + 0 F4), 20.4 MB, 912s
#     EPHPCACHE INCOMPLETE (24 base + 0 F4 of 24; 0 failed, 0 not attempted)
#
# That final line is a FALSE NEGATIVE and the cached outputs are good; this
# fix is for the next run, not a reason to repeat that one. Same bug was
# found and fixed in make_ep2hpcache.py.
rep('''n_f4 = sum(1 for p in written if p.name.startswith("edgeprob_F4_"))''',
    '''n_f4 = sum(1 for p in written
           if p.name.startswith("edgeprob_F4hp_"))''')

rep('EDGEPROB INCOMPLETE', 'EPHPCACHE INCOMPLETE')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ephpcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
