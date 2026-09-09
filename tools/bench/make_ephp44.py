"""ephpcache.py -> ephp44.py: the edge head over the 59 NEW hpara primaries.

`hpara44` (GPU, ~90 min) gives the last 59 44b6 movies their hpara detections
so hpara can be scored on all 71 instead of a 12-movie draw carrying 2 GT
divisions. Those detections need the head run over them, exactly as `ep44b6`
followed `fuse44b6` on the refine side.

WHY THE HEAD MUST BE RE-RUN AT ALL, from ephpcache's own docstring: the
harness builds candidate pairs by gating on the DISTANCE between detections
(`GATE_UM = 10.0`). hpara holds the detection set fixed -- same NMS
survivors, same count, same order -- but moves every point by a median
0.43 um, and that walks some pairs across the 10 um line in both directions.
So the cached `edgeprob_F4hp_` rows must be produced on the hpara
coordinates; joining the refine ones by (t, i, j) silently loses rows
(`44b6_0b24845f t=0: 40 of 990 harness pairs have no edgeprob row`).

THE ONE CHANGE, and it makes the kernel idempotent. `ephpcache` selects
`names_f4 = every movie that has an fdet_F4hp_ cache`. With both
`biohub-hparacache` (140) and `biohub-hpara44` (59) attached that is 199, and
re-running the 140 already cached would burn about two GPU hours for files
that exist. So the selection becomes **has an `fdet_F4hp_` cache AND has no
`edgeprob_F4hp_` cache yet** -- attach `biohub-ephpcache` alongside and the
kernel does exactly the work that is missing, whatever that turns out to be.
The 6bba completeness assert is replaced by one that says what this run is:
59 movies, all 44b6.

NO ANCHOR IS POSSIBLE -- by construction these outputs differ from the refine
ones. The check that IS made is inherited and is the important one: each
movie's per-frame primary counts must equal the shipped
`edgeprob_F4_<name>.npz`'s, because hpara44 added and removed no detection.
For these 59 that comparison cache comes from `biohub-ep44b6`.

Attach: ephpcache's sources, PLUS faustine649/biohub-hpara44 (the new
primaries), faustine649/biohub-ep44b6 (the n_per_frame comparison), and
faustine649/biohub-ephpcache (so the 140 already done are skipped). GPU.
"""
import os

src = open("ephpcache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-ephpcache"',
    'SCRIPT_VERSION = "2026-09-09a-ephp44"')

rep('''names_f4 = [n for n in ALL_TRAIN if n in set(CACHES_F4)]
assert len(names_f4) == len(CACHES_F4), (len(names_f4), len(CACHES_F4))
assert len(names_f4) >= 128, (
    f"only {len(names_f4)} hpara caches -- the bench scores 128 6bba "
    f"movies and every one of them needs an edgeprob_F4hp_ row")
_six = [n for n in ALL_TRAIN if n.startswith("6bba_")]
_gap = [n for n in _six if n not in set(names_f4)]
assert not _gap, (
    f"{len(_gap)} of the 128 6bba movies have no fdet_F4hp_ cache: "
    f"{_gap[:5]}")''',
    '''# ONLY WHAT IS MISSING. hparacache built 140 fdet_F4hp_ caches and
# ephpcache ran the head over all 140; hpara44 adds 59 more. Selecting on
# "has primaries" alone would redo the 140 and burn ~2 GPU hours writing
# files that already exist, so select on "has primaries and has NO head cache
# yet". Attach biohub-ephpcache and this kernel does exactly the remainder.
DONE_HP = _walk_files(lambda fn: fn[len("edgeprob_F4hp_"):-len(".npz")]
                      if fn.startswith("edgeprob_F4hp_")
                      and fn.endswith(".npz") else None)
print(f"edgeprob_F4hp_ caches already mounted: {len(DONE_HP)}", flush=True)
names_f4 = [n for n in ALL_TRAIN
            if n in set(CACHES_F4) and n not in set(DONE_HP)]
print(f"  fdet_F4hp_ primaries: {len(CACHES_F4)}; already headed: "
      f"{len(DONE_HP)}; to run here: {len(names_f4)}", flush=True)
assert names_f4, (
    "every movie with hpara primaries already has an edgeprob_F4hp_ cache -- "
    "either hpara44 is not attached, or this kernel has nothing to do")
assert len(names_f4) == 59, (
    f"expected the 59 44b6 movies hpara44 built, got {len(names_f4)}: "
    f"{names_f4[:5]} -- attach faustine649/biohub-hpara44 AND "
    f"faustine649/biohub-ephpcache, or the selection is not the remainder")
_not44 = [n for n in names_f4 if not n.startswith("44b6_")]
assert not _not44, (
    f"{len(_not44)} non-44b6 movies in the remainder ({_not44[:5]}) -- "
    f"ephpcache was supposed to have covered every 6bba movie already")
print(f"  the remainder is 59 44b6 movies, so after this kernel all 71 44b6 "
      f"carry hpara primaries AND hpara head probabilities, and hpara can be "
      f"scored on 71 instead of the 12-movie draw that holds 2 GT divisions.",
      flush=True)''')

rep('''print(f"jobs: {len(names_f4)} F4hp datasets (every movie hparacache "
      f"produced; the bench scores the 128 6bba of them)", flush=True)''',
    '''print(f"jobs: {len(names_f4)} F4hp datasets -- the movies that have hpara "
      f"primaries and no head cache yet", flush=True)''')

rep('EPHPCACHE OK', 'EPHP44 OK')
rep('EPHPCACHE INCOMPLETE', 'EPHP44 INCOMPLETE')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ephp44.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
