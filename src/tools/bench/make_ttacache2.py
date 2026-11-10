"""ttacache.py -> ttacache2.py: tta4 for the 59 44b6 pool movies ttacache skipped.

WHY, and it is the working-agreement's new rule applied to its own newest
case. `ttacache` scoped itself to "the 128 6bba the bench SCORES plus the 12
44b6 of the eval draw" = 140. That was right for the question it was written
for -- what does TTA do to the SCORE -- but it is wrong for the question the
user then asked for: *"哦，排序器水土不服，那先重训也行"*.

A RETRAINED test needs tta4 in the ranker's TRAINING rows, and `rkpool` at
`RANK_TRAIN="F4"` trains from `LAB4`, built with `EP_OF["F4"]` over **all 199**
movies. With ttacache's 140 that would be 140 tta4 rows against 59 noTTA ones
-- a mixed training set, which is a smaller version of exactly the confound
that made `ttabench`'s −0.0051 unreadable. 59 movies at ttaprobe's measured
99-119 s is about **1.8 GPU hours**, which is the whole cost of removing it.

    working-agreement.md, 2026-09-09: "the default is the whole population
    that any bench in this repo scores -- 199 train movies for the
    competition sets"

THE SELECTION IS THE REMAINDER, as in `ephp44`: every movie with an
`fdet_F4_` cache that has no `edgeprob_tta4_` cache yet. Attach
`biohub-ttacache` and this kernel does exactly the work that is missing,
and asserts the remainder is the 59 44b6 pool movies rather than trusting it.

⚠ These 59 are inside dctrain's 175-movie pool, so F4's fourth heatmap (ep05
DeepCenter) is in-sample on them -- the same caveat `fuse44b6` and `ep44b6`
carry. No detection claim may be made from these caches. They exist so a
tta4-trained ranker can be fitted on the same 199 movies the noTTA one is.

The inherited row-set anchor still applies and is the important one: every
tta4 cache must hold the SAME (t, i, j) rows as its `edgeprob_F4_` twin,
because `ttaprobe` holds `det_logits` at view 0 on purpose so the detections
cannot move. Only the probability values may differ.

Attach: ttacache's sources PLUS faustine649/biohub-ttacache (so the 140
already done are skipped), faustine649/biohub-fuse44b6 and
faustine649/biohub-ep44b6. GPU, ~1.8 h.
"""
import os

src = open("ttacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-ttacache"',
    'SCRIPT_VERSION = "2026-09-09b-ttacache2"')

rep('''CACHE_NAMES = sorted(n for n in CACHES_F4
                     if n.startswith("6bba_") or n in set(SET_44B6))
_n6 = sum(1 for n in CACHE_NAMES if n.startswith("6bba_"))
print(f"cache scope: {len(CACHE_NAMES)} movies ({_n6} 6bba + "
      f"{len(CACHE_NAMES) - _n6} 44b6)", flush=True)
assert _n6 == 128, (f"{_n6} 6bba movies carry an fdet_F4_ cache, want 128 -- "
                    f"attach biohub-fuse6bba as well as biohub-fusecache")
jobs = [(f"F4|{a}|{v}", n) for a, v in ARM_VIEWS for n in CACHE_NAMES]
assert len(jobs) == len(CACHE_NAMES) == 140, (len(jobs), len(CACHE_NAMES))''',
    '''# THE REMAINDER, not a hardcoded list. ttacache did 140 (the 128 6bba the
# bench scores + the 12 44b6 of the eval draw); a RETRAINED tta4 arm needs the
# ranker's whole 199-movie feature pool, so the 59 44b6 pool movies are what
# is left. Selecting on "has F4 primaries and has no tta4 cache yet" makes
# this kernel idempotent and self-describing.
DONE_TTA = _walk_files(lambda fn: fn[len("edgeprob_tta4_"):-len(".npz")]
                       if fn.startswith("edgeprob_tta4_")
                       and fn.endswith(".npz") else None)
print(f"edgeprob_tta4_ caches already mounted: {len(DONE_TTA)}", flush=True)
# AND inside ALL_TRAIN. v1 selected on "has F4 primaries" alone and its own
# assert caught the consequence: **207 fdet_F4_ caches are mounted, not 199**
# -- eight of them are not train movies at all -- so the remainder came out
# 67 instead of 59. Caching eight extra movies would have cost ~15 GPU
# minutes and, far worse, silently widened the ranker's feature pool to a
# different population than the noTTA arm's. The eight are NAMED below
# rather than dropped quietly.
_extra = sorted(n for n in CACHES_F4 if n not in set(ALL_TRAIN))
if _extra:
    print(f"  {len(_extra)} fdet_F4_ caches are NOT train movies and are "
          f"excluded: {' '.join(_extra)}", flush=True)
CACHE_NAMES = sorted(n for n in CACHES_F4
                     if n in set(ALL_TRAIN) and n not in set(DONE_TTA))
_n6 = sum(1 for n in CACHE_NAMES if n.startswith("6bba_"))
print(f"cache scope: {len(CACHES_F4)} fdet_F4_ caches mounted, "
      f"{len(ALL_TRAIN)} of them train movies, {len(DONE_TTA)} already "
      f"cached, {len(CACHE_NAMES)} to run "
      f"({_n6} 6bba + {len(CACHE_NAMES) - _n6} 44b6)", flush=True)
assert CACHE_NAMES, (
    "every movie with F4 primaries already has a tta4 cache -- either "
    "biohub-ttacache is not attached, or this kernel has nothing to do")
assert _n6 == 0, (
    f"{_n6} 6bba movies are in the remainder ({[n for n in CACHE_NAMES if n.startswith('6bba_')][:5]})"
    f" -- ttacache was supposed to have covered all 128 of them already")
assert len(CACHE_NAMES) == 59, (
    f"expected the 59 44b6 pool movies, got {len(CACHE_NAMES)} -- attach "
    f"faustine649/biohub-fuse44b6 (their F4 primaries) AND "
    f"faustine649/biohub-ttacache (so the 140 done are skipped)")
print(f"  after this kernel all 199 train movies carry a tta4 edge cache, so "
      f"rkpool at RANK_TRAIN=F4 can be TRAINED on tta4 rows instead of being "
      f"handed them at solve time -- which is the confound that made "
      f"ttabench's -0.0051 unreadable.", flush=True)
jobs = [(f"F4|{a}|{v}", n) for a, v in ARM_VIEWS for n in CACHE_NAMES]
assert len(jobs) == len(CACHE_NAMES) == 59, (len(jobs), len(CACHE_NAMES))''')

rep('''      f"movies. ttaprobe's tta4 cost 99-119 s per movie, so expect ~4.1 h.",''',
    '''      f"movies. ttaprobe's tta4 cost 99-119 s per movie and 44b6's F4 node "
      f"sets run about 1.8x 6bba's, so expect roughly 1.8-2.5 h.",''')

rep('print("TTACACHE OK"', 'print("TTACACHE2 OK"')
rep('TTACACHE INCOMPLETE', 'TTACACHE2 INCOMPLETE')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ttacache2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
