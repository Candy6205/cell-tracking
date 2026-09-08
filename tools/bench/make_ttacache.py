"""ttaprobe.py -> ttacache.py: the tta4 edge probabilities for ALL 140 movies.

WHY THIS EXISTS, and it is a correction rather than a new idea. `ttascore`
measured that D4-averaging the U-Net features makes the edge head's own AUC
worse (−0.004, 12/12 movies). I turned that into "the ~20 GPU hours are
refused", which the user took apart in two moves:

  *"边概率头变了和它不值之间的关联是？"* -- there was none. The ranker never
  ran, the ILP never ran, nothing was scored, and working-agreement.md
  records that offline edge AUC anti-predicted this pipeline three times.

  *"为了不动GPU而降级测试更是蠢得离谱"* -- and that is exactly what I then
  did. `ttabench` scores 128 movies of which only TWELVE change, because
  twelve is all `ttaprobe` cached. The dilution is tenfold, and it is
  self-inflicted.

THE ARITHMETIC I SHOULD HAVE DONE FIRST. `ttaprobe`'s own log: tta4 costs
99-119 s per 100-frame movie. 140 movies is **about 4.1 GPU hours**. The
quota at the time of writing has 21.3 h left. The "20 GPU hours" I refused
was an estimate for THREE edge-probability datasets at tta8 -- a much bigger
scope than the question needs. Answering "does edge-feature TTA cost score
on 6bba" needs the FIRST head at tta4 over the movies the bench scores, and
that is this.

WHAT IT IS. `ttaprobe` with the readout cut off and the job list widened:

  * `VIEWS = 4` fixed (`ttascore`: tta8 was indistinguishable from tta4, so
    the extra 3 GPU hours of the 8-view arm buy nothing measurable);
  * 140 movies -- the 128 6bba the bench scores plus the 12 44b6 of the eval
    draw -- matching `hparacache` v3's scope exactly, so the two arms of
    every bench cover the same set;
  * the labelling and AUC tail deleted. `ttascore` already did that
    measurement, and it is the half that had the bug.

OUTPUT: `edgeprob_tta4_<name>.npz`, the same keys as `edgeprob_F4_<name>.npz`
(t, i, j, p_fwd, p_rev), so `ttabench` reads it by changing one prefix.

THE ANCHOR IS FREE AND IT IS THE ONE THAT MATTERS. The (t, i, j) rows are
produced by the same gate over the same `fdet_F4_` primaries, so they must
match `edgeprob_F4_`'s EXACTLY, movie by movie. Detections are untouched
(the wrapper leaves `det_logits` at view 0), so anything else is a bug. The
run checks every movie it caches against the shipped cache and refuses to
report OK if a single row set differs.

Attach: ttaprobe's sources (the support pack, cell-tracking, edge-ranker,
cache-train, poolsize, fusecache, fuse6bba) PLUS biohub-edgeprob for the
row-set anchor. GPU, ~4.1 h.
"""
import os

src = open("ttaprobe.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-ttaprobe"'
    if 'SCRIPT_VERSION = "2026-09-09a-ttaprobe"' in src
    else 'SCRIPT_VERSION = "2026-09-08a-ttaprobe"',
    'SCRIPT_VERSION = "2026-09-09a-ttacache"')

# ---- one arm, 140 movies ------------------------------------------------
rep('''ARM_VIEWS = [("noTTA", 1), ("tta4", 4), ("tta8", 8)]''',
    '''# ONE arm. ttascore measured tta8 at −0.00408 AUC against tta4's −0.00417 --
# indistinguishable -- so the 8-view arm's extra ~3 GPU hours buy nothing
# that could be read.
ARM_VIEWS = [("tta4", 4)]''')

rep('''jobs = [(f"F4|{a}|{v}", n) for a, v in ARM_VIEWS for n in HOLDOUT_6BBA]
assert len(jobs) == 36, len(jobs)
assert len(jobs) == len(ARM_VIEWS) * len(HOLDOUT_6BBA) == 36''',
    '''# The 128 6bba the bench SCORES, plus the 12 44b6 of the eval draw:
# hparacache v3's scope, so both arms of every bench cover the same movies.
# ttaprobe ran 12 because it was answering a yes/no about the features; this
# is answering it about the SCORE, and a 12-of-128 arm dilutes tenfold.
CACHE_NAMES = sorted(n for n in CACHES_F4
                     if n.startswith("6bba_") or n in set(SET_44B6))
_n6 = sum(1 for n in CACHE_NAMES if n.startswith("6bba_"))
print(f"cache scope: {len(CACHE_NAMES)} movies ({_n6} 6bba + "
      f"{len(CACHE_NAMES) - _n6} 44b6)", flush=True)
assert _n6 == 128, (f"{_n6} 6bba movies carry an fdet_F4_ cache, want 128 -- "
                    f"attach biohub-fuse6bba as well as biohub-fusecache")
jobs = [(f"F4|{a}|{v}", n) for a, v in ARM_VIEWS for n in CACHE_NAMES]
assert len(jobs) == len(CACHE_NAMES) == 140, (len(jobs), len(CACHE_NAMES))''')

rep('''print(f"jobs: {len(names)} base + {len(names_f4)} F4 = {len(jobs)}",
      flush=True)''',
    '''print(f"jobs: {len(jobs)} = {len(ARM_VIEWS)} arm x {len(CACHE_NAMES)} "
      f"movies. ttaprobe's tta4 cost 99-119 s per movie, so expect ~4.1 h.",
      flush=True)''')

# ---- cut the labelling / AUC tail; ttascore already did it, correctly ---
_CUT = "def _label_pairs(name, z):"
assert src.count(_CUT) == 1
src = src[:src.index(_CUT)]

src += r'''# ---- THE ANCHOR: the row sets must be the shipped ones ------------------
# Every cached (t, i, j) comes from the SAME GATE_UM gate over the SAME
# fdet_F4_ primaries as edgeprob_F4_, and the wrapper leaves det_logits at
# view 0 on purpose, so the detections are untouched. The row sets must
# therefore be identical, movie by movie. If they are not, the arms of any
# bench built on this are not scoring the same pairs and nothing downstream
# compares anything.
print("\n=== ANCHOR: cached (t, i, j) row sets against edgeprob_F4_ ===",
      flush=True)
_EPF4 = _walk_files(lambda fn: (fn[len("edgeprob_F4_"):-len(".npz")]
                                if fn.startswith("edgeprob_F4_")
                                and fn.endswith(".npz") else None))
print(f"  edgeprob_F4_ caches available for comparison: {len(_EPF4)}",
      flush=True)
_ok = _bad = _nocheck = 0
_first = []
for _n in CACHE_NAMES:
    _p = OUT / f"edgeprob_tta4_{_n}.npz"
    if not _p.exists():
        continue
    if _n not in _EPF4:
        _nocheck += 1
        continue
    _a, _b = np.load(_p), np.load(_EPF4[_n])
    if all(np.array_equal(np.asarray(_a[c]), np.asarray(_b[c]))
           for c in ("t", "i", "j")):
        _ok += 1
    else:
        _bad += 1
        if len(_first) < 3:
            _first.append((_n, len(_a["t"]), len(_b["t"])))
    _a.close()
    _b.close()
print(f"  {_ok} movies reproduce the shipped row set EXACTLY, {_bad} differ, "
      f"{_nocheck} had no edgeprob_F4_ to check against", flush=True)
if _bad:
    print(f"  MISMATCH on {_bad}: {_first} -- the tta4 caches are NOT "
          f"row-compatible with the shipped ones and ttabench cannot pair "
          f"them", flush=True)

_have = sum(1 for _n in CACHE_NAMES
            if (OUT / f"edgeprob_tta4_{_n}.npz").exists())
print(f"\n{_have} of {len(CACHE_NAMES)} tta4 caches written", flush=True)
if _have == len(CACHE_NAMES) and _bad == 0:
    print("TTACACHE OK", flush=True)
else:
    print(f"TTACACHE INCOMPLETE ({_have} of {len(CACHE_NAMES)} written, "
          f"{_bad} row-set mismatches)", flush=True)
'''

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
out = os.path.abspath(os.path.join(base, "ttacache.py"))
open(out, "w", encoding="utf-8", newline="\n").write(src)
print("written ttacache.py", len(src))
