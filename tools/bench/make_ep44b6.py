"""edgeprob_cache.py -> ep44b6.py: the official head's probabilities over the
F4 primaries of the 59 44b6 train movies that biohub-edgeprob does not already
hold (fdet_F4_ from biohub-fuse44b6). No base pass -- all 199 are in
biohub-edgeprob already.

The 12 manifest-excluded 44b6 (edgeprob_cache's own SET_44B6, = fusecache's
EX_44B6) keep the caches in biohub-edgeprob, so nothing an anchor reads is
recomputed. 116 + 12 6bba + 59 + 12 44b6 = all 199 movies with F4 primaries
and F4 head probabilities, which is what a 44b6 division bench needs.
"""
src = open("edgeprob_cache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-03a"', 'SCRIPT_VERSION = "2026-09-07a-ep44b6"')
rep('names_f4 = list(EVAL_24)',
    '''names_f4 = [n for n in ALL_TRAIN
            if n.startswith("44b6_") and n not in set(SET_44B6)]
assert len(names_f4) == 59, len(names_f4)
assert not (set(names_f4) & set(EVAL_24)), "an evaluation dataset is in the build list"
print(f"44b6 F4 to build ({len(names_f4)}): {' '.join(names_f4)}", flush=True)''')
rep('assert len(CACHES_F4) >= 24, (', 'assert len(CACHES_F4) >= 59, (')
rep('jobs = [("base", n) for n in names] + [("F4", n) for n in names_f4]',
    'jobs = [("F4", n) for n in names_f4]          # the base pass is in biohub-edgeprob')
rep('assert len(jobs) == len(names) + len(names_f4)', 'assert len(jobs) == len(names_f4)')
rep('            and n_base == len(names) and n_f4 == len(names_f4))',
    '            and n_f4 == len(names_f4))')
open("ep44b6.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
