"""edgeprob_cache.py -> ep6bba.py: the official head's probabilities over the
F4 primaries of the 116 non-holdout 6bba train movies (fdet_F4_ from
biohub-fuse6bba). No base pass (all 199 are in biohub-edgeprob already)."""
src = open("edgeprob_cache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-03a"', 'SCRIPT_VERSION = "2026-09-06a-ep6bba"')
rep('names_f4 = list(EVAL_24)',
    '''names_f4 = [n for n in ALL_TRAIN
            if n.startswith("6bba_") and n not in set(HOLDOUT_6BBA)]
assert len(names_f4) == 116, len(names_f4)''')
rep('assert len(CACHES_F4) >= 24, (', 'assert len(CACHES_F4) >= 116, (')
rep('jobs = [("base", n) for n in names] + [("F4", n) for n in names_f4]',
    'jobs = [("F4", n) for n in names_f4]          # the base pass is in biohub-edgeprob')
rep('assert len(jobs) == len(names) + len(names_f4)', 'assert len(jobs) == len(names_f4)')
rep('            and n_base == len(names) and n_f4 == len(names_f4))',
    '            and n_f4 == len(names_f4))')
open("ep6bba.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
