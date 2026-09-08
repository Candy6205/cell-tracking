"""fuse6bba.py -> fuse44b6.py: the F4 blend's detection caches for the 59
44b6 train movies that do NOT already have one. The 12 manifest-excluded
44b6 (fusecache's EX_44B6, = dctrain's held-out 12, = edgeprob_cache's
SET_44B6) keep the fdet_F4_ files in biohub-fusecache, so every anchor that
reads them keeps reading the same nodes. F4 arm only, no Zebrahub pass.

Same shape as fuse6bba: 116 of 128 6bba were built there, 12 kept; here 59
of 71 44b6 are built, 12 kept. After this the whole 199-movie train set has
F4 primaries, and the fork gate can be trained on 44b6 forks for the first
time.

NOTE on the DeepCenter branch: these 59 are INSIDE dctrain's 175-movie pool,
so F4's fourth heatmap is in-sample on them -- exactly as it is on
fuse6bba's 116. The bench is symmetric between the two domains; it is not
an out-of-sample detection measurement, and no detection claim may be made
from it.
"""
src = open("fuse6bba.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-06a-fuse6bba"',
    'SCRIPT_VERSION = "2026-09-07a-fuse44b6"')

rep('''NAMES_6BBA = [n for n in names_all
              if n.startswith("6bba_") and n not in set(HOLDOUT_6BBA)]
assert len(NAMES_6BBA) == 116, len(NAMES_6BBA)''',
    '''NAMES_44B6 = [n for n in names_all
              if n.startswith("44b6_") and n not in set(EX_44B6)]
assert len(NAMES_44B6) == 59, len(NAMES_44B6)
assert all(n in _SEEN for n in NAMES_44B6), \\
    "a 44b6 movie outside dctrain's pool slipped into the build list"
print(f"44b6 to build ({len(NAMES_44B6)}): {' '.join(NAMES_44B6)}", flush=True)
print(f"44b6 already cached in biohub-fusecache ({len(EX_44B6)}): "
      f"{' '.join(EX_44B6)}", flush=True)''')

rep('''      f"{len(NAMES_6BBA)} 6bba datasets (the 12-movie holdout excluded), "''',
    '''      f"{len(NAMES_44B6)} 44b6 datasets (the 12 manifest-excluded ones "
      f"already cached in biohub-fusecache), "''')
rep('for nm in NAMES_6BBA:', 'for nm in NAMES_44B6:')
rep('    print(f"  {done}/{len(NAMES_6BBA)} {nm}: T={T}, peaks>=0.15 "',
    '    print(f"  {done}/{len(NAMES_44B6)} {nm}: T={T}, peaks>=0.15 "')
rep('print(f"\\nFUSE6BBA DONE', 'print(f"\\nFUSE44B6 DONE')

open("fuse44b6.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
