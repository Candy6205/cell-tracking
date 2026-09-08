"""fusecache.py -> fuse6bba.py: the F4 blend's detection caches for the 116
6bba train movies that are not in the 12-movie holdout (those 12 already
have fdet_F4_ files in biohub-fusecache and must not be recomputed, so the
anchors keep reading the same nodes). F4 arm only, no Zebrahub pass."""
src = open("fusecache.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


def line_start(marker, start=0):
    i = src.index(marker, start)
    return src.rfind(NL, 0, i) + 1


rep('SCRIPT_VERSION = "2026-08-29a-fusecache"', 'SCRIPT_VERSION = "2026-09-06a-fuse6bba"')
rep('''    ("W1101",   (1.0, 1.0, 0.5, 1.0)),
    ("DCheavy", (1.0, 1.0, 0.5, 2.0)),
]''', ''']''')

# no zebrahub crops
start = line_start('CROPS = None')
end = line_start('OUT = pathlib.Path("/kaggle/working")')
src = src[:start] + '''NAMES_6BBA = [n for n in names_all
              if n.startswith("6bba_") and n not in set(HOLDOUT_6BBA)]
assert len(NAMES_6BBA) == 116, len(NAMES_6BBA)
''' + src[end:]

rep('''      f"{len(HOLDOUT_6BBA) + len(EX_44B6) + len(ZH_FILES)} datasets, "''',
    '''      f"{len(NAMES_6BBA)} 6bba datasets (the 12-movie holdout excluded), "''')
rep('for nm in HOLDOUT_6BBA + EX_44B6:', 'for nm in NAMES_6BBA:')
rep('    print(f"  {done}/32 {nm}: T={T}, peaks>=0.15 "',
    '    print(f"  {done}/{len(NAMES_6BBA)} {nm}: T={T}, peaks>=0.15 "')

start = line_start('for f in ZH_FILES:')
end = line_start('print(f"\\nFUSECACHE DONE')
src = src[:start] + src[end:]
rep('print(f"\\nFUSECACHE DONE', 'print(f"\\nFUSE6BBA DONE')
open("fuse6bba.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
