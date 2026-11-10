"""ep2hpcache.py -> ep2lncache.py: the second edge head on lnrescue's rescue set (children + parent hypotheses) (fdet_F4ln_). The twin of make_eplncache.py; see there.
Output: edgeprob2_F4ln_<name>.npz. The n_per_frame check against the
shipped edgeprob2_F4_ cache is recorded, not enforced (the counts differ by
design). Attach: ep2hpcache's sources with biohub-hparacache replaced by
biohub-lnrescue. GPU T4.
"""
import os

src = open("ep2hpcache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


def cut(start, end, new):
    global src
    assert src.count(start) == 1, (src.count(start), start[:80])
    i = src.index(start)
    j = src.index(end, i)
    src = src[:i] + new + src[j:]


rep('SCRIPT_VERSION = "2026-09-08a-ep2hpcache"', 'SCRIPT_VERSION = "2026-09-17a-ep2lncache"')
rep("fdet_F4hp_", "fdet_F4ln_", n=src.count("fdet_F4hp_"))
rep('label = name if arm == "base" else f"F4hp_{name}"', 'label = name if arm == "base" else f"F4ln_{name}"')
rep("edgeprob2_F4hp_", "edgeprob2_F4ln_", n=src.count("edgeprob2_F4hp_"))
cut("        if _old.shape != _new.shape or not np.array_equal(_old, _new):",
    "        _z.close()",
    "        if _old.shape != _new.shape or not np.array_equal(_old, _new):\n"
    "            NPF_BAD.append(name)      # expected: hypcache appended parents in this movie\n")
cut('    print(f"n_per_frame check:', "    print(f\"EP2HPCACHE",
    '    print(f"n_per_frame against the shipped edgeprob2_F4_ counts: {len(NPF_BAD)} of "\n'
    '          f"{len(names_f4) - len(NPF_NOCHK)} checkable movies DIFFER (expected: hypcache appends "\n'
    '          f"primaries); {len(NPF_NOCHK)} not checkable", flush=True)\n')
rep('''    print(f"EP2HPCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "''',
    '''    print(f"EP2LNCACHE OK "''')
rep("EP2HPCACHE INCOMPLETE", "EP2LNCACHE INCOMPLETE")
assert "F4hp_" not in src.replace("fdet_F4hp", "").replace("edgeprob2_F4hp", ""), "a stale F4hp_ label survives"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ep2lncache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written ep2lncache.py", len(src))
