"""ep2hpcache.py -> ep2hypcache.py: the second edge head on hypcache's union set (children + parent hypotheses) (fdet_F4hyp_). The twin of make_ephypcache.py; see there.
Output: edgeprob2_F4hyp_<name>.npz. The n_per_frame check against the
shipped edgeprob2_F4_ cache is recorded, not enforced (the counts differ by
design). Attach: ep2hpcache's sources with biohub-hparacache replaced by
biohub-hypcache. GPU T4.
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


rep('SCRIPT_VERSION = "2026-09-08a-ep2hpcache"', 'SCRIPT_VERSION = "2026-09-14a-ep2hypcache"')
rep("fdet_F4hp_", "fdet_F4hyp_", n=src.count("fdet_F4hp_"))
rep('label = name if arm == "base" else f"F4hp_{name}"', 'label = name if arm == "base" else f"F4hyp_{name}"')
rep("edgeprob2_F4hp_", "edgeprob2_F4hyp_", n=src.count("edgeprob2_F4hp_"))
cut("        if _old.shape != _new.shape or not np.array_equal(_old, _new):",
    "        _z.close()",
    "        if _old.shape != _new.shape or not np.array_equal(_old, _new):\n"
    "            NPF_BAD.append(name)      # expected: hypcache appended parents in this movie\n")
cut('    print(f"n_per_frame check:', "    print(f\"EP2HPCACHE",
    '    print(f"n_per_frame against the shipped edgeprob2_F4_ counts: {len(NPF_BAD)} of "\n'
    '          f"{len(names_f4) - len(NPF_NOCHK)} checkable movies DIFFER (expected: hypcache appends "\n'
    '          f"primaries); {len(NPF_NOCHK)} not checkable", flush=True)\n')
rep('''    print(f"EP2HPCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "''',
    '''    print(f"EP2HYPCACHE OK "''')
rep("EP2HPCACHE INCOMPLETE", "EP2HYPCACHE INCOMPLETE")
assert "F4hp_" not in src.replace("fdet_F4hp", "").replace("edgeprob2_F4hp", ""), "a stale F4hp_ label survives"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ep2hypcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written ep2hypcache.py", len(src))
