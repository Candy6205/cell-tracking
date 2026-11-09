"""ep2hpcache.py -> ep2hrcache.py: the second edge head on hrefine's re-placed primaries (fdet_F4hr_).
The twin of make_ephrcache.py; see there. Output: edgeprob2_F4hr_<name>.npz. The n_per_frame check against
the shipped edgeprob2_F4_ cache stays ENFORCED (hrefine moves primaries, never adds or removes one).
Attach: ep2hpcache's sources with biohub-hparacache replaced by biohub-hrefine. GPU T4.
"""
import os

src = open("ep2hpcache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-ep2hpcache"', 'SCRIPT_VERSION = "2026-09-23a-ep2hrcache"')
rep("fdet_F4hp_", "fdet_F4hr_", n=src.count("fdet_F4hp_"))
rep('label = name if arm == "base" else f"F4hp_{name}"', 'label = name if arm == "base" else f"F4hr_{name}"')
rep("edgeprob2_F4hp_", "edgeprob2_F4hr_", n=src.count("edgeprob2_F4hp_"))
rep('''    print(f"EP2HPCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "''',
    '''    print(f"EP2HRCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "''')
rep("EP2HPCACHE INCOMPLETE", "EP2HRCACHE INCOMPLETE")
assert "F4hp_" not in src.replace("fdet_F4hp", "").replace("edgeprob2_F4hp", ""), "a stale F4hp_ label survives"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ep2hrcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written ep2hrcache.py", len(src))
