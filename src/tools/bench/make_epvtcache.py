"""ephpcache.py -> epvtcache.py: the first edge head, re-run on hypcache's
union set (fdet_F4vt_).

Same cut as make_ephpcache.py made from ep6bba.py, with three differences:
  * the detection cache prefix is `fdet_F4vt_` (biohub-mapvote) and the
    output label `F4vt_<name>`, so the file is `edgeprob_F4vt_<name>.npz`;
  * the n_per_frame cross-check against the shipped `edgeprob_F4_` cache is
    RECORDED, NOT ENFORCED: mapvote appends vote candidates, so the per-frame counts
    differ by design (the number of differing movies is printed);
  * nothing else -- the head, its weights and its crops are untouched, so the
    arm differs from F4HP in the detection set only.
Attach: ephpcache's sources with biohub-hparacache replaced by biohub-mapvote.
GPU T4, ~1.5-2 h (140 movies).
"""
import os

src = open("ephpcache.py", encoding="utf-8").read()


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


rep('SCRIPT_VERSION = "2026-09-08a-ephpcache"', 'SCRIPT_VERSION = "2026-09-19a-epvtcache"')
rep("fdet_F4hp_", "fdet_F4vt_", n=src.count("fdet_F4hp_"))
rep('label = name if arm == "base" else f"F4hp_{name}"', 'label = name if arm == "base" else f"F4vt_{name}"')
rep("edgeprob_F4hp_", "edgeprob_F4vt_", n=src.count("edgeprob_F4hp_"))
# the count check: record, do not fail, do not shout
cut("        if _old.shape != _new.shape or not np.array_equal(_old, _new):",
    "        _z.close()",
    "        if _old.shape != _new.shape or not np.array_equal(_old, _new):\n"
    "            NPF_BAD.append(name)      # expected: mapvote appended candidates in this movie\n")
cut('print(f"n_per_frame check:', "if complete and not NPF_BAD:",
    'print(f"n_per_frame against the shipped edgeprob_F4_ counts: {len(NPF_BAD)} of "\n'
    '      f"{len(names_f4) - len(NPF_NOCHK)} checkable movies DIFFER (expected: mapvote appends "\n'
    '      f"primaries; a movie with zero parents would match); {len(NPF_NOCHK)} not checkable", flush=True)\n')
rep("if complete and not NPF_BAD:", "if complete:")
rep("EPHPCACHE OK", "EPVTCACHE OK")
rep("EPHPCACHE INCOMPLETE", "EPVTCACHE INCOMPLETE")
assert "F4hp_" not in src.replace("fdet_F4hp", "").replace("edgeprob_F4hp", ""), "a stale F4hp_ label survives"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "epvtcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written epvtcache.py", len(src))
