"""ephpcache.py -> ephrcache.py: the first edge head, re-run on hrefine's re-placed primaries (fdet_F4hr_).

Same cut as make_eplncache.py made from ephpcache.py, with ONE difference that matters: hrefine moves every
primary and adds or removes none, so the n_per_frame check against the shipped `edgeprob_F4_` cache is
ENFORCED exactly as ephpcache enforces it (a count mismatch means the two arms are not scoring the same cells).
  * the detection cache prefix is `fdet_F4hr_` (biohub-hrefine) and the output label `F4hr_<name>`, so the
    file is `edgeprob_F4hr_<name>.npz`;
  * nothing else -- the head, its weights and its crops are untouched, so the arm differs from F4HP in the
    detection COORDINATES only.
Attach: ephpcache's sources with biohub-hparacache replaced by biohub-hrefine. GPU T4, ~1.5-2 h (140 movies).
"""
import os

src = open("ephpcache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-ephpcache"', 'SCRIPT_VERSION = "2026-09-23a-ephrcache"')
rep("fdet_F4hp_", "fdet_F4hr_", n=src.count("fdet_F4hp_"))
rep('label = name if arm == "base" else f"F4hp_{name}"', 'label = name if arm == "base" else f"F4hr_{name}"')
rep("edgeprob_F4hp_", "edgeprob_F4hr_", n=src.count("edgeprob_F4hp_"))
rep("EPHPCACHE OK", "EPHRCACHE OK")
rep("EPHPCACHE INCOMPLETE", "EPHRCACHE INCOMPLETE")
assert "F4hp_" not in src.replace("fdet_F4hp", "").replace("edgeprob_F4hp", ""), "a stale F4hp_ label survives"
assert "if complete and not NPF_BAD:" in src, "the n_per_frame check must stay enforced"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ephrcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written ephrcache.py", len(src))
