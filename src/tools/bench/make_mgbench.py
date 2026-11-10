"""dupsup.py -> mgbench_<tag>.py: the round-2 bench on a hard-merged detection set.

Usage: py tools/bench/make_mgbench.py F4mg90   -> writes mgbench_f4mg90.py

Same shape as make_hmbench.py (the bench on hmerge's set): DET_SRC becomes the
merged set written by mergecache (fdet_<tag>_ caches, heads edgeprob_<tag>_ /
edgeprob2_<tag>_ remapped from the union heads), no suppression arms, REF
only; the round-2 anchors are demoted to soft (they are EXPECTED to move);
the comparison with the shipped placement is offline: pair
`mgbench_REF_<TAG>.csv` with `dupsup_REF_F4HP.csv` through tools/hmpair.py
(same 128 movies, same folds, same lineage) and read the gated cells (REF +
joint) against round 2's 0.9433.

Attach: dupsup's 16 sources + biohub-mergecache. CPU, ~3 h.
"""
import os
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "F4mg90"
assert TAG.startswith("F4mg") and TAG[4:].isdigit(), TAG
UP = TAG.upper()                       # the DET_SRC token, e.g. F4MG90
src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', f'SCRIPT_VERSION = "2026-09-15a-mgbench-{TAG}"')

# ---- the detection source -------------------------------------------------------
rep('DET_SRC = "F4HP"\nassert DET_SRC in ("F4", "F4HP"), DET_SRC',
    f'DET_SRC = "{UP}"\nassert DET_SRC in ("F4", "F4HP", "{UP}"), DET_SRC')
rep('if DET_SRC == "F4HP":', f'if DET_SRC in ("F4HP", "{UP}"):', n=src.count('if DET_SRC == "F4HP":'))
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))', f'FDET_F4HP = _walk_files(_npz_prefix("fdet_{TAG}_"))')
rep('print(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)', f'print(f"fdet_{TAG} caches: {{len(FDET_F4HP)}}", flush=True)')
# mergecache writes the 128 6bba movies only; the 12 44b6 eval movies stay on hpara coordinates
rep('    assert len(FDET_F4HP) >= 140, (', '    assert len(FDET_F4HP) >= 128, (')
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    f'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("{TAG}_")}}', n=2)
rep('EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("F4hp_")}',
    f'EDGEPROB_F4HP = {{k[len("{TAG}_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("{TAG}_")}}')
rep('EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("F4hp_")}',
    f'EDGEPROB2_F4HP = {{k[len("{TAG}_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("{TAG}_")}}')

# ---- no arms: REF only -----------------------------------------------------------
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = []             # mgbench: REF only; the comparison with F4HP is offline')

# ---- anchors: the round-2 values are the reference, not a gate ----------------------
rep('''      f"0.9146 -> {'OK' if _r2 else 'MISS'}", flush=True)''',
    f'''      f"0.9146 -> {{'SAME' if _r2 else 'MOVED (expected: DET_SRC {UP})'}}", flush=True)''')
rep('''      f"{'OK' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MISS'}", flush=True)''',
    f'''      f"{{'SAME' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MOVED (expected: DET_SRC {UP})'}}", flush=True)''')
rep("if len(ROWS) == len(SIX) and not INCOMPLETE and _r2:", "if len(ROWS) == len(SIX) and not INCOMPLETE:")
rep('print("DUPSUP OK", flush=True)', 'print("MGBENCH OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] MGBENCH INCOMPLETE:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"mgbench_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"mgbench_arm_{a_}.csv"')

assert src.count(f"{TAG}_") >= 6 and f'DET_SRC = "{UP}"' in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", f"mgbench_{TAG.lower()}.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print(f"written mgbench_{TAG.lower()}.py", len(src))
