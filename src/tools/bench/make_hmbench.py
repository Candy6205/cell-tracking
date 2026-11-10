"""dupsup.py -> hmbench.py: the round-2 bench on hmerge's detection set.

WHAT IT MEASURES. The whole shipped pipeline (features, +BIDIR fold rankers,
ILP at floor 0.48 / DIV_W 0.55, the post-link patches, gate6 and joint fork
gates, the official scorer) run with DET_SRC = "F4HM90": the hpara
detections of the 140 bench movies with saddle-keyed pair merging (hmerge,
saddle >= 0.9 within 9.5 um), and BOTH edge heads re-run on those points
(ephmcache / ep2hmcache), because the pair set changes and the cached
(t, i, j) rows do not join. No suppression arms: DUP_ARMS is empty, so the
run is REF only. The comparison with the shipped placement is OFFLINE: pair
`hmbench_REF_F4HM90.csv` with `dupsup_REF_F4HP.csv` (same 128 movies, same
folds, same harness lineage; the REF rows of bridgerank / divadmit / dupsup
are identical) and bootstrap edge_J, adj_J and score per movie; read the
gated cells (REF + gate6, REF + joint) against round 2's 0.9408 / 0.9433.

ANCHORS. ANCHOR_D187 / +BIDIR / REF are demoted to soft exactly as the F4HP
run demotes the F4 ones: they are EXPECTED to move -- that is the
measurement -- so they are printed with round 2's values, never asserted.
Attach: dupsup's 16 sources + biohub-hmerge + biohub-ephmcache +
biohub-ep2hmcache. CPU, ~3 h.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-12a-hmbench"')

# ---- the detection source -------------------------------------------------------
rep('DET_SRC = "F4HP"\nassert DET_SRC in ("F4", "F4HP"), DET_SRC',
    'DET_SRC = "F4HM90"\nassert DET_SRC in ("F4", "F4HP", "F4HM90"), DET_SRC')
rep('if DET_SRC == "F4HP":', 'if DET_SRC in ("F4HP", "F4HM90"):', n=src.count('if DET_SRC == "F4HP":'))
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))', 'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hm90_"))')
rep('print(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)', 'print(f"fdet_F4hm90 caches: {len(FDET_F4HP)}", flush=True)')
# the heads: keep excluding the hpara labels from the BASE dicts, add the hm ones, and point the override at hm
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("F4hm90_")}', n=2)
rep('EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("F4hp_")}',
    'EDGEPROB_F4HP = {k[len("F4hm90_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("F4hm90_")}')
rep('EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("F4hp_")}',
    'EDGEPROB2_F4HP = {k[len("F4hm90_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("F4hm90_")}')

# ---- no suppression arms: REF only ------------------------------------------------
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = []             # hmbench: REF only; the comparison with F4HP is offline')

# ---- anchors: the round-2 values are the reference, not a gate ----------------------
rep('''      f"0.9146 -> {'OK' if _r2 else 'MISS'}", flush=True)''',
    '''      f"0.9146 -> {'SAME' if _r2 else 'MOVED (expected: DET_SRC F4HM90)'}", flush=True)''')
rep('''      f"{'OK' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MISS'}", flush=True)''',
    '''      f"{'SAME' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MOVED (expected: DET_SRC F4HM90)'}", flush=True)''')
rep("if len(ROWS) == len(SIX) and not INCOMPLETE and _r2:", "if len(ROWS) == len(SIX) and not INCOMPLETE:")
rep('print("DUPSUP OK", flush=True)', 'print("HMBENCH OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] HMBENCH INCOMPLETE:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"hmbench_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"hmbench_arm_{a_}.csv"')

assert src.count("F4hm90_") >= 6 and 'DET_SRC = "F4HM90"' in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hmbench.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written hmbench.py", len(src))
