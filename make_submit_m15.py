"""Generate submit_m15.py from submit_f4.py by asserted replacement.

Single variable vs ref 55889500 (F4, LB 0.925): ep05 points that the F4
average missed are ADDED as primaries -- mergescore's M15 arm, the only
merge arm that passed the clean-domain rule (44b6 +0.0060, zh +0.0104,
6bba -0.0064 non-voting).

Faithfulness to the offline arm matters in one non-obvious place: the
ep05 cache was built by peaking at CAND_THR 0.05, refining, running the
4 um NMS, and only THEN filtering score >= tau. Thresholding before the
NMS would let a 0.16 peak survive a 0.20 neighbour that suppressed it in
the cache, so this script keeps the cache's order.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "submit_f4.py"
DST = pathlib.Path(__file__).parent / "submit_m15.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""submit_f4.py -- ref 55817630 (LB 0.922) with ep05 DeepCenter as '
    'a 4th heatmap.',
    '"""submit_m15.py -- ref 55889500 (F4, LB 0.925) plus the ep05 points '
    'F4 missed.')

rep("SINGLE VARIABLE vs 55817630: detect() averages FOUR heatmaps instead of",
    "SINGLE VARIABLE vs 55889500: detect() still averages the same FOUR\n"
    "heatmaps, but an ep05 peak whose own score is >= MERGE_TAU 0.15 and\n"
    "which lands farther than MERGE_UM 4.0 um from EVERY point of that\n"
    "average is appended as a PRIMARY node at score UNET_THRESH. This is\n"
    "mergescore's M15 arm, the only one of tau 0.50/0.30/0.15 that passed\n"
    "the pre-registered clean-domain rule: adj_J vs F4 on D32 was +0.0060\n"
    "on 44b6 and +0.0104 on zh, -0.0064 on the non-voting 6bba. The 44b6\n"
    "gain is edge_J (0.8214 -> 0.8299), earned while the node deficit\n"
    "shrank -10.7% -> -7.4%, i.e. the local adj_J bookkeeping was working\n"
    "AGAINST the arm and it won anyway. Node growth 44b6 +3.94%, 6bba\n"
    "+4.07%, zh +2.03%; at the measured LB node tax (+10.3% -> -0.002) a\n"
    "+4% add costs about -0.0008, so the clean-domain edge gain has room.\n"
    "RISK ON RECORD: mergescore printed per-dataset detail for the\n"
    "DISCARDED M30 arm only, so the 44b6 +0.0060 was never checked for\n"
    "concentration in one or two datasets.\n"
    "\n"
    "(inherited from F4) detect() averages FOUR heatmaps instead of")

rep('SCRIPT_VERSION = "2026-08-30a-submit-f4"',
    'SCRIPT_VERSION = "2026-08-31a-submit-m15"')

# --- the merge constants + a shared peak extractor -----------------------
rep("def detect(vol):",
    "MERGE_TAU = 0.15    # mergescore's winning tau; == UNET_THRESH\n"
    "MERGE_UM = 4.0      # the separation our own NMS already enforces\n"
    "_MERGE_COUNT = [0, 0]      # [base points, ep05 points added]\n"
    "\n"
    "\n"
    "def _peaks(h, vol, thr):\n"
    "    \"\"\"F4's peak -> unpool -> refine -> physical NMS, unchanged.\"\"\"\n"
    "    pk = peak_local_max(h, min_distance=1, threshold_abs=thr,\n"
    "                        exclude_border=False)\n"
    "    if len(pk) == 0:\n"
    "        return np.zeros((0, 3)), np.zeros(0)\n"
    "    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)\n"
    "    coords = pk.astype(float)\n"
    "    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2\n"
    "    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2\n"
    "    ref = np.array([_refine(vol, c)[0] for c in coords])\n"
    "    return _physical_nms(ref, sc, NMS_UM)\n"
    "\n"
    "\n"
    "def detect(vol):")

rep("""    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)""",
    """    assert len(hs) == 4, f"expected 3 xl branches + DC, got {len(hs)}"
    h = np.mean(hs, axis=0)
    c0, s0 = _peaks(h, vol, DETECT_THRESH)
    _MERGE_COUNT[0] += len(c0)
    # ep05 alone, peaked/NMS'd at CAND_THR exactly as the dcdet_ep05
    # caches were, THEN cut at MERGE_TAU -- see the module docstring.
    cd, sd = _peaks(hs[3], vol, CAND_THR)
    cd = cd[sd >= MERGE_TAU]
    if len(cd) and len(c0):
        d, _ = cKDTree(c0 * SCALE[None, :]).query(cd * SCALE[None, :])
        cd = cd[d > MERGE_UM]
    if not len(cd):
        return c0, s0
    _MERGE_COUNT[1] += len(cd)
    return (np.concatenate([c0, cd], 0),
            np.concatenate([s0, np.full(len(cd), UNET_THRESH)], 0))""")

rep('print("SUBMIT RANKER DONE", flush=True)',
    '_b, _a = _MERGE_COUNT\n'
    'print(f"MERGE: {_a:,} ep05 points added on {_b:,} F4 points "\n'
    '      f"({100.0 * _a / max(_b, 1):.2f}% nodes) -- offline arm read "\n'
    '      f"+3.94% 44b6 / +4.07% 6bba / +2.03% zh", flush=True)\n'
    'print("SUBMIT RANKER DONE", flush=True)')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
