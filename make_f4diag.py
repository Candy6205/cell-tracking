"""Generate f4diag.py from edgediag.py by asserted replacement.

THE QUESTION (the user's, 2026-09-01, correcting my misread): trace
**F4** -- the pipeline that actually ships at LB 0.925 -- through every
stage on 6bba and 44b6, NODES and EDGES both. Not M15's added points.
M15 already failed and why it failed is not the question.

edgediag (2026-08-29) built exactly this instrument but ran it on the
0.922 pipeline's detections. This swaps the detection source to F4's
cached heatmap-average detections and runs BASE and F4 side by side, so
every number is also a statement about what the +0.003 actually did.

EDGE SIDE (edgediag's machinery, untouched):
  every GT edge charged to the FIRST stage that loses it --
  det_miss_subthresh / det_miss_cold / no_candidate / below_floor /
  node_absent / ilp_rejected / killed_post / recovered
  every FINAL edge classified against GT --
  tp_like / fp_confusion / fp_half / fp_invisible

NODE SIDE (new; almost all of it is already computed inside diag_one
and was simply never reported):
  detections -> primaries / repair band -> in graph -> ILP-linked ->
  post-link killed -> final, plus the final degree histogram, plus the
  GT-node ledger (matched by a primary / only by a repair-band
  candidate / cold-missed entirely).

ANCHOR: BASE-6bba must read adj_J 0.9293, edgediag's own and dcpool's.
F4 is then one documented change away from an anchored harness. Both
arms use D187, which is what the 0.925 submission ships.

zebrahub dropped: the question named 6bba and 44b6, and dropping it
roughly halves the runtime.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "edgediag.py"
DST = pathlib.Path(__file__).parent / "f4diag.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('SCRIPT_VERSION = "2026-08-29a-dcpool"',
    'SCRIPT_VERSION = "2026-09-01a-f4diag"')

# ---- 1. the F4 detection caches + an arm-aware loader -------------------
rep('''def load_dets(name):
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]''',
    '''def _walk_npz(prefix):
    """mergescore's bounded walk (trap 20: prune the competition tree and
    the zarr/geff trees or this takes minutes)."""
    import os
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


FDET_F4 = _walk_npz("fdet_F4_")
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"


def load_dets(name, arm="BASE"):
    """BASE = the xl three-branch average that shipped at 0.922.
    F4 = the same with ep05 DeepCenter as a fourth equal-weight map,
    which is what ships at 0.925 (ref 55889500)."""
    z = np.load(CACHES[name] if arm == "BASE" else FDET_F4[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]''')

# ---- 2. the node-side ledger, folded in where the matching already is ---
rep('''    # ---- final-edge (FP-side) classification --------------------------''',
    '''    # ---- node-side ledger ---------------------------------------------
    # Almost everything here was already computed for the edge pass and
    # simply never reported.
    nn = NAGG.setdefault(domain, defaultdict(int))
    for t in range(T):
        s_all = np.asarray(dets[t][1], float).reshape(-1)
        nn["dets_total"] += len(s_all)
        nn["repair_band"] += int(((s_all >= CAND_THR)
                                  & (s_all < UNET_THRESH)).sum())
        nn["primaries"] += len(seeds[t][0])
    nn["in_graph"] += len(seed_id)
    _lk = set()
    for a_, b_ in edges0:
        _lk.add(a_)
        _lk.add(b_)
    _sid = set(seed_id.values())
    nn["ilp_linked"] += len(_lk & _sid)
    nn["nodes_after_ilp"] += len(nodes0)
    nn["final_nodes"] += len(nodes_f)
    nn["final_from_seed"] += len(_sid & set(nodes_f))
    nn["final_repair"] += len([g for g in nodes_f if g not in _sid])
    nn["killed_post"] += len(_sid - set(nodes_f))
    _dg = defaultdict(int)
    for a_, b_ in edges_f:
        _dg[a_] += 1
        _dg[b_] += 1
    for g in nodes_f:
        nn[f"deg{min(_dg.get(g, 0), 3)}"] += 1
    # GT-node ledger: matched by a primary, reachable only through the
    # sub-threshold repair band, or nothing there at all.
    for r in range(len(gt_t)):
        if int(gt_t[r]) >= T:
            continue
        nn["gt_nodes"] += 1
        if r in m_seed:
            nn["gt_matched_primary"] += 1
        elif near_pool.get(r):
            nn["gt_subthresh_only"] += 1
        else:
            nn["gt_cold"] += 1

    # ---- final-edge (FP-side) classification --------------------------''')

rep("""CATS = ["det_miss_subthresh", "det_miss_cold", "no_candidate",
        "below_floor", "node_absent", "ilp_rejected", "killed_post",
        "recovered"]""",
    """CATS = ["det_miss_subthresh", "det_miss_cold", "no_candidate",
        "below_floor", "node_absent", "ilp_rejected", "killed_post",
        "recovered"]
NAGG = {}          # domain -> node-side counters""")

# ---- 3. two arms, no zebrahub ------------------------------------------
rep('''# ---- the two competition domains ---------------------------------------
for domain, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    td0 = time.time()
    outd = pdir("DIAG", domain)
    for name in names:
        dets = load_dets(name)
        T = len(dets)''',
    '''# ---- two arms x the two competition domains -----------------------------
for _arm in ("BASE", "F4"):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    domain = f"{_arm}-{_dm}"
    td0 = time.time()
    outd = pdir("DIAG", domain)
    for name in names:
        dets = load_dets(name, _arm)
        T = len(dets)''')

rep('''    res[domain], PER2[domain] = score_dir(outd)
    print(f"  {domain}: {len(names)} datasets ({time.time() - td0:.0f}s)",
          flush=True)''',
    '''    res[domain], PER2[domain] = score_dir(outd)
    print(f"  {domain}: {len(names)} datasets ({time.time() - td0:.0f}s)",
          flush=True)''')

rep('''# ---- zebrahub -----------------------------------------------------------
if SET_ZH:''',
    '''# ---- zebrahub: dropped, the question named 6bba and 44b6 ----------------
if False:''')

rep('''DOMAINS = [d for d in ("6bba", "44b6", "zh") if d in res]
print("\\n=== ANCHOR / official scorer ===", flush=True)
print("  6bba must read adj_J 0.9293 (dcpool, same machinery).", flush=True)''',
    '''DOMAINS = [d for d in ("BASE-6bba", "F4-6bba", "BASE-44b6", "F4-44b6")
           if d in res]
print("\\n=== ANCHOR / official scorer ===", flush=True)
print("  BASE-6bba must read adj_J 0.9293 (dcpool and edgediag, same "
      "machinery). BASE-44b6 must read 0.8185 (fusescore). F4 is then "
      "ONE documented change away from an anchored harness: the "
      "detection cache. Both arms run D187, which is what ref 55889500 "
      "ships.", flush=True)''')

# ---- 4. the node report -------------------------------------------------
rep('''print("\\n=== WHERE EVERY GT EDGE GOES (charged to the FIRST losing "
      "stage) ===", flush=True)''',
    '''def _p(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "  -  "


print("\\n=== NODES, STAGE BY STAGE ===", flush=True)
print("  detections split into PRIMARIES (score >= 0.15, the only ones "
      "that can become nodes) and the REPAIR BAND [0.05, 0.15) which "
      "only _gap_close may draw on. Percentages are of primaries unless "
      "stated. `final_repair` are nodes _gap_close created FROM that "
      "band, so they are not primaries at all.", flush=True)
print(f"  {'arm-domain':<12} {'dets':>10} {'primary':>10} {'band':>9} "
      f"{'in_graph':>9} {'ILP':>8} {'killed':>8} {'final':>10} "
      f"{'+repair':>8}", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    pr = n["primaries"]
    print(f"  {d:<12} {n['dets_total']:>10,} {pr:>10,} "
          f"{n['repair_band']:>9,} {_p(n['in_graph'], pr):>9} "
          f"{_p(n['ilp_linked'], pr):>8} {_p(n['killed_post'], pr):>8} "
          f"{n['final_nodes']:>10,} {n['final_repair']:>8,}", flush=True)

print("\\n=== FINAL-OUTPUT DEGREE (3 = 3 or more) ===", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    tot = n["final_nodes"]
    print(f"  {d:<12} n={tot:>9,}  " + "  ".join(
        f"deg{k}={_p(n.get(f'deg{k}', 0), tot)}" for k in (0, 1, 2, 3)),
        flush=True)

print("\\n=== THE GT-NODE LEDGER: what the detector did to real cells ===",
      flush=True)
print("  matched = a PRIMARY sits within the match radius. subthresh = "
      "no primary, but a 0.05-0.15 candidate is there (the detector saw "
      "it and the gate rejected it). cold = nothing there at all. Only "
      "the annotated cells are counted, and 6bba annotates 9.71% of its "
      "cells against 44b6's 0.77%.", flush=True)
print(f"  {'arm-domain':<12} {'GT nodes':>10} {'matched':>10} "
      f"{'subthresh':>11} {'cold':>10}", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    g = n["gt_nodes"]
    print(f"  {d:<12} {g:>10,} "
          f"{n['gt_matched_primary']:>6,} {_p(n['gt_matched_primary'], g):>3} "
          f"{n['gt_subthresh_only']:>6,} {_p(n['gt_subthresh_only'], g):>4} "
          f"{n['gt_cold']:>6,} {_p(n['gt_cold'], g):>3}", flush=True)

print("\\n=== WHERE EVERY GT EDGE GOES (charged to the FIRST losing "
      "stage) ===", flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
