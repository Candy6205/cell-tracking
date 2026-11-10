"""hparabench_ref.py -> gapcross.py: the post-link candidates from the
public 0.946 notebook, CROSSED rather than tested one at a time.

The user, going to bed 2026-09-08: "一个一个干吧，注意不同候选之间会不会互相
影响，比如说一个放开限制一个收紧限制之类的". That is exactly the situation
here, and it is not hypothetical:

  * OUR only pre-fork-gate leaderboard win was LOOSENING the gap budget,
    GAP1_CAP_FRAC 0.003 -> 0.02 (+0.002 LB). It worked because the budget
    was provably BINDING: 30 of 83 datasets hit the cap with 64% of demand
    unrepaired.
  * THEIR 0.939 -> 0.941 step TIGHTENED the gap radius, GAP1_GATE_UM
    5.8 -> 5.0. Ours has never moved off the public baseline's 9.0.

Tightening the radius proposes fewer repairs. If it proposes few enough,
the budget stops binding and our +0.002 evaporates -- a loosened knob and a
tightened knob on the SAME stage. Testing either alone would read as a
result and be an artifact of the other. So the gap arms are a 2x2.

THE SECOND INTERACTION, and it makes one candidate a provable no-op alone.
Their adaptive short-track rescue only ever considers components with
`SHORT_TRACK_RESCUE_MIN_LEN <= len < OUTPUT_MIN_TRACK_LEN`, i.e. length 4 or
5 at their settings. WE RUN SHORT_MIN = 4, so nothing of length 4 or 5 is
ever deleted here and that window is EMPTY. Porting the rescue alone would
change nothing and would look like a null result about the rescue. The unit
under test is therefore (SHORT_MIN 6 + rescue), not the rescue.

ARMS, all sharing ONE ILP solve per movie -- `patch_final` rebuilds from a
fresh copy of nodes0/edges0, so the expensive stage is paid once:

  REF     gate 9.0  cap 0.02   short 4   rescue off   <- the shipped 0.944
  g5      gate 5.0  cap 0.02   short 4   off          their radius, our budget
  g5c3    gate 5.0  cap 0.003  short 4   off          both of theirs
  g9c3    gate 9.0  cap 0.003  short 4   off          the pre-cap20 baseline
  s6      gate 9.0  cap 0.02   short 6   off          re-test of SHORT_MIN
  s6r     gate 9.0  cap 0.02   short 6   RESCUE ON    the actual candidate

g9c3 is a CONTROL, not a candidate: it must reproduce the loss that cap20
was adopted to fix. If it does not, the harness is not measuring the gap
stage and none of the other five rows mean anything.

NOT INCLUDED, and why -- both were checked against the record first rather
than copied off the notebook:

  * ILP disappearance 1.5 -> 2. `ilpsweep` (2026-09-05) already ran the
    notebook's ILP settings on our nodes: +0.0002 / -0.0115 / -0.0010, and
    the chains it deletes are 62%/91%/58% TRUE edges. REF is the local
    optimum. Their 2 is tuned for their node set, which is much smaller
    (18,618 against our 23,371 on the same movie).
  * ILP division weight 0.55 -> 1.2. Not a parameter difference, an
    ARCHITECTURAL one: they push the ILP to emit divisions natively and
    filter with geometry + DeepCenter, we suppress it and let the learned
    fork gate decide. 1.2 here would flood the graph with divisions the
    gate was never fitted to arbitrate.

ANCHOR: REF must reproduce divbench9 over the 128 -- edge_J 0.9104,
adj_J 0.9133, div_J 0.1337, score 0.9266. Asserted, not printed.
"""
import os

src = open("hparabench_ref.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparabench"',
    'SCRIPT_VERSION = "2026-09-08c-gapcross"')
rep('DET_SRC = "F4"', 'DET_SRC = "F4"      # gapcross is a REFINE-coordinate run')

# ---- the arm table, and the rescue itself -------------------------------
rep('''GAP1_CAP_FRAC = 0.02     # SHIPPED (cap20, ref 55817630)''',
    '''GAP1_CAP_FRAC = 0.02     # SHIPPED (cap20, ref 55817630)

# (arm, GAP1_GATE_UM, GAP1_CAP_FRAC, SHORT_MIN, rescue)
PATCH_ARMS = [
    ("g5",   5.0, 0.02,  4, False),   # their radius, our budget
    ("g5c3", 5.0, 0.003, 4, False),   # both of theirs
    ("g9c3", 9.0, 0.003, 4, False),   # CONTROL: the pre-cap20 baseline
    ("s6",   9.0, 0.02,  6, False),   # SHORT_MIN alone, re-tested
    ("s6r",  9.0, 0.02,  6, True),    # SHORT_MIN 6 + the rescue
]
RESCUE_MIN_LEN = 4
RESCUE_MIN_MEAN_PROB = 0.88
RESCUE_MAX_MEAN_DIST_UM = 3.0
RESCUE_MAX_NODES_FRAC = 0.012
RESCUE_MAX_NODES_ABS = 120
RESCUE_TRIGGER_REMOVED_FRAC = 0.10
RESCUE = False           # rebound per arm
RESCUE_LOG = {}''')

# ---- _short_filter gains the rescue, off by default ---------------------
rep('''def _short_filter(nodes, edges):''',
    '''def _rescue_short(nodes, edges, comps, keep, pfun):
    """The 0.946 notebook's adaptive short-track rescue, ported.

    Their gate values (0.88 mean edge prob, 3.0 um mean edge length) are on
    THEIR probability distribution. Ported unchanged here on purpose: the
    first question is whether the MECHANISM does anything at all on our
    graphs, and re-deriving the thresholds before knowing that would be
    tuning against noise. If the mechanism fires and helps, the thresholds
    become the next question, not the first one.
    """
    removed = len(nodes) - len(keep)
    if removed <= 0:
        return keep
    if removed / max(len(nodes), 1) < RESCUE_TRIGGER_REMOVED_FRAC:
        RESCUE_LOG.setdefault("not_triggered", 0)
        RESCUE_LOG["not_triggered"] += 1
        return keep
    budget = min(RESCUE_MAX_NODES_ABS,
                 max(0, int(round(len(nodes) * RESCUE_MAX_NODES_FRAC))))
    props = []
    for members in comps:
        ms = set(members)
        if ms & keep:
            continue
        if not (RESCUE_MIN_LEN <= len(members) < SHORT_MIN):
            continue
        ps, ds = [], []
        for a_, b_ in edges:
            if a_ in ms and b_ in ms:
                p_ = pfun(a_, b_) if pfun is not None else float("nan")
                if p_ == p_:
                    ps.append(float(p_))
                ds.append(float(np.linalg.norm(
                    (nodes[a_]["xyz"] - nodes[b_]["xyz"]) * SCALE)))
        if not ds:
            continue
        mp = float(np.mean(ps)) if ps else 0.0
        md = float(np.mean(ds))
        if mp < RESCUE_MIN_MEAN_PROB or md > RESCUE_MAX_MEAN_DIST_UM:
            continue
        props.append((mp - 0.02 * md + 0.004 * len(members),
                      len(members), tuple(members)))
    props.sort(reverse=True)
    used = 0
    for _, size, members in props:
        if used + size > budget:
            continue
        keep.update(members)
        used += size
        RESCUE_LOG["components"] = RESCUE_LOG.get("components", 0) + 1
    RESCUE_LOG["nodes"] = RESCUE_LOG.get("nodes", 0) + used
    RESCUE_LOG["proposed"] = RESCUE_LOG.get("proposed", 0) + len(props)
    return keep


def _short_filter(nodes, edges):''')

# ---- hook the rescue into the filter, and thread pfun through -----------
rep('''    keep = set()
    for members in comps.values():
        if len(members) >= SHORT_MIN or \\
                any(out_count[n] >= 2 for n in members):
            keep.update(members)
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}''',
    '''    keep = set()
    for members in comps.values():
        if len(members) >= SHORT_MIN or \\
                any(out_count[n] >= 2 for n in members):
            keep.update(members)
    if RESCUE:
        keep = _rescue_short(nodes, edges, list(comps.values()), keep,
                             PATCH_PFUN[0])
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}''')

# `_short_filter` has no pfun argument and is called from patch_final, which
# does. A module-level slot is uglier than threading it through, and safer:
# the signature stays byte-identical, so REF cannot change behaviour.
rep('''def _rescue_short(nodes, edges, comps, keep, pfun):''',
    '''PATCH_PFUN = [None]


def _rescue_short(nodes, edges, comps, keep, pfun):''')

rep('''    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}''',
    '''    cand, cand_sc, cand_trees = pool
    PATCH_PFUN[0] = pfun
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}''')

# ---- run the five extra arms off the ONE solve --------------------------
rep('''    cells = []
    for arm, dw in DIVW_ARMS:''',
    '''    for _arm, _gate, _cap, _smin, _resc in PATCH_ARMS:
        _keep = (GAP1_GATE_UM, GAP1_CAP_FRAC, SHORT_MIN, RESCUE)
        globals().update(GAP1_GATE_UM=_gate, GAP1_CAP_FRAC=_cap,
                         SHORT_MIN=_smin, RESCUE=_resc)
        try:
            _op = pathlib.Path(f"{TMP}/patch_{_arm}") / f"{name}.geff"
            _op.parent.mkdir(parents=True, exist_ok=True)
            PATCH_CAPS.clear()
            patch_final(nodes0, edges0, pool, _op, pfun=pfun)
            for _k2, _v2 in PATCH_CAPS.items():
                CAPS_BY_ARM.setdefault(_arm, defaultdict(int))[_k2] += _v2
            PATCH_ROWS[_arm].append(score_one(_op, name))
            PATCH_PER[_arm][name] = PATCH_ROWS[_arm][-1]
        finally:
            (globals().__setitem__("GAP1_GATE_UM", _keep[0]),
             globals().__setitem__("GAP1_CAP_FRAC", _keep[1]),
             globals().__setitem__("SHORT_MIN", _keep[2]),
             globals().__setitem__("RESCUE", _keep[3]))

    cells = []
    for arm, dw in DIVW_ARMS:''')

rep('''ROWS = []
PER = {}''',
    '''ROWS = []
PER = {}
PATCH_ROWS = {a[0]: [] for a in PATCH_ARMS}
PATCH_PER = {a[0]: {} for a in PATCH_ARMS}''')

# ---- the report: the 2x2 read as a 2x2, and the anchor asserted ---------
rep('''print("\\n=== ANCHORS / official scorer ===", flush=True)''',
    '''print(f"\\n=== THE GAP 2x2 AND THE SHORT-TRACK PAIR ===", flush=True)
print("  All six rows come from the SAME ILP solve per movie; only the "
      "post-link stage differs. Read the 2x2 as a 2x2: the question is "
      "whether tightening the radius removes the budget's reason to be "
      "loose, not whether either helps alone.", flush=True)
_ref = _with_ratio(ROWS)
print(f"  {'arm':<6} {'gate':>5} {'cap':>6} {'short':>6} {'resc':>5} "
      f"{'adj_J':>8} {'div_J':>7} {'score':>8} {'d score':>9}")
print(f"  {'REF':<6} {9.0:>5} {0.02:>6} {4:>6} {'off':>5} "
      f"{_ref['adj_edge_jaccard']:>8.4f} {_ref['division_jaccard']:>7.4f} "
      f"{_ref['score']:>8.4f} {0.0:>+9.4f}")
for _a, _g, _c, _s, _r in PATCH_ARMS:
    if not PATCH_ROWS[_a]:
        print(f"  {_a:<6} (no rows)")
        continue
    _w = _with_ratio(PATCH_ROWS[_a])
    print(f"  {_a:<6} {_g:>5} {_c:>6} {_s:>6} {('on' if _r else 'off'):>5} "
          f"{_w['adj_edge_jaccard']:>8.4f} {_w['division_jaccard']:>7.4f} "
          f"{_w['score']:>8.4f} {_w['score'] - _ref['score']:>+9.4f}")
print(f"\\n  CONTROL: g9c3 is the pre-cap20 baseline and MUST be clearly "
      f"below REF. cap20 was adopted for +0.002 on the leaderboard; if "
      f"g9c3 does not lose here, this harness is not measuring the gap "
      f"stage and no other row above means anything.", flush=True)
print(f"  rescue log: {RESCUE_LOG if RESCUE_LOG else 'never fired'}",
      flush=True)
print(f"  (rescue at SHORT_MIN 4 is a provable no-op -- its window is "
      f"[4, SHORT_MIN) -- which is why s6r pairs it with SHORT_MIN 6.)",
      flush=True)

print("\\n=== ANCHORS / official scorer ===", flush=True)''')

rep('''print("HPARABENCH OK", flush=True)''',
    '''_g9c3 = _with_ratio(PATCH_ROWS["g9c3"]) if PATCH_ROWS["g9c3"] else None
    if _g9c3 is not None and _g9c3["score"] >= _ref["score"]:
        print(f"  [degraded] CONTROL FAILED: g9c3 {_g9c3['score']:.4f} is "
              f"not below REF {_ref['score']:.4f}. The gap stage is not "
              f"being exercised; do not read the other arms.", flush=True)
    print("GAPCROSS OK", flush=True)''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "gapcross.py")
open(os.path.abspath(out), "w", encoding="utf-8",
     newline="\n").write(src)
print("written", len(src))
