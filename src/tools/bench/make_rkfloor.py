"""eprank3.py -> rkfloor.py: is the retrain's loss real, or the wrong floor?

The user: *"为啥重训会更差，好想知道这个原因"*. Chasing that killed the answer
I had given.

WHAT IS ALREADY RULED OUT, by eprank3's own numbers. I wrote that the BASE
pool acts as augmentation, so removing the mismatch removes the
regularisation. **That is a claim about generalisation, and edge AUC is flat
in five independent readings** -- rkpool folds 0.9935 / 0.9917 against
0.9935 / 0.9918, eprank3 val 0.9926 both, F4-6bba 0.9929 both, F4-44b6
0.9871 against 0.9872. Plain overfitting to the F4 pool is out for the same
reason. **The retrained ranker orders edges exactly as well.**

WHAT MOVED are the threshold-sensitive outputs: `div_J` −13% against `edge_J`
−0.13%, and fewer nodes surviving on BOTH domains. Two thirds of the 6bba
score loss is divisions, and the ILP does not read the ranking -- it reads
the probability VALUE against a floor.

AND THE SIMPLEST CALIBRATION STORY IS MEASURED AND FALSE. Both v4 pkls
unpickle to intercepts 0.016 logit apart (−0.012710 against +0.003145),
which is exactly the training base-rate difference (0.497 -> 0.501 positive)
and therefore class balance rather than anything learned -- **and it points
the wrong way**, since a higher intercept raises probabilities while what was
observed is fewer nodes surviving. If calibration is the mechanism it is a
change in the trees' shape, which cannot be read off a pkl.

THE CONFOUND, and it is mine. `eprank3` has **no `PROD_FLOOR` at all**: it
inherits `edgeprob_rank2`'s `ILP_FLOOR = 0.10` and solves with it, while what
ships is 0.48. So the retrain was scored at a scale belonging neither to the
new ranker nor to production -- the same shape of error as `divradius`
freezing the fork gate, three experiments earlier.

WHAT THIS KERNEL DOES, and why it is cheap: **nothing is retrained.** Both
v4 pkls already exist as kernel outputs, so this loads them, builds features
for the **24 evaluation movies only** (eprank3 built 199), and crosses
FLOOR x PKL. It also prints each ranker's probability QUANTILES on the same
gated rows, which is the calibration measurement that cannot be made
offline.

FLOORS: 0.10 (eprank3's, and ilpsweep's REF), 0.30, 0.48 (what ships, ref
56030644), 0.60. Two of those carry recorded anchors, so the sweep machinery
is validated before any new number is read:

    BASE pkl, floor 0.10 -> F4-6bba 0.9261, F4-44b6 0.8515
    BASE pkl, floor 0.48 -> F4-6bba 0.9261, F4-44b6 0.8384

6bba is 0.9261 at both because ilpsweep measured its F48 delta as +0.0000;
44b6's is −0.0131. That +0.0000 is exactly what hid the difference from me
when I first wrote the constant down as 0.48.

READING, fixed here. If the loss is the operating point, the F4 pkl has its
own best floor and the gap shrinks or reverses there. If it is real, no floor
recovers it. ⚠ Either way this answers the LOCAL question only: floor 0.48
ships because it **won on LB** (+0.001, submit-f48, −1.43% bytes) while
locally reading +0.0000 / −0.0131, so local and LB already disagree about
this knob.

Attach: eprank3's sources PLUS faustine649/biohub-eprank3 and
faustine649/biohub-eprank3-ref (the two v4 pkls). CPU only.
"""
import os

src = open("eprank3.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-eprank3"',
    'SCRIPT_VERSION = "2026-09-09a-rkfloor"')

# ---- no pool pass: nothing is retrained --------------------------------
rep('_jobs = [(n, POOL_ARM, "pool") for n in POOL175]',
    '''# NOTHING IS RETRAINED, so the 175-movie pool pass is dead weight -- it
# existed only to build X_pool. Dropping it takes the feature phase from 199
# passes to 24 and is the whole reason this kernel is affordable.
_jobs = []''')

rep('''X_pool = np.concatenate(X_pool_l).astype(np.float32)
y_pool = np.concatenate(y_pool_l)''',
    '''X_pool = (np.concatenate(X_pool_l).astype(np.float32) if X_pool_l
          else np.zeros((0, NCOL_ALL), np.float32))
y_pool = np.concatenate(y_pool_l) if y_pool_l else np.zeros(0, np.int64)''')

# ---- the arms are the two LOADED v4 pkls -------------------------------
rep('ARM_ORDER = ["+BIDIR"]                                  # trained',
    '''ARM_ORDER = []                     # nothing is trained in this kernel
# The two v4 pkls eprank3 / eprank3-ref wrote, loaded rather than refitted.
# Same 21 columns, same pool of 175 names; the ONLY difference between them
# is whether their training rows came from BASE or F4 primaries.
PKL_ARMS = [("v4_BASE", "edge_ranker_ep175v4_bidir_base.pkl"),
            ("v4_F4", "edge_ranker_ep175v4_bidir_f4.pkl")]
FLOORS = [0.10, 0.30, 0.48, 0.60]
# floor -> {(arm, domain): recorded adj_J}. Two cells at 0.10 (eprank3-ref,
# ilpsweep REF) and two at 0.48 (ilpsweep's F48 delta +0.0000 / -0.0131).
FLOOR_ANCHOR = {
    (0.10, "v4_BASE", "F4-6bba"): 0.9261,
    (0.10, "v4_BASE", "F4-44b6"): 0.8515,
    (0.48, "v4_BASE", "F4-6bba"): 0.9261,
    (0.48, "v4_BASE", "F4-44b6"): 0.8384,
}
FLOOR_TOL = 0.0005''')

# ---- solve_one takes a floor -------------------------------------------
rep('''def solve_one(dets, T, seeds, feats, model, cols):''',
    '''def solve_one(dets, T, seeds, feats, model, cols, floor=None):''')
rep('''            if pr >= ILP_FLOOR:''',
    '''            if pr >= (ILP_FLOOR if floor is None else floor):''')

# ---- cut stage B and replace it with the cross -------------------------
_CUT = 'print(f"\\n=== STAGE B: full pipeline, {len(EVAL_ARMS)} arms x {len(DOMAINS)} "'
assert src.count(_CUT) == 1
src = src[:src.index(_CUT)]

src += r'''# ---- THE CROSS: floor x pkl, on the 24 evaluation movies ---------------
#
# eprank3 measured the retrain at ILP_FLOOR = 0.10 and called the result a
# loss. 0.10 is not what ships (0.48 is, ref 56030644) and it is not the new
# ranker's own operating point either. This crosses them instead of freezing
# one, which is the correction the user made three experiments earlier when
# divradius froze the fork gate.
_PKL = {}
for _tag, _fn in PKL_ARMS:
    _p = next(pathlib.Path("/kaggle/input").glob(f"*/**/{_fn}"), None)
    assert _p is not None, (
        f"{_fn} not found under /kaggle/input -- attach "
        f"faustine649/biohub-eprank3 (the F4 pkl) and "
        f"faustine649/biohub-eprank3-ref (the BASE one)")
    with open(_p, "rb") as _f:
        _b = pickle.load(_f)
    assert len(_b["pool"]) == 175, (_tag, len(_b["pool"]))
    assert list(_b["feature_names"]) == BASE_NAMES + ["p_fwd", "p_rev",
                                                      "h_fused"], _tag
    _PKL[_tag] = (_b["model"], [int(c) for c in _b["cols"]])
    print(f"  {_tag}: {_fn} from {_p} | pool {len(_b['pool'])} | "
          f"version {_b.get('version')}", flush=True)
assert len(_PKL) == 2

print(f"\n=== FLOOR x PKL: {len(FLOORS)} floors x {len(PKL_ARMS)} rankers x "
      f"{len(NAMES_EVAL)} datasets, nothing retrained ===", flush=True)
print(f"  floors {FLOORS}; 0.10 is eprank3's and ilpsweep's REF, 0.48 is "
      f"what ships (ref 56030644, LB +0.001).", flush=True)
print(f"  ANCHORS, {len(FLOOR_ANCHOR)} cells: v4_BASE at 0.10 must read "
      f"F4-6bba 0.9261 / F4-44b6 0.8515, and at 0.48 must read 0.9261 / "
      f"0.8384. 6bba is the same at both floors because ilpsweep measured "
      f"its F48 delta as +0.0000 -- which is exactly what hid this constant "
      f"from me when I first wrote it down as 0.48.", flush=True)

TMP = "/tmp/rkfloor"
pathlib.Path(TMP).mkdir(parents=True, exist_ok=True)
CELL = {}                     # (floor, tag, domain) -> aggregate row
CELL_PER = {}                 # (floor, tag, domain) -> {name: row}
PQ = {}                       # (tag, domain) -> probability quantiles
t_x = time.time()

for det_arm in DET_ARMS:
    for role, names in (("6bba", HOLDOUT_6BBA), ("44b6", SET_44B6)):
        domain = f"{det_arm}-{role}"
        for name in names:
            if (det_arm, name) not in FEATS:
                continue
            t0 = time.time()
            dets = load_dets(name, det_arm)
            T = len(dets)
            seeds, feats = FEATS[(det_arm, name)]

            # THE CALIBRATION MEASUREMENT, and it is nearly free. Both
            # rankers score the SAME gated rows, so the quantiles are
            # directly comparable. This is what could not be read off the
            # pkls: their intercepts differ by 0.016 logit (class balance,
            # and in the wrong direction), so if calibration is the story it
            # lives in the trees.
            for _tag, (_m, _c) in _PKL.items():
                _pp = np.concatenate(
                    [_m.predict_proba(r[:, _c])[:, 1]
                     for _p_, r, _P, _C in feats if len(r)]) \
                    if any(len(r) for _p_, r, _P, _C in feats) \
                    else np.zeros(0)
                PQ.setdefault((_tag, domain), []).append(_pp)

            cells = []
            for fl in FLOORS:
                for _tag, (_m, _c) in _PKL.items():
                    n0, e0, p0 = solve_one(dets, T, seeds, feats, _m, _c,
                                           floor=fl)
                    outp = (pathlib.Path(TMP) / f"{fl}_{_tag}_{domain}"
                            / f"{name}.geff")
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    patch_final(n0, e0, p0, outp)
                    row = score_one(outp, name)
                    CELL_PER.setdefault((fl, _tag, domain), {})[name] = row
                    cells.append(f"{fl:.2f}/{_tag[3:]:<4}{_adj_of(row):.4f}")
                    del n0, e0, p0
            print(f"  {name} " + "  ".join(cells)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            del dets
            FEATS.pop((det_arm, name), None)
            gc.collect()

for key, per in CELL_PER.items():
    CELL[key] = _with_ratio(list(per.values()))

print(f"\n=== PROBABILITY QUANTILES on the SAME gated rows "
      f"(the calibration the pkls could not show) ===", flush=True)
print(f"  {'domain':<12}{'ranker':<10}{'rows':>10}" +
      "".join(f"{f'p{q}':>9}" for q in (5, 25, 50, 75, 95)) +
      f"{'>=0.10':>9}{'>=0.48':>9}", flush=True)
for domain in sorted({d for _t, d in PQ}):
    for _tag, _ in PKL_ARMS:
        v = np.concatenate(PQ[(_tag, domain)])
        print(f"  {domain:<12}{_tag:<10}{len(v):>10,}" +
              "".join(f"{np.percentile(v, q):>9.4f}"
                      for q in (5, 25, 50, 75, 95)) +
              f"{np.mean(v >= 0.10):>9.4f}{np.mean(v >= 0.48):>9.4f}",
              flush=True)

print(f"\n=== adj_J, floor x ranker ({(time.time() - t_x) / 60:.1f} min) ===",
      flush=True)
FLOOR_FAIL = []
for domain in sorted({d for _f, _t, d in CELL}):
    print(f"  -- {domain} --", flush=True)
    print(f"  {'floor':>7}" + "".join(f"{t:>12}" for t, _ in PKL_ARMS)
          + f"{'F4 - BASE':>12}", flush=True)
    for fl in FLOORS:
        a = CELL.get((fl, "v4_BASE", domain))
        b = CELL.get((fl, "v4_F4", domain))
        if a is None or b is None:
            continue
        va, vb = float(a["adj_edge_jaccard"]), float(b["adj_edge_jaccard"])
        mark = ""
        want = FLOOR_ANCHOR.get((fl, "v4_BASE", domain))
        if want is not None:
            ok = abs(va - want) <= FLOOR_TOL
            mark = f"   ANCHOR want {want:.4f} -> {'OK' if ok else 'MISS'}"
            if not ok:
                FLOOR_FAIL.append((fl, domain, va, want))
        print(f"  {fl:>7.2f}{va:>12.4f}{vb:>12.4f}{vb - va:>+12.4f}{mark}",
              flush=True)

print(f"\n=== READING ===", flush=True)
print("  Fixed before the numbers: if the retrain's loss is the OPERATING "
      "POINT, the F4 ranker has its own best floor and (F4 - BASE) shrinks "
      "or turns positive somewhere in the sweep. If the loss is REAL, "
      "(F4 - BASE) stays negative at every floor. A third outcome the "
      "pre-registration should allow: both rankers may prefer a floor "
      "neither eprank3 nor production uses, which would be a finding about "
      "the floor and not about the retrain.", flush=True)
print("  ⚠ LOCAL ONLY. Floor 0.48 ships because it WON on LB (+0.001, "
      "submit-f48, -1.43% bytes) while reading +0.0000 / -0.0131 locally, "
      "so local and LB already disagree about this knob. Nothing here can "
      "pick the shipped floor; it can only say whether the retrain's loss "
      "survives re-tuning.", flush=True)

for _k, _per in sorted(CELL_PER.items()):
    fl, _tag, domain = _k
    _csv = pathlib.Path("/kaggle/working") / f"rkfloor_{fl:.2f}_{_tag}_{domain}.csv"
    _keys = sorted({k for r in _per.values() for k in r})
    with open(_csv, "w", encoding="utf-8", newline="\n") as _fh:
        _fh.write("dataset," + ",".join(_keys) + "\n")
        for _n in sorted(_per):
            _fh.write(f"{_n}," + ",".join(repr(_per[_n].get(_k2))
                                          for _k2 in _keys) + "\n")
print(f"\n  {len(CELL_PER)} per-dataset CSVs written; every cell is paired "
      f"by dataset so any two can be bootstrapped against each other.",
      flush=True)

assert not FLOOR_FAIL, (
    f"the v4_BASE anchors did not reproduce: {FLOOR_FAIL}. The floor sweep "
    f"machinery is wrong and the F4 column cannot be read against it.")
print("\nRKFLOOR OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "rkfloor.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
