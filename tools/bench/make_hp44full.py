"""divbench5.py -> hp44full.py / hp44full_ref.py: hpara on ALL 71 44b6 movies.

The user, on being handed 44b6 as a "null" measured on a 12-movie draw whose
div_J was 0.0000 in both arms: *"思考，把44b6全跑一遍不行吗，你这分裂全挂0"*.

WHY THE 12-MOVIE READ CANNOT SETTLE 44b6, in this record's own numbers. The
seeded draw (`SET_44B6`) contains **2 GT divisions**; both were FN in both
arms, so div_J was 0.0000 either way and the domain's score WAS its adj_J.
All 71 carry **26** (progress.md:3532). And `divbench5` v2 already showed the
small draw misreads this domain on exactly this kind of quantity: divbench3's
12 said the fork gate was worth +0.0020 adj_J on 2 GT divisions; the full 71
said **+0.0014 on 26**.

WHY divbench5 IS THE RIGHT ANCESTOR, and hp44 was not. `hp44` ran the
production `MODEL_BIDIR`, which has 59 of the 71 in its pool -- fine for the
12 held out of it, impossible for the rest. divbench5 already solves that:
two out-of-fold rankers over 44b6, each trained on all 128 6bba plus the
other half of 44b6, so **every one of the 71 is out of fold.** It also
already runs all 71 (117.9 min, both hard anchors exact).

THE TWO ARMS. `DET_SRC` picks which detection and head caches stage B reads
for the 71; everything else -- the fold rankers, their BASE-primary training
rows, the ILP configuration, the patches, the scorer -- is identical.

    hp44full_ref.py   DET_SRC = "F4"     refine placement
    hp44full.py       DET_SRC = "F4HP"   hpara placement

THE OVERRIDE REPLACES, IT NEVER ADDS. hparabench v6 died on this: merging
`fdet_F4hp_` into `fdet_F4_` gave the hpara arm coverage the control lacked
and the arms stopped being the same experiment. Here the hpara caches must
cover all 71 BENCH movies and nothing outside them, asserted both ways.

ANCHORS. The control arm carries three, and the strongest is not one of
divbench5's own: **REF over all 71 must read `6/61/20`, div_J 0.0690,
adj_J 0.8868, score 0.8937** -- divbench5 v2's recorded REF row, on 71 movies
rather than 12. Plus the two inherited 12-movie ones (ANCHOR_D187 at floor
0.10 = 0.8353 hard, +BIDIR pkl at floor 0.48 = 0.8384 soft). In the F4HP arm
all three move by construction and are printed-and-warned rather than
asserted; the move is the measurement.

WHAT IS ADDED: a per-movie CSV (`hp44full_<DET_SRC>.csv`). divbench5 keeps
`PER[name]` for all 71 and writes only fork tables, so the two arms could not
be paired. With the CSV, `tools/hparapair2.py`'s aggregation and paired
bootstrap read them directly -- same movies, same order.

Blocked on `ephp44` (the head over hpara44's 59 new primaries). Attach:
divbench5's sources PLUS faustine649/biohub-hpara44, biohub-hparacache,
biohub-ephp44 and biohub-ephpcache. CPU only, ~2 h per arm.
"""
import os

src = open("divbench5.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07b-divbench5"',
    'SCRIPT_VERSION = "2026-09-09a-hp44full"\n'
    '# Which placement stage B solves on the 71. The fold rankers train on\n'
    '# BASE primaries in BOTH arms, exactly as divbench5 and hparabench do --\n'
    '# and eprank3 (2026-09-09) measured that matching them to the deployment\n'
    '# coordinates LOSES, so this is the better configuration as well as the\n'
    '# consistent one.\n'
    'DET_SRC = "F4"')

# ---- split the hpara head caches out BEFORE _EP_ALL is deleted --------
# `edgeprob_F4hp_<m>.npz` matches the "edgeprob_" prefix, and "F4hp_" does not
# start with "F4_", so divbench5's split would drop every one of them into the
# BASE dict under a key nothing ever looks up -- the trap ttabench documented.
# Pulled out here, and removed from EDGEPROB, whether or not this arm uses it.
rep('''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}
del _EP_ALL''',
    '''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}
_HP_EP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()
          if k.startswith("F4hp_")}
EDGEPROB = {k: v for k, v in EDGEPROB.items() if not k.startswith("F4hp_")}
del _EP_ALL''')

# ---- the override, replace-only, over exactly the 71 -------------------
rep('''FDET_F4 = _walk_files(_npz_prefix("fdet_F4_"))
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"''',
    '''FDET_F4 = _walk_files(_npz_prefix("fdet_F4_"))
_HP_DET = _walk_files(_npz_prefix("fdet_F4hp_"))
print(f"fdet_F4 caches: {len(FDET_F4)}; fdet_F4hp {len(_HP_DET)}; "
      f"edgeprob_F4hp {len(_HP_EP)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"
if DET_SRC == "F4HP":
    # THE OVERRIDE REPLACES, IT NEVER ADDS. hparabench v6 died on the other
    # rule: merging the hpara caches in gave the hpara arm 116 movies of
    # coverage the control did not have, and the two arms stopped being the
    # same experiment. Restricted to keys that already exist, and the 71
    # BENCH movies are asserted complete further down, once BENCH is defined.
    _dov = sorted(set(FDET_F4) & set(_HP_DET))
    _eov = sorted(set(EDGEPROB_F4) & set(_HP_EP))
    FDET_F4 = {**FDET_F4, **{k: _HP_DET[k] for k in _dov}}
    EDGEPROB_F4 = {**EDGEPROB_F4, **{k: _HP_EP[k] for k in _eov}}
    print(f"  hpara override: {len(_dov)} detection and {len(_eov)} head "
          f"entries REPLACED; {len(set(_HP_DET) - set(_dov))} hpara detection "
          f"caches ignored because the refine arm has none for those movies",
          flush=True)''')

# ---- assert the 71 are complete, where BENCH exists --------------------
rep('''_missing = [n for n in BENCH if n not in EDGEPROB_F4]
assert not _missing, (f"no edgeprob F4 cache for {len(_missing)} 44b6 movies: "
                      f"{_missing[:5]} -- attach biohub-ep44b6 AND biohub-edgeprob")''',
    '''_missing = [n for n in BENCH if n not in EDGEPROB_F4]
assert not _missing, (f"no edgeprob F4 cache for {len(_missing)} 44b6 movies: "
                      f"{_missing[:5]} -- attach biohub-ep44b6 AND biohub-edgeprob")
if DET_SRC == "F4HP":
    _nd = [n for n in BENCH if n not in _HP_DET]
    assert not _nd, (
        f"{len(_nd)} of the {len(BENCH)} 44b6 movies have no fdet_F4hp_ cache "
        f"({_nd[:4]}) -- attach faustine649/biohub-hpara44 (the 59) AND "
        f"faustine649/biohub-hparacache (the 12 of the eval draw). A partial "
        f"override would score some movies on hpara and some on refine and "
        f"still print a plausible aggregate.")
    _ne = [n for n in BENCH if n not in _HP_EP]
    assert not _ne, (
        f"{len(_ne)} of the {len(BENCH)} 44b6 movies have no edgeprob_F4hp_ "
        f"cache ({_ne[:4]}) -- attach faustine649/biohub-ephp44 AND "
        f"faustine649/biohub-ephpcache. The head MUST be re-run on hpara "
        f"points: GATE_UM = 10.0 gates candidate pairs by DISTANCE, and a "
        f"median 0.43 um move walks pairs across it in both directions, so "
        f"the refine rows do not join.")
    print(f"  all {len(BENCH)} 44b6 movies carry hpara primaries AND hpara "
          f"head probabilities: this arm is hpara on the whole domain, "
          f"26 GT divisions instead of the 12-movie draw's 2.", flush=True)''')

# ---- the anchors that cannot hold in the hpara arm ---------------------
rep('''print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)''',
    '''# divbench5 v2's RECORDED REF ROW over all 71, which is a stronger anchor
# than either 12-movie one because it is the same population this run
# measures. Hard on the control, printed-and-warned on the hpara arm.
ANCHOR71 = dict(tp=6, fp=61, fn=20, div_J=0.0690, adj_J=0.8868, score=0.8937)
_g71 = (TP, FP, FN, base_J, float(RES["adj_edge_jaccard"]),
        float(RES["score"]))
_w71 = (ANCHOR71["tp"], ANCHOR71["fp"], ANCHOR71["fn"], ANCHOR71["div_J"],
        ANCHOR71["adj_J"], ANCHOR71["score"])
_hit71 = (_g71[:3] == _w71[:3]
          and all(abs(a - b) <= ANCHOR_TOL for a, b in zip(_g71[3:], _w71[3:])))
if DET_SRC == "F4":
    print(f"  ANCHOR71 REF over all {len(ROWS)}: {TP}/{FP}/{FN} div_J "
          f"{base_J:.4f} adj_J {_g71[4]:.4f} score {_g71[5]:.4f}  want "
          f"6/61/20 0.0690 0.8868 0.8937 (divbench5 v2) -> "
          f"{'OK' if _hit71 else 'MISS'}", flush=True)
    if not _hit71:
        ANCHOR_FAIL.append(("REF71", "F4-44b6", _g71[4], ANCHOR71["adj_J"]))
else:
    print(f"  [expected] REF over all {len(ROWS)}: {TP}/{FP}/{FN} div_J "
          f"{base_J:.4f} adj_J {_g71[4]:.4f} score {_g71[5]:.4f} against the "
          f"control's 6/61/20 0.0690 0.8868 0.8937. This arm changes the "
          f"coordinates, so the move IS the measurement. Pair the CSVs; an "
          f"aggregate cannot say whether one movie carries it.", flush=True)
    # The two 12-movie anchors are built on SET_44B6, whose coordinates this
    # arm also moves. Demote them the way hparabench demotes its own.
    _moved = [f for f in ANCHOR_FAIL if f[0] in ("ANCHOR_D187", "REF71")]
    ANCHOR_FAIL = [f for f in ANCHOR_FAIL if f not in _moved]
    for _f in _moved:
        print(f"  [expected] {_f[0]} {_f[1]} moved to {_f[2]:.4f} from "
              f"{_f[3]:.4f} -- it is built on movies this arm re-places",
              flush=True)
print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)

# ---- the per-movie CSV divbench5 never wrote --------------------------
# It keeps PER[name] for all 71 and writes only the fork tables, so the two
# arms could not be paired. tools/hparapair2.py's aggregation and paired
# bootstrap read this directly: same movies, same order in both arms.
_csv = pathlib.Path("/kaggle/working") / f"hp44full_{DET_SRC}.csv"
_keys = sorted({k for r in PER.values() for k in r})
with open(_csv, "w", encoding="utf-8", newline="\\n") as _fh:
    _fh.write("dataset,fold," + ",".join(_keys) + "\\n")
    for _n in sorted(PER):
        _fh.write(f"{_n},{FOLD_OF[_n]}," +
                  ",".join(repr(PER[_n].get(_k)) for _k in _keys) + "\\n")
print(f"  {_csv.name} written ({len(PER)} movies, {len(_keys)} columns)",
      flush=True)''')

rep('print("DIVBENCH5 OK", flush=True)', 'print("HP44FULL OK", flush=True)')
rep('print(f"[degraded] DIVBENCH5 INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)',
    'print(f"[degraded] HP44FULL INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)')
rep('print(f"\\nDIVBENCH5 DONE in', 'print(f"\\nHP44FULL DONE in')

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for fname, det in (("hp44full.py", "F4HP"), ("hp44full_ref.py", "F4")):
    body = src.replace('DET_SRC = "F4"', f'DET_SRC = "{det}"', 1)
    assert body.count(f'DET_SRC = "{det}"') >= 1, det
    open(os.path.abspath(os.path.join(base, fname)), "w",
         encoding="utf-8", newline="\n").write(body)
    print("written", fname, len(body), "DET_SRC", det)
