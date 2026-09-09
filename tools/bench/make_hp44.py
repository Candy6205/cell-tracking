"""hparabench_ref.py -> hp44.py: the hpara arm on 44b6, the missing domain.

WHY IT WAS MISSING, and I told the user the wrong thing about it. divbench9
defines `GROUPS = (("6bba", HOLDOUT_6BBA), ("44b6", SET_44B6))` and
`DOMAINS`, and **neither is ever used** -- dead code inherited from an
ancestor. The bench scores 6bba and nothing else. `SET_44B6` exists in it
only to hold 12 movies OUT of the ranker's training pool and to feed the
joint fork gate. So after `hparabench` round 2 the score of the hpara
placement stood at:

    6bba      +0.0031 ungated over 128 movies, 90% [+0.0014, +0.0049]
    44b6      never measured
    zebrahub  zhpara, running

and this project's own rule is three domains or it does not count.

THE BLOCKER, and the way round it. Stage B's two fold rankers are each
trained on *all 71* 44b6 movies plus the other 64 6bba ones, so every 44b6
evaluation through them is IN-SAMPLE -- which is exactly why divbench9's own
44b6 anchors are marked soft. But `MODEL_BIDIR`
(`edge_ranker_ep175v3_bidir.pkl`, pool D175 = 199 - 12 6bba - 12 44b6) has
NOT seen the 12 evaluation movies, and it is the production pickle. So the
44b6 arm runs the PRODUCTION ranker at the PRODUCTION configuration
(disappearance 1.5, floor 0.48) on 12 movies it never saw.

ANCHOR: the F4 arm must reproduce divbench9's recorded `+BIDIR F4-44b6` row,
adj_J **0.8515** with div_J 0.0000 (44b6's 12 movies contain no GT division
this pipeline recovers). Asserted for DET_SRC=F4, printed and expected to
move for F4HP -- the move is the measurement.

WHAT IS CUT: stage B's 128-movie 6bba loop, the DIV_W arms, the fork tables
and the gate cross. All of it is about 6bba and all of it is already done.
Stage A is kept whole, because it is what computes the F4 features for the
12 44b6 movies in the first place.

RUN IT TWICE, once per DET_SRC, and pair the per-movie CSVs -- same movies,
same order, so the same paired bootstrap `tools/hparapair2.py` does for 6bba
works on these.

Attach: hparabench's sources. For DET_SRC=F4HP add biohub-hparacache,
biohub-ephpcache, biohub-ep2hpcache. CPU only.
"""
import os

src = open("hparabench_ref.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-hparabench"',
    'SCRIPT_VERSION = "2026-09-09a-hp44"')

_CUT = 'print(f"\\n=== STAGE B: {len(SIX)} 6bba movies on F4 primaries'
assert src.count(_CUT) == 1
src = src[:src.index(_CUT)]

src += r'''# ---- 44b6, the domain this bench has never scored ----------------------
#
# The production ranker on the 12 movies it was held out of. Both arms run
# byte-identical code; the only difference is which detection cache
# `load_dets(name, "F4")` opens, which DET_SRC decided at the top.
ANCHOR_44B6_ADJ = 0.8515        # divbench9 ANCHOR_SOFT ("+BIDIR", "F4", "44b6")
ANCHOR_44B6_DIV = 0.0000
ANCHOR_44B6_TOL = 0.0005

TMP = "/tmp/hp44"
pathlib.Path(TMP).mkdir(parents=True, exist_ok=True)

print(f"\n=== 44b6: {len(SET_44B6)} movies, PRODUCTION ranker "
      f"(edge_ranker_ep175v3_bidir, pool D175) at the production "
      f"configuration (disappearance {ILP_DISAPP}, floor {PROD_FLOOR}), "
      f"DET_SRC={DET_SRC} ===", flush=True)
print("  Stage B's two fold rankers each train on ALL 71 44b6 movies, so a "
      "44b6 row through them is in-sample -- which is why divbench9's 44b6 "
      "anchors are soft and why it never scored the domain. D175 held these "
      "12 out, so this arm is clean.", flush=True)
print(f"  ANCHOR (DET_SRC=F4 only): adj_J {ANCHOR_44B6_ADJ:.4f}, div_J "
      f"{ANCHOR_44B6_DIV:.4f}. For F4HP it is printed and expected to move; "
      f"the move is the measurement.", flush=True)

ROWS44_OUT = []
PER44 = {}
t44 = time.time()
for name in SET_44B6:
    if time.time() - t_all > BUDGET_S:
        print(f"  BUDGET reached; {len(SET_44B6) - len(ROWS44_OUT)} movies "
              f"not attempted", flush=True)
        break
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    n0, e0, p0, _aux = solve_one(dets, T, seeds, feats, MODEL_BIDIR,
                                 MCOLS_BIDIR, disapp=ILP_DISAPP,
                                 floor=PROD_FLOOR)
    outp = pathlib.Path(f"{TMP}/{DET_SRC}") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    PATCH_CAPS.clear()
    patch_final(n0, e0, p0, outp)
    row = score_one(outp, name)
    ROWS44_OUT.append(row)
    PER44[name] = row
    print(f"  [{len(ROWS44_OUT)}/{len(SET_44B6)}] {name}: T={T} "
          f"nodes={sum(len(c) for c, _s in dets):,} "
          f"adj_J {_adj_of(row):.4f} div {row['division_tp']}/"
          f"{row['division_fp']}/{row['division_fn']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del dets, seeds, feats, n0, e0, p0, _aux
    gc.collect()

assert ROWS44_OUT, "no 44b6 movie was scored"
R44 = _with_ratio(ROWS44_OUT)
print(f"\n=== 44b6 AGGREGATE, {len(ROWS44_OUT)} movies, DET_SRC={DET_SRC} "
      f"({(time.time() - t44) / 60:.1f} min) ===", flush=True)
for k in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard",
          "total_node_ratio", "score"):
    v = R44.get(k, float("nan"))
    print(f"  {k:<22}{float(v):>12.4f}", flush=True)
print(f"  div TP/FP/FN "
      f"{sum(r['division_tp'] for r in ROWS44_OUT)}/"
      f"{sum(r['division_fp'] for r in ROWS44_OUT)}/"
      f"{sum(r['division_fn'] for r in ROWS44_OUT)}", flush=True)

_got = float(R44["adj_edge_jaccard"])
_hit = abs(_got - ANCHOR_44B6_ADJ) <= ANCHOR_44B6_TOL
if DET_SRC == "F4":
    print(f"\n  ANCHOR +BIDIR F4-44b6 adj_J {_got:.4f} want "
          f"{ANCHOR_44B6_ADJ:.4f} -> {'OK' if _hit else 'MISS'} "
          f"({_got - ANCHOR_44B6_ADJ:+.4f})", flush=True)
    assert _hit, (f"the 44b6 control does not reproduce divbench9's recorded "
                  f"+BIDIR row ({_got:.4f} vs {ANCHOR_44B6_ADJ:.4f}); the "
                  f"F4HP arm would not be readable against it")
else:
    print(f"\n  [expected] +BIDIR F4-44b6 adj_J {_got:.4f} against the F4 "
          f"arm's anchor {ANCHOR_44B6_ADJ:.4f} ({_got - ANCHOR_44B6_ADJ:+.4f})"
          f" -- this arm changes the coordinates, so the move IS the "
          f"measurement. Pair the CSVs; a single aggregate cannot say "
          f"whether one movie carries it.", flush=True)

_csv = pathlib.Path("/kaggle/working") / f"hp44_{DET_SRC}.csv"
_keys = sorted({k for r in PER44.values() for k in r})
with open(_csv, "w", encoding="utf-8", newline="\n") as _fh:
    _fh.write("dataset,fold," + ",".join(_keys) + "\n")
    for _n in sorted(PER44):
        _fh.write(f"{_n},-1," + ",".join(repr(PER44[_n].get(_k))
                                         for _k in _keys) + "\n")
print(f"\n  {_csv.name} written ({len(PER44)} movies, {len(_keys)} columns). "
      f"Same movies and same order in both arms, so tools/hparapair2.py "
      f"pairs them directly.", flush=True)

if len(ROWS44_OUT) == len(SET_44B6):
    print("HP44 OK", flush=True)
else:
    print(f"HP44 INCOMPLETE ({len(ROWS44_OUT)} of {len(SET_44B6)})",
          flush=True)
'''

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for fname, det in (("hp44.py", "F4HP"), ("hp44_ref.py", "F4")):
    body = src.replace('DET_SRC = "F4"', f'DET_SRC = "{det}"', 1)
    assert body.count(f'DET_SRC = "{det}"') >= 1, det
    open(os.path.abspath(os.path.join(base, fname)), "w",
         encoding="utf-8", newline="\n").write(body)
    print("written", fname, len(body), "DET_SRC", det)
