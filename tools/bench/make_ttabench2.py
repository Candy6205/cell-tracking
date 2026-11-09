"""rkpool.py -> ttabench2.py: the MATCHED edge-feature TTA test.

WHAT IS OPEN. `ttabench` (2026-09-09) measured tta4 through the whole
pipeline at adj_J -0.0051 / edge_J -0.0074, both intervals excluding zero --
but its fold rankers were trained on noTTA rows and deployed on tta4 rows,
so the number is "TTA plus a train/deploy mismatch" with no way to split the
two. The matched test trains the ranker on tta4 rows and deploys it on tta4
rows. `ttacache` (140 movies) and `ttacache2` (59 movies) cached the first
head at tta4 for all 199 train movies, both COMPLETE 2026-09-09, both with
the (t, i, j) row-set anchor against edgeprob_F4_ passing on every movie.
Nothing has read them.

WHAT THIS IS. rkpool.py (RANK_TRAIN="F4", DET_SRC="F4") with ONE change:
every `edgeprob_F4_<name>.npz` is replaced by `edgeprob_tta4_<name>.npz`,
all 199. RANK_TRAIN="F4" builds LAB4 from EP_OF["F4"] over all 199 movies,
so the fold rankers now learn on tta4 columns, and stage B solves the same
128 6bba movies on tta4 columns. Detections are byte-identical (ttaprobe
held det_logits at view 0), the (t, i, j) rows are identical, only p_fwd /
p_rev differ.

CONTROL. rkpool's F4 arm: F4-trained on noTTA, deployed on noTTA --
52/225/73, div_J 0.1486, adj_J 0.9141, score 0.9290, per-movie CSV
`bench_csv/rkpool_F4.csv`. Same 128 movies, same folds, same stage B.
Pair with `tools/rkpoolpair.py bench_csv/rkpool_F4.csv <this csv>`.

ANCHORS. ANCHOR_D187 loads a fixed 18-column pkl that never reads the head,
so 0.9216 on the 12 holdout movies must reproduce EXACTLY -- that is what
licenses pairing against rkpool_F4.csv. The +BIDIR pkl reads p_fwd / p_rev /
h_fused and WILL move on tta4 columns; it is soft and printed. REF_FORKS is
demoted by RANK_TRAIN != "BASE" as in rkpool.

NOT COVERED. The second head (edgeprob2_F4_, 24 movies) stays single-view in
both arms; ttacache cached the first head only. Same scope as ttabench.

Attach: rkpool's 12 kernel sources plus faustine649/biohub-ttacache and
faustine649/biohub-ttacache2. CPU, ~3.3 h (rkpool's F4 arm took 194.7 min).
"""
import os

src = open("rkpool.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


assert 'RANK_TRAIN = "F4"' in src, "cut from rkpool.py (the F4 arm), not the ref"
assert 'DET_SRC = "F4"\nassert DET_SRC in ("F4", "F4HP"), DET_SRC' in src

rep('SCRIPT_VERSION = "2026-09-09a-rkpool"', 'SCRIPT_VERSION = "2026-09-11a-ttabench2"')

# ---- 1. swap the first head's caches for the tta4 ones, all 199 ---------
rep('''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}
del _EP_ALL''',
    '''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}

# ttacache / ttacache2 write edgeprob_tta4_<movie>.npz. Those match the
# "edgeprob_" prefix and would otherwise land in the BASE dict under a key
# nothing looks up. Split them out, then REPLACE the F4 head with them.
EP_TTA4 = {k[len("tta4_"):]: v for k, v in _EP_ALL.items()
           if k.startswith("tta4_")}
EDGEPROB = {k: v for k, v in EDGEPROB.items() if not k.startswith("tta4_")}
del _EP_ALL
assert len(EP_TTA4) == 199, (
    f"{len(EP_TTA4)} tta4 caches, want 199 -- attach both "
    f"faustine649/biohub-ttacache (140) and faustine649/biohub-ttacache2 (59)")
assert set(EP_TTA4) == set(EDGEPROB_F4), (
    f"tta4 and F4 head caches cover different movies: only-tta4 "
    f"{sorted(set(EP_TTA4) - set(EDGEPROB_F4))[:4]}, only-F4 "
    f"{sorted(set(EDGEPROB_F4) - set(EP_TTA4))[:4]}")
# THE ANCHOR THAT MAKES THE JOIN LEGAL: every tta4 cache must hold the SAME
# (t, i, j) rows as its edgeprob_F4_ twin (ttaprobe holds det_logits at
# view 0, so the detections and the gate cannot move). Both cache kernels
# checked this against the F4 caches they could see; this re-checks all 199
# against the F4 caches THIS harness reads, before any CPU is spent.
_rows_bad, _worst_dp, _n_same_p = [], 0.0, 0
for _m in sorted(EP_TTA4):
    _a, _b = np.load(EP_TTA4[_m]), np.load(EDGEPROB_F4[_m])
    if not (np.array_equal(_a["t"], _b["t"]) and np.array_equal(_a["i"], _b["i"])
            and np.array_equal(_a["j"], _b["j"])):
        _rows_bad.append(_m)
    else:
        _d = max(float(np.abs(np.asarray(_a[c], float)
                              - np.asarray(_b[c], float)).max())
                 for c in ("p_fwd", "p_rev"))
        _worst_dp = max(_worst_dp, _d)
        _n_same_p += int(_d == 0.0)
    _a.close(); _b.close()
assert not _rows_bad, (
    f"{len(_rows_bad)} tta4 caches have a different (t, i, j) row set from "
    f"their edgeprob_F4_ twin ({_rows_bad[:4]}) -- the join would be wrong")
print(f"  ANCHOR tta4 row sets: 199 of 199 movies carry exactly the "
      f"edgeprob_F4_ rows; worst |dp| over p_fwd/p_rev {_worst_dp:.3e}; "
      f"{_n_same_p} movies with probabilities unchanged (want 0: a tta4 cache "
      f"identical to noTTA would mean the average was never applied)",
      flush=True)
assert _n_same_p == 0, f"{_n_same_p} tta4 caches are bit-identical to noTTA"
EDGEPROB_F4 = EP_TTA4
print("  edgeprob: FIRST head switched to tta4 on all 199 movies. With "
      "RANK_TRAIN=F4 the fold rankers TRAIN on tta4 rows (LAB4 over all 199) "
      "and stage B DEPLOYS them on tta4 rows -- the matched test ttabench "
      "could not run. The SECOND head (edgeprob2_F4_, 24 movies) stays "
      "single-view in both arms; ttacache cached the first head only.",
      flush=True)''')

# ---- 2. the +BIDIR pkl anchor reads the head and will move: say so ------
rep('''    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.9261 soft (ilpsweep F48) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.9261) <= ANCHOR_TOL else 'MISS'}",
          flush=True)''',
    '''    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.9261 soft (ilpsweep F48) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.9261) <= ANCHOR_TOL else 'MISS'}"
          f"  [this pkl reads p_fwd/p_rev/h_fused, which are tta4 here, so a "
          f"move is expected: it is the shipped noTTA-trained ranker on tta4 "
          f"columns, ttabench's unmatched case on 12 movies]",
          flush=True)''')

# ---- 3. print what the REF row is paired against ------------------------
rep('''print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)''',
    '''print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)
print(f"  THE MATCHED TTA TEST. Compare the REF row above against, on these "
      f"same 128 movies:\\n"
      f"      (F4 ranker on noTTA, deployed noTTA)  52/225/73  div_J 0.1486  "
      f"adj_J 0.9141  score 0.9290   rkpool (the control, rkpool_F4.csv)\\n"
      f"      (BASE ranker on noTTA, deployed noTTA) 50/249/75  div_J 0.1337  "
      f"adj_J 0.9133  score 0.9266   rkpool_ref\\n"
      f"      ttabench's UNMATCHED tta4 on 12 movies: adj_J -0.0051, "
      f"edge_J -0.0074 against a noTTA-trained ranker.\\n"
      f"      (F4 ranker on tta4, deployed tta4)    <- this run.", flush=True)''')

# ---- 4. the CSV must not overwrite rkpool's -----------------------------
rep('f"rkpool_{RANK_TRAIN}.csv"', 'f"ttabench2_{RANK_TRAIN}_tta4.csv"')
rep('print("HPARABENCH OK", flush=True)', 'print("TTABENCH2 OK", flush=True)')
rep('print(f"[degraded] HPARABENCH INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)',
    'print(f"[degraded] TTABENCH2 INCOMPLETE: {len(ROWS)}/{len(SIX)}", flush=True)')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ttabench2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written ttabench2.py", len(src), "DET_SRC F4, RANK_TRAIN F4, first head tta4 x199")
