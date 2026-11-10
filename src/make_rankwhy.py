"""Generate rankwhy.py from edgewhy.py by asserted replacement.

THE QUESTION. edgewhy measured that the edge ranker is effectively
binary: on 6bba the correct pair is its top choice 98.26% of the time
and those edges are lost 0.81% of the time, while a correct pair sitting
at rank 1 is lost 84.93% of the time with p_true 0.148 against p_top
0.793. Roughly a quarter of every lost edge on 6bba (149 of 598) is that
one event -- the ranker put something else first.

Nothing on record says WHAT it put first. Two possibilities with
opposite repairs:

  * the preferred target is a SPURIOUS detection -- a peak with no cell
    under it. Then detection precision is the lever, and every node the
    detector invents is a distractor that can outrank a real successor.
  * the preferred target is a REAL neighbouring cell, usually the true
    successor of some OTHER source. Then it is a genuine assignment
    ambiguity, the ILP's global solve is the only thing that could break
    it, and it did not.

Zebrahub decides which: its GT covers every cell, so "is that peak a
cell" is answerable there. On 6bba and 44b6 an unannotated preferred
target is recorded as unknown rather than guessed.

SECOND BLOCK, same run: `node_absent`. On 6bba 73 lost edges (12% of the
loss) have both endpoints detected AND the correct pair as the ranker's
TOP choice at p_true 0.56, and the source still never reached the solved
graph -- the ILP dropped it. That bucket has never been opened. This
records, for each such node, its best outgoing and incoming probability
and how many other sources were bidding for the same target.

Same harness, same three domains, same anchors as edgewhy: F4-6bba
adj_J 0.9216, F4-44b6 0.8353, F4-zh 0.8118.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "edgewhy.py"
DST = pathlib.Path(__file__).parent / "rankwhy.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""edgewhy -- under what conditions does F4 lose an edge or link the '
    'wrong pair?',
    '"""rankwhy -- what does the edge ranker prefer when it puts the right '
    'target second?')
rep('SCRIPT_VERSION = "2026-09-02a-edgewhy"',
    'SCRIPT_VERSION = "2026-09-02b-rankwhy"')

# ---- 1. the inverse maps the two blocks need ---------------------------
rep('''    a = AGG.setdefault(domain, {c: 0 for c in CATS})''',
    r'''    # ---- rankwhy: primaries backwards to GT, and GT targets as a set --
    seed_gt = {ti_: r_ for r_, ti_ in m_seed.items()}
    gt_is_target = set()
    for u_, v_ in gt_edges:
        if int(gt_t[int(v_)]) == int(gt_t[int(u_)]) + 1:
            gt_is_target.add(int(v_))
    # how many sources at t list a given target j at t+1 as a candidate
    bidders = defaultdict(int)
    best_in_of = {}
    for (t_, i_, j_), p_ in prob_of.items():
        bidders[(t_ + 1, j_)] += 1
        k_ = (t_ + 1, j_)
        if k_ not in best_in_of or p_ > best_in_of[k_]:
            best_in_of[k_] = float(p_)

    a = AGG.setdefault(domain, {c: 0 for c in CATS})''')

# ---- 2. record what the ranker preferred -------------------------------
rep('''        erows.append((float(CATS.index(bucket)),''',
    r'''        # ---- what did the ranker put first instead? -------------------
        if su is not None and sv is not None and np.isfinite(rank_t) \
                and rank_t >= 1.0:
            lst2 = cand_by_src.get((tu, su[1]))
            j_top = lst2[0][1]
            tgt_gt = seed_gt.get((tv, j_top))
            P_ = seeds[tu][0]
            C_ = seeds[tv][0]
            src_um = np.asarray(P_[su[1]], float) * SCALE
            top_um = np.asarray(C_[j_top], float) * SCALE
            true_um = np.asarray(C_[sv[1]], float) * SCALE
            RROWS.setdefault(domain, []).append((
                float(rank_t),
                float(p_true) if np.isfinite(p_true) else np.nan,
                float(p_top) if np.isfinite(p_top) else np.nan,
                1.0 if tgt_gt is not None else 0.0,
                1.0 if (tgt_gt is not None
                        and tgt_gt in gt_is_target) else 0.0,
                float(np.linalg.norm(top_um - src_um)),
                float(np.linalg.norm(true_um - src_um)),
                float(np.linalg.norm(top_um - true_um)),
                float(seeds[tv][1][j_top]),
                float(seeds[tv][1][sv[1]]),
                float(bidders.get((tv, j_top), 0)),
                1.0 if bucket == "recovered" else 0.0))
        # ---- a node the ILP dropped despite a top-ranked correct pair --
        if bucket == "node_absent" and su is not None:
            lst3 = cand_by_src.get((tu, su[1]), ())
            best_out = float(lst3[0][0]) if lst3 else np.nan
            best_in = best_in_of.get((tu, su[1]), np.nan)
            NROWS.setdefault(domain, []).append((
                best_out, best_in,
                float(len(lst3)),
                float(bidders.get((tu, su[1]), 0)),
                float(seeds[tu][1][su[1]]),
                float(tu) / max(1, T)))
        erows.append((float(CATS.index(bucket)),''')

# ---- 3. stores ----------------------------------------------------------
rep('''EROWS = {}         # domain -> [array of per-GT-edge covariate rows]''',
    '''EROWS = {}         # domain -> [array of per-GT-edge covariate rows]
RROWS = {}         # domain -> one row per GT edge whose correct pair
                   # was NOT the ranker's first choice
NROWS = {}         # domain -> one row per node_absent source
RCOLS = ["rank", "p_true", "p_top", "top_is_gt", "top_is_gt_target",
         "d_src_top", "d_src_true", "d_top_true", "sc_top", "sc_true",
         "bidders_for_top", "survived"]
NCOLS = ["best_out", "best_in", "n_cand", "bidders_for_self",
         "sc_self", "t_frac"]''')

# ---- 4. the report ------------------------------------------------------
rep('''print(f"\\nEDGEWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
print("\n\n=== WHAT THE RANKER PREFERRED INSTEAD ===", flush=True)
print("  One row per GT edge whose correct pair was NOT the ranker's "
      "first choice. `top is a cell` = the preferred target matched an "
      "annotated GT node; on 6bba and 44b6 most cells are unannotated so "
      "this is a LOWER BOUND there and only zebrahub measures it. `top "
      "is another cell's successor` = that GT node is the target of some "
      "GT edge, i.e. the ranker tried to take a successor that belongs "
      "to a different track.", flush=True)
for d in DOMAINS:
    R = RROWS.get(d)
    if not R:
        continue
    A = np.asarray(R, np.float64)
    n = len(A)
    print(f"\n  -- {d}: {n:,} GT edges with the correct pair at rank >= 1 --",
          flush=True)
    print(f"  preferred target IS an annotated cell:      "
          f"{int(A[:, 3].sum()):>7,}  {100.0 * A[:, 3].mean():>5.1f}%",
          flush=True)
    print(f"  ... and is another cell's GT successor:     "
          f"{int(A[:, 4].sum()):>7,}  {100.0 * A[:, 4].mean():>5.1f}%",
          flush=True)
    print(f"  the lost edge survived anyway:              "
          f"{int(A[:, 11].sum()):>7,}  {100.0 * A[:, 11].mean():>5.1f}%",
          flush=True)
    print(f"  distances um   src->preferred {np.median(A[:, 5]):>6.2f}   "
          f"src->true {np.median(A[:, 6]):>6.2f}   "
          f"preferred->true {np.median(A[:, 7]):>6.2f}", flush=True)
    print(f"  detection score   preferred {np.median(A[:, 8]):>6.3f}   "
          f"true {np.median(A[:, 9]):>6.3f}", flush=True)
    print(f"  sources bidding for the preferred target: median "
          f"{np.median(A[:, 10]):>4.1f}", flush=True)
    closer = float((A[:, 5] < A[:, 6]).sum())
    print(f"  the preferred target is CLOSER than the true one: "
          f"{int(closer):,} ({100.0 * closer / n:.1f}%)", flush=True)
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 1e9)):
        k = float(((A[:, 7] >= lo) & (A[:, 7] < hi)).sum())
        lbl = f"{lo:g}-{hi:g} um" if hi < 1e8 else f"{lo:g}+ um"
        print(f"    preferred sits {lbl:<9} from the true target: "
              f"{int(k):>7,} ({100.0 * k / n:>5.1f}%)", flush=True)

print("\n=== THE NODES THE SOLVER DROPPED (node_absent) ===", flush=True)
print("  Both endpoints detected, the correct pair the ranker's own top "
      "choice, and the source never reached the solved graph. best_out "
      "is its best outgoing probability, best_in its best incoming one, "
      "bidders_for_self how many sources in the previous frame listed it "
      "as a candidate.", flush=True)
for d in DOMAINS:
    N = NROWS.get(d)
    if not N:
        continue
    A = np.asarray(N, np.float64)
    def _m(c):
        v = A[:, c]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")
    print(f"\n  -- {d}: {len(A):,} dropped sources --", flush=True)
    print(f"  best_out  median {_m(0):.3f}   best_in median {_m(1):.3f}   "
          f"n_cand median {_m(2):.1f}   bidders median {_m(3):.1f}   "
          f"det score median {_m(4):.3f}", flush=True)
    noin = float((~np.isfinite(A[:, 1])).sum())
    print(f"  had NO incoming candidate at all: {int(noin):,} "
          f"({100.0 * noin / len(A):.1f}%)", flush=True)
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        k = float(((A[:, 0] >= lo) & (A[:, 0] < hi)).sum())
        print(f"    best_out in {lo:.1f}-{hi:.1f}: {int(k):>6,} "
              f"({100.0 * k / len(A):>5.1f}%)", flush=True)

print("\n=== reading it ===", flush=True)
print("  If the preferred target is usually NOT a cell, the distractor "
      "is a spurious detection and detection precision is the lever. If "
      "it usually IS a cell, and specifically another track's successor, "
      "the ranker is being asked to break a tie it cannot see and only "
      "the global solve could fix it -- which it did not, since these "
      "edges are lost 85% of the time. The distance columns separate "
      "those two: a spurious peak tends to sit ON the true target, a "
      "rival cell sits a cell-width away.", flush=True)
print("  ⚠ On 6bba 9.71% of cells are annotated and on 44b6 0.77%, so "
      "`top is an annotated cell` is a floor in those columns, not a "
      "rate. zebrahub's ~97.5% annotation is the only one that reads as "
      "a rate.", flush=True)
print(f"\nRANKWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
