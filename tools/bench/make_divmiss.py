"""rkedge.py -> divmiss.py: where do the missing endpoints sit, relative to
the detector's peaks? GT against the hpara peak caches, no ranker, no ILP.

TWO QUESTIONS, both from the 2026-09-11 read of `bench_csv/rkedge/`:

1. The division funnel on the shipped arm loses 21 divisions to
   `daughter_absent` and 8 to `mother_absent` (of 70 lost, 121 GT). "Absent"
   there means no node of the SOLVED graph within MATCH_UM 5.0 of the GT
   point. A node can be absent from the graph because (a) no peak exists
   near the cell at any score, (b) a peak exists only in the sub-threshold
   band [0.05, 0.15) -- the repair pool -- or (c) a primary (>= 0.15) exists
   and the ILP dropped it as an unlinked node. (a) needs a detector, (b) is
   reachable by letting `_add_safe_divisions` draw second daughters from the
   pool, (c) by letting it draw from dropped primaries. This run counts
   (a)/(b)/(c) for every mother and daughter of every GT division.

2. 1,805 of the 3,534 true edges the shipped linker misses (both endpoints
   detected) are swaps: the source took a false target 1.9 um away along
   its own velocity while the labelled true child sits 6.5 um away, 51% of
   them two or more z-planes off. The reading to test is that a second cell
   stacked along z within the 5 um match radius absorbed the GT identity.
   This run dumps every primary with its nearest and second-nearest GT node
   so the swap rows in `rkedge_edges.csv` can be joined to coordinates
   locally by (dataset, t, i).

OUTPUT (all in /kaggle/working):
  divmiss_gt.csv      one row per GT node of the 128 movies: dataset, row,
                      node id, t, z y x (voxel), parent row, number of
                      children, is_mother, is_daughter; nearest primary
                      (distance um, score, index i in the frame's primary
                      list = the rkedge edge-table index), second-nearest
                      primary distance; nearest pool peak (0.05 <= s < 0.15)
                      distance and score; nearest peak of any score.
  divmiss_prims/<name>.npz  per movie: t, i, z, y, x, s, d_gt, gt_row,
                      d_gt2, gt_row2 for every primary (float32/int32).
  A printed funnel: mothers / daughters by (primary within 5 um, pool
  within 5 um, nothing within 5 um), plus the same for all annotated nodes.

Primaries are `seeds_of`'s selection (s >= UNET_THRESH 0.15) in cache
order, so `i` here is the `i` of the edge table. DET_SRC is F4HP, so the
peaks are hpara-placed, the same points cell C was solved on.

Attach: rkedge's 16 sources verbatim. CPU, minutes.
"""
import os

src = open("rkedge.py", encoding="utf-8").read()
cut = "# ---- FEATURES + JOIN: all 199 train movies on BASE -----------------------"
assert src.count(cut) == 1
head = src[:src.index(cut)]
assert 'DET_SRC = "F4HP"' in head, "rkedge.py is not an F4HP run"
assert head.count('SCRIPT_VERSION = "2026-09-10a-rkedge"') == 1
head = head.replace('SCRIPT_VERSION = "2026-09-10a-rkedge"',
                    'SCRIPT_VERSION = "2026-09-11a-divmiss"')

tail = r'''
# ---- DIVMISS: GT against the peaks, no ranker, no ILP -------------------
FORK_LOG = []        # referenced by _add_safe_divisions, which this run never calls
t_all = time.time()
SIX = sorted(n for n in ALL_TRAIN if n.startswith("6bba_"))
assert len(SIX) == 128, len(SIX)
_missing = [n for n in SIX if n not in FDET_F4]
assert not _missing, f"no fdet cache for {len(_missing)} 6bba movies: {_missing[:5]}"
OUTD = pathlib.Path("/kaggle/working")
PRIMD = OUTD / "divmiss_prims"
PRIMD.mkdir(exist_ok=True)
BIG = 1e9

COLS = ["dataset", "row", "gid", "t", "z", "y", "x", "parent_row", "n_kids",
        "is_mother", "is_daughter", "d_prim", "s_prim", "i_prim", "d_prim2",
        "d_pool", "s_pool", "d_any", "s_any"]
fh = open(OUTD / "divmiss_gt.csv", "w", encoding="utf-8", newline="\n")
fh.write(",".join(COLS) + "\n")
FUN = {k: defaultdict(int) for k in ("mother", "daughter", "all")}
N_GT_DIV = 0
N_ROWS = 0
for k_, name in enumerate(SIX):
    t0 = time.time()
    dets = load_dets(name, "F4")                 # hpara points (F4HP override)
    T = len(dets)
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1)
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    kids = defaultdict(list)
    parent = {}
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
            parent[rb] = ra
    mothers = {r for r, ks in kids.items() if len(ks) >= 2}
    daughters = {k for r in mothers for k in kids[r]}
    N_GT_DIV += len(mothers)
    # per-frame trees
    trees = {}
    for t in range(T):
        c_all = np.asarray(dets[t][0], float).reshape(-1, 3)
        s_all = np.asarray(dets[t][1], float).reshape(-1)
        pm = s_all >= UNET_THRESH
        qm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        trees[t] = dict(
            P=(c_all[pm] * SCALE, s_all[pm], cKDTree(c_all[pm] * SCALE) if pm.any() else None),
            Q=(c_all[qm] * SCALE, s_all[qm], cKDTree(c_all[qm] * SCALE) if qm.any() else None),
            A=(c_all * SCALE, s_all, cKDTree(c_all * SCALE) if len(c_all) else None))
    # GT rows
    gt_by_t = defaultdict(list)
    for r in range(len(g_t)):
        gt_by_t[int(g_t[r])].append(r)
    gtree = {t: (rows_, cKDTree(g_pos[rows_] * SCALE)) for t, rows_ in gt_by_t.items()}
    for r in range(len(g_t)):
        t = int(g_t[r])
        if t not in trees:
            continue
        pu = g_pos[r] * SCALE
        out = dict(dataset=name, row=r, gid=ids[r], t=t, z=g_pos[r, 0], y=g_pos[r, 1],
                   x=g_pos[r, 2], parent_row=parent.get(r, -1), n_kids=len(kids.get(r, [])),
                   is_mother=int(r in mothers), is_daughter=int(r in daughters))
        P, Q, A = trees[t]["P"], trees[t]["Q"], trees[t]["A"]
        if P[2] is not None:
            kq = min(2, len(P[0]))
            dd, ii = P[2].query(pu, k=kq)
            dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
            out["d_prim"], out["s_prim"], out["i_prim"] = float(dd[0]), float(P[1][ii[0]]), int(ii[0])
            out["d_prim2"] = float(dd[1]) if kq > 1 else BIG
        else:
            out["d_prim"], out["s_prim"], out["i_prim"], out["d_prim2"] = BIG, 0.0, -1, BIG
        if Q[2] is not None:
            dq, iq = Q[2].query(pu)
            out["d_pool"], out["s_pool"] = float(dq), float(Q[1][int(iq)])
        else:
            out["d_pool"], out["s_pool"] = BIG, 0.0
        if A[2] is not None:
            da, ia = A[2].query(pu)
            out["d_any"], out["s_any"] = float(da), float(A[1][int(ia)])
        else:
            out["d_any"], out["s_any"] = BIG, 0.0
        fh.write(",".join(str(out[c]) if not isinstance(out[c], float) else f"{out[c]:.4f}"
                          for c in COLS) + "\n")
        N_ROWS += 1
        bucket = ("primary_5um" if out["d_prim"] <= MATCH_UM else
                  "pool_5um" if out["d_pool"] <= MATCH_UM else
                  "any_peak_5um" if out["d_any"] <= MATCH_UM else "no_peak_5um")
        FUN["all"][bucket] += 1
        if r in mothers:
            FUN["mother"][bucket] += 1
        if r in daughters:
            FUN["daughter"][bucket] += 1
    # primaries with their nearest two GT nodes
    rec = defaultdict(list)
    for t in range(T):
        P = trees[t]["P"]
        n = len(P[0])
        if n == 0:
            continue
        rec["t"].append(np.full(n, t, np.int32))
        rec["i"].append(np.arange(n, dtype=np.int32))
        rec["zyx"].append((P[0] / SCALE).astype(np.float32))
        rec["s"].append(P[1].astype(np.float32))
        if t in gtree:
            rows_, tr = gtree[t]
            kq = min(2, len(rows_))
            dd, ii = tr.query(P[0], k=kq)
            dd = np.asarray(dd).reshape(n, kq); ii = np.asarray(ii).reshape(n, kq)
            rec["d_gt"].append(dd[:, 0].astype(np.float32))
            rec["gt_row"].append(np.asarray(rows_)[ii[:, 0]].astype(np.int32))
            if kq > 1:
                rec["d_gt2"].append(dd[:, 1].astype(np.float32))
                rec["gt_row2"].append(np.asarray(rows_)[ii[:, 1]].astype(np.int32))
            else:
                rec["d_gt2"].append(np.full(n, BIG, np.float32))
                rec["gt_row2"].append(np.full(n, -1, np.int32))
        else:
            rec["d_gt"].append(np.full(n, BIG, np.float32))
            rec["gt_row"].append(np.full(n, -1, np.int32))
            rec["d_gt2"].append(np.full(n, BIG, np.float32))
            rec["gt_row2"].append(np.full(n, -1, np.int32))
    np.savez_compressed(PRIMD / f"{name}.npz",
                        **{k: np.concatenate(v) for k, v in rec.items()})
    print(f"[{k_ + 1}/128] {name}: T={T} gt_nodes={len(g_t)} divisions={len(mothers)} "
          f"primaries={sum(len(trees[t]['P'][0]) for t in range(T))} "
          f"{time.time() - t0:.0f}s", flush=True)
    del dets, trees, gtree, rec
    gc.collect()
fh.close()

print(f"\n=== DIVMISS: {N_ROWS} GT nodes, {N_GT_DIV} GT divisions over 128 6bba movies, "
      f"hpara peaks; nearest peak within MATCH_UM {MATCH_UM} ===", flush=True)
for grp in ("mother", "daughter", "all"):
    tot = sum(FUN[grp].values())
    print(f"  {grp:9s} n={tot:7d}  " + "  ".join(
        f"{b} {FUN[grp][b]} ({FUN[grp][b] / max(1, tot):.1%})"
        for b in ("primary_5um", "pool_5um", "any_peak_5um", "no_peak_5um")), flush=True)
print("  primary_5um = a peak with s >= 0.15 within 5 um (the funnel's 'absent' "
      "can still hold if the ILP dropped it); pool_5um = only a [0.05, 0.15) "
      "peak; any_peak_5um = only a peak below 0.05 (should be 0: the cache "
      "floor is 0.05); no_peak_5um = nothing the detector emitted.", flush=True)
warning_summary()
print(f"\nDIVMISS DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("DIVMISS OK", flush=True)
'''
out_src = head + tail
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "divmiss.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written divmiss.py", len(out_src))
