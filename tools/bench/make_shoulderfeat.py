"""shoulderfeat: what ELSE is true about the 313 shoulder cells, from the
tracking graph alone -- no images, no detector, no GPU.

THE QUESTION (user, 2026-09-08): "this point has no feature except sitting on
a slope?!" Offline work on gtsplit's CSVs answered it twice and both answers
were about the heatmap: the cells are 3-4x dimmer (h at the GT voxel p50 0.096
against 0.366), and the rate is a MOVIE-level property (0.00%-10.58%, three
movies carry two thirds). Detection density was tested and REFUTED.

Everything measured so far needs the heatmap, which is exactly what fails on
these cells. This run asks the geffs instead, because a feature living in the
GEOMETRY or the LINEAGE would be knowable without the thing that is broken.

WHAT IT MEASURES

  A. DEPTH. Light-sheet signal attenuates with z. If shoulder cells sit deep
     in the stack then "dim" is not a property of the cell but of where it is
     -- and depth is knowable at inference time with no labels at all.
     Reported as the shoulder rate per z-decile, z normalised inside each
     movie so the stacks pool, and again in raw um.

  B. Is the dimness EXPLAINED by depth? h_gt is carried in from the CSVs and
     correlated against z. If the dim cells are the deep cells, A and the
     earlier h_gt finding are one fact, not two. NOTE the CSVs only dumped
     h_gt for every GT cell, so this correlation runs on the shoulder cells
     alone unless the columns are there for all -- the code checks.

  C. TIME. Same table by t-decile. The embryo gets denser as it goes.

  D. LINEAGE STRUCTURE -- the one with an immediate use. gap_close can only
     bridge a hole it can span, so it matters enormously whether the 313 are
     isolated single frames or long runs on the same cell:
       - run length of consecutive shoulder frames within a track,
       - whether a shoulder node sits next to a division, a track start or a
         track end.
     313 spread thin over 313 different cells for one frame each is a
     gap-close problem. 313 concentrated on 40 cells for 8 frames each is
     not, and no bridge will ever reach them.

  E. CROWDING, at the GT level this time. Reported but WEAK and labelled so:
     GT annotates 9.71% of 6bba's cells and 0.77% of 44b6's, so the distance
     to the nearest ANNOTATED neighbour is mostly a fact about the annotation,
     not about the tissue. It cannot refute anything; it is here because the
     earlier density test used detections and someone will ask.

NO ANCHOR IS POSSIBLE against a previous run -- nothing has read the geffs
this way before. The internal check instead: all 313 embedded node_ids must be
found in the geffs, and the per-movie counts must reproduce the CSVs' 77 / 44
/ 40 / 33 / 27 / 22 ... exactly. A silent id mismatch would make every table
below a table about randomly chosen cells.
"""
import os

SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")
PTS = open(SC + "shoulder_all.py", encoding="utf-8").read()
assert PTS.count("\n    (") == 313, PTS.count("\n    (")
PTS += "\n" + open(SC + "measured_movies.py", encoding="utf-8").read()
assert PTS.count("\n    '") == 24, PTS.count("\n    '")

BODY = '''"""shoulderfeat -- see the module docstring of make_shoulderfeat.py."""
import pathlib
import sys
import time
import collections
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-09-08b-shoulderfeat"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets)

ROOT = find_root(None)
SCALE = np.array([1.625, 0.40625, 0.40625])   # z, y, x um per voxel, fallback
NDEC = 10

''' + PTS + '''

BY_MOVIE = defaultdict(list)
for _m, _t, _n, _hg, _hw in SHOULDER:
    BY_MOVIE[_m].append((_t, _n, _hg, _hw))
print(f"{len(SHOULDER)} shoulder cells over {len(BY_MOVIE)} movies", flush=True)

t0 = time.time()

# v1 ran over all 199 train movies of the two domains. That was wrong: only 24
# of them were ever measured by gtpeak, so the other 175 contributed ~120,000
# cells labelled "not shoulder" because nobody looked, and every rate in every
# table was diluted by whatever those movies' z and t distributions happen to
# be. v1's decile tables are superseded. Its PER-MOVIE table was never
# affected -- each row uses only its own movie's cells.
MOVIES = [d for d in list_datasets(ROOT, "train") if d in set(MEASURED)]
assert len(MOVIES) == len(MEASURED), (len(MOVIES), len(MEASURED))
print(f"{len(MOVIES)} MEASURED movies -- the other train movies in these two "
      f"domains have no shoulder labels at all and would only dilute the "
      f"denominators", flush=True)

ROWS = []
PER_MOVIE = []


def _runs(sh_ids, kids, par):
    """Lengths of maximal chains of shoulder cells that are linked in the GT.

    A chain is walked back to its first shoulder node and then forward, so a
    run is counted once no matter which of its members the outer loop reaches
    first. Divisions make the lineage a tree, not a path: a run that hits a
    fork continues down ONE branch and the other branch, being unvisited, is
    picked up as a separate run. That is the conservative reading -- it can
    only make runs look SHORTER, never longer, so "n of these are a single
    frame" is never inflated by it.
    """
    runs, seen = [], set()
    for n in sorted(sh_ids):
        if n in seen:
            continue
        cur, guard = n, 0
        while guard < 100000:
            guard += 1
            ps = [p for p in par.get(cur, ()) if p in sh_ids and p not in seen]
            if not ps:
                break
            cur = ps[0]
        L = 0
        while cur in sh_ids and cur not in seen:
            seen.add(cur)
            L += 1
            nx = [c for c in kids.get(cur, ()) if c in sh_ids and c not in seen]
            if not nx:
                break
            cur = nx[0]
        if L:
            runs.append(L)
    return runs


def _links(node_ids, edges):
    """parent -> children and child -> parents, on the raw geff ids."""
    kids = defaultdict(list)
    par = defaultdict(list)
    known = set(int(n) for n in node_ids)
    for a, b in edges:
        a, b = int(a), int(b)
        if a in known and b in known:
            kids[a].append(b)
            par[b].append(a)
    return kids, par


for name in MOVIES:
    g = read_geff(geff_path(ROOT, "train", name))
    sc = np.asarray(g.get("scale") or SCALE, float)
    nid = np.asarray(g["node_ids"], np.int64)
    tt = np.asarray(g["t"], float)
    zz = np.asarray(g["z"], float) * sc[0]
    yy = np.asarray(g["y"], float) * sc[1]
    xx = np.asarray(g["x"], float) * sc[2]
    hits = {int(n): (hg, hw) for _t, n, hg, hw in BY_MOVIE.get(name, [])}
    is_sh = np.array([int(n) in hits for n in nid], bool)
    if int(is_sh.sum()) != len(hits):
        print(f"  ! {name}: {int(is_sh.sum())} of {len(hits)} shoulder ids "
              f"found in the geff -- IDS DO NOT MATCH, every table is suspect",
              flush=True)
    lo_z, hi_z = np.percentile(zz, [1.0, 99.0])
    zn = (zz - lo_z) / max(1e-9, hi_z - lo_z)
    lo_t, hi_t = float(tt.min()), float(tt.max())
    tn = (tt - lo_t) / max(1e-9, hi_t - lo_t)

    nn = np.full(len(nid), np.nan)
    byt = defaultdict(list)
    for i, t_ in enumerate(tt):
        byt[int(t_)].append(i)
    for t_, idx in byt.items():
        if len(idx) < 2:
            continue
        p = np.stack([zz[idx], yy[idx], xx[idx]], 1)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn[idx] = d.min(axis=1)

    kids, par = _links(nid, g["edges"])
    sh_ids = set(int(n) for n in nid[is_sh])
    at_div = np.zeros(len(nid), bool)
    for i, n in enumerate(nid):
        n = int(n)
        if len(kids.get(n, ())) > 1:
            at_div[i] = True
            continue
        for p_ in par.get(n, ()):
            if len(kids.get(p_, ())) > 1:
                at_div[i] = True
        for c_ in kids.get(n, ()):
            if len(kids.get(c_, ())) > 1:
                at_div[i] = True

    runs = _runs(sh_ids, kids, par)
    assert sum(runs) == len(sh_ids), (sum(runs), len(sh_ids), name)

    for i, n in enumerate(nid):
        hg, _hw = hits.get(int(n), (np.nan, np.nan))
        ROWS.append((name, name[:4], bool(is_sh[i]), float(zn[i]),
                     float(zz[i]), float(tn[i]), float(nn[i]), bool(at_div[i]),
                     len(par.get(int(n), ())) == 0,
                     len(kids.get(int(n), ())) == 0, float(hg)))
    PER_MOVIE.append((name, len(nid), int(is_sh.sum()), runs,
                      float(np.median(zn[is_sh])) if is_sh.any() else np.nan,
                      float(np.median(zn)),
                      float(np.median(zz[is_sh])) if is_sh.any() else np.nan,
                      float(np.median(zz))))
    print(f"  {name:<16} N={len(nid):>5}  shoulder={int(is_sh.sum()):>3}  "
          f"z {zz.min():.0f}-{zz.max():.0f} um  "
          f"runs={sorted(runs, reverse=True)[:8]}  ({time.time() - t0:.0f}s)",
          flush=True)

DOM = np.array([r[1] for r in ROWS])
SH = np.array([r[2] for r in ROWS], bool)
ZN = np.array([r[3] for r in ROWS], float)
ZU = np.array([r[4] for r in ROWS], float)
TN = np.array([r[5] for r in ROWS], float)
NNB = np.array([r[6] for r in ROWS], float)
DIV = np.array([r[7] for r in ROWS], bool)
TSTART = np.array([r[8] for r in ROWS], bool)
TEND = np.array([r[9] for r in ROWS], bool)
HG = np.array([r[10] for r in ROWS], float)
print(f"\\n{len(ROWS)} GT cells, {int(SH.sum())} of them shoulder "
      f"({100.0 * SH.mean():.2f}%)", flush=True)
assert int(SH.sum()) == len(SHOULDER), (int(SH.sum()), len(SHOULDER))


def decile_table(val, title, unit=""):
    print(f"\\n=== {title} ===", flush=True)
    print(f"  {'bin':<16} {'6bba n':>8} {'6bba sh':>8} {'rate':>7}   "
          f"{'44b6 n':>8} {'44b6 sh':>8} {'rate':>7}", flush=True)
    ok = np.isfinite(val)
    edges = np.percentile(val[ok], np.linspace(0, 100, NDEC + 1))
    edges[-1] += 1e-9
    for k in range(NDEC):
        m = ok & (val >= edges[k]) & (val < edges[k + 1])
        cells = []
        for dm in ("6bba", "44b6"):
            q = m & (DOM == dm)
            n = int(q.sum())
            s = int((q & SH).sum())
            cells.append(f"{n:>8} {s:>8} {100.0 * s / max(1, n):>6.2f}%")
        lbl = f"{edges[k]:.2f}-{edges[k + 1]:.2f}{unit}"
        print(f"  {lbl:<16} " + "   ".join(cells), flush=True)


decile_table(ZN, "A. DEPTH: shoulder rate by normalised z (0 = one edge of "
                 "the stack)")
# v1's per-movie table showed the shoulder cells sit nearer a z EDGE than
# their own movie's cells in 17 of 18 movies (sign test p = 1.5e-5), at BOTH
# ends, which a monotone depth table cannot express. So fold the axis: 0 =
# hard against either edge, 0.5 = the middle of the stack.
decile_table(np.minimum(ZN, 1.0 - ZN),
             "A3. DISTANCE TO THE NEARER z EDGE (0 = at an edge, 0.5 = "
             "mid-stack) -- THE ONE THAT MATTERS")
decile_table(ZU, "A2. DEPTH in raw um (pools movies with different stacks)",
             "um")
decile_table(TN, "C. TIME: shoulder rate by normalised t")
decile_table(NNB, "E. CROWDING: nearest ANNOTATED neighbour in um -- WEAK")
print("  ^ GT annotates 9.71% of 6bba and 0.77% of 44b6, so this is mostly a "
      "fact\\n    about the annotation, not the tissue. It cannot refute "
      "anything.", flush=True)

print("\\n=== B. is the DIMNESS explained by DEPTH? ===", flush=True)
n_h = int(np.isfinite(HG).sum())
print(f"  h_gt is available for {n_h} of {len(ROWS)} cells"
      + ("" if n_h > len(SHOULDER) + 10 else
         " -- the CSVs carried it for the shoulder cells only, so this is a "
         "within-group\\n  correlation: among cells that already failed, are "
         "the deeper ones the dimmer ones?"), flush=True)
for dm in ("6bba", "44b6"):
    q = (DOM == dm) & np.isfinite(HG)
    if int(q.sum()) < 10:
        continue
    r = float(np.corrcoef(ZN[q], HG[q])[0, 1])
    print(f"  {dm}: corr(normalised z, h at the GT voxel) = {r:+.3f} over "
          f"{int(q.sum())} cells", flush=True)
    for k in range(4):
        m = q & (ZN >= k / 4) & (ZN < (k + 1) / 4 + (1e-9 if k == 3 else 0))
        if m.any():
            print(f"    z {k / 4:.2f}-{(k + 1) / 4:.2f}: n={int(m.sum()):>3} "
                  f"h_gt p50 {float(np.median(HG[m])):.3f}", flush=True)

print("\\n=== D. LINEAGE: where do the 313 sit in the tracks? ===", flush=True)
for dm in ("6bba", "44b6"):
    q = DOM == dm
    tot = int((q & SH).sum())
    if not tot:
        continue
    print(f"  {dm}: {tot} shoulder cells of {int(q.sum())} GT cells",
          flush=True)
    for lbl, msk in (("next to a division", DIV),
                     ("track START (no parent)", TSTART),
                     ("track END (no child)", TEND)):
        a = int((q & SH & msk).sum())
        b = int((q & msk).sum())
        print(f"    {lbl:<24} {a:>4} / {tot} = {100.0 * a / tot:>5.1f}%   "
              f"(all GT: {100.0 * b / max(1, int(q.sum())):>5.1f}%)", flush=True)

ALLRUNS = [L for _n, _N, _s, rr, *_ in PER_MOVIE for L in rr]
print("\\n  run length of CONSECUTIVE shoulder frames on one lineage:",
      flush=True)
if ALLRUNS:
    c = collections.Counter(ALLRUNS)
    for L in sorted(c):
        print(f"    {L:>2} frame(s): {c[L]:>4} runs = {L * c[L]:>4} cells "
              f"({100.0 * L * c[L] / len(SHOULDER):>5.1f}% of the 313)",
              flush=True)
    print(f"    {len(ALLRUNS)} runs in total, longest {max(ALLRUNS)}",
          flush=True)
    iso = c.get(1, 0)
    print(f"\\n  READ IT AS: {iso} of the {len(ALLRUNS)} runs are a SINGLE "
          f"frame. Those are the ones a\\n  one-frame bridge can reach at all; "
          f"the rest are holes gap_close cannot span at\\n  GAP_SPAN=1, and no "
          f"amount of better point-picking changes that.", flush=True)

print("\\n=== per movie ===", flush=True)
print(f"  {'movie':<16} {'N':>6} {'sh':>4} {'rate':>7} {'z50 sh':>8} "
      f"{'z50 all':>8} {'um sh':>7} {'um all':>7}", flush=True)
for nm, N, s, rr, z1, z2, u1, u2 in sorted(PER_MOVIE, key=lambda r: -r[2]):
    print(f"  {nm:<16} {N:>6} {s:>4} {100.0 * s / max(1, N):>6.2f}% "
          f"{z1:>8.3f} {z2:>8.3f} {u1:>7.1f} {u2:>7.1f}", flush=True)

print(f"\\nSHOULDERFEAT DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("SHOULDERFEAT OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "shoulderfeat.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(BODY)
print("written", len(BODY))
