"""lnpersist: which of lnrescue's new peaks to admit -- priced BEFORE the heads run, on the CPU.

WHY (2026-09-17). lnrescue appended 45,803 rescue peaks to the shipped F4HP detections (128 6bba
movies); 403 lie within 7 um of an annotated cell, and they concentrate at low scores (240 of them
below 0.20, where only 1 peak in 143 is a cell). A score threshold alone therefore either admits
8,400 noise peaks (0.20) or keeps ~60 cells (0.25). This script measures threshold-free admission
rules on the same peaks, per score bin, with the annotated share of what each keeps:
  P1  another rescue peak within PERSIST_UM in the previous OR the next frame
  P2  ... in the previous AND the next frame
  G   a shipped primary (score >= 0.15) within PERSIST_UM in the previous or the next frame
      (the peak fills a hole in an existing track: no shipped detection within 7 um in ITS frame
      by construction, but one sits at its position one frame away)
and, for every annotated peak, whether its GT node has a GT edge to a node that a shipped primary
detects within 7 um ("linkable": the edge the rescue could turn from FN to TP).
v2 adds the DIVISION view: is the annotated peak a newborn daughter (its GT parent has two
children), and are its mother and its sister detected by the shipped primaries -- i.e. is the
peak the missing daughter of a division the shipped pass cannot emit (the ~19 detector-blind lost
divisions of divmiss)? Also whether it is a mother (two GT children) whose daughters are detected.

Attach: faustine649/cell-tracking (code) + competition data + biohub-lnrescue. CPU, ~10 min.
"""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-17b-lnpersist"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 7.0
PERSIST_UM = 4.0
BINS = [(0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 1.01)]
RULES = ["ALL", "P1", "P2", "G", "P1|G", "P1&G"]

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
if code is not None:
    sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff  # noqa: E402

ROOT = find_root(None)
FDET = {}
for d_ in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4ln_*.npz"):
    FDET[d_.name[len("fdet_F4ln_"):-len(".npz")]] = d_
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4ln caches (attach biohub-lnrescue)"
print(f"movies {len(SIX)}", flush=True)
OUT = pathlib.Path("/kaggle/working")


def _near(tree, pts, r):
    if tree is None or len(pts) == 0:
        return np.zeros(len(pts), bool)
    d, _ = tree.query(pts)
    return d <= r


TOT = defaultdict(float)
fh = open(OUT / "lnpersist_peaks.csv", "w", encoding="utf-8", newline="\n")
fh.write("dataset,t,score,p1,p2,g,ann,d_gt,link_edges,gt_deg,daughter,mother_det,sister_det,mother,kids_det\n")
DIV = defaultdict(int)
rows = []
t0 = time.time()
for k_, name in enumerate(SIX):
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    ship, newp, news = [], [], []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], np.float32).reshape(-1)
        nn = int(z[f"nnew{t}"]) if f"nnew{t}" in z.files else 0
        c0, s0 = (c[:-nn], s[:-nn]) if nn else (c, s)
        ship.append(c0[s0 >= UNET_THRESH] * SCALE)
        newp.append(c[-nn:] * SCALE if nn else np.zeros((0, 3)))
        news.append(s[-nn:] if nn else np.zeros(0))
    ship_tree = [cKDTree(p) if len(p) else None for p in ship]
    new_tree = [cKDTree(p) if len(p) else None for p in newp]
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_id = np.asarray(gtg["node_ids"], int)
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    idx_of = {int(i): k for k, i in enumerate(g_id)}
    nbr = defaultdict(list)
    kids = defaultdict(list)
    parent = {}
    for a, b in np.asarray(gtg["edges"], int).reshape(-1, 2):
        ia, ib = idx_of[a], idx_of[b]
        if g_t[ia] > g_t[ib]:
            ia, ib = ib, ia
        nbr[ia].append(ib)
        nbr[ib].append(ia)
        kids[ia].append(ib)
        parent[ib] = ia
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    g_det = np.zeros(len(g_id), bool)
    for tt, ix in gt_rows.items():
        if tt < T:
            g_det[ix] = _near(ship_tree[tt], g_pos[ix], MATCH_UM)
    mv = defaultdict(float)
    for t in range(T):
        pts, sc = newp[t], news[t]
        if len(pts) == 0:
            continue
        pm = _near(new_tree[t - 1], pts, PERSIST_UM) if t > 0 else np.zeros(len(pts), bool)
        pp = _near(new_tree[t + 1], pts, PERSIST_UM) if t + 1 < T else np.zeros(len(pts), bool)
        gm = _near(ship_tree[t - 1], pts, PERSIST_UM) if t > 0 else np.zeros(len(pts), bool)
        gp = _near(ship_tree[t + 1], pts, PERSIST_UM) if t + 1 < T else np.zeros(len(pts), bool)
        p1, p2, g = pm | pp, pm & pp, gm | gp
        if t in gt_tree:
            d_gt, j_gt = gt_tree[t].query(pts)
        else:
            d_gt, j_gt = np.full(len(pts), np.inf), np.zeros(len(pts), int)
        ann = d_gt <= MATCH_UM
        for i in range(len(pts)):
            link = deg = 0
            dau = m_det = s_det = mom = k_det = 0
            if ann[i]:
                gi = gt_rows[t][j_gt[i]]
                deg = len(nbr[gi])
                link = sum(1 for nb in nbr[gi] if g_det[nb])
                pa = parent.get(gi)
                if pa is not None and len(kids[pa]) == 2:
                    dau = 1
                    m_det = int(g_det[pa])
                    s_det = int(g_det[[c_ for c_ in kids[pa] if c_ != gi][0]])
                if len(kids[gi]) == 2:
                    mom = 1
                    k_det = sum(int(g_det[c_]) for c_ in kids[gi])
                b_ = next(lo for lo, hi in BINS if lo <= sc[i] < hi)
                DIV[f"ann|{b_:.2f}"] += 1
                if dau:
                    DIV[f"dau|{b_:.2f}"] += 1
                    if m_det and s_det:
                        DIV[f"dau_ms|{b_:.2f}"] += 1
                if mom:
                    DIV[f"mom|{b_:.2f}"] += 1
                    if k_det == 2:
                        DIV[f"mom_kk|{b_:.2f}"] += 1
            fh.write(f"{name},{t},{sc[i]:.4f},{int(p1[i])},{int(p2[i])},{int(g[i])},{int(ann[i])},{d_gt[i]:.2f},{link},{deg},"
                     f"{dau},{m_det},{s_det},{mom},{k_det}\n")
            flags = {"ALL": True, "P1": p1[i], "P2": p2[i], "G": g[i], "P1|G": p1[i] or g[i], "P1&G": p1[i] and g[i]}
            for lo, hi in BINS:
                if lo <= sc[i] < hi:
                    for r_, f_ in flags.items():
                        if f_:
                            mv[f"{r_}|{lo:.2f}|n"] += 1
                            mv[f"{r_}|{lo:.2f}|ann"] += int(ann[i])
                            mv[f"{r_}|{lo:.2f}|link"] += link
    for k2, v in mv.items():
        TOT[k2] += v
    n_all = sum(mv[f"ALL|{lo:.2f}|n"] for lo, _ in BINS)
    a_all = sum(mv[f"ALL|{lo:.2f}|ann"] for lo, _ in BINS)
    n_p1 = sum(mv[f"P1|{lo:.2f}|n"] for lo, _ in BINS)
    a_p1 = sum(mv[f"P1|{lo:.2f}|ann"] for lo, _ in BINS)
    n_g = sum(mv[f"G|{lo:.2f}|n"] for lo, _ in BINS)
    a_g = sum(mv[f"G|{lo:.2f}|ann"] for lo, _ in BINS)
    rows.append((name, T, int(n_all), int(a_all), int(n_p1), int(a_p1), int(n_g), int(a_g)))
    print(f"  [{k_ + 1}/{len(SIX)}] {name}: T={T} new {n_all:.0f} (ann {a_all:.0f}) | P1 keeps {n_p1:.0f} (ann {a_p1:.0f}) | "
          f"G keeps {n_g:.0f} (ann {a_g:.0f})  {time.time() - t0:.0f}s", flush=True)
fh.close()
with open(OUT / "lnpersist_movies.csv", "w", encoding="utf-8", newline="\n") as f2:
    f2.write("dataset,T,new,new_ann,p1,p1_ann,g,g_ann\n")
    for r in rows:
        f2.write(",".join(str(v) for v in r) + "\n")

print(f"\n=== LNPERSIST over {len(SIX)} movies: rescue peaks kept by each admission rule, per score bin ===", flush=True)
print("  rule   bin           kept   annotated  ann share   linkable GT edges   (kept = peaks admitted; annotated = a GT node "
      "within 7 um; linkable = that node's GT edges to a cell the shipped detections see)", flush=True)
for r_ in RULES:
    for lo, hi in BINS:
        n, a, l_ = TOT[f"{r_}|{lo:.2f}|n"], TOT[f"{r_}|{lo:.2f}|ann"], TOT[f"{r_}|{lo:.2f}|link"]
        print(f"  {r_:<6} [{lo:.2f},{hi:.2f})  {n:>8,.0f}  {a:>9,.0f}  {a / max(1, n):>9.4f}  {l_:>10,.0f}", flush=True)
    n = sum(TOT[f"{r_}|{lo:.2f}|n"] for lo, _ in BINS)
    a = sum(TOT[f"{r_}|{lo:.2f}|ann"] for lo, _ in BINS)
    l_ = sum(TOT[f"{r_}|{lo:.2f}|link"] for lo, _ in BINS)
    print(f"  {r_:<6} all           {n:>8,.0f}  {a:>9,.0f}  {a / max(1, n):>9.4f}  {l_:>10,.0f}", flush=True)
print("  reading: the shipped detections' annotated share is 0.0548; a rule whose kept peaks reach that share adds cells "
      "at the detector's own noise rate. Linkable edges bound the FN edges a rule can recover (each admitted noise peak "
      "costs ~1 node in the node term and can take a false link).", flush=True)
print("\n=== DIVISION VIEW of the annotated rescue peaks (a newborn daughter = its GT parent has two children; "
      "m+s det = mother AND sister within 7 um of a shipped primary, so admitting the peak completes a division the "
      "shipped pass cannot emit; mother = two GT children, kk det = both daughters detected) ===", flush=True)
print(f"  {'bin':<14}{'annotated':>10}{'daughter':>10}{'m+s det':>9}{'mother':>8}{'kk det':>8}", flush=True)
for lo, hi in BINS:
    print(f"  [{lo:.2f},{hi:.2f})  {DIV[f'ann|{lo:.2f}']:>10}{DIV[f'dau|{lo:.2f}']:>10}{DIV[f'dau_ms|{lo:.2f}']:>9}"
          f"{DIV[f'mom|{lo:.2f}']:>8}{DIV[f'mom_kk|{lo:.2f}']:>8}", flush=True)
print(f"  {'all':<14}{sum(v for k, v in DIV.items() if k.startswith('ann|')):>10}"
      f"{sum(v for k, v in DIV.items() if k.startswith('dau|')):>10}{sum(v for k, v in DIV.items() if k.startswith('dau_ms|')):>9}"
      f"{sum(v for k, v in DIV.items() if k.startswith('mom|')):>8}{sum(v for k, v in DIV.items() if k.startswith('mom_kk|')):>8}", flush=True)
print("  reading: divmiss counted ~19 of the 70 lost divisions as detector-blind (a daughter with no primary). A rescue peak "
      "that is such a daughter, with mother and sister detected, is one division the linker could now take: +1 div TP is "
      "worth ~+0.0006 score, 30x an edge.", flush=True)
print(f"\nLNPERSIST DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("LNPERSIST OK", flush=True)
