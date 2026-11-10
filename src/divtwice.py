"""divtwice: for the 22 'timing' divisions (divtime), what the GT and the raw detections say about the second daughter
that seems to exist frames before the annotated division -- without the linker.

Per case (mother at t0, daughters at t0+1), from the GT alone:
  (a) divided twice? the mother's annotated ancestry back 30 frames: frame of the nearest earlier division on it, if any;
      and does either daughter divide again within 30 frames after (a fast cycle would make both plausible);
  (b) daughters at t0+1: their separation, and each one's distance to the mother at t0 (the 'second' daughter = the farther);
  (c) other GT tracks near the second daughter: any annotated node NOT on this lineage within 6 um of the second daughter's
      position at t0+1 or of its back-tracked positions (a separately annotated cell sitting where the daughter 'comes from');
and from the F4HP detections alone (score >= 0.15, no linker):
  (d) back-track the second daughter from its detection at t0+1: frame by frame the nearest detection within 5 um of the
      previous position, stopping when none; report how many frames back a detection persists at that spot, its distance
      to the mother's annotated chain over those frames (min / median), and the same walk for the first daughter and for
      the on-time controls;
  (e) the match itself: distance from each GT daughter to its nearest detection at t0+1, and whether the nearest detection
      of the second daughter is closer to some OTHER GT node at t0+1.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py) + biohub-hparacache (fdet_F4hp_). CPU.
"""
import io
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-24a-divtwice"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
INPUT_ROOT = "/kaggle/input"
SCALE = np.array([1.625, 0.40625, 0.40625])
SCORE_MIN = 0.15
BACK_DIV = 30            # frames of ancestry searched for an earlier division
FWD_DIV = 30             # frames after for a daughter's own division
STEP_UM = 5.0            # back-tracking step radius on detections
MAX_BACK = 60
NEAR_UM = 6.0            # 'another annotated cell sits here'

CASES_CSV = """idx,kind,movie,t_div,d2_start_off,eye
1,timing,6bba_05db0fb1,62,-6,not_split
2,timing,6bba_09961292,48,-49,undecidable
3,timing,6bba_0e7c0d07,57,-13,undecidable
4,timing,6bba_12665c0e,39,-15,not_split
5,timing,6bba_1d0d8384,14,-12,not_split
6,timing,6bba_32db13fc,5,-6,early_real
7,ontime,6bba_48816121,16,0,control
8,timing,6bba_57b7cc1e,12,-4,undecidable
9,timing,6bba_57b7cc1e,23,-11,early_real
10,timing,6bba_57b7cc1e,77,-26,undecidable
11,timing,6bba_6321a359,8,-9,early_real
12,timing,6bba_705ec2c9,42,-3,early_real
13,timing,6bba_74686d6a,24,-16,early_real
14,timing,6bba_786893ac,63,-2,undecidable
15,timing,6bba_78a7bd97,56,-2,early_real
16,timing,6bba_7af54fde,52,-2,undecidable
17,timing,6bba_7af54fde,55,-5,not_split
18,timing,6bba_969618f6,71,-2,undecidable
19,ontime,6bba_9e23430b,89,0,control
20,timing,6bba_ab78413d,67,-2,undecidable
21,timing,6bba_afb141ff,48,-6,not_split
22,ontime,6bba_bb9f20c3,31,0,control
23,ontime,6bba_cdcfe533,52,0,control_gt_error
24,timing,6bba_debd7bfa,12,-4,early_real
25,timing,6bba_ebdf3b34,13,-4,undecidable
26,timing,6bba_fc5f39dc,54,-55,undecidable
"""


def _walk_files(match):
    hits = {}
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: fn[len(prefix):-len(".npz")] if fn.startswith(prefix) and fn.endswith(".npz") else None


def gt_graph(gt):
    g_t = np.asarray(gt["t"], float).round().astype(int)
    pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gs, gp = defaultdict(list), {}
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            gs[ra].append(rb)
            gp[rb] = ra
    return g_t, pos, gs, gp


def ancestry(gp, m, n):
    out, cur = [m], m
    for _ in range(n):
        cur = gp.get(cur)
        if cur is None:
            break
        out.append(cur)
    return out


def descendants(gs, d, n):
    """rows reachable from d within n frames, with the frame offset; and the offset of the first division on the way"""
    out, front, first_div = [d], [d], None
    for k in range(1, n + 1):
        nxt = []
        for x in front:
            ch = gs.get(x, [])
            if len(ch) >= 2 and first_div is None:
                first_div = k - 1
            nxt.extend(ch)
        if not nxt:
            break
        out.extend(nxt)
        front = nxt
    return out, first_div


def backtrack(det, t_start, p_start, T):
    """walk back from p_start (um) at t_start through the nearest detection within STEP_UM per frame; positions by frame"""
    path = {t_start: p_start}
    p = p_start
    for t in range(t_start - 1, max(-1, t_start - MAX_BACK - 1), -1):
        C, Sc = det[f"c{t}"], det[f"s{t}"]
        C = C[Sc >= SCORE_MIN] * SCALE
        if not len(C):
            break
        d = np.linalg.norm(C - p, axis=1)
        j = int(np.argmin(d))
        if d[j] > STEP_UM:
            break
        p = C[j]
        path[t] = p
    return path


def nearest_det(det, t, p):
    C, Sc = det[f"c{t}"], det[f"s{t}"]
    C = C[Sc >= SCORE_MIN] * SCALE
    if not len(C):
        return None, np.inf
    d = np.linalg.norm(C - p, axis=1)
    j = int(np.argmin(d))
    return C[j], float(d[j])


if __name__ == "__main__":
    t_all = time.time()
    FDET = _walk_files(_npz_prefix("fdet_F4hp_"))
    cases = pd.read_csv(io.StringIO(CASES_CSV))
    assert not (set(cases.movie) - set(FDET)), "attach biohub-hparacache"
    rows = []
    for name, grp in cases.groupby("movie", sort=True):
        gt = read_geff(geff_path(ROOT, "train", name))
        g_t, pos, gs, gp = gt_graph(gt)
        det = np.load(FDET[name])
        T = max(int(k[1:]) for k in det.files if k.startswith("c")) + 1
        trees = {}
        for r in grp.itertuples():
            t0 = int(r.t_div)
            divs = [m for m, ss in gs.items() if len(ss) >= 2 and g_t[m] == t0]
            assert len(divs) == 1, (name, t0, len(divs))
            m = divs[0]
            d1, d2 = gs[m][:2]
            # (b) geometry in the GT
            dm = [float(np.linalg.norm(pos[d] - pos[m])) for d in (d1, d2)]
            if dm[1] < dm[0]:
                d1, d2, dm = d2, d1, dm[::-1]
            sep = float(np.linalg.norm(pos[d1] - pos[d2]))
            # (a) divided twice?
            anc = ancestry(gp, m, BACK_DIV)
            prev_div = [int(t0 - g_t[a]) for a in anc[1:] if len(gs.get(a, [])) >= 2]
            prev_div_off = -min(prev_div) if prev_div else None
            lineage_rows = set(anc)
            nxt_div = {}
            for lab, d in (("d1", d1), ("d2", d2)):
                desc, fd = descendants(gs, d, FWD_DIV)
                lineage_rows |= set(desc)
                nxt_div[lab] = fd
            # (d) back-track on detections
            p2_det, e2 = nearest_det(det, t0 + 1, pos[d2])
            p1_det, e1 = nearest_det(det, t0 + 1, pos[d1])
            pm_det, em = nearest_det(det, t0, pos[m])
            mchain = {int(g_t[a]): pos[a] for a in anc}
            walk2 = backtrack(det, t0 + 1, p2_det, T) if p2_det is not None else {}
            walk1 = backtrack(det, t0 + 1, p1_det, T) if p1_det is not None else {}
            n_back2 = (t0 + 1) - min(walk2) if walk2 else 0
            n_back1 = (t0 + 1) - min(walk1) if walk1 else 0
            dist_m2 = [float(np.linalg.norm(walk2[t] - mchain[t])) for t in walk2 if t in mchain and t <= t0]
            dist_m1 = [float(np.linalg.norm(walk1[t] - mchain[t])) for t in walk1 if t in mchain and t <= t0]
            # (c) other annotated tracks near the second daughter's back-tracked positions (t0-5 .. t0+1)
            others = 0
            other_ids = set()
            for t in range(max(0, t0 - 5), t0 + 2):
                p = walk2.get(t)
                if p is None:
                    continue
                cand = np.where((g_t == t))[0]
                cand = [c for c in cand if c not in lineage_rows]
                if not cand:
                    continue
                dd = np.linalg.norm(pos[cand] - p, axis=1)
                for c, x in zip(cand, dd):
                    if x <= NEAR_UM:
                        others += 1
                        other_ids.add(int(c))
            # (e) is d2's nearest detection closer to another GT node at t0+1?
            g_here = np.where(g_t == t0 + 1)[0]
            steal = None
            if p2_det is not None and len(g_here):
                dd = np.linalg.norm(pos[g_here] - p2_det, axis=1)
                j = int(np.argmin(dd))
                steal = (int(g_here[j]) != int(d2)) and dd[j] < e2
            rows.append(dict(idx=int(r.idx), kind=r.kind, eye=r.eye, movie=name, t0=t0, d2_start_off=int(r.d2_start_off),
                             gt_sep=round(sep, 2), gt_d1_to_m=round(dm[0], 2), gt_d2_to_m=round(dm[1], 2),
                             prev_div_off=prev_div_off, d1_next_div=nxt_div["d1"], d2_next_div=nxt_div["d2"],
                             m_det_err=round(em, 2), d1_det_err=round(e1, 2), d2_det_err=round(e2, 2),
                             d2_nearest_det_belongs_elsewhere=steal,
                             det_back_d2=int(n_back2), det_back_d1=int(n_back1),
                             d2_to_mother_min=round(min(dist_m2), 2) if dist_m2 else None,
                             d2_to_mother_med=round(float(np.median(dist_m2)), 2) if dist_m2 else None,
                             d1_to_mother_min=round(min(dist_m1), 2) if dist_m1 else None,
                             other_gt_near_d2_path=int(len(other_ids)),
                             mother_chain_len=len(anc) - 1))
        print(f"  {name}: {len(grp)} cases", flush=True)
    df = pd.DataFrame(rows).sort_values("idx")
    df.to_csv(OUT / "divtwice.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n=== DIVTWICE: per case (um; det_back = frames a detection persists at the daughter's spot before t0+1, walking back "
          "through the nearest detection within 5 um each frame; d2_to_mother = that walk's distance to the mother's annotated chain) ===")
    print(df.drop(columns=["movie"]).to_string(index=False))
    print("\n  by eye class: median det_back_d2 / d2_to_mother_med / gt_sep / d2_det_err / other_gt_near")
    for lab, g in df.groupby("eye"):
        print(f"    {lab:<18} n {len(g):>2}  det_back_d2 {g.det_back_d2.median():>4.0f}  d2_to_mother_med {g.d2_to_mother_med.median():>5.1f}  "
              f"gt_sep {g.gt_sep.median():>5.1f}  d2_det_err {g.d2_det_err.median():>4.1f}  other_gt_near {g.other_gt_near_d2_path.median():>3.0f}  "
              f"prev_div {int(g.prev_div_off.notna().sum())}  d2_next_div {int(g.d2_next_div.notna().sum())}")
    print(f"\nDIVTWICE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("DIVTWICE OK", flush=True)
