"""forkvel: does a real daughter ever continue the mother's velocity? (the user, 2026-09-24: a daughter cannot keep the
mother's speed and direction; only a false fork does, because nothing divided and the cell just kept moving.)

Velocity continuity of a fork p -> {c1, c2} with the mother's predecessor m (all in um):
    v_m   = p - m                       the mother's own velocity into the fork
    v_ci  = ci - p                      each branch's velocity out of the fork
    cont  = min_i |v_ci - v_m|          how well the best branch continues the mother (0 = a straight continuation)
    cont2 = min_i |(ci' - ci) - v_m|    the same one frame later (ci' = ci's single successor), when it exists
    sep   = |v_c1 - v_c2|               the daughters' relative velocity (real daughters move apart)
Populations:
    (A) every fork in the shipped graph (biohub-submit-nc35's submission.csv, 4 visible movies = train copies), labelled against
        the GT where the mother matches an annotated node within 7 um (TP if that node divides in the GT, FP if it continues,
        uncounted otherwise);
    (B) every GT division in the same 4 movies and in every 6bba / 44b6 train movie (true by definition);
    (B') the same GT divisions with every node SNAPPED to the nearest F4HP detection within 5 um (v2): the annotation's own
        jitter (integer voxels, z quantised to 1.6 um) inflates cont on GT coordinates, so the fair comparison with the
        shipped forks -- which sit on detector coordinates -- is on detector coordinates for both;
    (C) the jitter baseline: for annotated non-dividing GT chains, |v(t+1) - v(t)| (what 'continuing' looks like for a cell
        that did not divide), and the same for the shipped graph's plain chain nodes.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py) + biohub-submit-nc35 (submission.csv) + biohub-hparacache + biohub-hpara44 (fdet_F4hp_). CPU.
"""
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-24b-forkvel"
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
MATCH_UM = 7.0
SNAP_UM = 5.0
SCORE_MIN = 0.15


def _walk_files(match):
    hits = {}
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: fn[len(prefix):-len(".npz")] if fn.startswith(prefix) and fn.endswith(".npz") else None


def snap_positions(det, pos, t):
    """each GT node -> the nearest detection (score >= SCORE_MIN) within SNAP_UM at its frame; NaN where none"""
    out = np.full_like(pos, np.nan)
    for f in np.unique(t):
        key = f"c{int(f)}"
        if key not in det.files:
            continue
        C, Sc = det[key], det[f"s{int(f)}"]
        C = C[Sc >= SCORE_MIN] * SCALE
        idx = np.where(t == f)[0]
        if not len(C) or not len(idx):
            continue
        d, j = cKDTree(C).query(pos[idx], k=1)
        ok = d <= SNAP_UM
        out[idx[ok]] = C[j[ok]]
    return out


def find_submission():
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
        if "submission.csv" in fnames and "submit-nc35" in root_:
            return pathlib.Path(root_) / "submission.csv"
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
        if "submission.csv" in fnames:
            return pathlib.Path(root_) / "submission.csv"
    return None


def graph_from(pos, t, edges):
    """pos (N,3) um, t (N,), edges (E,2) row indices -> succ, pred dicts"""
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[int(a)].append(int(b))
        pred[int(b)].append(int(a))
    return succ, pred


def fork_rows(pos, t, succ, pred):
    """one row per node with exactly 2 children and exactly 1 predecessor"""
    rows = []
    for p, ch in succ.items():
        if len(ch) != 2 or len(pred.get(p, [])) != 1:
            continue
        m = pred[p][0]
        c1, c2 = ch
        if not (t[c1] == t[p] + 1 and t[c2] == t[p] + 1 and t[m] == t[p] - 1):
            continue
        if not np.all(np.isfinite(pos[[m, p, c1, c2]])):
            continue
        v_m = pos[p] - pos[m]
        v1, v2 = pos[c1] - pos[p], pos[c2] - pos[p]
        cont = min(np.linalg.norm(v1 - v_m), np.linalg.norm(v2 - v_m))
        c2s = []
        for c in (c1, c2):
            s = succ.get(c, [])
            if len(s) == 1 and t[s[0]] == t[c] + 1:
                c2s.append(np.linalg.norm((pos[s[0]] - pos[c]) - v_m))
        cont2 = min(c2s) if c2s else np.nan
        rows.append(dict(p=p, t=int(t[p]), speed_m=float(np.linalg.norm(v_m)), cont=float(cont), cont2=float(cont2),
                         sep=float(np.linalg.norm(v1 - v2)), d1=float(np.linalg.norm(v1)), d2=float(np.linalg.norm(v2))))
    return pd.DataFrame(rows)


def jitter_rows(pos, t, succ, pred, cap=20000):
    """|v(t+1) - v(t)| on plain chain nodes (1 in, 1 out, both neighbours also plain in time)"""
    out = []
    for x, ch in succ.items():
        if len(ch) != 1 or len(pred.get(x, [])) != 1:
            continue
        m, s = pred[x][0], ch[0]
        if t[m] != t[x] - 1 or t[s] != t[x] + 1:
            continue
        if not np.all(np.isfinite(pos[[m, x, s]])):
            continue
        out.append(float(np.linalg.norm((pos[s] - pos[x]) - (pos[x] - pos[m]))))
        if len(out) >= cap:
            break
    return np.array(out)


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n 0"
    return f"n {len(v):>6}  " + "  ".join(f"p{p} {np.percentile(v, p):5.2f}" for p in ps)


if __name__ == "__main__":
    t_all = time.time()
    sub_path = find_submission()
    assert sub_path is not None, "attach biohub-submit-nc35"
    sub = pd.read_csv(sub_path)
    print(f"submission {sub_path}: {len(sub):,} rows, datasets {sorted(sub.dataset.unique())}", flush=True)
    names_all = sorted(p.name.replace(".zarr", "") for p in (ROOT / "train").glob("*.zarr")) if (ROOT / "train").exists() else []
    if not names_all:
        names_all = sorted({p.name.split(".")[0] for p in ROOT.rglob("*.geff")})
    print(f"train movies with GT: {len(names_all)}", flush=True)

    # (A) shipped forks on the 4 visible movies, labelled against the GT
    A = []
    jit_pipe = []
    for name, g in sub.groupby("dataset"):
        nodes = g[g.row_type == "node"]
        ids = nodes.node_id.astype(int).values
        row_of = {int(v): i for i, v in enumerate(ids)}
        pos = nodes[["z", "y", "x"]].astype(float).values * SCALE
        tt = nodes.t.astype(int).values
        e = g[g.row_type == "edge"]
        edges = [(row_of[int(a)], row_of[int(b)]) for a, b in zip(e.source_id.astype(int), e.target_id.astype(int)) if int(a) in row_of and int(b) in row_of]
        succ, pred = graph_from(pos, tt, edges)
        F = fork_rows(pos, tt, succ, pred)
        jit_pipe.append(jitter_rows(pos, tt, succ, pred))
        # GT for the labels
        gt = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gt["t"], float).round().astype(int)
        g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
        g_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
        g_succ = defaultdict(list)
        for a, b in gt["edges"]:
            ra, rb = g_row.get(int(a)), g_row.get(int(b))
            if ra is not None and rb is not None:
                g_succ[ra].append(rb)
        lab = []
        for r in F.itertuples():
            p = int(r.p)
            cand = np.where(g_t == tt[p])[0]
            if not len(cand):
                lab.append("uncounted")
                continue
            d = np.linalg.norm(g_pos[cand] - pos[p], axis=1)
            j = int(np.argmin(d))
            if d[j] > MATCH_UM:
                lab.append("uncounted")
            else:
                lab.append("TP" if len(g_succ.get(int(cand[j]), [])) >= 2 else "FP")
        F["label"] = lab
        F["movie"] = name
        A.append(F)
        print(f"  {name}: nodes {len(nodes):,}, forks {len(F):,} (TP {sum(x == 'TP' for x in lab)}, FP {sum(x == 'FP' for x in lab)})", flush=True)
    A = pd.concat(A, ignore_index=True)
    jit_pipe = np.concatenate(jit_pipe)

    # (B) GT divisions on every train movie, and (C) GT jitter
    FDET = _walk_files(_npz_prefix("fdet_F4hp_"))
    print(f"fdet_F4hp_ caches: {len(FDET)}", flush=True)
    B, BD = [], []
    jit_gt, jit_gtd = [], []
    for name in names_all:
        gt = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gt["t"], float).round().astype(int)
        g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
        g_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
        edges = [(g_row[int(a)], g_row[int(b)]) for a, b in gt["edges"] if int(a) in g_row and int(b) in g_row]
        succ, pred = graph_from(g_pos, g_t, edges)
        F = fork_rows(g_pos, g_t, succ, pred)
        F["movie"] = name
        B.append(F)
        jit_gt.append(jitter_rows(g_pos, g_t, succ, pred))
        if name in FDET:
            det = np.load(FDET[name])
            g_snap = snap_positions(det, g_pos, g_t)
            FD = fork_rows(g_snap, g_t, succ, pred)
            FD["movie"] = name
            BD.append(FD)
            jit_gtd.append(jitter_rows(g_snap, g_t, succ, pred, cap=2000))
    B = pd.concat(B, ignore_index=True)
    BD = pd.concat(BD, ignore_index=True)
    jit_gt = np.concatenate(jit_gt)
    jit_gtd = np.concatenate(jit_gtd)
    BD.to_csv(OUT / "forkvel_gt_snapped.csv", index=False)
    A.to_csv(OUT / "forkvel_shipped.csv", index=False)
    B.to_csv(OUT / "forkvel_gt.csv", index=False)

    print("\n=== FORKVEL: velocity continuity, um per frame (cont = the best branch's |v_branch - v_mother|; cont2 one frame later; "
          "sep = the two branches' relative velocity; speed_m = the mother's speed) ===")
    print(f"  jitter baseline |v(t+1) - v(t)|, GT plain chain nodes:      {q(jit_gt)}")
    print(f"  jitter baseline, shipped graph plain chain nodes (4 movies): {q(jit_pipe)}")
    print(f"  jitter baseline, GT plain chain nodes SNAPPED to detections:  {q(jit_gtd)}")
    print(f"  GT divisions SNAPPED to detections (all four nodes within 5 um): cont   {q(BD.cont)}")
    print(f"                                                                  cont2  {q(BD.cont2)}")
    print(f"                                                                  sep    {q(BD.sep)}")
    print(f"                                                                  speed  {q(BD.speed_m)}")
    print(f"  GT divisions, all train movies with a mother predecessor: cont   {q(B.cont)}")
    print(f"                                                            cont2  {q(B.cont2)}")
    print(f"                                                            sep    {q(B.sep)}")
    print(f"                                                            speed  {q(B.speed_m)}")
    for lab in ("TP", "FP", "uncounted"):
        g = A[A.label == lab]
        print(f"  shipped forks {lab:<9}: cont   {q(g.cont)}")
        print(f"  {'':<23} cont2  {q(g.cont2)}")
        print(f"  {'':<23} sep    {q(g.sep)}")
    # separation power of cont between GT divisions (true) and shipped uncounted forks (mostly false), and TP vs FP
    from sklearn.metrics import roc_auc_score
    for feat in ("cont", "cont2", "sep"):
        x1, x0 = BD[feat].values, A.loc[A.label == "uncounted", feat].values
        m1, m0 = np.isfinite(x1), np.isfinite(x0)
        y = np.r_[np.ones(m1.sum()), np.zeros(m0.sum())]
        auc = roc_auc_score(y, np.r_[x1[m1], x0[m0]])
        tp, fp = A[A.label == "TP"][feat].values, A[A.label == "FP"][feat].values
        tp, fp = tp[np.isfinite(tp)], fp[np.isfinite(fp)]
        auc2 = roc_auc_score(np.r_[np.ones(len(tp)), np.zeros(len(fp))], np.r_[tp, fp]) if len(tp) >= 3 and len(fp) >= 3 else np.nan
        print(f"  AUC {feat:<5} GT divisions (snapped) vs shipped uncounted forks {auc:.3f}; shipped TP vs FP (n {len(tp)} / {len(fp)}) {auc2:.3f}")
    for cut in (0.5, 1.0, 1.5, 2.0):
        print(f"  cont <= {cut}: GT divisions raw {np.mean(B.cont <= cut):.3f}  snapped {np.mean(BD.cont <= cut):.3f}  shipped uncounted {np.mean(A.loc[A.label == 'uncounted', 'cont'] <= cut):.3f}  "
              f"shipped FP {np.mean(A.loc[A.label == 'FP', 'cont'] <= cut):.3f}  shipped TP {np.mean(A.loc[A.label == 'TP', 'cont'] <= cut):.3f}")
    print(f"\nFORKVEL DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("FORKVEL OK", flush=True)
