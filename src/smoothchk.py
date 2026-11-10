"""smoothchk: how smooth is the linker, measured on the SAME cells as the truth (the user, 2026-09-27: "所以排序器到底有多平滑").

forkvel (2026-09-24) compared the jitter |v(t+1) - v(t)| of the shipped graph's plain chains (all linked nodes, 4 visible
movies: median 0.63 um/frame) with that of the annotated chains (199 train movies: 1.95) and called the linker's chains
'three times smoother'. That comparison has no control: different movies, different cells, and a chain that the linker
breaks at a jump contributes no jitter at all. This kernel puts both on the same annotated cells of the 4 visible movies:

    every annotated plain step m -> x -> s (consecutive frames, x with one GT predecessor and one GT successor), each of the
    three nodes snapped to the nearest node of the shipped graph within SNAP_UM at its frame; j_gt = |(s - x) - (x - m)| on the
    shipped graph's coordinates. What the shipped graph did at x:
        correct  x -> s is its only outgoing edge
        fork     x -> s and one more child
        swap     x has an outgoing edge, none of them to s      (a wrong edge; j_chosen = the chosen edge's jitter)
        break    x has no outgoing edge                          (a missing edge)
    binned by j_gt, plus the reverse question for the swaps: is the chosen edge smoother than the true one, and nearer?
    and the same-cell jitter distributions: j_gt against the shipped chain's own jitter at the same x (whatever it linked).
Attach: the competition data + faustine649/cell-tracking (viz_cells.py) + biohub-submit-nc35 (submission.csv). CPU, ~1 min.
"""
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-27a-smoothchk"
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
SNAP_UM = 5.0
BINS = [0, 1, 2, 3, 4, 6, 8, 12, np.inf]


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


def graph_from(edges):
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[int(a)].append(int(b))
        pred[int(b)].append(int(a))
    return succ, pred


def snap_to_graph(pos, t, g_pos, g_t):
    """each GT node -> the nearest shipped node at its frame within SNAP_UM; -1 where none"""
    out = np.full(len(g_pos), -1, int)
    for f in np.unique(g_t):
        idx = np.where(g_t == f)[0]
        cand = np.where(t == f)[0]
        if not len(cand) or not len(idx):
            continue
        d, j = cKDTree(pos[cand]).query(g_pos[idx], k=1)
        ok = d <= SNAP_UM
        out[idx[ok]] = cand[j[ok]]
    return out


def plain_chain_jitter(pos, t, succ, pred, x):
    """the shipped chain's own jitter at x (1 in, 1 out, consecutive frames), NaN otherwise"""
    if len(succ.get(x, [])) != 1 or len(pred.get(x, [])) != 1:
        return np.nan
    p0, s0 = pred[x][0], succ[x][0]
    if t[p0] != t[x] - 1 or t[s0] != t[x] + 1:
        return np.nan
    return float(np.linalg.norm((pos[s0] - pos[x]) - (pos[x] - pos[p0])))


def step_rows(pos, t, succ, pred, g_t, g_succ, g_pred, snap, movie):
    """one row per annotated plain step m -> x -> s with all three snapped; what the shipped graph did at x"""
    rows = []
    n_steps = n_unsnapped = 0
    for x, ch in g_succ.items():
        if len(ch) != 1 or len(g_pred.get(x, [])) != 1:
            continue
        m, s = g_pred[x][0], ch[0]
        if g_t[m] != g_t[x] - 1 or g_t[s] != g_t[x] + 1:
            continue
        n_steps += 1
        M, X, S = snap[m], snap[x], snap[s]
        if M < 0 or X < 0 or S < 0:
            n_unsnapped += 1
            continue
        v_in = pos[X] - pos[M]
        j_gt = float(np.linalg.norm((pos[S] - pos[X]) - v_in))
        d_true = float(np.linalg.norm(pos[S] - pos[X]))
        out = succ.get(X, [])
        if S in out:
            what = "correct" if len(out) == 1 else "fork"
            Y = S
        elif out:
            what = "swap"
            Y = out[0] if len(out) == 1 else min(out, key=lambda y: np.linalg.norm((pos[y] - pos[X]) - v_in))
        else:
            what = "break"
            Y = -1
        j_chosen = float(np.linalg.norm((pos[Y] - pos[X]) - v_in)) if Y >= 0 else np.nan
        d_chosen = float(np.linalg.norm(pos[Y] - pos[X])) if Y >= 0 else np.nan
        in_ok = int(pred.get(X, []) == [M])
        s_in = pred.get(S, [])
        s_taken = int(bool(s_in) and s_in[0] != X)
        rows.append(dict(movie=movie, x=int(X), t=int(t[X]), j_gt=j_gt, d_true=d_true, what=what, j_chosen=j_chosen, d_chosen=d_chosen,
                         in_ok=in_ok, s_taken=s_taken, j_pipe=plain_chain_jitter(pos, t, succ, pred, X)))
    return pd.DataFrame(rows), n_steps, n_unsnapped


def jitter_all(pos, t, succ, pred):
    xs = [x for x, ch in succ.items() if len(ch) == 1 and len(pred.get(x, [])) == 1 and t[pred[x][0]] == t[x] - 1 and t[ch[0]] == t[x] + 1]
    return np.array(xs, int), np.array([float(np.linalg.norm((pos[succ[x][0]] - pos[x]) - (pos[x] - pos[pred[x][0]]))) for x in xs])


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n 0"
    return f"n {len(v):>7,}  " + "  ".join(f"p{p} {np.percentile(v, p):5.2f}" for p in ps)


def pct(mask):
    return float(np.mean(mask)) * 100 if len(mask) else float("nan")


if __name__ == "__main__":
    t_all = time.time()
    sub_path = find_submission()
    assert sub_path is not None, "attach biohub-submit-nc35"
    sub = pd.read_csv(sub_path)
    print(f"submission {sub_path}: {len(sub):,} rows, datasets {sorted(sub.dataset.unique())}", flush=True)
    R = []
    jit_all_pipe, jit_ann_pipe = [], []
    for name, g in sub.groupby("dataset"):
        nodes = g[g.row_type == "node"]
        ids = nodes.node_id.astype(int).values
        row_of = {int(v): i for i, v in enumerate(ids)}
        pos = nodes[["z", "y", "x"]].astype(float).values * SCALE
        tt = nodes.t.astype(int).values
        e = g[g.row_type == "edge"]
        edges = [(row_of[int(a)], row_of[int(b)]) for a, b in zip(e.source_id.astype(int), e.target_id.astype(int)) if int(a) in row_of and int(b) in row_of]
        succ, pred = graph_from(edges)
        gt = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gt["t"], float).round().astype(int)
        g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
        g_row = {int(v): i for i, v in enumerate(gt["node_ids"])}
        g_edges = [(g_row[int(a)], g_row[int(b)]) for a, b in gt["edges"] if int(a) in g_row and int(b) in g_row]
        g_succ, g_pred = graph_from(g_edges)
        snap = snap_to_graph(pos, tt, g_pos, g_t)
        D, n_steps, n_unsnapped = step_rows(pos, tt, succ, pred, g_t, g_succ, g_pred, snap, name)
        R.append(D)
        xs, ja = jitter_all(pos, tt, succ, pred)
        jit_all_pipe.append(ja)
        ann = set(int(v) for v in snap if v >= 0)
        jit_ann_pipe.append(ja[[x in ann for x in xs]] if len(xs) else ja)
        print(f"  {name}: shipped nodes {len(nodes):,} edges {len(edges):,}; GT nodes {len(g_pos):,}, snapped {int((snap >= 0).sum()):,}; "
              f"annotated plain steps {n_steps:,}, unsnapped {n_unsnapped:,}, measured {len(D):,}: "
              + ", ".join(f"{k} {int((D.what == k).sum()):,}" for k in ("correct", "fork", "swap", "break")), flush=True)
    D = pd.concat(R, ignore_index=True)
    D.to_csv(OUT / "smoothchk_steps.csv", index=False)
    jit_all_pipe = np.concatenate(jit_all_pipe)
    jit_ann_pipe = np.concatenate(jit_ann_pipe)

    print("\n=== SMOOTHCHK: the shipped graph at every annotated plain step, 4 visible movies (um per frame) ===")
    print(f"  steps measured {len(D):,}: " + ", ".join(f"{k} {int((D.what == k).sum()):,} ({pct(D.what == k):.1f}%)" for k in ("correct", "fork", "swap", "break")))
    print(f"  the true target taken by another source (s_taken): swaps {pct(D[D.what == 'swap'].s_taken):.0f}%, breaks {pct(D[D.what == 'break'].s_taken):.0f}%")
    print("\n  by the truth's own jitter j_gt (what the cell actually did at x):")
    print(f"  {'j_gt bin':<12} {'n':>7} {'correct':>8} {'fork':>6} {'swap':>6} {'break':>6} | {'swap: chosen smoother':>22} {'chosen nearer':>14} {'med j_chosen':>12} {'med d_true':>10} {'med d_chosen':>12}")
    D["bin"] = pd.cut(D.j_gt, BINS, right=False)
    for b, g in D.groupby("bin", observed=True):
        sw = g[g.what == "swap"]
        print(f"  {str(b):<12} {len(g):>7,} {pct(g.what == 'correct'):>7.1f}% {pct(g.what == 'fork'):>5.1f}% {pct(g.what == 'swap'):>5.1f}% {pct(g.what == 'break'):>5.1f}% | "
              f"{pct(sw.j_chosen < sw.j_gt):>21.0f}% {pct(sw.d_chosen < sw.d_true):>13.0f}% "
              f"{sw.j_chosen.median() if len(sw) else float('nan'):>12.2f} {g.d_true.median():>10.2f} {sw.d_chosen.median() if len(sw) else float('nan'):>12.2f}")
    print("\n  same cells, two chains -- the jitter the truth has at x against the jitter the shipped chain has at x (1 in, 1 out, whatever it linked):")
    print(f"    truth   j_gt   at every measured step:                     {q(D.j_gt)}")
    print(f"    truth   j_gt   where the shipped step is correct:          {q(D[D.what == 'correct'].j_gt)}")
    print(f"    truth   j_gt   where the shipped step is a swap:           {q(D[D.what == 'swap'].j_gt)}")
    print(f"    truth   j_gt   where the shipped step is a break:          {q(D[D.what == 'break'].j_gt)}")
    print(f"    shipped j_pipe at the same x (where it has a plain chain):  {q(D.j_pipe)}")
    print(f"    shipped j_pipe at the swaps:                               {q(D[D.what == 'swap'].j_pipe)}")
    print(f"    shipped jitter, every plain chain node of the 4 movies:    {q(jit_all_pipe)}   (forkvel's 0.63 line)")
    print(f"    shipped jitter, plain chain nodes that carry an annotation:  {q(jit_ann_pipe)}")
    sw = D[D.what == "swap"]
    print(f"\n  the swaps ({len(sw):,}): the chosen edge is smoother than the true one in {pct(sw.j_chosen < sw.j_gt):.0f}%, nearer in {pct(sw.d_chosen < sw.d_true):.0f}%; "
          f"median j_chosen {sw.j_chosen.median():.2f} vs j_gt {sw.j_gt.median():.2f}; median d_chosen {sw.d_chosen.median():.2f} vs d_true {sw.d_true.median():.2f}")
    br = D[D.what == "break"]
    print(f"  the breaks ({len(br):,}): median j_gt {br.j_gt.median():.2f}, d_true {br.d_true.median():.2f}; the true target had another source in {pct(br.s_taken):.0f}%")
    print("  reading: 'the linker is smoother than the truth' is a wrong-edge bias only if the swaps pick the smoother edge well above 50% "
          "and the swap rate rises with j_gt; if the error at high j_gt is mostly breaks, the bias sits in the edge price (floor / disappear), not in the ranking.")
    print(f"\nSMOOTHCHK OK in {(time.time() - t_all) / 60:.1f} min", flush=True)
