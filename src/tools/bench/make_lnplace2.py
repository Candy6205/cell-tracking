"""lnplace2.py: is the annotated cell behind a rescue peak really an UNDETECTED cell at that frame?

WHY (2026-09-18). The user's question after lnplace: does a rescue peak correspond to a cell at all? lntrace
called a rescue peak "annotated" when a GT node lies within 7 um at the same frame and that node's GT
neighbour at t-1 or t+1 has a shipped primary within 7 um. It never asked whether the GT node at frame t
ITSELF already has a shipped primary: a peak 5.7 um from an annotated cell that a shipped primary holds at
1.5 um is a second point next to a detected cell, and the ranker is right to refuse its edges. lnplace
found the raw image nearly flat at these GT centroids (contrast 0.1-0.2 against 0.6 for detected cells),
which fits either a genuinely dim missed cell or an annotation placed where no nucleus is (interpolated
between frames where the cell was visible).

WHAT. Standalone CPU kernel, no solve. For every annotated rescue peak of lnrank_edges.csv (arm LN20), with
its class (linked / ranker_refused / below_tau / other):
  * the matched GT node g at frame t: distance to the nearest SHIPPED primary at t and that primary's score
    ("already detected" = within 3 um); the peak's own distance to the nearest shipped primary;
  * the GT track around g: its neighbours at t-1 and t+1, their distance to the nearest shipped primary at
    their frame, the raw contrast (fine smoothing, over the crop median) at g and at both neighbours -- does
    the cell go dark at t only?; the step lengths and the deviation of g from the midpoint of its neighbours
    (an interpolated annotation sits on the line: deviation ~0);
  * a control: GT nodes of the same frames that a shipped primary holds within 2 um (detected cells), up to
    30 per movie, the same columns.
Summary per class, and the refused class split by "already detected at t" or not. lnplace2_peaks.csv,
lnplace2_control.csv. Attach: biohub-lnrescue, biohub-lnrank, the competition data, the code dataset, the
support pack. CPU ~4 min.
"""
import os

ln = open("lnrank.py", encoding="utf-8").read()


def between(s, a, b, include_b=False):
    i = s.index(a)
    j = s.index(b, i + len(a))
    return s[i:j + (len(b) if include_b else 0)]


pre = between(ln, "from __future__ import annotations\n", "SCALE = np.array([1.625, 0.40625, 0.40625])\n", include_b=True)
pre = pre.replace('SCRIPT_VERSION = "2026-09-18a-lnrank"', 'SCRIPT_VERSION = "2026-09-18a-lnplace2"')
assert 'SCRIPT_VERSION = "2026-09-18a-lnplace2"' in pre and "from viz_cells import" in pre
warn = between(ln, "_WARN_COUNT = defaultdict(int)\n", "def pass1_tracks(dets):")
walk = between(ln, "def _walk_files(match):", "# ---- the official head's probabilities")
lnload = between(ln, "def ln_load(name):", "LN_PERSIST_UM = 4.0")

BODY = r'''
root = find_root(None)
print(f"root {root}", flush=True)
import pandas as pd                                                          # noqa: E402
from scipy import ndimage as _ndi                                            # noqa: E402
from viz_cells import open_image as _open_image                              # noqa: E402

LN_FDET = _walk_files(_npz_prefix("fdet_F4ln_"))
print(f"rescue caches (fdet_F4ln_): {len(LN_FDET)}", flush=True)
assert len(LN_FDET) >= 100, "attach biohub-lnrescue"
_csv = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/lnrank_edges.csv")]
assert _csv, "attach biohub-lnrank (lnrank_edges.csv)"
df = pd.read_csv(_csv[0])
df = df[df.arm == "LN20"].reset_index(drop=True)
print(f"lnrank_edges.csv: {len(df)} LN20 rows over {df.dataset.nunique()} movies", flush=True)

UNET_THRESH = 0.15                # seeds_of: the harness keeps primaries at score >= this; q_i indexes THAT list
HALF_UM = 8.0
SIG = (0.6, 1.5, 1.5)             # forkimg's contrast crop
DET_UM = 3.0                      # a shipped primary this close to a GT node = the cell is detected
CTRL_UM = 2.0                     # control cells: a shipped primary within this of the GT node
LIN_UM = 0.3                      # a GT node this close to the midpoint of its two neighbours looks interpolated


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _crop(img, t, xyz_vox, sigma):
    if t < 0 or t >= img.shape[0]:
        return None, None
    hz = int(np.ceil(HALF_UM / SCALE[0]))
    hy = int(np.ceil(HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    try:
        crop = np.asarray(img[t, z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1])
    except Exception:                                                        # noqa: BLE001
        crop = np.asarray(img[t])[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return None, None
    return _ndi.gaussian_filter(crop.astype(np.float32), sigma), np.array([z0, y0, x0], float)


def con_at(img, t, xyz_um):
    """contrast over the crop median at a position (and the maximum within 2 um of it); NaN when unreadable."""
    xyz_vox = np.asarray(xyz_um, float) / SCALE
    sm, off = _crop(img, t, xyz_vox, SIG)
    if sm is None:
        return np.nan, np.nan
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    pos_um = (idx + off) * SCALE
    v = sm.reshape(-1)
    bg = float(np.median(v))
    d = np.linalg.norm(pos_um - np.asarray(xyz_um, float), axis=1)
    j0 = int(np.argmin(d))
    near = d <= 2.0
    mx = float(v[near].max()) if near.any() else float(v[j0])
    return (float(v[j0]) - bg) / max(bg, 1e-6), (mx - bg) / max(bg, 1e-6)


def peak_class(cats):
    cats = set(cats)
    if "G_tp" in cats:
        return "linked"
    if "C_below_floor" in cats:
        return "ranker_refused"
    if cats == {"A_below_tau"}:
        return "below_tau"
    return "other"


peaks = {}
for (name, t, qi), g_ in df.groupby(["dataset", "t", "q_i"]):
    peaks[(name, int(t), int(qi))] = dict(cls=peak_class(g_.cat.tolist()), q_score=float(g_.q_score.iloc[0]),
                                          d_gt_csv=float(g_.d_gt.iloc[0]), admitted=int(g_.admitted.iloc[0]))
print(f"peaks: {len(peaks)}; per class " + str(pd.Series([v['cls'] for v in peaks.values()]).value_counts().to_dict()),
      flush=True)


def cell_row(img, g, t, g_t, g_pos, nbr, ship):
    """everything about one GT node g at frame t: detection at t, the track around it, the raw contrast along it."""
    row = {}
    st = ship(t)
    if st is not None:
        d_, i_ = st[0].query(g_pos[g])
        row["d_gt_ship"], row["s_score"] = float(d_), float(st[1][i_])
    else:
        row["d_gt_ship"], row["s_score"] = np.nan, np.nan
    row["con_gt"], row["con_gt_max2"] = con_at(img, t, g_pos[g])
    prev = [gn for gn in nbr[g] if int(g_t[gn]) == t - 1]
    nxt = [gn for gn in nbr[g] if int(g_t[gn]) == t + 1]
    row["deg"] = len(nbr[g])
    row["has_prev"], row["has_next"] = int(bool(prev)), int(bool(nxt))
    for side, lst, tt in (("prev", prev, t - 1), ("next", nxt, t + 1)):
        if lst:
            gn = lst[0]
            st2 = ship(tt)
            row[f"d_{side}_ship"] = float(st2[0].query(g_pos[gn])[0]) if st2 is not None else np.nan
            row[f"con_{side}"], row[f"con_{side}_max2"] = con_at(img, tt, g_pos[gn])
            row[f"step_{side}"] = float(np.linalg.norm(g_pos[gn] - g_pos[g]))
        else:
            row[f"d_{side}_ship"] = row[f"con_{side}"] = row[f"con_{side}_max2"] = row[f"step_{side}"] = np.nan
    if prev and nxt:
        mid = 0.5 * (g_pos[prev[0]] + g_pos[nxt[0]])
        row["lin_dev"] = float(np.linalg.norm(g_pos[g] - mid))
    else:
        row["lin_dev"] = np.nan
    return row


ROWS, CTRL = [], []
t_all = time.time()
rng = np.random.default_rng(0)
by_movie = defaultdict(list)
for k_, v_ in peaks.items():
    by_movie[k_[0]].append((k_[1], k_[2], v_))
n_skip = 0
for mi, name in enumerate(sorted(by_movie)):
    t0 = time.time()
    if name not in LN_FDET:
        warn_once("no_fdet", f"{name}: no fdet_F4ln_ cache")
        continue
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    nbr = defaultdict(list)
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is not None and rb is not None:
            nbr[ra].append(rb)
            nbr[rb].append(ra)
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    dets, nnew = ln_load(name)
    seeds = [seeds_of(dets[t]) for t in range(len(dets))]
    try:
        img = _open_image(pathlib.Path(root) / "train" / f"{name}.zarr")
    except Exception as _e:                                                  # noqa: BLE001
        warn_once("no_img", f"{name}: image not opened ({_e})")
        continue
    ship_cache = {}

    def _ship(t):
        if t < 0 or t >= len(seeds):
            return None
        if t not in ship_cache:
            c_, s_ = seeds[t]
            n_ = len(c_) - nnew[t]
            ship_cache[t] = (cKDTree(c_[:n_] * SCALE), s_[:n_]) if n_ > 0 else None
        return ship_cache[t]

    used_g = set()
    for t, qi, meta in by_movie[name]:
        c_all = seeds[t][0]
        if qi >= len(c_all) or t not in gt_tree:
            n_skip += 1
            continue
        c_um = c_all[qi] * SCALE
        d0, j_ = gt_tree[t].query(c_um)
        g = int(gt_rows[t][int(j_)])
        used_g.add((t, g))
        row = cell_row(img, g, t, g_t, g_pos, nbr, _ship)
        st = _ship(t)
        row.update(dataset=name, t=t, q_i=qi, **meta, d_gt0=float(d0),
                   d_peak_ship=float(st[0].query(c_um)[0]) if st is not None else np.nan,
                   con_peak=con_at(img, t, c_um)[0])
        ROWS.append(row)
    # control: GT nodes of the same frames held by a shipped primary within CTRL_UM, up to 30 per movie
    frames = sorted({t for t, _q, _m in by_movie[name]})
    cands = []
    for t in frames:
        st = _ship(t)
        if st is None or t not in gt_rows:
            continue
        ix = gt_rows[t]
        dd, jj = st[0].query(g_pos[ix])
        for k_ in np.where(dd <= CTRL_UM)[0]:
            g = int(ix[k_])
            if (t, g) not in used_g:
                cands.append((t, g))
    if cands:
        for ci in rng.choice(len(cands), size=min(30, len(cands)), replace=False):
            t, g = cands[ci]
            row = cell_row(img, g, t, g_t, g_pos, nbr, _ship)
            row.update(dataset=name, t=t, g=g)
            CTRL.append(row)
    if mi % 16 == 0 or mi == len(by_movie) - 1:
        print(f"  [{mi + 1}/{len(by_movie)}] {name}: {len(by_movie[name])} peaks, {time.time() - t0:.1f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)

P = pd.DataFrame(ROWS)
C = pd.DataFrame(CTRL)
P.to_csv("/kaggle/working/lnplace2_peaks.csv", index=False)
C.to_csv("/kaggle/working/lnplace2_control.csv", index=False)
print(f"\n{len(P)} peaks ({n_skip} skipped), {len(C)} control cells; recomputed d_gt matches lnrank's csv within 0.01 um for "
      f"{int((np.abs(P.d_gt0 - P.d_gt_csv) < 0.01).sum())} of {len(P)}", flush=True)


def _q(s, f=0.5):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, f)) if len(s) else float("nan")


def _frac(m):
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    return 100.0 * float(np.mean(m)) if len(m) else float("nan")


CLASSES = ("ranker_refused", "linked", "below_tau", "other", "ALL", "control")


def _sel(cls):
    return C if cls == "control" else (P if cls == "ALL" else P[P.cls == cls])


print("\n=== IS THE ANNOTATED CELL BEHIND THE PEAK DETECTED AT FRAME t? (the GT node matched to the peak; medians in um) ===", flush=True)
print(f"  {'class':<15} {'n':>4} {'d_gt0':>6} | {'GT->ship':>8} {'<=3':>5} {'s_score':>7} | {'peak->ship':>10} {'<=3':>5} | "
      f"{'prev det':>8} {'next det':>8} | {'con t-1':>7} {'con t':>6} {'con t+1':>7} | {'max2 t':>6} | {'lin_dev':>7} {'<0.3':>5} {'step':>5}",
      flush=True)
for cls in CLASSES:
    s = _sel(cls)
    if not len(s):
        continue
    d_gt0 = f"{_q(s.d_gt0):>6.2f}" if "d_gt0" in s else f"{'--':>6}"
    p_ship = f"{_q(s.d_peak_ship):>10.2f} {_frac(s.d_peak_ship <= DET_UM):>4.0f}%" if "d_peak_ship" in s else f"{'--':>10} {'--':>5}"
    print(f"  {cls:<15} {len(s):>4} {d_gt0} | {_q(s.d_gt_ship):>8.2f} {_frac(s.d_gt_ship <= DET_UM):>4.0f}% {_q(s.s_score):>7.2f} | "
          f"{p_ship} | {_frac(s.d_prev_ship <= DET_UM):>7.0f}% {_frac(s.d_next_ship <= DET_UM):>7.0f}% | "
          f"{_q(s.con_prev):>7.2f} {_q(s.con_gt):>6.2f} {_q(s.con_next):>7.2f} | {_q(s.con_gt_max2):>6.2f} | "
          f"{_q(s.lin_dev):>7.2f} {_frac(s.lin_dev < LIN_UM):>4.0f}% {_q(pd.concat([s.step_prev, s.step_next])):>5.2f}", flush=True)
print("  columns: d_gt0 = peak to its GT node; GT->ship = the GT node to the nearest shipped primary at t, '<=3' = the cell is "
      "already detected, s_score = that primary's score; peak->ship = the peak to the nearest shipped primary; prev/next det = "
      "the GT neighbour at t-1 / t+1 has a shipped primary within 3 um; con = raw contrast over the crop median at the GT "
      "node's position at t-1 / t / t+1, max2 = the maximum within 2 um at t; lin_dev = the node's distance from the midpoint "
      "of its two neighbours (interpolated annotations sit on the line), step = the track's step length per frame.", flush=True)

print("\n=== the refused peaks split by whether the annotated cell already has a shipped primary at t ===", flush=True)
s = P[P.cls == "ranker_refused"]
for lab, m in (("cell DETECTED at t (GT->ship <= 3)", s.d_gt_ship <= DET_UM), ("cell UNDETECTED at t (GT->ship > 3)", s.d_gt_ship > DET_UM)):
    r = s[m]
    if len(r):
        print(f"  {lab:<38} n={len(r):>3}  d_gt0 {_q(r.d_gt0):.2f}  peak->ship {_q(r.d_peak_ship):.2f}  con t-1/t/t+1 "
              f"{_q(r.con_prev):.2f}/{_q(r.con_gt):.2f}/{_q(r.con_next):.2f}  max2 {_q(r.con_gt_max2):.2f}  lin_dev {_q(r.lin_dev):.2f} "
              f"(<0.3: {_frac(r.lin_dev < LIN_UM):.0f}%)  q_score {_q(r.q_score):.2f}", flush=True)
print("\n=== the same split for every class: share of peaks whose annotated cell is already detected at t ===", flush=True)
for cls in ("ranker_refused", "linked", "below_tau", "other", "ALL"):
    r = _sel(cls)
    if len(r):
        print(f"  {cls:<15} n={len(r):>3}  detected {_frac(r.d_gt_ship <= DET_UM):>3.0f}%  undetected {_frac(r.d_gt_ship > DET_UM):>3.0f}%  "
              f"(GT->ship median {_q(r.d_gt_ship):.2f} um)", flush=True)
print("  reading: 'detected' peaks are second points next to a cell a shipped primary already holds -- the ranker is right to "
      "refuse them and they are not missed cells; for the 'undetected' ones, compare con t against con t-1 / t+1 (does the cell "
      "go dark at t only?) and lin_dev against the control (is the annotation at t an interpolation?).", flush=True)
warning_summary()
print(f"\nLNPLACE2 DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("LNPLACE2 OK", flush=True)
'''

src = pre + "\n" + warn + "\n" + walk + "\n" + lnload + BODY
assert src.count("def seeds_of(") == 1 and src.count("def cell_row(") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lnplace2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written lnplace2.py", len(src))
