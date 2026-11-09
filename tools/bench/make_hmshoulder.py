"""hplace.py -> hmshoulder.py: is a missed annotated cell a SECOND BLOB under its
host's heatmap response, findable without the annotation?

WHY (2026-09-18). gtpeak: at 95% of the annotated cells F4 misses, the map
responds (0.10 at the cell, 0.15 within 3.25 um) but has no local maximum
there even at the 0.05 floor; the nearest maximum is 6-7 um away on a
neighbour. lnplace / lnplace2: the raw image carries no placement signal at
these cells (contrast 0.17-0.19 against 0.64 at detected cells), blobsplit
closed the raw-image splitter (AUC 0.70-0.77, 16 wrong splits per cell found).
The heatmap is the one place left where the cell leaves a mark: a shoulder on
the host's blob. This measures whether that shoulder can be located blind --
by fitting the host's blob and reading the residual -- and what the same
procedure invents at ordinary cells (the false-fire rate that priced blobsplit
out).

WHAT IT DOES. The shipped heatmap pass (branch_maps4: xl b0, b1, b2 and ep05
DeepCenter; f4 = their mean on the pooled grid, 1.625 um isotropic), anchored
frame by frame against biohub-hparacache's fdet_F4hp_ caches (the hpara half
of detect_both must reproduce c{t} exactly). Sites, built in the kernel from
the GT and the caches with blobsplit's definitions:
  host   an annotated cell with no shipped primary within 7 um (the scorer's
         radius) and one within 12 um -- that primary is the host
  ctrl   a primary holding an annotated cell (2.5 um), no other primary
         within 8 um, no other annotated cell within 10 um
  dense  a primary holding an annotated cell with another primary within 8 um
Controls come from the host frames only (2 of each per frame, seeded), so the
GPU pass touches only frames that carry a host.

At every site, blind to the annotation, on the f4 map within 13 um of the
host with voxels within 6 um of any OTHER shipped primary left out:
  1. one Gaussian blob + constant fitted to the host (least squares);
  2. the residual's maximum at least 3 um from the host (position, value);
  3. a radial-median residual (the map minus its shell median around the
     fitted centre -- a plateau-shaped host leaves no ring here) and ITS
     maximum;
  4. two Gaussian blobs + constant, started from 2 and from 3, lower SSE kept:
     the second blob's centre, amplitude, width, and the SSE drop.
A second blob is a FIRE when it sits >= 4 um from every shipped primary; at a
host site it is a HIT when it is also within 3 um of the missed cell. Ordinary
cells price the fires: at 30 / 50 / 70% hit recall, the fire rate on clean
and dense singles and the wrong-per-hit ratio at blobsplit's base rate.
Host diagnostics (never a feature): f4 at the cell, its maximum within 2 um,
the dip along host -> cell, the value 2 um beyond the cell, whether any of the
five maps has a local maximum within 3 um of the cell at the 0.05 floor, and
the pooled raw intensity at the cell against the frame's 40th percentile (the
DeepCenter-style supervision trains everything below it as background).

Outputs: hmshoulder_sites.csv (one row per site), the printed summary,
HMSHOULDER OK / ANCHOR DRIFT.

Kaggle: T4 (hplace's sources; kernel source biohub-hparacache).
"""
import os
import sys

src = open("hplace.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-13b-hplace"', 'SCRIPT_VERSION = "2026-09-18a-hmshoulder"')
rep("from scipy.spatial import cKDTree\n", "from scipy import ndimage as _ndi\nfrom scipy.optimize import least_squares\nfrom scipy.spatial import cKDTree\n")

DOC = '''"""hmshoulder -- is a missed annotated cell a second blob under its host's
heatmap response, findable without the annotation? See
tools/bench/make_hmshoulder.py for the design. Built from hplace.py: the same
branch_maps4 / detect_both pass, anchored against biohub-hparacache.

SCRIPT_VERSION 2026-09-18a-hmshoulder
"""
'''
doc_end = src.index('"""', 3) + 3
assert src.startswith('"""') and doc_end < src.index("import base64"), doc_end
head = DOC + src[doc_end:src.index("# ---- HPLACE:")]
helpers = src[src.index("UNPOOL_OFF = "):src.index("def load_sites")] + src[src.index("def to_pooled"):src.index("SUB = [")]
assert "SITES_B64" not in head and "SITES_B64" not in helpers

BLOCK = r'''
# ---- HMSHOULDER: a second blob under the host's response, blind to the annotation ------------------
MISS_UM = 7.0            # a cell with no shipped primary within this is missed (the scorer's radius)
HOST_UM = 12.0           # the host = the nearest shipped primary, within this
CTRL_MATCH_UM = 2.5
LONE_PRIM_UM = 8.0
LONE_GT_UM = 10.0
N_CTRL_PER_FRAME = 2     # clean singles and dense singles, from the host frames only
PATCH_UM = 13.0          # fit window: the pooled f4 map within this of the host
MASK_UM = 6.0            # voxels within this of ANOTHER shipped primary are left out of the fit
EXCL_UM = 3.0            # the second blob is looked for at least this far from the host
FIRE_UM = 4.0            # a second blob within this of any shipped primary is not a fire
HIT_UM = 3.0             # a second blob within this of the missed cell is a hit
SIG0 = 1.3               # starting sigma, pooled voxels (2.1 um)
LM_FLOOR = 0.05          # peak-finding floor for 'does map X have a local maximum near the cell'
LM_UM = 3.0
RNG = np.random.default_rng(0)


def build_sites(gt, prim):
    """gt: t -> annotated positions (um); prim: per t (coords_vox, scores) of the shipped primaries.
    host: an annotated cell with no primary within MISS_UM and one within HOST_UM (its host);
    ctrl: a primary holding an annotated cell (2.5 um), no other primary within 8, no other GT within 10;
    dense: a primary holding an annotated cell with another primary within 8 um.
    Controls are drawn from the host frames only, N_CTRL_PER_FRAME of each, seeded."""
    hosts, ctrl, dense = defaultdict(list), defaultdict(list), defaultdict(list)
    for t in sorted(gt):
        if t >= len(prim):
            continue
        C = np.asarray(prim[t][0], float).reshape(-1, 3)
        S = np.asarray(prim[t][1], float).reshape(-1)
        if len(C) == 0:
            continue
        g_pos = np.asarray(gt[t], float).reshape(-1, 3)
        if len(g_pos) == 0:
            continue
        Cu = C * SCALE
        ptree = cKDTree(Cu)
        gtree = cKDTree(g_pos)
        d_near, i_near = ptree.query(g_pos)
        for k in np.where(d_near > MISS_UM)[0]:
            if d_near[k] <= HOST_UM:
                i = int(i_near[k])
                n8 = len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1
                hosts[t].append(dict(kind="host", t=t, d=C[i].copy(), s=float(S[i]), g=g_pos[k] / SCALE,
                                     d_gnear=float(d_near[k]), n8=n8))
        for k in np.where(d_near <= CTRL_MATCH_UM)[0]:
            i = int(i_near[k])
            n8 = len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1
            n_gt = len(gtree.query_ball_point(g_pos[k], LONE_GT_UM)) - 1
            row = dict(kind="", t=t, d=C[i].copy(), s=float(S[i]), g=None, d_gnear=float(d_near[k]), n8=n8)
            if n8 == 0 and n_gt == 0:
                row["kind"] = "ctrl"
                ctrl[t].append(row)
            elif n8 >= 1:
                row["kind"] = "dense"
                dense[t].append(row)
    out = []
    for t in sorted(hosts):
        out.extend(hosts[t])
        for pool in (ctrl[t], dense[t]):
            if len(pool) > N_CTRL_PER_FRAME:
                pick = RNG.choice(len(pool), N_CTRL_PER_FRAME, replace=False)
                pool = [pool[i] for i in sorted(pick)]
            out.extend(pool)
    return out


def um_grid(P):
    """Full-resolution um of pooled grid points (N, 3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    return np.stack([P[:, 0] * SCALE[0], (P[:, 1] * POOL + UNPOOL_OFF) * SCALE[1],
                     (P[:, 2] * POOL + UNPOOL_OFF) * SCALE[2]], 1)


def um_pooled(p):
    return um_grid(np.asarray(p, float)[None, :])[0]


def _gauss(P, c, sg):
    return np.exp(-0.5 * ((P - c) ** 2).sum(1) / (sg * sg))


def fit_site(m, pd_, others_um):
    """m: the pooled f4 map; pd_: the host in pooled coords; others_um: every other shipped primary (um).
    Blind to the annotation. Returns the one-blob fit, the residual maximum, the radial-median residual
    maximum and the two-blob fit; positions in pooled coords. None when the window is too small."""
    Z, Y, X = m.shape
    r = int(np.ceil(PATCH_UM / PSCALE[0]))
    z0, y0, x0 = (int(round(float(v))) for v in pd_)
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return None
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = np.asarray(m[zs, ys, xs], np.float64)
    gz, gy, gx = np.mgrid[zs, ys, xs]
    P = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1).astype(np.float64)
    v = sub.ravel()
    Pu = um_grid(P)
    hu = um_pooled(pd_)
    keep = np.linalg.norm(Pu - hu, axis=1) <= PATCH_UM
    others_um = np.asarray(others_um, float).reshape(-1, 3)
    if len(others_um):
        keep &= cKDTree(others_um).query(Pu)[0] >= MASK_UM
    P, v, Pu = P[keep], v[keep], Pu[keep]
    if len(v) < 200:
        return None
    a0 = max(float(m[z0, y0, x0]), 0.05)
    pd_ = np.asarray(pd_, float)

    def f1(p):
        return p[0] * _gauss(P, p[1:4], p[4]) + p[5]

    lo1 = [0.0, pd_[0] - 2.0, pd_[1] - 2.0, pd_[2] - 2.0, 0.6, 0.0]
    hi1 = [2.0, pd_[0] + 2.0, pd_[1] + 2.0, pd_[2] + 2.0, 4.0, 0.6]
    x1 = np.clip(np.array([a0, pd_[0], pd_[1], pd_[2], SIG0, 0.0]), lo1, hi1)
    r1 = least_squares(lambda p: f1(p) - v, x1, bounds=(lo1, hi1), method="trf", max_nfev=300)
    p1 = r1.x
    res = v - f1(p1)
    sse1 = float((res ** 2).sum())
    c1 = p1[1:4]
    rr = np.linalg.norm(um_grid(P) - um_pooled(c1), axis=1)
    far = rr >= EXCL_UM
    if not far.any():
        return None
    k = int(np.argmax(np.where(far, res, -np.inf)))
    out = dict(n_fit=int(len(v)), A1=float(p1[0]), s1=float(p1[4]), b1=float(p1[5]), sse1=sse1,
               c1=c1.copy(), r_res=float(res[k]), c_res=P[k].copy())
    # the radial-median residual: the map minus its shell median around the fitted centre
    sh = np.round(rr / PSCALE[0]).astype(int)
    prof = np.zeros(int(sh.max()) + 1)
    for j in range(len(prof)):
        mj = sh == j
        prof[j] = float(np.median(v[mj])) if mj.any() else 0.0
    rad = v - np.interp(rr, np.arange(len(prof)) * PSCALE[0], prof)   # the shell profile, interpolated in radius
    k2 = int(np.argmax(np.where(far, rad, -np.inf)))
    out.update(r_rad=float(rad[k2]), c_rad=P[k2].copy())

    # two blobs, started from each residual maximum; the lower SSE wins
    def f2(p):
        return p[0] * _gauss(P, p[1:4], p[4]) + p[5] + p[6] * _gauss(P, p[7:10], p[10])

    lo2 = lo1 + [0.0, float(zs.start), float(ys.start), float(xs.start), 0.6]
    hi2 = hi1 + [2.0, float(zs.stop - 1), float(ys.stop - 1), float(xs.stop - 1), 4.0]
    starts = [(res[k], P[k])]
    if np.linalg.norm((P[k] - P[k2]) * PSCALE) > 2.0:
        starts.append((rad[k2], P[k2]))
    best = None
    for amp, c in starts:
        x2 = np.clip(np.concatenate([p1, [max(float(amp), 0.01)], c, [SIG0]]), lo2, hi2)
        r2 = least_squares(lambda p: f2(p) - v, x2, bounds=(lo2, hi2), method="trf", max_nfev=400)
        sse2 = float(((f2(r2.x) - v) ** 2).sum())
        if best is None or sse2 < best[0]:
            best = (sse2, r2.x)
    sse2, p2 = best
    A1b, c1b, s1b, bb, A2, c2, s2 = p2[0], p2[1:4], p2[4], p2[5], p2[6], p2[7:10], p2[10]
    if np.linalg.norm((c2 - pd_) * PSCALE) < np.linalg.norm((c1b - pd_) * PSCALE):   # the fit swapped the blobs
        A1b, c1b, s1b, A2, c2, s2 = A2, c2, s2, A1b, c1b, s1b
    out.update(A1b=float(A1b), s1b=float(s1b), bb=float(bb), A2=float(A2), s2=float(s2), c2=np.asarray(c2, float).copy(),
               sse2=sse2, dsse=float((sse1 - sse2) / max(sse1, 1e-12)))
    return out


def map_at(m, p):
    z, y, x = (int(round(float(v))) for v in p)
    Z, Y, X = m.shape
    if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
        return float(m[z, y, x])
    return float("nan")


def max_near(m, p, R):
    """Maximum of the map within R um of pooled point p."""
    Z, Y, X = m.shape
    r = int(np.ceil(R / PSCALE[0])) + 1
    z0, y0, x0 = (int(round(float(v))) for v in p)
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = m[zs, ys, xs]
    if sub.size == 0:
        return float("nan")
    gz, gy, gx = np.mgrid[zs, ys, xs]
    d = np.sqrt(((gz - p[0]) * PSCALE[0]) ** 2 + ((gy - p[1]) * PSCALE[1]) ** 2 + ((gx - p[2]) * PSCALE[2]) ** 2)
    inR = d <= R
    return float(sub[inR].max()) if inR.any() else float("nan")


def along(m, pa, pb, step_um=0.5):
    """The map sampled (trilinear) from pooled pa to pooled pb."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    n = max(3, int(np.ceil(np.linalg.norm((pb - pa) * PSCALE) / step_um)) + 1)
    pts = pa[None, :] + (pb - pa)[None, :] * np.linspace(0.0, 1.0, n)[:, None]
    return _ndi.map_coordinates(m, pts.T, order=1, mode="nearest")


def dip_g(m, pg, pd_):
    """min of the map along cell -> host, over the value at the cell (1 = monotone rise, no dip)."""
    vals = along(m, pg, pd_)
    return float(vals.min() / max(float(vals[0]), 1e-6))


def far2_g(m, pg, pd_, um=2.0):
    """The map 2 um beyond the cell, away from the host, over the value at the cell."""
    pg, pd_ = np.asarray(pg, float), np.asarray(pd_, float)
    u = (pg - pd_) * PSCALE
    nrm = np.linalg.norm(u)
    if nrm < 1e-6:
        return float("nan")
    q = pg + (u / nrm) * um / PSCALE
    vals = along(m, pg, q)
    return float(vals[-1] / max(float(vals[0]), 1e-6))


def ball_mean(vol_p, p, r=1):
    z, y, x = (int(round(float(v))) for v in p)
    Z, Y, X = vol_p.shape
    sub = vol_p[max(0, z - r):min(Z, z + r + 1), max(0, y - r):min(Y, y + r + 1), max(0, x - r):min(X, x + r + 1)]
    return float(sub.mean()) if sub.size else float("nan")


# ---- the caches, the sites, the frame loop ------------------------------------------------------------
FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
print(f"F4hp caches: {len(SIX)} 6bba movies", flush=True)

MAP_NAMES = ["f4", "b0", "b1", "b2", "dc"]
COLS = (["s_host", "n8", "d_gnear", "d_dg", "f4_g", "f4_g2", "f4_d", "dip_g", "far2_g"]
        + [f"lm3_{mp}" for mp in MAP_NAMES] + ["I_g", "I_d", "p40", "n_fit", "A1", "s1", "b1", "sse1",
        "r_res", "d_res_g", "d_res_d", "d_res_prim", "r_rad", "d_rad_g", "d_rad_d", "d_rad_prim",
        "A1b", "s1b", "A2", "s2", "dsse", "d_c2_g", "d_c2_d", "d_c2_prim"])
OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "hmshoulder_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t,dz,dy,dx,gz,gy,gx," + ",".join(COLS) + "\n")
RES = defaultdict(list)
COUNT = defaultdict(int)
t0 = time.time()
n_frames = n_bad = n_ann = n_missed = n_prim_all = 0
ANCHOR_FAILS = []
nan = float("nan")
for k_, nm in enumerate(SIX):
    tm = time.time()
    zf = np.load(FDET[nm])
    T = max(int(k[1:]) for k in zf.files if k.startswith("c")) + 1
    prim = []
    for t in range(T):
        c = np.asarray(zf[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(zf[f"s{t}"], float).reshape(-1)
        prim.append((c[s >= UNET_THRESH], s[s >= UNET_THRESH]))
        n_prim_all += int((s >= UNET_THRESH).sum())
    gtg = read_geff(geff_path(ROOT, "train", nm))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    gt = {int(tt): g_pos[g_t == tt] for tt in np.unique(g_t)}
    n_ann += len(g_t)
    sites = build_sites(gt, prim)
    n_missed += sum(1 for r_ in sites if r_["kind"] == "host")
    by_t = defaultdict(list)
    for r_ in sites:
        by_t[r_["t"]].append(r_)
    if not by_t:
        print(f"  [{k_ + 1}/{len(SIX)}] {nm}: no host site", flush=True)
        continue
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    for t in sorted(by_t):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        maps = dict(zip(["b0", "b1", "b2", "dc"], hs))
        maps["f4"] = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        o_ = np.asarray(zf[f"c{t}"], np.float32)
        n_ = np.asarray(ch, np.float32)
        if o_.shape != n_.shape or not np.array_equal(o_, n_):
            n_bad += 1
            if len(ANCHOR_FAILS) < 6:
                ANCHOR_FAILS.append((nm, t, o_.shape, n_.shape))
        praw = pool_xy(vol)
        p40 = float(np.percentile(praw[::2, ::2, ::2], 40))
        C_t = np.asarray(prim[t][0], float).reshape(-1, 3)
        Cu_t = C_t * SCALE
        ptree = cKDTree(Cu_t) if len(Cu_t) else None
        f4 = maps["f4"]
        for row in by_t[t]:
            kind = row["kind"]
            d = np.asarray(row["d"], float)
            du = d * SCALE
            pd_ = to_pooled(d)
            others = Cu_t[np.linalg.norm(Cu_t - du, axis=1) > 0.5] if len(Cu_t) else np.zeros((0, 3))
            o = {c_: nan for c_ in COLS}
            o["s_host"], o["n8"], o["d_gnear"] = row["s"], row["n8"], row["d_gnear"]
            g = row["g"]
            if g is not None:
                gu = np.asarray(g, float) * SCALE
                pg = to_pooled(g)
                o["d_dg"] = float(np.linalg.norm(gu - du))
                o["f4_g"] = map_at(f4, pg)
                o["f4_g2"] = max_near(f4, pg, 2.0)
                o["f4_d"] = map_at(f4, pd_)
                o["dip_g"] = dip_g(f4, pg, pd_)
                o["far2_g"] = far2_g(f4, pg, pd_)
                for mp in MAP_NAMES:
                    o[f"lm3_{mp}"] = place_one(maps[mp], vol, pg, thr=LM_FLOOR, R=LM_UM)["d"]
                o["I_g"] = ball_mean(praw, pg)
            o["I_d"] = ball_mean(praw, pd_)
            o["p40"] = p40
            fit = fit_site(f4, pd_, others)
            if fit is not None:
                for kk in ("n_fit", "A1", "s1", "b1", "sse1", "r_res", "r_rad", "A1b", "s1b", "A2", "s2", "dsse"):
                    o[kk] = fit[kk]
                for tag, ck in (("res", "c_res"), ("rad", "c_rad"), ("c2", "c2")):
                    cu = um_pooled(fit[ck])
                    o[f"d_{tag}_d"] = float(np.linalg.norm(cu - du))
                    o[f"d_{tag}_prim"] = float(ptree.query(cu)[0]) if ptree is not None else nan
                    if g is not None:
                        o[f"d_{tag}_g"] = float(np.linalg.norm(cu - gu))
            RES[kind].append(o)
            COUNT[kind] += 1
            gs = ",".join(f"{v:.3f}" for v in g) if g is not None else "nan,nan,nan"
            fh.write(f"{kind},{nm},{t},{d[0]:.3f},{d[1]:.3f},{d[2]:.3f},{gs},"
                     + ",".join(f"{o[c_]:.5g}" if o[c_] == o[c_] else "nan" for c_ in COLS) + "\n")
        n_frames += 1
        del hs, maps, vol, praw, f4
    fh.flush()
    zf.close()
    del img
    gc.collect()
    print(f"  [{k_ + 1}/{len(SIX)}] {nm}: {len(by_t)} frames, sites host {sum(1 for r_ in sites if r_['kind'] == 'host')} "
          f"ctrl {sum(1 for r_ in sites if r_['kind'] == 'ctrl')} dense {sum(1 for r_ in sites if r_['kind'] == 'dense')}, "
          f"anchor misses so far {n_bad}, {time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if n_frames and k_ == 0:
        assert n_bad == 0, f"ANCHOR FAILED on the first movie: {ANCHOR_FAILS[:3]}"
        print("  ANCHOR OK on movie 1: every computed frame reproduces fdet_F4hp_ exactly", flush=True)
fh.close()


# ---- summary -----------------------------------------------------------------------------------------
def q(vals, f):
    v = np.array([x for x in vals if x == x], float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


def share(vals, pred):
    v = [x for x in vals if x == x]
    return float(np.mean([pred(x) for x in v])) if v else float("nan")


def col(kind, c_):
    return [o[c_] for o in RES[kind]]


from sklearn.metrics import roc_auc_score  # noqa: E402

H, Cc, Dd = RES["host"], RES["ctrl"], RES["dense"]
print(f"\n=== HMSHOULDER: {n_frames} frames, sites host {len(H)} / ctrl {len(Cc)} / dense {len(Dd)}; anchor misses {n_bad}; "
      f"annotated GT nodes {n_ann:,}, missed at {MISS_UM} um with a host within {HOST_UM} um: {n_missed:,} "
      f"({n_missed / max(1, n_ann):.2%}); shipped primaries in these movies {n_prim_all:,} ===", flush=True)
print(f"\n  --- the missed cell g against its host d (n = {len(H)}) ---", flush=True)
print(f"  d(d, g) median {q(col('host', 'd_dg'), .5):.2f} um (p25 {q(col('host', 'd_dg'), .25):.2f}, p75 {q(col('host', 'd_dg'), .75):.2f}); "
      f"host score median {q(col('host', 's_host'), .5):.2f}; other primaries within 8 um of the host: median {q(col('host', 'n8'), .5):.0f}, "
      f">= 1 share {share(col('host', 'n8'), lambda x: x >= 1):.2f}", flush=True)
print(f"  f4 at g median {q(col('host', 'f4_g'), .5):.3f} (p25 {q(col('host', 'f4_g'), .25):.3f}, p75 {q(col('host', 'f4_g'), .75):.3f}), "
      f"max within 2 um {q(col('host', 'f4_g2'), .5):.3f}, >= 0.05 share {share(col('host', 'f4_g2'), lambda x: x >= 0.05):.2f}; "
      f"f4 at d median {q(col('host', 'f4_d'), .5):.3f}", flush=True)
print(f"  dip along g -> d (min / value at g) median {q(col('host', 'dip_g'), .5):.3f}, >= 0.97 (no dip: a monotone rise to the host) "
      f"share {share(col('host', 'dip_g'), lambda x: x >= 0.97):.2f}, < 0.8 share {share(col('host', 'dip_g'), lambda x: x < 0.8):.2f}; "
      f"value 2 um beyond g / at g median {q(col('host', 'far2_g'), .5):.3f}, < 0.9 share {share(col('host', 'far2_g'), lambda x: x < 0.9):.2f}", flush=True)
lm_any = [min([o[f'lm3_{mp}'] for mp in MAP_NAMES[1:] if o[f'lm3_{mp}'] == o[f'lm3_{mp}']] or [nan]) for o in H]
print("  a local maximum (floor 0.05) within 3 um of g: " + ", ".join(
    f"{mp} {share([0.0 if o[f'lm3_{mp}'] != o[f'lm3_{mp}'] else 1.0 for o in H], lambda x: x == 1.0):.2f}" for mp in MAP_NAMES)
    + f"; any single map {share([0.0 if x != x else 1.0 for x in lm_any], lambda x: x == 1.0):.2f}", flush=True)
bg = [1.0 if o["I_g"] <= o["p40"] else 0.0 for o in H if o["I_g"] == o["I_g"]]
print(f"  pooled raw intensity at g <= the frame's 40th percentile (trained as background) share {np.mean(bg) if bg else nan:.2f}; "
      f"I(g) / I(d) median {q([o['I_g'] / o['I_d'] for o in H if o['I_d'] == o['I_d'] and o['I_d'] > 0], .5):.3f}; "
      f"controls' I(d) <= p40 share ctrl {share([1.0 if o['I_d'] <= o['p40'] else 0.0 for o in Cc], lambda x: x == 1.0):.2f}, "
      f"dense {share([1.0 if o['I_d'] <= o['p40'] else 0.0 for o in Dd], lambda x: x == 1.0):.2f}", flush=True)
print(f"\n  --- the fits (medians host / ctrl / dense) ---", flush=True)
for c_, lab in (("n_fit", "voxels fitted"), ("A1", "one-blob amplitude"), ("s1", "one-blob sigma (pooled vox)"), ("b1", "one-blob floor"),
                ("A2", "second blob amplitude"), ("s2", "second blob sigma"), ("dsse", "SSE drop, two blobs vs one"),
                ("r_res", "residual max (>= 3 um from host)"), ("r_rad", "radial-median residual max")):
    print(f"  {lab:<36}{q(col('host', c_), .5):>9.3f}{q(col('ctrl', c_), .5):>9.3f}{q(col('dense', c_), .5):>9.3f}", flush=True)
print(f"\n  --- where the second point lands, at the hosts (d -> g in um; fire = >= {FIRE_UM} um from every shipped primary; "
      f"hit = fire and within {HIT_UM} um of g) ---", flush=True)
print(f"  {'method':<24}{'d->g med':>10}{'p75':>8}{'<=3um':>8}{'fire':>8}{'hit':>8}{'d->host med':>13}{'fire ctrl':>11}{'fire dense':>12}", flush=True)
METHODS = (("two-blob centre c2", "c2", "A2"), ("one-blob residual max", "res", "r_res"), ("radial residual max", "rad", "r_rad"))
for lab, tag, sc in METHODS:
    dg = col("host", f"d_{tag}_g")
    fire_h = [1.0 if o[f"d_{tag}_prim"] >= FIRE_UM else 0.0 for o in H if o[f"d_{tag}_prim"] == o[f"d_{tag}_prim"]]
    hit_h = [1.0 if (o[f"d_{tag}_prim"] >= FIRE_UM and o[f"d_{tag}_g"] <= HIT_UM) else 0.0 for o in H if o[f"d_{tag}_g"] == o[f"d_{tag}_g"]]
    print(f"  {lab:<24}{q(dg, .5):>10.2f}{q(dg, .75):>8.2f}{share(dg, lambda x: x <= HIT_UM):>8.2f}"
          f"{np.mean(fire_h) if fire_h else nan:>8.2f}{np.mean(hit_h) if hit_h else nan:>8.2f}{q(col('host', f'd_{tag}_d'), .5):>13.2f}"
          f"{share([o[f'd_{tag}_prim'] for o in Cc], lambda x: x >= FIRE_UM):>11.2f}{share([o[f'd_{tag}_prim'] for o in Dd], lambda x: x >= FIRE_UM):>12.2f}", flush=True)
print(f"\n  --- separation and price (a site's score is 0 unless its second point fires) ---", flush=True)
N_PRIM = 2_091_008
frac_host = n_missed / max(1, n_ann)
print(f"  base rate: hosts ~ {frac_host:.2%} of primaries (~{frac_host * N_PRIM:,.0f} of {N_PRIM:,}); a wrong fire puts a second point "
      f"next to a real cell (~1 edge at risk), a hit recovers ~2 edges (break-even ~2 wrong per hit)", flush=True)


def scored(rows, tag, sc):
    out = []
    for o in rows:
        v = o[sc]
        f = o[f"d_{tag}_prim"]
        out.append(v if (v == v and f == f and f >= FIRE_UM) else 0.0)
    return np.asarray(out, float)


for lab, tag, sc in METHODS + (("two-blob SSE drop", "c2", "dsse"),):
    sh_, sc_, sd_ = scored(H, tag, sc), scored(Cc, tag, sc), scored(Dd, tag, sc)
    hit = np.asarray([1.0 if (o[f"d_{tag}_g"] == o[f"d_{tag}_g"] and o[f"d_{tag}_g"] <= HIT_UM) else 0.0 for o in H], float)
    if len(sh_) < 30 or len(sc_) < 30 or len(sd_) < 30:
        continue
    auc_c = roc_auc_score(np.r_[np.ones(len(sh_)), np.zeros(len(sc_))], np.r_[sh_, sc_])
    auc_d = roc_auc_score(np.r_[np.ones(len(sh_)), np.zeros(len(sd_))], np.r_[sh_, sd_])
    print(f"  {lab} ({sc}): AUC host vs clean {auc_c:.3f}, host vs dense {auc_d:.3f}; host fires {np.mean(sh_ > 0):.2f}, "
          f"hits {np.mean((sh_ > 0) & (hit > 0)):.2f}", flush=True)
    hs_ = sh_[(sh_ > 0) & (hit > 0)]
    for rec in (0.3, 0.5, 0.7):
        need = rec * len(sh_)
        if len(hs_) < need:
            print(f"    {rec:.0%} hit recall unreachable: at most {len(hs_) / len(sh_):.0%} of the hosts get a hit at any threshold", flush=True)
            continue
        thr = float(np.sort(hs_)[::-1][int(np.ceil(need)) - 1])
        fc, fd = float((sc_ >= thr).mean()), float((sd_ >= thr).mean())
        f_all = 0.5 * (fc + fd)
        wrong = f_all * (1 - frac_host) / max(rec * frac_host, 1e-9)
        print(f"    at {rec:.0%} hit recall (thr {thr:.3f}): fires on {fc:.3f} of clean singles, {fd:.3f} of dense singles "
              f"-> ~{wrong:.1f} wrong fires per hit (break-even ~2)", flush=True)
print("\n  reading: if the two-blob centre lands within 3 um of g for most hosts while ordinary cells rarely fire at that "
      "amplitude, the shoulder is a placement source and a heatmap-side splitter has a price; if the fires at "
      "ordinary cells cost more than ~2 per hit, the shoulder is not separable from the map's own texture, as the raw "
      "image was not (blobsplit). Control fires are an upper bound: an unannotated cell next to a control is a real "
      "second blob the annotation cannot confirm.", flush=True)
print(f"\nHMSHOULDER DONE in {(time.time() - t0) / 60:.1f} min; hmshoulder_sites.csv written", flush=True)
print("HMSHOULDER OK" if n_bad == 0 else f"HMSHOULDER ANCHOR DRIFT on {n_bad} frames", flush=True)
'''

out_src = head + helpers + BLOCK
for must in ("def build_sites(", "def fit_site(", "def place_one(", "def to_pooled(", "from scipy.optimize import least_squares",
             "from scipy import ndimage as _ndi", "fdet_F4hp_", "HMSHOULDER OK", 'SCRIPT_VERSION = "2026-09-18a-hmshoulder"'):
    assert out_src.count(must) >= 1, must
assert out_src.count("def load_sites") == 0 and "SITES_B64" not in out_src
assert out_src.count("MASK_UM = 6.0") == 1 and out_src.count("N_CTRL_PER_FRAME = 2") == 1
out = sys.argv[1] if len(sys.argv) > 1 else "hmshoulder.py"
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print(f"wrote {out}: {len(out_src.splitlines())} lines")
