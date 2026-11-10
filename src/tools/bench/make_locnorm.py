"""hparacache.py -> locnorm.py: does LOCAL contrast normalisation of the detector's input
surface the cells the shipped detector misses?

WHY (2026-09-17). blobsplit: the annotated cells the detector misses (1,817, 1.6%) are DIM
cells in BRIGHT surroundings -- their raw intensity is at the local background (contrast
median 0.09 against 1.81 for a clean single nucleus), 25% dimmer than the neighbour 8 um
away whose blob they sit on, with no raw maximum of their own in 93% of cases. The shipped
input normalisation is GLOBAL per volume (pool_norm: percentile 50 -> 99.5 of the pooled
frame), so a dim cell in a bright region reaches the network with near-zero contrast. This
kernel feeds the same four networks a LOCALLY normalised input and asks, at the missed
cells, whether a peak appears -- and at ordinary cells, what it costs.

ARMS (the normalisation applied to the pooled volume before pool_norm / _dc_normalize):
  GLOBAL   the shipped pass (anchor: detections must equal fdet_F4hp_ frame by frame)
  BGSUB15  minus a Gaussian local mean, sigma 15 um
  LOCZ15   (x - local mean) / (local std + 0.1 global std), sigma 15 um
  LOCZ30   the same at sigma 30 um
Gaussians run as separable conv3d on the GPU.

SITES, built in-kernel from the GT and the fdet_F4hp_ caches, on the movies with >= 30
missed cells (20 movies, ~1,140 missed cells, ~830 frames):
  missed   an annotated cell with no F4HP primary within 7 um (the scorer's radius)
  ctrl     an annotated cell with a primary within 2.5 um, no other primary within 8 um and
           no other annotated cell within 10 um (<= 10 per frame)
Per site and arm: distance to the nearest detection (score >= 0.15), the fused map's value
at the cell, a raw local maximum of the fused map (>= 0.05) within 5 um. Per frame and arm:
detections >= 0.15, detections not within 7 um of any GLOBAL detection ("new"), and how
many of those sit within 7 um of an annotated cell (the annotated share of new peaks
against the frame's overall annotated share says whether new peaks are cells or noise).

Attach: hplace's sources + biohub-hparacache. GPU T4, ~1.5 h.
"""
import os

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-17a-locnorm"')
cut = "names_all = sorted(p.name[:-len(\".zarr\")] for p in TRAIN_DIR.glob(\"*.zarr\"))"
assert src.count(cut) == 1
head = src[:src.index(cut)]

tail = r'''
# ---- LOCNORM ------------------------------------------------------------------------
LN_MODE = "GLOBAL"
LN_SIGMA = {"BGSUB15": 15.0, "LOCZ15": 15.0, "LOCZ30": 30.0}
PSCALE_UM = SCALE[0]                       # the pooled grid is isotropic 1.625 um
ARMS_LN = ["GLOBAL", "BGSUB15", "LOCZ15", "LOCZ30"]
MISS_UM = 7.0
CTRL_MATCH_UM = 2.5
LONE_PRIM_UM = 8.0
LONE_GT_UM = 10.0
N_CTRL_PER_FRAME = 10
MIN_MISSED_PER_MOVIE = 30
_pool_xy_raw = pool_xy


def _gauss3d(p, sigma_vox):
    """Separable Gaussian blur of a (Z, Y, X) float32 array on DEVICE, reflect padding."""
    t = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))[None, None].to(DEVICE)
    r = int(np.ceil(2.5 * sigma_vox))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32, device=DEVICE) / sigma_vox) ** 2)
    k = k / k.sum()
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = r
        pad[2 * (4 - dim) + 1] = r
        n = t.shape[dim]
        if n <= r:                                  # a thin axis: pad by replication instead
            t = torch.nn.functional.pad(t, pad, mode="replicate")
        else:
            t = torch.nn.functional.pad(t, pad, mode="reflect")
        t = torch.nn.functional.conv3d(t, k.view(*shape))
    return t[0, 0].cpu().numpy()


def local_norm(p, mode):
    if mode == "GLOBAL":
        return p
    s_vox = LN_SIGMA[mode] / PSCALE_UM
    mu = _gauss3d(p, s_vox)
    if mode == "BGSUB15":
        return (p - mu).astype(np.float32)
    var = _gauss3d(p * p, s_vox) - mu * mu
    sd = np.sqrt(np.clip(var, 0.0, None))
    return ((p - mu) / (sd + 0.1 * float(p.std()) + 1e-6)).astype(np.float32)


def pool_xy(vol, f=POOL):                   # shadows the name pool_norm / branch_maps4 look up at call time
    return local_norm(_pool_xy_raw(vol, f), LN_MODE)


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    off = (POOL - 1) / 2.0
    return np.array([zyx[0], (zyx[1] - off) / POOL, (zyx[2] - off) / POOL])


def map_value(h, p_vox):
    z, y, x = (int(round(v)) for v in to_pooled(p_vox))
    Z, Y, X = h.shape
    if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
        return float(h[z, y, x])
    return float("nan")


def raw_max_within(h, p_vox, R_um, thr=0.05):
    pA = to_pooled(p_vox)
    r = int(np.ceil(R_um / PSCALE_UM)) + 1
    Z, Y, X = h.shape
    z0, y0, x0 = (int(round(v)) for v in pA)
    zs, ys, xs = slice(max(0, z0 - r), min(Z, z0 + r + 1)), slice(max(0, y0 - r), min(Y, y0 + r + 1)), slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = h[zs, ys, xs]
    if sub.size == 0:
        return 0.0, float("nan")
    pk = peak_local_max(sub, min_distance=1, threshold_abs=thr, exclude_border=False)
    if len(pk) == 0:
        return 0.0, float("nan")
    pk = pk + np.array([zs.start, ys.start, xs.start])
    d = np.linalg.norm((pk - pA[None, :]) * PSCALE_UM, axis=1)
    k = int(np.argmin(d))
    return float(d[k] <= R_um), float(sub.max())


# ---- sites -------------------------------------------------------------------------------
FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
RNG = np.random.default_rng(0)
SITES = {}          # name -> {t: [(kind, g_vox)]}
COUNT = defaultdict(int)
print("building sites from the GT and the F4HP caches...", flush=True)
for name in SIX:
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1)
    per_t = defaultdict(list)
    n_miss = 0
    for t in range(T):
        rows = np.where(g_t == t)[0]
        if len(rows) == 0:
            continue
        c = np.asarray(z[f"c{t}"], float)
        s = np.asarray(z[f"s{t}"], float)
        C = c[s >= UNET_THRESH]
        if len(C) == 0:
            continue
        Cu = C * SCALE
        Gu = g_pos[rows] * SCALE
        ptree, gtree = cKDTree(Cu), cKDTree(Gu)
        d_near, i_near = ptree.query(Gu)
        miss = np.where(d_near > MISS_UM)[0]
        for k in miss:
            per_t[t].append(("missed", g_pos[rows[k]]))
        n_miss += len(miss)
        if len(miss):
            pool = []
            for k in np.where(d_near <= CTRL_MATCH_UM)[0]:
                i = int(i_near[k])
                if len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1 == 0 and len(gtree.query_ball_point(Gu[k], LONE_GT_UM)) - 1 == 0:
                    pool.append(("ctrl", g_pos[rows[k]]))
            if len(pool) > N_CTRL_PER_FRAME:
                pool = [pool[i] for i in RNG.choice(len(pool), N_CTRL_PER_FRAME, replace=False)]
            per_t[t].extend(pool)
    if n_miss >= MIN_MISSED_PER_MOVIE:
        SITES[name] = dict(per_t)
        COUNT["movies"] += 1
        COUNT["missed"] += n_miss
        COUNT["ctrl"] += sum(1 for t in per_t for k_, _g in per_t[t] if k_ == "ctrl")
        COUNT["frames"] += len(per_t)
print(f"sites: {dict(COUNT)} (movies with >= {MIN_MISSED_PER_MOVIE} missed cells)", flush=True)
assert COUNT["movies"] > 0

# ---- the pass ------------------------------------------------------------------------------
OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "locnorm_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t,gz,gy,gx," + ",".join(f"{a}_{k}" for a in ARMS_LN for k in ("d_det", "h_g", "rawmax5", "hmax5")) + "\n")
ff = open(OUT / "locnorm_frames.csv", "w", encoding="utf-8", newline="\n")
ff.write("dataset,t,n_gt," + ",".join(f"{a}_{k}" for a in ARMS_LN for k in ("n_det", "n_new", "n_new_ann", "n_det_ann")) + "\n")
RES = {a: defaultdict(list) for a in ARMS_LN}          # arm -> kind -> rows
FR = {a: defaultdict(float) for a in ARMS_LN}
ANCHOR_BAD = 0
n_frames = 0
t0 = time.time()
for k_, name in enumerate(sorted(SITES)):
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    zc = np.load(FDET[name])
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos_all = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    tm = time.time()
    for t in sorted(SITES[name]):
        vol = np.asarray(img[t])
        gt_t = g_pos_all[g_t == t]
        gtree = cKDTree(gt_t) if len(gt_t) else None
        det = {}
        for arm in ARMS_LN:
            LN_MODE = arm
            hs = branch_maps4(vol)
            h = sum(hs) / 4.0
            cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
            det[arm] = (ch[s >= UNET_THRESH] if len(ch) else np.zeros((0, 3)), h)
            if arm == "GLOBAL":
                o = np.asarray(zc[f"c{t}"], np.float32)
                if o.shape != np.asarray(ch, np.float32).shape or not np.array_equal(o, np.asarray(ch, np.float32)):
                    ANCHOR_BAD += 1
            del hs
        LN_MODE = "GLOBAL"
        g_tree = cKDTree(det["GLOBAL"][0] * SCALE) if len(det["GLOBAL"][0]) else None
        frame_row = [name, str(t), str(len(gt_t))]
        for arm in ARMS_LN:
            D, h = det[arm]
            Du = D * SCALE
            n_det = len(D)
            if g_tree is not None and n_det:
                dd, _ = g_tree.query(Du)
                new = dd > MISS_UM
            else:
                new = np.ones(n_det, bool)
            n_new = int(new.sum())
            if gtree is not None and n_det:
                dg, _ = gtree.query(Du)
                n_det_ann = int((dg <= MISS_UM).sum())
                n_new_ann = int(((dg <= MISS_UM) & new).sum())
            else:
                n_det_ann = n_new_ann = 0
            for k2, v in (("n_det", n_det), ("n_new", n_new), ("n_new_ann", n_new_ann), ("n_det_ann", n_det_ann)):
                FR[arm][k2] += v
            frame_row += [str(n_det), str(n_new), str(n_new_ann), str(n_det_ann)]
        ff.write(",".join(frame_row) + "\n")
        for kind, g in SITES[name][t]:
            row = [kind, name, str(t)] + [f"{v:.3f}" for v in g]
            for arm in ARMS_LN:
                D, h = det[arm]
                d_det = float(cKDTree(D * SCALE).query(g * SCALE)[0]) if len(D) else float("nan")
                h_g = map_value(h, g)
                rm5, hmax5 = raw_max_within(h, g, 5.0)
                RES[arm][kind].append((d_det, h_g, rm5, hmax5))
                row += [f"{d_det:.3f}", f"{h_g:.4f}", f"{rm5:.0f}", f"{hmax5:.4f}"]
            fh.write(",".join(row) + "\n")
        n_frames += 1
        del vol, det
        if n_frames % 50 == 0:
            gc.collect()
    fh.flush()
    ff.flush()
    print(f"  [{k_ + 1}/{len(SITES)}] {name}: {len(SITES[name])} frames, anchor misses so far {ANCHOR_BAD}, "
          f"{time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if k_ == 0:
        assert ANCHOR_BAD == 0, "ANCHOR FAILED on the first movie: GLOBAL does not reproduce fdet_F4hp_"
        print("  ANCHOR OK on movie 1: GLOBAL reproduces the fdet_F4hp_ cache exactly", flush=True)
fh.close()
ff.close()

# ---- summary ---------------------------------------------------------------------------------
print(f"\n=== LOCNORM: {n_frames} frames, {len(SITES)} movies; anchor misses {ANCHOR_BAD} ===", flush=True)
print(f"  {'arm':<8}{'missed found<=7um':>18}{'found<=5um':>11}{'h at cell med':>14}{'raw max<=5um':>13}{'hmax5 med':>10} | "
      f"{'ctrl found<=2.5':>15}{'ctrl d med':>11}{'ctrl lost>7':>12}", flush=True)
for arm in ARMS_LN:
    M = np.array(RES[arm]["missed"], float).reshape(-1, 4)
    C = np.array(RES[arm]["ctrl"], float).reshape(-1, 4)
    print(f"  {arm:<8}{np.mean(M[:, 0] <= 7):>18.3f}{np.mean(M[:, 0] <= 5):>11.3f}{np.nanmedian(M[:, 1]):>14.3f}{np.mean(M[:, 2] == 1):>13.3f}{np.nanmedian(M[:, 3]):>10.3f} | "
          f"{np.mean(C[:, 0] <= 2.5):>15.3f}{np.nanmedian(C[:, 0]):>11.2f}{np.mean(C[:, 0] > 7):>12.3f}", flush=True)
print(f"\n  per frame, summed: detections >= 0.15; 'new' = not within 7 um of a GLOBAL detection; annotated = within 7 um of a GT node", flush=True)
print(f"  {'arm':<8}{'n_det':>10}{'vs GLOBAL':>10}{'n_new':>9}{'new annotated':>14}{'new ann share':>14}{'all ann share':>14}", flush=True)
base = FR["GLOBAL"]["n_det"]
for arm in ARMS_LN:
    f_ = FR[arm]
    print(f"  {arm:<8}{f_['n_det']:>10,.0f}{f_['n_det'] / max(base, 1):>10.3f}{f_['n_new']:>9,.0f}{f_['n_new_ann']:>14,.0f}"
          f"{f_['n_new_ann'] / max(f_['n_new'], 1):>14.3f}{f_['n_det_ann'] / max(f_['n_det'], 1):>14.3f}", flush=True)
print("\n  reading: a normalisation that finds a real share of the missed cells (found <= 7 um) while its new peaks are "
      "annotated at about the frame's overall annotated share is adding cells; new peaks annotated far below that share "
      "are noise. Placement at ctrl cells (d med, lost > 7) is the cost on cells already found.", flush=True)
print(f"\nLOCNORM DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("LOCNORM OK" if ANCHOR_BAD == 0 else f"LOCNORM ANCHOR DRIFT on {ANCHOR_BAD} frames", flush=True)
'''
out_src = head + tail
assert "def pool_xy(vol, f=POOL):" in head and out_src.count("_pool_xy_raw = pool_xy") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "locnorm.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written locnorm.py", len(out_src))
