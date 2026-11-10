"""hparacache.py -> lnrescue.py: the RESCUE detection cache -- the shipped F4HP detections
plus the peaks a locally normalised pass finds where the shipped pass found nothing.

WHY (2026-09-17). locnorm: with the input's local background (Gaussian mean, sigma 15 um)
subtracted before the same four networks, 19% of the annotated cells the shipped pass
misses get a detection within 7 um, but shipped cells drift (1.3% of clean singles lost
beyond 7 um) and half the new peaks are noise. So the normalised pass is used only to ADD:
a BGSUB15 detection is kept iff no shipped F4HP detection (score >= 0.15) lies within
RESCUE_UM of it; every shipped detection stays exactly where it was.

OUTPUT per 6bba movie: fdet_F4ln_<name>.npz with c{t} / s{t} = the fdet_F4hp_ arrays with
the new peaks appended after them (score = the BGSUB15 fused value, >= 0.15), and nnew{t} =
how many were appended (they are the LAST nnew{t} primaries of the frame, so a bench can
drop the ones below a score threshold by rank). Per movie the log prints the new peaks by
score bin with their annotated share (within 7 um of a GT node) against the shipped
detections' share, so the threshold can be chosen before the heads run.

Attach: hplace's sources + biohub-hparacache. GPU T4: one normalised 4-map pass per frame,
~0.9 s/frame, 128 movies ~3.2 h.
"""
import os

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-17a-lnrescue"')
cut = "names_all = sorted(p.name[:-len(\".zarr\")] for p in TRAIN_DIR.glob(\"*.zarr\"))"
assert src.count(cut) == 1
head = src[:src.index(cut)]

tail = r'''
# ---- LNRESCUE ---------------------------------------------------------------------------
LN_MODE = "GLOBAL"
LN_SIGMA_UM = 15.0
PSCALE_UM = SCALE[0]
RESCUE_UM = 7.0
BUDGET_S = 11.0 * 3600
_pool_xy_raw = pool_xy


def _gauss3d(p, sigma_vox):
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
        t = torch.nn.functional.pad(t, pad, mode="replicate" if t.shape[dim] <= r else "reflect")
        t = torch.nn.functional.conv3d(t, k.view(*shape))
    return t[0, 0].cpu().numpy()


def local_norm(p, mode):
    if mode == "GLOBAL":
        return p
    mu = _gauss3d(p, LN_SIGMA_UM / PSCALE_UM)
    return (p - mu).astype(np.float32)


def pool_xy(vol, f=POOL):                   # shadows the name pool_norm / branch_maps4 look up at call time
    return local_norm(_pool_xy_raw(vol, f), LN_MODE)


FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
OUT = pathlib.Path("/kaggle/working")
BINS = [(0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 1.01)]
TOT = defaultdict(float)
t_all = time.time()
done = 0
for k_, name in enumerate(SIX):
    if time.time() - t_all > BUDGET_S:
        print(f"BUDGET: stopping before {name}", flush=True)
        break
    tm = time.time()
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    out = {}
    stat = defaultdict(float)
    for t in range(T):
        c0 = np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3)
        s0 = np.asarray(z[f"s{t}"], np.float32).reshape(-1)
        prim = c0[s0 >= UNET_THRESH] * SCALE
        vol = np.asarray(img[t])
        LN_MODE = "BGSUB15"
        hs = branch_maps4(vol)
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        LN_MODE = "GLOBAL"
        del hs, vol
        keep = s >= UNET_THRESH
        ch, s = ch[keep], s[keep]
        if len(ch) and len(prim):
            d, _ = cKDTree(prim).query(ch * SCALE)
            new = d > RESCUE_UM
        else:
            new = np.ones(len(ch), bool)
        cn, sn = ch[new].astype(np.float32), s[new].astype(np.float32)
        out[f"c{t}"] = np.concatenate([c0, cn]) if len(cn) else c0
        out[f"s{t}"] = np.concatenate([s0, sn]) if len(sn) else s0
        out[f"nnew{t}"] = np.int32(len(cn))
        gt_t = g_pos[g_t == t]
        gtree = cKDTree(gt_t) if len(gt_t) else None
        stat["shipped"] += len(prim)
        if gtree is not None and len(prim):
            stat["shipped_ann"] += int((gtree.query(prim)[0] <= RESCUE_UM).sum())
        if len(cn):
            dg = gtree.query(cn * SCALE)[0] if gtree is not None else np.full(len(cn), np.inf)
            for lo, hi in BINS:
                m = (sn >= lo) & (sn < hi)
                stat[f"new_{lo:.2f}"] += int(m.sum())
                stat[f"newann_{lo:.2f}"] += int((m & (dg <= RESCUE_UM)).sum())
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"fdet_F4ln_{name}.npz", **out)
    done += 1
    for k2, v in stat.items():
        TOT[k2] += v
    n_new = sum(stat[f"new_{lo:.2f}"] for lo, _hi in BINS)
    n_new_ann = sum(stat[f"newann_{lo:.2f}"] for lo, _hi in BINS)
    print(f"  [{done}/{len(SIX)}] {name}: T={T}, shipped primaries {stat['shipped']:,.0f} (annotated {stat['shipped_ann'] / max(1, stat['shipped']):.3f}), "
          f"new peaks {n_new:,} (+{n_new / max(1, stat['shipped']):.1%}; annotated {n_new_ann / max(1, n_new):.3f}); by score "
          + " ".join(f"[{lo:.2f}: {stat[f'new_{lo:.2f}']:.0f}/{stat[f'newann_{lo:.2f}']:.0f}]" for lo, _hi in BINS)
          + f"  {time.time() - tm:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    del img, out, z
    gc.collect()

print(f"\n=== LNRESCUE over {done} movies: shipped primaries {TOT['shipped']:,.0f}, annotated share {TOT['shipped_ann'] / max(1, TOT['shipped']):.4f} ===", flush=True)
print(f"  {'new-peak score':<16}{'count':>9}{'+% of shipped':>14}{'annotated':>10}{'ann share':>10}{'x shipped share':>16}", flush=True)
base = TOT["shipped_ann"] / max(1, TOT["shipped"])
for lo, hi in BINS:
    n, a = TOT[f"new_{lo:.2f}"], TOT[f"newann_{lo:.2f}"]
    print(f"  [{lo:.2f}, {hi:.2f})   {n:>9,.0f}{n / max(1, TOT['shipped']):>14.2%}{a:>10,.0f}{a / max(1, n):>10.4f}{(a / max(1, n)) / max(base, 1e-9):>16.2f}", flush=True)
print("  reading: 'x shipped share' near 1 means the new peaks in that bin are cells as often as the shipped detections "
      "are; well below 1 means noise. Choose the bench's score threshold from this table.", flush=True)
print(f"\nLNRESCUE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("LNRESCUE OK" if done == len(SIX) else f"LNRESCUE INCOMPLETE {done}/{len(SIX)}", flush=True)
'''
out_src = head + tail
assert out_src.count("_pool_xy_raw = pool_xy") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lnrescue.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written lnrescue.py", len(out_src))
