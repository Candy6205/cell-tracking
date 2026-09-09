"""dcmerge.py -> zhpara.py: the hpara placement, measured on ZEBRAHUB.

THE USER'S RULE, 2026-09-08: "比赛的验证集两个都不是，所以最好两个都跑，甚至
zebrahub也要跑". The hidden set is neither 6bba nor 44b6, so a placement
change has to be measured on every domain we can reach, and zebrahub is the
third instrument.

WHY dcmerge IS THE BASE AND NOT divbench9. Zebrahub has no cached edge-head
probabilities and no fork gate; `dcmerge`'s `solve_one` runs the ranker
straight off `pair_features(dets, None, T)`, so a zebrahub arm needs NOTHING
but a detection cache. `zhhpcache` produced exactly that -- `zhdet_hp_<stem>`
with the same keys as the shipped `zhdet_<stem>`, the same detections in the
same order, placed by the parabolic vertex instead of the shipped `_refine`.
So this is a one-word swap on the load, and every stage after it is
byte-identical code.

WHAT IS CUT. Everything from dcmerge's "two competition domains" loop to the
end of its report: the DeepCenter sweep, the 6bba and 44b6 domains, and the
three-domain verdict machinery, all of which ask a different question. What
is KEPT is every definition above that line -- `zh_load`, `solve_one`,
`patch_and_save`, `apply_config`, `per_sample_metrics`, `summarise`, the
metric -- so the pipeline being measured is dcmerge's, unmodified.

THE ANCHOR, and it is a hard one. dcmerge v2 recorded its zebrahub BASE row:
edge_J 0.8036, adj_J 0.8093, div_J 0.0230, on `sorted(ZH)[:8]`. The REF arm
here runs the same eight crops through the same code and must reproduce it.
If it does not, the harness moved and nothing below is readable.

Two more anchors that cost nothing and would catch the failure modes this
swap actually has:

  * the GROUND TRUTH must be identical between the arms. zhhpcache carries
    the label arrays through; if it dropped or reordered one, the two arms
    would be scored against different truths and the delta would be noise
    about the labels.
  * the DETECTION COUNT per frame must be identical. hpara moves points, it
    does not add or remove them -- hparacache's and zhhpcache's own anchors
    say so -- so a count difference here means the wrong cache was loaded.

THE MEASUREMENT. Every crop carrying both caches (41, against dcmerge's 8),
scored with the official metric against zebrahub's own dense labels; the
8-crop anchor subset is reported separately so the anchor stays legible next
to the wider number. Per-crop deltas and a paired bootstrap over crops,
because an aggregate cannot show whether one crop carries the whole thing --
the failure this comparison is most exposed to, and the one hparabench
round 1 walked into.

Attach: faustine649/biohub-zebrahub-extract (the crops),
faustine649/biohub-zh-detect (zhdet_), faustine649/biohub-zhhpcache
(zhdet_hp_), plus dcmerge's own sources. CPU only.
"""
import os

src = open("dcmerge.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION 2026-08-28h-dcmerge', 'SCRIPT_VERSION 2026-09-09a-zhpara')
if 'SCRIPT_VERSION = "2026-08-28h-dcmerge"' in src:
    rep('SCRIPT_VERSION = "2026-08-28h-dcmerge"',
        'SCRIPT_VERSION = "2026-09-09a-zhpara"')

# ---- the cache split ------------------------------------------------------
# dcmerge's glob is `zhdet_*.npz`, which now also matches `zhdet_hp_*.npz`
# and would file those under the key "hp_<stem>" -- a silent second copy of
# every crop under a name nothing looks up. Split them at the source.
rep('''ZH = {f.stem[len("zhdet_"):]: f
      for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
N_ZH = 8
SET_ZH = sorted(ZH)[:N_ZH]''',
    '''_ZH_ALL = {f.stem[len("zhdet_"):]: f
           for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
ZH_REF = {k: v for k, v in _ZH_ALL.items() if not k.startswith("hp_")}
ZH_HP = {k[len("hp_"):]: v for k, v in _ZH_ALL.items()
         if k.startswith("hp_")}
ZH = ZH_REF                     # `zh_load` reads whichever dict ZH names
N_ZH = 8
# dcmerge v2 scored `sorted(ZH)[:8]`, and its recorded zebrahub BASE row is
# the hard anchor below. Computed off ZH_REF so attaching the hpara caches
# cannot change WHICH eight crops the anchor is taken on.
SET_ZH = sorted(ZH_REF)[:N_ZH]
ANCHOR8 = list(SET_ZH)
BOTH = sorted(set(ZH_REF) & set(ZH_HP))
# v1 was OOM-KILLED after 9 of 41 crops (89 min). Two fixes, and the second
# is the one that mattered: order the crops SMALLEST FIRST so a kill costs
# the fewest, and write the CSV after every crop so a kill costs nothing at
# all. v1's nine crops survived only because they were printed to the log.
# The eight anchor crops are forced to the front regardless of size -- an
# unanchored partial run is not readable.
_size = {k: v.stat().st_size for k, v in ZH_REF.items()}
BOTH = ANCHOR8 + sorted((n for n in BOTH if n not in set(ANCHOR8)),
                        key=lambda n: _size.get(n, 0))
print(f"  zhdet caches: refine {len(ZH_REF)}, hpara {len(ZH_HP)}, "
      f"crops carrying BOTH {len(BOTH)}", flush=True)
assert len(BOTH) >= 8, (
    f"only {len(BOTH)} crops carry both caches -- attach "
    f"faustine649/biohub-zhhpcache")
assert set(ANCHOR8) <= set(BOTH), (
    "the 8 anchor crops are not all present in the hpara cache; the anchor "
    "and the measurement would be on different samples")''')

# ---- `zh_load` reads the arm's dict --------------------------------------
rep('''    z = np.load(ZH[name])''',
    '''    z = np.load(ZH[name])       # ZH is rebound per arm; see the driver''')

# ---- cut everything from the competition loop to the end ------------------
_CUT = "# ---- the two competition domains ---------------------------------------"
assert src.count(_CUT) == 1
src = src[:src.index(_CUT)]


src += r'''# ---- ZEBRAHUB: refine vs hpara, one config, two detection sources -------
#
# dcmerge's BASE arm is `_CAP` plus SHORT_MIN 4, and that is the
# configuration its recorded zebrahub row was measured under. It is also
# still what ships: gapcross's REF arm (2026-09-09) is gate 9.0, cap 0.02,
# short 4, rescue off, and it beat every alternative it was crossed against.
SHIPPED = {"GAP_NEED_CAND": False, "GAP1_CAP_FRAC": 0.02, "SHORT_MIN": 4}

# dcmerge v2, recorded in progress.md: zebrahub BASE edge_J 0.8036,
# adj_J 0.8093, div_J 0.0230 over these 8 crops. The REF arm runs the same
# code on the same crops and must reproduce it.
ZH_ANCHOR = {"edge_jaccard": 0.8036, "adj_edge_jaccard": 0.8093,
             "division_jaccard": 0.0230}
ZH_ANCHOR_TOL = 0.0005

BUDGET_S = 10.0 * 3600      # Kaggle kills at 12 h; stop cleanly first

print(f"\n=== ZEBRAHUB: does hpara's placement convert, on a third "
      f"instrument? {len(BOTH)} crops x 2 detection sources ===", flush=True)
print("  The ONLY difference between the arms is where each detection was "
      "placed inside its voxel. Same detections, same order, same count, "
      "same ground truth, same solver, same post-link configuration.",
      flush=True)
print(f"  HARD ANCHOR: the REF arm over {len(ANCHOR8)} crops must reproduce "
      f"dcmerge v2's zebrahub BASE -- edge_J 0.8036, adj_J 0.8093, "
      f"div_J 0.0230. A miss means the harness moved and nothing below is "
      f"readable.", flush=True)

def _checkpoint():
    """Rewrite the per-crop CSV after every crop. A run that is killed at
    crop 9 of 41 should still hand back nine crops of data."""
    if not done:
        return
    ks = sorted({k for t, _ in ARM_SRC for n in done for k in ZROWS[t][n]})
    with open("/kaggle/working/zhpara.csv", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("crop,arm,n_gt," + ",".join(ks) + "\n")
        for n in done:
            for t, _ in ARM_SRC:
                fh.write(f"{n},{t},{NGT[n]}," +
                         ",".join(repr(ZROWS[t][n].get(k)) for k in ks) + "\n")


GT_DIR = pathlib.Path("/kaggle/working/ps_zh_gt")
GT_DIR.mkdir(parents=True, exist_ok=True)
import shutil                                                  # noqa: E402
ARM_SRC = [("REF", ZH_REF), ("HP", ZH_HP)]
ZROWS = {tag: {} for tag, _ in ARM_SRC}
NGT = {}
MOVED = []
t_zh = time.time()
done = []
for name in BOTH:
    if time.time() - t_zh > BUDGET_S:
        print(f"  BUDGET reached after {len(done)} crops; the rest are "
              f"skipped and every table below says so", flush=True)
        break
    t0 = time.time()
    # Ground truth first, read through the REFINE cache. It does not depend
    # on the arm, so reading it once means the two arms are scored against
    # ONE object rather than two that are merely asserted to be equal.
    ZH = ZH_REF
    apply_config(None, None)
    _d0, T, gt_arr, gt_e, n_cross = zh_load(name)
    n_ref = [len(c) for c, _s in _d0]
    gp = GT_DIR / f"{name}.geff"
    dr._save(str(gp), gt_arr, gt_e)
    _r = td.graph.IndexedRXGraph.from_geff(str(gp))
    gtg = _r[0] if isinstance(_r, tuple) else _r

    # ANCHOR 1: the hpara cache must carry the SAME labels. If zhhpcache
    # dropped or reordered one, the arms would be scored against different
    # truths and the delta would be a statement about the labels.
    ZH = ZH_HP
    apply_config(None, None)
    _d1, T1, gt_arr1, gt_e1, _nc1 = zh_load(name)
    n_hp = [len(c) for c, _s in _d1]
    assert T1 == T, f"{name}: T {T1} != {T}"
    assert gt_arr1.shape == gt_arr.shape and np.allclose(gt_arr1, gt_arr), (
        f"{name}: the hpara cache carries DIFFERENT ground-truth nodes")
    assert gt_e1.shape == gt_e.shape and np.array_equal(gt_e1, gt_e), (
        f"{name}: the hpara cache carries DIFFERENT ground-truth edges")
    # ANCHOR 2: hpara RE-PLACES, it does not re-detect. Equal counts frame
    # by frame is what makes the two arms the same experiment.
    assert n_hp == n_ref, (
        f"{name}: detection COUNT differs between the arms "
        f"({sum(n_hp)} vs {sum(n_ref)}) -- this is the wrong cache")
    # And how far the points actually moved, so a null result can be told
    # apart from a no-op. refinecmp measured |hpara - refine| p50 ~0.43 um
    # on the competition domains; zebrahub is a different voxel size.
    _dd = np.concatenate([
        np.linalg.norm((np.asarray(a, float).reshape(-1, 3)
                        - np.asarray(b, float).reshape(-1, 3))
                       * SCALE[None, :], axis=1)
        for (a, _sa), (b, _sb) in zip(_d1, _d0) if len(a)])
    MOVED.append(_dd)
    del _d0, _d1

    for tag, srcd in ARM_SRC:
        ZH = srcd
        apply_config(SHIPPED, None)
        dets, T2, _ga, _ge, _nc = zh_load(name)
        nodes0, edges0, pool = solve_one(dets, T2)
        outp = pdir(tag, "zh") / f"{name}.geff"
        patch_and_save(nodes0, edges0, pool, outp)
        rr = td.graph.IndexedRXGraph.from_geff(str(outp))
        g = rr[0] if isinstance(rr, tuple) else rr
        er = compute_metric(g, gtg, scale=tuple(SCALE))
        rec = (node_recall(g, gtg)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        ZROWS[tag][name] = per_sample_metrics(er, float(len(gt_arr)), rec)
        del dets, nodes0, edges0, pool, g
        gc.collect()
    ZH = ZH_REF
    apply_config(None, None)
    NGT[name] = int(len(gt_arr))
    done.append(name)
    print(f"  [{len(done)}/{len(BOTH)}] {name}: T={T} nodes={sum(n_ref):,} "
          f"gt={len(gt_arr):,} moved p50 {np.percentile(_dd, 50):.3f} um  "
          f"adj_J REF {_adj_of(ZROWS['REF'][name]):.4f} -> HP "
          f"{_adj_of(ZROWS['HP'][name]):.4f}  ({time.time() - t0:.0f}s)",
          flush=True)
    # THE FREES GO AFTER THE PRINT, which reads gt_arr and _dd. v2 put them
    # before it and died on the first crop with `NameError: gt_arr` -- 20
    # minutes of CPU for something no syntax check can catch.
    #
    # v1 held every crop's GT graph and both prediction graphs for the whole
    # run; zebrahub crops reach 50,494 nodes and nine of them was enough to
    # be OOM-killed. The geff directories go too: they are written under
    # /kaggle/working and nothing reads them once the metric has run.
    del gtg, _r, gt_arr, gt_e, _dd, n_ref, n_hp
    for _tag, _ in ARM_SRC:
        for _g in pathlib.Path(f"/kaggle/working/ps_zh_{_tag}").glob("*.geff"):
            shutil.rmtree(_g, ignore_errors=True)
    _checkpoint()
    gc.collect()

print(f"\n  {len(done)} of {len(BOTH)} crops scored in "
      f"{(time.time() - t_zh) / 60:.1f} min", flush=True)
if MOVED:
    _all = np.concatenate(MOVED)
    print(f"  |hpara - refine| over {len(_all):,} detections: p50 "
          f"{np.percentile(_all, 50):.3f} um, p90 "
          f"{np.percentile(_all, 90):.3f}, max {_all.max():.3f}. A null "
          f"result is only readable next to this number -- if the points "
          f"did not move there is nothing to convert.", flush=True)


def agg(tag, names):
    return _with_ratio([ZROWS[tag][n] for n in names])


# ---- the anchor ---------------------------------------------------------
_a8 = [n for n in ANCHOR8 if n in done]
ANCHOR_FAIL = []
if len(_a8) == len(ANCHOR8):
    _r = agg("REF", _a8)
    print(f"\n=== HARD ANCHOR: the REF arm over dcmerge v2's own "
          f"{len(_a8)} crops ===", flush=True)
    for k, want in ZH_ANCHOR.items():
        got = float(_r[k])
        ok = abs(got - want) <= ZH_ANCHOR_TOL
        print(f"  {k:<20} got {got:.4f}  want {want:.4f}  "
              f"{'OK' if ok else 'MISS'}", flush=True)
        if not ok:
            ANCHOR_FAIL.append((k, got, want))
else:
    # The 8 anchor crops run FIRST, so this only fires if the run died
    # inside them -- in which case there is nothing to read anyway.
    ANCHOR_FAIL.append(("anchor crops incomplete", len(_a8), len(ANCHOR8)))
assert not ANCHOR_FAIL, (
    f"zebrahub anchor miss: {ANCHOR_FAIL} -- the harness is not dcmerge's "
    f"any more, so the hpara delta below is not readable")
print("  the harness is dcmerge's, unmodified", flush=True)

# ---- the measurement ----------------------------------------------------
KEYS = ["edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score",
        "total_node_ratio"]
for label, names in (("ANCHOR 8", _a8), (f"ALL {len(done)}", done)):
    print(f"\n=== ZEBRAHUB, {label} crops ===", flush=True)
    print(f"  {'arm':<8}" + "".join(f"{k:>20}" for k in KEYS), flush=True)
    rows = {}
    for tag, _ in ARM_SRC:
        rows[tag] = agg(tag, names)
        print(f"  {tag:<8}" + "".join(
            f"{float(rows[tag].get(k, float('nan'))):>20.4f}" for k in KEYS),
            flush=True)
    print(f"  {'delta':<8}" + "".join(
        f"{float(rows['HP'].get(k, float('nan'))) - float(rows['REF'].get(k, float('nan'))):>+20.4f}"
        for k in KEYS), flush=True)

# ---- per crop, because an aggregate cannot show concentration -----------
print(f"\n=== PER CROP (all {len(done)}) ===", flush=True)
print(f"  {'crop':<26}{'gt':>8}{'REF adj':>10}{'HP adj':>10}{'d adj':>10}"
      f"{'REF edge':>10}{'HP edge':>10}{'d edge':>10}", flush=True)
up_a = up_e = 0
for n in done:
    a0, a1 = _adj_of(ZROWS["REF"][n]), _adj_of(ZROWS["HP"][n])
    e0 = float(ZROWS["REF"][n].get("edge_jaccard", float("nan")))
    e1 = float(ZROWS["HP"][n].get("edge_jaccard", float("nan")))
    up_a += int(a1 > a0)
    up_e += int(e1 > e0)
    print(f"  {n:<26}{NGT[n]:>8,}{a0:>10.4f}{a1:>10.4f}{a1 - a0:>+10.4f}"
          f"{e0:>10.4f}{e1:>10.4f}{e1 - e0:>+10.4f}", flush=True)
print(f"  crops where hpara wins: adj_J {up_a}/{len(done)}, "
      f"edge_J {up_e}/{len(done)}", flush=True)

# ---- paired bootstrap over crops ---------------------------------------
print(f"\n=== PAIRED BOOTSTRAP over the {len(done)} crops, 2,000 draws, on "
      f"the AGGREGATE. The SAME resample is used for both arms, so the crop "
      f"draw cancels and what is left is the placement ===", flush=True)
_rng = np.random.default_rng(0)
_draws = [[done[i] for i in _rng.integers(0, len(done), len(done))]
          for _ in range(2000)]
_A = [(agg("REF", nm), agg("HP", nm)) for nm in _draws]
for k in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"):
    d = np.asarray([float(h.get(k, float("nan")))
                    - float(r.get(k, float("nan"))) for r, h in _A])
    d = d[d == d]
    if not len(d):
        continue
    lo, hi = np.percentile(d, [5, 95])
    print(f"  {k:<22} {np.mean(d):>+9.5f}  90% CI [{lo:>+9.5f}, {hi:>+9.5f}]"
          f"   P(>0) = {np.mean(d > 0):.3f}", flush=True)

_csv = pathlib.Path("/kaggle/working/zhpara.csv")
_ks = sorted({k for t, _ in ARM_SRC for n in done for k in ZROWS[t][n]})
with open(_csv, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("crop,arm,n_gt," + ",".join(_ks) + "\n")
    for n in done:
        for t, _ in ARM_SRC:
            fh.write(f"{n},{t},{NGT[n]}," +
                     ",".join(repr(ZROWS[t][n].get(k)) for k in _ks) + "\n")
print(f"\n  {_csv.name} written ({len(done)} crops x 2 arms)", flush=True)

print("\n  HOW TO READ IT. Zebrahub is the third instrument and the only "
      "one whose ground truth is dense, so a sign here is worth more than "
      "its size. It is also a DIFFERENT pipeline from the competition "
      "arm -- dcmerge's solver has no edge head and no fork gate -- so "
      "this answers 'does better placement help the LINKER', not 'does it "
      "help the shipped submission'. hparabench answers the second.",
      flush=True)

if len(done) == len(BOTH):
    print("ZHPARA OK", flush=True)
else:
    print(f"ZHPARA INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)
'''

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
out = os.path.abspath(os.path.join(base, "zhpara.py"))
open(out, "w", encoding="utf-8", newline="\n").write(src)
print("written zhpara.py", len(src))
