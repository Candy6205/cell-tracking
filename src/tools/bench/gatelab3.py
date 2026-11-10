"""gatelab3 -- PAIRED comparison of gate variants (same resampled movies), with
the per-movie GT counts recovered from divbench3's log so div_J is exact under
resampling. The marginal interval in gatelab2 answers 'how well is div_J
measured'; this answers 'is variant B better than variant A', which is the
question that decides anything."""
import json
exec(open("gatelab.py").read().split("print(\"\n\" + \"=\" * 78)\nprint(\"STEP 2")[0])

PM = json.load(open("db3_per_movie.json"))
movies = np.array(sorted(PM))
gt_of = {m: PM[m]["gt"] for m in movies}
idx_of = {m: np.where(movie == m)[0] for m in movies}
assert sum(gt_of.values()) == 125

VARIANTS = {}
VARIANTS["2-fold, 20 feats (SHIPPED)"] = oof(HGB, ALL, movie, 2, quiet=True)[0]
VARIANTS["3-fold, 20 feats"] = oof(HGB, ALL, movie, 3, quiet=True)[0]
VARIANTS["5-fold, 20 feats"] = oof(HGB, ALL, movie, 5, quiet=True)[0]
KEEP6 = [FEATS.index(f) for f in ("sc_q", "dv", "sc_p", "t_frac", "pf_c1", "n_orph")]
KEEP2 = [FEATS.index(f) for f in ("sc_q", "dv")]
VARIANTS["3-fold, 6 feats (ablation-positive)"] = oof(HGB, KEEP6, movie, 3, quiet=True)[0]
VARIANTS["3-fold, 2 feats (sc_q, dv)"] = oof(HGB, KEEP2, movie, 3, quiet=True)[0]
VARIANTS["2-fold, 6 feats"] = oof(HGB, KEEP6, movie, 2, quiet=True)[0]


def J_of(p, ii, thr):
    keep = np.isfinite(p[ii]) & (p[ii] > thr)
    tp = int((lab[ii][keep] == 2).sum())
    fp = int((lab[ii][keep] == 1).sum())
    return tp, fp


print("=" * 78)
print("PAIRED BOOTSTRAP over movies (2000 draws), cut p>0.03, exact per-movie N_GT")
print("=" * 78)
rng = np.random.default_rng(20260907)
draws = [rng.choice(movies, size=len(movies), replace=True) for _ in range(2000)]
JS = {}
for nm, p in VARIANTS.items():
    tp0, fp0 = J_of(p, np.arange(len(lab)), 0.03)
    J0 = tp0 / (tp0 + fp0 + (125 - tp0))
    arr = []
    for d in draws:
        ii = np.concatenate([idx_of[m] for m in d])
        ngt = sum(gt_of[m] for m in d)
        tp, fp = J_of(p, ii, 0.03)
        arr.append(tp / max(1, tp + fp + (ngt - tp)))
    JS[nm] = np.array(arr)
    print(f"   {nm:<38} point div_J {J0:.4f} ({tp0} TP / {fp0} FP)   "
          f"boot mean {np.mean(JS[nm]):.4f} 90% [{np.quantile(JS[nm], .05):.4f}, "
          f"{np.quantile(JS[nm], .95):.4f}]")

base = "2-fold, 20 feats (SHIPPED)"
print(f"\n   PAIRED difference against '{base}' (same movies in every draw):")
for nm in VARIANTS:
    if nm == base:
        continue
    d = JS[nm] - JS[base]
    sig = "yes" if (np.quantile(d, .05) > 0 or np.quantile(d, .95) < 0) else "NO"
    print(f"   {nm:<38} d div_J {np.mean(d):+.4f}  90% [{np.quantile(d, .05):+.4f}, "
          f"{np.quantile(d, .95):+.4f}]  distinguishable from 0: {sig}")

print("\n" + "=" * 78)
print("WHAT WOULD IT TAKE? div_J as a function of how well 50 TP separate from 249 FP")
print("=" * 78)
print("   (a gate that keeps R of the 50 true forks and F of the 249 false ones)")
print(f"   {'keep TP':>8} {'keep FP':>8} {'div_J':>7} {'d score vs 0.1337':>18}")
for tp in (50, 48, 46, 44, 42):
    for fp in (0, 10, 20, 30, 45):
        J = tp / (tp + fp + (125 - tp))
        print(f"   {tp:>8} {fp:>8} {J:>7.4f} {0.1 * (J - 0.1337):>+18.4f}")
    print()
