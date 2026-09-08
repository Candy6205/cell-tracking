"""gatelab2 -- what the fork gate would have to be to close the 0.244 -> 0.40 gap,
and how much of the difference between gate variants is noise."""
exec(open("gatelab.py").read().split("print(\"\n\" + \"=\" * 78)\nprint(\"STEP 2")[0])

p3, _ = oof(HGB, ALL, movie, 3, label="3-fold", quiet=True)
p5, _ = oof(HGB, ALL, movie, 5, label="5-fold", quiet=True)

print("\n" + "=" * 78)
print("A -- div_J depends ONLY on the ranking WITHIN the 299 counted forks")
print("    (uncounted forks never enter div_J; they only cost edges in adj_J)")
print("=" * 78)
print(f"   {'TP kept':>8} {'FP kept':>8} {'div_J':>7} {'d score':>8}   <- 2-fold ranking")
pc = p2[counted]
lc = lab[counted]
o = np.argsort(-np.nan_to_num(pc, nan=-1))
for keep_tp in (50, 48, 46, 44, 42, 40, 36, 30):
    # walk down the ranking until keep_tp true forks have been kept
    tp = fp = 0
    for i in o:
        if lc[i] == 2:
            tp += 1
        else:
            fp += 1
        if tp == keep_tp:
            break
    J = tp / (tp + fp + (N_GT - tp))
    print(f"   {tp:>8} {fp:>8} {J:>7.4f} {0.1 * (J - 0.1337):>+8.4f}")
print("   (reading: to keep all 50 true forks this ranking drags every false one "
      "with it;\n    the shipped cut keeps 42 and sheds 202 of 249)")

print("\n" + "=" * 78)
print("B -- how big is the noise? bootstrap over MOVIES, 2000 draws, cut p>0.03")
print("=" * 78)
rng = np.random.default_rng(20260907)
movies = np.unique(movie)
idx_of = {m: np.where(movie == m)[0] for m in movies}
# GT divisions per movie = TP + (this movie's share of FN). FN is not per-fork,
# so resample movies and scale N_GT by the number of movies drawn.
gt_per_movie = {}
for m in movies:
    gt_per_movie[m] = int((lab[idx_of[m]] == 2).sum())
tot_tp = sum(gt_per_movie.values())
# N_GT per movie is unknown from the table; distribute the 125 GT events in
# proportion to TP is WRONG. Instead resample and hold the per-movie unknown
# FN fixed by scaling: N_GT_draw = 125 * (#movies drawn with replacement)/128,
# which is exact in expectation.
for name, p in (("2-fold", p2), ("3-fold", p3), ("5-fold", p5)):
    Js = []
    for _ in range(2000):
        draw = rng.choice(movies, size=len(movies), replace=True)
        ii = np.concatenate([idx_of[m] for m in draw])
        keep = np.isfinite(p[ii]) & (p[ii] > 0.03)
        tp = int((lab[ii][keep] == 2).sum())
        fp = int((lab[ii][keep] == 1).sum())
        Js.append(tp / max(1, tp + fp + (N_GT - tp)))
    Js = np.array(Js)
    print(f"   {name}: div_J {np.mean(Js):.4f}  90% interval "
          f"[{np.quantile(Js, 0.05):.4f}, {np.quantile(Js, 0.95):.4f}]  "
          f"width {np.quantile(Js, 0.95) - np.quantile(Js, 0.05):.4f}")

print("\n" + "=" * 78)
print("C -- feature ablation, 3-fold (drop one at a time; pooled AUC on counted)")
print("=" * 78)
base_p, base_a = oof(HGB, ALL, movie, 3, quiet=True)
print(f"   all {len(FEATS)}: pooled AUC {base_a:.4f}")
rows = []
for j, f in enumerate(FEATS):
    cols = [c for c in ALL if c != j]
    _, a = oof(HGB, cols, movie, 3, quiet=True)
    rows.append((a - base_a, f, a))
for d, f, a in sorted(rows):
    print(f"   drop {f:<9} AUC {a:.4f}  {d:+.4f}")

print("\n" + "=" * 78)
print("D -- the two fork sources: does one gate for both cost anything?")
print("=" * 78)
src = X[:, FEATS.index("src")].astype(int)
for s, nm in ((1, "patch"), (0, "ilp")):
    m = counted & (src == s)
    tp_ = int((lab[m] == 2).sum())
    fp_ = int((lab[m] == 1).sum())
    a = auc((lab[m] == 2).astype(int), p3[m])
    print(f"   {nm:<6} counted {int(m.sum()):>3} ({tp_} TP / {fp_} FP)  "
          f"3-fold AUC on this source alone {a:.4f}")
    for thr in (0.03,):
        k = m & (p3 > thr)
        print(f"          at p>{thr}: keeps {int((lab[k] == 2).sum())} TP, "
              f"{int((lab[k] == 1).sum())} FP")
