"""divtrip.py -> divscore.py: the two triple scorers trained on everything, pickled for the submission.

WHY (2026-09-25). hypz / hypz3: a division variable per pooled triple, priced by scorer B's logit, adds +0.002 over the
shipped composition after the gate (Z10). The hidden-test submission has to compute the triples itself, so it needs the
scorers as files: scorer A (geometry only) picks the top 20 triples per frame, scorer B (geometry + raw-image columns)
prices them. divtrip fitted both out of fold for the measurement and pickled nothing. The user (2026-09-25): "不缺就搭吧".

WHAT. divtrip.py's functions verbatim (the geometry, the labeller, the image features, _hgb / _w / fit_oof / fit_full),
with a new main: pass 1 on the 199 train movies (every triple, the positives + ~6,000 negatives per movie, 121 positives)
-> scorer A trained on all of it (its out-of-fold AUC by divtrip's 5 movie folds printed for the record); the pooled
candidate table (biohub-divtrip's divtrip_candidates.csv, 389,914 rows, 83 positives) -> scorer B trained on all of it
(out of fold printed too); S0 recorded as hypz's -9.0 (the logit at which a triple costs DIV_W) and the rule behind it
(1.15 discounted triples per frame) with the measured out-of-fold quantile. Output divscore.pkl: {"A": model, "B": model,
"geo": GEO, "img": IMG, "topk": 20, "s0": -9.0, "geometry": {R_PARENT, R_SISTER, R_MID, D_AB_MIN, SNAP_UM, SCORE_MIN},
"image": {FI_*}, "n_train": {...}, "auc_oof": {...}, "version"}.
Attach: divtrip's sources (competition data, faustine649/cell-tracking, biohub-hparacache, biohub-hpara44) + the kernel
faustine649/biohub-divtrip. CPU, ~15 min.
"""
src = open("divtrip.py", encoding="utf-8").read()
head = src[:src.index('if __name__ == "__main__":')]
head = head.replace('SCRIPT_VERSION = "', 'SCRIPT_VERSION = "2026-09-25a-divscore"  # divtrip was "', 1)
head = head.replace("import pandas as pd\n", "import pandas as pd\nimport pickle\n", 1)
assert head.count("2026-09-25a-divscore") == 1

MAIN = r'''
Z_S0 = -9.0                 # hypz: the logit of scorer B at which a triple costs DIV_W (0.55); 0.10 per logit above it
Z_PER_FRAME = 1.15          # the rule behind it: about this many triples per frame get a discount


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    t_all = time.time()
    FDET = _walk_files(_npz_prefix("fdet_F4hp_"))
    names = sorted(FDET)
    print(f"movies with F4hp caches: {len(names)}", flush=True)
    assert len(names) == 199, "attach biohub-hparacache + biohub-hpara44 (divtrip had 199)"
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(names))
    FOLD = {nm: int(perm[i] % N_FOLDS) for i, nm in enumerate(names)}

    # ---- pass 1: every triple of every movie, the positives + a uniform negative sample (divtrip verbatim) ------
    rows, ylab, fold_of = [], [], []
    n_pos_all, n_tri_all, n_frames = 0, 0, 0
    t_p1 = time.time()
    for k_, name in enumerate(names):
        det = np.load(FDET[name])
        T = max(int(k[1:]) for k in det.files if k.startswith("c")) + 1
        gts = gt_triples(read_geff(geff_path(ROOT, "train", name)))
        PR = {t: primaries(det, t) for t in range(T)}
        um = {t: PR[t][0] * SCALE for t in range(T)}
        keep_X, keep_y = [], []
        for t in range(T - 1):
            n_frames += 1
            P, S = um[t], PR[t][1]
            Pn, Sn = um[t + 1], PR[t + 1][1]
            idx, X = triples_frame(P, S, Pn, Sn, um[t - 1] if t >= 1 else None, um[t + 2] if t + 2 < T else None, t, T)
            if not len(idx):
                continue
            y, _m = label_triples(idx, P, Pn, gts.get(t, []))
            n_tri_all += len(idx)
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            take = min(len(neg), max(1, int(round(N_NEG_PER_MOVIE / max(1, T - 1)))))
            sel = np.concatenate([pos, rng.choice(neg, take, replace=False)]) if len(neg) else pos
            keep_X.append(X[sel])
            keep_y.append(y[sel])
            n_pos_all += len(pos)
        if keep_X:
            rows.append(np.concatenate(keep_X))
            ylab.append(np.concatenate(keep_y))
            fold_of.append(np.full(sum(len(x) for x in keep_y), FOLD[name]))
        if (k_ + 1) % 40 == 0:
            print(f"  [{k_ + 1}/{len(names)}] {time.time() - t_p1:.0f}s", flush=True)
    X1 = np.concatenate(rows); y1 = np.concatenate(ylab); f1 = np.concatenate(fold_of)
    print(f"=== pass 1 ({(time.time() - t_p1) / 60:.1f} min): {n_tri_all:,} triples over {n_frames:,} frames, sample "
          f"{len(y1):,} rows, {int(y1.sum())} positives ===", flush=True)
    pA = fit_oof(X1, y1, f1, N_FOLDS, tag="scorer A (geometry) out of fold")
    okA = np.isfinite(pA)
    auc_A = float(roc_auc_score(y1[okA], pA[okA]))
    A_full = fit_full(X1, y1)
    print(f"  scorer A pooled out-of-fold AUC {auc_A:.4f}; trained on all {len(y1):,} rows", flush=True)

    # ---- scorer B on the pooled candidate table (divtrip pass 2's output) ---------------------------------------
    hit = _walk_files(lambda fn: fn if fn == "divtrip_candidates.csv" else None)
    assert hit, "attach the kernel faustine649/biohub-divtrip (divtrip_candidates.csv)"
    C = pd.read_csv(hit["divtrip_candidates.csv"])
    for c_ in ["movie", "fold", "t", "p", "a", "b", "label", "p_A"] + GEO + IMG:
        assert c_ in C.columns, c_
    XB = C[GEO + IMG].to_numpy(float); yB = C["label"].to_numpy(int); fB = C["fold"].to_numpy(int)
    assert (C["p_A"].notna()).all() and C.groupby(["movie", "t"]).size().max() <= TOPK, "the pool is not divtrip's top-20"
    pB = fit_oof(XB, yB, fB, N_FOLDS, tag="scorer B (geometry + image) out of fold")
    okB = np.isfinite(pB)
    auc_B = float(roc_auc_score(yB[okB], pB[okB]))
    B_full = fit_full(XB, yB)
    sB = _logit(pB[okB])
    n_fr_pool = C.loc[okB].groupby(["movie", "t"]).ngroups
    s_sorted = np.sort(sB)[::-1]
    s0_rule = float(s_sorted[min(len(s_sorted) - 1, int(round(Z_PER_FRAME * n_fr_pool)))])
    s_full = _logit(B_full.predict_proba(XB)[:, 1])
    print(f"  scorer B pooled out-of-fold AUC {auc_B:.4f}; trained on all {len(yB):,} pooled rows ({int(yB.sum())} positives, "
          f"{C.movie.nunique()} movies); out-of-fold logit p50 / p90 / p99 {np.quantile(sB, .5):.2f} / {np.quantile(sB, .9):.2f} / "
          f"{np.quantile(sB, .99):.2f}; S0 by the {Z_PER_FRAME}-per-frame rule {s0_rule:.2f} (hypz shipped {Z_S0}); "
          f"the full-fit model's logits on its own pool: above {Z_S0} for {int((s_full > Z_S0).sum()):,} rows "
          f"({int(((s_full > Z_S0) & (yB == 1)).sum())} positives) against {int((sB > Z_S0).sum()):,} out of fold "
          f"({int(((sB > Z_S0) & (yB[okB] == 1)).sum())})", flush=True)

    blob = {"A": A_full, "B": B_full, "geo": list(GEO), "img": list(IMG), "topk": TOPK, "s0": Z_S0,
            "per_frame_rule": Z_PER_FRAME, "s0_oof_rule": s0_rule,
            "geometry": {"R_PARENT": R_PARENT, "R_SISTER": R_SISTER, "R_MID": R_MID, "D_AB_MIN": D_AB_MIN,
                         "SNAP_UM": SNAP_UM, "SCORE_MIN": SCORE_MIN},
            "image": {"half_um": FI_HALF_UM, "peak_um": FI_PEAK_UM, "size_um": FI_SIZE_UM, "sigma_vox": list(FI_SIGMA_VOX)},
            "n_train": {"A_rows": int(len(y1)), "A_pos": int(y1.sum()), "B_rows": int(len(yB)), "B_pos": int(yB.sum()),
                        "movies": len(names)},
            "auc_oof": {"A": auc_A, "B": auc_B}, "version": SCRIPT_VERSION,
            "hgb": "HistGradientBoostingClassifier(max_iter=200, lr=0.05, min_samples_leaf=20, early_stopping=False), "
                   "positives weighted to the negatives' mass"}
    with open(OUT / "divscore.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    print(f"  divscore.pkl written ({(OUT / 'divscore.pkl').stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"DIVSCORE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("DIVSCORE OK", flush=True)
'''
out = head + MAIN
assert out.count("import pickle") == 1 and "import pandas as pd" in out, "divtrip's imports"
open("divscore.py", "w", encoding="utf-8", newline="\n").write(out)
print("written divscore.py", out.count("\n"), "lines")
