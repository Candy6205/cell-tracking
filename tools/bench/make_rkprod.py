"""rkpool.py -> rkprod.py: turn the MEASUREMENT into a SHIPPABLE pkl.

WHAT IS MISSING BETWEEN rkpool AND A SUBMISSION. rkpool measured the retrain
with two OUT-OF-FOLD rankers over the bench's movies -- 135 movies each, fitted
inside the bench so the 128 scored movies could be read out of fold. A
submission cannot use those: `submit_joint.py` loads ONE ranker trained on the
**D175 pool** (`MIN_POOL = 170`), the 175 train movies that exclude both
holdout groups. That pkl does not exist for F4 primaries. This kernel makes it.

The estimator, the columns, the seed, the validation split and the refit are
`edgeprob_rank2.py:1707-1731` exactly -- `HistGradientBoostingClassifier(
max_iter=300, random_state=0)`, a seeded 10% split for the val AUC, then a
refit on the whole pool. The ONLY difference from what ships is which table the
rows come from:

    edge_ranker_prod_base_bidir.pkl   LAB  = BASE primaries  (the CONTROL)
    edge_ranker_prod_f4_bidir.pkl     LAB4 = F4 primaries    (the CANDIDATE)

⚠ THE ANCHOR IS THE WHOLE REASON THE CONTROL IS TRAINED TOO. The leaderboard
control for the F4 pkl is ref 56086066 (LB 0.944), which loaded
`edge_ranker_ep175_bidir.pkl` -- a pkl this kernel did not make. So before the
F4 pkl can be read as a single-variable change, a BASE pkl built by THIS code
has to be shown to behave like the shipped one. The kernel scores both on the
12 HOLDOUT_6BBA movies' BASE rows, which are in neither pool, and reports the
number that actually matters to the pipeline: **how often the two disagree
about which side of the production floor 0.48 a candidate pair falls on**,
because that is what decides whether the ILP is offered the pair at all.

  * disagreement ~ 0  -> the swap is single-variable. Submit the F4 pkl alone
    and read it against 0.944.
  * disagreement large -> the BASE pkl is also a change, so BOTH pkls get
    submitted and the F4 one is read against the BASE one, not against 0.944.

The kernel does not decide that; it prints it.

WHAT IS DELETED. Stage B and everything after it. rkpool already solved the
128 movies twice and the answer is in the record (+0.0023 [+0.0007, +0.0040],
P 0.990, 77/128 movies up). Re-solving here would cost three hours to learn
nothing. The full F4 features are not kept either -- only the labelled rows --
which is what lets all 199 movies be labelled on BOTH primaries in one pass.

COST: no GPU, no submission slot. Features for 199 movies on two primary sets,
then two fits.

Attach rkpool's sources, unchanged.
"""
import os

src = open("rkpool.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-rkpool"',
    'SCRIPT_VERSION = "2026-09-10a-rkprod"')

assert src.count('RANK_TRAIN = "F4"') == 1
# RANK_TRAIN stays "F4" so the feature loop labels all 199 movies on F4 rows;
# both pkls are trained below, so the constant no longer selects an arm. Say so
# where the constant is, rather than leaving a reader to work it out.
rep('RANK_TRAIN = "F4"',
    'RANK_TRAIN = "F4"        # rkprod: NOT an arm selector any more -- it is\n'
    '                         # what makes the loop build LAB4 for all 199.\n'
    '                         # Both pkls are trained, from LAB and from LAB4.')

# ---- 1. do not keep the full F4 features; rkprod never solves --------------
rep('''        if _solved:
            FEATS[name] = (seeds4, feats4)
        if role == "6bba" or RANK_TRAIN == "F4":
            LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4
        if not _solved:            # pool-only 44b6: labelled, then released
            del seeds4, feats4''',
    '''        # rkprod solves nothing, so the full features are NEVER kept -- only
        # the labelled rows. That is what lets all 199 movies be labelled on
        # BOTH primary sets inside one kernel's memory.
        LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4, seeds4, feats4''')

# ---- 2. cut stage B and everything after it --------------------------------
MARK = "# ---- STAGE A: two out-of-fold +BIDIR rankers ---"
assert src.count(MARK) == 1, src.count(MARK)
src = src[:src.index(MARK)] + r'''# ---- THE TWO PRODUCTION RANKERS -------------------------------------------
PROD_FLOOR_CHK = 0.48        # ref 56030644 / LB 0.933: the ILP candidate floor
COLS = ARMS["+BIDIR"]
assert COLS[:NFEAT_BASE] == list(range(NFEAT_BASE)), COLS
EXTRA = [NEW_NAMES[c] for c in COLS[NFEAT_BASE:]]
assert EXTRA == ["p_fwd", "p_rev", "h_fused"], EXTRA
assert len(COLS) == 21, len(COLS)

# The pool is the shipped one, asserted rather than assumed: a ranker that saw
# an evaluation movie would make every local number on this project unreadable.
PROD_POOL = list(POOL175)
assert len(PROD_POOL) == 175, len(PROD_POOL)
assert not (set(PROD_POOL) & set(HOLDOUT_6BBA)), "pool overlaps the 6bba holdout"
assert not (set(PROD_POOL) & set(SET_44B6)), "pool overlaps the 44b6 draw"
assert set(PROD_POOL) == set(POOL_BIDIR), (
    f"this pool is not the shipped ranker's: only-here "
    f"{sorted(set(PROD_POOL) - set(POOL_BIDIR))[:4]} only-there "
    f"{sorted(set(POOL_BIDIR) - set(PROD_POOL))[:4]}")
for _t, _tab in (("LAB", LAB), ("LAB4", LAB4)):
    _need = [n for n in PROD_POOL if n not in _tab]
    assert not _need, f"{_t} is missing {len(_need)} pool movies: {_need[:4]}"

print(f"\n=== THE TWO PRODUCTION RANKERS, D175 pool ({len(PROD_POOL)} movies) ===",
      flush=True)
print("  Estimator, columns, seed, split and refit = edgeprob_rank2.py:1707-1731\n"
      "  exactly (HistGB max_iter 300, random_state 0, seeded 10% val split,\n"
      "  then a refit on the whole pool). The ONLY difference between the two\n"
      "  rows below is which table the rows come from.", flush=True)

MODELS_PROD, ROWS_PROD = {}, {}
for tag, table, fname in (("BASE", LAB, "edge_ranker_prod_base_bidir.pkl"),
                          ("F4", LAB4, "edge_ranker_prod_f4_bidir.pkl")):
    t0 = time.time()
    Xa = np.concatenate([table[n][0] for n in PROD_POOL]).astype(np.float32)[:, COLS]
    ya = np.concatenate([table[n][1] for n in PROD_POOL])
    perm = np.random.default_rng(0).permutation(len(ya))
    n_val = max(1000, len(ya) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa[ti], ya[ti])
    val_auc = _auc(ya[vi], m.predict_proba(Xa[vi])[:, 1])
    m.fit(Xa, ya)                          # refit on the whole pool
    MODELS_PROD[tag] = m
    ROWS_PROD[tag] = (len(ya), int(ya.sum()), val_auc)
    with open(f"/kaggle/working/{fname}", "wb") as f:
        pickle.dump({"model": m,
                     "cols": list(COLS),
                     "pool": list(PROD_POOL),
                     "extra_cols": EXTRA,
                     "feature_names": BASE_NAMES + EXTRA,
                     "n_features": len(COLS),
                     "h_fused": "2*p_fwd*p_rev/(p_fwd+p_rev); p_fwd where "
                                "p_rev is NaN",
                     "version": SCRIPT_VERSION,
                     "rank_train": tag,
                     "detector": "F4",
                     "n_train_rows": int(len(ya)),
                     "n_train_pos": int(ya.sum()),
                     "val_auc": float(val_auc)}, f)
    print(f"  {tag:<4} {fname}  {len(ya)} rows ({ya.mean():.3f} pos)  "
          f"val AUC {val_auc:.4f}  {time.time() - t0:.0f}s", flush=True)
    del Xa, ya
    gc.collect()

# ---- THE ANCHOR -----------------------------------------------------------
# Both pkls are new. The leaderboard control for the F4 one is ref 56086066,
# which loaded a pkl this kernel did not make, so the question that decides
# whether ONE submission is enough is: does a BASE-trained ranker from THIS
# code behave like the shipped one? Read on the 12 HOLDOUT_6BBA movies' BASE
# rows -- in neither pool, so held out for both models.
print(f"\n=== THE ANCHOR: this code's BASE ranker against the SHIPPED one ===",
      flush=True)
print(f"  Read on the {len(HOLDOUT_6BBA)} HOLDOUT_6BBA movies' BASE rows, "
      f"held out of both pools.", flush=True)
_hx = np.concatenate([LAB[n][0] for n in HOLDOUT_6BBA]).astype(np.float32)
_hy = np.concatenate([LAB[n][1] for n in HOLDOUT_6BBA])
p_ship = MODEL_BIDIR.predict_proba(_hx[:, MCOLS_BIDIR])[:, 1]
p_ctl = MODELS_PROD["BASE"].predict_proba(_hx[:, COLS])[:, 1]
p_f4 = MODELS_PROD["F4"].predict_proba(_hx[:, COLS])[:, 1]
print(f"  {len(_hy)} rows, {int(_hy.sum())} positive", flush=True)
print(f"  {'pair':<22}{'AUC':>9}{'max|dp|':>10}{'mean|dp|':>10}"
      f"{'floor 0.48 side differs':>26}", flush=True)


def _cmp(nm, pa, pb):
    d = np.abs(pa - pb)
    side = float(((pa > PROD_FLOOR_CHK) != (pb > PROD_FLOOR_CHK)).mean())
    print(f"  {nm:<22}{_auc(_hy, pb):>9.4f}{d.max():>10.5f}{d.mean():>10.6f}"
          f"{side * 100:>25.3f}%", flush=True)
    return side


print(f"  {'SHIPPED ep175_bidir':<22}{_auc(_hy, p_ship):>9.4f}"
      f"{0.0:>10.5f}{0.0:>10.6f}{'--':>26}", flush=True)
_s_ctl = _cmp("this code, BASE", p_ship, p_ctl)
_s_f4 = _cmp("this code, F4", p_ship, p_f4)

print("\n  READING, fixed here before the numbers are looked at:", flush=True)
print("    the BASE row's floor-side disagreement is the control's own\n"
      "    contribution. If it is SMALL against the F4 row's, the F4 pkl is a\n"
      "    single-variable swap and is submitted alone, read against LB 0.944.\n"
      "    If the two are comparable, the BASE pkl is itself a change and BOTH\n"
      "    pkls get submitted, the F4 one read against the BASE one.", flush=True)
print(f"    BASE {_s_ctl * 100:.3f}%   F4 {_s_f4 * 100:.3f}%   ratio "
      f"{(_s_f4 / _s_ctl if _s_ctl else float('inf')):.1f}x", flush=True)

print(f"\nRKPROD DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("RKPROD OK", flush=True)
'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "rkprod.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
