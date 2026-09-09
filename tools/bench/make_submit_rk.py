"""submit_joint.py -> submit_rk.py / submit_rkctl.py: the SAME pipeline with
the edge ranker retrained on the primaries it is actually deployed on, and
DIV_W moved to 0.35 with it.

WHAT CHANGES. Two constants and one file:

    RANKER_NAME   edge_ranker_ep175_bidir.pkl -> edge_ranker_prod_f4_bidir.pkl
    ILP_DIV_W     0.55 -> 0.35

Nothing else -- not the detector, not the placement, not the floor, not the
fork gate, not the patches, not the 21 columns. The new pkl comes from
`rkprod`, which trains on the SAME D175 pool with the SAME estimator, seed,
validation split and refit as `edgeprob_rank2.py:1707-1731`; the only
difference is that its rows are built from F4 primaries (the xl three-branch
mean PLUS ep05 DeepCenter) instead of BASE ones (that mean alone), which is
what the pipeline actually solves.

WHAT IT MEASURED. rkpool, 128 6bba movies, out-of-fold rankers, official
scorer: ungated REF 0.9266 -> 0.9290, **+0.0023**, 90% [+0.0007, +0.0040],
P 0.990, 77/128 movies up, not concentrated (top 8 = 27% of the gain). The
control reproduced seven recorded numbers exactly. fk44f4 then gave it 44b6's
71 movies: **+0.0019**, 90% [-0.0004, +0.0048], P 0.906.

⚠ WHY DIV_W MOVES WITH IT, and why that is not a second variable. The retrain
LOSES in the shipped configuration: REF + joint gate reads 0.9410 against what
ships at 0.9424. It only pays once the solver is allowed to make more forks --
dw35 + joint = **0.9449**, the highest cell measured anywhere on this project.
And DIV_W 0.35 on its own is worth +0.0002 (0.9424 -> 0.9426), so it is not an
independent driver; it is the operating point at which the retrained ranker's
extra division candidates can be used. Shipping the retrain at 0.55 would be
shipping a configuration already measured as negative.

⚠ THE GAIN IS ENTIRELY DIVISIONS, and one channel is mildly against it. On
44b6's 71 movies adj_edge_jaccard is flat to four decimals (-0.0001) and
edge_jaccard is NEGATIVE at P 0.084 -- the closest thing to a distinguishable
loss in this candidate's record. The score rises because 0.1 x (+0.0201 of
division_jaccard) outweighs it.

⚠ AND NO MECHANISM IS OFFERED. Three accounts of why the retrain gains have
died on measurement: augmentation (flat held-out AUC across all four cells of
rkhp's 2x2, 0.99250-0.99267), the train/deploy mismatch (rkhp's best cell has
the LARGEST mismatch), and fork-population coherence (jointgate: fixing it is
worth -0.0005 here). The +0.0023 stands; the explanation does not exist. By the
2026-08-28 rule this is the weaker of the two live candidates, and it is being
submitted anyway because submission slots are not scarce (5/day, 19 days) and
this is the cheapest available test of that rule.

⚠ DO NOT COMBINE WITH hpara. rkhp: hpara alone 0.9298, this alone 0.9290, both
together 0.9282 where additive would be 0.9321; the retrain ON TOP of hpara is
-0.0016 with a 90% interval of [-0.0030, -0.0001].

THE TWO ARMS THIS GENERATOR WRITES. submit_rk.py ships the F4 pkl. Whether
submit_rkctl.py is ALSO needed is decided by rkprod's anchor, not here: the
leaderboard control for the F4 pkl is ref 56086066 (LB 0.944), which loaded a
pkl rkprod did not make, so if rkprod's BASE pkl does not behave like the
shipped one then the BASE arm has to be submitted too and the F4 one read
against it rather than against 0.944.

ATTACH `faustine649/biohub-rkprod` AS A KERNEL SOURCE, in addition to
submit_joint's sources. biohub-edgeprob-rank may stay attached -- the pkl
filenames differ, so the /kaggle/input walk cannot pick up the wrong one.
"""
import os

base = open("submit_joint.py", encoding="utf-8").read()

ARMS = [
    # out file,         SCRIPT_VERSION tag, pkl,                          tag
    ("submit_rk.py", "2026-09-10a-submit-rk",
     "edge_ranker_prod_f4_bidir.pkl", "F4",
     "the ranker RETRAINED on F4 primaries -- the coordinates the pipeline "
     "actually solves"),
    ("submit_rkctl.py", "2026-09-10a-submit-rkctl",
     "edge_ranker_prod_base_bidir.pkl", "BASE",
     "rkprod's BASE-trained CONTROL -- same pool, same estimator, same seed, "
     "BASE primaries, i.e. what ships, rebuilt. Submitted ONLY if rkprod's "
     "anchor says its behaviour differs from edge_ranker_ep175_bidir.pkl"),
]

for out_name, version, pkl, rank_train, what in ARMS:
    src = base

    def rep(old, new, n=1):
        global src
        assert src.count(old) == n, (out_name, src.count(old), old[:70])
        src = src.replace(old, new)

    rep('SCRIPT_VERSION = "2026-09-08a-submit-joint"',
        f'SCRIPT_VERSION = "{version}"')

    # ---- 1. the ranker ---------------------------------------------------
    rep('RANKER_NAME = "edge_ranker_ep175_bidir.pkl"',
        f'RANKER_NAME = "{pkl}"      # rkprod, RANK_TRAIN={rank_train}')

    rep('''assert _pk is not None, (
    f"{RANKER_NAME} not found under /kaggle/input -- attach "
    f"faustine649/biohub-edgeprob-rank as a kernel source. No other ranker "
    f"pkl is accepted by this submission.")''',
        '''assert _pk is not None, (
    f"{RANKER_NAME} not found under /kaggle/input -- attach "
    f"faustine649/biohub-rkprod as a kernel source. No other ranker "
    f"pkl is accepted by this submission.")''')

    # Refuse to run on the wrong pkl rather than silently shipping the other
    # arm under this arm's name: the two rkprod pkls differ in ONE field and
    # are otherwise interchangeable, which is exactly when a mix-up is silent.
    rep('''CLF = _blob["model"]
assert hasattr(CLF, "predict_proba"), (''',
        f'''assert _blob.get("rank_train") == "{rank_train}", (
    f"{{RANKER_NAME}} says rank_train={{_blob.get('rank_train')!r}}, not "
    f"{rank_train!r} -- this is the other rkprod arm's pkl, not this one's")
assert _blob.get("version", "").endswith("rkprod"), (
    f"{{RANKER_NAME}} was written by {{_blob.get('version')!r}}, not rkprod")
CLF = _blob["model"]
assert hasattr(CLF, "predict_proba"), (''')

    rep('''print(f"ranker {RANKER_NAME} from {_pk} | pool {len(_pool)} datasets | "
      f"version {_blob.get('version')} | {len(_cols)} features = 18 + "
      f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)''',
        '''print(f"ranker {RANKER_NAME} from {_pk} | pool {len(_pool)} datasets | "
      f"version {_blob.get('version')} | rank_train {_blob.get('rank_train')} | "
      f"{_blob.get('n_train_rows')} rows, {_blob.get('n_train_pos')} positive | "
      f"val AUC {_blob.get('val_auc')} | {len(_cols)} features = 18 + "
      f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)''')

    # ---- 2. the operating point the retrain needs ------------------------
    rep('ILP_DIV_W = 0.55   # set from the fine sweep\'s argmax',
        'ILP_DIV_W = 0.35   # WAS 0.55 (the fine sweep\'s argmax for the BASE-trained\n'
        '                   # ranker). rkpool\'s gated cross: the retrained ranker at\n'
        '                   # 0.55 + the joint gate reads 0.9410, BELOW what ships at\n'
        '                   # 0.9424; at 0.35 it reads 0.9449, the best cell measured\n'
        '                   # anywhere. DIV_W 0.35 alone is +0.0002, so this is the\n'
        '                   # operating point the retrain needs, not a second change.')

    # ---- 3. say so in the log -------------------------------------------
    rep('''print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)''',
        f'''print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
print("  {what}\\n"
      "  + ILP_DIV_W 0.55 -> 0.35. Everything else is ref 56086066 (LB 0.944).\\n"
      "  rkpool, 128 6bba movies, official scorer: ungated REF 0.9266 -> 0.9290\\n"
      "  (+0.0023, 90% [+0.0007, +0.0040], 77/128 up); dw35 + joint gate 0.9449\\n"
      "  against 0.9424. fk44f4, 71 44b6 movies: +0.0019 [-0.0004, +0.0048],\\n"
      "  all of it divisions (adj_J flat, edge_J -0.0014 at P 0.084).", flush=True)''')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       out_name)
    open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
    print("written", out_name, len(src))
