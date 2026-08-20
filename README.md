# cell tracking — local working folder

Biohub Cell Tracking During Development (Kaggle). This folder holds the code;
the data lives on Kaggle (199 train datasets, ~167 GB).

Written material lives in `docs/`, which is the master copy:

- `docs/strategy.md` — where we are, what to do next, and why
- `docs/progress.md` — confirmed findings, experiment log, traps hit
- `docs/public-solutions-recon.md` — what the public leaderboard actually
  contains, read from the forum and the API
- `docs/holdout-split.md` — the public weights' training split (largely
  superseded)
- `docs/competition-brief.md` — task, data format, metric
- `docs/engineering-rules.md` — conventions every script follows

## Files

| File | Purpose |
|---|---|
| `viz_cells.py` | `inspect` / `viz` / `gif` / `survey`, plus all shared I/O |
| `submit_baseline.py` | `predict` / `score` / `sweep` / `verify` — classical baseline **and the local scorer** |
| `cell_count.py` | per-dataset density / cell-count estimation |
| `check_weights.py` | inspect the published checkpoints for their training split |
| `error_analysis.py` | classify every edge error: detection / dropped / confusion / spurious |
| `division_features.py` | do geometry, intensity or elongation distinguish a dividing cell? |
| `division_recovery.py` | add missed divisions back into an ILP graph, post-hoc |
| `division_strip_cell.py` | static filmstrip of one GT division |
| `sweep_edge_threshold.py` | sweep the linker threshold offline from one inference run |
| `make_fake_data.py` | synthetic data for smoke tests (numbers meaningless) |
| `run_cells.md` | ready-to-paste Kaggle cells, one fenced block per cell |
| `read_gate.py` | read the DeepCenter tuning files (threshold sweep, history) |

## State of play

| | LB | local, 12 held-out `6bba` |
|---|---|---|
| classical DoG pipeline, submitted 2026-08-17 | 0.681 (rank 1940/2470) | ~0.45 |
| official pipeline, greedy linking | — | 0.8847 |
| **official + ILP + app 0.0 / disapp 1.5, submitted 2026-08-19** | **0.876** (rank 1560/2560) | **0.9269** |
| live public frontier, read from the LB 2026-08-20 | 0.957 #1 / 0.935 #20 | — |

The 0.9269 is the official repo plus pilkwang's public weights and two
command-line flags — no DeepCenter, no second seed, none of the ~40 `BIOHUB_*`
post-processing knobs. It improved 12 of 12 held-out datasets over greedy, and
**it is what scored 0.876 on the public leaderboard**.

Local is optimistic by 0.051, so use local scores for **paired deltas only**.

The classical DoG pipeline is capped around 0.80–0.85 (node-recall ceiling
0.921, and an edge needs both endpoints) and is no longer the main line.

**The differentiator is the measurement loop, not the model.** Public LB is 29%
of the test data and resolves about ±0.03; our local scorer covers 199 datasets
and compares configurations *paired*, which is far more sensitive.

## The local scorer

`submit_baseline.run_score` / `run_verify` print:

```
adj_edge_jaccard   the edge term
division_jaccard   approximate; meaningless below ~50 datasets
score_full         adj_J + 0.1*div_J  <- the leaderboard-comparable number
detect_ceiling     GT edges with BOTH endpoints detected
link_rate          of those, the fraction actually linked
                   micro_J = detect_ceiling x link_rate   (exact when FP=0)
```

That decomposition is the point: a single score cannot tell you whether to work
on detection or linking. On the public pipeline it said 38% detection / 62%
linking.

## Kaggle quickstart

See `run_cells.md` — one fenced block per Kaggle cell. Always print
`SCRIPT_VERSION` for all three modules first —
Dataset caching serves stale code without raising.

`--limit` is a **stride** in `score` mode, not the first N: names are sorted, so
the first 71 are one imaging batch carrying only 15% of the score weight.

## Local smoke test (no competition data needed)

```bash
python make_fake_data.py
python viz_cells.py inspect --root ./fake_data
python submit_baseline.py score --root ./fake_data --limit 8 --per-frame 30
```

Synthetic data is 8 frames of 16x64x64 — **the numbers are meaningless**, it
only proves the plumbing works. It uses the same blosc codec and
one-chunk-per-frame layout as the real data, so it also exercises the zarr-free
readers.

## Things that are easy to forget

1. **Over-detect.** The count penalty's coefficient is 0.1 and FP is
   structurally ~0, so raw Jaccard climbs faster than the penalty takes away.
   Measured optimum for the DoG detector: 2x the estimated count. Reasoning from
   the formula alone gave the opposite, wrong answer.
2. **Prune unlinked nodes.** They cost node-count penalty and can never make a
   TP. Worth +0.015, and TP went *up* because one-to-one matching freed GT
   nodes for linked predictions.
3. **Scale distances physically**: Z=1.625 um, XY=0.40625 um, read per dataset.
   The 7 um match radius is physical.
4. **`node_recall` is a misleading target** — pruning lowered it while raising
   the score. Watch TP.
5. **Select on leave-one-group-out, never random CV** — train has two imaging
   batches and random CV rewards memorizing which one.
