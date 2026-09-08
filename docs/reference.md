# Reference — stable facts

> Reader: a future AI session with no context. Everything here is lookup
> material that stays true until explicitly superseded. Events and
> measurements-with-dates live in `progress.md` (newest first). One fact
> lives in one place; this file is that place for anything below.
> Grep anchors: every section is a `##` with a stable name.

## Timeline

| Date | Event |
|---|---|
| 2026-09-22 | Team merger / new-entrant deadline (confirmed via API) |
| 2026-09-29 | Final submission (UTC+8: 09-30 07:59) |

## The metric

`score = adjusted_edge_jaccard + 0.1 * division_jaccard`

- `N_true` = train geff metadata field `estimated_number_of_nodes`. Test has
  none.
- Penalty `max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))`, no upper
  clamp, weak coefficient.
- Not a pure micro-average: `adj_edge_jaccard` is per-dataset, averaged with
  weights `w_i = TP + FP + FN`. This is exactly why a flat constant caps out.
- Edge FP rule: an edge counts toward the denominator if **either** endpoint
  matched a GT node of the right degree (one matched endpoint is enough).
  History of getting this wrong: progress.md `2026-08-18 — the scorer bug`.
- Edges with `t_target != t_source + 1` are discarded, not penalised; merges
  collapse to one predicted edge per matched GT edge; out-degree capped at 2.
- A missed detection costs twice: the GT edge becomes an FN and the predicted
  edge leaving the matched neighbour becomes an FP.
- Division metric: 574 lines of local-topology validation in
  `division_metrics.py`, deliberately NOT ported. Our
  `score_divisions_simple` is a lenient upper bound — trust the repo's
  number. A fork lying entirely among unannotated cells enters none of the
  FP sets and is not penalised (same leniency as node/edge terms). Do not
  exploit: unmatched forks add no TP, and fork-spam is the genre of the
  hub-node exploit patched 2026-07-17.

## Train-set survey (199 datasets)

| Metric | Value |
|---|---|
| Datasets | 199 (`44b6` 71, `6bba` 128) |
| Shape — identical everywhere | `(100, 64, 256, 256)` uint16 |
| Voxel scale | (1.625, 0.40625, 0.40625) um, uniform |
| GT nodes / edges | 133,318 / 128,883 |
| Divisions | 151 (1 per 854 edges; 56% of datasets have none) |
| N_true total | 4,725,117 — GT labels only 2.82% of it |
| True cells/frame | min 38 / median 179 / max 786 (20x spread) |
| Pooled displacement | p50 1.82 / p99 6.71 / p100 60.76 um |
| Brightness q0.999 across datasets | 615..5398 = **8.8x spread** (44b6 7.8x, 6bba 7.9x) |

Brightness source: organizer-stored `image_statistics.quantiles` in each
dataset's zarr attrs (keys 0.0/0.001/0.01/0.1/0.9/0.99/0.999/1.0); the
official `io.py` normalises with them. Per-dataset quantile normalisation is
mandatory, by design. Stored quantiles are whole-movie; a single frame can
sit ~2x below them (bleaching/drift) — compute crop-local quantiles for
training crops.

By imaging batch:

| | `44b6` (71) | `6bba` (128) |
|---|---|---|
| Share of score weight | 15% | **85%** |
| True cells/frame (median) | 327 | 97 |
| Annotated fraction (median) | 0.77% | 9.71% |
| Displacement p99 (median) | 4.90 um | 7.81 um |
| Duplicate-frame rate | 0% | 7.47% |

- `corr(annotation density, true density) = -0.20` — negative. N_true cannot
  be inferred from the GT; hence `cell_count.py`.
- `corr(true density, displacement p99) = -0.39` — denser videos move less.
- 43 of 199 datasets have GT that does not span all 100 frames. The
  node-count penalty counts the whole movie, so detection must run on every
  frame.
- The competition zarrs are **zarr format 3** (`zarr.json` metadata);
  `zarr<3` cannot open them. Read attrs as plain JSON, or install `zarr>=3`.

## Zebrahub external data

Host permission (Thibaut Goldsborough, 2026-08-13, discussion /734330):
imaging data AND `*_tracks.csv` are free to use; no overlap with the test
set. Source bucket (use this, never the `kkunizaw/biohub-zh001r` mirror —
its labels failed alignment at 16.6%):

    https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/

All five movies passed the label-to-image alignment check on 2026-08-21
(kernel `faustine649/zebrahub-align-spotcheck`; "align" = local-max intensity
in a ~2.5x3x3 um window around track points vs random points, same chunk):

| movie | shape (t,c,z,y,x) | track units | disp p50/p99 um | align track-p50 vs random | verdict |
|---|---|---|---|---|---|
| ZSNS001 | (791,1,448,2174,2423) | **um** | 1.52 / 5.13 | 118 vs 6 | PASS |
| ZSNS001_tail | (791,1,420,1217,1091) | **um** | 1.32 / 5.15 | 138 vs 5 | PASS |
| ZSNS002 | — | no tracks.csv | — | — | unusable |
| ZSNS003 | (515,1,258,1929,2038) | **voxel** | 1.32 / 6.64 | 360 vs 0 | PASS |
| ZSNS004 | (600,1,294,1926,2581) | **voxel** | 1.58 / 8.61 | 297 vs 0 | PASS |
| ZSNS005 | (600,1,291,1910,2750) | **voxel** | 1.39 / 8.16 | 810 vs 0 | PASS |

Facts every consumer of this data must handle:

1. **Units differ per movie.** ZSNS003/004/005 store voxel indices (integer
   coordinates; y max exceeds the um extent but fits the voxel extent).
   Detect, never assume: coordinate max vs physical extent. Convert per axis
   (voxel (1.24, 0.439, 0.439) um — but read it from the zarr attrs).
2. **Columns differ per movie.** `_tail`: `ParentTrackID`, `NodeID`,
   `t_hier_id`, `area`. 003/004/005 add `id`/`parent_id`. Normalise names.
3. **Brightness varies 10x across movies** (chunk p99 126..1293) — same
   regime as the competition's own 8.8x; per-crop quantile normalisation.
4. **Labels are machine-generated** (automated tracking, not human GT):
   ~10% of ZSNS001 points sit on near-background intensity. Filter or
   down-weight by local brightness when training.
   **MEASURED 2026-08-30 (`biohub-zhcomplete`): the labels are NOT
   complete.** ~2.5% of cells (defensible estimate; <= ~7% worst case)
   have no label at all -- whole-track misses, not per-frame dropouts
   (their own tracks flicker at ~0.00%). So zh PRECISION numbers for
   any detector are understated by roughly that amount; zh recall
   denominators remain the best available by 10-100x over the
   competition's 2.8% annotation, with this footnote.
5. Scale: ZSNS001 alone has 21,697,591 nodes and 635,042 divisions vs the
   competition's 133,318 and 151. Divisions are plentiful in every movie.
6. Sampled density per 104-um cube brackets the competition's 50-519
   cells/frame, but placement decides what you get — see
   `zebrahub_extract.py` (density-bin placement, SCRIPT_VERSION >= 2026-08-21d).
7. Zebrahub zarrs are zarr v2 over HTTP; `zarr<3` opens them. In zarr 2,
   `arr[t, 0][slices]` materialises the whole multi-GB frame first — always
   index in ONE call: `arr[t, 0, z0:z1, y0:y1, x0:x1]`.

## Submission mechanics

Code Competition, hidden test set.

- Notebook submissions only; 12 h limit; internet disabled; output must be
  named exactly `submission.csv`.
- The visible `test/` holds 4 datasets that are copies of train files
  (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` —
  quantile tables match digit for digit). Real data is swapped in at
  rescore. Never hardcode dataset count or names.
- Those 4 have ground truth under train, so the real submission CSV can be
  scored locally — `submit_baseline.py verify`.

Runtime budget (measured 2026-08-20 from the scored submission's own cell
timestamps; T4, `predict_video` + ILP): install 26 s, import+load 11 s,
inference 94.2 s/dataset -> **~5.2 h for the ~199-volume hidden test**.
Roughly 2.3x budget spare — enough for TTA or a wider candidate search.
Corroborated by forum reports (visible->submission runtime ratio 40-50x).

Update, the user 2026-09-02: the latest shipped pipeline (F4, ref
55889500 family) takes **about 6 h** on the hidden test, and that figure
is already generous -- it will not run over. So the spare against the
12 h limit is ~6 h. For scale: watershed masks on the heatmap cost
~80 s per 100-frame movie (`biohub-zh-masks`, 41 crops in 54 min GPU),
i.e. ~4.4 h for 199 volumes -- it would fit, with ~1.5 h to spare.

Residual risk: the notebook falls back to greedy if `pyscipopt` fails to
import; the warning only appears in the rescore log. Confirm ILP ran before
reading a score as the ILP number.

Offline dependencies: the classical path needs no installs (zarr v3 chunks
read via `blosc2`, ships with the Kaggle image; `open_image()` and
`read_geff()` have verified zarr-free fallbacks). The official pipeline
needs `torch`, `tracksdata`, `zarr`, `rustworkx` from an offline wheel
dataset — pilkwang's support pack ships 62 wheels and an install command.
Do NOT install every wheel in the pack; it clobbers numpy (see
`submission_notebook.md`).

## Running the official pipeline

```bash
git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git
pip install -e . ; pip install "tracksdata @ git+.../tracksdata@main" zarr tqdm

python scripts/predict_unet_transformer.py \
    --splits splits.json --split 0 \
    --weights .../unet_transformer/split_0/edge_predictor_best.pth \
    --det-threshold 0.99 --use-ilp \
    --ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5 \
    --evaluate
```

- The package is `tracking_cellmot`; pilkwang's repackage renames it
  `biohub_tracking`.
- `--debug-video` takes a path with no `.zarr` suffix — `predict()` uses the
  path's name for the output filename, and `x.zarr.geff` will not match GT.
- `predict()` wipes every .geff in the output dir on each call, so `--slice`
  runs do not accumulate.
- `--use-ilp` needs `pyscipopt`; `--ilp-division-weight` is the direct knob
  on fork creation (a fork is taken iff
  `edge_prob > division_weight - appearance_weight`).
- pip prints a wall of conflicts about `bigframes`, `ydata-profiling`,
  `google-colab`, `cuml`, `cudf` — none are in the pipeline.

Key defaults inside `predict_unet_transformer.py` (read 2026-08-20):
`det_threshold 0.5` (we pass 0.99), `det_tta True`, `pool_kernel_um 3.0`,
`edge_activation softmax` (normalises over sources: for each target j,
`sum_i probs[i,j] = 1`), `threshold 0.5` = the edge-candidate gate — the
only thing limiting what the linker may consider. With `use_ilp=True` the
greedy per-node capacity caps are disabled entirely.

## Where the code lives

Local folder `D:\programming\cell tracking\` (flat; do not reorganise
mid-competition — Kaggle pushes and cross-file imports depend on paths):

| File | Purpose |
|---|---|
| `kag.py` | the Kaggle CLI with its OAuth token kept fresh — use for every CLI call |
| `viz_cells.py` | inspect / viz / gif / survey, plus all shared I/O |
| `cell_count.py` | features / fit / predict — estimate N_true from images |
| `submit_baseline.py` | predict / score / sweep / verify, and the local scorer |
| `cache_candidates.py` | run GPU inference once, dump (coords, edges, probs) per dataset |
| `link_replay.py` | re-solve linking from the cache on CPU |
| `run_sweep.py` | Kaggle entry point for cached linking sweeps |
| `cache_det_sweep.py` | det-threshold variant of the cache (one inference pass per value) |
| `stranded_audit.py` | count stranded targets / real edges behind the 0.5 gate |
| `run_audit.py` | Kaggle entry point for the stranded audit |
| `survey_match.py` | match-radius / survey checks |
| `division_ceiling.py` | division-term headroom arithmetic |
| `division_features.py` | do geometry/intensity/elongation distinguish a dividing cell? |
| `division_recovery.py` | add missed divisions back into an ILP graph, post-hoc |
| `division_strip_cell.py` / `division_gif_cell.py` | filmstrip / gif of one GT division |
| `sweep_edge_threshold.py` | greedy-only offline threshold sweep (superseded by the cache) |
| `ranker_scale.py` | BASE/EXT/PURE column views of the edge features — proved scale-invariance changes nothing |
| `finetune_detect.py` | fine-tune the detection head (per-crop quantile normalisation lives here) |
| `zebrahub_extract.py` | cut competition-shaped crops out of Zebrahub (see its header) |
| `error_analysis.py` | classify every edge error: detection / dropped / confusion / spurious |
| `check_weights.py` | inspect published checkpoints for their training split |
| `read_gate.py` | read DeepCenter's tuning files |
| `make_fake_data.py` | synthetic smoke-test data (numbers meaningless) |
| `run_cells.md` | Kaggle cells, one fenced block per cell |
| `submission_notebook.md` | the cells that produced the 0.876 submission |

Every script prints `SCRIPT_VERSION` (trap 11). Kaggle workflow, CLI/MCP
split, and its own traps: `kaggle-toolchain.md`.

## Published-artefact facts (read 2026-08-18)

- **DeepCenter's split is a batch split**: train = exactly the 71 `44b6_`,
  val = exactly the 128 `6bba_` (`split_manifest.json`; the author's group
  split collapsed on a two-group dataset). Consequence: every `6bba_`
  dataset is guaranteed unseen by DeepCenter — the held-out pool is
  `6bba`-only for that reason.
- DeepCenter ships `best.pt` (epoch 2) and `checkpoint_last.pt` (epoch 500);
  the gate sweep was calibrated on `best.pt`, the 0.916 notebook loads
  `checkpoint_last.pt`. ~~No detection-level comparison exists between
  them. Do not read the 7x val_loss as "got worse" — the loss is
  positive-unlabeled BCE that punishes finding unannotated cells.~~
  **MEASURED 2026-08-28 by `biohub-dcprobe`. USE `best.pt`.
  `checkpoint_last.pt` IS DEAD:**

  | | head bias | head weight \|mean\| | Σ per-tensor \|mean\| | peaks/frame |
  |---|---|---|---|---|
  | `best.pt` (ep 2) | −0.325 | 0.305 | **15.9** | **114 – 654** |
  | `checkpoint_last.pt` (ep 500) | −0.585 | 0.739 | **39.9** | **0 – 1** |

  Its largest logit anywhere in a 64³ volume is −0.34, i.e. "no cell
  anywhere", and its weights have grown 2.5×. That is divergence, on
  6bba, which is DeepCenter's own validation split. Both checkpoints
  record the same `best_score` (−0.04500306, roughly epoch 2), so
  training never improved after epoch 2 and ran on to 500.

  ⚠ **The struck-through sentence above caused this.** It is a warning
  *not to conclude* "worse" from that loss. It was read as a positive
  argument that the epoch-500 checkpoint finds MORE unannotated cells —
  which is what a repair pool wants — and that reasoning picked the dead
  checkpoint and cost three runs and ~70 min of GPU. **A note saying
  "do not infer X from this" is not evidence for not-X.**
- The primary U-Net-transformer's split is unknown (`split_0`
  checkpoint carries no file list). See `holdout-split.md`.

## Historical: the classical DoG pipeline

Superseded as a main line (LB 0.681 vs official pipeline's 0.876). Findings
that still hold (directions only — its adj_J numbers came from the broken
pre-2026-08-18 scorer):

1. Under-detecting is catastrophic, over-detecting is cheap (J=0.84 at 2.92x
   N_true vs J=0.18 at 0.28x).
2. Unlinked nodes are pure cost — pruning raised adj_J and TP went up.
3. `node_recall` is a misleading target — watch TP.
4. `ratio` and `nms_um` are coupled; re-sweep after either changes.
5. A per-dataset count estimate is worth +0.155 over the best flat constant
   (the gap is the node-count penalty being mis-distributed).

Submitted config for the record: `sigma_um=2.0 sigma_ratio=1.6 nms_um=3.0
gate_um=10.0 per_frame=100 ratio=2.0 max_gap=1 prune_isolated=True
truncate=3.0`; 92 s/dataset single-core.

## Traps already hit

1. napari is unusable — desktop GUI, Kaggle is headless. Use matplotlib MIP.
2. Kaggle has no zarr, and submissions have no internet.
3. OME-Zarr is a multiscale pyramid — `zarr.open()` returns a group.
4. argparse dies in notebooks — use `parse_known_args()`.
5. matplotlib's default interpolation blurs microscopy — `nearest`.
6. Full-Z projection is mush — 64 planes x 1.625 um is ~15 nuclei deep.
7. matplotlib's default font has no CJK glyphs — figure text in English.
8. Cowork's cloud working directory does not persist across sessions.
9. Folder access is fixed at session start.
10. `--limit` taking the first N covers one imaging batch only — use a
    stride. Runs with different `limit` are NOT comparable.
11. Kaggle Dataset caching plus `sys.modules` caching runs stale code
    without raising. Hence `SCRIPT_VERSION` in every file.
12. Cross-file imports of private helpers break when the Dataset copy is a
    version behind.
13. Too much output kills a Kaggle cell — never print per dataset in a loop.
14. Recomputing detection inside a sweep — caching made sweeps 4.5x faster,
    byte-identical.
15. Parameters interact — re-sweep couplings after any change.
16. `!pip install` with internet disabled spins ~10 minutes before failing.
17. A `.geff` IS a zarr store. "The geff library is optional" is not "zarr
    is optional".
18. `node_recall` is a misleading target.
19. Reimplementing an official metric is a liability. Ours was wrong for
    weeks in the flattering direction. Cross-check against the official
    scorer on the same predicted graphs after every scorer change.
20. `Path("/kaggle/input").rglob("*")` appears to hang — the competition
    mount holds millions of zarr chunk files. Prune `.zarr` trees.
21. A silent cell is not a hung cell — check the GPU meter before killing.
22. In zarr 2, `arr[t, 0][slices]` materialises the whole frame before
    slicing — index in one call.
23. The competition zarrs are zarr format 3; `zarr<3` cannot open them
    (Zebrahub's are v2, which `zarr>=3` reads too).
24. PowerShell `Out-File`/`>` writes a UTF-8 BOM that Kaggle's CLI JSON
    parser rejects — write `kernel-metadata.json` without BOM.
25. A Kaggle kernel title that does not slugify to the metadata `id` makes
    the next push 409 — keep title and id consistent.
