# Engineering rules

> Applies to every script in this project, new and existing.
> Each rule exists because something broke, not because it sounded nice.

---

## 1. Degradation is never silent

**Every fallback path must emit a warning.** That includes:

- an import failing and an alternative implementation taking over
  (e.g. `zarr` missing -> `RawZarrImage`)
- a metadata field being absent and a default substituted
  (e.g. no voxel scale -> `DEFAULT_SCALE`)
- a lookup missing and a global default used
  (e.g. dataset not in `--target-csv` -> flat per-frame default)
- a model being unusable and a statistic returned instead
  (e.g. missing features -> predict the global median)
- any `try/except` that swallows an exception and carries on

**Why:** a silent fallback produces a *plausible wrong number*. A crash is
better — it tells you immediately where the problem is. A run that quietly used
the default voxel scale hands you a full set of physical-distance statistics
that are all wrong and all look normal. That class of bug costs an order of
magnitude more time to find than a crash.

**How:** use `viz_cells.warn_once(key, msg)`, and call `warning_summary()` at
the end of a run.

```python
warn_once("scale_default",
          f"no voxel scale in {name} -> falling back to {DEFAULT_SCALE}; "
          f"every physical distance below assumes that value")
```

- prints the first occurrence, counts the rest — compatible with rule 2
- the end-of-run table shows how many times each degradation fired
- the message must state **what value was substituted** and **what it affects**;
  "using default" alone is not enough

When importing these helpers across files, wrap it in `try/except ImportError`
and **warn about the import failure itself** — see the top of
`submit_baseline.py`.

## 2. Do not flood the output

A Kaggle cell can die from too much output. So:

- never print one line per dataset in a loop; use `_progress(i, total, every)`,
  default one line per 25 datasets
- `every=0` silences it completely
- **errors and degradations are exempt** — they must be visible — but they go
  through `warn_once` so repeats do not spam
- truncate large JSON dumps to ~400 characters

## 3. Every file carries a `SCRIPT_VERSION`

Kaggle Dataset caching plus `sys.modules` caching will run stale code **without
raising anything** — it just gives you an old correct answer. Put a
`SCRIPT_VERSION` at the top of each file, bump it on every edit, and print it
right after import:

```python
import importlib, viz_cells, cell_count, submit_baseline
for m in (viz_cells, cell_count, submit_baseline):
    importlib.reload(m)
    print(m.__name__, m.SCRIPT_VERSION)
```

All must match. A mismatch means the Dataset was not updated — not a bug in the
code.

## 4. Read physical quantities per dataset; never hardcode

Voxel size lives in each zarr's own metadata
(`multiscales[0].datasets[0].coordinateTransformations[0].scale[-3:]`).
`DEFAULT_SCALE` is a fallback only, and triggering it must warn (rule 1).
Call `read_scale(zarr_path)` before any distance computation.

## 5. Subsample with a stride, never take the first N

Dataset names are sorted, and the first 71 are all `44b6_`. `limit=12` taking
the first 12 means looking at one imaging batch only — and that batch carries
just 15% of the score weight. Any `--limit` used for scoring or validation must
use `names[::step]`.

## 6. Validate the submission path with the real file

Do not validate only the in-memory arrays. The `verify` mode reads the **actual
CSV that will be submitted**, exercising id remapping, coordinate rounding and
row ordering. The visible test split happens to duplicate train datasets, so
their ground truth is available — do not waste that.

## 7. Everything written to a file is in English

Code, comments, printed output, and documentation alike.

## 8. Scripts use `parse_known_args`

Jupyter injects `-f kernel.json` into `sys.argv`; `parse_args()` responds with
`SystemExit(2)`.

## 9. Never walk `/kaggle/input` without pruning

`Path("/kaggle/input").rglob("*")` looks harmless and is not. When the
competition dataset is attached, that mount holds ~200 `.zarr` stores made of
millions of one-chunk-per-frame files. `sorted(rglob(...))` forces the whole
traversal before printing anything: the cell shows the section header, then
appears to hang for minutes, then dumps everything at once — which also
violates rule 2.

Walk with `os.scandir` and prune any directory whose name ends in `.zarr`,
`.geff`, `.n5` or `.ome` before descending, cap the number of lines printed,
and `flush=True` every print so progress is visible while it runs.
`check_weights.py` is the reference implementation.

## 10. A notebook prints whatever a function RETURNS

Rule 2 covers what a script *prints*. It does not cover what a function
*returns* — and Jupyter auto-displays the last expression of a cell, so
`ea.run(...)` returning a list of 10313 records dumps all 10313 records with no
`print` anywhere in the code. This has now happened twice.

For any function meant to be called from a notebook:

- **return a small summary** — a dict of counts, a scalar. Never the raw
  per-item records.
- **write the detailed data to a file** and print the path
- print `-- send this file, do not print it` beside the path, so attaching the
  file is the obvious next move rather than pasting the output

`error_analysis.run()` is the reference implementation: it tees its report into
`error_summary.txt`, writes `error_edges.csv`, and returns a five-key dict.

## 11. Prefer the official implementation over a reimplementation

Ours was wrong for weeks, in the flattering direction, and every tuning decision
rode on it. Where the official code can be run, run it. Use a reimplementation
only for fast paired sweeps, and cross-check the two on the same predictions
after every change to either.

---

*Last updated: 2026-08-18*
