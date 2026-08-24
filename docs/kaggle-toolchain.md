# Kaggle toolchain — how to actually run things

*Written 2026-08-20, after working out what does and does not function.*

Two separate channels, and each is only good at one half of the job.

| | Kaggle MCP connector | official `kaggle` CLI |
|---|---|---|
| endpoint | `www.kaggle.com/mcp` | `www.kaggle.com/api/v1` |
| auth | the `KGAT_…` bearer token in the Claude connector config | `kaggle auth login` once, then `kag.py` keeps it fresh (trap 1b) |
| read competitions / leaderboard / forums / notebooks | **yes** | partly |
| read a finished run's **stdout log** | **yes** | no |
| list and fetch a run's **output files** | **yes**, with signed URLs | no |
| push code | yes, but see below | **yes** |
| **attach data sources** | **NO** | **yes** |

**The working loop is: push with the CLI, read with the connector.**

```powershell
py kag.py kernels push -p "D:\programming\cell tracking"
```

then poll `get_notebook_session_status` and read
`list_notebook_session_output`, which returns the whole stdout log inline plus
a signed URL per output file. `curl` fetches those URLs directly.

`kag.py` is the CLI with its OAuth token kept fresh — same arguments, same
output as `py -m kaggle`. Use it for **every** CLI call; trap 1b says why.

---

## The traps, each one cost a run

### 1. `save_notebook` over MCP wipes data sources

Verified with four field combinations on 2026-08-20 —
`datasetDataSources` alone, `datasetDataSourcesSetter` alone, both together,
and neither. In every case `/kaggle/input` came up **empty**.

Worse, pushing to an **existing** notebook that already had sources attached
**removes them**, and converts `kernel_type` from `notebook` to `script`. That
is how `cell-tracking-during-development` lost its inputs; v6 survived only in
Kaggle's version history.

**Never push a data-dependent kernel through the connector.** Use the CLI.

### 1b. Call the CLI through `kag.py`, never `py -m kaggle` directly

```powershell
py kag.py kernels push -p "D:\programming\cell tracking"
py kag.py kernels status faustine649/<slug>
py kag.py --token-status     # how long the token has left, no network call
py kag.py --refresh          # force a mint now
```

`kag.py` refreshes the OAuth access token when, and only when, it is close to
expiring, then hands the command straight to `py -m kaggle`. Same arguments,
same output. It exists because of two bugs stacked on each other.

**SUPERSEDED 2026-08-24: the earlier text here said the CLI "does NOT
auto-refresh" and told you to prepend `KAGGLE_API_TOKEN=$(py -m kaggle auth
print-access-token)` to every invocation. Both halves were wrong, and the
recipe is what broke the account's minting.**

What is actually true, read out of kagglesdk 0.1.37:

- **The CLI does auto-refresh — 30 minutes too late.**
  `KaggleCredentials.access_token_has_expired` computes
  `expiration < now - timedelta(minutes=30)`, subtracting a safety margin that
  should be added. So the token is presented as valid for the first half hour
  *past* its expiry. Every call in that window fails with "Authentication
  required", or — on `kernels status` — the MISLEADING "Permission
  'kernels.get' was denied / wrong slug / private notebook". Thirty minutes
  later it recovers by itself. The old entry only ever saw the dead window.
- **`auth print-access-token` mints a brand new 12-hour token every time and
  never saves it.** It calls `generate_access_token` unconditionally; only
  `refresh_access_token` writes the result back to `credentials.json`. So the
  old recipe, prepended to a 60-second waiter loop as instructed, minted 60
  tokens an hour. On 2026-08-24 the endpoint answered
  `429 Too Many Requests for .../GenerateAccessToken` and the CLI could not
  mint at all for a while. It cleared within the hour, but do not go back.

`kag.py` uses `refresh_access_token()` — which mints **and** saves — under a
30-minute margin. Steady state is one mint per 12 hours regardless of how many
commands or waiter iterations run. Because it writes `credentials.json`, a
plain `py -m kaggle` also works for the rest of that window; `kag.py` is what
keeps the window from ever closing.

If it reports `429`, stop: do not retry in a loop. The stored token keeps
working until its own expiry, and `--token-status` says how long that is.

### 2. The connector's token does not work for the REST API

The `KGAT_…` token authenticates `www.kaggle.com/mcp` and returns **401
Unauthenticated** on `www.kaggle.com/api/v1`. It is MCP-scoped. The CLI needs
its own credential from `kaggle auth login`; neither `~/.kaggle/access_token`
nor `KAGGLE_API_TOKEN` accepts the MCP token.

### 3. Without `machine_shape` you may get a P100, and it fails outright

The Kaggle image's torch is built for **sm_70 and up**. A Tesla P100 is
**sm_60**, so every kernel launch dies with:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

`kernel-metadata.json` must carry:

```json
"machine_shape": "NvidiaTeslaT4"
```

**This is a submission risk, not just a dev annoyance.** The 0.876 submission
happened to run on a T4. Check the machine before trusting any timed run, and
check it before the final submission.

`cache_candidates.py` now asserts `torch.cuda.get_device_capability(0)[0] >= 7`
and exits immediately rather than failing per dataset.

### 4. `kaggle datasets version -r zip` flattens one directory level

The code dataset used to serve its files under `cell-tracking/cell tracking/`.
Re-uploading with `-r zip` extracted them to the dataset **root**, so

```python
sys.path.insert(0, "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking")
```

no longer resolves. Any new code should probe both layouts — see
`_first_existing` in `run_sweep.py`. Without `-r`, the CLI silently skips
subdirectories entirely and the push 400s.

### 5. A kernel-output mount path is not guessable

`kernel_sources` attaches another kernel's output, but not at
`/kaggle/input/<slug>/`. Search for it instead of hardcoding — and search
**with pruning**: rule 9 applies, the competition mount holds millions of zarr
chunk files and a blind `rglob` appears to hang. `_find_npz_dir` in
`run_sweep.py` is the reference implementation.

---

### 6. Waiting on a run: do not poll in a loop of tool calls

A CPU sweep runs for one to two hours. Checking its status by calling the
connector over and over burns the session for nothing and tells the user
nothing they did not already know.

Kaggle only releases a run's log after it finishes, so there is no partial
progress to read either. Start one background waiter and let it notify:

```bash
while true; do
  s=$(py kag.py kernels status faustine649/<slug> 2>&1)
  if echo "$s" | grep -qiE "complete|error|cancel"; then echo "$s"; break; fi
  sleep 60
done
```

run in the background. The grep must cover **error and cancel as well as
complete** -- a waiter that only matches success stays silent through a
crash, and silence is indistinguishable from still-running.

`Start-Sleep` in the foreground is blocked by the harness, deliberately.

### 7. Every completed kernel run emails the account owner

Batch experiments into one kernel where they share a mount and a wheel
install, rather than pushing three that each spend 30 s installing the same
54 wheels and each send mail.

### 7. Only TWO batch sessions at a time -- GPU OR CPU (2026-08-23)

A third `kernels push` on a GPU kernel is refused outright:

```
Kernel push error: Maximum batch GPU session count of 2 reached.
```

`CORRECTED 2026-08-24`: the cap is on batch sessions, not on GPU ones.
A CPU kernel (`enable_gpu: false`) occupies a slot just the same -- a
push was refused with one GPU and one CPU kernel running.

The push does NOT queue -- it fails, and if the metadata was already
edited you can end up having pushed an earlier version of the script by
accident. Check what is running before pushing:

```bash
py kag.py kernels status faustine649/<slug>
```

### 8. Backslashes are eaten one layer before Python sees them

Patch scripts fed through a heredoc lose an escape level, so
`s.index('print("\nFOO")')` silently fails to match and the patch
either asserts or, worse, half-applies. **Locate by line content, never
by a search string containing an escape sequence:**

```python
b = next(i for i, x in enumerate(lines) if "FOO" in x)
```

This cost two failed patches and one accidental push of an unpatched
script.

## Measured costs

| | value |
|---|---|
| inference, one dataset, T4, `predict_video` + candidate dump | **54–68 s** |
| full 12-dataset candidate cache | **11.3 min**, 2.8 MB total |
| cache size per dataset | 0.1–1.0 MB |
| the whole 199-dataset train set, extrapolated | ~3.3 h GPU, ~30 MB |

`SUPERSEDED 2026-08-24`: this section used to end "Quota is not the scarce
resource; wall-clock attention is." **That is now false and acting on it is
expensive.** See below.

---

## 9. Quota mechanics, measured 2026-08-24

On 2026-08-24 the weekly GPU quota stood at **102,089 s of 108,000 s (28.4 of
30 h) with five days to refresh** — about 98 minutes left. Read it before
planning a session:

```bash
# MCP connector: get_accelerator_quota
```

### What the quota charges for

Measured by reading `time_used` around known events:

| | charged? |
|---|---|
| sitting in the QUEUE | **no** — `biohub-hoct-gate` queued 67 min of its 88 and only ~10 min of quota moved |
| a session that STARTS, from container-up | **yes, all of it** |
| **mounting data sources** | **yes** — and this is the trap |
| a CPU kernel (`enable_gpu: false`) | **no** — separate pool |

**A kernel that dies instantly still pays the full mount.** `biohub-det-ablate2`
v1 hit an `assert` **7.7 seconds** into the script and burned **~4.7 minutes**
of quota. Nearly all of it was mounting the 87.6 GB competition data and the
7.5 GB Zebrahub crops — which that script never read. It was attached only to
satisfy an `assert ROOT is not None` inherited from `cache_train.py`'s preamble.

**So: never attach a data source the script does not read.** The competition
data alone is 87.6 GB across 21,494 files.

### Queue time is not free, it is just not quota

Near the cap Kaggle deprioritises: the same day, GPU kernels queued **18 to 67
minutes** while CPU kernels started in **under 2 minutes**. That costs the
session's wall clock, not the weekly budget. Both are real; do not confuse them.

Note the batch-session cap is separate again (trap 7): two at a time, and a CPU
kernel occupies a slot exactly like a GPU one.

### Most of this project's GPU spend never needed a GPU

Of ~234 GPU-minutes spent on 2026-08-24, roughly **140 went to kernels whose
entire input was under 100 MB of cached detections** — div-gate, div-forks,
div-kinematics, div-train, ranker-scale. None ran a detector. `pyscipopt` is
CPU, sklearn's GBM is CPU.

`biohub-cpu-probe` established where the real boundary is. On a CPU machine the
environment builds, caches load, the ranker predicts, **the capacity-2 ILP
solves in 7 seconds**, and forks score. Only the official metric fails, on one
line:

```
tracking_cellmot/io.py:199, in _process_on_gpu
    torch.from_numpy(image).pin_memory().to(torch_device, non_blocking=True)
RuntimeError: Cannot access accelerator device when none is available.
```

The old note "official metric needs a GPU machine even to score" is true as
written and much narrower than it sounds: **the metric never touches the
image.** `score_dir` uses `dset.tracks` and `dset.scale` and nothing else; the
GPU was demanded only to load and normalise an array that is then ignored.
`open_dataset` takes `load_image: bool = True` — pass **`load_image=False`** and
the transfer is skipped.

### CONFIRMED 2026-08-24: the ILP and the scorer run on CPU, bit for bit

`biohub-cpu-probe2` reproduced div-gate's unpruned row EXACTLY on a CPU
machine:

| | GPU (biohub-div-gate) | CPU (biohub-cpu-probe2) |
|---|---|---|
| score | 0.9363 | **0.9363** |
| adj_J | 0.9220 | **0.9220** |
| div_J | 0.1429 | **0.1429** |
| div TP/FP/FN | 5/23/7 | **5/23/7** |
| scorable forks | 553 | **553** |
| p_div median / p90 | 0.1812 / 0.7484 | **0.1812 / 0.7484** |

Every figure identical. `load_image=False` is equivalent, not an
approximation. The 12-dataset global solve costs 623 s on CPU against 419 s on
GPU — about 1.5x slower, for **zero quota and no queue**.

**So: any kernel that starts from cached detections belongs on a CPU machine.**
That is the ranker, the ILP, every sweep, and all scoring. Reserve GPU strictly
for work that touches raw pixels — detector inference, mask building, training.

The pattern this came from is worth keeping: "official metric needs a GPU
machine even to score" sat in a handoff bullet list for days with no entry
behind it, and every kernel since inherited `enable_gpu: true` from
`ranker_fine` without anyone testing it. **An unverified constraint copied
forward is as expensive as a bug.** The same day, the edge ranker's 18 features
turned out to be a wholesale transplant nobody here had chosen either.

---

*Last updated: 2026-08-24 (quota mechanics, section 9)*
