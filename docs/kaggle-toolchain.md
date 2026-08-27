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

### 6b. A SUBMISSION needs a waiter too (2026-08-27)

This rule was applied to kernels only, for days. **A submission's hidden
-test rescore is a second long-running job with no notification**, and it
was being checked by hand -- the user caught it: *"那么两个任务为什么只
有一个等待器呢?!"*

```bash
while true; do
  line=$(py kag.py competitions submissions <comp> 2>&1 | grep -m1 '^ *<ref>')
  if echo "$line" | grep -qE "SubmissionStatus\.(COMPLETE|ERROR)"; then
    echo "$line"; break
  fi
  sleep 300
done
```

Two details that are not optional:

- **Match `SubmissionStatus.COMPLETE`, never a score-shaped pattern.**
  The submission DESCRIPTION sits on the same line, and ours routinely
  contain numbers like `0.920` and `0.921` (pre-registered predictions),
  so grepping for a score matches the description and fires instantly.
- **Poll at 300 s, not 60.** The rescore takes hours; a 60-second loop
  is 60 CLI calls an hour for nothing. Kernel waiters stay at 60 s
  because CPU sweeps finish in one to two hours.

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

### 9. `machine_shape` is not optional on a GPU kernel

`biohub-det-ablate3` v1 was given a **Tesla P100 (sm_60)**, which the
installed PyTorch does not support -- it builds for sm_70 and above. The
kernel printed

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
GPU present but conv3d unusable -> CPU
```

and silently continued on CPU. This is not bad luck: kernels that ran
fine on GPU carry

```json
"machine_shape": "NvidiaTeslaT4"
```

in their metadata (confirmed on `biohub-zh-masks`, which ran 54 min on a
T4). Omit the field and Kaggle picks, and P100 is in the pool.

**Every GPU kernel's metadata must pin `"machine_shape":
"NvidiaTeslaT4"`.** Two failure modes if you do not, and the second is
worse than the first:

- the run dies, or
- **it does not die.** The detector falls back to CPU, produces the same
  numbers 30x slower, and burns the wall clock while looking healthy.

### 10. Recover a working kernel's sources instead of reconstructing them

`det_ablate3` v1 also died on `unet3d_bright.pt not found`, because the
assembly copied the SCRIPT from `zh_masks.py` but the metadata was
written from scratch and guessed the wrong weight datasets. The correct
ones are `xiaoleilian/biohub-unet3d-weights` and
`xiaoleilian/biohub-unet3d-weights-v2models`; the guess was
`pilkwang/biohub-tracking-support-pack-50ep-v1`, which is a different
pack.

The fix takes one command and settles both this and `machine_shape`:

```bash
py kag.py kernels pull faustine649/<a-kernel-that-worked> -p /tmp/x -m
```

**When an assembly reuses a script's code, reuse its metadata too.**
Weights are resolved by `glob("/kaggle/input/**/<name>")`, so a wrong
mount is invisible until the assert fires minutes into the run.


## 11. Colab as a third worker: never run the job INSIDE the exec cell

### The one rule

**`colab exec` must only ever start a job and return. Never wait in it.**

```bash
# in the exec cell -- returns in under a second
nohup python /content/job.py > /content/job.log 2>&1 &
# then poll with separate short execs
tail -40 /content/job.log
```

The job then survives the client disconnecting, its progress is visible
while it runs, and a lost session costs the run but not the answer.

### The failure that established this, and its correct diagnosis

2026-08-25. The environment was solved end to end — version-matched
`tracksdata 0.1.0rc6.dev3`, data uploaded, the whole chain proven on one
dataset (`6bba_2312ac41: nodes 11523 edges 11205, 21s`, ILP through
SCIP, post-link patches, geff save, official scorer). The 12-dataset
acceptance run then returned `TimeoutError: Timeout waiting for output`
and nothing else, and the session was later reclaimed.

**The first write-up of this called it a platform limit and parked
Colab. That was wrong** — the user, who runs long Colab jobs in another
session, said so immediately. The simpler explanation fits every
observation:

- `--timeout 3600` is **the client waiting for output, not the job
  running**. The traceback comes from `jupyter_kernel_client`, not from
  Colab.
- The job needed longer than 60 minutes and was not hung. On Kaggle the
  same 12-dataset ILP stage occupies a large share of `post_sweep`'s
  1323 s `6bba` block on a 4-vCPU machine; a smaller Colab CPU VM taking
  1–2 hours is ordinary. The estimate that killed it — "dataset 1 took
  21 s, so 12 take 4 minutes" — was the actual error.
- The client giving up **orphaned** the run; the session then went idle
  and was reclaimed. **The client was killed first, and by us.**

One failed run does not establish a platform limit. Prefer the
explanation that requires nothing new to be true.

### Two properties of `colab exec` that make the rule necessary

- It runs the file as ONE interactive cell and **buffers all stdout
  until the cell finishes**, so a job printing a line per dataset shows
  nothing for its whole duration.
- On timeout it returns the traceback **and none of the run's output**,
  so there is no way to tell how far it got.

Both disappear once the job writes to a file and the cell returns
immediately.

### Sessions

**Multiple concurrent sessions are allowed** (confirmed by the user
2026-08-25). Use a distinct `-s` name and do not attach to a session you
did not create — `colab sessions` shows the user's own work alongside
ours, and taking over the wrong one interferes with a job you cannot
see.

### ⚠ "Session appears to be lost (404/401). Cleaning up." IS A FALSE NEGATIVE

Observed three times, and the third with direct proof. **The VM does not
die. The CLI deletes its own local name → session record on one
transient error**, and after that every command taking `-s <name>` —
including `stop` — cannot reach a machine that is still running and
still being billed.

The proof, when `ct3` went "lost":

| | |
|---|---|
| keep-alive daemon | **still running**, PID 1205, 3.1 hours |
| the assignment | **still listed** by the server |
| `sessions.json` | the name was **gone** |

Two earlier write-ups in this file blamed idle-pruning and then a dead
keep-alive daemon. **Both were wrong**, and both were built on the
assumption that the VM had died — which was never checked. `colab
sessions` prints `[?]` for exactly this state: a server-side assignment
with no local record.

**The fix is `heal()` in `tools/colab/cx.sh`**, which runs before every
command that touches a live session: if the name is missing and exactly
one orphaned assignment exists, `tools/colab/adopt.py` rebuilds the
record from the live assignment using the CLI's own client, so the proxy
token is fetched by the library and never handled by us. It refuses to
guess when more than one orphan exists, because one of them could be the
user's.

`adopt.py --list` shows which assignments have names and which do not.
Recovering a stranded VM to stop it is the same two commands.

Two smaller traps found alongside:

- `colab status -s <missing>` prints "not found" and **exits 0**, so an
  `|| new` guard never fires. Read `sessions.json` instead.
- `colab keep-alive` takes **positional** `{endpoint} {session_name}`,
  not `-s`, and terminates itself after 24 h.
- Scripts written by the Windows-side file tools get **CRLF** endings and
  bash rejects them (`$'\r': command not found`). Convert to LF before
  running anything through wsl.

### Everything below still holds

Established 2026-08-25 on the user's WSL install of `google-colab-cli`.
The user's laptop itself is NOT usable -- it runs at 70-80% memory with
about 3 GB free, and the pipeline would be killed. Colab's VM has
12.7 GB.

### The CLI

- **Always pass `--auth=adc`.** `colab --help` says the default is
  `oauth2` while the bundled `colab skill` says `adc`; they contradict,
  and the bare command launches a browser consent flow that an agent
  cannot complete. `colab --auth=adc sessions` verifies auth read-only.
- Call the binary by ABSOLUTE PATH (`/home/<user>/.local/bin/colab`).
  Exporting `$PATH` through Windows -> wsl -> bash expands the WINDOWS
  path, which contains `Program Files (x86)`, and the unquoted
  parentheses break the inner shell.
- From Git Bash, prefix `MSYS_NO_PATHCONV=1` or `/mnt/c/...` is rewritten
  into a Windows path with the Git install prefix glued on.
- Put multi-line python in a FILE and pass it with `exec -f`. A heredoc
  through the Windows shell AND wsl loses a quoting level twice; it ate
  the quotes inside an f-string and produced a SyntaxError on the VM.
- `colab upload <local> <remote>` needs a full remote FILENAME, not a
  directory; a trailing-slash destination 500s. Upload wheels under a
  short name, then rename on the VM -- pip rejects a wheel whose
  filename is not PEP 427, so `td.whl` must be restored to its real
  name before installing.
- `colab run script.py` = new + exec + stop, and its own chatter goes to
  stderr while the script's output goes to stdout. That is the shape a
  waiter wants. But it destroys the VM, so anything needing uploaded
  data should use a long-lived `colab new` session instead.
- **A session burns compute units until `colab stop`.** Stop it whenever
  the work is blocked, not just when it is finished.

### The VM, measured

| | Kaggle CPU | Colab | user's WSL |
|---|---|---|---|
| cores | 4 | **2** | 16 |
| RAM | 30 GB | 12.7 GB | 7 GB (3 GB free on the host) |
| Python | 3.12 | **3.13.15** | 3.12.3 |

Colab is SLOWER per job than Kaggle, not faster. Its value is being a
third slot outside Kaggle's two-at-a-time cap.

### ⭐ Version-matching the ILP stack on Python 3.13 -- SOLVED

The support pack's wheels are `cp312` and mostly will not install on
3.13. Three of them decide ILP behaviour and all three can be matched:

| | support pack | how to get it on 3.13 |
|---|---|---|
| `tracksdata` | `0.1.0rc6.dev3+g980c2d30a`, **py3-none-any** | NOT on PyPI (a git build). Install `tracksdata==0.1.0rc8` first so the DEPENDENCIES resolve, then `pip install --force-reinstall --no-deps` the support pack wheel over it. |
| `ilpy` | `0.6.0`, py3-none-any | PyPI has it |
| `pyscipopt` | `6.2.1`, cp312 | **PyPI publishes a cp313 wheel of the same version** |

Verified: `tracksdata 0.1.0rc6.dev3+g980c2d30a` imports, `ILPSolver` is
present, `pyscipopt 6.2.1` is present. Matching these three matters
because a different MIP backend breaks ties differently, and this
project's linking problems are full of near-ties -- a Colab number
computed on a different solver could not be compared to any recorded
anchor.

⚠ **Version matching is not the test.** The test is whether a dataset
the pipeline has already scored reproduces its recorded adj_J on the
VM. Matching versions only raises the odds. That check has NOT been run.

### What still blocks it

The pipeline's preamble needs, beyond the three packages above:

1. `git clone royerlab/kaggle-cell-tracking-competition` for
   `tracking_cellmot` -- fine, the VM has internet;
2. the support pack's `repo/src` and `repo/scripts` on `sys.path` for
   `predict_unet_transformer` -- 209 KB, uploadable;
3. `division_recovery.py` and `viz_cells.py` -- both are in THIS repo,
   uploadable;
4. `detcache_*.npz` -- 65 MB for all 199, uploadable;
5. **the GT.** `geff_path(root, "train", name)` reads the competition
   data, which is 87.6 GB and whose geffs are directories of many small
   files. Downloading them one API call at a time is impractical.

**(5) is the blocker and it is circular:** exporting the GT cleanly as
one small npz per dataset needs a Kaggle kernel, and Kaggle's two slots
are what the Colab worker exists to relieve. Queue the export the moment
a slot frees; after that Colab needs no Kaggle slot again.

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
| **submitting to the competition** | **NO** (user, 2026-08-25) — see below |

### ⚠ A submission does NOT cost GPU quota. Stop saying it does.

Corrected by the user on 2026-08-25 after I priced a submission at
"about 11.8 h of GPU" three separate times while recommending for and
against making one. **The rescore Kaggle runs on the hidden test is not
charged to the weekly accelerator budget.**

What a submission actually costs:

- **one of the day's submission slots**, and one of the two final
  selection slots at the end;
- the wall-clock wait for the rescore, which matters only for planning.

### ⚠ And 11.8 h is an ESTIMATE that has never been measured

Also corrected by the user, same day. The number comes from ref
55707206's own description, which I wrote:

> RUNTIME RISK: 210 s/dataset measured vs the 0.917 pipeline's 86 s;
> **~200 hidden volumes** puts this near 11.8 h against the 12 h cap

That is `210 × 200 = 11.67 h`, and **both factors are assumptions**:

- **210 s/dataset** was measured on the 4 visible movies, not the hidden
  ones;
- **~200 hidden volumes** is a guess, taken from the train split having
  199. Nothing establishes the hidden split is the same size.

**Nothing on record measures the hidden rescore's duration.** The
submission record carries the submit timestamp and no completion time;
the kernel's `last_run_time` is the visible run, which is a different
job. The user's standing observation is the only direct evidence there
is, and it points the other way: **no submission has ever still been
running the next morning**, so the real figure is likely under 8 h.

This matters because the estimate has already been used to block work —
`progress.md` calls `biohub-thr-sweep`'s +0.0033 threshold change "too
risky to submit on its own", and 11.8 h is the entire reason. That
verdict rests on an unverified extrapolation and should be revisited.

**The broken factor is the 210 s, not the ~200.** The user pointed this
out too: the community's own estimate of the hidden split is also about
200, so that half is probably fine. The 210 s/dataset is the problem,
and the reason is visible in this project's own logs — **per-dataset
cost varies by more than an order of magnitude**:

| 6bba dataset | nodes | solve time |
|---|---|---|
| `6bba_3abfe10a` | 53,504 | **495 s** |
| `6bba_aeee7805` | 4,809 | **8 s** |

11x the nodes, 60x the time, because linking cost runs about `n^1.5` in
candidate pairs. **A mean over 4 arbitrary movies is not an estimator of
the mean over 200** when the distribution is that skewed — and the 4
visible movies were chosen by the competition, not sampled by us.

Same error as everything else in this section: **a number measured on
one population applied to another.**

### The practical rule

**Stop pricing submissions with this number.** The observed fact is that
no submission has ever failed to finish, over many runs. Until a run
actually times out, the 12 h cap is not a binding constraint and must
not be used to block a change. If the margin ever needs checking, check
it by looking at whether a run timed out — not by re-deriving 11.8 h.

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
