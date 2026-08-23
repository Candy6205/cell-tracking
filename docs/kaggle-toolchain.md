# Kaggle toolchain — how to actually run things

*Written 2026-08-20, after working out what does and does not function.*

Two separate channels, and each is only good at one half of the job.

| | Kaggle MCP connector | official `kaggle` CLI |
|---|---|---|
| endpoint | `www.kaggle.com/mcp` | `www.kaggle.com/api/v1` |
| auth | the `KGAT_…` bearer token in the Claude connector config | `kaggle auth login` (OAuth, cached locally) |
| read competitions / leaderboard / forums / notebooks | **yes** | partly |
| read a finished run's **stdout log** | **yes** | no |
| list and fetch a run's **output files** | **yes**, with signed URLs | no |
| push code | yes, but see below | **yes** |
| **attach data sources** | **NO** | **yes** |

**The working loop is: push with the CLI, read with the connector.**

```powershell
py -m kaggle kernels push -p "D:\programming\cell tracking"
```

then poll `get_notebook_session_status` and read
`list_notebook_session_output`, which returns the whole stdout log inline plus
a signed URL per output file. `curl` fetches those URLs directly.

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

### 1b. The CLI's access token expires and it does NOT auto-refresh (2026-08-22)

Some hours after `kaggle auth login`, every CLI call starts failing with
"Authentication required" — or, on `kernels status`, a MISLEADING
"Permission 'kernels.get' was denied / wrong slug / private notebook". The
refresh token in `~/.kaggle/credentials.json` is still good; the CLI just
never uses it on the API path. Fix: mint a fresh token and inject it per
invocation:

```powershell
$env:KAGGLE_API_TOKEN = (py -m kaggle auth print-access-token 2>$null | Out-String).Trim()
py -m kaggle <whatever>
```

```bash
export KAGGLE_API_TOKEN=$(py -m kaggle auth print-access-token 2>/dev/null)
```

Shell state does not persist between tool calls — prepend this to EVERY
CLI invocation, including background waiter loops.

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
  s=$(py -m kaggle kernels status faustine649/<slug> 2>&1)
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

### 7. Only TWO batch GPU sessions at a time (2026-08-23)

A third `kernels push` on a GPU kernel is refused outright:

```
Kernel push error: Maximum batch GPU session count of 2 reached.
```

The push does NOT queue -- it fails, and if the metadata was already
edited you can end up having pushed an earlier version of the script by
accident. Check what is running before pushing:

```bash
py -m kaggle kernels status faustine649/<slug>
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

GPU quota is 30 h/week and does not accumulate, but across the remaining weeks
that is well over 100 h. **A 12-dataset validation pass costs about 11 minutes.
Quota is not the scarce resource; wall-clock attention is.**

---

*Last updated: 2026-08-23*
