#!/bin/bash
# Drive the Colab session. Called as: wsl -d Ubuntu-24.04 -- bash <this> <cmd> [args]
#
# THE RULE, learned the hard way on 2026-08-25: `colab exec` must only
# ever START a job and return. Never wait inside it.
#
#   * `--timeout N` is the CLIENT waiting for output, not the job
#     running. When it fires you get a jupyter_kernel_client traceback
#     and NONE of the run's output, and the job is orphaned.
#   * `colab exec` buffers all stdout until the cell finishes, so a
#     waiting cell shows no progress at all.
#
# So: `run` launches detached and returns in a second; `poll` tails the
# log with a separate short exec. A lost session then costs the run, not
# the answer.
#
# Calls the binary by ABSOLUTE PATH on purpose. Exporting $PATH through
# the Windows->wsl->bash chain expands the Windows PATH, which contains
# "Program Files (x86)", and the unquoted parentheses break the inner
# shell. Nothing here needs PATH.
COLAB=/home/faustine/.local/bin/colab
# OUR session. Never attach to one we did not create -- the user runs
# their own Colab work and `colab sessions` lists theirs too.
#
# CT_SESS overrides it. Needed because the CLI DELETES its own local
# name->session record the moment one request returns 404/401
# ("Session 'ct2' appears to be lost. Cleaning up."), after which the
# name is gone even though the VM is still listed by `colab sessions`
# and still running. Recovering such a session means addressing it by
# the machine id that `sessions` prints.
SESS=${CT_SESS:-ct2}
SP=/mnt/c/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/8724d11c-e5b5-4ba3-be32-60f80bbf0f50/scratchpad
LOG=/content/job.log
PY=$HOME/.local/share/pipx/venvs/google-colab-cli/bin/python

# ---- SELF-HEAL, and the reason it is necessary -------------------------
# Observed three times, and the third time with direct proof: the CLI
# prints "Session 'X' appears to be lost (404/401). Cleaning up." and
# DELETES its own local name->session record, while the VM keeps running
# and the keep-alive daemon keeps pinging it. When ct3 went "lost", the
# keep-alive was still alive (PID 1205, 3.1 hours) and the assignment was
# still listed by the server -- only the NAME was gone.
#
# It is a false negative on a transient error, not a dead VM. So before
# any operation, if the name is missing but exactly one orphaned
# assignment exists, take it back. adopt.py rebuilds the record from the
# live assignment using the CLI's own client, so the proxy token is
# fetched by the library and never handled here.
heal() {
  have=$("$PY" - "$SESS" <<'PY' 2>/dev/null
import json, os, sys
p = os.path.expanduser('~/.config/colab-cli/sessions.json')
try:
    d = json.load(open(p))
except Exception:
    d = {}
print('yes' if sys.argv[1] in d else 'no')
PY
)
  [ "$have" = "yes" ] && return 0
  echo "[cx] '$SESS' has no local record; looking for an orphan to adopt" >&2
  ORPH=$(cd /tmp && "$PY" "$SP/adopt.py" --list 2>/dev/null \
         | grep '<ORPHAN>' | awk '{print $1}')
  N=$(echo "$ORPH" | grep -c .)
  if [ "$N" != "1" ]; then
    echo "[cx] $N orphaned assignments -- refusing to guess which is ours." >&2
    echo "[cx] Run adopt.py --list and adopt by endpoint explicitly." >&2
    return 1
  fi
  (cd /tmp && "$PY" "$SP/adopt.py" "$ORPH" "$SESS") >&2
}

case "$1" in
  # ---- create a session AND hold its keep-alive open ---------------------
  # `colab new` spawns a keep-alive daemon that pings the assignment so
  # Colab does not idle-prune the VM, and records its pid. Invoked as
  #     wsl -d Ubuntu-24.04 -- bash cx.sh new -s ct2
  # the whole process tree is torn down the instant the command returns
  # and the daemon dies with it, so `hold` runs it as a LONG-LIVED
  # background task on the Windows side instead.
  #
  # ⚠ CORRECTION, and it matters because the first version of this
  # comment claimed the opposite. **Losing the keep-alive is NOT what
  # made ct/ct2/ct3 unreachable.** When ct3 went "lost" the keep-alive
  # was still running (PID 1205, 3.1 hours) and the VM was still listed
  # by the server. Only the CLI's local NAME record had been deleted.
  # The keep-alive is worth having on its own merits; it is not the fix
  # for "session appears to be lost". `heal` above is.
  hold)
    # `colab status -s <missing>` prints "not found" and EXITS 0, so an
    # `|| new` guard never fires. Read the local record instead; it is
    # the same file `keep-alive` needs the endpoint from anyway.
    ep_of() {
      python3 - "$SESS" <<'PY'
import json, os, sys
p = os.path.expanduser('~/.config/colab-cli/sessions.json')
try:
    d = json.load(open(p))
except Exception:
    d = {}
s = d.get(sys.argv[1])
print(s.get('endpoint', '') if s else '')
PY
    }
    EP=$(ep_of)
    if [ -z "$EP" ]; then
      echo "creating session $SESS ..."
      "$COLAB" --auth=adc new -s "$SESS" || exit 1
      EP=$(ep_of)
    fi
    if [ -z "$EP" ]; then
      echo "still no local record for $SESS; cannot keep alive"
      exit 1
    fi
    # `keep-alive` takes POSITIONAL {endpoint} {session_name}, not -s,
    # and terminates itself after 24h.
    echo "holding $SESS ($EP) open; this must NOT exit"
    exec "$COLAB" --auth=adc keep-alive "$EP" "$SESS"
    ;;

  # ---- short synchronous cell: setup, probes, anything under a minute
  exec)
    heal || exit 1
    shift
    exec timeout 300 "$COLAB" --auth=adc exec -s "$SESS" --timeout 240 \
        -f "$SP/$1"
    ;;

  # ---- upload a real file. `run` base64s the payload into the cell,
  # which is fine for a script and hopeless for 15 MB of caches.
  # Takes an ABSOLUTE local path so it can reach outside $SP.
  up)
    heal || exit 1
    shift
    exec timeout 600 "$COLAB" --auth=adc upload -s "$SESS" "$1" "$2"
    ;;

  # ---- upload a local file to /content, then launch it DETACHED
  run)
    heal || exit 1
    shift
    NAME=$(basename "$1")
    printf '%s\n' \
      "import base64, pathlib, subprocess" \
      "pathlib.Path('/content/$NAME').write_bytes(base64.b64decode('$(base64 -w0 "$SP/$1")'))" \
      "pathlib.Path('$LOG').write_text('')" \
      "subprocess.Popen(['bash','-c','cd /content && nohup python -u $NAME >> $LOG 2>&1 &'])" \
      "print('launched $NAME detached; poll with cx.sh poll')" \
      > "$SP/_launch.py"
    exec timeout 300 "$COLAB" --auth=adc exec -s "$SESS" --timeout 240 \
        -f "$SP/_launch.py"
    ;;

  # ---- read the log without blocking on the job
  poll)
    heal || exit 1
    N=${2:-40}
    printf '%s\n' \
      "import pathlib, subprocess" \
      "p = pathlib.Path('$LOG')" \
      "print('log', p.stat().st_size if p.exists() else 'MISSING', 'bytes')" \
      "print(subprocess.run(['bash','-c','tail -n $N $LOG'],capture_output=True,text=True).stdout)" \
      "r = subprocess.run(['bash','-c','pgrep -af \"python -u\" || true'],capture_output=True,text=True)" \
      "print('still running:', r.stdout.strip() or 'NOTHING -- the job has exited')" \
      > "$SP/_poll.py"
    exec timeout 300 "$COLAB" --auth=adc exec -s "$SESS" --timeout 240 \
        -f "$SP/_poll.py"
    ;;

  *)
    exec timeout 300 "$COLAB" --auth=adc "$@"
    ;;
esac
