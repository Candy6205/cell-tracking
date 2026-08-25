#!/bin/bash
# Poll the detached Colab job until it stops, then print the tail.
#
# MSYS_NO_PATHCONV=1: from Git Bash, /mnt/c/... is rewritten into a
# Windows path with the Git install prefix glued on
# (D:/Miao-Yunzai/Git/mnt/c/...), which is why the first attempt at
# driving wsl from here failed with "No such file or directory".
export MSYS_NO_PATHCONV=1
SP=/mnt/c/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/8724d11c-e5b5-4ba3-be32-60f80bbf0f50/scratchpad

echo "colab waiter started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on session ct2"
for i in $(seq 1 60); do
  sleep 180
  OUT=$(wsl -d Ubuntu-24.04 -- bash "$SP/cx.sh" poll 60 2>&1)
  if echo "$OUT" | grep -q 'NOTHING -- the job has exited'; then
    echo "=== COLAB JOB EXITED after ~$((i * 3)) min ==="
    echo "$OUT"
    exit 0
  fi
  if echo "$OUT" | grep -q "Session .* appears to be lost"; then
    echo "=== SESSION LOST after ~$((i * 3)) min ==="
    echo "$OUT"
    exit 0
  fi
done
echo "=== colab waiter gave up after 180 min, job still running ==="
