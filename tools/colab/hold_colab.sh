#!/bin/bash
# Long-lived holder for OUR Colab session. Must be started as a
# background task on the Windows side and left running -- see the `hold`
# case in cx.sh for why. If this exits, the VM is idle-pruned about
# fifty minutes later and everything on it is lost.
export MSYS_NO_PATHCONV=1
SP=/mnt/c/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/8724d11c-e5b5-4ba3-be32-60f80bbf0f50/scratchpad

echo "hold started $(date -u '+%Y-%m-%d %H:%M:%S UTC') for session ct3"
CT_SESS=ct3 wsl -d Ubuntu-24.04 -- bash -c "CT_SESS=ct3 bash '$SP/cx.sh' hold"
echo "hold EXITED $(date -u '+%Y-%m-%d %H:%M:%S UTC') -- the session will be pruned"
