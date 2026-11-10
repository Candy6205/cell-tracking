"""Re-adopt an orphaned Colab assignment by endpoint.

The CLI deletes its local name->session record whenever one request
returns 404/401 ("Cleaning up"), but the VM keeps running and keeps
being billed. After that, every command that takes `-s <name>` --
including `stop` -- cannot reach it. Two VMs have been stranded that way.

This rebuilds the local record from the LIVE server assignment, using
the CLI's own client so the proxy token is fetched by the library and
never passes through here.

Usage: python3 adopt.py <endpoint> <name>
       python3 adopt.py --list
"""
import sys

from colab_cli.common import state
from colab_cli.state import SessionState

# The CLI's --auth=adc global option sets this. Without it the library
# falls back to the oauth2 InstalledAppFlow, which prompts on stdin for
# an authorization code -- a browser consent flow for the user's own
# Google account, which is theirs to complete and not something to drive
# from here. It already failed that way once; setting it explicitly.
state.auth_provider = "adc"


def live():
    sessions, assignments = state.sync_sessions()
    known = {s.endpoint: s.name for s in sessions.values()}
    return assignments, known


if len(sys.argv) == 2 and sys.argv[1] == "--list":
    assignments, known = live()
    for a in assignments:
        print(f"  {a.endpoint}  name={known.get(a.endpoint, '<ORPHAN>')}")
    sys.exit(0)

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)

endpoint, name = sys.argv[1], sys.argv[2]
assignments, known = live()
match = [a for a in assignments if a.endpoint == endpoint]
if not match:
    print(f"no live assignment with endpoint {endpoint}")
    sys.exit(1)
a = match[0]
if endpoint in known:
    print(f"{endpoint} is already tracked as '{known[endpoint]}'")
    sys.exit(0)

rpi = getattr(a, "runtime_proxy_info", None)
s = SessionState(
    name=name,
    token=(getattr(rpi, "token", "") if rpi
           else getattr(a, "runtime_proxy_token", "")),
    url=(getattr(rpi, "url", "") if rpi else ""),
    endpoint=endpoint,
    variant=a.variant.name,
    accelerator=a.accelerator.value,
)
store = getattr(state, "sessions", None) or getattr(state, "store", None)
if store is None or not hasattr(store, "add"):
    print("cannot find the session store on `state`; attributes:",
          [x for x in dir(state) if not x.startswith("_")])
    sys.exit(1)
store.add(s)
print(f"adopted {endpoint} as '{name}' "
      f"(token length {len(s.token)}, not printed)")
