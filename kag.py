"""kag.py -- the Kaggle CLI with its OAuth token kept fresh.

Use this instead of `py -m kaggle`:

    py kag.py kernels status faustine649/biohub-div-train
    py kag.py kernels push -p "D:/programming/cell tracking"

    py kag.py --token-status     how long the stored token is good for (no network)
    py kag.py --refresh          force a refresh now

WHY THIS EXISTS -- two bugs, one in the SDK and one in our own workaround.

1. kagglesdk 0.1.37 refreshes the access token 30 minutes too LATE.
   `KaggleCredentials.access_token_has_expired` reads

       expiration < now - timedelta(minutes=30)

   where a safety margin should be ADDED, not subtracted. So the CLI keeps
   presenting a token it knows is dead for the first half hour past expiry,
   and every call in that window fails with "Authentication required" -- or,
   on `kernels status`, a misleading "Permission 'kernels.get' was denied".
   After 30 minutes it does recover on its own.

2. The workaround we adopted on 2026-08-22 --

       $env:KAGGLE_API_TOKEN = (py -m kaggle auth print-access-token).Trim()

   -- calls `generate_access_token` unconditionally and never SAVES the result,
   so it mints a brand new 12-hour token on every single invocation. Prepended
   to a 60-second waiter loop, as the toolchain doc told us to, that is 60
   mints an hour. On 2026-08-24 the mint endpoint started answering
   `429 Too Many Requests`.

This wrapper refreshes through `refresh_access_token()`, which mints AND writes
the result back to credentials.json, and only when less than REFRESH_MARGIN of
the token's 12-hour life remains. Steady state is one mint per 12 hours no
matter how many commands or waiter iterations run.
"""

SCRIPT_VERSION = "kag-1"

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

CREDENTIALS = os.path.expanduser("~/.kaggle/credentials.json")

# Refresh once the token has under this much life left. The SDK issues
# 12-hour tokens, so this fires at most once per 12 hours of use.
REFRESH_MARGIN = timedelta(minutes=30)


def _read_credentials():
    """Return (access_token_expiry_utc, has_refresh_token). Never returns secrets."""
    try:
        with open(CREDENTIALS, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None, False
    except (OSError, json.JSONDecodeError) as exc:
        print("WARNING: %s is unreadable (%s) -- treating the token as absent, "
              "which forces a mint on the next call" % (CREDENTIALS, exc),
              file=sys.stderr)
        return None, False

    has_refresh = bool(data.get("refresh_token"))
    raw = data.get("access_token_expiration")
    if not data.get("access_token") or not raw:
        return None, has_refresh
    try:
        expiry = datetime.fromisoformat(raw)
    except ValueError:
        print("WARNING: access_token_expiration is not an ISO timestamp (%r) -- "
              "treating the token as expired, which forces a mint" % (raw,),
              file=sys.stderr)
        return None, has_refresh
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry, has_refresh


def token_life():
    """Seconds of validity left on the stored token; None if there is no usable one."""
    expiry, _ = _read_credentials()
    if expiry is None:
        return None
    return expiry - datetime.now(timezone.utc)


def _minutes(delta):
    return int(delta.total_seconds() // 60)


def refresh(force=False):
    """Mint a new access token and SAVE it. Returns True if the token is now usable."""
    left = token_life()
    if not force and left is not None and left > REFRESH_MARGIN:
        return True

    expiry, has_refresh = _read_credentials()
    if not has_refresh:
        print("ERROR: no refresh token in %s. Run:  py -m kaggle auth login"
              % CREDENTIALS, file=sys.stderr)
        return False

    try:
        from kagglesdk import KaggleClient, KaggleCredentials
    except ImportError as exc:
        print("ERROR: cannot import kagglesdk (%s). Run:  py -m pip install -U kaggle"
              % exc, file=sys.stderr)
        return False

    why = "forced" if force else (
        "no usable token stored" if left is None
        else "%d min of life left, under the %d min margin"
             % (_minutes(left), _minutes(REFRESH_MARGIN)))
    print("kag: refreshing the Kaggle access token (%s)" % why, file=sys.stderr)

    try:
        with KaggleClient() as client:
            creds = KaggleCredentials.load(client=client)
            if creds is None:
                print("ERROR: kagglesdk could not load %s. Run:  py -m kaggle auth login"
                      % CREDENTIALS, file=sys.stderr)
                return False
            creds.refresh_access_token()   # mints AND writes credentials.json
    except Exception as exc:              # noqa: BLE001 -- the message matters more than the type
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            msg = ("the mint endpoint is rate-limited (429). Do NOT retry in a loop. "
                   "The stored token, if any, still works until it expires")
        elif status == 401:
            msg = ("the refresh token was rejected (401) -- it has been revoked or has "
                   "expired. Run:  py -m kaggle auth login")
        else:
            msg = str(exc)
        print("ERROR: token refresh failed: %s" % msg, file=sys.stderr)
        left = token_life()
        if left is not None and left > timedelta(0):
            print("kag: falling back to the stored token, valid for another %d min -- "
                  "commands will work until then and fail after" % _minutes(left),
                  file=sys.stderr)
            return True
        return False

    left = token_life()
    if left is None:
        print("WARNING: refresh reported success but no expiry was written to %s; "
              "the next call may still fail" % CREDENTIALS, file=sys.stderr)
        return True
    print("kag: token refreshed, good for %d min" % _minutes(left), file=sys.stderr)
    return True


def status():
    expiry, has_refresh = _read_credentials()
    print("credentials: %s" % CREDENTIALS)
    print("refresh token present: %s" % ("yes" if has_refresh else "NO -- run 'py -m kaggle auth login'"))
    if expiry is None:
        print("access token: none stored; the next command will mint one")
        return 0 if has_refresh else 1
    left = expiry - datetime.now(timezone.utc)
    print("access token expires: %s UTC" % expiry.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    if left <= timedelta(0):
        print("access token: EXPIRED %d min ago; the next command will refresh it" % _minutes(-left))
    else:
        print("access token: valid for another %d min%s" % (
            _minutes(left),
            "" if left > REFRESH_MARGIN else " -- inside the refresh margin, the next command will renew it"))
    return 0


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 0
    if argv[0] == "--token-status":
        return status()
    if argv[0] == "--refresh":
        return 0 if refresh(force=True) else 1

    if not refresh():
        return 1
    return subprocess.call([sys.executable, "-m", "kaggle"] + argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
