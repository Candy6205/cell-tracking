"""klog -- fetch ONLY a kernel's log, in seconds instead of minutes.

`py kag.py kernels output <kernel> -p <dir>` downloads every output file, and
the bench kernels write their geff intermediates to /kaggle/working: reading
`eprank3_ref`'s log meant pulling 25 MB of zarr chunk directories first, and
`zhpara`'s meant tens of thousands of tiny files. Several times tonight the
log arrived ten minutes after the question.

`KaggleApi.kernels_output` takes a `file_pattern` regex, and the log write is
UNCONDITIONAL -- it happens whether or not any file matched:

    log = response.log
    if log and page_token is None:
        outfile = os.path.join(target_dir, kernel_slug + ".log")

So a pattern that matches nothing gives the log alone.

    py tools/klog.py biohub-eprank3            # -> the tail, printed
    py tools/klog.py biohub-eprank3 --full     # the whole thing
    py tools/klog.py biohub-eprank3 -n 120     # last 120 lines
    py tools/klog.py biohub-eprank3 --grep AUC # matching lines only

The owner defaults to faustine649. Token refresh goes through kag.py, so this
inherits the once-per-12-hours behaviour and cannot trigger the 429 that the
old refresh loop did.

⚠ A SIGKILLed session returns an EMPTY log -- that is a finding, not a
failure of this tool, and it is reported as such. See working-agreement.md,
"A killed kernel loses stdout".
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import kag  # noqa: E402


def fetch(slug, owner="faustine649"):
    """Return the kernel's log as text ('' if the session left none)."""
    kag.refresh()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as td:
        # A pattern no output file can match: the log is written anyway.
        api.kernels_output(f"{owner}/{slug}", td,
                           file_pattern=r"^\0never-matches\0$", quiet=True)
        p = pathlib.Path(td) / f"{slug}.log"
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return ""
    try:                                  # Kaggle returns a JSON stream list
        return "".join(e.get("data", "") for e in json.loads(raw))
    except json.JSONDecodeError:
        return raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slug")
    ap.add_argument("--owner", default="faustine649")
    ap.add_argument("-n", "--lines", type=int, default=60)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--grep", default=None)
    ap.add_argument("--save", default=None, help="also write the text here")
    a = ap.parse_args(argv)

    slug = a.slug.split("/")[-1]
    owner = a.slug.split("/")[0] if "/" in a.slug else a.owner
    text = fetch(slug, owner)
    if text is None:
        print(f"{owner}/{slug}: no log file returned at all", file=sys.stderr)
        return 2
    if text == "":
        print(f"{owner}/{slug}: the log is EMPTY (0 bytes). On Kaggle that "
              f"means the session was SIGKILLed -- stdout is gone, not "
              f"truncated. Look at the OUTPUT FILES instead: a checkpoint "
              f"CSV, a fsynced trace, or which per-unit artefacts were left "
              f"behind. See working-agreement.md.", file=sys.stderr)
        return 1
    if a.save:
        pathlib.Path(a.save).write_text(text, encoding="utf-8")
        print(f"[{len(text)} chars -> {a.save}]", file=sys.stderr)
    lines = text.split("\n")
    if a.grep:
        lines = [l for l in lines if a.grep in l]
        print(f"[{len(lines)} of {text.count(chr(10)) + 1} lines match "
              f"{a.grep!r}]", file=sys.stderr)
    elif not a.full:
        print(f"[{len(text)} chars, last {min(a.lines, len(lines))} of "
              f"{len(lines)} lines]", file=sys.stderr)
        lines = lines[-a.lines:]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
