"""checkgen -- the one-second gate every generated kernel should pass.

WHY. `zhpara` v2 died on its first crop with `NameError: gt_arr`, twenty
minutes into a CPU run, because a `del` I added landed ABOVE the print that
reads the name. `py -c "import ast; ast.parse(...)"` -- the check this
project has always run before pushing -- cannot see that: the syntax is
fine. `pyflakes` sees it in one second, and it tracks `del`:

    for i in range(3):
        x = i * 2
        del x
        print(x)          -> undefined name 'x'

That is exactly the shape of the bug.

A WHITELIST, NOT A BLACKLIST. The first version of this file blacklisted the
noise and treated everything else as fatal. It immediately failed four
healthy kernels on "f-string is missing placeholders" -- an `f` prefix on a
string with no braces, which is cosmetic. Naming the fatal classes
explicitly is the only form that cannot cry wolf, and a gate that cries wolf
gets ignored, which is worse than not having one.

  FATAL    undefined name, local variable referenced before assignment,
           syntax error, duplicate argument, 'return'/'yield' outside
           function, two starred expressions, assertion is always true
           (an assert on a tuple -- it silently never fires)
  warn     anything pyflakes reports that is not on either list: printed,
           not fatal, so a new failure mode is never silently dropped
  quiet    imported but unused, redefinition of unused, f-string missing
           placeholders, star-import noise. Assembled scripts -- built by
           splicing one kernel's source into another's -- carry these by
           construction, and burying two real errors in forty cosmetic ones
           is how the real ones get skipped.

Usage:  py tools/checkgen.py zhpara.py ttacache.py ...
        py tools/checkgen.py            (every *.py changed in the worktree)

Exit code 1 if anything FATAL is found, so it can go in front of a push.
"""
import subprocess
import sys
import ast
import pathlib

FATAL = (
    "undefined name",
    "local variable",
    "duplicate argument",
    "outside function",
    "two starred expressions",
    "assertion is always true",
    "invalid syntax",
    "f-string: ",
)
QUIET = (
    "imported but unused",
    "redefinition of unused",
    "unable to detect undefined names",
    "may be undefined, or defined from star imports",
    "is assigned to but never used",
    "f-string is missing placeholders",
)


def targets():
    if len(sys.argv) > 1:
        return [pathlib.Path(a) for a in sys.argv[1:]]
    out = subprocess.run(["git", "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    ps = []
    for line in out.splitlines():
        f = line[3:].strip().strip('"')
        if f.endswith(".py") and pathlib.Path(f).exists():
            ps.append(pathlib.Path(f))
    return ps


def main():
    files = targets()
    if not files:
        print("nothing to check")
        return 0
    bad = 0
    for f in files:
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"FATAL {f}:{e.lineno}: SYNTAX {e.msg}")
            bad += 1
            continue
        r = subprocess.run([sys.executable, "-m", "pyflakes", str(f)],
                           capture_output=True, text=True)
        fatal, warn = [], []
        for ln in (r.stdout + r.stderr).splitlines():
            if not ln.strip() or any(s in ln for s in QUIET):
                continue
            (fatal if any(s in ln for s in FATAL) else warn).append(ln)
        for ln in fatal:
            print("FATAL " + ln)
        for ln in warn:
            print(" warn " + ln)
        bad += len(fatal)
        if fatal:
            state = "FAIL " + str(len(fatal))
        elif warn:
            state = "ok (" + str(len(warn)) + " warn)"
        else:
            state = "ok"
        print("  " + str(f) + ": " + state
              + "  (" + str(len(src.splitlines())) + " lines)")
    if bad:
        print("\n" + str(bad) + " FATAL finding(s). Do NOT push -- these are "
              "runtime failures, and a kernel takes hours to tell you the "
              "same thing.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
