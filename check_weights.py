"""
Inspect the published weight artefacts to find out what each checkpoint trained
on. CPU only, seconds to run.

Attach:
    pilkwang/biohub-tracking-support-pack-50ep-v1
    pilkwang/biohub-deepcenter-unet3d-center-prior-v1

OUTPUT DISCIPLINE (engineering-rules.md rule 2): this summarises, it does not
dump. Nothing here prints a raw array or a full name list.

WALK DISCIPLINE: /kaggle/input also holds the competition data, which is
~200 .zarr stores containing millions of chunk files. A plain rglob("*") over
it appears to hang for many minutes. This walks with os.scandir and PRUNES
.zarr directories and the competition mount, so it finishes in seconds.
"""

import json
import os

ROOT = "/kaggle/input"

# The 19 datasets our split derivation predicts the primary model held out.
DERIVED_HOLDOUT = {
    "6bba_268e1230", "44b6_d754aa59", "6bba_d1acb6ff", "6bba_7af54fde",
    "44b6_d5e7d891", "6bba_5c824876", "6bba_7b5d3b2c", "6bba_afb141ff",
    "6bba_2312ac41", "44b6_1574802b", "6bba_aeee7805", "6bba_283bf9f1",
    "6bba_3abfe10a", "6bba_c328f2fd", "44b6_e57ff5c6", "6bba_61dd1e0d",
    "6bba_c27cba08", "44b6_706092f0", "6bba_3a1849c2",
}

SPLITTY = ("split", "fold", "train", "val", "test", "holdout", "seed",
           "data_dir", "splits")

MAX_FILES = 400          # hard stop, so a surprise mount cannot flood the cell
PRUNE_SUFFIX = (".zarr", ".geff", ".n5", ".ome")
PRUNE_NAME = {"__pycache__", ".git"}


def _p(msg=""):
    print(msg, flush=True)


def _short(v, n=120):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"... [{len(s)} chars]"


def walk():
    """Every file under ROOT except pruned trees. Returns (paths, pruned, hit_cap)."""
    out, pruned, stack = [], [], [ROOT]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError as e:
            _p(f"[degraded] cannot list {d}: {e}")
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if e.name in PRUNE_NAME or e.name.endswith(PRUNE_SUFFIX):
                    pruned.append(e.path)
                    continue
                stack.append(e.path)
            elif e.is_file(follow_symlinks=False):
                out.append(e.path)
                if len(out) > MAX_FILES * 20:
                    return sorted(out), pruned, True
    return sorted(out), pruned, False


def files(paths, pruned, capped):
    _p("=== files ===")
    if capped:
        _p("[degraded] file walk hit its cap; listing is incomplete")
    wheels = shown = skipped = 0
    for p in paths:
        if p.endswith(".whl"):
            wheels += 1
            continue
        if shown >= MAX_FILES:
            skipped += 1
            continue
        _p(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p) // 1024} KB")
        shown += 1
    _p(f"  (+{wheels} wheels, +{skipped} more files, not listed)")
    if pruned:
        _p(f"  ({len(pruned)} data trees pruned, e.g. "
           f"{os.path.relpath(pruned[0], ROOT)})")


def jsons(paths):
    _p("\n=== json: keys, and any list of dataset names ===")
    found = False
    for p in paths:
        if not p.endswith(".json"):
            continue
        found = True
        name = os.path.relpath(p, ROOT)
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            _p(f"  {name}: unreadable ({e})")
            continue
        if not isinstance(d, dict):
            _p(f"  {name}: {type(d).__name__}")
            continue
        _p(f"  {name}: keys = {sorted(d)}")
        for k, v in d.items():
            # Every list of strings gets the overlap test, whatever it is
            # called -- a list named "all" answers the question as well as
            # one named "val".
            if isinstance(v, list):
                names = [x for x in v if isinstance(x, str)]
                if names:
                    hit = DERIVED_HOLDOUT & set(names)
                    _p(f"    {k}: {len(names)} datasets, "
                       f"{len(hit)}/19 of our derived holdout in it")
                    if 0 < len(hit) <= 25:
                        _p(f"      overlap: {sorted(hit)}")
                else:
                    _p(f"    {k}: list[{len(v)}] (not dataset names)")
            elif any(s in k.lower() for s in SPLITTY):
                _p(f"    {k}: {_short(v)}")
    if not found:
        _p("  none found")


def checkpoints(paths):
    _p("\n=== checkpoints: scalar metadata only ===")
    cks = [p for p in paths if p.endswith((".pt", ".pth", ".ckpt"))]
    if not cks:
        _p("  none found")
        return
    import torch
    for p in cks:
        _p(f"  {os.path.relpath(p, ROOT)}  "
           f"({os.path.getsize(p) // 2**20} MB)")
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            _p(f"    could not load: {e}")
            continue
        if not isinstance(ck, dict):
            _p(f"    {type(ck).__name__}, not a metadata dict")
            continue
        for k, v in ck.items():
            if hasattr(v, "shape") or (isinstance(v, dict) and len(v) > 20):
                _p(f"    {k}: <weights/state, skipped>")
            elif isinstance(v, list) and len(v) > 3:
                names = [x for x in v if isinstance(x, str)]
                if names:
                    hit = DERIVED_HOLDOUT & set(names)
                    _p(f"    {k}: {len(names)} datasets, "
                       f"{len(hit)}/19 of our derived holdout in it")
                else:
                    _p(f"    {k}: list[{len(v)}], last = {_short(v[-1])}")
            else:
                _p(f"    {k}: {_short(v, 200)}")


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit("no /kaggle/input -- attach the weight datasets first")
    _p(f"walking {ROOT} (pruning .zarr trees) ...")
    paths, pruned, capped = walk()
    _p(f"{len(paths)} files, {len(pruned)} trees pruned")
    files(paths, pruned, capped)
    jsons(paths)
    checkpoints(paths)
    _p("\nThe question: does any artefact name its own train/val datasets? "
       "If a split list overlaps our 19 derived holdout datasets, that "
       "component trained on them and our 'held-out' score is contaminated.")


if __name__ == "__main__":
    main()
