#!/usr/bin/env python3
"""Fail a push or pull request that adds training dumps or log data.

A changed file (added or modified) fails when either of these is true:

* it lives under the top-level ``data/``, ``exports/``, or ``logs/`` directory
* its name ends in ``.jsonl``, ``.parquet``, ``.csv``, ``.ndjson``, ``.npy``,
  or ``.pt`` and the blob is larger than 50 KB (51200 bytes)

Deletions are allowed. Files already on the base commit are not inspected
unless this push or pull request changes them.

Model weights stay on the path this repo already uses: ``*.pt`` and
``*.tar.gz`` are gitignored, checkpoint notes live under ``manifests/``, and
the binaries are stored in S3 or attached to a GitHub Release. This check
does not look at release assets. A ``.pt`` or ``.tar.gz`` path that is
already tracked on the base ref may still be updated, so an existing
checkpoint blob is not treated as a newly published weight. A new
``.tar.gz`` is rejected. That exception does not cover ``data/``,
``exports/``, or ``logs/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SIZE_LIMIT_BYTES = 50 * 1024
DUMP_SUFFIXES = (".jsonl", ".parquet", ".csv", ".ndjson", ".npy", ".pt")
RESTRICTED_ROOTS = frozenset({"data", "exports", "logs"})
WEIGHT_SUFFIXES = (".pt", ".tar.gz")


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(["git", *args], capture_output=True)
    if check and result.returncode != 0:
        sys.stderr.write(result.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"git {' '.join(args)} failed with status {result.returncode}")
    return result


def rev_exists(rev: str) -> bool:
    result = _run_git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], check=False)
    return result.returncode == 0


def _empty_tree() -> str:
    result = subprocess.run(
        ["git", "hash-object", "-t", "tree", "--stdin"],
        input=b"",
        capture_output=True,
        check=True,
    )
    return result.stdout.decode().strip()


def blob_size(rev: str, path: str) -> int | None:
    result = _run_git(["cat-file", "-s", f"{rev}:{path}"], check=False)
    if result.returncode != 0:
        return None
    return int(result.stdout.decode().strip())


def changed_files(base: str, head: str, mode: str) -> list[tuple[str, str]]:
    """Return (status, path) for files this range adds or modifies.

    ``mode`` is ``two`` (``git diff base head``) or ``triple``
    (``git diff base...head``). Rename detection is off so a rename is an
    add of the destination plus a delete of the source.
    """
    if mode == "triple":
        spec = [f"{base}...{head}"]
    elif mode == "two":
        spec = [base, head]
    else:
        raise SystemExit(f"unknown diff mode {mode!r}")
    result = _run_git(
        ["diff", "--no-renames", "--name-status", "-z", "--no-ext-diff", *spec]
    )
    parts = result.stdout.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(parts):
        status = parts[index].decode("utf-8", "surrogateescape")
        index += 1
        if not status:
            continue
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(parts):
                raise SystemExit("truncated git diff name-status output")
            index += 1  # old path
            path = parts[index].decode("utf-8", "surrogateescape")
            index += 1
        else:
            if index >= len(parts):
                raise SystemExit("truncated git diff name-status output")
            path = parts[index].decode("utf-8", "surrogateescape")
            index += 1
        if code == "D":
            continue
        if code in {"A", "M", "T", "R", "C"}:
            entries.append((code, path))
    return entries


def restricted_dir(path: str) -> bool:
    parts = path.split("/")
    return bool(parts) and parts[0].lower() in RESTRICTED_ROOTS and len(parts) > 1


def dump_suffix(path: str) -> str | None:
    lower = path.lower()
    for suffix in DUMP_SUFFIXES:
        if lower.endswith(suffix):
            return suffix
    return None


def is_weight_path(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def grandfathered_weight(base: str, path: str) -> bool:
    """True when this path is a weight blob already tracked on the base ref."""
    if restricted_dir(path) or not is_weight_path(path):
        return False
    return blob_size(base, path) is not None


def violation(base: str, head: str, path: str) -> str | None:
    if grandfathered_weight(base, path):
        return None
    if restricted_dir(path):
        size = blob_size(head, path)
        size_note = f"{size} bytes" if size is not None else "unreadable blob"
        return f"{path}: file under {path.split('/')[0]}/ ({size_note})"
    if path.lower().endswith(".tar.gz"):
        size = blob_size(head, path)
        size_note = f"{size} bytes" if size is not None else "unreadable blob"
        return (
            f"{path}: new model archive ({size_note}). Weights can be published "
            "only when that path is already tracked; new binaries belong in S3 "
            "or a GitHub Release, with the checkpoint noted under manifests/."
        )
    suffix = dump_suffix(path)
    if suffix is None:
        return None
    size = blob_size(head, path)
    if size is None:
        return f"{path}: {suffix} dataset dump could not be sized"
    if size > SIZE_LIMIT_BYTES:
        return (
            f"{path}: {suffix} dataset dump is {size} bytes "
            f"(limit is {SIZE_LIMIT_BYTES} bytes / 50 KB)"
        )
    return None


def collect_violations(base: str, head: str, mode: str) -> list[str]:
    found: list[str] = []
    for _status, path in changed_files(base, head, mode):
        message = violation(base, head, path)
        if message:
            found.append(message)
    return found


def default_range() -> tuple[str, str, str]:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "pull_request":
        base = os.environ.get("PR_BASE_SHA") or ""
        head = os.environ.get("PR_HEAD_SHA") or "HEAD"
        if not base:
            raise SystemExit("PR_BASE_SHA is required for pull_request events")
        return base, head, "triple"
    if event == "push":
        before = os.environ.get("PUSH_BEFORE", "")
        after = os.environ.get("PUSH_AFTER") or os.environ.get("GITHUB_SHA") or "HEAD"
        if before and set(before) != {"0"}:
            return before, after, "two"
        for candidate in ("origin/main", "main"):
            if rev_exists(candidate):
                return candidate, after, "triple"
        return _empty_tree(), after, "two"
    for candidate in ("origin/main", "main"):
        if rev_exists(candidate):
            return candidate, "HEAD", "triple"
    raise SystemExit("Could not find main or origin/main to diff against")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Base commit. Defaults to the GitHub event range.")
    parser.add_argument("--head", help="Head commit. Defaults to the GitHub event range.")
    parser.add_argument(
        "--mode",
        choices=("two", "triple"),
        help="two = git diff base head; triple = git diff base...head",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.base or args.head or args.mode:
        if not args.base or not args.head:
            raise SystemExit("--base and --head are both required when overriding the range")
        base, head, mode = args.base, args.head, args.mode or "two"
    else:
        base, head, mode = default_range()
    if not rev_exists(base):
        raise SystemExit(f"base revision not found: {base}")
    if not rev_exists(head):
        raise SystemExit(f"head revision not found: {head}")
    files = changed_files(base, head, mode)
    violations = collect_violations(base, head, mode)
    if violations:
        sys.stderr.write(
            "This push or pull request adds training data, log data, or a new "
            "model archive, which cannot be published in this public repo.\n"
        )
        sys.stderr.write(
            "Model weights are unchanged by this rule when they are already "
            "tracked (manifests document the checkpoint; *.pt and *.tar.gz stay "
            "gitignored, with binaries in S3 or a GitHub Release).\n"
        )
        for item in violations:
            sys.stderr.write(f"  - {item}\n")
        return 1
    print(
        f"No new training or log data in {mode}-dot {base}..{head} "
        f"({len(files)} changed file(s) checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
