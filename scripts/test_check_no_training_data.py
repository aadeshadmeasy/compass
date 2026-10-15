#!/usr/bin/env python3
"""Tests for the public-repo training and log data check."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_no_training_data as check  # noqa: E402

LIMIT = check.SIZE_LIMIT_BYTES


def git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Compass Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Compass Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        capture_output=True,
        env=env,
        check=True,
    )
    return result.stdout.decode()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


class CheckNoTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "Compass Test")
        (self.repo / "README.md").write_text("public\n")
        manifests = self.repo / "manifests" / "stage-a"
        manifests.mkdir(parents=True)
        (manifests / "README.md").write_text("checkpoint notes\n")
        (manifests / "metrics.json").write_text("{}\n")
        # An already-tracked weight, the way a historical checkpoint blob would be.
        (manifests / "model.pt").write_bytes(b"W" * (LIMIT + 100))
        (manifests / "model.tar.gz").write_bytes(b"T" * (LIMIT + 100))
        sample = self.repo / "data"
        sample.mkdir()
        (sample / "routing_trajectories_sample.jsonl").write_bytes(b"{" * (LIMIT + 10))
        self.base = commit_all(self.repo, "base")
        self._old = os.getcwd()
        os.chdir(self.repo)

    def tearDown(self) -> None:
        os.chdir(self._old)
        self._tmp.cleanup()

    def violations_since_base(self) -> list[str]:
        head = git(self.repo, "rev-parse", "HEAD").strip()
        return check.collect_violations(self.base, head, "two")

    def test_unrelated_change_passes(self) -> None:
        (self.repo / "README.md").write_text("public\nmore\n")
        commit_all(self.repo, "docs")
        self.assertEqual(self.violations_since_base(), [])

    def test_manifest_metrics_update_passes(self) -> None:
        path = self.repo / "manifests" / "stage-a" / "metrics.json"
        path.write_text('{"recall": 0.7}\n')
        commit_all(self.repo, "metrics")
        self.assertEqual(self.violations_since_base(), [])

    def test_existing_weight_update_passes(self) -> None:
        path = self.repo / "manifests" / "stage-a" / "model.pt"
        path.write_bytes(b"W" * (LIMIT + 500))
        commit_all(self.repo, "weight")
        self.assertEqual(self.violations_since_base(), [])

    def test_existing_archive_update_passes(self) -> None:
        path = self.repo / "manifests" / "stage-a" / "model.tar.gz"
        path.write_bytes(b"T" * (LIMIT + 800))
        commit_all(self.repo, "archive")
        self.assertEqual(self.violations_since_base(), [])

    def test_new_archive_fails(self) -> None:
        path = self.repo / "manifests" / "stage-b" / "model.tar.gz"
        path.parent.mkdir()
        path.write_bytes(b"tar")
        commit_all(self.repo, "new archive")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("model.tar.gz", messages[0])
        self.assertIn("already tracked", messages[0])

    def test_new_large_weight_fails(self) -> None:
        path = self.repo / "manifests" / "stage-b" / "model.pt"
        path.parent.mkdir()
        path.write_bytes(b"N" * (LIMIT + 1))
        commit_all(self.repo, "new weight")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("model.pt", messages[0])

    def test_new_small_pt_outside_data_passes(self) -> None:
        (self.repo / "notes.pt").write_bytes(b"n" * LIMIT)
        commit_all(self.repo, "tiny tensor")
        self.assertEqual(self.violations_since_base(), [])

    def test_directories_fail_at_any_size(self) -> None:
        (self.repo / "logs").mkdir()
        (self.repo / "logs" / "app.log").write_text("x\n")
        (self.repo / "exports").mkdir()
        (self.repo / "exports" / "out.txt").write_text("y\n")
        (self.repo / "data" / "extra.json").write_text("{}\n")
        commit_all(self.repo, "dumps")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 3)
        joined = "\n".join(messages)
        self.assertIn("logs/app.log", joined)
        self.assertIn("exports/out.txt", joined)
        self.assertIn("data/extra.json", joined)

    def test_case_variant_directory_fails(self) -> None:
        (self.repo / "Data").mkdir()
        (self.repo / "Data" / "row.csv").write_text("a,b\n")
        commit_all(self.repo, "case")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("Data/row.csv", messages[0])

    def test_modifying_tracked_sample_fails(self) -> None:
        path = self.repo / "data" / "routing_trajectories_sample.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        commit_all(self.repo, "grow sample")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("data/routing_trajectories_sample.jsonl", messages[0])

    def test_small_dump_outside_restricted_dirs_passes(self) -> None:
        (self.repo / "docs").mkdir()
        (self.repo / "docs" / "example.jsonl").write_text('{"id": 1}\n')
        commit_all(self.repo, "tiny example")
        self.assertEqual(self.violations_since_base(), [])

    def test_large_dump_outside_restricted_dirs_fails(self) -> None:
        (self.repo / "train.csv").write_bytes(b"a,b\n" + b"1,2\n" * (LIMIT // 2))
        commit_all(self.repo, "csv")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("train.csv", messages[0])
        self.assertIn(".csv", messages[0])

    def test_size_limit_is_exclusive(self) -> None:
        (self.repo / "exact.npy").write_bytes(b"\0" * LIMIT)
        commit_all(self.repo, "exact")
        self.assertEqual(self.violations_since_base(), [])
        (self.repo / "over.ndjson").write_bytes(b"\0" * (LIMIT + 1))
        commit_all(self.repo, "over")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("over.ndjson", messages[0])

    def test_parquet_suffix_and_deletion(self) -> None:
        (self.repo / "logs").mkdir()
        (self.repo / "logs" / "old.log").write_text("gone\n")
        commit_all(self.repo, "add log")
        self.assertTrue(self.violations_since_base())
        (self.repo / "logs" / "old.log").unlink()
        commit_all(self.repo, "delete log")
        # Net diff from the original base still contains the add-then-delete,
        # so compare only the delete commit against its parent.
        parent = git(self.repo, "rev-parse", "HEAD^").strip()
        head = git(self.repo, "rev-parse", "HEAD").strip()
        self.assertEqual(check.collect_violations(parent, head, "two"), [])

    def test_new_large_parquet_fails(self) -> None:
        (self.repo / "rows.parquet").write_bytes(b"P" * (LIMIT + 1))
        commit_all(self.repo, "parquet")
        messages = self.violations_since_base()
        self.assertEqual(len(messages), 1)
        self.assertIn("rows.parquet", messages[0])

    def test_cli_success_and_failure(self) -> None:
        script = ROOT / "scripts" / "check_no_training_data.py"
        head = git(self.repo, "rev-parse", "HEAD").strip()
        ok = subprocess.run(
            [sys.executable, str(script), "--base", self.base, "--head", head, "--mode", "two"],
            cwd=self.repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        (self.repo / "exports").mkdir()
        (self.repo / "exports" / "leak.txt").write_text("nope\n")
        bad_head = commit_all(self.repo, "leak")
        bad = subprocess.run(
            [sys.executable, str(script), "--base", self.base, "--head", bad_head, "--mode", "two"],
            cwd=self.repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn("exports/leak.txt", bad.stderr)


class GitignoreTest(unittest.TestCase):
    def test_repo_gitignore_matches_policy(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text()
        for needle in ("data/**", "exports/**", "logs/**", "*.jsonl", "*.parquet", "*.csv", "*.ndjson", "*.npy", "*.pt"):
            self.assertIn(needle, gitignore)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / ".gitignore").write_text(gitignore)
            (repo / "data").mkdir()
            (repo / "data" / "routing_trajectories_sample.jsonl").write_text("sample\n")
            (repo / "data" / "tool_catalog_sample.jsonl").write_text("sample\n")
            (repo / "data" / "new.jsonl").write_text("nope\n")
            (repo / "exports").mkdir()
            (repo / "exports" / "a.txt").write_text("nope\n")
            (repo / "logs").mkdir()
            (repo / "logs" / "a.log").write_text("nope\n")
            (repo / "manifests").mkdir()
            (repo / "manifests" / "metrics.json").write_text("{}\n")
            (repo / "manifests" / "model.pt").write_text("weight\n")
            (repo / "manifests" / "model.tar.gz").write_bytes(b"tar")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            )
            names = set(staged.stdout.split())
            self.assertIn("data/routing_trajectories_sample.jsonl", names)
            self.assertIn("data/tool_catalog_sample.jsonl", names)
            self.assertIn("manifests/metrics.json", names)
            self.assertNotIn("data/new.jsonl", names)
            self.assertNotIn("exports/a.txt", names)
            self.assertNotIn("logs/a.log", names)
            self.assertNotIn("manifests/model.pt", names)
            self.assertNotIn("manifests/model.tar.gz", names)


if __name__ == "__main__":
    unittest.main()
