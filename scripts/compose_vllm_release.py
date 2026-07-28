#!/usr/bin/env python3
"""Compose a release tree from a fresh base branch and pinned pull requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CompositionError(RuntimeError):
    pass


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CompositionError(f"{' '.join(args)} failed: {detail}")
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha(repository: str, ref: str) -> str:
    result = _run(["git", "ls-remote", repository, ref])
    matches = [line.split()[0] for line in result.stdout.splitlines() if line]
    if len(matches) != 1 or not SHA_RE.fullmatch(matches[0]):
        raise CompositionError(
            f"expected exactly one SHA for {repository} {ref}, got {matches}"
        )
    return matches[0]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"cannot read manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise CompositionError("manifest schema_version must be 1")
    for key in ("name", "repository", "base_ref", "pull_requests"):
        if key not in manifest:
            raise CompositionError(f"manifest is missing {key!r}")
    if not manifest["base_ref"].startswith("refs/heads/"):
        raise CompositionError("base_ref must be a full refs/heads/... ref")

    seen: set[int] = set()
    for entry in manifest["pull_requests"]:
        number = entry.get("number")
        head = entry.get("head")
        ref = entry.get("ref", f"refs/pull/{number}/head")
        if not isinstance(number, int) or number <= 0 or number in seen:
            raise CompositionError(f"invalid or duplicate PR number: {number!r}")
        if not isinstance(head, str) or not SHA_RE.fullmatch(head):
            raise CompositionError(f"PR #{number} must pin a 40-character head SHA")
        if ref != f"refs/pull/{number}/head":
            raise CompositionError(f"PR #{number} has unexpected ref {ref!r}")
        entry["ref"] = ref
        seen.add(number)
    return manifest


def compose(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = _load_manifest(manifest_path)
    repository = str(manifest["repository"])
    base_ref = str(manifest["base_ref"])
    base_sha = _remote_sha(repository, base_ref)

    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / "integration.patch"
    lock_path = output_dir / "integration.lock.json"

    with tempfile.TemporaryDirectory(prefix="source-release-compose-") as temp:
        checkout = Path(temp) / "checkout"
        _run(["git", "init", "--quiet", str(checkout)])
        _run(["git", "remote", "add", "origin", repository], cwd=checkout)
        _run(["git", "fetch", "--quiet", "--no-tags", "origin", base_ref], cwd=checkout)
        _run(["git", "checkout", "--quiet", "--detach", base_sha], cwd=checkout)

        merge_env = os.environ.copy()
        merge_env.update(
            {
                "GIT_AUTHOR_NAME": "Release Composer",
                "GIT_AUTHOR_EMAIL": "release-composer@local",
                "GIT_COMMITTER_NAME": "Release Composer",
                "GIT_COMMITTER_EMAIL": "release-composer@local",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            }
        )

        resolved_prs: list[dict[str, Any]] = []
        for entry in manifest["pull_requests"]:
            number = int(entry["number"])
            ref = str(entry["ref"])
            expected_head = str(entry["head"])
            remote_head = _remote_sha(repository, ref)
            if remote_head != expected_head:
                raise CompositionError(
                    f"PR #{number} moved: manifest pins {expected_head}, "
                    f"remote is {remote_head}; review and update the manifest"
                )

            local_ref = f"refs/remotes/origin/release-pr-{number}"
            _run(
                ["git", "fetch", "--quiet", "--no-tags", "origin", f"{ref}:{local_ref}"],
                cwd=checkout,
            )
            fetched_head = _run(["git", "rev-parse", local_ref], cwd=checkout).stdout.strip()
            if fetched_head != expected_head:
                raise CompositionError(
                    f"PR #{number} fetch mismatch: expected {expected_head}, got {fetched_head}"
                )

            already_in_base = (
                _run(
                    ["git", "merge-base", "--is-ancestor", expected_head, base_sha],
                    cwd=checkout,
                    check=False,
                ).returncode
                == 0
            )
            disposition = "already_in_base" if already_in_base else "merged"
            if not already_in_base:
                merge = _run(
                    [
                        "git",
                        "merge",
                        "--no-ff",
                        "--no-edit",
                        "-m",
                        f"release composition: PR #{number}",
                        expected_head,
                    ],
                    cwd=checkout,
                    env=merge_env,
                    check=False,
                )
                if merge.returncode:
                    detail = merge.stderr.strip() or merge.stdout.strip()
                    raise CompositionError(f"PR #{number} conflicts with the clean stack: {detail}")

            resolved_prs.append(
                {
                    "number": number,
                    "ref": ref,
                    "head": expected_head,
                    "title": entry.get("title", ""),
                    "disposition": disposition,
                }
            )

        _run(["git", "diff", "--check", base_sha, "HEAD"], cwd=checkout)
        result_tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=checkout).stdout.strip()
        patch = subprocess.run(
            ["git", "diff", "--binary", "--full-index", base_sha, "HEAD"],
            cwd=checkout,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        patch_path.write_bytes(patch)

        _run(["git", "reset", "--hard", "--quiet", base_sha], cwd=checkout)
        _run(["git", "clean", "-fdx", "--quiet"], cwd=checkout)
        if patch:
            _run(["git", "apply", "--index", str(patch_path)], cwd=checkout)
        applied_tree = _run(["git", "write-tree"], cwd=checkout).stdout.strip()
        if applied_tree != result_tree:
            raise CompositionError(
                f"patch verification produced tree {applied_tree}, expected {result_tree}"
            )

    lock = {
        "schema_version": 1,
        "name": manifest["name"],
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "base": {
            "repository": repository,
            "ref": base_ref,
            "commit": base_sha,
        },
        "pull_requests": resolved_prs,
        "result": {
            "tree": result_tree,
            "patch": patch_path.name,
            "patch_sha256": _sha256(patch_path),
        },
    }
    temporary_lock = lock_path.with_suffix(".json.tmp")
    temporary_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary_lock.replace(lock_path)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        lock = compose(args.manifest, args.output_dir)
    except CompositionError as exc:
        parser.error(str(exc))
    print(json.dumps(lock, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
