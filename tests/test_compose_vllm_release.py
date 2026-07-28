from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.compose_vllm_release import CompositionError, compose


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _commit(repo: Path, name: str, contents: str, message: str) -> str:
    (repo / name).write_text(contents)
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repositories(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Test")
    _git(source, "config", "user.email", "test@example.com")
    base = _commit(source, "model.py", "base\n", "base")
    _git(source, "branch", "-M", "dev/gilded-gnosis")
    _git(tmp_path, "init", "--quiet", "--bare", str(remote))
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "--quiet", "origin", "dev/gilded-gnosis")
    return source, remote, base


def _create_pr(
    source: Path,
    remote: Path,
    number: int,
    name: str,
    contents: str,
) -> str:
    _git(source, "checkout", "--quiet", "-B", f"pr-{number}", "dev/gilded-gnosis")
    head = _commit(source, name, contents, f"PR {number}")
    _git(source, "push", "--quiet", str(remote), f"HEAD:refs/pull/{number}/head")
    _git(source, "checkout", "--quiet", "dev/gilded-gnosis")
    return head


def _manifest(path: Path, remote: Path, prs: list[tuple[int, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-release",
                "repository": str(remote),
                "base_ref": "refs/heads/dev/gilded-gnosis",
                "pull_requests": [
                    {"number": number, "head": head} for number, head in prs
                ],
            }
        )
    )
    return path


def test_composition_uses_latest_clean_base_and_replays_patch(tmp_path: Path) -> None:
    source, remote, old_base = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")

    new_base = _commit(source, "base-update.py", "new base\n", "advance base")
    _git(source, "push", "--quiet", "origin", "dev/gilded-gnosis")
    assert new_base != old_base

    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, pr_head)])
    output = tmp_path / "output"
    lock = compose(manifest, output)

    assert lock["base"]["commit"] == new_base
    assert lock["pull_requests"][0]["disposition"] == "merged"
    assert (output / "integration.patch").is_file()

    replay = tmp_path / "replay"
    _git(tmp_path, "clone", "--quiet", str(remote), str(replay))
    _git(replay, "checkout", "--quiet", "--detach", new_base)
    _git(replay, "apply", "--index", str(output / "integration.patch"))
    assert _git(replay, "write-tree") == lock["result"]["tree"]
    assert (replay / "base-update.py").read_text() == "new base\n"
    assert (replay / "feature.py").read_text() == "feature\n"


def test_composition_rejects_moved_pr_head(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, "0" * 40)])

    with pytest.raises(CompositionError, match="PR #1 moved"):
        compose(manifest, tmp_path / "output")
    assert pr_head != "0" * 40


def test_composition_identifies_conflicting_pr(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    first = _create_pr(source, remote, 1, "model.py", "first\n")
    second = _create_pr(source, remote, 2, "model.py", "second\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, first), (2, second)])

    with pytest.raises(CompositionError, match="PR #2 conflicts"):
        compose(manifest, tmp_path / "output")


@pytest.mark.parametrize(
    ("component", "lock_sha256", "tree"),
    [
        (
            "vllm",
            "0c4ef43a7f01c4e0ac011edd813af1f840c272fdb253fb7598dbc15cd0880c2f",
            "99287e8898587f536b5710e25d1b65229f1d6d78",
        ),
        (
            "sparkinfer",
            "f3e066067bfa35f5c6d86778c16fee091305e57e715203f89266f122980f682b",
            "4ecc87fbe51090b7932e3ba8fa06d9649296ba38",
        ),
    ],
)
def test_r5_release_archive_matches_lock(
    component: str,
    lock_sha256: str,
    tree: str,
) -> None:
    release_dir = (
        REPOSITORY_ROOT
        / "patches"
        / "releases"
        / "gilded-gnosis-v20-r5"
        / component
    )
    lock_bytes = (release_dir / "integration.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    patch_bytes = (release_dir / lock["result"]["patch"]).read_bytes()

    assert hashlib.sha256(lock_bytes).hexdigest() == lock_sha256
    assert hashlib.sha256(patch_bytes).hexdigest() == lock["result"]["patch_sha256"]
    assert lock["result"]["tree"] == tree
