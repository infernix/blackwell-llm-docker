#!/usr/bin/env python3
"""Fail a clean release when an open base-branch PR was not reviewed."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_EXCLUSION_DISPOSITIONS = {
    "alternative",
    "experimental",
    "out_of_scope",
    "superseded",
}


class AuditError(RuntimeError):
    """Release PR audit failed."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def _repository_slug(repository: str) -> str:
    path = urllib.parse.urlparse(repository).path.removesuffix(".git").strip("/")
    if len(path.split("/")) != 2:
        raise AuditError(f"unsupported GitHub repository URL: {repository!r}")
    return path


def _fetch_open_pulls(slug: str, base: str) -> list[dict[str, Any]]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    pulls: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"state": "open", "base": base, "per_page": 100, "page": page}
        )
        request = urllib.request.Request(
            f"https://api.github.com/repos/{slug}/pulls?{query}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "blackwell-llm-release-audit",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                batch = json.load(response)
        except Exception as exc:
            raise AuditError(f"cannot query open PRs for {slug}: {exc}") from exc
        if not isinstance(batch, list):
            raise AuditError(f"GitHub returned a non-list PR payload for {slug}")
        pulls.extend(batch)
        if len(batch) < 100:
            return pulls
        page += 1


def audit_manifest(manifest: dict[str, Any], pulls: list[dict[str, Any]]) -> None:
    included = manifest.get("pull_requests")
    excluded = manifest.get("reviewed_exclusions")
    if not isinstance(included, list) or not isinstance(excluded, list):
        raise AuditError("manifest must contain pull_requests and reviewed_exclusions")

    classified: dict[int, tuple[str, str]] = {}
    for disposition, entries in (("included", included), ("excluded", excluded)):
        for entry in entries:
            number = int(entry["number"])
            head = str(entry["head"])
            if number in classified:
                raise AuditError(f"PR #{number} is classified more than once")
            if not _SHA_RE.fullmatch(head):
                raise AuditError(f"PR #{number} has an invalid pinned head {head!r}")
            if disposition == "excluded":
                category = entry.get("disposition")
                reason = entry.get("reason")
                if category not in _EXCLUSION_DISPOSITIONS or not reason:
                    raise AuditError(f"PR #{number} has an incomplete exclusion record")
            classified[number] = (disposition, head)

    errors: list[str] = []
    for pull in pulls:
        number = int(pull["number"])
        live_head = str(pull["head"]["sha"])
        record = classified.get(number)
        if record is None:
            errors.append(f"open PR #{number} is not classified")
            continue
        disposition, pinned_head = record
        if live_head != pinned_head:
            errors.append(
                f"open PR #{number} ({disposition}) moved: "
                f"pinned {pinned_head}, live {live_head}"
            )
    if errors:
        raise AuditError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--pulls-json",
        type=Path,
        help="Use a saved GitHub pulls response instead of the network.",
    )
    args = parser.parse_args()

    try:
        manifest = _load_json(args.manifest)
        if args.pulls_json:
            pulls = _load_json(args.pulls_json)
        else:
            slug = _repository_slug(str(manifest["repository"]))
            base = str(manifest["base_ref"]).removeprefix("refs/heads/")
            pulls = _fetch_open_pulls(slug, base)
        audit_manifest(manifest, pulls)
    except (AuditError, KeyError, TypeError, ValueError) as exc:
        print(f"release PR audit failed: {exc}", file=sys.stderr)
        return 1

    print(f"release PR audit: PASS ({len(pulls)} open base-branch PRs classified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
