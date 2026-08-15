from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_release_prs.py"
_SPEC = importlib.util.spec_from_file_location("audit_release_prs", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

AuditError = _MODULE.AuditError
audit_manifest = _MODULE.audit_manifest


def _manifest() -> dict:
    return {
        "pull_requests": [{"number": 10, "head": "a" * 40}],
        "reviewed_exclusions": [
            {
                "number": 11,
                "head": "b" * 40,
                "disposition": "superseded",
                "reason": "Covered by PR #10.",
            }
        ],
    }


def _pull(number: int, head: str) -> dict:
    return {"number": number, "head": {"sha": head}}


def test_audit_accepts_pinned_included_and_excluded_prs() -> None:
    audit_manifest(_manifest(), [_pull(10, "a" * 40), _pull(11, "b" * 40)])


@pytest.mark.parametrize("disposition", ["research-only", "unsupported"])
def test_audit_accepts_release_status_exclusions(disposition: str) -> None:
    manifest = _manifest()
    manifest["reviewed_exclusions"][0]["disposition"] = disposition

    audit_manifest(manifest, [_pull(11, "b" * 40)])


def test_audit_rejects_unclassified_open_pr() -> None:
    with pytest.raises(AuditError, match="open PR #12 is not classified"):
        audit_manifest(_manifest(), [_pull(12, "c" * 40)])


def test_audit_rejects_moved_excluded_pr() -> None:
    with pytest.raises(AuditError, match="open PR #11 .* moved"):
        audit_manifest(_manifest(), [_pull(11, "c" * 40)])


def test_audit_rejects_duplicate_classification() -> None:
    manifest = _manifest()
    manifest["reviewed_exclusions"][0]["number"] = 10

    with pytest.raises(AuditError, match="PR #10 is classified more than once"):
        audit_manifest(manifest, [])
