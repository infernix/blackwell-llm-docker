from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.verify_remote_gpu_validation import REQUIRED_TESTS, verify_receipt


IMAGE = "registry.example/vllm:test"
IMAGE_ID = "sha256:" + "1" * 64
VLLM_TREE = "2" * 40
B12X_TREE = "3" * 40


def _receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "vllm_tree": VLLM_TREE,
        "b12x_tree": B12X_TREE,
        "validator_host": "gpu-host.example",
        "gpu_ids": [4, 5, 6, 7],
        "validated_at": "2026-08-03T12:00:00Z",
        "tests": {name: "pass" for name in REQUIRED_TESTS},
    }


def _verify(receipt: dict[str, object]) -> None:
    verify_receipt(
        receipt,
        image=IMAGE,
        image_id=IMAGE_ID,
        vllm_tree=VLLM_TREE,
        b12x_tree=B12X_TREE,
    )


def test_accepts_receipt_for_exact_image_and_source_trees() -> None:
    _verify(_receipt())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "registry.example/vllm:other"),
        ("image_id", "sha256:" + "4" * 64),
        ("vllm_tree", "5" * 40),
        ("b12x_tree", "6" * 40),
    ],
)
def test_rejects_receipt_for_different_artifact(
    field: str, value: str
) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match=f"receipt {field} mismatch"):
        _verify(receipt)


def test_rejects_receipt_with_missing_release_test() -> None:
    receipt = deepcopy(_receipt())
    tests = receipt["tests"]
    assert isinstance(tests, dict)
    tests["prefill"] = "fail"
    with pytest.raises(ValueError, match="prefill"):
        _verify(receipt)


def test_accepts_legacy_sparkinfer_tree_field() -> None:
    receipt = _receipt()
    receipt["sparkinfer_tree"] = receipt.pop("b12x_tree")
    _verify(receipt)
