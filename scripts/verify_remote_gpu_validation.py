#!/usr/bin/env python3
"""Verify that remote GPU validation belongs to the exact release image."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_TESTS = (
    "source_contracts",
    "runtime_contracts",
    "model_startup",
    "correctness",
    "decode",
    "prefill",
)


def verify_receipt(
    receipt: dict[str, Any],
    *,
    image: str,
    image_id: str,
    vllm_tree: str,
    b12x_tree: str,
) -> None:
    """Validate identity and required test results in a release receipt."""
    backend_tree_key = (
        "b12x_tree" if "b12x_tree" in receipt else "sparkinfer_tree"
    )
    expected = {
        "schema_version": 1,
        "image": image,
        "image_id": image_id,
        "vllm_tree": vllm_tree,
        backend_tree_key: b12x_tree,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"receipt {key} mismatch: expected {value!r}, "
                f"got {receipt.get(key)!r}"
            )

    host = receipt.get("validator_host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("receipt validator_host must be a non-empty string")

    gpu_ids = receipt.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
    ):
        raise ValueError("receipt gpu_ids must be a non-empty list of GPU IDs")

    validated_at = receipt.get("validated_at")
    if not isinstance(validated_at, str):
        raise ValueError("receipt validated_at must be an ISO-8601 timestamp")
    try:
        datetime.fromisoformat(validated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "receipt validated_at must be an ISO-8601 timestamp"
        ) from exc

    tests = receipt.get("tests")
    if not isinstance(tests, dict):
        raise ValueError("receipt tests must be an object")
    failed = [name for name in REQUIRED_TESTS if tests.get(name) != "pass"]
    if failed:
        raise ValueError(
            "receipt is missing passing release tests: " + ", ".join(failed)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--vllm-tree", required=True)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--b12x-tree")
    backend.add_argument("--sparkinfer-tree")
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    verify_receipt(
        receipt,
        image=args.image,
        image_id=args.image_id,
        vllm_tree=args.vllm_tree,
        b12x_tree=args.b12x_tree or args.sparkinfer_tree,
    )
    print(f"Remote GPU validation receipt: PASS ({args.receipt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
