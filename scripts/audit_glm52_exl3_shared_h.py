#!/usr/bin/env python3
"""Audit the physical shared-H layout of a GLM-5.2 EXL3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


_SHARED_FIELDS = (
    ("gate_proj", "suh"),
    ("up_proj", "suh"),
    ("down_proj", "svh"),
)
_LOCAL_FIELDS = (
    ("gate_proj", "svh"),
    ("up_proj", "svh"),
    ("down_proj", "suh"),
)


def _parse_layers(value: str) -> tuple[int, ...]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            if stop < start:
                raise argparse.ArgumentTypeError(f"invalid layer range: {part}")
            layers.update(range(start, stop + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return tuple(sorted(layers))


def _shape(handle: Any, key: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in handle.get_slice(key).get_shape())
    except KeyError as exc:
        raise ValueError(f"missing tensor: {key}") from exc


def audit_checkpoint(
    checkpoint: Path,
    *,
    layers: tuple[int, ...],
    tp_size: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, Any]:
    config = json.loads((checkpoint / "config.json").read_text())
    metadata = config.get("hybrid_tr3_tail", {})
    if metadata.get("rotation_layout") != "shared_h_v1":
        raise ValueError("checkpoint does not declare rotation_layout=shared_h_v1")

    bitmap = json.loads((checkpoint / "tier_bitmap.json").read_text())
    layer_results: list[dict[str, Any]] = []
    for layer in layers:
        shard = checkpoint / f"model-layer-{layer:03d}.safetensors"
        if not shard.is_file():
            raise ValueError(f"missing layer shard: {shard.name}")
        bitrates = bitmap.get(str(layer), {}).get("k")
        if not isinstance(bitrates, list) or len(bitrates) != num_experts:
            raise ValueError(
                f"layer {layer} tier bitmap must contain {num_experts} entries"
            )
        invalid_bits = sorted({int(bits) for bits in bitrates} - {3, 4})
        if invalid_bits:
            raise ValueError(f"layer {layer} has unsupported bitrates: {invalid_bits}")

        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            shared_keys: set[str] = set()
            for rank in range(tp_size):
                for projection, field in _SHARED_FIELDS:
                    key = (
                        f"model.layers.{layer}.mlp.experts.shared_h."
                        f"{projection}.rank{rank}.{field}"
                    )
                    shared_keys.add(key)
                    shape = _shape(handle, key)
                    if shape not in {(hidden_size,), (1, hidden_size)}:
                        raise ValueError(
                            f"{key} must be one physical H row, got shape={shape}"
                        )

            for expert in range(num_experts):
                for rank in range(tp_size):
                    for projection, field in _SHARED_FIELDS:
                        old_key = (
                            f"model.layers.{layer}.mlp.experts.{expert}."
                            f"{projection}.rank{rank}.{field}"
                        )
                        if old_key in keys:
                            raise ValueError(
                                f"shared-H checkpoint retains per-expert tensor: {old_key}"
                            )
                    for projection, field in _LOCAL_FIELDS:
                        key = (
                            f"model.layers.{layer}.mlp.experts.{expert}."
                            f"{projection}.rank{rank}.{field}"
                        )
                        shape = _shape(handle, key)
                        if shape not in {
                            (intermediate_size,),
                            (1, intermediate_size),
                        }:
                            raise ValueError(
                                f"{key} must remain expert-local, got shape={shape}"
                            )

            unexpected_shared = {
                key for key in keys if ".experts.shared_h." in key
            } - shared_keys
            if unexpected_shared:
                raise ValueError(
                    "unexpected shared-H tensors: "
                    + ", ".join(sorted(unexpected_shared)[:8])
                )
            exl3_tensor_count = sum(
                ".mlp.experts." in key
                and any(
                    key.endswith(suffix)
                    for suffix in (".trellis", ".mcg", ".suh", ".svh")
                )
                for key in keys
            )
            expected_exl3_tensors = num_experts * tp_size * 3 * 3 + tp_size * 3
            if exl3_tensor_count != expected_exl3_tensors:
                raise ValueError(
                    f"layer {layer} has {exl3_tensor_count} EXL3 tensors; "
                    f"expected {expected_exl3_tensors}"
                )

        layer_results.append(
            {
                "layer": layer,
                "k3_experts": sum(int(bits) == 3 for bits in bitrates),
                "k4_experts": sum(int(bits) == 4 for bits in bitrates),
                "exl3_tensor_count": exl3_tensor_count,
            }
        )

    bytes_per_element = 2
    physical_per_rank = (
        len(layers) * len(_SHARED_FIELDS) * hidden_size * bytes_per_element
    )
    legacy_per_rank = physical_per_rank * num_experts
    saved_per_rank = legacy_per_rank - physical_per_rank
    return {
        "checkpoint": str(checkpoint.resolve()),
        "rotation_layout": metadata["rotation_layout"],
        "producer_version": metadata.get("producer_version"),
        "layers": list(layers),
        "layer_count": len(layers),
        "tp_size": tp_size,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "shared_h_physical_bytes_per_rank": physical_per_rank,
        "shared_h_legacy_bytes_per_rank": legacy_per_rank,
        "shared_h_saved_bytes_per_rank": saved_per_rank,
        "shared_h_saved_mib_per_rank": saved_per_rank / (1 << 20),
        "shared_h_saved_bytes_checkpoint": saved_per_rank * tp_size,
        "layer_results": layer_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("3-78"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_checkpoint(
        args.checkpoint,
        layers=args.layers,
        tp_size=args.tp_size,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
