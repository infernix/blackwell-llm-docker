from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


_SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_glm52_exl3_shared_h.py"
_SPEC = importlib.util.spec_from_file_location("audit_glm52_exl3_shared_h", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _checkpoint(tmp_path: Path, *, retain_old_h: bool = False) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "hybrid_tr3_tail": {
                    "rotation_layout": "shared_h_v1",
                    "producer_version": "test",
                }
            }
        )
    )
    (checkpoint / "tier_bitmap.json").write_text(json.dumps({"3": {"k": [3, 4]}}))
    tensors: dict[str, torch.Tensor] = {}
    for projection, field in _MODULE._SHARED_FIELDS:
        tensors[f"model.layers.3.mlp.experts.shared_h.{projection}.rank0.{field}"] = (
            torch.ones(8, dtype=torch.float16)
        )
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.3.mlp.experts.{expert}.{projection}.rank0"
            tensors[f"{prefix}.trellis"] = torch.ones(1, dtype=torch.int16)
            tensors[f"{prefix}.mcg"] = torch.ones((), dtype=torch.int32)
        for projection, field in _MODULE._LOCAL_FIELDS:
            tensors[
                f"model.layers.3.mlp.experts.{expert}.{projection}.rank0.{field}"
            ] = torch.ones(4, dtype=torch.float16)
    if retain_old_h:
        tensors["model.layers.3.mlp.experts.0.gate_proj.rank0.suh"] = torch.ones(
            8, dtype=torch.float16
        )
    save_file(tensors, checkpoint / "model-layer-003.safetensors")
    return checkpoint


def test_audit_accepts_physical_shared_rows(tmp_path: Path) -> None:
    result = _MODULE.audit_checkpoint(
        _checkpoint(tmp_path),
        layers=(3,),
        tp_size=1,
        num_experts=2,
        hidden_size=8,
        intermediate_size=4,
    )
    assert result["shared_h_physical_bytes_per_rank"] == 48
    assert result["shared_h_saved_bytes_per_rank"] == 48
    assert result["layer_results"][0]["exl3_tensor_count"] == 21


def test_audit_rejects_retained_per_expert_h_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retains per-expert tensor"):
        _MODULE.audit_checkpoint(
            _checkpoint(tmp_path, retain_old_h=True),
            layers=(3,),
            tp_size=1,
            num_experts=2,
            hidden_size=8,
            intermediate_size=4,
        )
