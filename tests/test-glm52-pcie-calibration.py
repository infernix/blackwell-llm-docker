from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "launchers" / "glm52-pcie-calibration.py"
SPEC = importlib.util.spec_from_file_location("glm52_pcie_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def _result(*, wire_mode: str = "bf16") -> dict:
    return {
        "policy": {
            "numeric_contract": "lossless-only",
            "compressed_dma_requires_explicit_opt_in": True,
            "dcp_ckv_prefetch_depth": 1,
            "dcp_query_split": 1,
            "dcp_query_split_min_context_tokens": 8192,
            "tp_allreduce": {
                "dma_wire_mode": wire_mode,
                "dma_min_bytes": 24 * 1024 * 1024,
            },
        }
    }


def _result_without_dma_crossover() -> dict:
    result = _result()
    result["policy"]["tp_allreduce"]["dma_min_bytes"] = 0
    return result


def test_validate_probe_result_accepts_only_lossless_policy() -> None:
    assert calibration.validate_probe_result(_result()) == {
        "prefetch_depth": 1,
        "query_split": 1,
        "query_split_min_context_tokens": 8192,
        "dma_min_bytes": 24 * 1024 * 1024,
    }

    with pytest.raises(ValueError, match="BF16 DMA"):
        calibration.validate_probe_result(_result(wire_mode="fp8-ring"))

    assert (
        calibration.validate_probe_result(_result_without_dma_crossover())[
            "dma_min_bytes"
        ]
        == "off"
    )


def test_fingerprint_is_order_sensitive() -> None:
    first = {"gpu_order": [{"uuid": "a"}, {"uuid": "b"}]}
    second = {"gpu_order": [{"uuid": "b"}, {"uuid": "a"}]}

    assert calibration.fingerprint(first) != calibration.fingerprint(second)


def test_collective_environment_tracks_only_relevant_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "8")
    monkeypatch.setenv("SPARKINFER_PCIE_DMA_PIECES", "2")
    monkeypatch.setenv("UNRELATED_VALUE", "ignored")

    environment = calibration._collective_environment()

    assert environment["NCCL_MIN_NCHANNELS"] == "8"
    assert environment["SPARKINFER_PCIE_DMA_PIECES"] == "2"
    assert "UNRELATED_VALUE" not in environment


def test_cache_rejects_wrong_fingerprint_and_numeric_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    record = {
        "schema": calibration.SCHEMA_VERSION,
        "fingerprint": "expected",
        "probe": _result(),
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    assert calibration._load_record(path, "expected") == record
    assert calibration._load_record(path, "different") is None

    record["probe"]["policy"]["numeric_contract"] = "compressed"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert calibration._load_record(path, "expected") is None
