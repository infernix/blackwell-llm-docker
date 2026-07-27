from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

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


def test_validate_probe_result_rejects_unsafe_or_inconsistent_policy() -> None:
    result = _result()
    result["policy"]["compressed_dma_requires_explicit_opt_in"] = False
    with pytest.raises(ValueError, match="compressed DMA"):
        calibration.validate_probe_result(result)

    result = _result()
    result["policy"]["dcp_query_split_min_context_tokens"] = 0
    with pytest.raises(ValueError, match="inconsistent query-split"):
        calibration.validate_probe_result(result)

    result = _result()
    result["policy"]["dcp_query_split"] = 0
    with pytest.raises(ValueError, match="inconsistent query-split"):
        calibration.validate_probe_result(result)


@pytest.mark.parametrize(
    ("value", "expected"),
    (("4", 4), ("N/A", None), ("[N/A]", None)),
)
def test_optional_nvidia_int(value: str, expected: int | None) -> None:
    assert calibration._optional_nvidia_int(value) == expected


def _probe_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tp_size=2,
        dcp_size=1,
        indexer_shards=1,
        hidden_size=6144,
        tp_rows=8192,
        ckv_record_bytes=656,
        context_tokens=(8192,),
        allreduce_rows=(1,),
        gpus=(0, 1),
        timeout=5.0,
        cache_dir=tmp_path,
    )


def test_run_probe_reports_timeout_with_captured_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("probe", 5.0, output="probe stalled")

    monkeypatch.setattr(calibration.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match=r"(?s)timed out.*probe stalled"):
        calibration._run_probe(_probe_args(tmp_path), tmp_path / "result.json")


def test_run_probe_reports_invalid_result_with_probe_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="probe complete\n"),
    )
    output = tmp_path / "result.json"
    output.write_text("not JSON", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"(?s)no valid result.*probe complete"):
        calibration._run_probe(_probe_args(tmp_path), output)


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
