"""TP3 profile defaults and caller overrides for DeepSeek V4.1 Flash."""

import json
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "serve-ds41-tp3.sh"


def launch(tmp_path, overrides=None, arguments=()):
    native = tmp_path / "serve-ds41-flash.sh"
    native.write_text(
        '#!/bin/bash\nexec "$PYTHON_BIN" -c '
        "'import json,os,sys; print(json.dumps({"
        '"args":sys.argv[1:],"env":dict(os.environ)}))' + '\' "$@"\n'
    )
    native.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": sys.executable,
        "VLLM_SOURCE_DIR": str(tmp_path),
        **(overrides or {}),
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def value_of(args, option):
    return args[args.index(option) + 1]


def test_tp3_defaults_follow_the_qualified_operating_point(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    args, env = payload["args"], payload["env"]
    assert value_of(args, "--cache-mode") == "lmcache"
    assert value_of(args, "--cache-transfer-mode") == "engine_driven"
    assert value_of(args, "--cache-l1-gib") == "48"
    assert value_of(args, "--cache-l2-enabled") == "false"
    assert value_of(args, "--kv-cache-memory-bytes") == "536870912"
    assert value_of(args, "--prefill-policy") == "round-robin"
    assert value_of(args, "--jit-monitor-mode") == "warn"
    assert value_of(args, "--scheduling-policy") == "priority"
    dash = args.index("--")
    assert args[dash:] == [
        "--", "--cpu-offload-gb", "8",
        "--cpu-offload-params", "experts", "vision", "aligner",
    ]
    assert env["TP_SIZE"] == "3"
    assert env["MAX_NUM_SEQS"] == "4"
    assert env["MAX_MODEL_LEN"] == "524288"
    assert env["MAX_NUM_BATCHED_TOKENS"] == "1024"
    assert env["GPU_MEMORY_UTILIZATION"] == "0.93"
    assert env["MODE"] == "off"
    assert env["ENGRAM_TABLE_MEMORY"] == "disk"
    assert env["B12X_SCALE_CODEC"] == "1"
    assert env["B12X_DEFERRED_WEIGHT_BANKS"] == "1"
    assert env["DS41_VMM_ENABLE"] == "1"
    assert env["DS41_OFFLOAD_FROM_ORDINAL"] == "20"
    assert env["DS41_OFFLOAD_TO_ORDINAL"] == "22"


def test_caller_cache_selection_replaces_the_profile_default(tmp_path):
    result = launch(tmp_path, arguments=("--cache-mode", "vram"))
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)["args"]
    assert args.count("--cache-mode") == 1
    assert value_of(args, "--cache-mode") == "vram"
    # Untouched defaults remain so the L1 size and KV cap stay coherent.
    assert "--cache-l1-gib" in args
    assert "--kv-cache-memory-bytes" in args


def test_caller_scheduling_selection_replaces_priority_default(tmp_path):
    result = launch(tmp_path, arguments=("--scheduling-policy=fcfs"))
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)["args"]
    assert args.count("--scheduling-policy") == 1
    assert "--scheduling-policy" in args or "--scheduling-policy=fcfs" in args


def test_engram_storage_validation(tmp_path):
    bad = launch(tmp_path, {"ENGRAM_TABLE_MEMORY": "ssd"})
    assert bad.returncode == 2
    assert "ram or disk" in bad.stderr
    for memory in ("ram", "disk"):
        result = launch(tmp_path, {"ENGRAM_TABLE_MEMORY": memory})
        assert result.returncode == 0, result.stderr
        env = json.loads(result.stdout)["env"]
        assert env["ENGRAM_TABLE_MEMORY"] == memory
