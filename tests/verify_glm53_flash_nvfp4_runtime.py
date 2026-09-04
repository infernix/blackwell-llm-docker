#!/usr/bin/env python3
# Extended verifier: R21 NVFP4 runtime + strict TP3 policy checks, --policy-only mode.
"""Verify the installed GLM-5.3-Flash NVFP4 runtime contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TP3_DISPATCHER = pathlib.Path("/usr/local/bin/serve-glm53-flash.sh")
TP3_LAUNCHER = pathlib.Path("/usr/local/bin/serve-glm53-flash-tp3-r21.sh")
BASE_DELEGATE = "/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh"
LOCKED_OPTIONS = (
    "--revision",
    "--speculative-config",
    "--moe-backend",
    "--model",
    "--config",
    "--enforce-eager",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--decode-context-parallel-size",
    "--cp-kv-cache-interleave-size",
    "--dcp-kv-cache-interleave-size",
    "--mamba-cache-mode",
    "--mm-encoder-tp-mode",
    "--enable-expert-parallel",
    "--disable-expert-parallel",
    "--disable-custom-all-reduce",
    "--cudagraph-capture-sizes",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--max-cudagraph-capture-size",
    "--gpu-memory-utilization",
    "--kv-cache-memory-bytes",
    "--num-gpu-blocks-override",
    "--kv-cache-dtype",
    "--additional-config",
    "--compilation-config",
    "--kv-transfer-config",
    "--kv-offloading-size",
    "--kv-offloading-backend",
)
LOCKED_ENV = (
    ("TP", "3"),
    ("DCP", "1"),
    ("MM_ENCODER_TP_MODE", "weights"),
    ("MODEL_REVISION", "378ca54585c46542bad1f3cb3ed0d73ae51cdb62"),
    ("DFLASH_MODEL_REVISION", "aea0ac8a05624512ca9e106c09c16087da998426"),
    ("LMCACHE_ENABLED", "0"),
    ("MAX_MODEL_LEN", "1048576"),
    ("MAX_NUM_SEQS", "32"),
    ("MAX_NUM_BATCHED_TOKENS", "4096"),
    ("PREFILL_SCHEDULE_INTERVAL", "1"),
    ("GPU_MEMORY_UTILIZATION", "0.93"),
    ("KV_CACHE_DTYPE", "fp8"),
    ("ATTENTION_BACKEND", "B12X"),
    ("MOE_BACKEND", "b12x"),
    ("LINEAR_BACKEND", "b12x"),
    ("MTP_ATTENTION_BACKEND", "B12X"),
    ("MTP_MOE_BACKEND", "marlin"),
    ("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x"),
    ("B12X_PCIE_ALLREDUCE", "1"),
    ("GLM53_KDA_DECODE_BACKEND", "auto"),
    ("GLM53_KDA_PREFILL_BACKEND", "flashkda"),
    ("CUDAGRAPH_MODE", "FULL_AND_PIECEWISE"),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="run only the TP3 policy checks (no imports/GPU state required)",
    )
    parser.add_argument("--vllm-version", required=not _is_policy_only(argv))
    parser.add_argument("--vllm-tree", required=not _is_policy_only(argv))
    parser.add_argument("--b12x-tree", required=not _is_policy_only(argv))
    return parser.parse_args()


def _is_policy_only(argv: list[str]) -> bool:
    return "--policy-only" in argv


def main(args: argparse.Namespace) -> None:
    assert importlib.metadata.version("vllm") == args.vllm_version
    assert importlib.metadata.version("b12x") == "1.3.0"
    assert importlib.metadata.version("flashinfer-python") == "0.6.18+cu133"
    assert importlib.metadata.version("instanttensor") == "0.1.9"
    assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.6.2"

    import b12x
    import torch
    import vllm
    from vllm.model_executor.models.registry import ModelRegistry

    assert torch.__version__.startswith("2.13.0")
    assert "Glm5NextForCausalLM" in ModelRegistry.get_supported_archs()
    assert "Glm5NextForConditionalGeneration" in ModelRegistry.get_supported_archs()

    vllm_path = pathlib.Path(vllm.__file__).resolve()
    b12x_path = pathlib.Path(b12x.__file__).resolve()
    assert vllm_path.is_relative_to("/opt/glm53-flash/vllm")
    assert b12x_path.is_relative_to("/opt/glm53-flash/b12x")
    assert pathlib.Path("/opt/glm53-flash/vllm/vllm/models/glm5next").is_dir()

    stable_ops_spec = importlib.util.find_spec("vllm._C_stable_libtorch")
    assert stable_ops_spec is not None and stable_ops_spec.origin is not None
    assert pathlib.Path(stable_ops_spec.origin).resolve(strict=True).is_file()
    if torch.cuda.is_available():
        importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm.vllm_flash_attn.layers.rotary")
    importlib.import_module("vllm.models.glm5next.nvidia.model")
    importlib.import_module("instanttensor")

    from vllm.models.deepseek_v4.nvidia import b12x_indexer
    from vllm.models.deepseek_v4.nvidia.b12x_indexer import B12xC4SparseIndexer
    from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer
    assert callable(B12xC4SparseIndexer.run_paged_topk)
    assert not hasattr(b12x_indexer, "_run_deepgemm_prefill_topk")
    assert not hasattr(Glm5NextPooledIndexer, "run_deepgemm_prefill_topk")

    assert len(args.vllm_tree) == 40
    assert len(args.b12x_tree) == 40
    for source_dir, expected_tree in (
        ("/opt/glm53-flash/vllm", args.vllm_tree),
        ("/opt/glm53-flash/b12x", args.b12x_tree),
    ):
        actual_tree = subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        assert actual_tree == expected_tree
        subprocess.run(
            ["git", "-C", source_dir, "diff", "--quiet", "HEAD", "--"],
            check=True,
        )


def verify_tp3_launcher() -> None:
    """Record strict TP3 support: dispatcher, lock list, and full policy env set.

    Structural only: the TP3 runtime is not qualified by these checks.
    """
    assert TP3_DISPATCHER.is_file()
    assert TP3_LAUNCHER.is_file()

    dispatcher_text = TP3_DISPATCHER.read_text()
    tp3_text = TP3_LAUNCHER.read_text()

    assert re.search(
        r"if \[\[ \$\{TP:-4\} == 3 && -z \$\{CACHE_MODE:-\} "
        r"&& \$\{LMCACHE_ENABLED:-0\} == 0 \]\]; then\n"
        r"  exec /usr/local/bin/serve-glm53-flash-tp3-r21\.sh \"\$@\"\nfi",
        dispatcher_text,
    )
    assert "CACHE_MODE must be vram, native, or lmcache" in dispatcher_text

    for name, value in LOCKED_ENV:
        if name in ("MODEL_REVISION", "DFLASH_MODEL_REVISION"):
            pattern = rf'lock_env {re.escape(name)} "\$\{{locked_\w+\}}"'
        else:
            pattern = rf"lock_env {re.escape(name)} {re.escape(value)}\b"
        assert re.search(pattern, tp3_text), f"missing lock: {name}"
    assert "local-inference-lab/GLM-5.3-Flash-NVFP4" in tp3_text
    assert "local-inference-lab/GLM-5.3-Flash-DFlash2" in tp3_text

    assert "locked_long_options" in tp3_text
    assert "is_locked_override" in tp3_text
    assert "rejects caller override" in tp3_text
    assert "split-page block-size variables" in tp3_text


def run_tp3_policy_cases() -> None:
    """Fail-closed behavioral cases: rejected inputs exit 2, accepted TP=3
    env set reaches the stubbed base launcher appended with the lock flags."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="glm53-tp3-policy-"))
    try:
        base_stub = tmp / "base-dflash2"
        base_stub.write_text(
            '#!/usr/bin/env bash\nprintf "STUB-REACHED:"\nprintf " %q" "$@"\nprintf "\\n"\nexit 0\n'
        )
        base_stub.chmod(0o755)
        sandbox_tp3 = tmp / "tp3.sh"
        sandbox_tp3.write_text(
            TP3_LAUNCHER.read_text().replace(BASE_DELEGATE, str(base_stub))
        )
        sandbox_tp3.chmod(0o755)
        sandbox_dispatcher = tmp / "dispatcher.sh"
        sandbox_dispatcher.write_text(
            TP3_DISPATCHER.read_text().replace(
                "/usr/local/bin/serve-glm53-flash-tp3-r21.sh", str(sandbox_tp3)
            )
        )
        sandbox_dispatcher.chmod(0o755)

        def output_env(**over: str) -> dict[str, str]:
            env = dict(os.environ)
            for key in (
                "CACHE_MODE",
                "LMCACHE_ENABLED",
                "DCP",
                "MM_ENCODER_TP_MODE",
                "MAX_NUM_SEQS",
                "MAX_MODEL_LEN",
                "MAX_NUM_BATCHED_TOKENS",
                "PREFILL_SCHEDULE_INTERVAL",
                "GPU_MEMORY_UTILIZATION",
                "KV_CACHE_DTYPE",
                "ATTENTION_BACKEND",
                "MOE_BACKEND",
                "LINEAR_BACKEND",
                "MTP_ATTENTION_BACKEND",
                "MTP_MOE_BACKEND",
                "VLLM_PCIE_ALLREDUCE_BACKEND",
                "B12X_PCIE_ALLREDUCE",
                "GLM53_KDA_DECODE_BACKEND",
                "GLM53_KDA_PREFILL_BACKEND",
                "CUDAGRAPH_MODE",
                "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE",
                "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE",
            ):
                env.pop(key, None)
            env.update(over)
            env.setdefault("TP", "3")
            return env

        def reject(desc: str, env: dict[str, str], args: list[str]) -> None:
            result = subprocess.run(
                ["bash", str(sandbox_tp3), *args], env=env,
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 2, (
                f"TP3 policy accepted disallowed input: {desc}; "
                f"rc={result.returncode} stderr={result.stderr}"
            )

        def accept(desc: str, env: dict[str, str], args: list[str]) -> subprocess.CompletedProcess:
            result = subprocess.run(
                ["bash", str(sandbox_tp3), *args], env=env,
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"TP3 policy rejected compliant input: {desc}; "
                f"rc={result.returncode} stderr={result.stderr}"
            )
            return result
        # Every locked long option (and each short alias) rejected as override.
        for option in LOCKED_OPTIONS:
            reject(f"locked override {option}", output_env(), [option, "1"])
            reject(f"locked override {option}=1", output_env(), [f"{option}=1"])
        for alias_args in (
            ["-tp", "3"],
            ["-dcp", "2"],
            ["-pp", "2"],
            ["-sc", '{"method":"mtp"}'],
            ["-cc", "{}"],
            ["-ep"],
        ):
            reject(f"locked alias {alias_args}", output_env(), alias_args)
        reject("positional model override", output_env(), ["some-model"])
        # Environment overrides are rejective at policy level. CACHE_MODE is
        # proven separately against the dispatcher below.
        reject("LMCACHE_ENABLED=1", output_env(LMCACHE_ENABLED="1"), [])
        reject("DCP=2", output_env(DCP="2"), [])
        for key, locked_value in (f for f in LOCKED_ENV if f[0] != "TP"):
            reject(
                f"env override {key}",
                output_env(**{key: "999999" if locked_value.isdigit() else "bogus"}),
                [],
            )
        reject("split target block size", output_env(VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE="2048"), [])
        reject("split mamba block size", output_env(VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE="2048"), [])
        reject("TP=4 fails closed in TP3 script", output_env(TP="4"), [])
        reject("TP=8 fails closed in TP3 script", output_env(TP="8"), [])
        reject("TP unset fails closed", output_env(TP=""), [])

        # Accepting paths:
        # 1) all environment at exact policy values
        full_env = output_env(**dict(LOCKED_ENV))
        result = accept("full policy env value set", full_env, [])
        assert "STUB-REACHED" in result.stdout, result.stdout
        # appended lock flags present
        for flag in ("--enable-expert-parallel", "--mm-encoder-tp-mode", "weights"):
            assert flag in result.stdout
        # 2) clean environment defaults to policy values
        result = accept("clean policy default", output_env(), [])
        assert "STUB-REACHED" in result.stdout

        # Dispatcher routing: clean TP=3 goes into the rewritten policy
        # launcher; CACHE_MODE=lmcache/native must not reach it.
        clean_dispatcher_run = subprocess.run(
            ["bash", str(sandbox_dispatcher)],
            env=output_env(),
            capture_output=True, text=True, timeout=30,
        )
        assert "STUB-REACHED" in clean_dispatcher_run.stdout, clean_dispatcher_run.stderr
        for cache_mode in ("native", "lmcache"):
            result = subprocess.run(
                ["bash", str(sandbox_dispatcher)],
                env=output_env(TP="3", CACHE_MODE=cache_mode),
                capture_output=True, text=True, timeout=30,
            )
            assert "STUB-REACHED" not in result.stdout, (
                f"CACHE_MODE={cache_mode} reached the TP3 policy launcher"
            )
        tp4_run = subprocess.run(
            ["bash", str(sandbox_dispatcher)],
            env=output_env(TP="4"),
            capture_output=True, text=True, timeout=30,
        )
        assert "STUB-REACHED" not in tp4_run.stdout, "TP=4 entered TP3 policy"
        tp8_run = subprocess.run(
            ["bash", str(sandbox_dispatcher)],
            env=output_env(TP="8"),
            capture_output=True, text=True, timeout=30,
        )
        assert "STUB-REACHED" not in tp8_run.stdout, "TP=8 entered TP3 policy"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if _is_policy_only(argv):
        args = parse_args(argv + ["--vllm-version", "x", "--vllm-tree", "0", "--b12x-tree", "0"])
        verify_tp3_launcher()
        run_tp3_policy_cases()
        print("TP3 policy: recorded (structural), fail-closed matrix: PASS")
    else:
        args = parse_args(argv)
        main(args)
