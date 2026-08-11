#!/usr/bin/env python3
"""Verify the Kimi-K3 HH runtime without loading model weights."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path
import time

import torch
from packaging.version import Version


FORBIDDEN_CUDA_WHEEL_PREFIXES = (
    "nvidia-cublas-",
    "nvidia-cuda-cupti-",
    "nvidia-cuda-nvrtc-",
    "nvidia-cuda-runtime-",
    "nvidia-cudnn-",
    "nvidia-cufft-",
    "nvidia-curand-",
    "nvidia-cusolver-",
    "nvidia-cusparse-",
    "nvidia-nccl-",
    "nvidia-nvjitlink-",
)

# nvidia-cudnn-frontend contains Python bindings and headers. It does not
# replace the cuDNN shared libraries supplied by the NVIDIA base image.
ALLOWED_CUDA_TOOLING_WHEELS = {"nvidia-cudnn-frontend"}


def _mapped_libraries(fragment: str) -> set[Path]:
    libraries: set[Path] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rsplit(maxsplit=1)[-1]
            if fragment in path and path.startswith("/"):
                libraries.add(Path(path))
    return libraries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instanttensor-version", required=True)
    parser.add_argument("--cutlass-dsl-version", required=True)
    parser.add_argument("--flashinfer-version", required=True)
    parser.add_argument("--torchvision-version", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--require-cuda-platform", action="store_true")
    parser.add_argument("--require-flashinfer-sampler", action="store_true")
    args = parser.parse_args()

    if args.require_flashinfer_sampler and not args.require_cuda_platform:
        parser.error("--require-flashinfer-sampler requires --require-cuda-platform")

    if args.require_cuda_platform:
        assert not torch.cuda.is_initialized()
        import vllm.v1.engine.async_llm  # noqa: F401

        assert not torch.cuda.is_initialized(), (
            "vLLM worker preload imports initialized CUDA; "
            "the forkserver cannot create CUDA workers safely"
        )

    import cutlass.cute as cute
    import flashinfer
    import instanttensor
    import torchvision
    import triton_kernels.matmul_ogs as matmul_ogs
    import transformers
    import xgrammar
    from b12x.attention import dense_mla
    from b12x.comm.pcie.pcie_dcp_a2a import SUPPORTED_WORLD_SIZES
    from vllm.model_executor.layers.activation import ensure_kimi_k3_activation_ops
    from vllm.models.kimi_k3.nvidia.kda import ensure_fused_kda_decode_op
    from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
        ensure_kimi_k3_cache_ops,
    )

    assert torch.__version__ == "2.13.0"
    assert torch.version.cuda == "13.3"
    assert torch.cuda.nccl.version() == (2, 31, 2)

    nccl_path = Path(os.environ["NCCL_LOCAL_INFERENCE_PATH"]).resolve(strict=True)
    nccl = ctypes.CDLL(str(nccl_path))
    version = ctypes.c_int()
    assert nccl.ncclGetVersion(ctypes.byref(version)) == 0
    assert version.value == 23102
    assert any(
        path.exists() and os.path.samefile(path, nccl_path)
        for path in _mapped_libraries("libnccl")
    )

    assert metadata.version("instanttensor") == args.instanttensor_version
    assert metadata.version("flashinfer-python") == args.flashinfer_version
    assert metadata.version("flashinfer-cubin") == args.flashinfer_version
    assert metadata.version("vllm") == args.vllm_version
    assert metadata.version("torchvision") == args.torchvision_version
    try:
        metadata.version("torchao")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("TorchAO must not be installed in the Kimi-K3 runtime")
    assert flashinfer.__file__
    assert instanttensor.__file__
    assert torchvision.__version__ == args.torchvision_version
    assert torchvision.ops.nms(
        torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([1.0]), 0.5
    ).tolist() == [0]
    assert metadata.version("nvidia-cutlass-dsl") == args.cutlass_dsl_version
    assert metadata.version("nvidia-cutlass-dsl-libs-base") == args.cutlass_dsl_version
    assert metadata.version("nvidia-cutlass-dsl-libs-cu13") == args.cutlass_dsl_version
    assert hasattr(cute.nvgpu.warp, "MmaMXF8Op")
    assert Version(metadata.version("xgrammar")) >= Version("0.2.5")
    assert Version(metadata.version("transformers")) >= Version("5.5.3")
    triton_kernels_root = Path(
        "/opt/kimi-k3-hh/vllm/vllm/third_party/triton_kernels"
    ).resolve(strict=True)
    assert Path(matmul_ogs.__file__).resolve(strict=True).is_relative_to(
        triton_kernels_root
    )
    assert xgrammar.__file__
    assert transformers.__file__

    assert dense_mla.__file__.startswith("/opt/kimi-k3-hh/b12x/")
    assert SUPPORTED_WORLD_SIZES == (2, 4, 8, 16)
    assert all(hasattr(dense_mla, name) for name in ("Caps", "plan", "bind", "compile", "run"))
    ensure_kimi_k3_cache_ops()
    assert ensure_fused_kda_decode_op()
    assert ensure_kimi_k3_activation_ops()

    if args.require_cuda_platform:
        from vllm.platforms import current_platform
        from vllm.triton_utils.importing import HAS_TRITON

        assert current_platform.is_cuda()
        assert HAS_TRITON

    if args.require_flashinfer_sampler:
        from vllm.v1.sample.ops.topk_topp_sampler import (
            TopKTopPSampler,
            flashinfer_sampler_supported,
        )

        assert flashinfer_sampler_supported()
        logits = torch.randn((1, 163840), dtype=torch.float32, device="cuda")
        top_k = torch.tensor([50], dtype=torch.int32, device="cuda")
        top_p = torch.tensor([0.95], dtype=torch.float32, device="cuda")
        sampler = TopKTopPSampler()
        assert sampler.forward.__name__ == "forward_cuda"

        torch.cuda.synchronize()
        first_start = time.perf_counter()
        token_ids, returned_logits = sampler(logits.clone(), {}, top_k, top_p)
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - first_start

        steady_start = time.perf_counter()
        for _ in range(100):
            token_ids, returned_logits = sampler(logits.clone(), {}, top_k, top_p)
        torch.cuda.synchronize()
        steady_milliseconds = (time.perf_counter() - steady_start) * 10

        assert token_ids.shape == (1,)
        assert returned_logits is None
        print(
            "FlashInfer sampler contract: PASS "
            f"first={first_seconds:.3f}s steady={steady_milliseconds:.3f}ms "
            f"token={token_ids.item()}"
        )

    installed = {
        distribution.metadata["Name"].lower()
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }
    forbidden = sorted(
        package
        for package in installed
        if package.startswith(FORBIDDEN_CUDA_WHEEL_PREFIXES)
        and package not in ALLOWED_CUDA_TOOLING_WHEELS
    )
    assert not forbidden, f"pip CUDA runtime overlays are installed: {forbidden}"

    print(
        "Kimi-K3 HH runtime contract: PASS "
        f"torch={torch.__version__} cuda={torch.version.cuda} nccl={version.value} "
        f"instanttensor={metadata.version('instanttensor')} "
        f"flashinfer={metadata.version('flashinfer-python')} "
        f"torchvision={metadata.version('torchvision')} "
        f"cutlass-dsl={metadata.version('nvidia-cutlass-dsl')} "
        f"xgrammar={metadata.version('xgrammar')}"
    )


if __name__ == "__main__":
    main()
