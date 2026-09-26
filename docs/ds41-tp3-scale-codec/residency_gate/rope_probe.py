"""DeepSeek V4 RoPE context tiers, exact prefix, and dtype-error probe."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope as baseline_build

ROOT = Path(__file__).resolve().parents[3]
VLLM_SOURCE = Path(os.environ.get("VLLM_SOURCE_DIR", ROOT.parent / "vllm"))
source = VLLM_SOURCE / "vllm/models/deepseek_v4/common/rope.py"
spec = importlib.util.spec_from_file_location("capped_rope", source)
capped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capped)

config = SimpleNamespace(
    rope_parameters={
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    },
    compress_rope_theta=160000,
    rope_theta=10000,
)

results = []
ctx = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu")))
ctx.__enter__()

for ratio in (0, 2):
    _ROPE_DICT.clear()
    base = baseline_build(
        config,
        head_dim=512,
        rope_head_dim=64,
        max_position_embeddings=1048576,
        compress_ratio=ratio,
    )
    source = base.cos_sin_cache
    for rows in (1048576, 786432, 524288):
        os.environ["DS41_ROPE_CACHE_ROWS"] = str(rows)
        _ROPE_DICT.clear()
        target = capped.build_deepseek_v4_rope(
            config,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=1048576,
            compress_ratio=ratio,
        ).cos_sin_cache
        assert target.shape == (rows, 64)
        assert torch.equal(target, source[:rows])
        sample_ids = torch.tensor([0, rows // 2, rows - 1])
        fp32 = target[sample_ids]
        bf16 = fp32.to(torch.bfloat16).float()
        fp8 = fp32.to(torch.float8_e4m3fn).float()
        results.append({
            "compress_ratio": ratio,
            "rows": rows,
            "fp32_bytes": target.nbytes,
            "sha256": hashlib.sha256(target.numpy().tobytes()).hexdigest(),
            "exact_prefix": True,
            "bf16_max_abs": (fp32 - bf16).abs().max().item(),
            "bf16_mean_abs": (fp32 - bf16).abs().mean().item(),
            "fp8_max_abs": (fp32 - fp8).abs().max().item(),
            "fp8_mean_abs": (fp32 - fp8).abs().mean().item(),
        })
ctx.__exit__(None, None, None)
print(json.dumps({"torch": torch.__version__, "results": results}, indent=2))
