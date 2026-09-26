"""Extend the existing real-checkpoint scratchless gate with deferred WS4.
Inside the isolated assembled image with this repository at /repo:
python /repo/docs/ds41-tp3-scale-codec/residency_gate/ws4_gpu_gate.py --experts 4 --tokens 1 16 --deterministic --report /tmp/ws4.json
Repeat under compute-sanitizer --tool memcheck. Never run in the live container.
"""
import dataclasses
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import torch

root = Path(__file__).resolve().parents[3]
vllm_source = Path(os.environ.get("VLLM_SOURCE_DIR", root.parent / "vllm"))
work = root / "docs/ds41-tp3-scale-codec"
sys.path.insert(0, str(work / "scratchless_gate"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
vmm = load("gate_vmm", vllm_source / "vllm/model_executor/offloader/vmm_shim.py")
r = load("gate_residency", vllm_source / "vllm/model_executor/layers/fused_moe/expert_residency.py")
from b12x.moe.fused_moe import _impl as native
original_build = native.build_tp_moe_fp4_binding
original_op = native.b12x_moe_fp4
caches = {}

def build(*, experts, **kwargs):
    payload = experts.representation_for('w4a8_mx')
    if payload.w13_sfb.dtype != torch.uint32:
        return original_build(experts=experts, **kwargs)
    allocations, weights = [], []
    for source in (payload.w13_rp, payload.w2_rp):
        alloc = vmm.MixedResidencyAllocation(source.nbytes, 0, set())
        alloc.operand_role = 'weight'
        target = alloc.wrap_torch(torch, source.numel(), source.dtype).view(source.shape)
        target.copy_(source)
        allocations.append(alloc)
        weights.append(target)
    torch.cuda.synchronize()
    payload = dataclasses.replace(payload, w13_rp=weights[0], w2_rp=weights[1])
    experts = dataclasses.replace(experts, w1_fp4=weights[0], w2_fp4=weights[1], representation=dataclasses.replace(experts.representation, value=payload))
    banks = r.DeferredWeightBanks(dict(zip(r.BANK_SLOTS, (*weights, payload.w13_sfb, payload.w2_sfb))), num_experts=experts.num_experts, device='cuda:0', owners=allocations)
    native.attach_expert_banks(experts, banks)
    binding = original_build(experts=experts, **kwargs)
    caches[id(binding)] = [banks, 0]
    return binding

def op(*, binding):
    entry = caches.get(id(binding))
    if entry is not None and not torch.cuda.is_current_stream_capturing():
        banks, count = entry
        # Reference gate interleaves 64 native samples before graph capture.
        # Exercise cold, hot-prefix, non-prefix replacement, and eviction.
        if count == 1:
            banks.promote_up_to(banks.expert_bytes * 2)
        elif count == 16:
            banks.replace_hot((banks.num_experts - 1,), budget_bytes=banks.expert_bytes)
        elif count == 32:
            banks.evict_all()
        entry[1] += 1
    return original_op(binding=binding)

native.build_tp_moe_fp4_binding = build
native.b12x_moe_fp4 = op
runpy.run_path(str(work / "scratchless_gate/gate.py"), run_name="__main__")
