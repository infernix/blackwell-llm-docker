"""Run only in an isolated approved GPU window, never the live container.
CUDA_VISIBLE_DEVICES=0 python3 docs/ds41-tp3-scale-codec/residency_gate/vmm_gpu_gate.py
compute-sanitizer --tool memcheck python3 docs/ds41-tp3-scale-codec/residency_gate/vmm_gpu_gate.py
"""
import importlib.util
import os
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[3]
vllm_source = Path(os.environ.get("VLLM_SOURCE_DIR", root.parent / "vllm"))
source = vllm_source / "vllm/model_executor/offloader/vmm_shim.py"
spec = importlib.util.spec_from_file_location("vmm", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with torch.cuda.device(0):
    torch.cuda.init()
    a = m.MixedResidencyAllocation(2 * 1024 * 1024, 0, set())
    t = a.wrap_torch(torch, a.size, torch.uint8)
    t.fill_(73)
    torch.cuda.synchronize()
    address = t.data_ptr()
    out = torch.empty_like(t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(t)
    assert a.promote_up_to(a.size - 1) == 0
    assert a.promote_up_to(a.size) == a.size
    assert t.data_ptr() == address
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == 73).all())
    assert a.evict_promoted() == a.size
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == 73).all())
    del graph, t, out
    a.free()
    a.free()
print('PASS: same VA, promotion, retained-host eviction, captured graph replay')
