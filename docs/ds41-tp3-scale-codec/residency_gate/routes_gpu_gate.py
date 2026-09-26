"""Focused CUDA route accounting gate; run in the inference image on a spare GPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
VLLM_SOURCE = Path(os.environ.get("VLLM_SOURCE_DIR", ROOT.parent / "vllm"))
source = VLLM_SOURCE / "vllm/model_executor/layers/fused_moe/expert_residency.py"
spec = importlib.util.spec_from_file_location("residency", source)
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
os.environ["B12X_EXPERT_ROUTES"] = "1"
os.environ["B12X_EXPERT_ROUTES_INTERVAL"] = "0.1"
torch.cuda.set_device(0)
prepared = dict(zip(r.BANK_SLOTS, (
    torch.arange(384 * 16, device="cuda").reshape(384, 16),
    torch.arange(384 * 8, device="cuda").reshape(384, 8),
    torch.zeros(384, dtype=torch.uint32, device="cuda"),
    torch.zeros(384, dtype=torch.uint32, device="cuda"))))
owners = [SimpleNamespace(va=prepared[n].data_ptr(), hot_bytes=0,
                          operand_role="weight", layer_ordinal=22)
          for n in r.BANK_SLOTS[:2]]
b = r.DeferredWeightBanks(prepared, num_experts=384, device="cuda:0", owners=owners)
expected = torch.zeros(384, dtype=torch.int64)
ids32 = torch.tensor([0, 2, 2, 383, 2, 0, 99], device="cuda", dtype=torch.int32)
ids64 = ids32.to(torch.int64)
with tempfile.TemporaryDirectory() as directory:
    b.record_routes(ids32)
    b.record_routes(ids64)
    torch.cuda.synchronize()
    assert b.dump_routes(directory)["counts"] == expected.tolist()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        b.record_routes(ids32)
        b.record_routes(ids64)
    graph.replay()
    torch.cuda.synchronize()
    assert b.dump_routes(directory)["total_routes"] == 0
    # Redirect only the daemon output, so the real background loop is exercised.
    original_dump = b.dump_routes
    b.dump_routes = lambda: original_dump(directory)
    b.replace_hot((0, 383), budget_bytes=b.expert_bytes * 2)
    b.enable_routes()
    b.record_routes(ids32)
    b.record_routes(ids64)
    expected.scatter_add_(0, ids64.cpu(), torch.ones(7, dtype=torch.int64) * 2)
    for _ in range(5):
        graph.replay()
    expected.scatter_add_(0, ids64.cpu(), torch.ones(7, dtype=torch.int64) * 10)
    torch.cuda.synchronize()
    snapshot = b.dump_routes()
    assert snapshot["counts"] == expected.tolist()
    assert snapshot["cold_routes"] == int(expected.sum() - expected[[0, 383]].sum())
    b.replace_hot((2, 99), budget_bytes=b.expert_bytes * 2)
    snapshot = b.dump_routes()
    assert snapshot["counts"] == expected.tolist()
    assert snapshot["cold_routes"] == 36
    assert snapshot["hot_hit_ratio"] == 48 / 84
    assert snapshot["cold_hit_ratio"] == 36 / 84
    assert snapshot["hot_ids"] == [2, 99]
    assert set(snapshot) == {"rank", "layer", "total_routes", "cold_routes", "hot_hit_ratio", "cold_hit_ratio", "hot_ids", "counts", "timestamp"}
    path = Path(directory) / "expert-routes-rank0-layer22.json"
    prior = snapshot["timestamp"]
    deadline = time.monotonic() + 5
    while json.loads(path.read_text())["timestamp"] <= prior:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    observed = json.loads(path.read_text())
    assert observed["counts"] == expected.tolist() and observed["layer"] == 22
    allocated = torch.cuda.memory_allocated()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    monitor = r.ExpertRouteMonitor(layer_ordinal=7, num_experts=384, device="cuda:0")
    monitor_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(monitor_graph):
        monitor.record_routes(ids32)
    monitor_graph.replay()
    torch.cuda.synchronize()
    assert monitor.dump_routes(directory)["total_routes"] == 0
    monitor.enable()
    monitor.record_routes(ids32)
    for _ in range(3):
        monitor_graph.replay()
    torch.cuda.synchronize()
    monitored = monitor.dump_routes(directory)
    assert monitored["total_routes"] == 28
    assert monitored["counts"][2] == 12
    assert monitored["layer"] == 7
    assert set(monitored) == {"rank", "layer", "total_routes", "counts", "timestamp"}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):
        graph.replay()
    end.record()
    end.synchronize()
    print(json.dumps({"result": "PASS", "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(), "cc": torch.cuda.get_device_capability(),
                      "two_kernel_replay_us": start.elapsed_time(end),
                      "counters": 384, "exact_routes_before_timing": 84,
                      "daemon_snapshot": True, "steady_gpu_allocation_delta": 0}))
    # Process exit ends the daemon; no later snapshots should target the removed dir.
    os._exit(0)
