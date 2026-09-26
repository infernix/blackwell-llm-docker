"""CPU behavioral gate using real CPU tensors, not a CUDA correctness claim.
uv run --no-project --with torch --index-url https://download.pytorch.org/whl/cpu python docs/ds41-tp3-scale-codec/residency_gate/bank_gate.py
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[3]
VLLM_SOURCE = Path(os.environ.get("VLLM_SOURCE_DIR", ROOT.parent / "vllm"))
source = VLLM_SOURCE / "vllm/model_executor/layers/fused_moe/expert_residency.py"
spec = importlib.util.spec_from_file_location("residency", source)
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)

class Gate(unittest.TestCase):
    def make(self):
        prepared = dict(zip(r.BANK_SLOTS, (torch.arange(64).reshape(4, 16), torch.arange(32).reshape(4, 8), torch.zeros(8, dtype=torch.uint32), torch.zeros(8, dtype=torch.uint32))))
        owners = [SimpleNamespace(va=prepared[n].data_ptr(), hot_bytes=0, operand_role='weight') for n in r.BANK_SLOTS[:2]]
        return r.DeferredWeightBanks(prepared, num_experts=4, device='cpu', owners=owners)
    @patch('torch.cuda.synchronize')
    def test_cache_and_eviction(self, sync):
        b = self.make()
        addresses = (b.w13_bank.data_ptr(), b.down_bank.data_ptr())
        scales = tuple(s.clone() for s in b.scales)
        self.assertEqual(b.hot_slabs, ())
        size = b.expert_bytes
        self.assertEqual(b.promote_up_to(size * 2 - 1), size)
        for slab, source, stride in zip(b.hot_slabs, b.sources, b.slot_stride_bytes):
            self.assertTrue(torch.equal(slab, source.view(torch.uint8).reshape(-1)[:stride]))
        self.assertEqual(b.replace_hot((3, 1), budget_bytes=size*2), size*2)
        self.assertEqual(b.hot_ids, (1, 3))
        for table, cold in zip((b.w13_bank, b.down_bank), b._cold_tables):
            self.assertTrue(torch.equal(table[[0, 2]], cold[[0, 2]]))
        self.assertEqual(b.evict_all(), size*2)
        self.assertEqual(addresses, (b.w13_bank.data_ptr(), b.down_bank.data_ptr()))
        for table, cold in zip((b.w13_bank, b.down_bank), b._cold_tables):
            self.assertTrue(torch.equal(table, cold))
        self.assertTrue(all(torch.equal(a, z) for a, z in zip(scales, b.scales)))
        self.assertEqual(b.hot_slabs, ())
    @patch('torch.cuda.synchronize')
    def test_publication_rollback(self, sync):
        b = self.make()
        original = torch.Tensor.copy_
        calls = [0]
        def fail_second_table(target, source, *args, **kwargs):
            if target.data_ptr() in (b.w13_bank.data_ptr(), b.down_bank.data_ptr()):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('injected table failure')
            return original(target, source, *args, **kwargs)
        with patch.object(torch.Tensor, 'copy_', fail_second_table):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                b.promote_up_to(b.expert_bytes)
        self.assertEqual(b.hot_ids, ())
        for table, cold in zip((b.w13_bank, b.down_bank), b._cold_tables):
            self.assertTrue(torch.equal(table, cold))
    def test_rejected_budget(self):
        b = self.make()
        with self.assertRaises(ValueError):
            b.replace_hot((0,), budget_bytes=b.expert_bytes-1)
        with self.assertRaises(ValueError):
            b.replace_hot((0, 0), budget_bytes=b.expert_bytes*2)

if __name__ == '__main__':
    unittest.main()
