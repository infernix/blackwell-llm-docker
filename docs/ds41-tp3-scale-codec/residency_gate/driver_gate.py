"""CPU driver-model gate; does not establish CUDA driver/kernel correctness.
Run: python3 docs/ds41-tp3-scale-codec/residency_gate/driver_gate.py
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
VLLM_SOURCE = Path(os.environ.get("VLLM_SOURCE_DIR", ROOT.parent / "vllm"))
PATH = VLLM_SOURCE / "vllm/model_executor/offloader/vmm_shim.py"
spec = importlib.util.spec_from_file_location("vmm_gate", PATH)
vmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vmm)

class Struct(ctypes.Structure):
    def __init__(self):
        self.loc = self.acc_loc = type('Loc', (), {})()

class Driver:
    _Prop = _Acc = Struct
    def __init__(self):
        self.cuda = self
        self.maps = {4096: (8, 1)}
        self.handles = {1}
        self.fail = None
        self.copies = []
    def check(self, rc, what):
        if rc:
            raise RuntimeError(what)
    def cuCtxSynchronize(self):
        return 0
    def cuMemCreate(self, out, size, prop, flags):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = 2
        self.handles.add(2)
        return 0
    def cuMemAddressReserve(self, out, *args):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = 8192
        return 0
    def cuMemMap(self, address, size, offset, handle, flags):
        assert offset == 0 and address not in self.maps
        if self.fail == 'map' and address == 4096 and handle == 2:
            self.fail = None
            return 1
        self.maps[address] = (size, handle)
        return 0
    def cuMemUnmap(self, address, size):
        assert self.maps[address][0] == size
        del self.maps[address]
        return 0
    def cuMemSetAccess(self, address, *args):
        if self.fail == 'access' and address == 4096 and self.maps[address][1] == 2:
            self.fail = None
            return 1
        return 0
    def cuMemcpyDtoD_v2(self, dest, src, size):
        if self.fail == 'copy':
            self.fail = None
            return 1
        self.copies.append((dest, src, size))
        return 0
    def cuMemRelease(self, handle):
        assert all(h != handle for _, h in self.maps.values())
        self.handles.remove(handle)
        return 0
    def cuMemAddressFree(self, address, size):
        assert address not in self.maps
        return 0

def owner():
    a = object.__new__(vmm.MixedResidencyAllocation)
    a.lib = Driver()
    a.va, a.size, a.device_index = 4096, 8, 0
    a.hot_bytes = 0
    a._cold_mapping = (4096, 8, 1)
    a._promoted_handle = None
    a._handles = [('cold', 1)]
    a._mappings = dict(a.lib.maps)
    a._freed = False
    return a

class Gate(unittest.TestCase):
    def test_budget_and_eviction(self):
        a = owner()
        self.assertEqual(a.promote_up_to(7), 0)
        self.assertEqual(a.promote_up_to(8), 8)
        self.assertEqual(a.lib.copies, [(8192, 4096, 8)])
        self.assertEqual(a.va, 4096)
        self.assertEqual(a.lib.handles, {1, 2})
        self.assertEqual(a.promote_up_to(100), 0)
        self.assertEqual(a.evict_promoted(), 8)
        self.assertEqual(a.lib.maps, {4096: (8, 1)})
        self.assertEqual(a.lib.handles, {1})
        a.free()
        a.free()
        self.assertEqual(a.lib.handles, set())
    def test_failure_atomicity(self):
        for failure in ('copy', 'map', 'access'):
            with self.subTest(failure=failure):
                a = owner()
                a.lib.fail = failure
                with self.assertRaises(RuntimeError):
                    a.promote_up_to(8)
                self.assertEqual(a.lib.maps, {4096: (8, 1)})
                self.assertEqual(a._mappings, a.lib.maps)
                self.assertEqual(a.lib.handles, {1})
                self.assertEqual(a.hot_bytes, 0)

if __name__ == '__main__':
    unittest.main()
