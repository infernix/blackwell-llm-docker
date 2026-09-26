"""Source-only specialization gate; no CuTe compilation or numerical proof.
Run: python3 docs/ds41-tp3-scale-codec/residency_gate/source_gate.py
"""
import ast
import os
from pathlib import Path

root = Path(__file__).resolve().parents[3]
vllm = Path(os.environ.get("VLLM_SOURCE_DIR", root.parent / "vllm"))
b12x = Path(os.environ.get("B12X_SOURCE_DIR", root.parent / "b12x"))
sources = (
    vllm / "vllm/model_executor/offloader/vmm_shim.py",
    vllm / "vllm/model_executor/offloader/uva_p4.py",
    b12x / "b12x/moe/_shared/kernels/dynamic.py",
    b12x / "b12x/moe/fused_moe/_impl.py",
    vllm / "vllm/model_executor/layers/fused_moe/expert_residency.py",
)
for source in sources:
    compile(source.read_text(), str(source), "exec")
dynamic = b12x / "b12x/moe/_shared/kernels/dynamic.py"
tree = ast.parse(dynamic.read_text())
parents = {}
for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
        parents[child] = parent

def ancestors(node):
    while node in parents:
        node = parents[node]
        yield node

loads = []
decoders = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if isinstance(node.func, ast.Name) and node.func.id == '_ld_global_u64':
        text = ast.unparse(node)
        if 'sfbw13_bank,' in text or 'sfbdown_bank,' in text:
            loads.append(node)
            assert any(isinstance(a, ast.If) and ast.unparse(a.test) == 'cutlass.const_expr(not self.scale_codec)' for a in ancestors(node)), text
    if isinstance(node.func, ast.Name) and node.func.id == '_w4a8_stage_codec_sfb':
        decoders.append(node)
        assert not any('bank_bytes' in ast.unparse(arg) for arg in node.args)
assert len(loads) == 4
assert len(decoders) == 4
# Execute the actual constructor recipe rejection predicate for both modes.
checks = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test).startswith('self.scale_codec and')]
assert len(checks) == 2
recipe = next(n for n in checks if 'quant_recipe' in ast.unparse(n.test))
from types import SimpleNamespace
expr = compile(ast.Expression(recipe.test), '<recipe>', 'eval')
for bank in (False, True):
    values = dict(self=SimpleNamespace(scale_codec=True, bank_mode=bank), quant_recipe='w4a8_mx', w4a8_repacked=True, w4a8_n64_repacked=False)
    assert not eval(expr, values)
    values['w4a8_n64_repacked'] = True
    assert eval(expr, values)
print('PASS: Python compile; codec+bank recipe; four codec sites; four raw SFB table loads excluded')
