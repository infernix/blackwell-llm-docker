#!/usr/bin/env python3
"""Isolated exact prepared-scale/native MoE gate; never loads a complete model."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import traceback

import torch
from safetensors import safe_open

from codec import encode, validate
from b12x.moe.fused_moe import _impl as native
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.moe.fused_moe.workloads import make_tuning_routes


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                device_free=free, device_total=total)


def substitute_scales(owner, first, second):
    """Respect frozen owners and canonical scale-storage validation."""
    payload = owner.representation_for("w4a8_mx")
    changed = dataclasses.replace(payload, w13_sfb=first, w2_sfb=second)
    return dataclasses.replace(owner, w1_blockscale=first, w2_blockscale=second,
                              representation=dataclasses.replace(owner.representation, value=changed))


def event_ms(fn, repeats):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


def run(args, report):
    if args.deterministic:
        os.environ['B12X_DYNAMIC_DETERMINISTIC_OUTPUT'] = '1'
    torch.cuda.set_device(0)
    torch.manual_seed(7341)
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__, cuda=torch.version.cuda)
    report['memory_initial'] = memory()
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    def load(expert, projection, suffix):
        key = f'layers.{args.layer}.ffn.experts.{expert}.{projection}.{suffix}'
        with safe_open(args.model / index[key], framework='pt', device='cpu') as f:
            return f.get_tensor(key)
    weights, scales = {}, {}
    for projection in ('w1', 'w3', 'w2'):
        weights[projection] = torch.stack([load(e, projection, 'weight') for e in range(args.experts)]).view(torch.uint8).cuda()
        scales[projection] = torch.stack([load(e, projection, 'scale') for e in range(args.experts)]).view(torch.uint8).cuda()
    e, n, packed_k = weights['w1'].shape
    k = packed_k * 2
    report['geometry'] = dict(experts=e, hidden=k, intermediate=n, layer=args.layer, source='real checkpoint experts starting at zero; no TP slicing')
    plan = native.plan_b12x_fp4_moe_weights(quant_modes='w4a8_mx', source_format='fp4_e8m0_k32', activation='silu', params_dtype=torch.bfloat16, num_experts=e, hidden_size=k, intermediate_size=n)
    one = torch.ones(e, device='cuda', dtype=torch.float32)
    owner = native.prepare_b12x_fp4_moe_weights(plan=plan, params_dtype=torch.bfloat16,
        w1_fp4=torch.cat((weights['w1'], weights['w3']), dim=1), w2_fp4=weights['w2'],
        w1_blockscale=torch.cat((scales['w1'], scales['w3']), dim=1), w2_blockscale=scales['w2'],
        w1_global_scale=one, w2_global_scale=one, a1_gscale=one, a2_gscale=one)
    del weights, scales
    payload = owner.representation_for('w4a8_mx')
    original = [payload.w13_sfb, payload.w2_sfb]
    compressed, rows = [], []
    for name, src in zip(('w13_sfb', 'w2_sfb'), original):
        encoded = native._encode_prepared_scale_plane(src)
        assert torch.equal(encoded, encode(src)), "startup encoder differs from reference"
        compressed.append(encoded)
        row = dict(name=name, raw_bytes=src.nbytes, persistent_bytes=encoded.nbytes,
                   decoded_global_scratch_bytes=0)
        rows.append(row)
        report['scales'] = rows
        row['staging'] = validate(src, encoded)
    altered = substitute_scales(owner, *compressed)
    report['persistent_raw_bytes'] = sum(t.nbytes for t in original)
    report['persistent_compressed_bytes'] = sum(t.nbytes for t in compressed)
    report['decoded_global_scratch_bytes'] = 0
    if args.probe_only:
        report['status'] = 'staging_passed'
        return
    report['cases'] = []
    for tokens in args.tokens:
        config = MoeDecodeConfig(backend='dynamic', route_planner='internal', max_active_clusters=None, dynamic_tile_m=args.tile_m, dynamic_route_mode=args.route_mode)
        caps = native.TPMoEScratchCaps(max_tokens=tokens, num_topk=min(8,e), device='cuda:0', weight_plan=plan, quant_mode='w4a8_mx', decode_config=config, deterministic_output=True)
        pool = native.allocate_tp_moe_workspace_pool()
        native.materialize_tp_moe_arena_workspaces(pool,caps=caps)
        a = torch.randn(tokens,k,device='cuda',dtype=torch.bfloat16) * 0.1
        if args.route_pattern is None:
            ids = torch.stack([torch.randperm(e, device='cuda')[:min(8, e)] for _ in range(tokens)]).to(torch.int32)
            probabilities = torch.full(ids.shape, 1 / ids.shape[1], device='cuda', dtype=torch.float32)
        else:
            ids = make_tuning_routes(tokens, min(8, e), e, device=a.device)[args.route_pattern]
            route_rows = torch.arange(tokens, device=a.device).unsqueeze(1)
            route_columns = torch.arange(min(8, e), device=a.device).unsqueeze(0)
            probabilities = torch.softmax(route_rows.float() * 0.03125 + route_columns.float() * 0.125, dim=-1).contiguous()
        out = torch.empty_like(a)
        common = dict(scratch=pool,a=a,topk_ids=ids,topk_weights=probabilities,output=out,quant_mode='w4a8_mx',decode_config=config)
        binding_a = native.build_tp_moe_fp4_binding(experts=owner,**common)
        binding_b = native.build_tp_moe_fp4_binding(experts=altered,**common)
        def op_a():
            native.b12x_moe_fp4(binding=binding_a)
        def op_b():
            native.b12x_moe_fp4(binding=binding_b)
        samples_a, samples_b = [], []
        parity_samples = 64
        for _ in range(parity_samples):
            op_a(); samples_a.append(out.clone())
            op_b(); samples_b.append(out.clone())
        stack_a = torch.stack(samples_a).float()
        stack_b = torch.stack(samples_b).float()
        a_min, a_max = stack_a.amin(dim=0), stack_a.amax(dim=0)
        tolerance = (a_max - a_min).max().item()
        ab_errors = [
            (samples_a[0].float() - sample.float()).abs().max().item()
            for sample in samples_b
        ]
        error = max(ab_errors, default=0.0)
        aa_errors = [
            (samples_a[0].float() - sample.float()).abs().max().item()
            for sample in samples_a[1:]
        ]
        bb_errors = [
            (samples_b[0].float() - sample.float()).abs().max().item()
            for sample in samples_b[1:]
        ]
        assert torch.isfinite(stack_a).all() and torch.isfinite(stack_b).all()
        assert error <= tolerance, (error, tolerance, aa_errors, bb_errors, ab_errors)
        before_graph = memory()
        graphs = []
        for fn in (op_a, op_b):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        after_graph = memory()
        row = dict(tokens=tokens, operation='b12x_moe_fp4 dynamic', parity_samples=parity_samples, native_repeatability_range_max_abs=tolerance, native_a_reference_samples=aa_errors, native_b_reference_samples=bb_errors, ab_max_abs=error, ab_reference_samples=ab_errors, bitwise=torch.equal(stack_a[0], stack_b[0]), pass_parity=True,
                   graph_ms={name:event_ms(g.replay,args.repeats) for name,g in zip(('native_a','compressed_native_b'),graphs)}, memory_before_graph=before_graph,memory_after_graph=after_graph, graph_allocated_delta=after_graph['allocated']-before_graph['allocated'], graph_reserved_delta=after_graph['reserved']-before_graph['reserved'])
        report['cases'].append(row)
        for graph in graphs:
            graph.reset()
    report['memory_final'] = memory()
    report['status'] = 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('/model'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--experts', type=int, default=96)
    parser.add_argument('--tokens', nargs='+',type=int,default=[1,16,128])
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--route-mode', choices=['direct', 'grouped'], default='grouped')
    parser.add_argument('--tile-m', type=int, default=16)
    parser.add_argument('--route-pattern', type=int)
    parser.add_argument('--deterministic', action='store_true')
    args=parser.parse_args()
    report={'status':'running','image':'sha256:d9d273c6890060eb5eae511a34efd724dd5c29b6dd767c55f149e0da04a7a4c3'}
    try:
        run(args,report)
    except Exception:
        report['status']='failed'
        report['traceback']=traceback.format_exc()
        raise
    finally:
        args.report.write_text(json.dumps(report,indent=2)+'\n')
