#!/usr/bin/env python3
"""Isolated exact prepared-scale/native MoE gate; never loads a complete model."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scale_codec_core as codec
from scale_codec_gpu import decode_one
from b12x.moe.fused_moe import _impl as native
from b12x.moe.fused_moe._tuning import MoeDecodeConfig


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


def closure_tensors(fn):
    return [c.cell_contents for c in fn.__closure__ or () if isinstance(c.cell_contents, torch.Tensor)]


def run(args, report):
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
    scratch = [torch.empty_like(t) for t in original]
    launches, rows = [], []
    for name, src, dst in zip(('w13_sfb', 'w2_sfb'), original, scratch):
        assert src.is_contiguous() and dst.is_contiguous()
        raw = src.view(torch.uint8).flatten().cpu().numpy().tobytes()
        blob = codec.encode_bytes(raw)
        launch = decode_one(torch, blob, len(raw), dst.view(torch.uint8).flatten())
        launches.append(launch)
        assert torch.equal(src.view(torch.uint8), dst.view(torch.uint8))
        resident = [t for t in closure_tensors(launch) if t.data_ptr() != dst.data_ptr()]
        rows.append(dict(name=name, shape=list(src.shape), dtype=str(src.dtype), raw_bytes=len(raw), encoded_file_bytes=len(blob), mode=blob[4], sha256=hashlib.sha256(raw).hexdigest(), decoder_resident_bytes=sum(t.nbytes for t in resident), resident_tensors=[dict(dtype=str(t.dtype), bytes=t.nbytes) for t in resident], scratch_bytes=dst.nbytes, exact=True))
    report['scales'] = rows
    altered = substitute_scales(owner, *scratch)
    assert altered.w1_fp4.data_ptr() == owner.w1_fp4.data_ptr()
    assert altered.w2_fp4.data_ptr() == owner.w2_fp4.data_ptr()
    def decode():
        for launch in launches:
            launch()
    byte_pairs = [(s.view(torch.uint8), d.view(torch.uint8)) for s,d in zip(original,scratch)]
    def copy():
        for s,d in byte_pairs:
            d.copy_(s)
    decode(); copy(); decode()
    report['codec'] = dict(scope='full prepared layer; native expert IDs retain canonical offsets', decoder_ms=event_ms(decode,args.repeats), raw_copy_ms=event_ms(copy,args.repeats), timing='CUDA events; uploaded descriptors and preallocated scratch; JIT warm before measurement; Python launch overhead can leave GPU idle')
    before = memory()
    codec_graphs = []
    for fn in (decode, copy):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        codec_graphs.append(graph)
    after = memory()
    report['codec']['graph_ms'] = {name: event_ms(g.replay, args.repeats)
                                  for name, g in zip(('decode', 'raw_copy'), codec_graphs)}
    report['codec']['memory_before_graph'] = before
    report['codec']['memory_after_graph'] = after
    report['codec']['graph_allocated_delta'] = after['allocated'] - before['allocated']
    report['codec']['graph_reserved_delta'] = after['reserved'] - before['reserved']
    for graph in codec_graphs:
        graph.reset()
    if args.codec_only:
        report['status'] = 'codec_only_passed_native_unverified'
        return
    report['cases'] = []
    for tokens in args.tokens:
        config = MoeDecodeConfig(backend='dynamic', route_planner='internal', max_active_clusters=None, dynamic_tile_m=16, dynamic_route_mode='grouped')
        caps = native.TPMoEScratchCaps(max_tokens=tokens, num_topk=min(8,e), device='cuda:0', weight_plan=plan, quant_mode='w4a8_mx', decode_config=config, deterministic_output=True)
        pool = native.allocate_tp_moe_workspace_pool()
        native.materialize_tp_moe_arena_workspaces(pool,caps=caps)
        a = torch.randn(tokens,k,device='cuda',dtype=torch.bfloat16) * 0.1
        ids = torch.stack([torch.randperm(e,device='cuda')[:min(8,e)] for _ in range(tokens)]).to(torch.int32)
        probabilities = torch.full(ids.shape,1/ids.shape[1],device='cuda',dtype=torch.float32)
        out = torch.empty_like(a)
        common = dict(scratch=pool,a=a,topk_ids=ids,topk_weights=probabilities,output=out,quant_mode='w4a8_mx',decode_config=config)
        binding_a = native.build_tp_moe_fp4_binding(experts=owner,**common)
        binding_b = native.build_tp_moe_fp4_binding(experts=altered,**common)
        def op_a():
            native.b12x_moe_fp4(binding=binding_a)
        def op_b():
            native.b12x_moe_fp4(binding=binding_b)
        samples_a, samples_b = [], []
        parity_samples = 32
        for _ in range(parity_samples):
            op_a(); samples_a.append(out.clone())
            decode(); op_b(); samples_b.append(out.clone())
        reference_a = samples_a[0]
        reference_b = samples_b[0]
        aa_errors = [
            (reference_a.float() - sample.float()).abs().max().item()
            for sample in samples_a[1:]
        ]
        bb_errors = [
            (reference_b.float() - sample.float()).abs().max().item()
            for sample in samples_b[1:]
        ]
        ab_errors = [
            (reference_a.float() - sample.float()).abs().max().item()
            for sample in samples_b
        ]
        tolerance = max([*aa_errors, *bb_errors], default=0.0)
        error = max(ab_errors, default=0.0)
        assert torch.isfinite(reference_a).all()
        assert all(torch.isfinite(sample).all() for sample in samples_b)
        assert error <= tolerance, (error, tolerance, aa_errors, bb_errors, ab_errors)
        def codec_native_b():
            decode()
            op_b()
        before_graph = memory()
        graphs = []
        for fn in (op_a, codec_native_b, decode, copy):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        after_graph = memory()
        row = dict(tokens=tokens, operation='b12x_moe_fp4 dynamic', parity_samples=parity_samples, native_repeatability_max_abs=tolerance, native_a_repeatability_samples=aa_errors, native_b_repeatability_samples=bb_errors, ab_max_abs=error, ab_samples=ab_errors, bitwise=all(torch.equal(reference_a, sample) for sample in samples_b), pass_parity=True,
                   graph_ms={name:event_ms(g.replay,args.repeats) for name,g in zip(('native_a','codec_plus_native_b','decode','raw_copy'),graphs)}, memory_before_graph=before_graph,memory_after_graph=after_graph, graph_allocated_delta=after_graph['allocated']-before_graph['allocated'], graph_reserved_delta=after_graph['reserved']-before_graph['reserved'])
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
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--tokens', nargs='+',type=int,default=[1,16,128])
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--codec-only', action='store_true')
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
