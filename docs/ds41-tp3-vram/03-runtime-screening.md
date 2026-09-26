# DS4.1 TP3 - installed-runtime persistent-buffer screening

Installed tree read with the container stopped (no mounts created; nothing restarted):
`/home/gerben/.local/share/containers/storage/overlay/254d59abfab2ece34bc3e4006fda45b406b4de12de0baedfc53bd561a381b053/diff/opt/venv/lib/python3.12/site-packages/` (abbreviated `<SP>/`; `/opt/lil/runtime` = `.../diff/opt/lil/runtime`).
Overlay: `/home/gerben/ai/llama-swap/config/ds41-tp3-offload/uva.py`. Placement: `/home/gerben/ai/llama-swap/config/ds41-tp3-launch.yaml`.

Effective sizing inputs (ds41-tp3-launch.yaml:6-23): tensor-parallel-size 3, max-num-seqs 2, max-num-batched-tokens 2048, max-cudagraph-capture-size 24, gpu-memory-utilization 0.98, mode off.
Checkpoint geometry (`/home/gerben/ai/models/distributedcognition-DeepSeek-V4.1-Flash-abliterated/config.json`): vocab_size 129280, hidden_size 5120, hc_mult 4, num_hidden_layers 40, num_attention_heads 64, head_dim 512, index_topk 512, candidate_topk_blocks 2048, candidate_source_layer_id 20, num_experts_per_tok 6, tie_word_embeddings false, max_position_embeddings 1048576.

## Anchor for all sizes
`<SP>/vllm/v1/worker/gpu_model_runner.py` delegates to the v2 runner. `<SP>/vllm/v1/worker/gpu/model_runner.py:235-237`:
```
235|        self.max_model_len = self.model_config.max_model_len
236|        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
237|        self.max_num_reqs = self.scheduler_config.max_num_seqs
```
So persistent buffers are functions of 2048 (tokens) or 2 (reqs), NOT of max_model_len - except the two flagged below.

## 1. Persistent buffers at startup (besides weights + KV + graphs)

| Buffer | Site (path:line) | Proportional to | Size here |
|---|---|---|---|
| FlashInfer trtllm module-global workspace | `<SP>/vllm/v1/attention/backends/flashinfer.py:106-112` `torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda")`; consumed at :2247, :2435 | fixed env default | **394 MiB** |
| FlashInfer per-attn-group impl workspace | same file `:1019-1043` | max_num_batched_tokens, floored by env default | **max(394 MiB, est)**; est = `2048 * num_qo_heads * head_dim * 16` (`FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM = 16`, :94-95; heads/head_dim :767-772) |
| sparse-MLA FI workspace `_fi_sparse_workspace` | `<SP>/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:384-400` | fixed base + optional DCP slab | **>= 394 MiB** (`_DEFAULT_WORKSPACE_BUFFER_SIZE = 394 * 1024 * 1024`, :326); slab only when dcp_world_size > 1 (`compute_trtllm_sparse_mla_workspace_bytes` early-return :351-352) |
| input_ids / positions / is_padding | `<SP>/vllm/v1/worker/gpu/input_batch.py:35-37` | max_num_batched_tokens | 8 KiB + 16 KiB + 2 KiB |
| query_start_loc / seq_lens / dcp_local_seq_lens | same file :38-45 | max_num_seqs | 12 B + 8 B + 8 B |
| RequestState.all_token_ids `[max_num_reqs, max_model_len]` int32 | `<SP>/vllm/v1/worker/gpu/states.py:31-39`, `uva_instead_of_gpu=True` | max_model_len - but HOST-side by design | 8 MiB pinned HOST, not device heap |
| last_sampled_tokens / draft_tokens / next_prefill_tokens | same file :65-85 | max_num_reqs | ~112 B |
| output_bin_counts `[max_num_reqs, vocab_size]` int32 | `<SP>/vllm/v1/worker/gpu/sample/penalties.py:40-42` | max_num_seqs x vocab | `2*129280*4 = 1,034,240 B` = **0.99 MiB** |
| prompt_bin_mask `[max_num_reqs, ceil(vocab/32)]` int32 | same file :32-37 | max_num_seqs x vocab/32 | ~32 KiB |
| bad_words / logit_bias / logprob / thinking_budget state | `sample/bad_words.py:22-32`, `sample/logit_bias.py:20-50`, `sample/states.py:22-26`, `sample/thinking_budget.py:52-67` | max_num_seqs | tens of KiB |
| RoutedExpertsCapturer.device_buffer `[max_num_batched_tokens, num_layers, num_experts_per_tok]` int32 | `<SP>/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:89-102` | max_num_batched_tokens | `2048*40*6*4 = 1,966,080 B` = **1.88 MiB** (matches documented formula, :56-60) |
| topk_indices_buffer `[max_num_batched_tokens, index_topk]` int32 | `<SP>/vllm/models/deepseek_v41/nvidia/model.py:485-489` | max_num_batched_tokens | `2048*512*4` = **4 MiB** |
| candidate_block_buffer `[max_num_batched_tokens, candidate_topk_blocks]` int32 | same file :498-502 (active: candidate_source_layer_id 20 > 0) | max_num_batched_tokens | `2048*2048*4` = **16 MiB** |
| _mtp_hidden_buffer `[max_num_batched_tokens, hc_dim]` | same file :560-571, gated on `spec_config.use_eagle() or uses_draft_model()` | max_num_batched_tokens | 80 MiB with spec decode; **None** here (`mode: off`, ds41-tp3-launch.yaml:23). `models/deepseek_v41/nvidia/dspark.py:105-109` allocates a second topk_indices_buffer in dspark mode. |
| boundary_capture `[3, num_reqs, NUM_BOUNDARY_CHECKPOINT_SLOTS]` int32 | `<SP>/vllm/v1/worker/gpu/model_runner.py:2399-2402` | max_num_reqs | ~200 B |
| routed-experts transit buffers | same file :7662-7685 (`max_tokens = scheduler_config.max_num_batched_tokens`) | max_num_batched_tokens | ~16 KiB device |
| UVA / staged-write pools | `<SP>/vllm/v1/worker/gpu/buffer_utils.py:29-175`; multiplier set at `<SP>/vllm/v1/worker/gpu_model_runner.py:247-249` `set_default_max_concurrency(vllm_config.max_concurrent_batches)`, floor `max(2, n)` (buffer_utils.py:24-26) | max_concurrent_batches x rows | pinned HOST |
| b12x attention scratch | `<SP>/b12x/attention/_shared/workspace.py:2005-2009` fused-indexer scratch; :2048-2051 `[max_paged_q_rows, max_page_table_width]`; :2414-2428 tmp_lse/final_lse; :2518-2521 KV gather scratch; :1244-1248 arena bytes (see `<SP>/b12x/_lib/scratch.py:26-37`) | query/page-table rows, never context length | MiB-scale |

Notes:
- The two FlashInfer workspaces are independent allocations (`trtllm_workspace_buffer` flashinfer.py:101-112 vs `_fi_sparse_workspace` flashinfer_mla_sparse.py:384-400); both are live for this model, so ~788 MiB/rank from these two alone.
- Head-room estimate uses per-rank heads (`get_num_attention_heads_from_layers`, flashinfer.py:767-769) and `head_dim = kv_cache_spec.head_size` (:772). head_dim 512 with <=24 heads/rank keeps est <= 394 MiB, so the env default binds.
- Explicit override interacts with the computed floor: flashinfer_mla_sparse.py:370-386 honours a smaller `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` but warns the trtllm-gen kernel may overflow.
- Non-UVA platforms move those pools onto the device: buffer_utils.py:41-57 (`NonUvaBuffer` allocates `torch.zeros(size, ..., device=current_platform.device_type)`), so UVA availability is itself a VRAM lever.
- all_token_ids deliberately costs no device memory: states.py:31-38 comment "This tensor can be extremely large (e.g., several GBs)... To save GPU memory, we use UVA instead of GPU" with `uva_instead_of_gpu=True`; buffer_utils.py:153-163 allocates it `device="cpu", pin_memory=True`.

### FlashInfer autotune dummy-run buffers
`<SP>/vllm/model_executor/warmup/kernel_warmup.py`:
```
329|    max_tokens = runner.scheduler_config.max_num_batched_tokens
331|    token_counts = [max_tokens]
335|        and max_tokens > _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
337|        token_counts.append(_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS)
338|    token_counts.extend(_flashinfer_deferred_moe_token_counts(runner))
```
`_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS = 32` (:305). Each count feeds `fi_utils.flashinfer_get_hybrid_num_tokens_buckets(num_tokens)` then `runner._dummy_run(num_tokens=...)` under `fi_utils.autotune(tuning_buckets=...)` (:348-362). With max_num_batched_tokens = 2048 the live bucket anchor is **2048 tokens**; the extra bf16 count only applies when `linear_backend == "flashinfer_cutedsl"`, and this placement resolves `linear_backend: b12x` (`/opt/lil/runtime/profiles/ds41-flash.yaml` defaults), so autotune runs the single 2048 bucket plus deferred-MoE counts (:308-325). Transient dummy-run peaks that set the allocator high-water mark, not persistent allocations.

## 2. Second full copy of loaded weights? No.
- `<SP>/vllm/model_executor/model_loader/default_loader.py:271-275` routes `load_format == "instanttensor"` to `instanttensor_weights_iterator`; `<SP>/vllm/model_executor/model_loader/weight_utils.py:1356-1365` opens with `copy=True`, comment: "copy=True yields tensors that own their memory, staying valid after the context exits or InstantTensor reuses its buffer." Each yielded tensor lands via `default_weight_loader` -> `copy_weight(param.data, loaded_weight)` (weight_utils.py:1440-1452). No accumulation of loaded tensors.
- Post-processed loaders stay in place: `composed_weight_loader` calls `loader(param, loaded_weight)` then `param.data.copy_(fn(param))` (weight_utils.py:1488-1496).
- One storage per logical weight: `<SP>/vllm/model_executor/layers/vocab_parallel_embedding.py:128-135` builds `Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition, ...))`.
- b12x preparation does not duplicate weights: plans key on caller storage - `<SP>/b12x/attention/paged/_preparation.py:255` `storage_key = (key, binding.k_cache.data_ptr(), binding.v_cache.data_ptr())` - and materialize against caller buffers (:265-269 `materialize_paged_resources(k_cache=..., v_cache=..., ...)`; operand list :37-41). Owner-side device tensors are only small constant LUTs (`b12x/_lib/quant/lut_e4m3.py:145-152`, `lut_fp16.py:55-69`, `iq2_xs.py:104-115`) plus per-shape scratch (`b12x/_lib/scratch.py:26-77`).
- The UVA overlay also avoids duplication: `config/ds41-tp3-offload/uva.py` `_maybe_offload_to_cpu` replaces storage (`cpu_data = p.data.to(device="cpu")`, then `p.data = get_accelerator_view_from_cpu_tensor(cpu_data)` with marker `_vllm_is_uva_offloaded`), consistent with measured 88.52 GiB/rank including the ~6.68 GiB offloaded slice.

## 3. embed.weight vs head.weight per rank
Two independent Parameters per rank; the existing tie path stays inactive.
- `<SP>/vllm/models/deepseek_v41/nvidia/model.py:506-513`: `self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens")` on the first PP rank.
- Same file :1098-1105: `self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))` on the last rank, else `PPMissingLayer()` (:1104-1105).
- Both checkpoint tensors load to distinct destinations - mapper :1009-1012: `"head.weight": "lm_head.weight"`, `"embed.weight": "embed_tokens.weight"`. `ParallelLMHead` subclasses `VocabParallelEmbedding` (vocab_parallel_embedding.py:707); each layer makes its own Parameter (:128-135).
- Sharing mechanism exists but inert: `vocab_parallel_embedding.py:161-165` (`layer.weight = embed_tokens.weight`) via `ParallelLMHead.tie_weights` (:763-765), called from `<SP>/vllm/model_executor/model_loader/utils.py:107-108` -> `model_loader/weight_tying.py:73-96`, which returns unless `model_config.word_embeddings_untied_by_checkpoint`; that flag is only set by `<SP>/vllm/config/model.py:1143-1157`, which returns immediately when `tie_word_embeddings` is falsy. Our checkpoint sets `tie_word_embeddings: false`, so both tables stay resident.
- Derived duplicate-table cost `[INFERENCE]` (padding math vocab_parallel_embedding.py:355-363, `DEFAULT_VOCAB_PADDING_SIZE = 64` :39, `pad_vocab_size` :168-170): padding_size = lcm(64, tp=3) = 192; padded vocab ceil(129280/192)*192 = 129,408; per rank 43,136 x 5120 x 2 B = 441,919,232 B = **~0.42 GiB per table**, ~0.84 GiB for both. Whether these params are also UVA-offloaded depends on the runtime `cpu_offload_params` set.
- Related: `VLLM_REPLICATE_EMBED` cannot combine with tied embeddings at TP>1 - `<SP>/vllm/model_executor/layers/fused_embed_norm.py:56-60` asserts `get_tensor_model_parallel_world_size() == 1`.

## 4. Startup env knobs that plausibly shrink persistent buffers HERE
1. `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` - biggest fixed lever. Readers: `<SP>/vllm/v1/attention/backends/flashinfer.py:110`, `:1020`; `<SP>/vllm/v1/attention/backends/prefill/flashinfer.py:92`; `prefill/trtllm_ragged.py:81`; floor logic `mla/flashinfer_mla_sparse.py:370-399`. Default `394 * 1024 * 1024` (`envs.py:240`, registry `envs.py:1848-1850`). At 2048 tokens the estimate floor is <= default for <=24 heads/rank, so an explicit smaller value shrinks both buffers; keep it >= the sparse-MLA computed minimum (warning flashinfer_mla_sparse.py:374-386).
2. `VLLM_PCIE_DMA_MIN_BYTES` (`off` vs current `6MB`) - gate `<SP>/vllm/distributed/device_communicators/b12x_pcie_all_reduce.py:164-168`; capacity plan :192-199 `elements = max_num_batched_tokens * hidden_size = 2048*5120 = 10,485,760` -> bf16 ~**20 MiB**, fp32 ~**40 MiB** per rank. Set at `/opt/lil/runtime/hardware/rtx-pro-6000-pcie.yaml:21`.
3. `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` / `VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE` / `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE` - read at b12x_pcie_all_reduce.py:104-159; one eager exchange buffer `max(allreduce, fused, 16)` passed as `eager_buffer_bytes`/`max_size` (:283-287). Current 96KB / 96KB / off (rtx-pro-6000-pcie.yaml:20-24) -> ~**96 KiB**.
4. `max_num_batched_tokens` (now 2048) - multiplier behind topk_indices_buffer, candidate_block_buffer, RoutedExpertsCapturer.device_buffer, FI prefill estimate, DMA capacities, b12x query/page-table scratch, autotune bucket anchor (kernel_warmup.py:329-331). All linear in it; `max-num-seqs: 2` already minimizes the request dimension.
5. `max_cudagraph_capture_size: 24` with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` - readers `<SP>/vllm/v1/worker/gpu/model_runner.py:737-739`, `<SP>/vllm/v1/worker/gpu/cudagraph_utils.py:908-909`, `<SP>/vllm/config/vllm.py:923-931`, `:2130-2134`. Controls graph-pool count; vllm.py:169-173 documents DeepseekV41ForCausalLM cannot torch.compile, which is why the placement pins it to 0.
6. `max_concurrent_batches` -> `set_default_max_concurrency` (gpu_model_runner.py:247-249; floor `max(2, n)` buffer_utils.py:24-26) multiplies staged-write/UVA pools (host-side). Keep UVA available so `NonUvaBuffer` (buffer_utils.py:41-57) never engages.
7. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` - readers `<SP>/vllm/v1/worker/gpu_worker.py:717-720`, `:776-803`, `:983-993`; allocates nothing but folds the cudagraph estimate into the KV-size decision and logs actual-vs-estimated, which is what makes the explicit 1.05 GiB KV figure reproducible. Env registry `envs.py:347`, `:2266-2268`.

Non-levers for VRAM on this config:
- `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024` - only an `enable=` flag: `<SP>/vllm/models/deepseek_v41/attention.py:820-821` and `models/deepseek_v4/attention.py:798-799`; the three aux streams come from model.py:478-483. Env registry `envs.py:328`, `:2181-2183`.
- `VLLM_USE_FLASHINFER_SAMPLER=1` - kernel selection only (`sample/sampler.py:64-66`, `:330-333`; registry `envs.py:51`, `:902-906`).
- `VLLM_USE_MEGA_AOT_ARTIFACT=1` / `VLLM_USE_STANDALONE_COMPILE=1` - host-side only: assert `<SP>/vllm/compilation/backends.py:97-99`; pre-split deepcopy `backends.py:1184-1185`; serialization source `:1276-1278`; cache factors `compilation/caching.py:327-330`, `:586-587`; registry `envs.py:776-778`, `:816`.
- `enable_jit_warmup` / `enable_flashinfer_autotune` (`<SP>/vllm/config/kernel.py:237-250`) change JIT/autotune behavior, not persistent size.

## Recoverable-footprint conclusion
Fixed non-weight startup footprint per rank ~= 394 MiB (trtllm FI workspace) + >=394 MiB (sparse-MLA FI workspace) + 16 MiB candidate_block_buffer + 4 MiB topk_indices_buffer + ~1.88 MiB routed-experts device_buffer + ~0.99 MiB output_bin_counts + ~50 KiB input/misc + ~60 MiB DMA capacity (bf16+fp32, when DMA min is set) + ~96 KiB PCIe exchange = **~0.9 GiB/rank**, of which the two FlashInfer workspaces are ~0.77 GiB. The duplicate embed/head pair adds ~0.42 GiB/rank (~0.84 GiB for both tables). Everything else is linear in max_num_batched_tokens and tens of MiB. These startup buffers therefore cannot alone supply ~6.7 GiB/rank; the remaining gap must come from weights, KV capacity, and CUDA-graph pool size.
