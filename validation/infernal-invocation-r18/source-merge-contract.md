# Infernal Invocation Source Merge Contract

## Status

**Qualified:** the source composition in
`patches/releases/infernal-invocation-r18` produced this immutable image:

```text
voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18
registry digest: sha256:414ec7d0d28358cfd8af0697f330f5c8acbb80e4dc4e5ba69c9fd5b5855ea804
image ID:        sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5
```

**Implemented but not merged:** the vLLM integration tree contains 28 open,
non-draft, mergeable pull requests. The B12X integration tree contains four
open, non-draft, mergeable pull requests. Every source pull request targets its
canonical branch.

**Merged:** B12X pull requests #228, #229, and #230 are in `b12x/master`.
Every LMCache change used by the image is in the declared LMCache release
branch.

The source composition contains no private patch. A `disposition: merged`
entry in an integration lock means that the builder merged the pull-request
head into the generated integration tree; GitHub merge state is recorded
separately in this document.

## Source Identities

| Component | Canonical base | Qualified integration tree |
|---|---|---|
| vLLM | `dev/infernal-invocation@6dc2f516688fe6f84c6994dcd20fddf296853a6c` | `f0fa1cefc1865d316c2478525f550e7646addc40` |
| B12X | `master@c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5` | `75787c7a7431b3bea414d2ebf5f2b8671b23eb33` |
| LMCache | `release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481` | `e045d729bc5c4c63a40e13d032f42923de97812f` |

## vLLM Merge Queue

Merge these pull requests into `dev/infernal-invocation` in numeric order:

1. [#285](https://github.com/local-inference-lab/vllm/pull/285) preserves resolved Hugging Face revisions across worker spawn.
2. [#286](https://github.com/local-inference-lab/vllm/pull/286) inherits target identity for in-checkpoint speculative drafts.
3. [#287](https://github.com/local-inference-lab/vllm/pull/287) defines DeepSeek V4 launch identity, graph capacity, and checkpoint contracts.
4. [#288](https://github.com/local-inference-lab/vllm/pull/288) enforces the B12X mHC input-shape contract for DeepSeek V4 MTP.
5. [#290](https://github.com/local-inference-lab/vllm/pull/290) profiles scheduler-reachable serving shapes before KV-cache sizing.
6. [#292](https://github.com/local-inference-lab/vllm/pull/292) enforces the canonical `b12x` package identifier.
7. [#294](https://github.com/local-inference-lab/vllm/pull/294) preserves grammar bitmask source widths after speculative-draft trimming.
8. [#295](https://github.com/local-inference-lab/vllm/pull/295) preserves XGrammar termination across token batches.
9. [#296](https://github.com/local-inference-lab/vllm/pull/296) absorbs duplicate DeepSeek V4 tool-call closers.
10. [#298](https://github.com/local-inference-lab/vllm/pull/298) requires request decode state before FULL CUDA graph dispatch.
11. [#300](https://github.com/local-inference-lab/vllm/pull/300) loads projection-mixed MCG K3/K4/K5 routed experts and persists online K6 payloads.
12. [#301](https://github.com/local-inference-lab/vllm/pull/301) enforces GLM-5.2 B12X sparse-MLA runtime contracts.
13. [#302](https://github.com/local-inference-lab/vllm/pull/302) activates reasoning-aware structural tool grammars from token zero.
14. [#303](https://github.com/local-inference-lab/vllm/pull/303) preserves DeepSeek V4 FP8 quantization with an in-checkpoint DSpark draft.
15. [#304](https://github.com/local-inference-lab/vllm/pull/304) publishes native filesystem KV blocks without replacing visible destination inodes.
16. [#308](https://github.com/local-inference-lab/vllm/pull/308) zeroes recycled physical blocks for heterogeneous attention caches.
17. [#309](https://github.com/local-inference-lab/vllm/pull/309) defers MLA DCP workspace allocation until CUDA materialization.
18. [#320](https://github.com/local-inference-lab/vllm/pull/320) validates accepted speculative tokens against structured-output grammar before scheduler commit.
19. [#415](https://github.com/local-inference-lab/vllm/pull/415) preserves FULL CUDA graph dispatch for DSpark capture-only contexts.
20. [#417](https://github.com/local-inference-lab/vllm/pull/417) parses the legacy direct DeepSeek V4 DSML tool-call form.
21. [#422](https://github.com/local-inference-lab/vllm/pull/422) restores failed external KV loads independently across heterogeneous cache groups.
22. [#423](https://github.com/local-inference-lab/vllm/pull/423) skips absent fused MTP indexer targets for split EXL3 projections.
23. [#429](https://github.com/local-inference-lab/vllm/pull/429) keeps DeepSeek V4 indexer scoring inside breakable CUDA graphs.
24. [#430](https://github.com/local-inference-lab/vllm/pull/430) sizes DeepSeek V4 C128A graph metadata from physical graph capacity.
25. [#431](https://github.com/local-inference-lab/vllm/pull/431) enforces the active-row contract for DeepSeek V4 sparse top-k.
26. [#432](https://github.com/local-inference-lab/vllm/pull/432) preserves GLM sparse-attention row semantics during prefill.
27. [#433](https://github.com/local-inference-lab/vllm/pull/433) resets cached MRV2 logits-processing state for each request.
28. [#434](https://github.com/local-inference-lab/vllm/pull/434) updates DeepSeek V4 sparse metadata without per-step host scalar extraction.

Pull request #289 is superseded by #430. Pull request #293 is superseded by
#422. Neither superseded pull request is part of the source composition.

## B12X Merge Queue

Merge these pull requests into `master` in numeric order:

1. [#145](https://github.com/local-inference-lab/b12x/pull/145) qualifies CUTLASS DSL 4.6.2.
2. [#221](https://github.com/local-inference-lab/b12x/pull/221) executes unpaired K6/MCG dense projections with a fused CuTe DSL kernel.
3. [#223](https://github.com/local-inference-lab/b12x/pull/223) executes projection-mixed MCG K3/K4/K5 routed experts.
4. [#227](https://github.com/local-inference-lab/b12x/pull/227) masks inactive routes in native W4A16 microkernels while preserving compile-time route bounds.

B12X pull requests
[#228](https://github.com/local-inference-lab/b12x/pull/228),
[#229](https://github.com/local-inference-lab/b12x/pull/229), and
[#230](https://github.com/local-inference-lab/b12x/pull/230) are merged into
`master`.

## LMCache State

LMCache pull requests #7 through #17, #22, and #23 are in
`release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481`.
The resulting tree is
`e045d729bc5c4c63a40e13d032f42923de97812f`; no LMCache pull request remains
for this image.

## Official vLLM Main Audit

The official `vllm-project/vllm/main` branch was audited at
`8f4a7f45c53ab52b17023d3ca804e477daa36a23`. The commits after
`203926c477` modify only ROCm CI configuration and
`tests/tools/test_docker_build_metadata_args.py`. They contain no DS4, GLM,
structured-output, KV-cache, CUDA-graph, quantization, or SM120 runtime change.
No additional official-main commit belongs in this source composition.

Kimi-K3 pull requests #418, #419, and #428, FLA pull request #420, and the
KVarN research pull requests #424 through #426 and B12X #231 have
model-specific or research-only contracts. They are not dependencies of the
DS4 and GLM profiles and are intentionally excluded.

## Qualification Evidence

- The Docker release, source-composition, and GLM launcher suites passed.
- Seventeen focused Python tests passed, and `pytest 8.4.1` remains installed
  in the final image.
- DeepSeek-V4-Flash-0731 TP2/DCP1 fixed probabilistic DSpark K5 captured FULL
  target, draft, and DFlash context-KV graphs.
- The 20-second C1 gate measured 164.46 aggregate tok/s and 64.40 target
  steps/s. Infernal Invocation r16 measured 172.93 tok/s and 64.03 target
  steps/s with the same scheduler configuration. The target-step rate did not
  regress; emitted-token throughput followed draft-acceptance variance.
- Strict structured output completed 160 of 160 requests at concurrency 8 with
  no malformed response or runtime failure.
- Native vLLM filesystem KV restored all 695 persisted objects after process
  restart, read 607,357,440 bytes from the filesystem tier, reproduced exact
  outputs, and left no temporary files.
- LMCache restored all 94 persistent chunks and a 24,064-token prefix after a
  complete LMCache and vLLM process restart.
- GLM-5.2 source, launcher, sparse-prefill, projection-mixed loading, online K6,
  and routed-expert tests passed. The rank-sliced EXL3 checkpoint correctly
  rejected a TP2 process because its declared runtime geometry is TP4.

Machine-readable evidence is stored in:

- `validation/infernal-invocation-r18-remote-gpu.json`
- `validation/infernal-invocation-r18/runtime-summary.json`
- `validation/infernal-invocation-r18/native-l2-restart-summary.json`
- `validation/infernal-invocation-r18/lmcache-disk-restart-summary.json`

## Completion Criteria

1. Merge every open pull request in the vLLM and B12X queues into its named
   canonical branch.
2. Recompose from the resulting branch heads with empty pull-request lists and
   empty `source_patches` lists.
3. Require the recomposed source trees to match the qualified integration
   trees, or document and qualify every intentional tree difference.
4. Repeat the Docker source-composition suites, focused source suites, DS4 FULL
   graph capture, C1 gate, strict-tool soak, native filesystem restart replay,
   and LMCache disk restart replay.
5. Preserve the r18 registry digest as the rollback identity until the
   branch-only composition passes those gates.
