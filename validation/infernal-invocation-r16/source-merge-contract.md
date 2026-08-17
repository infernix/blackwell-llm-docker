# Infernal Invocation r16 source merge contract

## Status

**Qualified:** the immutable release composition described here produced the
following image:

```text
voipmonitor/vllm:infernal-invocation-vllm5beffc4-b12xa4a0bc8-fi1ac6942-cu133-torch213-20260817-r16
registry digest: sha256:ff9d4f2402ed88b1ae7ca3a6886c80a64d72993f1a593380c8cb6f193437567d
image ID:        sha256:b8ce67bd8ed86ad9a77affe63105b1ace4f7a6a8e09b41e1ba5deb9379a3e81e
```

**Implemented but not merged:** the vLLM integration tree contains 22 open,
non-draft pull requests. The B12X integration tree contains four open,
non-draft pull requests. Every source PR targets its canonical branch.

**Merged:** the LMCache release branch contains every LMCache change used by
the image. B12X pull requests #228, #229, and #230 are present in
`b12x/master`.

The release composition contains no private source patch. The
`disposition: merged` value in an integration lock means that the builder
merged a PR head into the generated source tree; it does not describe the
GitHub PR state.

## Source identities

| Component | Canonical base | Qualified integration tree |
|---|---|---|
| vLLM | `dev/infernal-invocation@d6e0bb7c813264f1293288b9de0d18f277be7c9f` | `5beffc48f7cd9d4ade076e4b6d1f117ac8e79d4a` |
| B12X | `master@c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5` | `a4a0bc8a8f5e56dbef85f9b46b0d74f6e8edb491` |
| LMCache | `release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481` | `e045d729bc5c4c63a40e13d032f42923de97812f` |

## vLLM merge queue

Merge these PRs into `dev/infernal-invocation` in numeric order:

1. https://github.com/local-inference-lab/vllm/pull/285 - preserve resolved Hugging Face revisions across worker spawn.
2. https://github.com/local-inference-lab/vllm/pull/286 - inherit target identity for in-checkpoint speculative drafts.
3. https://github.com/local-inference-lab/vllm/pull/287 - define DeepSeek V4 launch identity, graph capacity, and checkpoint contracts.
4. https://github.com/local-inference-lab/vllm/pull/288 - enforce the B12X mHC input-shape contract for DeepSeek V4 MTP.
5. https://github.com/local-inference-lab/vllm/pull/289 - bound DeepSeek V4 sparse metadata to active context.
6. https://github.com/local-inference-lab/vllm/pull/290 - profile scheduler-reachable serving shapes before KV-cache sizing.
7. https://github.com/local-inference-lab/vllm/pull/292 - enforce the canonical `b12x` package identifier.
8. https://github.com/local-inference-lab/vllm/pull/293 - recover hybrid KV-cache load failures independently across cache groups.
9. https://github.com/local-inference-lab/vllm/pull/294 - preserve grammar bitmask source widths after speculative-draft trimming.
10. https://github.com/local-inference-lab/vllm/pull/295 - stop XGrammar draft validation when grammar state terminates.
11. https://github.com/local-inference-lab/vllm/pull/296 - absorb duplicate DeepSeek V4 tool-call closers.
12. https://github.com/local-inference-lab/vllm/pull/298 - require request decode state before FULL CUDA graph dispatch.
13. https://github.com/local-inference-lab/vllm/pull/300 - load projection-mixed MCG K3/K4/K5 routed experts and persist online K6 payloads.
14. https://github.com/local-inference-lab/vllm/pull/301 - enforce GLM-5.2 B12X sparse-MLA runtime contracts.
15. https://github.com/local-inference-lab/vllm/pull/302 - activate reasoning-aware structural tool grammars from token zero.
16. https://github.com/local-inference-lab/vllm/pull/303 - preserve DeepSeek V4 FP8 quantization with an in-checkpoint DSpark draft.
17. https://github.com/local-inference-lab/vllm/pull/304 - publish native filesystem KV blocks without replacing visible destination inodes.
18. https://github.com/local-inference-lab/vllm/pull/308 - zero recycled physical blocks for heterogeneous attention caches.
19. https://github.com/local-inference-lab/vllm/pull/309 - defer MLA DCP workspace allocation until CUDA materialization.
20. https://github.com/local-inference-lab/vllm/pull/320 - validate accepted speculative tokens against structured-output grammar before scheduler commit.
21. https://github.com/local-inference-lab/vllm/pull/415 - preserve FULL CUDA graph dispatch for DSpark capture-only contexts.
22. https://github.com/local-inference-lab/vllm/pull/417 - parse the legacy direct DeepSeek V4 DSML tool-call form.

## B12X merge queue

Merge these PRs into `master` in numeric order:

1. https://github.com/local-inference-lab/b12x/pull/145 - qualify CUTLASS DSL 4.6.2.
2. https://github.com/local-inference-lab/b12x/pull/221 - execute unpaired K6/MCG dense projections with the fused CuTe DSL kernel.
3. https://github.com/local-inference-lab/b12x/pull/223 - execute projection-mixed MCG K3/K4/K5 routed experts.
4. https://github.com/local-inference-lab/b12x/pull/227 - mask inactive routes in native W4A16 microkernels while preserving compile-time route bounds.

B12X pull requests
[228](https://github.com/local-inference-lab/b12x/pull/228),
[229](https://github.com/local-inference-lab/b12x/pull/229), and
[230](https://github.com/local-inference-lab/b12x/pull/230) are already in
`master@c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5`.

## LMCache state

LMCache pull requests #7-#17, #22, and #23 are already in
`release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481`.
The resulting tree is
`e045d729bc5c4c63a40e13d032f42923de97812f`; no LMCache PR remains for this
release.

## Qualification evidence

- The Docker release and source-composition suites passed.
- The focused B12X suite passed 309 tests with 18 skips.
- DeepSeek-V4-Flash-0731 TP2/DCP1 fixed probabilistic DSpark K5 captured FULL
  target, draft, and DFlash context-KV graphs.
- The 20-second C1 gate measured 172.93 aggregate tok/s and 64.03 target
  steps/s. The Gilded Gnosis r27 control measured 173.18 tok/s and 62.96
  steps/s on the same direct-attached host.
- Strict structured output completed 160/160 requests at concurrency 8 with no
  malformed response or runtime failure.
- Native vLLM filesystem KV restored 695/695 objects after process restart,
  read 1,394,892,800 bytes from disk, and reproduced exact outputs.
- LMCache restored a 24,064-token prefix after L1 eviction and again after a
  complete vLLM and LMCache process restart.
- GLM-5.2 projection-mixed loading, online K6, routed-expert graph replay, and
  sparse-MLA contracts passed focused tests. Full-checkpoint GLM E2E is not
  qualified by this receipt because the validation host cannot fit the TP4
  EXL3 or TP8 NVFP4 profiles.
- GLM SQG support is unsupported and absent from the composition.

Machine-readable evidence:

- https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16-remote-gpu.json
- https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16/runtime-summary.json
- https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16/native-l2-restart-summary.json
- https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16/lmcache-disk-restart-summary.json

## Completion criteria

1. Merge every PR in the vLLM and B12X queues into its named canonical branch.
2. Recompose from the resulting branch heads with empty PR lists and empty
   `source_patches` lists.
3. Require the recomposed source trees to match the qualified integration
   trees, or document and qualify every intentional tree difference.
4. Repeat the Docker source-composition suites, focused B12X suite, DS4 FULL
   graph capture, C1 gate, strict-tool soak, native filesystem restart replay,
   and LMCache disk restart replay.
5. Preserve the r16 registry digest as the rollback identity until the
   branch-only composition passes those gates.
