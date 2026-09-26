# DS4.1 TP3 — best lossless combination proposal (22)

Question: regardless of effort, which combination is best while staying lossless? Or all?

## Answer: yes — ALL lossless items compose, and they are mutually reinforcing.

Every lossless item targets a DIFFERENT resource, so they stack without interference:

| # | item | resource touched | interaction with others |
|---|---|---|---|
| L1 KV right-size 1.05->0.5 GiB | GPU VRAM (KV pool) | none — KV bytes are disjoint from weights/workspaces |
| L2 ws3 one-shot patch | comm TIME (us), not bytes | reduces the COST of residual expert offload (PCIe step overhead) — pure synergy |
| L3 UVA hardening (#54972+#56413) | allocator ceiling + host-RAM transient | makes deeper offload walks SAFE and halves host spike — prerequisite for comfortable P4 testing |
| L4 NCCL buff/channel shrink | NCCL ring buffers | +0.10-0.15 GiB; only marginal-risk item — hold unless VRAM-critical |
| L10 P4 VMM mixed-residency | WHERE expert bytes live + PCIe traffic | CONSUMES L1: converts KV-trim headroom into cold experts returning to GPU; cuts PCIe 4-5x on the rest |

## The combined end-state (all of L1+L2+L3+L10, L4 optional)

- GPU: +0.547 GiB/rank freed by KV trim, then spent by P4 on ~0.5 GiB of COLD experts resident again
- Host RAM: ~0.5 GiB less expert bytes pinned (P4 moves cold->GPU); L3 halves the walk transient
- PCIe on text decode: ords 20-22 reads drop ~215 -> ~50 MiB/step (4-5x) because only COLD experts (2-5% of routes) remain host-backed; PLUS ws3 one-shot cuts per-all-reduce 44.9->6.0us
- Decode/prefill correctness: zero changes to kernels, plans, graphs (single base pointer preserved); functional suite gates every step
- Net text perf: strictly faster (less PCIe wait + faster all-reduce), no capability loss (KV still covers 2x~203k worst case)

## Effort/sequencing (do not batch code with config)

Phase 1 (config+overlay, one restart): L1 + L2 + L3. Gate: validate-config, functional suite (11), graph capture 2/2, KV tokens check (~567k expected), ws3 one-shot init line present.
Phase 2 (probe, no restart): 20-line VMM HOST-location platform probe on GB202.
Phase 3 (~300 LOC shim, one restart): L10 P4. Gate: same functional suite + numerics vs resident baseline on 1-GPU microbench + TTFT A/B probe (17 sec 5).
Phase 4 (optional, only if VRAM-critical later): L4 NCCL shrink with prefill A/B.

## What "all" does NOT include (kept out, reasons)
- L11/L13 (upstream-gated/accuracy): track only.
- Anything lossy: user constraint.
- LMCache L1 cap cut (48->8): NOT lossless-adjacent to this goal - it frees HOST RAM only and LMCache reuse may return; hold unless host RAM becomes critical again.

## Risk register
- L1: if a >520k-context request ever arrives, it 400s (validation) instead of running - acceptable per observed traffic (max 67 tokens seen; adapter worst case 203k routes to qwen).
- L2: b12x ws3 is upstream-unqualified (PR #297 deferred) - one-shot init failure falls back to PYNCCL automatically (b12x_pcie_all_reduce.py:295-302), so worst case = today behavior.
- L10: platform probe failure kills the design cheaply; UVA fallback stays intact; graph capture unaffected (pointer static).
