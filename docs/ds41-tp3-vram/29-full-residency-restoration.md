# Full routed-expert residency restoration

Date: 2026-09-25

The TP3 launch command accidentally dropped the `uva_p4.py`, `b12x_layers.py`,
and `rope.py` mounts while adding the PCIe one-shot threshold. That made VMM
conversion fall back to stock UVA, left ordinals 20-22 host-backed, and stranded
about 6 GiB/rank of reclaimed VRAM.

The placement command was restored from proven commit `c7f7cf1` while retaining
`VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB`.

Managed cold-start evidence:

- configuration validation: `config OK`
- watched reload: `Configuration Reloaded`
- only tower/aligner stock offload remains: `Total CPU offloaded parameters: 0.9`
  GiB/rank
- ordinals 20, 21, and 22 each converted to VMM regions without fallback
- each ordinal promoted `2,264,924,160` bytes/rank with `hot=384 cold=0`
- all 384 routed experts are GPU-resident in each targeted layer
- 512 MiB KV pool: 536,819 tokens, 1.02x at 524,288 tokens
- steady physical free VRAM: 344 MiB/rank after inference
- authenticated llama-swap completion returned `READY` with `finish_reason=stop`
- `make test-all`, `gofmt -l .`, and `git diff --check` passed

The 256 MiB promotion reserve is intentionally retained. No routed-expert CPU
fallback is accepted as a successful boot.
