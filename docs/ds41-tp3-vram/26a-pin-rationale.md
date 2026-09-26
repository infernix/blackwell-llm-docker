# L3 pin_memory rationale (26a)

Advisor flagged keeping pin_memory on the UVA path (fork CUDA branch does not re-pin). Measured in-container:
- pinned-first: 0.2 ms per 32 MiB view creation; unpinned+cudaHostRegister: 6.1 ms (one-time cost at load)
- RSS after unpinned path is LOWER (906 vs 938 MiB) - the register path pins the EXISTING pageable buffer in place;
  the pin_memory path creates a SECOND buffer (pageable->pinned copy), so per offloaded param the transient
  host footprint is 2x the param size (the 1.9x from upstream #56410).
- Steady-state serving path is unchanged either way: registered pages ARE pinned (cudaHostRegister pins).
- Per-layer one-time cost of skipping pin: ~420 ms/layer for 2.24 GiB - acceptable at load (3 layers = ~1.3 s).
DECISION: keep the 56413 port (pin only on fallback path). Rationale recorded; net -6.72 GiB transient host spike.
