# DS4.1 TP3 — combine + offload-budget decision (07)

## Survivors
From 06-candidates.md: A(control)=B=B2=C in measured steady state (used 96190 MiB / free ~1102 MiB per GPU 0-2). No env/graph candidate produced a measurable reclaim:
- FI workspace trim: neutral (workspaces evidently not the dominant fixed cost at this shape).
- max-cudagraph-capture-size 24->16: neutral; graph pool is only ~0.02 GiB here.
Combined expected saving therefore ~0 MiB/rank. Final settings unchanged: capture size 24, no FI workspace override.

## Correctness gate
Each candidate run reached health 200 and passed the smoke chat completion (finish_reason stop, numeric reply, latency 0.72-0.77 s). Full repo functional suite not re-run because the surviving configuration is byte-identical to the previously validated live configuration (validate-config exit 0; capture size restored to 24 and verified).

## Offload arithmetic (measured, per rank, GPUs 0-2)
- steady used 96190 MiB / free ~1102 MiB (~1.07 GiB).
- current offload = 6.68 GiB across ordinals >=20: ords 20/21/22 contribute ~2.24/2.24/2.20 GiB (engine log overlay walk).
- raising DS41_OFFLOAD_FROM_ORDINAL by one step (20->21) moves ~2.24 GiB back onto the GPU; available steady headroom is ~1.07 GiB -> does not fit even for a single step.
- Conclusion from direct measurement: with the current weight/KV/graph footprint, dropping the decoder-half expert offload entirely (or even partially by one ordinal step) is NOT possible on GPUs 0-2. The 6.68 GiB offload must stay.

## What remains viable if more margin is ever needed
Trim the explicit KV pool (kv-cache-memory-bytes, currently 1124073472 ~= 1.05 GiB -> ~0.5 GiB at half) trading context capacity; everything else measured already sits at its floor.
