# DS4.1 TP3 — vision-offload isolation runs (10)

Goal: keep the 2.35 GiB vision+aligner saving WITHOUT offloading expert ord 23 (text-path cost).

## Run vis2 (budget cap 7.7 GiB, no TO_ORDINAL): FAILED to cap
- walk: towers 0.904 + ords 20-22 (6.72) = 7.624 spent, remaining 0.076 GiB; ord 23 expert params are
  3-6 MiB each, so per-param budget checks bled into ord 23. Total 9.03 again. Byte budgets cannot
  express "stop at ordinal N" when the next module has small parameters.

## Run vis3 (DS41_OFFLOAD_TO_ORDINAL=22 via placement env list): env never reached the container
- placement env items are set on the sh wrapper process on the HOST (proxy/process.go:444), and
  `podman run` without -e does not forward them. Walk still hit ord 23 (9.03 total).
- NOTE: this means DS41_OFFLOAD_FROM_ORDINAL=20 in the placement env list ALSO never reaches the
  container; the default 20 in uva.py:68 made it look like it did. Non-default ordinal values would
  silently not apply for any placement launched this way.

## Run vis4 (explicit -e DS41_OFFLOAD_FROM_ORDINAL=20 -e DS41_OFFLOAD_TO_ORDINAL=22 on podman run): SUCCESS
- walk: towers (0.14+0.77) + ords 20,21,22 only = 7.63 GiB total (ord 23 correctly excluded)
- Model loading took 86.17 GiB/rank (same as vis run: towers gone)
- steady used 95192 MiB / free 2100 MiB per GPU 0-2
  = control 96190 - vision/aligner (2.35) + ord23 back on GPU (1.41) => -998 MiB net vs control
- KV unchanged 1,131,122 tokens; smoke 200 in 0.42 s; container stopped; GPUs idle after.

## Where this lands
| config | offloaded (GiB/rank) | steady free (MiB) | text-path PCIe bytes/step |
|---|---|---|---|
| control (04/06) | 6.68 (ords 20-22) | ~1102 | baseline |
| vis (09) | 9.03 (towers + ords 20-23) | ~3496 | +1.41 GiB ord 23 reads — SLOWER decode |
| vis4 (final) | 7.63 (towers + ords 20-22) | ~2100 | identical to control |

vis4 is the lossless optimum for text speed: +1 GiB free per rank vs control, zero extra PCIe
traffic on the decode/prefill path, vision-only slowdown accepted.
