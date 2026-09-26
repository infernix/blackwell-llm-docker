# DS4.1 TP3 — managed-path functional verification (11)

Purpose: prove commit 9aaf34e's shipped config (env in placement env-list, no -e on cmd) was BROKEN,
then verify the fixed config through llama-swap's managed start.

## Finding: shipped placement env-list does not reach the container
- proxy/process.go:444 puts placement env items on the HOST sh wrapper process only;
  podman run forwards nothing without -e. Commit 9aaf34e shipped DS41_OFFLOAD_TO_ORDINAL=22 in the
  env-list (line 67) with no matching -e on the cmd (line 70) => container would see default 999 and
  walk to ord 23 (9.03 GiB) — the opposite of the verified 7.63 state.
## Fix
- moved both DS41 vars onto the podman run cmd as -e flags (validate-config OK, diff-check clean).

## Managed-path verification (llama-swap started the placement itself; model re-enabled via
  POST /api/models/enable/deepseek-v4.1-flash; cold start through proxy)

- container env confirmed: DS41_OFFLOAD_FROM_ORDINAL=20, DS41_OFFLOAD_TO_ORDINAL=22 (podman inspect Config.Env)
- walk: towers 0.14+0.77, ords 20/21/22 = 2.24 each; TOTAL 7.63 GiB — ord 23 ABSENT (as designed)
- steady used 95208 MiB / free 2084 MiB per GPU 0-2 (GPU3 companions untouched)
- CUDA graph capture COMPLETED: "Capturing CUDA graphs (FULL): 2/2" (not skipped)

## Enforced functional suite (through llama-swap proxy, model=medium/multimodal alias)

| check | expected | observed | verdict |
|---|---|---|---|
| Basic chat | successful response | "Hello", finish stop | PASS |
| Reasoning | content contains 3/10 | content "3/10", reasoning 12 tokens, finish stop | PASS |
| Tool calling | get_weather({"city":"Utrecht"}) | tool_calls[0].function.arguments = {"city": "Utrecht"} | PASS |
| Image input | returns red | content "Red" | PASS |
| Multimodal alias | ALIAS READY | content "ALIAS READY", served DeepSeek-V4.1-Flash | PASS |

Note: cold-start request latency 136 s includes model load (~64 s) + graph capture + KV init; subsequent
checks 0.6-3.6 s. The image-input check exercised the UVA-offloaded vision tower + aligner via PCIe (PASS).
