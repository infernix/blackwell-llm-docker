# DS4.1 TP3 — Checkpoint byte-level screening (02)

Source of truth: `/home/gerben/ai/models/distributedcognition-DeepSeek-V4.1-Flash-abliterated` — 48 `model-*.safetensors` shards + `model.safetensors.index.json` + `config.json`. Census computed from safetensors headers only (`data_offsets`), host python3 3.14.7. Tensor payload = sum(s1-s0). Header note: these shards carry an 8-byte little-endian header-length prefix (e.g. file 1: `f0 6a 00 00 ...` -> len 27376); header JSON follows at offset 8.

## 1. Per-name-class byte census (measured)

96,085 tensors total. Tensor payload sum = 510,286,023,000 B = 475.241 GiB — exactly `metadata.total_size` in `model.safetensors.index.json` (510286023000). On-disk bytes across the 48 shards: 510,296,708,312.

| class | tensors | bytes | GiB |
|---|---|---|---|
| experts payload (I8) | 47,232 | 278,585,671,680 | 259.453 |
| engram embed tables (F8_E4M3) | 2 | 196,613,849,600 | 183.111 |
| expert scales (F8_E8M0) | 47,232 | 17,411,604,480 | 16.216 |
| bf16-class survivors (layers.* attn/hc/index, non-expert) | 1,009 | 8,114,775,488 | 7.557 |
| engram other (wkv F8_E4M3 + scales F8_E8M0 + q/k BF16) | 10 | 6,459,226,640 | 6.016 |
| shared_experts (F8_E4M3 + scales) | 258 | 1,523,232,000 | 1.419 |
| vision (BF16) | 259 | 823,685,120 | 0.767 |
| mtp.* (non-expert parts; all mtp.* combined = 7.388 GiB) | 79 | 607,156,872 | 0.565 |
| aligner (BF16) | 4 | 146,821,120 | 0.137 |
| TOTAL | 96,085 | 510,286,023,000 | 475.241 |

Cross-check vs known totals: experts I8 259.453 GiB CONFIRMED exactly; expert scales F8_E8M0 16.216 GiB CONFIRMED exactly; engram tables F8_E4M3 183.111 GiB CONFIRMED exactly; engram scales measured 5.723 GiB vs quoted 5.722 (rounding); BF16 survivors = sum of all BF16 dtype bytes 3,952,883,712 = 3.681 GiB CONFIRMED exactly. Whole-checkpoint figure corrects to 475.2 GiB binary (510.3 GB decimal), not ~497 GiB.

Dominant shapes: experts I8 (2304,2560) n=31,488 -> 172.969 GiB; experts I8 (5120,1152) n=15,744 -> 86.484 GiB; expert scales F8_E8M0 (2304,160) 10.811 + (5120,72) 5.405 GiB; engram tables F8_E4M3 (384016682,256) 91.557 + (384006168,256) 91.554 GiB; engram embed scales F8_E8M0 (384016682,8)+(384006168,8) = 5.723 GiB.

Evidence: python3 header census (hub job ckpt-screen2/ckpt-screen3): reads u64 header len, json header, sums `data_offsets` deltas per class; output `TOTAL_PAYLOAD_BYTES 510286023000 475.241` and CLASS lines above.

## 2. embed.weight vs head.weight

Both BF16, shape [129280, 5120], 1,323,827,200 bytes (1.2329 GiB) each. sha256 over the exact byte range (seek BASE+s0, read s1-s0):

- `embed.weight` (model-00002-of-00048.safetensors, s0=0): `78adec772ae091c82dc7fed0d7ca6cd534751cd42cc612a7e4b71a83166e1655`
- `head.weight` (model-00043-of-00048.safetensors, s0=0): `68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789`

VERDICT: NOT byte-identical (hashes differ). Head is genuinely untied despite identical shape/dtype. Consistent with `config.json` text_config: `"tie_word_embeddings": false`. No `lm_head`/`embed_weight` alias keys exist in the index; every shard header has empty `__metadata__` ({} for all 48 files). `norm.weight` BF16 [5120] sha256 `9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428` (in shard 43, offset 1323827200).

## 3. Duplicate scan

- BF16 survivors (embed/head/aligner/markov/vision/hc/index BF16, all dtypes==BF16): 520 tensors -> 520 distinct sha256 hashes, 0 duplicate hash groups. No exact duplicates found in this class.
- Expert weights sample-check: 4 layer groups (layers 0-9,10-19,20-29,30-39 pools 12,672 / 11,520 / 11,520 / 11,520 name-counts incl. w1/w3), random.seed(0), 20 sampled per group = 80 tensors hashed: 80/80 distinct hashes; identical-pair rate 0.00. Experts do not share byte-identical weights; no dedup win available there.

## 4. Per-TP-rank share arithmetic (GPUs 0-2, TP3, EP3)

| class | sharding | GiB total | est GiB/rank |
|---|---|---|---|
| experts payload+scales, layers 0-19 (encoder half) | split EP3 | 141.196 | 47.065 |
| experts payload+scales, layers 20-39 (decoder half) | split EP3, UVA-offloaded to host RAM | 134.473 | 44.824 (host-backed; counts partially toward GPU VA) |
| engram embed tables + scales | row-sharded /3 | 189.127 | 63.042 if in-GPU; largely host-resident in practice |
| shared_experts | col/row-shard /3 | 1.419 | 0.473 |
| attn/hc fp8 + bf16 layer survivors (non-expert, non-engram) | mostly /3, tiny vectors replicated | ~9.99 (7.557+1.562+... incl.) | ~3.7 |
| embed.weight | vocab-shard /3 | 1.233 | 0.411 (per_rank_ceil 43,094 rows, pad-to-128 = 43,136; padded = raw = 0.411) |
| head.weight | vocab-shard /3 | 1.233 | 0.411 |
| vision tower | replicated (or TP-shardable) | 0.767 | ~0.26 or 0.767 replicated |
| aligner | col/row-shard /3 | 0.137 | ~0.046 |
| mtp.* (dspark, 3 layers) | split/shard /3 | 7.388 | ~2.46 |

Measured reference point: model loading WITH decoder-half expert UVA offload active = 88.52 GiB/rank (explicit KV 1.05 GiB; cards 95.6 GiB usable). Assumption recorded, not derived: layers>=20 experts (~44.8 GiB/rank if evenly split) plus most engram table memory are host/UVA-backed views, so they commit GPU VA only partially at steady state — which is why measured 88.52 sits between fully-sharded (~158.4 GiB/rank naive all-/3 sum) and the ~6.68 GiB/rank currently claimed as pure-offload footprint. All bytes in this section come from the header census above; split factors 3 are TP3/EP3 placement assumptions consistent with the ds41_3gpu_karmic_r1 placement, not measured per-rank allocations.
