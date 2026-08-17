#!/usr/bin/env python3
"""Qualify a Kimi-K3 prefix restore from an external LMCache server.

The test stores a chunk-aligned prompt, clears only vLLM's GPU prefix cache,
and submits the identical token sequence again. Qualification requires token
hits in both vLLM connector metrics and LMCache lookup metrics; wall-clock
improvement alone is not accepted as cache evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT_SEED = (
    "External prefix-cache qualification uses a deterministic technical document. "
    "Each paragraph describes the same cache-transfer invariant: the serving engine "
    "keeps its active GPU working set while a CPU-only process owns reusable prefix "
    "objects. A qualified restore must be visible in token counters, preserve the "
    "generated token, and occur only after the local GPU prefix cache is cleared.\n"
)

ANSWER_SUFFIX = (
    "\nA cache restore must preserve the exact model state. "
    "The integer result of two plus two is"
)


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 900,
) -> tuple[bytes, float]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), time.perf_counter() - started


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 900,
) -> tuple[dict[str, Any], float]:
    body, seconds = _request(url, method=method, payload=payload, timeout=timeout)
    value = json.loads(body)
    assert isinstance(value, dict), value
    return value, seconds


def _prometheus_values(text: str) -> dict[str, float]:
    """Sum Prometheus samples by metric name while ignoring comments."""
    values: dict[str, float] = {}
    pattern = re.compile(
        r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
        r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match is None:
            continue
        name = match.group("name")
        values[name] = values.get(name, 0.0) + float(match.group("value"))
    return values


def _metrics(url: str) -> tuple[str, dict[str, float]]:
    body, _ = _request(url, timeout=30)
    text = body.decode()
    return text, _prometheus_values(text)


def _metric(values: dict[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        if name in values:
            return values[name]
    return 0.0


def _reset_local(base_url: str, *, attempts: int = 60) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for _ in range(attempts):
        response, _ = _json_request(
            f"{base_url}/reset_prefix_cache?reset_external=false",
            method="POST",
            timeout=30,
        )
        last = response
        if response.get("success") is True:
            return response
        time.sleep(1)
    raise RuntimeError(f"vLLM local prefix-cache reset did not complete: {last}")


def _clear_external(cache_url: str) -> dict[str, Any]:
    response, _ = _json_request(
        f"{cache_url}/cache/clear?tier=l1",
        method="POST",
        timeout=60,
    )
    assert response.get("status") == "ok", response
    return response


def _tokenize(base_url: str, model: str, token_count: int) -> list[int]:
    suffix, _ = _json_request(
        f"{base_url}/tokenize",
        method="POST",
        payload={
            "model": model,
            "prompt": ANSWER_SUFFIX,
            "add_special_tokens": False,
        },
        timeout=120,
    )
    suffix_tokens = suffix.get("tokens")
    assert isinstance(suffix_tokens, list) and suffix_tokens, suffix
    prefix_count = token_count - len(suffix_tokens)
    assert prefix_count > 0, (token_count, len(suffix_tokens))

    # The first request determines how many seed paragraphs are needed. A
    # second request obtains a sufficiently long deterministic prefix. The
    # fixed arithmetic suffix makes a corrupt restore change an otherwise
    # stable greedy token instead of relying on an arbitrary document boundary.
    sample, _ = _json_request(
        f"{base_url}/tokenize",
        method="POST",
        payload={"model": model, "prompt": PROMPT_SEED, "add_special_tokens": True},
        timeout=120,
    )
    sample_tokens = sample.get("tokens")
    assert isinstance(sample_tokens, list) and sample_tokens, sample
    repeats = prefix_count // len(sample_tokens) + 2
    encoded, _ = _json_request(
        f"{base_url}/tokenize",
        method="POST",
        payload={
            "model": model,
            "prompt": PROMPT_SEED * repeats,
            "add_special_tokens": True,
        },
        timeout=120,
    )
    tokens = encoded.get("tokens")
    assert isinstance(tokens, list) and len(tokens) >= prefix_count, encoded
    result = tokens[:prefix_count] + suffix_tokens
    assert len(result) == token_count
    assert all(isinstance(token, int) for token in result)
    return result


def _complete(
    base_url: str, model: str, tokens: list[int]
) -> tuple[dict[str, Any], float]:
    request = {
        "model": model,
        "prompt": tokens,
        "temperature": 0,
        "max_tokens": 1,
        "ignore_eos": True,
        "logprobs": 20,
    }
    return _json_request(
        f"{base_url}/v1/completions",
        method="POST",
        payload=request,
        timeout=900,
    )


def _choice_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    assert isinstance(choices, list) and len(choices) == 1, response
    text = choices[0].get("text")
    assert isinstance(text, str), response
    return text


def _choice_token_logprob(response: dict[str, Any]) -> float:
    choices = response.get("choices")
    assert isinstance(choices, list) and len(choices) == 1, response
    logprobs = choices[0].get("logprobs")
    assert isinstance(logprobs, dict), response
    token_logprobs = logprobs.get("token_logprobs")
    assert isinstance(token_logprobs, list) and len(token_logprobs) == 1, response
    value = token_logprobs[0]
    assert isinstance(value, int | float), response
    return float(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cache-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=24576)
    parser.add_argument("--chunk-tokens", type=int, default=12288)
    parser.add_argument("--store-wait-seconds", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_tokens < args.chunk_tokens:
        parser.error("--prompt-tokens must cover at least one LMCache chunk")
    if args.prompt_tokens % args.chunk_tokens:
        parser.error("--prompt-tokens must be an exact multiple of --chunk-tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models, _ = _json_request(f"{args.base_url}/v1/models", timeout=30)
    advertised = {
        entry.get("id")
        for entry in models.get("data", [])
        if isinstance(entry, dict)
    }
    assert args.model in advertised, (args.model, advertised)

    tokens = _tokenize(args.base_url, args.model, args.prompt_tokens)
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode()
    token_receipt = {
        "count": len(tokens),
        "sha256": hashlib.sha256(token_bytes).hexdigest(),
        "first_16": tokens[:16],
        "last_16": tokens[-16:],
        "generator": (
            "PROMPT_SEED repetition followed by deterministic truncation and "
            "the tokenized ANSWER_SUFFIX"
        ),
    }
    _write_json(args.output_dir / "tokens.json", token_receipt)

    _reset_local(args.base_url)
    external_clear = _clear_external(args.cache_url)
    before_vllm_text, _before_vllm = _metrics(f"{args.base_url}/metrics")
    before_cache_text, _before_cache = _metrics(f"{args.cache_url}/metrics")

    cold_response, cold_seconds = _complete(args.base_url, args.model, tokens)
    time.sleep(args.store_wait_seconds)
    local_reset = _reset_local(args.base_url)
    before_hit_vllm_text, before_hit_vllm = _metrics(f"{args.base_url}/metrics")
    before_hit_cache_text, before_hit_cache = _metrics(f"{args.cache_url}/metrics")

    hit_response, hit_seconds = _complete(args.base_url, args.model, tokens)
    time.sleep(2)
    after_vllm_text, after_vllm = _metrics(f"{args.base_url}/metrics")
    after_cache_text, after_cache = _metrics(f"{args.cache_url}/metrics")

    _write_json(args.output_dir / "cold-response.json", cold_response)
    _write_json(args.output_dir / "external-hit-response.json", hit_response)

    vllm_query_names = (
        "vllm:external_prefix_cache_queries_total",
        "vllm_external_prefix_cache_queries_total",
    )
    vllm_hit_names = (
        "vllm:external_prefix_cache_hits_total",
        "vllm_external_prefix_cache_hits_total",
    )
    cache_query_names = (
        "lmcache_mp_lookup_requested_tokens_total",
        "lmcache_mp_lookup_requested_total",
    )
    cache_hit_names = (
        "lmcache_mp_lookup_hit_tokens_total",
        "lmcache_mp_lookup_hit_total",
    )
    vllm_query_delta = _metric(after_vllm, vllm_query_names) - _metric(
        before_hit_vllm, vllm_query_names
    )
    vllm_hit_delta = _metric(after_vllm, vllm_hit_names) - _metric(
        before_hit_vllm, vllm_hit_names
    )
    cache_query_delta = _metric(after_cache, cache_query_names) - _metric(
        before_hit_cache, cache_query_names
    )
    cache_hit_delta = _metric(after_cache, cache_hit_names) - _metric(
        before_hit_cache, cache_hit_names
    )
    expected_restore_tokens = (
        (args.prompt_tokens - 1) // args.chunk_tokens
    ) * args.chunk_tokens

    assert _choice_text(cold_response) == _choice_text(hit_response), (
        cold_response,
        hit_response,
    )
    assert vllm_query_delta >= expected_restore_tokens, vllm_query_delta
    assert vllm_hit_delta == expected_restore_tokens, vllm_hit_delta
    assert cache_query_delta == expected_restore_tokens, cache_query_delta
    assert cache_hit_delta == expected_restore_tokens, cache_hit_delta

    cold_token_logprob = _choice_token_logprob(cold_response)
    external_hit_token_logprob = _choice_token_logprob(hit_response)

    summary = {
        "artifact_kind": "Kimi-K3 external LMCache restore qualification",
        "status": "qualified",
        "base_url": args.base_url,
        "cache_url": args.cache_url,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "chunk_tokens": args.chunk_tokens,
        "expected_restore_tokens": expected_restore_tokens,
        "token_sha256": token_receipt["sha256"],
        "cold_response_seconds": cold_seconds,
        "external_hit_response_seconds": hit_seconds,
        "vllm_external_query_token_delta": vllm_query_delta,
        "vllm_external_hit_token_delta": vllm_hit_delta,
        "lmcache_lookup_requested_token_delta": cache_query_delta,
        "lmcache_lookup_hit_token_delta": cache_hit_delta,
        "external_clear_response": external_clear,
        "local_reset_response": local_reset,
        "cold_output": _choice_text(cold_response),
        "external_hit_output": _choice_text(hit_response),
        "cold_token_logprob": cold_token_logprob,
        "external_hit_token_logprob": external_hit_token_logprob,
        "token_logprob_absolute_delta": abs(
            cold_token_logprob - external_hit_token_logprob
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)
    for name, text in (
        ("before-vllm.prom", before_vllm_text),
        ("before-lmcache.prom", before_cache_text),
        ("before-hit-vllm.prom", before_hit_vllm_text),
        ("before-hit-lmcache.prom", before_hit_cache_text),
        ("after-vllm.prom", after_vllm_text),
        ("after-lmcache.prom", after_cache_text),
    ):
        (args.output_dir / name).write_text(text, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {error.url}: {body}") from error
