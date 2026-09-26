#!/usr/bin/env python3
"""Standalone scale-codec CLI: inspect, self-test, roundtrip, encode, verify,
bench-decode. CPU-first; GPU work is opt-in via --gpu / bench-decode and is
marked not_run with the precise reason when unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scale_codec_core as core  # noqa: E402

DEFAULT_MODEL = "/path/to/DeepSeek-V4.1-Flash"


def _select(args) -> list[dict]:
    rows = core.enumerate_selected(args.model, args.include)
    if args.all:
        return rows
    if args.limit_tensors and args.limit_tensors > 0:
        rows = rows[: args.limit_tensors]
    if args.max_bytes and args.max_bytes > 0:
        capped, total = [], 0
        for r in rows:
            if total >= args.max_bytes:
                break
            capped.append(r)
            total += r["length"]
        rows = capped
    return rows


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _emit(report: dict, args) -> None:
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        base = args.output
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        with open(base + ".report.json", "w", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_inspect(args) -> int:
    rows = _select(args)
    total_bytes = sum(r["length"] for r in rows)
    sizes = {r["length"] for r in rows}
    dtypes = {r["dtype"] for r in rows}
    full = args.all or not (args.limit_tensors or args.max_bytes)
    report = {
        "command": "inspect",
        "mode": "full" if full else "bounded-subset",
        "model_dir": args.model,
        "include": args.include,
        "selected_tensors": len(rows),
        "raw_bytes": total_bytes,
        "distinct_tensor_byte_sizes": sorted(sizes),
        "dtypes": sorted(dtypes),
        "first_tensors": [r["name"] for r in rows[:3]],
    }
    facts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "audit_facts.json")
    if os.path.exists(facts_path):
        with open(facts_path, encoding="utf-8") as f:
            facts = json.load(f)
        scope = facts.get("scope", {})
        report["audit_facts"] = {
            "selected_tensors": scope.get("selected_tensors"),
            "raw_bytes": scope.get("raw_bytes"),
        }
        mismatches = []
        if scope.get("selected_tensors") is not None and scope["selected_tensors"] != len(rows):
            mismatches.append({
                "field": "selected_tensors",
                "expected": scope["selected_tensors"], "actual": len(rows)})
        expected_bytes = scope.get("raw_bytes")
        full_span = sum(r["length"] for r in core.enumerate_selected(
            args.model, args.include))
        if expected_bytes is not None and full_span != expected_bytes:
            mismatches.append({
                "field": "raw_bytes", "expected": expected_bytes,
                "actual": full_span})
        report["audit_facts_mismatches"] = mismatches
    _emit(report, args)
    return 0


def cmd_self_test(args) -> int:
    from tests.test_scale_codec import run_all
    results = run_all(gpu=args.gpu)
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] in ("skipped", "not_run"))
    report = {
        "command": "self-test",
        "gpu_requested": bool(args.gpu),
        "tests": results,
        "passed": passed, "failed": failed, "skipped_or_not_run": skipped,
    }
    _emit(report, args)
    return 0 if failed == 0 else 1


def cmd_roundtrip(args) -> int:
    rows = _select(args)
    full = args.all or not (args.limit_tensors or args.max_bytes)
    n_tensors = 0
    n_bytes = 0
    mismatches = []
    io_t = enc_t = dec_t = ver_t = 0.0
    hashes = {}
    for row in rows:
        (src, dt0) = _timed(lambda r=row: core.read_tensor_bytes(r))
        (blob, dt1) = _timed(lambda s=src: core.encode_bytes(s))
        (got, dt2) = _timed(lambda b=blob: core.decode_bytes(b))
        t0 = time.perf_counter()
        ok = got == src
        hashes[row["name"]] = {
            "source_sha256": core.sha256_bytes(src),
            "decoded_sha256": core.sha256_bytes(got),
            "raw_length": len(src), "encoded_length": len(blob),
            "match": ok,
        }
        if not ok:
            a = np.frombuffer(src, dtype=np.uint8)
            b = np.frombuffer(got, dtype=np.uint8)
            diff = np.flatnonzero(a != b)
            first = int(diff[0]) if diff.size else -1
            mismatches.append({
                "name": row["name"], "count": int(diff.size),
                "first_mismatch": first,
                "expected": int(a[first]) if first >= 0 else None,
                "actual": int(b[first]) if first >= 0 else None,
            })
        dt3 = time.perf_counter() - t0
        io_t += dt0; enc_t += dt1; dec_t += dt2; ver_t += dt3
        n_tensors += 1
        n_bytes += len(src)
    report = {
        "command": "roundtrip",
        "mode": "full" if full else "bounded-subset",
        "streamed_no_persist": True,
        "tensors_verified": n_tensors,
        "bytes_verified": n_bytes,
        "mismatched_tensors": len(mismatches),
        "mismatch_details": mismatches[:10],
        "timings_seconds": {
            "io": round(io_t, 4), "encode": round(enc_t, 4),
            "decode": round(dec_t, 4), "verify_compare": round(ver_t, 4),
        },
        "hashes_sample": dict(list(hashes.items())[:5]),
    }
    _emit(report, args)
    return 0 if not mismatches else 1


def cmd_encode(args) -> int:
    rows = _select(args)
    out_prefix = args.output or _default_out_prefix()
    blobs = []
    enc_t = 0.0
    io_t = 0.0
    for row in rows:
        (src, dt0) = _timed(lambda r=row: core.read_tensor_bytes(r))
        (blob, dt1) = _timed(lambda s=src: core.encode_bytes(s))
        blobs.append(blob)
        io_t += dt0
        enc_t += dt1
    manifest = core.write_sidecar(
        out_prefix, rows, blobs, args.model, args.include, force=args.force)
    raw_total = sum(b["raw_length"] for b in manifest["tensors"])
    sidecar_bytes = os.path.getsize(out_prefix + ".sbuf")
    manifest_bytes = os.path.getsize(out_prefix + ".smanifest.json")
    report = {
        "command": "encode",
        "mode": "full" if (args.all or not (args.limit_tensors or args.max_bytes))
                else "bounded-subset",
        "sidecar": out_prefix + ".sbuf",
        "manifest": out_prefix + ".smanifest.json",
        "tensors": manifest["tensor_count"],
        "raw_bytes": raw_total,
        "encoded_sidecar_bytes": sidecar_bytes,
        "manifest_bytes": manifest_bytes,
        "total_serialized_bytes": sidecar_bytes + manifest_bytes,
        "header_bytes": 16 * manifest["tensor_count"],
        "timings_seconds": {"io": round(io_t, 4), "encode": round(enc_t, 4)},
    }
    # report file itself must not collide with the sidecar prefix
    saved, args.output = args.output, out_prefix + ".cli"
    _emit(report, args)
    args.output = saved
    return 0


def _default_out_prefix() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "out", "scales-g256")


def cmd_verify(args) -> int:
    out_prefix = args.output or _default_out_prefix()
    man_path = out_prefix + ".smanifest.json"
    bin_path = out_prefix + ".sbuf"
    manifest = core.read_manifest(man_path)
    source_rows = core.enumerate_selected(
        manifest["model_dir"], manifest["include"]
    )
    source_by_name = {row["name"]: row for row in source_rows}
    checked = 0
    checked_bytes = 0
    mismatches = []
    sidecar_size = os.path.getsize(bin_path)
    with open(bin_path, "rb") as sidecar:
        for ent in manifest["tensors"]:
            sidecar.seek(ent["blob_offset"])
            blob = sidecar.read(ent["blob_length"])
            if len(blob) != ent["blob_length"]:
                mismatches.append({
                    "name": ent["name"],
                    "reason": f"sidecar truncated at offset {ent['blob_offset']}",
                })
                continue
            try:
                decoded = core.decode_bytes(blob)
            except ValueError as exc:
                mismatches.append({
                    "name": ent["name"], "reason": f"invalid blob: {exc}"
                })
                continue
            match = source_by_name.get(ent["name"])
            if match is None:
                mismatches.append({
                    "name": ent["name"], "reason": "tensor missing from checkpoint"
                })
                continue
            src = core.read_tensor_bytes(match)
            src_hash = core.sha256_bytes(src)
            decoded_hash = core.sha256_bytes(decoded)
            if ent["source_sha256"] != src_hash:
                mismatches.append({
                    "name": ent["name"], "reason": "source sha256 changed"
                })
                continue
            if ent.get("decoded_sha256", ent["source_sha256"]) != decoded_hash:
                mismatches.append({
                    "name": ent["name"], "reason": "decoded sha256 mismatch"
                })
                continue
            if decoded != src:
                a = np.frombuffer(src, dtype=np.uint8)
                b = np.frombuffer(decoded, dtype=np.uint8)
                limit = min(a.size, b.size)
                diff = np.flatnonzero(a[:limit] != b[:limit])
                first = int(diff[0]) if diff.size else limit
                mismatches.append({
                    "name": ent["name"],
                    "reason": "bytes differ",
                    "first_mismatch": first,
                    "expected": int(a[first]) if first < a.size else None,
                    "actual": int(b[first]) if first < b.size else None,
                })
                continue
            checked += 1
            checked_bytes += len(src)
    report = {
        "command": "verify",
        "sidecar": bin_path,
        "manifest": man_path,
        "tensors_verified": checked,
        "bytes_verified": checked_bytes,
        "encoded_bytes_on_disk": sidecar_size,
        "manifest_bytes_on_disk": os.path.getsize(man_path),
        "total_serialized_bytes": sidecar_size + os.path.getsize(man_path),
        "mismatched_tensors": len(mismatches),
        "mismatch_details": mismatches[:10],
    }
    saved, args.output = args.output, out_prefix + ".cli"
    _emit(report, args)
    args.output = saved
    return 0 if not mismatches else 1


def cmd_bench_decode(args) -> int:
    report = {"command": "bench-decode"}
    try:
        import scale_codec_gpu as gpu
    except Exception as exc:  # pragma: no cover
        report["gpu"] = {"status": "not_run",
                         "reason": f"import failed: {exc}"}
        _emit(report, args)
        return 0
    result = gpu.bench(args)
    report["gpu"] = result
    _emit(report, args)
    return 0


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="scale_codec",
        description="Lossless E8M0 expert-scale codec prototype "
                    "(binary_exceptions256).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="checkpoint directory holding *.safetensors shards")
    p.add_argument("--include", default=core.DEFAULT_INCLUDE,
                   help="tensor-name regex (default: routed-expert w1/w2/w3 scales)")
    p.add_argument("--limit-tensors", type=int, default=8,
                   help="max tensors in bounded mode (default 8; ignored by --all)")
    p.add_argument("--max-bytes", type=int, default=0,
                   help="cap selected bytes in bounded mode (0 = only --limit-tensors)")
    p.add_argument("--seed", type=int, default=1234,
                   help="seed for synthetic fixtures/random selection")
    p.add_argument("--output", default=None,
                   help="output prefix for reports/sidecars")
    p.add_argument("--all", action="store_true", dest="all",
                   help="stream the complete selected population")
    p.add_argument("--gpu", action="store_true", help="enable GPU tests")
    p.add_argument("--force", action="store_true", help="overwrite existing sidecar")
    p.add_argument("command", choices=["inspect", "self-test", "roundtrip",
                                       "encode", "verify", "bench-decode"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fn = {
        "inspect": cmd_inspect,
        "self-test": cmd_self_test,
        "roundtrip": cmd_roundtrip,
        "encode": cmd_encode,
        "verify": cmd_verify,
        "bench-decode": cmd_bench_decode,
    }[args.command]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
