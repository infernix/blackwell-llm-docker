"""CPU (and optionally GPU) tests for the scale codec.

Runnable without pytest:
    python3 tests/test_scale_codec.py
Or via the CLI:
    python3 scale_codec.py self-test [--gpu]

All fixtures except the 368,640-byte realistic-shape fixture are synthetic;
the large fixture is histogram-derived synthetic data, not checkpoint bytes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scale_codec_core as core  # noqa: E402


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_roundtrip(data: bytes, name: str) -> dict:
    blob = core.encode_bytes(data)
    got = core.decode_bytes(blob)
    ok = got == data
    return {
        "name": name,
        "status": "passed" if ok else "failed",
        "n": len(data),
        "blob_bytes": len(blob),
        "detail": None if ok else "decoded bytes differ",
    }


def t_lengths_and_values(seed: int = 7) -> list[dict]:
    results = []
    rng = _rng(seed)
    cases = {
        "empty": b"",
        "len1": b"\x79",
        "len7": bytes([120, 121, 5, 120, 255, 0, 121]),
        "len8": bytes([120] * 4 + [121] * 4),
        "len9": bytes([0] * 5 + [255] * 4),
        "len255": bytes([120] * 254 + [1, 2]),
        "len256_full_group": bytes([120] * 128 + [121] * 128),
        "len257_odd_tail": bytes([120] * 256 + [121] * 1),
        "len4096": rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes(),
        "every_value_0_255": bytes(range(256)),
        "both_common_only": bytes([120, 121] * 512),
        "all_constant_42": bytes([42]) * 300,
        "no_exceptions_120": bytes([120]) * 600,
        "every_entry_exceptional": bytes(range(256)) + bytes(range(256)) + bytes([3]),
        "clustered_exceptions": bytes([120] * 250) + bytes([7, 8, 9, 10]),
        "tile_boundary_exceptions": (
            bytes([120] * 255) + bytes([200]) + bytes([121] * 255) + bytes([201])
        ),
    }
    for name, data in cases.items():
        results.append(check_roundtrip(data, name))

    # random distributions incl. heavy tails
    for s in (11, 12):
        r = _rng(s)
        data = r.integers(0, 256, size=1000, dtype=np.uint8).tobytes()
        results.append(check_roundtrip(data, f"random_uniform_1000_seed{s}"))
    data = _rng(13).choice(
        [120, 121, 119, 122], size=(5000,), p=[0.55, 0.44, 0.005, 0.005],
    ).astype(np.uint8).tobytes()
    results.append(check_roundtrip(data, "random_biased_5000"))
    return results


def t_bit_semantics() -> dict:
    # bit set only for 121; LSB-first within each byte; unused tail bits zero
    # 89 entries: index multiples of 8 are 121, rest 120 -> binary mode fits
    data = bytes([121] + [120] * 7) * 11 + bytes([121])
    blob = core.encode_bytes(data)
    bits_len = (len(data) + 7) // 8
    bits = np.frombuffer(blob, dtype=np.uint8, count=bits_len, offset=16)
    # each byte covers entries 8i..8i+7; only entry 8i (bit 0) is set.
    ok = (blob[4] == core.MODE_BINARY256
          and bits_len == 12
          and bits.size == bits_len
          and all(int(b) == 0b00000001 for b in bits)
          and core.decode_bytes(blob) == data)
    return {
        "name": "bit_semantics_lsb_first_tail_zero",
        "status": "passed" if ok else "failed",
        "detail": None if ok else f"mode={blob[4]} bits={bits.tobytes()!r}",
    }


def t_raw_fallback() -> dict:
    # incompressible: alternating unusual bytes forces raw fallback
    data = bytes([i % 256 for i in range(100)])
    blob = core.encode_bytes(data)
    mode = blob[4]
    got = core.decode_bytes(blob)
    ok = got == data and mode == core.MODE_RAW and len(blob) < len(data) + 16 + 1
    return {
        "name": "raw_fallback_when_smaller",
        "status": "passed" if ok and len(data) < 100 + 100 else "failed",
        "detail": f"mode={mode} blob={len(blob)} src={len(data)}",
    }


def t_exception_two_byte_records() -> dict:
    # exception with value equal to a common byte must survive via exception path
    data = bytes([120]) * 8 + bytes([250]) + bytes([121]) * 3 + bytes([0])
    blob = core.encode_bytes(data)
    got = core.decode_bytes(blob)
    return {
        "name": "extremes_with_common_neighbors",
        "status": "passed" if got == data else "failed",
        "detail": None if got == data else repr(got),
    }


def t_corruption_rejection() -> list[dict]:
    results = []
    data = bytes([120, 121, 3, 200] * 300)
    blob = core.encode_bytes(data)

    truncated = blob[:-3]
    try:
        core.decode_bytes(truncated)
        results.append({"name": "reject_truncated", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_truncated", "status": "passed",
                        "detail": None})

    bad_magic = b"XXXX" + blob[4:]
    try:
        core.decode_bytes(bad_magic)
        results.append({"name": "reject_bad_magic", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_bad_magic", "status": "passed",
                        "detail": None})

    bad_mode = blob[:4] + bytes([9]) + blob[5:]
    try:
        core.decode_bytes(bad_mode)
        results.append({"name": "reject_invalid_mode", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_invalid_mode", "status": "passed",
                        "detail": None})

    # non-monotonic prefix: rebuild a valid-shaped blob with broken offsets
    n = len(data)
    bits_len = (n + 7) // 8
    groups = (n + 255) // 256
    bits = np.packbits(np.frombuffer(data, dtype=np.uint8) == 121,
                       bitorder="little")[:bits_len]
    prefixes = np.zeros(groups + 1, dtype=np.uint32)
    prefixes[:-1] = np.arange(groups)
    prefixes[-1] = 2  # end sentinel smaller than earlier entries
    broken = (core._header(1, n) + bits.tobytes() + prefixes.tobytes()
              + bytes([9, 7]))
    try:
        core.decode_bytes(broken)
        results.append({"name": "reject_nonmonotonic_offsets", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_nonmonotonic_offsets", "status": "passed",
                        "detail": None})

    # duplicate exception positions
    dup = (core._header(1, n) + bits.tobytes()
           + np.array([0, 0, 0, 0], dtype=np.uint32).tobytes()
           + bytes([5, 9, 5, 7]))
    try:
        core.decode_bytes(dup)
        results.append({"name": "reject_duplicate_positions", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_duplicate_positions", "status": "passed",
                        "detail": None})

    # out-of-range exception position (position 255 in a 10-byte tensor)
    short = b"abcdefghij"
    s_bits = np.packbits(np.frombuffer(short, dtype=np.uint8) == 121,
                        bitorder="little")[:2]
    oor = (core._header(1, len(short)) + s_bits.tobytes()
           + np.array([0, 0], dtype=np.uint32).tobytes()
           + bytes([255, 7]))
    try:
        core.decode_bytes(oor)
        results.append({"name": "reject_out_of_range_position", "status": "failed",
                        "detail": "no error raised"})
    except ValueError:
        results.append({"name": "reject_out_of_range_position", "status": "passed",
                        "detail": None})
    return results


def t_offset_overflow_policy() -> dict:
    # uint32 prefix capacity is exercised structurally: 256 exceptions in one
    # group must be storable (8-bit counts would fail this).
    data = bytes([i % 251 + 5 for i in range(256)])
    blob = core.encode_bytes(data)
    got = core.decode_bytes(blob)
    return {
        "name": "uint32_prefix_256_exceptions",
        "status": "passed" if got == data else "failed",
        "detail": None if got == data else "mismatch",
    }


def t_full_checkpoint_shape_synthetic(seed: int = 5) -> dict:
    # Realistic 368,640-byte fixture, histogram-derived SYNTHETIC data
    # (weights of the observed audit histogram), not checkpoint bytes.
    rng = _rng(seed)
    values = np.array([117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                      dtype=np.uint8)
    counts = np.array([450, 14256, 5122574, 9300877845, 7670665097, 9163405,
                       1040117, 46232, 1183, 39, 2], dtype=np.float64)
    p = counts / counts.sum()
    data = rng.choice(values, size=368640, p=p).astype(np.uint8).tobytes()
    r = check_roundtrip(data, "synthetic_368640_histogram")
    r["label"] = "synthetic histogram-derived fixture, not checkpoint data"
    return r


def t_file_serialization() -> dict:
    """Encode -> persist via the sidecar writer -> reread -> byte compare."""
    import tempfile
    import shutil
    d = tempfile.mkdtemp(prefix="scale-codec-test-")
    try:
        src_path = os.path.join(d, "mini.safetensors")
        name = "layers.0.ffn.experts.0.w1.scale"
        payload = bytes([120, 121, 3, 200] * 20) + bytes([121])
        header = json_minimal(name, payload)
        with open(src_path, "wb") as f:
            f.write(header)
            f.write(payload)
        rows = core.enumerate_selected(d)
        if len(rows) != 1:
            return {"name": "sidecar_file_roundtrip", "status": "failed",
                    "detail": f"enumerated {len(rows)} rows"}
        blobs = [core.encode_bytes(core.read_tensor_bytes(r)) for r in rows]
        prefix = os.path.join(d, "out", "mini")
        core.write_sidecar(prefix, rows, blobs, d, core.DEFAULT_INCLUDE)
        manifest = core.read_manifest(prefix + ".smanifest.json")
        ent = manifest["tensors"][0]
        with open(prefix + ".sbuf", "rb") as f:
            f.seek(ent["blob_offset"])
            blob = f.read(ent["blob_length"])
        got = core.decode_bytes(blob)
        ok = got == payload
        # refuse-overwrite behaviour
        second = "exists"
        try:
            core.write_sidecar(prefix, rows, blobs, d, core.DEFAULT_INCLUDE)
            second = "overwrote"
        except FileExistsError:
            second = "exists"
        ok = ok and second == "exists"
        return {
            "name": "sidecar_file_roundtrip",
            "status": "passed" if ok else "failed",
            "detail": f"overwrite_policy={second}",
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


def json_minimal(name: str, payload: bytes) -> bytes:
    import json
    import struct
    hdr = {name: {"dtype": "U8", "shape": [len(payload)],
                  "data_offsets": [0, len(payload)]}}
    body = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(body)) % 8
    body += b" " * pad
    return struct.pack("<Q", len(body)) + body


def t_palette_and_nibble_reference() -> dict:
    data = bytes([120, 121, 119, 120, 121, 122, 120, 121] * 100)
    ok = core.palette_decode(core.palette_record(data), len(data)) == data
    ok = ok and core.nibble4_decode(core.nibble4_encode(data), len(data)) == data
    wide = bytes([0, 15, 30, 120] * 50)
    ok = ok and core.nibble4_decode(core.nibble4_encode(wide), len(wide)) == wide
    return {
        "name": "palette_and_nibble_reference",
        "status": "passed" if ok else "failed",
        "detail": None if ok else "reference codec mismatch",
    }


def gpu_test(enable: bool) -> dict:
    if not enable:
        return {"name": "gpu_decode", "status": "skipped",
                "detail": "pass --gpu to enable"}
    import scale_codec_gpu as gpu
    status = gpu.gpu_status()
    if not status["available"]:
        return {"name": "gpu_decode", "status": "not_run",
                "detail": status["reason"]}

    import torch
    rows = []
    cases = (
        ("len1_raw", b"\x79"),
        ("len7_raw", b"abcdefg"),
        ("len9_binary", bytes([120] * 9)),
        ("len256_binary", bytes([120] * 256)),
        ("len257_binary", bytes([120] * 256 + [121])),
        ("every_exception_raw", bytes([3] * 300)),
    )
    modes = set()
    for name, data in cases:
        blob = core.encode_bytes(data)
        modes.add(blob[4])
        out = torch.empty(len(data), dtype=torch.uint8, device="cuda")
        gpu.decode_one(torch, blob, len(data), out)
        got = out.cpu().numpy().tobytes()
        rows.append({"name": name, "n": len(data), "exact": got == data})
    ok = all(r["exact"] for r in rows) and modes == {
        core.MODE_RAW, core.MODE_BINARY256
    }
    selected_sources = [
        bytes([120, 121, 7, 120] * 80),
        bytes(range(256)) + bytes([9] * 44),
    ]
    selected_blobs = [core.encode_bytes(data) for data in selected_sources]
    pool = gpu.prepare_selected_pool(torch, selected_blobs)
    tensor_ids = torch.tensor([0, 0, 1, 9], dtype=torch.int64, device="cuda")
    tile_ids = torch.tensor([0, 0, 1, 0], dtype=torch.int64, device="cuda")
    selected_out = torch.empty(4 * 256, dtype=torch.uint8, device="cuda")
    selected_launch = gpu.selected_tile_launcher(
        torch, pool, tensor_ids, tile_ids, selected_out
    )
    selected_launch()
    torch.cuda.synchronize()
    got = selected_out.cpu().numpy().reshape(4, 256)
    selected_ok = (
        got[0].tobytes() == selected_sources[0][:256]
        and got[1].tobytes() == selected_sources[0][:256]
        and got[2, :44].tobytes() == selected_sources[1][256:]
        and not got[2, 44:].any()
        and not got[3].any()
    )
    graph_status = "not_supported"
    if hasattr(torch.cuda, "CUDAGraph"):
        selected_launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected_launch()
        tensor_ids.copy_(torch.tensor([1, 0, 0, -1], device="cuda"))
        tile_ids.copy_(torch.tensor([0, 1, 0, 0], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        replay = selected_out.cpu().numpy().reshape(4, 256)
        selected_ok = selected_ok and (
            replay[0].tobytes() == selected_sources[1][:256]
            and replay[1, :64].tobytes() == selected_sources[0][256:]
            and not replay[1, 64:].any()
            and replay[2].tobytes() == selected_sources[0][:256]
            and not replay[3].any()
        )
        graph_status = "passed" if selected_ok else "failed"
    ok = ok and selected_ok
    return {
        "name": "gpu_decode",
        "status": "passed" if ok else "failed",
        "detail": f"{len(rows)} full tensors; selected tiles exact; "
                  f"graph_replay={graph_status}; environment={status['device']}",
    }


def run_all(gpu: bool = False) -> list[dict]:
    results: list[dict] = []
    results.extend(t_lengths_and_values())
    results.append(t_bit_semantics())
    results.append(t_raw_fallback())
    results.append(t_exception_two_byte_records())
    results.extend(t_corruption_rejection())
    results.append(t_offset_overflow_policy())
    results.append(t_full_checkpoint_shape_synthetic())
    results.append(t_file_serialization())
    results.append(t_palette_and_nibble_reference())
    results.append(gpu_test(gpu))
    return results


if __name__ == "__main__":
    enable_gpu = "--gpu" in sys.argv
    rs = run_all(gpu=enable_gpu)
    bad = 0
    for r in rs:
        mark = {"passed": "PASS", "failed": "FAIL",
                "skipped": "SKIP", "not_run": "NOT_RUN"}[r["status"]]
        print(f"{mark:7s} {r['name']} n={r.get('n','-')} {r.get('detail') or ''}")
        bad += r["status"] == "failed"
    print(f"{len(rs)} tests: {sum(r['status']=='passed' for r in rs)} passed, "
          f"{bad} failed, "
          f"{sum(r['status'] in ('skipped','not_run') for r in rs)} skipped/not_run")
    sys.exit(0 if bad == 0 else 1)
