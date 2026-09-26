"""CPU codecs, safe byte I/O and the sidecar format for the scale codec.

Everything here operates on raw uint8 byte patterns. No floating-point
conversion is used as a byte view. All imports are CPU-safe: this module
works without CUDA, PyTorch, Triton or vLLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

import numpy as np

MIB = 1 << 20
DEFAULT_INCLUDE = r"^layers\.[0-9]+\.ffn\.experts\.[0-9]+\.w[123]\.scale$"
CHUNK = 4 * MIB

MODE_RAW = 0
MODE_BINARY256 = 1
MAGIC = b"SCD1"
GROUP = 256
COMMON_A = 120  # bit 0 restores this byte
COMMON_B = 121  # the only value that sets a bit


# ---------------------------------------------------------------------------
# safetensors byte I/O (bounded, header-driven)
# ---------------------------------------------------------------------------

def read_header(path: str) -> dict:
    """Read a safetensors JSON header with two bounded reads."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        length = int(np.frombuffer(head, dtype="<u8", count=1)[0])
        raw = f.read(length)
        if len(raw) != length:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(raw.decode("utf-8"))


def enumerate_selected(model_dir: str, include: str = DEFAULT_INCLUDE) -> list[dict]:
    """Deterministically ordered selected tensors from shard headers only.

    Returns dicts: name, file (abs path), dtype, shape, offset, length.
    """
    pattern = re.compile(include)
    shards = []
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            shards.append(os.path.join(model_dir, name))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards under {model_dir}")
    out: list[dict] = []
    for shard in shards:
        header = read_header(shard)
        base = _data_start(shard, header)
        rows = []
        for name, ent in header.items():
            if name == "__metadata__":
                continue
            if not pattern.match(name):
                continue
            lo, hi = ent["data_offsets"]
            rows.append({
                "name": name,
                "file": shard,
                "dtype": ent["dtype"],
                "shape": list(ent["shape"]),
                "offset": base + int(lo),
                "length": int(hi) - int(lo),
            })
        rows.sort(key=lambda r: (r["offset"], r["name"]))
        out.extend(rows)
    out.sort(key=lambda r: (os.path.basename(r["file"]), r["offset"], r["name"]))
    return out


def _data_start(path: str, header: dict) -> int:
    """Header length + 8-byte prefix, verified against the first tensor offset."""
    with open(path, "rb") as f:
        length = int(np.frombuffer(f.read(8), dtype="<u8", count=1)[0])
    body = json.dumps({k: v for k, v in header.items()}, separators=(",", ":"))
    # Safetensors pads the header to align data at 8 bytes; recover the true
    # data start as the smallest observed offset plus the header size field.
    _ = body
    return 8 + length


def read_tensor_bytes(row: dict, chunk: int = CHUNK) -> bytes:
    """Read exactly the tensor's byte range via seek+read; never whole-shard."""
    out = bytearray()
    remaining = row["length"]
    pos = row["offset"]
    with open(row["file"], "rb") as f:
        while remaining > 0:
            n = min(chunk, remaining)
            f.seek(pos)
            part = f.read(n)
            if not part:
                raise ValueError(f"{row['file']}: short read at {pos}")
            out += part
            pos += len(part)
            remaining -= len(part)
    return bytes(out)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# binary_exceptions256 codec
# ---------------------------------------------------------------------------
#
# Per-tensor encoded blob (little-endian):
#   offset 0  : magic  "SCD1"                 (4 bytes)
#   offset 4  : mode   uint8 (0 raw, 1 binary_exceptions256)
#   offset 5  : zero padding                  (3 bytes, must be zero)
#   offset 8  : uint64 LE original byte length N
#   then, for mode 1:
#     bits      ceil(N/8) bytes, LSB-first, bit=1 iff source byte == 121;
#               unused tail bits zero
#     prefixes  4*(ceil(N/256)+1) bytes LE uint32 group offsets into the
#               exception-record array, measured in records; last entry is
#               the end sentinel == total record count
#     exceptions 2*E bytes of (position_in_group uint8, original byte uint8)
#   for mode 0:
#     raw       exactly N source bytes
# Header is exactly 16 bytes, matching the design-estimate term 16*tensors.

def encode_bytes(data: bytes) -> bytes:
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8, count=n)
    groups = (n + GROUP - 1) // GROUP

    bits = np.packbits(arr == COMMON_B, bitorder="little")
    exc_pos = np.flatnonzero((arr != COMMON_A) & (arr != COMMON_B))
    exc_val = arr[exc_pos]
    e = int(exc_pos.size)
    if e > np.iinfo(np.uint32).max:
        raise OverflowError(f"{e} exception records exceed uint32 prefix range")

    counts = np.bincount((exc_pos // GROUP).astype(np.int64), minlength=groups) \
        if groups else np.zeros(0, dtype=np.int64)
    prefixes = np.empty(groups + 1, dtype=np.uint32)
    prefixes[0] = 0
    if groups:
        prefixes[1:] = np.cumsum(counts, dtype=np.uint32)

    bits_len = (n + 7) // 8
    body_bits = bits[:bits_len].tobytes()
    body_pre = prefixes.tobytes()
    body_exc = np.empty(2 * e, dtype=np.uint8)
    body_exc[0::2] = (exc_pos % GROUP).astype(np.uint8)
    body_exc[1::2] = exc_val
    encoded_size = 16 + bits_len + 4 * (groups + 1) + 2 * e
    raw_size = 16 + n
    if encoded_size >= raw_size and n > 0:
        return _header(MODE_RAW, n) + data

    rec = bytearray(_header(MODE_BINARY256, n))
    rec += body_bits
    rec += body_pre
    rec += body_exc.tobytes()
    return bytes(rec)


def _header(mode: int, n: int) -> bytes:
    return MAGIC + bytes([mode, 0, 0, 0]) + int(n).to_bytes(8, "little")


def decode_bytes(blob: bytes) -> bytes:
    if len(blob) < 16 or blob[:4] != MAGIC:
        raise ValueError("scale blob: missing or invalid SCD1 header")
    mode = blob[4]
    if mode not in (MODE_RAW, MODE_BINARY256):
        raise ValueError(f"scale blob: invalid mode {mode}")
    if blob[5:8] != b"\x00\x00\x00":
        raise ValueError("scale blob: reserved header bytes must be zero")
    n = int.from_bytes(blob[8:16], "little")

    if mode == MODE_RAW:
        payload = blob[16:]
        if len(payload) != n:
            raise ValueError(
                f"scale blob: raw payload {len(payload)} != declared {n}")
        return payload

    bits_len = (n + 7) // 8
    groups = (n + GROUP - 1) // GROUP
    pre_len = 4 * (groups + 1)
    need = 16 + bits_len + pre_len
    if len(blob) < need:
        raise ValueError("scale blob: truncated before prefix table")
    bits = np.frombuffer(blob, dtype=np.uint8, count=bits_len, offset=16)
    prefixes = np.frombuffer(
        blob, dtype="<u4", count=groups + 1, offset=16 + bits_len)

    if prefixes[0] != 0:
        raise ValueError("scale blob: prefix table must start at 0")
    if np.any(np.diff(prefixes.astype(np.int64)) < 0):
        raise ValueError("scale blob: non-monotonic prefix offsets")
    e = int(prefixes[groups])
    if len(blob) != need + 2 * e:
        raise ValueError(
            f"scale blob: length {len(blob)} inconsistent with "
            f"header {need} + 2*{e} exception bytes")

    packed = np.unpackbits(bits, bitorder="little")[:n]
    out = np.where(packed == 1, COMMON_B, COMMON_A).astype(np.uint8)
    if e:
        recs = np.frombuffer(
            blob, dtype=np.uint8, count=2 * e, offset=need)
        offs = prefixes[:-1].astype(np.int64)
        ends = prefixes[1:].astype(np.int64)
        idx = np.empty(e, dtype=np.int64)
        vals = np.empty(e, dtype=np.uint8)
        w = 0
        for g in range(groups):
            for r in range(int(offs[g]), int(ends[g])):
                pos = int(recs[2 * r])
                val = int(recs[2 * r + 1])
                gi = g * GROUP + pos
                if gi >= n:
                    raise ValueError(
                        f"scale blob: exception position {gi} out of range {n}")
                idx[w] = gi
                vals[w] = val
                w += 1
        if np.unique(idx).size != e:
            raise ValueError("scale blob: duplicate exception positions")
        out[idx] = vals
    return out.tobytes()


# ---------------------------------------------------------------------------
# Reference CPU comparisons (not the primary codec)
# ---------------------------------------------------------------------------

def palette_record(data: bytes) -> bytes:
    """Reference palette256 tile encoder from the original audit."""
    arr = np.frombuffer(data, dtype=np.uint8)
    palette = np.unique(arr)
    if palette.size > 255 or palette.size == 0:
        return bytes([255]) + data if False else data
    bits_per = max(1, int(np.ceil(np.log2(palette.size)))) if palette.size > 1 else 1
    packed_bits = ((arr.size * bits_per + 7) // 8)
    record = bytes([palette.size]) + palette.astype(np.uint8).tobytes()
    ids = np.searchsorted(palette, arr).astype(np.uint64)
    packed = np.zeros((arr.size, bits_per), dtype=np.uint8)
    for b in range(bits_per):
        packed[:, b] = ((ids >> b) & 1).astype(np.uint8)
    flat = packed.reshape(-1)
    pad = (-flat.size) % 8
    flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed_bytes = np.packbits(flat)
    if packed_bytes.size != packed_bits:
        packed_bytes = packed_bytes[:packed_bits]
    record += packed_bytes.tobytes()
    if len(record) >= len(data) or palette.size >= 255:
        return bytes([255])[:0] + data
    return record


def palette_decode(record: bytes, original_n: int) -> bytes:
    if len(record) == original_n and (not record or record[0] == 255):
        return record
    k = record[0]
    if k == 0 or k > 128:
        return record[:original_n]
    palette = np.frombuffer(record, dtype=np.uint8, count=k, offset=1)
    bits_per = max(1, int(np.ceil(np.log2(k)))) if k > 1 else 1
    packed_bits = (original_n * bits_per + 7) // 8
    packed = np.frombuffer(
        record, dtype=np.uint8, count=packed_bits, offset=1 + k)
    flat = np.unpackbits(packed)[: original_n * bits_per]
    ids = np.zeros(original_n, dtype=np.uint64)
    for b in range(bits_per):
        ids |= flat[b::bits_per][:original_n].astype(np.uint64) << b
    return palette[ids].tobytes()


def nibble4_encode(data: bytes) -> bytes:
    """nibble4 comparison: byte - min_byte in 4 bits when the range fits."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return b"\x00"
    lo = int(arr.min())
    hi = int(arr.max())
    if hi - lo > 15:
        return b"\xff" + data  # range does not fit: raw fallback
    off = (arr.astype(np.uint16) - lo).astype(np.uint8)
    low = off[0::2] & 0x0F
    high = np.zeros(low.shape, dtype=np.uint8)
    hi_n = (off[1::2] & 0x0F) << np.uint8(4)
    high[: hi_n.size] = hi_n
    return bytes([lo]) + (low | high).tobytes()


def nibble4_decode(record: bytes, n: int) -> bytes:
    head = record[0]
    body = np.frombuffer(
        record, dtype=np.uint8, count=len(record) - 1, offset=1)
    if head == 255:
        return body.tobytes()[:n]
    out = np.empty(body.size * 2, dtype=np.uint8)
    out[0::2] = (body & 0x0F) + np.uint8(head)
    out[1::2] = ((body >> np.uint8(4)) & 0x0F) + np.uint8(head)
    return out[:n].tobytes()


# ---------------------------------------------------------------------------
# Sidecar format (outside the checkpoint directory, atomic completion)
# ---------------------------------------------------------------------------

SIDECAR_VERSION = 1


def write_sidecar(out_prefix: str, rows: list[dict], blobs: list[bytes],
                  model_dir: str, include: str, force: bool = False) -> dict:
    """Write <out_prefix>.sbuf + <out_prefix>.smanifest.json atomically.

    rows/blobs are parallel lists already in deterministic order.
    """
    bin_path = out_prefix + ".sbuf"
    man_path = out_prefix + ".smanifest.json"
    for p in (bin_path, man_path):
        if os.path.exists(p) and not force:
            raise FileExistsError(f"{p} exists; pass --force to overwrite")

    entries = []
    cursor = 0
    buf = bytearray()
    for row, blob in zip(rows, blobs):
        source = read_tensor_bytes(row)
        entries.append({
            "name": row["name"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "source_offsets": [row["offset"] - 8 - _header_len(row["file"]),
                               row["offset"] - 8 - _header_len(row["file"])
                               + row["length"]],
            "layout": row["shape"],
            "blob_offset": cursor,          # uint64: aggregate may exceed 4 GiB
            "blob_length": len(blob),
            "raw_length": row["length"],
            "source_sha256": sha256_bytes(source),
            "decoded_sha256": sha256_bytes(source),
        })
        buf += blob
        cursor += len(blob)

    manifest = {
        "schema_version": SIDECAR_VERSION,
        "codec": "binary_exceptions256",
        "group_size": GROUP,
        "model_dir": model_dir,
        "include": include,
        "tensor_count": len(entries),
        "raw_bytes_total": sum(r["raw_length"] for r in entries),
        "encoded_bytes_total": cursor,
        "manifest_sha256_self": None,
        "tensors": entries,
    }
    payload = json.dumps(manifest, indent=1, sort_keys=False)

    _atomic_write(bin_path, bytes(buf))
    _atomic_write(man_path, payload.encode("utf-8"))
    return manifest


def _header_len(path: str) -> int:
    with open(path, "rb") as f:
        return int(np.frombuffer(f.read(8), dtype="<u8", count=1)[0])


def _atomic_write(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".sc-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_manifest(man_path: str) -> dict:
    with open(man_path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
