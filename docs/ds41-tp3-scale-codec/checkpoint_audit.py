#!/usr/bin/env python3
"""Read-only, CPU-only redundancy audit for a local safetensors checkpoint.

Python >=3.10; dependency: numpy. No torch, CUDA, pickle, or network calls.
This measures stored bytes. It does NOT modify a checkpoint or prove VRAM savings.
See checkpoint_audit_README.md for scopes, accounting, and limitations.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np

MIB = 1 << 20
GIB = 1 << 30

@dataclass(frozen=True)
class Tensor:
    file: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"Duplicate JSON key: {k}")
        result[k] = v
    return result


def inventory(path: Path) -> list[Tensor]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        index = path / 'model.safetensors.index.json'
        if index.exists():
            obj = json.loads(index.read_text(), object_pairs_hook=unique_object)
            files = [path / x for x in sorted(set(obj['weight_map'].values()))]
        else:
            indexes = list(path.glob('*.safetensors.index.json'))
            if indexes:
                raise ValueError('Nonstandard index found. Use one snapshot with a '
                                 'model.safetensors.index.json, or a single shard file.')
            files = sorted(path.glob('*.safetensors'))
    else:
        raise ValueError(f'Checkpoint not found: {path}')
    if not files:
        raise ValueError('No safetensors files found. Point at ONE checkpoint snapshot.')
    tensors, names = [], set()
    for file in files:
        file = file.resolve()
        size = file.stat().st_size
        with file.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Truncated header: {file}')
            hlen = struct.unpack('<Q', prefix)[0]
            if not 2 <= hlen <= min(100_000_000, size - 8):
                raise ValueError(f'Invalid safetensors header length: {file}')
            h = json.loads(f.read(hlen), object_pairs_hook=unique_object)
        ranges = []
        for name, m in h.items():
            if name == '__metadata__':
                continue
            if name in names:
                raise ValueError(f'Duplicate tensor name across shards: {name}; '
                                 'do not mix checkpoints/revisions.')
            a, b = m['data_offsets']
            shape = tuple(m['shape'])
            if (not all(type(v) is int for v in (a, b)) or
                not 0 <= a <= b <= size - 8 - hlen or
                not all(type(v) is int and v >= 0 for v in shape)):
                raise ValueError(f'Invalid tensor metadata: {name}')
            names.add(name)
            tensors.append(Tensor(str(file), name, m['dtype'], shape,
                                  8 + hlen + a, b - a))
            if b > a:
                ranges.append((a, b))
        ranges.sort()
        if any(a1 < b0 for (_, b0), (a1, _) in zip(ranges, ranges[1:])):
            raise ValueError(f'Overlapping tensor ranges: {file}')
    return sorted(tensors, key=lambda t: (t.file, t.offset, t.name))


class Reader:
    """Bounded file-descriptor cache; pread leaves independent reads race-free."""
    def __init__(self):
        self.fds = collections.OrderedDict()

    def read(self, t: Tensor, offset: int, n: int) -> bytes:
        if offset < 0 or n < 0 or offset + n > t.nbytes:
            raise ValueError('Read outside tensor')
        fd = self.fds.pop(t.file, None)
        if fd is None:
            fd = os.open(t.file, os.O_RDONLY)
        self.fds[t.file] = fd
        if len(self.fds) > 16:
            _, old = self.fds.popitem(last=False)
            os.close(old)
        parts, done = [], 0
        while done < n:
            b = os.pread(fd, n - done, t.offset + offset + done)
            if not b:
                raise OSError(f'Short read: {t.file} / {t.name}')
            parts.append(b)
            done += len(b)
        return b''.join(parts)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class Progress:
    def __init__(self, total: int):
        self.total, self.done, self.last = total, 0, 0.0

    def add(self, n: int):
        self.done += n
        now = time.monotonic()
        if now - self.last >= 5 or self.done == self.total:
            print(f'  Read {self.done/GIB:.3f} / {self.total/GIB:.3f} GiB',
                  file=sys.stderr, flush=True)
            self.last = now


def category(t: Tensor) -> str:
    n = t.name.lower()
    if 'engram' in n:
        return 'engram'
    if re.search(r'(^|\.)(mtp|draft|drafter|dspark)(\.|_)', n):
        return 'draft_or_mtp'
    if re.search(r'(^|\.)(vision|visual|aligner)(\.|_)', n):
        return 'vision_or_aligner'
    if re.search(r'(^|\.)experts\.', n):
        return 'routed_expert_scales' if 'scale' in n else 'routed_expert_other'
    return 'other'


def run_inventory(ts: list[Tensor]) -> dict:
    buckets = {}
    for t in ts:
        key = (category(t), t.dtype)
        b = buckets.setdefault(key, {'category': key[0], 'dtype': key[1],
                                     'tensors': 0, 'bytes': 0, 'examples': []})
        b['tensors'] += 1
        b['bytes'] += t.nbytes
        if len(b['examples']) < 3:
            b['examples'].append(t.name)
    rows = list(buckets.values())
    for b in sorted(rows, key=lambda x: -x['bytes']):
        print(f"{b['category']:24s} {b['dtype']:12s} {b['tensors']:8d} tensors  "
              f"{b['bytes']/GIB:10.3f} GiB")
        for e in b['examples']:
            print('    ' + e)
    return {'buckets': rows, 'tensors': [asdict(t) for t in ts],
            'note': 'Categories are name-based hints, not verified residency or quantization.'}


def palette_record(data: bytes) -> bytes:
    """Reference tile encoder. Raw if no smaller; record length tags raw mode."""
    x = np.frombuffer(data, np.uint8)
    if not len(x):
        return data
    palette, ids = np.unique(x, return_inverse=True)
    k = len(palette)
    bits = (k - 1).bit_length()
    size = 1 + k + (len(x) * bits + 7) // 8
    if size >= len(x) or k == 256:
        return data
    if bits:
        bitstream = ((ids[:, None] >> np.arange(bits)) & 1).astype(np.uint8)
        packed = np.packbits(bitstream.reshape(-1), bitorder='little').tobytes()
    else:
        packed = b''
    return bytes([k]) + palette.tobytes() + packed


def palette_decode(record: bytes, original_n: int) -> bytes:
    if len(record) == original_n:
        return record
    k = record[0]
    bits = (k - 1).bit_length()
    palette = np.frombuffer(record[1:1+k], np.uint8)
    if not bits:
        return np.full(original_n, palette[0], np.uint8).tobytes()
    unpacked = np.unpackbits(np.frombuffer(record[1+k:], np.uint8), bitorder='little')
    b = unpacked[:original_n*bits].reshape(original_n, bits)
    ids = (b.astype(np.uint16) << np.arange(bits)).sum(axis=1)
    return palette[ids].tobytes()


def run_scales(ts: list[Tensor], r: Reader, group: int) -> dict:
    accepted = {'U8', 'I8', 'F8_E8M0', 'F8_E8M0FNU'}
    bad = [t for t in ts if t.dtype not in accepted or math.prod(t.shape) != t.nbytes]
    if bad:
        examples = ', '.join(f'{t.name} ({t.dtype})' for t in bad[:5])
        raise ValueError('Scale audit requires one-byte entries. Refine --include/--exclude. '
                         f'Unsupported selection: {examples}')
    histogram = np.zeros(256, np.int64)
    khist = np.zeros(257, np.int64)
    results = []
    original = sum(t.nbytes for t in ts)
    p = Progress(original)
    step = max(group, (MIB // group) * group)
    bits = np.array([0] + [(k - 1).bit_length() for k in range(1, 257)], np.int64)
    candidate = 1 + np.arange(257) + ((group * bits + 7) // 8)
    costs = np.minimum(candidate, group)
    verified = 0
    for t in ts:
        body, tiles = 0, 0
        local_hist = np.zeros(257, np.int64)
        for pos in range(0, t.nbytes, step):
            data = r.read(t, pos, min(step, t.nbytes-pos))
            x = np.frombuffer(data, np.uint8)
            histogram += np.bincount(x, minlength=256)
            full = len(x) // group
            if full:
                a = np.sort(x[:full*group].reshape(full, group), axis=1)
                k = 1 + np.count_nonzero(a[:, 1:] != a[:, :-1], axis=1)
                counts = np.bincount(k, minlength=257)
                local_hist += counts
                body += int(np.dot(counts, costs))
                tiles += full
            tail = data[full*group:]
            if tail:
                body += len(palette_record(tail))
                tiles += 1
            # Bounded smoke tests only; full scan counts are exact, not sampled.
            if verified < 128 and data:
                test = data[:group]
                encoded = palette_record(test)
                if palette_decode(encoded, len(test)) != test:
                    raise AssertionError('Palette round-trip failed')
                verified += 1
            p.add(len(data))
        # 16-byte per-tensor header and 64-bit offset table, including sentinel.
        candidate_bytes = body + 16 + 8 * (tiles + 1)
        encoded_bytes = min(t.nbytes, candidate_bytes)  # whole-tensor raw fallback
        khist += local_hist
        results.append({'name': t.name, 'raw_bytes': t.nbytes,
                        'modelled_encoded_bytes': encoded_bytes,
                        'net_saving_bytes': t.nbytes-encoded_bytes,
                        'tiles': tiles,
                        'uses_compressed_representation': candidate_bytes < t.nbytes})
    encoded = sum(x['modelled_encoded_bytes'] for x in results)
    prob = histogram[histogram > 0] / max(original, 1)
    entropy = float(-(prob * np.log2(prob)).sum())
    print(f'Scale bytes: {original/GIB:.6f} GiB')
    print(f'Modelled palette representation: {encoded/GIB:.6f} GiB')
    print(f'Modelled net saving: {(original-encoded)/GIB:.6f} GiB '
          f'({100*(original-encoded)/max(original,1):.2f}%)')
    print(f'Marginal byte entropy: {entropy:.4f} bits/entry (not a GPU codec result)')
    return {'scan': 'full selected scale tensors', 'raw_bytes': original,
            'modelled_encoded_bytes': encoded, 'net_saving_bytes': original-encoded,
            'group_entries': group, 'histogram_256': histogram.tolist(),
            'unique_values_per_full_group': khist.tolist(),
            'marginal_entropy_bits_per_byte': entropy,
            'roundtrip_smoke_tests': verified, 'tensors': results,
            'layout': '16-byte tensor header; uint64 offsets plus sentinel; each tile '
                      'raw or [uint8 palette_length, palette bytes, bit-packed indices]. '
                      'Whole-tensor raw fallback; existing tensor directory reused.',
            'excludes': 'GPU alignment, kernel layout changes, GPU scratch and integration metadata. '
                        'No encoded checkpoint is written. No inference kernels are tested.'}


def run_local_blocks(ts: list[Tensor], r: Reader, args) -> dict:
    window = args.window_mib * MIB
    sizes = sorted(set(args.block_bytes))
    if any(b <= 0 or window % b for b in sizes):
        raise ValueError('Each block size must be positive and divide --window-mib bytes')
    cumulative, total = [], 0
    for t in ts:
        total += (t.nbytes + window - 1) // window
        cumulative.append(total)
    wanted = min(total, max(1, args.sample_mib * MIB // window))
    indices = range(total) if args.full else sorted(random.Random(args.seed).sample(range(total), wanted))
    stats = {b: dict(block_bytes=b, examined_bytes=0, full_blocks=0,
                    duplicate_occurrences=0, zero_blocks=0, gross_duplicate_bytes=0,
                    modelled_encoded_bytes=0, net_saving_bytes=0) for b in sizes}
    nominal = sum(t.nbytes for t in ts) if args.full else len(indices)*window
    p, manifests = Progress(nominal), []
    per_tensor = {b: {} for b in sizes}
    for wi in indices:
        ti = bisect.bisect_right(cumulative, wi)
        previous = cumulative[ti-1] if ti else 0
        offset = (wi-previous)*window
        t = ts[ti]
        data = r.read(t, offset, min(window, t.nbytes-offset))
        if not args.full:
            manifests.append({'tensor': t.name, 'offset': offset, 'bytes': len(data)})
        for b in sizes:
            n, tail = divmod(len(data), b)
            records = np.frombuffer(data[:n*b], dtype=f'V{b}')
            u = int(np.unique(records).size) if n else 0  # exact bytes, no hashing
            index_bytes = ((u-1).bit_length()+7)//8 if u > 1 else 0
            # Raw record, or 16-byte dictionary header + dictionary + indices + tail.
            # All records also require a tensor-level offset directory, counted below.
            candidate = 16 + u*b + n*index_bytes + tail
            stored = min(len(data), candidate)
            accounting = per_tensor[b].setdefault(ti, [0, 0, 0])
            accounting[0] += len(data)
            accounting[1] += stored
            accounting[2] += 1
            s = stats[b]
            s['examined_bytes'] += len(data)
            s['full_blocks'] += n
            s['duplicate_occurrences'] += n-u
            s['zero_blocks'] += int(np.count_nonzero(records == np.void(bytes(b)))) if n else 0
            s['gross_duplicate_bytes'] += (n-u)*b
        p.add(len(data))
    for b, s in stats.items():
        for raw, body, windows in per_tensor[b].values():
            stored = min(raw, body + 16 + 8*(windows+1))
            s['modelled_encoded_bytes'] += stored
            s['net_saving_bytes'] += raw-stored
        print(f"{b:7d}-byte blocks: gross duplicate bytes "
              f"{s['gross_duplicate_bytes']/MIB:.3f} MiB; modelled net saving "
              f"{s['net_saving_bytes']/MIB:.3f} MiB / {s['examined_bytes']/MIB:.3f} MiB")
    return {'scan': 'full' if args.full else 'sample',
            'scope': 'independent tensor-aligned windows; no cross-window deduplication',
            'window_bytes': window, 'seed': args.seed, 'results': list(stats.values()),
            'sample_manifest': manifests,
            'note': 'Exact byte comparisons within each window. Do not extrapolate a sample '
                    'duplicate fraction globally. Even --full misses cross-window repetition. '
                    'Sizes are for separate alternative dictionaries; never add their savings. '
                    'Accounting includes a 16-byte tensor header, uint64 window offsets plus '
                    'sentinel, 16-byte compressed-window headers, and byte-width indices. '
                    'Raw fallback per window and per tensor. In sample mode, sizes describe '
                    'only the selected windows as a logical subset, not the full checkpoint. '
                    'Existing tensor metadata reused; no GPU alignment/scratch included.'}


def equal_tensors(a: Tensor, b: Tensor, r: Reader) -> bool:
    return all(r.read(a, p, min(4*MIB, a.nbytes-p)) ==
               r.read(b, p, min(4*MIB, b.nbytes-p))
               for p in range(0, a.nbytes, 4*MIB))


def run_tensors(ts: list[Tensor], r: Reader) -> dict:
    groups = collections.Counter((t.dtype, t.shape, t.nbytes) for t in ts if t.nbytes)
    selected = [t for t in ts if t.nbytes and groups[(t.dtype,t.shape,t.nbytes)] > 1]
    seen, matches = {}, []
    p = Progress(sum(t.nbytes for t in selected))
    for t in selected:
        h = hashlib.sha256()
        for pos in range(0, t.nbytes, 4*MIB):
            b = r.read(t, pos, min(4*MIB, t.nbytes-pos))
            h.update(b)
            p.add(len(b))
        key = (t.dtype, t.shape, t.nbytes, h.digest())
        representatives = seen.setdefault(key, [])
        for rep in representatives:
            if equal_tensors(t, rep, r):
                matches.append({'tensor': t.name, 'same_as': rep.name,
                                'bytes': t.nbytes, 'sha256': h.hexdigest()})
                break
        else:
            representatives.append(t)
    saving = sum(m['bytes'] for m in matches)
    print(f'Whole-tensor exact duplicates: {len(matches)}; gross {saving/GIB:.6f} GiB')
    return {'scope': 'same dtype and shape, whole tensors, all selected shards',
            'hashed_bytes': sum(t.nbytes for t in selected), 'matches': matches,
            'gross_duplicate_bytes': saving,
            'verification': 'SHA-256 candidates verified by full byte comparison',
            'note': 'Checkpoint duplication only. Existing runtime sharing can make VRAM '
                    'saving zero. Sharing weight payloads does not permit discarding distinct scales.'}


def run_global_blocks(ts: list[Tensor], r: Reader, args) -> dict:
    b = args.block_bytes
    if b < 4096:
        raise ValueError('Global scan requires block size >=4096; use local-blocks for small blocks')
    count = sum(t.nbytes//b for t in ts)
    if count > 10_000_000 and not args.allow_large:
        raise ValueError(f'{count:,} blocks exceeds guardrail. Use a larger block or --allow-large. '
                         'SQLite scratch and scan work scale with block count.')
    database = Path(args.database)
    if database.exists():
        raise ValueError('SQLite index already exists; use a new --database path')
    database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    try:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA cache_size=-65536')
        db.execute('CREATE TABLE blocks (digest BLOB, tensor INTEGER, off INTEGER, '
                   'PRIMARY KEY(digest,tensor,off)) WITHOUT ROWID')
        db.execute('CREATE TABLE tensors (id INTEGER PRIMARY KEY, metadata TEXT)')
        db.executemany('INSERT INTO tensors VALUES (?,?)',
                       ((i,json.dumps(asdict(t))) for i,t in enumerate(ts)))
        db.commit()
        duplicates = unique = zeros = 0
        examples = []
        p = Progress(count*b)
        zero_block = bytes(b)
        step = max(b, (4*MIB//b)*b)
        for ti, t in enumerate(ts):
            full = t.nbytes//b*b
            for start in range(0, full, step):
                data = r.read(t, start, min(step, full-start))
                for off in range(0, len(data), b):
                    block = data[off:off+b]
                    digest = hashlib.sha256(block).digest()
                    found = None
                    for ri, ro in db.execute('SELECT tensor,off FROM blocks WHERE digest=?', (digest,)):
                        if r.read(ts[ri], ro, b) == block:
                            found = (ri, ro)
                            break
                    zeros += block == zero_block
                    if found is None:
                        db.execute('INSERT INTO blocks VALUES (?,?,?)', (digest,ti,start+off))
                        unique += 1
                    else:
                        duplicates += 1
                        if len(examples) < 30:
                            examples.append({'tensor': t.name, 'offset': start+off,
                                             'same_as': ts[found[0]].name,
                                             'same_as_offset': found[1]})
                db.commit()
                p.add(len(data))
        width = ((unique-1).bit_length()+7)//8 if unique > 1 else 0
        payload_saving = duplicates*b
        index_bytes = count*width
        net = payload_saving-index_bytes
        print(f'Global {b}-byte blocks: {count:,}; duplicates {duplicates:,}; '
              f'gross {payload_saving/GIB:.6f} GiB')
        print(f'Flat dictionary after {width}-byte indices: {net/GIB:.6f} GiB saving '
              '(signed; excludes additional GPU layout/scratch)')
        return {'scope': 'all selected tensors, tensor-relative aligned full blocks only',
                'block_bytes': b, 'full_blocks': count, 'unique_blocks': unique,
                'duplicate_occurrences': duplicates, 'zero_blocks': zeros,
                'raw_full_block_bytes': count*b, 'gross_duplicate_bytes': payload_saving,
                'flat_dictionary_index_bytes_per_block': width,
                'flat_dictionary_index_bytes': index_bytes,
                'net_saving_bytes_with_uint32_indices': payload_saving-count*4 if unique <= 2**32 else None,
                'net_saving_bytes_with_uint64_indices': payload_saving-count*8,
                'flat_dictionary_net_saving_bytes_before_runtime_overhead': net,
                'unexamined_tail_bytes': sum(t.nbytes % b for t in ts),
                'examples': examples, 'sqlite_index': str(database.resolve()),
                'verification': 'SHA-256 candidates verified by exact block byte comparison',
                'note': 'Not a CUDA page-sharing measurement. Excludes unaligned matches, '
                        'cross-tensor-boundary blocks, runtime repacking, TP ownership, '
                        'metadata beyond flat indices, and GPU scratch. Do not add savings '
                        'from different block sizes or whole-tensor scans.'}
    finally:
        db.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('inventory','scales','local-blocks','tensors','global-blocks'):
        q = sub.add_parser(name)
        q.add_argument('checkpoint', type=Path)
        q.add_argument('--include', default='.*', help='regex on actual checkpoint tensor names')
        q.add_argument('--exclude', default=None, help='regex exclusion, applied after include')
        q.add_argument('--output', type=Path, required=True, help='new JSON report path')
        if name == 'scales':
            q.add_argument('--group', type=int, default=256)
        if name == 'local-blocks':
            q.add_argument('--block-bytes', nargs='+', type=int, default=[16,64,256,4096])
            q.add_argument('--window-mib', type=int, default=4)
            q.add_argument('--sample-mib', type=int, default=512)
            q.add_argument('--seed', type=int, default=20260924)
            q.add_argument('--full', action='store_true')
        if name == 'global-blocks':
            q.add_argument('--block-bytes', type=int, default=65536)
            q.add_argument('--database', required=True)
            q.add_argument('--allow-large', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError(f'Report exists: {args.output}; choose a new path')
    if args.command == 'scales' and not 1 <= args.group <= 65536:
        raise ValueError('--group must be 1..65536')
    if args.command == 'local-blocks' and (args.window_mib < 1 or args.sample_mib < 1):
        raise ValueError('Window and sample MiB must be positive integers')
    inc, exc = re.compile(args.include), re.compile(args.exclude) if args.exclude else None
    ts = [t for t in inventory(args.checkpoint) if inc.search(t.name)
          and not (exc and exc.search(t.name))]
    if not ts:
        raise ValueError('No tensor names matched; inspect inventory first')
    source_stats = {t.file: (os.stat(t.file).st_size, os.stat(t.file).st_mtime_ns) for t in ts}
    print(f'{len(ts):,} selected tensors; {sum(t.nbytes for t in ts)/GIB:.3f} GiB stored bytes',
          file=sys.stderr)
    reader = Reader()
    try:
        if args.command == 'inventory':
            report = run_inventory(ts)
        elif args.command == 'scales':
            report = run_scales(ts,reader,args.group)
        elif args.command == 'local-blocks':
            report = run_local_blocks(ts,reader,args)
        elif args.command == 'tensors':
            report = run_tensors(ts,reader)
        else:
            report = run_global_blocks(ts,reader,args)
    finally:
        reader.close()
    for file, old in source_stats.items():
        st = os.stat(file)
        if (st.st_size, st.st_mtime_ns) != old:
            raise OSError(f'Source changed during scan: {file}; results discarded')
    report.update(command=args.command, include=args.include, exclude=args.exclude,
                  selected_tensors=len(ts), selected_bytes=sum(t.nbytes for t in ts),
                  source_files=[{'path':p,'bytes':v[0],'mtime_ns':v[1]}
                                for p,v in source_stats.items()],
                  warning='Offline byte audit, NOT measured recoverable VRAM or GPU performance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(f'Report: {args.output}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; no complete report written.', file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, json.JSONDecodeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
