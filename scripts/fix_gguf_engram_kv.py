#!/usr/bin/env python3
"""Repair the engram metadata of a DeepSeek-V4.1 GGUF that was written before the converter fix.

Two things went wrong in those files and both live in the header of the first shard:

  1. the four engram keys carry a hardcoded `deepseek4.` prefix, so a `deepseek41` model looks for
     `deepseek41.engram.head_count` and finds nothing
  2. the five constants the hash actually needs are absent, because gguf-py's add_array() maps every
     Python int to INT32, the 47 bit multipliers raised struct.error, and a broad except downgraded
     that to a warning

Tensor data is untouched. Existing key/value pairs are re-emitted byte for byte, apart from the
four that get renamed, so nothing this script does not understand can be corrupted by it.

  python fix_gguf_engram_kv.py shard1.gguf out.gguf --model-dir /path/to/DeepSeek-V4.1-Flash
"""
import argparse
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"

# value type tags
T_UINT32 = 4
T_INT32  = 5
T_STRING = 8
T_ARRAY  = 9
T_UINT64 = 10

FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    i = 41
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def build_token_map(model_dir):
    """Case folded, accent stripped vocabulary, exactly as the reference builds it."""
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    sentinel = ""  # private use char, so a lone space survives Strip()
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tok.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tok)
    for tid in range(len(tok)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


def build_constants(model_dir, layer_ids, max_ngram, n_heads, vocab_size, pad_raw):
    import numpy as np

    token_map, compressed = build_token_map(model_dir)

    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed) // 2)
    mults = []
    for lid in layer_ids:
        rng = np.random.default_rng(10007 * lid)
        mults.extend(int(v) * 2 + 1 for v in rng.integers(0, bound, size=(max_ngram,), dtype=np.int64))

    primes, seen = [], set()
    for _ in layer_ids:
        for _ in range(max_ngram - 1):
            cur = vocab_size - 1
            for _ in range(n_heads):
                cur = _next_prime(cur, seen)
                seen.add(cur)
                primes.append(cur)

    per_layer = (max_ngram - 1) * n_heads
    offsets = []
    for l in range(len(layer_ids)):
        acc = 0
        for b in range(per_layer):
            offsets.append(acc)
            acc += primes[l * per_layer + b]

    return {
        "multipliers": mults,
        "primes": primes,
        "offsets": offsets,
        "token_map": token_map,
        "pad_id": token_map[pad_raw],
        "compressed_vocab": compressed,
    }


def kv_uint32(v):
    return struct.pack("<I", T_UINT32) + struct.pack("<I", v)


def kv_array(elem_type, values):
    fmt = {T_INT32: "<i", T_UINT64: "<Q"}[elem_type]
    out = [struct.pack("<I", T_ARRAY), struct.pack("<I", elem_type), struct.pack("<Q", len(values))]
    out.extend(struct.pack(fmt, int(v)) for v in values)
    return b"".join(out)


def kv_entry(key, value_bytes):
    k = key.encode("utf-8")
    return struct.pack("<Q", len(k)) + k + value_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model-dir", required=True, help="the original checkpoint, for its tokenizer")
    ap.add_argument("--arch", default="deepseek41")
    ap.add_argument("--engram-vocab", type=int, default=16_000_000)
    ap.add_argument("--engram-pad-id", type=int, default=2)
    args = ap.parse_args()

    f = open(args.src, "rb")
    assert f.read(4) == GGUF_MAGIC, "not a gguf"
    version, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")

    def skip_value(t):
        if t == T_STRING:
            n, = struct.unpack("<Q", f.read(8))
            f.seek(n, os.SEEK_CUR)
        elif t == T_ARRAY:
            et, = struct.unpack("<I", f.read(4))
            cnt, = struct.unpack("<Q", f.read(8))
            if et == T_STRING:
                for _ in range(cnt):
                    n, = struct.unpack("<Q", f.read(8))
                    f.seek(n, os.SEEK_CUR)
            else:
                f.seek(FIXED[et] * cnt, os.SEEK_CUR)
        else:
            f.seek(FIXED[t], os.SEEK_CUR)

    kvs = []          # (key, raw value bytes including the type tag)
    seen_keys = set()
    for _ in range(n_kv):
        key = rstr()
        vstart = f.tell()
        t, = struct.unpack("<I", f.read(4))
        skip_value(t)
        vend = f.tell()
        f.seek(vstart)
        raw = f.read(vend - vstart)
        kvs.append([key, raw])
        seen_keys.add(key)

    tensor_info_start = f.tell()
    for _ in range(n_tensors):
        rstr()
        ndim, = struct.unpack("<I", f.read(4))
        f.seek(8 * ndim, os.SEEK_CUR)
        f.seek(4, os.SEEK_CUR)    # ggml type
        f.seek(8, os.SEEK_CUR)    # offset
    tensor_info_end = f.tell()
    f.seek(tensor_info_start)
    tensor_info_raw = f.read(tensor_info_end - tensor_info_start)

    alignment = 32
    for key, raw in kvs:
        if key == "general.alignment":
            alignment, = struct.unpack("<I", raw[4:8])

    data_start = (tensor_info_end + alignment - 1) // alignment * alignment

    # --- rename the mis-prefixed keys -------------------------------------------------
    renamed = 0
    for kv in kvs:
        if kv[0].startswith("deepseek4.engram."):
            kv[0] = args.arch + "." + kv[0][len("deepseek4."):]
            renamed += 1

    def get_scalar(name):
        for key, raw in kvs:
            if key == name:
                t, = struct.unpack("<I", raw[:4])
                return struct.unpack("<I" if t in (T_UINT32,) else "<i", raw[4:8])[0]
        return None

    layer_ids = None
    for key, raw in kvs:
        if key == f"{args.arch}.engram.layer_ids":
            et, = struct.unpack("<I", raw[4:8])
            cnt, = struct.unpack("<Q", raw[8:16])
            fmt = {T_INT32: "<i", T_UINT32: "<I", T_UINT64: "<Q"}[et]
            sz = FIXED[et]
            layer_ids = [struct.unpack(fmt, raw[16 + i * sz: 16 + (i + 1) * sz])[0] for i in range(cnt)]

    n_heads = get_scalar(f"{args.arch}.engram.head_count")
    max_ngram = get_scalar(f"{args.arch}.engram.max_ngram_size")
    if layer_ids is None or n_heads is None or max_ngram is None:
        sys.exit("could not read the engram layer ids, head count or ngram size from the header")

    print(f"  arch={args.arch} layer_ids={layer_ids} heads={n_heads} max_ngram={max_ngram}")
    print(f"  renamed {renamed} mis-prefixed keys")

    const = build_constants(args.model_dir, layer_ids, max_ngram, n_heads,
                            args.engram_vocab, args.engram_pad_id)
    print(f"  compressed vocab {const['compressed_vocab']}, token map {len(const['token_map'])}, "
          f"{len(const['primes'])} primes, pad_id {const['pad_id']}")
    print(f"  first multipliers {const['multipliers'][:3]} (max bits "
          f"{max(const['multipliers']).bit_length()})")

    additions = [
        (f"{args.arch}.engram.multipliers", kv_array(T_UINT64, const["multipliers"])),
        (f"{args.arch}.engram.primes",      kv_array(T_UINT64, const["primes"])),
        (f"{args.arch}.engram.offsets",     kv_array(T_UINT64, const["offsets"])),
        (f"{args.arch}.engram.token_map",   kv_array(T_INT32,  const["token_map"])),
        (f"{args.arch}.engram.pad_id",      kv_uint32(const["pad_id"])),
    ]
    additions = [(k, v) for k, v in additions if k not in {kv[0] for kv in kvs}]
    print(f"  adding {len(additions)} keys")

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", version)
    header += struct.pack("<Q", n_tensors)
    header += struct.pack("<Q", len(kvs) + len(additions))
    for key, raw in kvs:
        header += kv_entry(key, raw)
    for key, raw in additions:
        header += kv_entry(key, raw)
    header += tensor_info_raw

    pad = (-len(header)) % alignment
    header += b"\x00" * pad

    src_size = os.path.getsize(args.src)
    print(f"  header {tensor_info_end} -> {len(header)} bytes, copying "
          f"{(src_size - data_start)/1e9:.1f} GB of tensor data")

    f.seek(data_start)
    with open(args.dst, "wb") as out:
        out.write(header)
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            out.write(chunk)

    print(f"  wrote {args.dst} ({os.path.getsize(args.dst)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
