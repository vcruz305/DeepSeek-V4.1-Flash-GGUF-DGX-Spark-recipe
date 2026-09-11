"""Add DeepSeek V4.1 (DeepseekV41ForCausalLM) conversion support to llama.cpp.

Upstream llama.cpp master carries a full DeepSeek V4 implementation in conversion/deepseek.py
(DeepseekV4Model, DeepseekV4DSparkModel, DeepseekV4FlashVisionModel). V4.1 is close enough to
subclass rather than reimplement. This patch adds the V4.1 pieces and fixes one bug that would
otherwise corrupt V4.1 weights silently.

Differences handled, each verified against the published DeepSeek-V4.1-Flash config.json and
model.safetensors.index.json rather than assumed:

1. FP8 dequantization block size. V4's dequant_model hardcodes repeat_interleave(128, ...),
   matching V4's weight_block_size of [128, 128]. V4.1 declares [32, 32]. Running the V4 path
   unchanged produces wrong weights with no error and fluent but incorrect output, so the block
   size is read from quantization_config instead.

2. num_hash_layers is absent from the V4.1 config. V4 reads it unconditionally and would raise
   KeyError, so it defaults to 0 here.

3. The V4.1 config nests its text parameters under text_config and its vision parameters under
   vision_config. V4 expects them flat.

4. Six tensor families exist in V4.1 that V4 does not map. Four need new enum entries;
   INDEXER_K_NORM and INDEXER_ATTN_K already exist upstream and are reused:
       layers.N.engram.embed.weight      layers.N.engram.k_weight
       layers.N.engram.q_weight          layers.N.engram.wkv.weight
       layers.N.attn.indexer.k_norm.weight
       layers.N.attn.indexer.wk.weight
   The engram tables are the largest single component of the model at roughly 196.6B parameters
   across layers 1 and 14, which is about 36 percent of the checkpoint.

5. The two engram tables need their own write path. Each is 384,006,168 x 256, so 98.3
   billion elements, and the inherited FP8 dequant materializes 393 GB of float32 per
   table. They are quantized in row blocks into a disk-backed memmap instead. Their scale
   layout also differs: [rows, 8], one scale per 32 columns of a single row, rather than
   the [rows/32, cols/32] tiling the generic path assumes.

6. V4.1 lacks attn.compressor.ape and the attn.indexer.compressor.* family that V4 maps. Those
   entries stay in the inherited map and simply go unused.

This patch covers CONVERSION only. Running the resulting GGUF additionally requires a llama.cpp
runtime graph for V4.1, which is separate work.

usage: python patch_llamacpp_v41.py <llama.cpp checkout>  [--revert] [--check]
"""

import pathlib
import shutil
import sys

# ---------------------------------------------------------------- constants.py

CONST_TENSOR_ENUM_ANCHOR = "    INDEXER_PROJ "
CONST_TENSOR_ENUM_NEW = """    ENGRAM_EMBD            = auto()
    ENGRAM_K               = auto()
    ENGRAM_Q               = auto()
    ENGRAM_WKV             = auto()
"""

CONST_TENSOR_NAME_ANCHOR = "    MODEL_TENSOR.INDEXER_PROJ:"
CONST_TENSOR_NAME_NEW = """    MODEL_TENSOR.ENGRAM_EMBD:               "blk.{bid}.engram_embd",
    MODEL_TENSOR.ENGRAM_K:                  "blk.{bid}.engram_k",
    MODEL_TENSOR.ENGRAM_Q:                  "blk.{bid}.engram_q",
    MODEL_TENSOR.ENGRAM_WKV:                "blk.{bid}.engram_wkv",
"""

# V4.1 gets its own architecture rather than riding on DEEPSEEK4.
#
# llama.cpp does not mirror the HF model_type, it drops the "_v": deepseek_v2 became
# deepseek2, deepseek_v3.2 became deepseek32, qwen2_moe became qwen2moe. Only 10 of the
# 150 arch strings in constants.py contain an underscore at all, so V4.1 is "deepseek41".
#
# The behavioural reason matters more than the convention. Sharing deepseek4 sends a V4.1
# file to the V4 loader, which asks for output_hc_fn, output_hc_base and output_hc_scale.
# V4.1 ships none of the three, while it does ship the per-layer hc_attn_* and hc_ffn_*,
# so the loader sees a confusing partial match instead of refusing the file. Riding on
# DEEPSEEK4 was also not strictly correct: V4.1 emits attn_kv_a_norm, which DEEPSEEK4's
# own declared tensor list does not contain.
#
# The list below is derived from the 39 tensor families the converter actually wrote,
# reverse mapped through TENSOR_NAMES, rather than copied from DEEPSEEK4 and trimmed.

# The engram KV keys, declared the way every other arch-scoped key is: with an {arch}
# placeholder. PerLayerEmbedding directly below is the closest existing analogue, and llama.cpp's
# own lazy-read comment pairs PLE and engrams for the same reason.
CONST_KV_ANCHOR = "    class PerLayerEmbedding:\n"
CONST_KV_NEW = '''    class Engram:
        LAYER_IDS      = "{arch}.engram.layer_ids"
        HEAD_COUNT     = "{arch}.engram.head_count"
        KEY_LENGTH     = "{arch}.engram.key_length"
        MAX_NGRAM_SIZE = "{arch}.engram.max_ngram_size"
        MULTIPLIERS    = "{arch}.engram.multipliers"
        PRIMES         = "{arch}.engram.primes"
        OFFSETS        = "{arch}.engram.offsets"
        TOKEN_MAP      = "{arch}.engram.token_map"
        PAD_ID         = "{arch}.engram.pad_id"

    class PerLayerEmbedding:
'''

CONST_ARCH_ENUM_ANCHOR = "    DEEPSEEK4        = auto()\n"
CONST_ARCH_ENUM_NEW = """    DEEPSEEK4        = auto()
    DEEPSEEK41       = auto()
"""

CONST_ARCH_NAME_ANCHOR = '    MODEL_ARCH.DEEPSEEK4:        "deepseek4",\n'
CONST_ARCH_NAME_NEW = '''    MODEL_ARCH.DEEPSEEK4:        "deepseek4",
    MODEL_ARCH.DEEPSEEK41:       "deepseek41",
'''

CONST_ARCH_TENSORS_ANCHOR = "    MODEL_ARCH.DEEPSEEK4: [\n"
CONST_ARCH_TENSORS_NEW = """    MODEL_ARCH.DEEPSEEK41: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_SINKS,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_EXP_PROBS_B_VL,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_KV,
        MODEL_TENSOR.ATTN_OUT_A,
        MODEL_TENSOR.ATTN_OUT_B,
        MODEL_TENSOR.HC_ATTN_FN,
        MODEL_TENSOR.HC_ATTN_BASE,
        MODEL_TENSOR.HC_ATTN_SCALE,
        MODEL_TENSOR.HC_FFN_FN,
        MODEL_TENSOR.HC_FFN_BASE,
        MODEL_TENSOR.HC_FFN_SCALE,
        MODEL_TENSOR.ATTN_COMPRESSOR_WKV,
        MODEL_TENSOR.ATTN_COMPRESSOR_WGATE,
        MODEL_TENSOR.ATTN_COMPRESSOR_NORM,
        MODEL_TENSOR.INDEXER_K_NORM,
        MODEL_TENSOR.ENGRAM_EMBD,
        MODEL_TENSOR.ENGRAM_K,
        MODEL_TENSOR.ENGRAM_Q,
        MODEL_TENSOR.ENGRAM_WKV,
        MODEL_TENSOR.INDEXER_PROJ,
        MODEL_TENSOR.INDEXER_ATTN_K,
        MODEL_TENSOR.INDEXER_ATTN_Q_B,
    ],
    MODEL_ARCH.DEEPSEEK4: [
"""

# ---------------------------------------------------------------- deepseek.py

V41_CLASS = '''

def _v41_is_prime(n: int) -> bool:
    """Trial division, deliberately not sympy.

    The reference uses sympy.isprime, but sympy is not a llama.cpp conversion dependency and
    pulling in a computer algebra system to test primality would be hard to justify. The
    candidates here sit just above engram_vocab_size, about 16 million, so trial division runs
    to sqrt(n) which is around 4000 and costs nothing. There are 48 primes to find in total.
    """
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    f = 41
    while f * f <= n:
        # 6k +/- 1 wheel, having already cleared the small primes above
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def _v41_find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above start that has not been handed out yet."""
    candidate = start + 1
    while not _v41_is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def _v41_build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map token ids to a compressed id space where normalized tokens collapse together."""
    from tokenizers import Regex, normalizers

    sentinel = "\\ue000"  # private-use char to preserve single spaces
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \\t\\r\\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def _v41_compute_hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int):
    """Generate odd multipliers for n-gram hashing, one per (layer, lookback)."""
    import numpy as np
    import torch

    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@ModelBase.register("DeepseekV41ForCausalLM")
@ModelBase.example("deepseek-ai/DeepSeek-V4.1-Flash")
class DeepseekV41Model(DeepseekV4Model):
    """DeepSeek V4.1. Subclasses V4 and overrides only where the checkpoint differs.

    See patch_llamacpp_v41.py for the verified list of differences. The important one is the
    FP8 block size: V4 is [128, 128] and V4.1 is [32, 32], and using the wrong value corrupts
    every dequantized weight without raising.
    """

    model_arch = gguf.MODEL_ARCH.DEEPSEEK41

    def _v41_flatten_hparams(self):
        """Promote the nested text_config to the top level.

        V4.1 nests its text parameters under text_config while the inherited V4 code expects
        them flat. Overriding load_hparams does not work, because base.__init__ calls
        ModelBase.load_hparams explicitly rather than through the instance, so the seam is the
        first consumer of hparams instead, which is index_tensors.
        """
        for key, value in (self.hparams.get("text_config") or {}).items():
            self.hparams.setdefault(key, value)
        # absent in V4.1; V4 reads it unconditionally
        self.hparams.setdefault("num_hash_layers", 0)

    def index_tensors(self, remote_hf_model_id=None):
        self._v41_flatten_hparams()
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        # the FP8 scale block size, read rather than assumed
        qcfg = raw.get("quantization_config") or {}
        block = qcfg.get("weight_block_size") or [128, 128]
        self._v41_block_rows = int(block[0])
        self._v41_block_cols = int(block[1] if len(block) > 1 else block[0])
        logger.info(
            "DeepSeek V4.1: fp8 weight_block_size %dx%d, engram layers %s",
            self._v41_block_rows, self._v41_block_cols,
            self.hparams.get("engram_layer_ids"),
        )

        self.block_count = self.hparams["num_hidden_layers"]
        if self.mtp_only:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        # the vision tower and its aligner are exported separately as an mmproj file
        if name.startswith(("vision.", "aligner.", "image_")):
            return None
        return super().filter_tensors(item)

    def dequant_model(self):
        """Same as V4 but with the block size taken from the checkpoint."""
        fp8_dtypes = self._float8_dtypes()
        tensors_to_remove: list[str] = []
        rows, cols = self._v41_block_rows, self._v41_block_cols

        def dequant_fp8_weight(weight: Tensor, scale: Tensor) -> Tensor:
            out_features, in_features = weight.shape
            scale_f = self._e8m0_to_float(scale)
            scale_f = scale_f.repeat_interleave(rows, 0)[:out_features]
            scale_f = scale_f.repeat_interleave(cols, 1)[:, :in_features]
            return weight.float() * scale_f

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".scale"):
                continue
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name not in self.model_tensors:
                continue
            weight = self.model_tensors[weight_name]
            scale = self.model_tensors[name]
            if weight().dtype not in fp8_dtypes:
                continue
            self.model_tensors[weight_name] = lambda w=weight, s=scale: dequant_fp8_weight(w(), s())
            self._dsv4_fp8_dequantized.add(weight_name)
            tensors_to_remove.append(name)

        for name in tensors_to_remove:
            del self.model_tensors[name]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (engram_ids := hparams.get("engram_layer_ids")) is not None:
            # These MUST carry the arch prefix, not a literal "deepseek4.". llama.cpp resolves
            # every LLM_KV_* as "{arch}.{key}", so a hardcoded prefix means the runtime looks up
            # deepseek41.engram.head_count and finds nothing, on a file that otherwise loads.
            arch = self.gguf_writer.arch
            self.gguf_writer.add_uint32(gguf.Keys.Engram.HEAD_COUNT.format(arch=arch), hparams["engram_n_heads"])
            self.gguf_writer.add_uint32(gguf.Keys.Engram.KEY_LENGTH.format(arch=arch), hparams["engram_head_dim"])
            self.gguf_writer.add_uint32(gguf.Keys.Engram.MAX_NGRAM_SIZE.format(arch=arch), hparams["engram_max_ngram_size"])
            self.gguf_writer.add_array(gguf.Keys.Engram.LAYER_IDS.format(arch=arch), engram_ids)

            # Generate and write engram hash constants
            import numpy as _np

            # A model with engram layers cannot run without these constants, so every step below
            # raises rather than warns: a file that is missing them loads and then hashes wrong.
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

            # Build compressed token map
            token_map, compressed_vocab_size = _v41_build_compressed_token_map(tokenizer)
            expected_vocab = hparams.get("engram_compressed_vocab_size")
            if compressed_vocab_size != expected_vocab:
                # every multiplier derives from this size, so a mismatch rehashes the whole table
                raise ValueError(f"compressed vocab size is {compressed_vocab_size}, config says {expected_vocab}")

            # Compute multipliers
            layer_ids = tuple(engram_ids)
            multipliers = _v41_compute_hash_multipliers(layer_ids, hparams["engram_max_ngram_size"], compressed_vocab_size)

            # Compute primes and offsets. Every (n-gram size, head) pair gets its own bucket, and
            # the search starts over at engram_vocab_size - 1 for each n-gram size, so the shared
            # `seen` set is what keeps the buckets distinct.
            max_ngram_size = hparams["engram_max_ngram_size"]
            n_heads = hparams["engram_n_heads"]
            primes_list, seen_primes = [], set()

            for layer_id in layer_ids:
                per_ngram = []
                for ngram_idx in range(max_ngram_size - 1):
                    current_search = hparams["engram_vocab_size"] - 1
                    sizes = []
                    for head_idx in range(n_heads):
                        current_search = _v41_find_next_prime(current_search, seen_primes)
                        seen_primes.add(current_search)
                        sizes.append(current_search)
                    per_ngram.append(sizes)
                primes_list.append(per_ngram)

            # [n_engram_layers, max_ngram_size - 1, n_heads]
            primes_array = _np.array(primes_list, dtype=_np.uint64)

            # each bucket starts where the previous one ended, in that same order flattened
            offsets_list = []
            for layer_primes in primes_array:
                flat_primes = layer_primes.flatten()
                offsets = _np.cumsum(_np.concatenate(([0], flat_primes[:-1])))
                offsets_list.append(offsets.reshape(layer_primes.shape))
            offsets_array = _np.array(offsets_list, dtype=_np.uint64)

            # Write constants to GGUF.
            # add_array() infers the element type from the first item and maps every Python int to
            # INT32, which would truncate the multipliers, so pass the element type explicitly and
            # flatten by hand: a gguf array is one dimensional.
            def add_u64(key, arr):
                self.gguf_writer.add_key_value(
                    key,
                    [int(x) for x in _np.asarray(arr).reshape(-1)],
                    gguf.GGUFValueType.ARRAY,
                    gguf.GGUFValueType.UINT64,
                )

            add_u64(gguf.Keys.Engram.MULTIPLIERS.format(arch=arch), multipliers.numpy())
            add_u64(gguf.Keys.Engram.PRIMES.format(arch=arch), primes_array)
            add_u64(gguf.Keys.Engram.OFFSETS.format(arch=arch), offsets_array)
            self.gguf_writer.add_key_value(
                gguf.Keys.Engram.TOKEN_MAP.format(arch=arch),
                [int(x) for x in token_map],
                gguf.GGUFValueType.ARRAY,
                gguf.GGUFValueType.INT32,
            )
            # the reference stores the padding token already mapped, so do the same here
            self.gguf_writer.add_uint32(
                gguf.Keys.Engram.PAD_ID.format(arch=arch),
                int(token_map[hparams.get("engram_pad_id", 2)]),
            )
            logger.info("Engram constants written: multipliers %s, primes %s, offsets %s, token_map %d",
                        multipliers.shape, primes_array.shape, offsets_array.shape, len(token_map))


    # rows per block when rewriting an engram table; 1M rows is about 1 GB of float32 scratch
    _V41_ENGRAM_CHUNK_ROWS = 1_000_000

    def _write_engram_table(self, bid: int) -> list[str]:
        """Quantize one engram table in row blocks, accumulating into a disk-backed memmap.

        The inherited FP8 path cannot be used here. It computes weight.float() * scale over the
        whole tensor, and one engram table is 384,006,168 x 256, so 98.3 billion elements, which
        is 393 GB as float32. A run that reaches this tensor collapses from 118 GiB free to 13 GiB
        and is killed. Every other tensor in the model converts normally.

        The scale layout also differs from the rest of the checkpoint. Linear weights carry a
        [rows/32, cols/32] scale matching weight_block_size [32, 32], while the engram scale is
        [rows, 8], which is one scale per 32 columns within a single row. Broadcasting it the way
        the generic path does would corrupt the table, so it is expanded along columns only.

        Reading is done straight from the safetensors shard rather than through the lazy tensor
        wrapper, because to_eager materializes the whole tensor before any slicing takes effect.
        """
        import json as _json
        import os as _os
        import tempfile as _tempfile

        import numpy as _np
        from safetensors import safe_open as _safe_open

        weight_name = f"layers.{bid}.engram.embed.weight"
        scale_name = f"layers.{bid}.engram.embed.scale"

        index_path = self.dir_model / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = _json.load(f)["weight_map"]
        shard = self.dir_model / weight_map[weight_name]

        qtype = gguf.GGMLQuantizationType.Q8_0
        block_elems = gguf.GGML_QUANT_SIZES[qtype][0]

        with _safe_open(str(shard), framework="pt") as f:
            wsl = f.get_slice(weight_name)
            n_rows, n_cols = (int(x) for x in wsl.get_shape())
            has_scale = scale_name in f.keys()
            ssl = f.get_slice(scale_name) if has_scale else None
            scale_groups = int(ssl.get_shape()[1]) if has_scale else 0

            if n_cols % block_elems:
                raise ValueError(
                    f"engram row width {n_cols} is not a multiple of the {qtype.name} block {block_elems}"
                )
            if has_scale and n_cols % scale_groups:
                raise ValueError(
                    f"engram row width {n_cols} is not divisible by its {scale_groups} scale groups"
                )
            per_group = n_cols // scale_groups if has_scale else 0

            row_bytes = int(gguf.quantize(_np.zeros((1, n_cols), dtype=_np.float32), qtype).nbytes)
            rows_per_chunk = min(int(self._V41_ENGRAM_CHUNK_ROWS), n_rows)
            n_chunks = (n_rows + rows_per_chunk - 1) // rows_per_chunk

            tmp_dir = _os.environ.get("V41_ENGRAM_TMPDIR") or _tempfile.gettempdir()
            tmp_path = _os.path.join(tmp_dir, f"engram_{bid}_{qtype.name}.bin")
            logger.info(
                "engram layer %d: %d x %d, scale groups %d, %s in %d blocks of %d rows, staging %.1f GB at %s",
                bid, n_rows, n_cols, scale_groups, qtype.name, n_chunks, rows_per_chunk,
                n_rows * row_bytes / 1e9, tmp_path,
            )

            out = _np.memmap(tmp_path, dtype=_np.uint8, mode="w+", shape=(n_rows, row_bytes))
            for ci, start in enumerate(range(0, n_rows, rows_per_chunk)):
                stop = min(start + rows_per_chunk, n_rows)
                chunk = wsl[start:stop, :].float()
                if has_scale:
                    s = self._e8m0_to_float(ssl[start:stop, :])
                    chunk = chunk * s.repeat_interleave(per_group, 1)[:, :n_cols]
                out[start:stop] = gguf.quantize(
                    chunk.cpu().numpy().astype(_np.float32), qtype
                ).reshape(stop - start, row_bytes)
                del chunk
                if ci % 25 == 0:
                    logger.info("  engram layer %d: %d / %d rows", bid, stop, n_rows)
            out.flush()

        new_name = self.format_tensor_name(gguf.MODEL_TENSOR.ENGRAM_EMBD, bid, ".weight")
        self.gguf_writer.add_tensor(new_name, out, raw_dtype=qtype)
        logger.info("engram layer %d: wrote %s as %s", bid, new_name, qtype.name)

        consumed = [weight_name]
        if has_scale:
            consumed.append(scale_name)
        return consumed

    def generate_extra_tensors(self):
        yield from super().generate_extra_tensors()

        consumed: list[str] = []
        for bid in (self.hparams.get("engram_layer_ids") or []):
            if f"layers.{bid}.engram.embed.weight" in self.model_tensors:
                consumed.extend(self._write_engram_table(int(bid)))
        for name in consumed:
            if name in self.model_tensors:
                del self.model_tensors[name]

    def _map_dsv4_tensor_name(self, name: str, bid):
        match = re.match(r"layers\\.(\\d+)\\.(.+)$", name)
        if match is not None:
            v41_only = {
                "engram.embed.weight": (gguf.MODEL_TENSOR.ENGRAM_EMBD, ".weight"),
                "engram.k_weight": (gguf.MODEL_TENSOR.ENGRAM_K, ".weight"),
                "engram.q_weight": (gguf.MODEL_TENSOR.ENGRAM_Q, ".weight"),
                "engram.wkv.weight": (gguf.MODEL_TENSOR.ENGRAM_WKV, ".weight"),
                "attn.indexer.k_norm.weight": (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".weight"),
                "attn.indexer.wk.weight": (gguf.MODEL_TENSOR.INDEXER_ATTN_K, ".weight"),
            }
            tensor_name = match.group(2)
            if tensor_name in v41_only:
                return v41_only[tensor_name]
        return super()._map_dsv4_tensor_name(name, bid)
'''


def patch_constants(path: pathlib.Path, revert: bool, check: bool) -> int:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.v41orig")

    if revert:
        if backup.exists():
            shutil.copy2(backup, path)
            print("  reverted  gguf/constants.py")
            return 0
        print("  no backup gguf/constants.py")
        return 4

    # Check if ALL expected changes are already present
    if "DEEPSEEK41" in text and "PAD_ID         = \"{arch}.engram.pad_id\"" in text:
        print("  ok        gguf/constants.py (already has all engram constants)")
        return 0

    anchors = (
        CONST_TENSOR_ENUM_ANCHOR,
        CONST_TENSOR_NAME_ANCHOR,
        CONST_KV_ANCHOR,
        CONST_ARCH_ENUM_ANCHOR,
        CONST_ARCH_NAME_ANCHOR,
        CONST_ARCH_TENSORS_ANCHOR,
    )
    for anchor in anchors:
        if anchor not in text:
            print(f"  NO MATCH  gguf/constants.py, missing anchor: {anchor.strip()!r}")
            return 5

    if check:
        print("  would patch gguf/constants.py")
        return 0

    # the engram tensor enum and its names are shared, the arch entries are new
    if "ENGRAM_EMBD" not in text:
        text = text.replace(CONST_TENSOR_ENUM_ANCHOR, CONST_TENSOR_ENUM_NEW + CONST_TENSOR_ENUM_ANCHOR, 1)
        text = text.replace(CONST_TENSOR_NAME_ANCHOR, CONST_TENSOR_NAME_NEW + CONST_TENSOR_NAME_ANCHOR, 1)

    # Handle Engram class: either add it or add the missing KV entries
    if "class Engram:" not in text:
        text = text.replace(CONST_KV_ANCHOR, CONST_KV_NEW, 1)
    else:
        # An older run of this patcher left an Engram class with only some of the keys.
        # Add the missing ones one at a time, so re-running never duplicates a line.
        anchor = "        MAX_NGRAM_SIZE = \"{arch}.engram.max_ngram_size\""
        for name, key in (
            ("MULTIPLIERS   ", "multipliers"),
            ("PRIMES        ", "primes"),
            ("OFFSETS       ", "offsets"),
            ("TOKEN_MAP     ", "token_map"),
            ("PAD_ID        ", "pad_id"),
        ):
            line = "        %s = \"{arch}.engram.%s\"" % (name, key)
            if line not in text:
                text = text.replace(anchor, anchor + "\n" + line, 1)

    # Add arch enum, name, and tensor map entries (only if not already present)
    if "DEEPSEEK41       = auto()" not in text:
        text = text.replace(CONST_ARCH_ENUM_ANCHOR, CONST_ARCH_ENUM_NEW, 1)
    if 'MODEL_ARCH.DEEPSEEK41:' not in text:
        text = text.replace(CONST_ARCH_NAME_ANCHOR, CONST_ARCH_NAME_NEW, 1)
    if "MODEL_ARCH.DEEPSEEK41: [" not in text:
        text = text.replace(CONST_ARCH_TENSORS_ANCHOR, CONST_ARCH_TENSORS_NEW, 1)

    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    print("  patched   gguf/constants.py")
    return 0


def patch_init(path: pathlib.Path, revert: bool, check: bool) -> int:
    """conversion/__init__.py maps an architecture name to the module that implements it.

    The @ModelBase.register decorator only runs once that module is imported, and the importer
    is driven by this map, so a class registered in deepseek.py stays invisible until the
    architecture appears here.
    """
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.v41orig")
    key = '    "DeepseekV41ForCausalLM": "deepseek",\n'
    anchor = '    "DeepseekV4ForCausalLM": "deepseek",\n'

    if revert:
        if backup.exists():
            shutil.copy2(backup, path)
            print("  reverted  conversion/__init__.py")
            return 0
        print("  no backup conversion/__init__.py")
        return 4

    if key in text:
        print("  ok        conversion/__init__.py (already maps V4.1)")
        return 0
    if anchor not in text:
        print("  NO MATCH  conversion/__init__.py has no DeepseekV4ForCausalLM entry to anchor on")
        return 5
    if check:
        print("  would patch conversion/__init__.py")
        return 0

    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(anchor, anchor + key, 1), encoding="utf-8")
    print("  patched   conversion/__init__.py")
    return 0


def patch_deepseek(path: pathlib.Path, revert: bool, check: bool) -> int:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.v41orig")

    if revert:
        if backup.exists():
            shutil.copy2(backup, path)
            print("  reverted  conversion/deepseek.py")
            return 0
        print("  no backup conversion/deepseek.py")
        return 4

    if "DeepseekV41ForCausalLM" in text:
        print("  ok        conversion/deepseek.py (already registers V4.1)")
        return 0
    if "class DeepseekV4Model" not in text:
        print("  NO MATCH  conversion/deepseek.py has no DeepseekV4Model to subclass")
        return 5
    if check:
        print("  would patch conversion/deepseek.py")
        return 0

    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.rstrip("\n") + "\n" + V41_CLASS, encoding="utf-8")
    print("  patched   conversion/deepseek.py")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = pathlib.Path(sys.argv[1])
    revert = "--revert" in sys.argv
    check = "--check" in sys.argv

    constants = root / "gguf-py" / "gguf" / "constants.py"
    deepseek = root / "conversion" / "deepseek.py"
    init = root / "conversion" / "__init__.py"
    for p in (constants, deepseek, init):
        if not p.exists():
            print(f"  MISSING   {p}")
            return 3

    rc = patch_constants(constants, revert, check)
    rc = patch_deepseek(deepseek, revert, check) or rc
    rc = patch_init(init, revert, check) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
