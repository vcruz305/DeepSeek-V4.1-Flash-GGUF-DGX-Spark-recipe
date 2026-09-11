# DeepSeek-V4.1-Flash to GGUF, on one NVIDIA DGX Spark

Reproducible **conversion** recipe for **DeepSeek-V4.1-Flash** (`DeepseekV41ForCausalLM`) into GGUF,
plus the llama.cpp architecture work it needs, on a single **DGX Spark / GB10 (SM121)**.

> Independent community engineering. Not affiliated with DeepSeek, NVIDIA, or ggml-org.
> This repo is the **conversion recipe and patches**, not a weight dump.

## Status, read this first

| Stage | State |
|---|---|
| Convert safetensors to GGUF | **Works.** Patch in `scripts/`, upstream PR open |
| llama.cpp loads a `deepseek41` file | **Works** for the engram, hyper-connection and MoE path |
| llama.cpp runs the full model | **Not yet.** The V4.1 sparse attention is unimplemented |

The GGUFs this recipe produces **do not generate text on upstream llama.cpp today**. V4.1's sparse
attention differs from V4's and is still being written; a file with a compressor is refused at load
rather than run through V4's path, which would silently drop the long range half of attention and
still produce fluent output. Track the runtime branch below.

| What | Where |
|---|---|
| Model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Conversion PR | [ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) |
| Conversion branch | [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp) `feat/deepseek-v41-convert` |
| Runtime branch (WIP) | [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp) `runtime/deepseek41` |

---

## Architecture name

`deepseek41`. llama.cpp drops the `_v` in architecture strings (`deepseek_v2` became `deepseek2`,
`deepseek_v3.2` became `deepseek32`), so `deepseek41` is the house style. vLLM uses `deepseek_v41`;
only 10 of roughly 150 llama.cpp arch strings contain an underscore, so that one is the outlier.

Files converted before 2026-09-10 carry `general.architecture = deepseek4` and are not loadable by
this work. They need redoing.

## What is new in V4.1

Against DeepSeek-V4, which llama.cpp already supports:

- **Engram**: n-gram keyed lookup tables at two layers, ~196.6B parameters between them, added into
  the residual stream through a learned gate
- **Hyper-connection lag**: each sublayer's mix coefficients are consumed by the *next* sublayer,
  so the last layer's FFN mix performs the final collapse and the model ships no `output_hc_*`
- **Sparse attention**: four KV source layers sharing one compressed stream, index keys derived
  from that shared latent, and a two-level candidate mask. This is the part that is unfinished

## Converting

Requires the full 510 GB of FP8 safetensors, roughly 1 TB of scratch, and patience. The engram
tables live in the last two shards, so their absence early in a run is ordering, not a bug.

```bash
git clone https://github.com/vcruz305/llama.cpp
cd llama.cpp && git checkout feat/deepseek-v41-convert

# or apply the patch to an upstream checkout
python scripts/patch_llamacpp_v41.py /path/to/llama.cpp          # --check / --revert also work

PYTHONPATH=$PWD/gguf-py python conversion/convert_hf_to_gguf.py \
    /path/to/DeepSeek-V4.1-Flash \
    --outfile /scratch/DeepSeek-V4.1-Flash-Q8_0.gguf \
    --outtype q8_0 --use-temp-file
```

Confirm the run logs `Engram constants written: ...`. If it does not, stop; see below.

## Gotchas that cost real time

**The engram constants silently never reached the file.** `gguf-py`'s `add_array()` infers the
element type from the first item and maps every Python int to INT32, with a literal
`TODO: need help with 64-bit types` beside it. The engram hash multipliers are 47-bit, so the write
raised `struct.error`, and a broad `except` downgraded that to one warning line. The result was a
GGUF with **none** of the engram constants, failing later at load with a confusing missing-key
error. Fixed by passing the element type explicitly and flattening by hand, since a gguf array is
one dimensional and the primes and offsets are not.

**The FP8 block size is not 128.** V4 declares `weight_block_size [128, 128]`; V4.1 declares
`[32, 32]`. Inheriting V4's dequantization yields fluent, wrong output with no error. Read it from
`quantization_config`.

**The engram scale layout differs from every other tensor.** Linear weights use `[rows/32, cols/32]`;
the engram scale is `[rows, 8]`, one scale per 32 columns of a single row. Applying the generic
broadcast corrupts the largest tensor in the model.

**The engram tables will OOM the box.** Each is 384,006,168 x 256, so 98.3G elements, and the
inherited path materializes `weight.float()` at 393 GB per table. Read slices straight from the
shard with `safe_open(...).get_slice()`, quantize in row blocks, and accumulate into a `np.memmap`.
Do not use `--dry-run`; it accumulates tensors in RAM and dies around 200 GB.

**`llama_kv_cache::build_input_k_rot` hangs on a cache with no layers.** Its width search is
`do { nrot *= 2; } while (n_embd_head_k_all % nrot == 0);`, and every width divides 0. It presents
as the model hanging at 100% of one core inside `graph_reserve` with nothing printed. Fixed on the
runtime branch.

## Measured sizes

Tensor payload, measured on cruz-spark. The architecture rename and the KV fix change metadata
only, so these still hold.

| Rung | Bytes | GiB |
|---|---:|---:|
| Q8_0 staging | 507,953,707,584 | 473.1 |
| Q3_K_M | 347,270,954,112 | 323.4 |
| Q2_K | 264,514,761,248 | 246.3 |

Ratios read high (0.684 and 0.521 of Q8_0) because the staging file is not uniformly Q8_0: the
experts arrive as MXFP4 at 4.25 bpw, so higher rungs move parts of the mixture *up*. At Q3_K_M
`ffn_down_exps` goes mxfp4 to q5_K and grows from 2295 to 2970 MiB. Expect Q5_K_M to be close
enough to Q8_0 to be poor value. The engram tables do follow the rung, 99,611 to 40,284 MiB each
between q8_0 and q3_K.

## Testing the runtime without the weights

`scripts/make_tiny_deepseek41.py` writes a ~1 MB four layer `deepseek41` file with random weights
and engram on layer 1, which exercises the loader and the graph without the 510 GB download.

```bash
python scripts/make_tiny_deepseek41.py tiny-deepseek41.gguf
./build/bin/llama-bench -m tiny-deepseek41.gguf -p 8 -n 4
./build/bin/llama-eval-callback -m tiny-deepseek41.gguf -p "hello world" -n 1 | grep engram
```

Two traps if you write your own: the KV keys are `{arch}.attention.indexer.head_count`, with a dot
rather than an underscore, and the tensor name strings are `attn_kv_a_norm`, `attn_output_a` and
`attn_output_b`, not the enum spellings. Give it the full 256 byte tokens or SPM tokenization
throws `unordered_map::at`; `llama-bench` hides this by using synthetic tokens.

## Verification done

- **Engram hash exact against the reference.** All 576 row indices for a 12 token sequence
  (12 tokens x 2 layers x 24 buckets) match `NgramHashState` from the checkpoint's own
  `inference/engram.py`. Covers the compressed token map, the PCG64 multipliers, the prime search,
  the offsets, the XOR chain, and the start of sequence padding.
- **Engram gate equivalent to float32 rounding**, max difference 2.4e-7 against the reference
  formulation.
- **Compressor pooling equivalent**, max difference 6e-8. The softmax runs across the group, one
  weight per channel; doing it across channels instead is finite, plausible and wrong by 4.2.

## License

MIT for the recipe and scripts here. DeepSeek-V4.1-Flash is under its own license; check the model
card before redistributing anything derived from it.
