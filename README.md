# Running DeepSeek-V4.1-Flash on one NVIDIA DGX Spark

llama.cpp recipe for **DeepSeek-V4.1-Flash** (`DeepseekV41ForCausalLM`, arch `deepseek41`) on a
single **DGX Spark / GB10 (SM121)**: how to serve it, what the engine needs, and where the work
currently stands.

> Independent community engineering. Not affiliated with DeepSeek, NVIDIA, or ggml-org.
> This repo is the recipe and the patches, not a weight dump.

## Status, 2026-09-11

**It generates.** DeepSeek-V4.1-Flash Q2_K on one DGX Spark GB10, CPU backend:

```
prompt: "The chemical symbol for gold is"
output: "The user is asking for the chemical symbol for gold. This"
[ Prompt: 2.5 t/s | Generation: 2.3 t/s ]
```

| Stage | State |
|---|---|
| Convert safetensors to GGUF | Works |
| Load, engram, hyper-connections | Works, verified against the reference |
| Sparse attention, shared streams, indexer | Works |
| Generate on the real weights | Works at Q2_K |
| Quality measured against the reference | Not done, see below |
| MTP head, vision | Present in the checkpoint, not mapped. Text only |
| Two level candidate mask | Not implemented, see the context cap |

Corpus NLL against the reference implementation has not been run, so the sample above is a smoke
test rather than a quality claim.

## The published GGUFs are fixed as of 2026-09-11

Files on the Hub were converted before two fixes landed, and both lived in the header of the first
shard of each rung: the engram keys carried a hardcoded `deepseek4.` prefix, and five of the nine
keys were missing entirely. Both are repaired on all five rungs now, so the files load as they are.

`scripts/fix_gguf_engram_kv.py` is kept for anyone holding an older copy. It re-emits every
existing key byte for byte and copies tensor data untouched, and it is a no-op on a file converted
after the fixes.

```bash
python scripts/fix_gguf_engram_kv.py     DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf     fixed/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf     --model-dir /path/to/DeepSeek-V4.1-Flash
```

It computes the constants from the checkpoint's tokenizer and config, re-emits every existing key
byte for byte, and copies the tensor data untouched. Symlink the remaining shards alongside the
output. Only the first shard of each rung carries file level metadata. The script is a no-op on a
file converted after these fixes.

| What | Where |
|---|---|
| Model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| GGUF | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) |
| Conversion PR | [ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) |
| Runtime branch | [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp) `runtime/deepseek41` |

---

## Serving it

### Build the engine

The runtime lives on a branch; upstream llama.cpp does not know this architecture yet.

```bash
git clone https://github.com/vcruz305/llama.cpp
cd llama.cpp && git checkout runtime/deepseek41

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build --config Release -j 12
```

`CMAKE_CUDA_ARCHITECTURES=121` is the GB10. Building for the wrong arch costs a silent fallback to
a slower path, so set it explicitly.

### Run

```bash
./build/bin/llama-cli     -m DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf     -lm mmap -ngl 99 -cmoe -ot "engram_embd.weight=CPU"     -fa on -c 2048 -b 2048 -ub 1024 -t 20     -p "Explain in two sentences why a mixture of experts model activates only a few experts per token."
```

Measured on one DGX Spark GB10, Q2_K, 2026-09-11:

| Config | Prompt | Generation |
|---|---:|---:|
| CPU build, no offload | 2.5 t/s | 2.3 t/s |
| CUDA, `-cmoe`, tuned batch | 1.9 t/s | **2.8 t/s** |

Sample output, temperature 0:

```
We need answer. Need explain why MoE activates only few experts per token. In two sentences.
Need likely: computational efficiency, sparse gating, top-k routing. We can say: MoE uses a
learned gating network that scores experts and routes each token to only top-k
```

**`-lm mmap` is the flag that makes this work at all, and it is not optional on GB10.** The default
load mode is `auto`, which disables mmap globally if any device reports no mmap support. The GB10
CUDA device reports exactly that, so a 246 GB model tries to allocate 261 GB up front and fails
with `unable to allocate CUDA_Host buffer`, or `CPU buffer` if you pass `--no-host`. Forcing
`-lm mmap` restores demand paging and the offload then works.

`-cmoe` keeps the mixture of experts weights on CPU and offloads the rest, and
`-ot "engram_embd.weight=CPU"` keeps the two engram tables there too, since they are designed to be
read from mmap on demand rather than held resident.

The offload is worth having but it is not transformative, because the expert FFN dominates and
stays on CPU either way. The real limit is that 246 GB against 121.7 GiB of unified memory is
two-times oversubscribed, so throughput is bounded by paging rather than by compute.

### Use the CPU build for anything above ~100 GB

A CUDA build pins host memory for CPU-resident weights, and pinned memory is capped by physical
RAM. On a 121.7 GiB Spark, Q1_0 at 106 GB loads and Q2_K at 246 GB fails with
`unable to allocate CUDA_Host buffer`. The CPU build uses plain mmap demand paging and handles it,
which is what the Q2_K numbers above were measured on.

### Which rung fits

The Spark has 121.7 GiB of unified memory, so the model does not fit whole at any rung and part of
it streams. **Start at Q2_K.** Q1_0 loads and runs, and emits the same token for every prompt: a
1-bit block format keeps one scale per block and one sign bit per weight, so the token embedding
table comes back as plus or minus a single magnitude per block. The embedding is the only place
prompt identity enters the model, so once it is a sign vector the prompts are nearly
indistinguishable. That rung is 1.53 bits per weight over the backbone, or 1.13 counting the
engram tables.

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

### Engram tables and memory

Two of the layers carry n-gram lookup tables, roughly 196.6B parameters between them, which is most
of the file. A token touches 24 rows of each, so they are created with `TENSOR_READ_LAZY` and read
on demand rather than resident. That needs mmap, so leave `--no-mmap` off.

## What this architecture is

Against DeepSeek-V4, which llama.cpp already supports:

- **Engram**: n-gram keyed lookup tables at two layers, added into the residual stream through a
  learned gate.
- **Hyper-connection lag**: each sublayer's mix coefficients are consumed by the *next* sublayer,
  so the last layer's FFN mix performs the final collapse and the model ships no `output_hc_*`.
- **Sparse attention**: four KV source layers sharing one compressed stream, index keys derived
  from that shared latent, and a two level candidate mask. Everything but the candidate mask is
  implemented.

The architecture string is `deepseek41`. llama.cpp drops the `_v` (`deepseek_v2` became
`deepseek2`, `deepseek_v3.2` became `deepseek32`), so `deepseek41` is the house style; vLLM's
`deepseek_v41` is the outlier. Files converted before 2026-09-10 carry `general.architecture =
deepseek4` and need redoing.

## Converting it yourself

Skip this if you are pulling the GGUFs from the Hub. It needs the full 510 GB of FP8 safetensors
and roughly 1 TB of scratch.

```bash
python scripts/patch_llamacpp_v41.py /path/to/llama.cpp    # --check / --revert also work

PYTHONPATH=$PWD/gguf-py python conversion/convert_hf_to_gguf.py \
    /path/to/DeepSeek-V4.1-Flash \
    --outfile /scratch/DeepSeek-V4.1-Flash-Q8_0.gguf \
    --outtype q8_0 --use-temp-file
```

Confirm the run logs `Engram constants written: ...`. The engram tables live in the last two shards,
so their absence early in a run is ordering, not a bug.

## Gotchas that cost real time

**The engram constants silently never reached the file.** `gguf-py`'s `add_array()` infers the
element type from the first item and maps every Python int to INT32, with a literal
`TODO: need help with 64-bit types` beside it. The hash multipliers are 47-bit, so the write raised
`struct.error`, and a broad `except` downgraded that to one warning line. The result was a GGUF with
**none** of the engram constants, failing later at load with a confusing missing-key error.

**The FP8 block size is not 128.** V4 declares `weight_block_size [128, 128]`; V4.1 declares
`[32, 32]`. Inheriting V4's dequantization yields fluent, wrong output with no error.

**The engram scale layout differs from every other tensor.** Linear weights use `[rows/32, cols/32]`;
the engram scale is `[rows, 8]`, one scale per 32 columns of a single row.

**The engram tables will OOM the box during conversion.** Each is 384,006,168 x 256, and the
inherited path materializes `weight.float()` at 393 GB per table. Read slices straight from the
shard, quantize in row blocks, accumulate into a `np.memmap`. Do not use `--dry-run`; it accumulates
tensors in RAM and dies around 200 GB.

**`llama_kv_cache::build_input_k_rot` hangs on a cache with no layers.** Its width search is
`do { nrot *= 2; } while (n_embd_head_k_all % nrot == 0);`, and every width divides 0. It presents
as a hang at 100% of one core inside `graph_reserve` with nothing printed. Fixed on the runtime
branch. The same search starts at 64, so a head dim below that yields a rotation wider than the
head; real models are 512 and 128, so it only bites toy sizes.

## Testing without the weights

`scripts/make_tiny_deepseek41.py` writes a ~1 MB six layer `deepseek41` file with random weights,
engram on layer 1 and compressor layers at two ratios, which exercises the loader and the graph
without the 510 GB download.

```bash
python scripts/make_tiny_deepseek41.py tiny-deepseek41.gguf
./build/bin/llama-bench -m tiny-deepseek41.gguf -p 8 -n 4
./build/bin/llama-eval-callback -m tiny-deepseek41.gguf -p "hello world" -n 1 | grep engram
```

Two traps if you write your own: the KV keys are `{arch}.attention.indexer.head_count`, with a dot
rather than an underscore, and the tensor name strings are `attn_kv_a_norm`, `attn_output_a` and
`attn_output_b`, not the enum spellings. Give it the full 256 byte tokens or SPM tokenization throws
`unordered_map::at`; `llama-bench` hides this by using synthetic tokens.

## Verification done

- **Engram hash exact against the reference.** All 576 row indices for a 12 token sequence
  (12 tokens x 2 layers x 24 buckets) match `NgramHashState` from the checkpoint's own
  `inference/engram.py`.
- **Engram gate equivalent to float32 rounding**, max difference 2.4e-7.
- **Compressor pooling equivalent**, max difference 6e-8. The softmax runs across the group, one
  weight per channel; doing it across channels instead is finite, plausible and wrong by 4.2.
- **Indexer scaling identical** to V4's existing `1/sqrt(index_head_dim * n_heads)`.
- **Graph structure confirmed** with `llama-eval-callback`: the layer 0 identity pre-mix, the one
  sublayer lag, a per-copy engram gate, and the full compressed path (pooled rows, index keys from
  the pre-rope latent, the indexer top-k and the mask that carries it into attention).

## License

MIT for the recipe and scripts here. DeepSeek-V4.1-Flash is under its own license; check the model
card before redistributing anything derived from it.
