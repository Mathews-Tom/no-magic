# Systems & Inference

The engineering that makes models fast, small, and deployable. These scripts demystify the optimizations that turn research prototypes into production systems.

## Scripts

Measured on Apple M-series, Python 3.12. Times are wall-clock.

| Script               | Algorithm                                                         | Time   | Status |
| -------------------- | ----------------------------------------------------------------- | ------ | ------ |
| `microattention.py`  | Attention variants compendium (MHA, GQA, MQA, sliding window)     | < 1s   | Pass   |
| `microbeam.py`       | Decoding strategies (greedy, top-k, top-p, beam, speculative)     | 1m 27s | Pass   |
| `microcheckpoint.py` | Activation/gradient checkpointing — trading compute for memory    | < 1s   | Pass   |
| `microflash.py`      | Flash Attention algorithmic simulation (tiling, online softmax)   | < 1s   | Pass   |
| `microkv.py`         | KV-cache mechanics (with vs. without, paged attention)            | 0m 33s | Pass   |
| `micropaged.py`      | PagedAttention — vLLM-style paged KV-cache memory management      | < 1s   | Pass   |
| `microparallel.py`   | Tensor and pipeline parallelism — distributed model inference     | 0m 27s | Pass   |
| `microquant.py`      | Weight quantization (INT8, INT4, per-channel vs. per-tensor)      | 1m 22s | Pass   |
| `microrope.py`       | Rotary Position Embedding (RoPE) — position via rotation matrices | < 1s   | Pass   |
| `microssm.py`        | State Space Models (Mamba-style) — linear-time sequence modeling  | 0m 34s | Pass   |

### Forward-Pass Scripts

`microattention.py`, `microflash.py`, `microcheckpoint.py`, `micropaged.py`, and `microrope.py` are **forward-pass comparisons** — they demonstrate algorithmic mechanics rather than training loops. This is an intentional exception to the train+infer rule: the pedagogical value is in comparing implementations side-by-side.

### Algorithmic Simulations

`microflash.py` is an **algorithmic simulation** of Flash Attention. Pure Python will be slower than standard attention. The script demonstrates _what_ Flash Attention does (tiled computation, online softmax), not _why_ it's fast in practice (GPU memory hierarchy optimization). Comments make this distinction explicit.

## Future Candidates

| Algorithm                             | What It Would Teach                                   | Notes                                                   |
| ------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| **Speculative Decoding (standalone)** | Draft-verify paradigm in depth                        | Currently part of microbeam; could be its own deep-dive |
| **Continuous Batching**               | Dynamic batching for throughput optimization          | The technique behind vLLM's performance                 |
| **Prefix Caching**                    | Sharing KV-cache across requests with common prefixes | Extension of microkv concepts                           |
| **Mixed Precision**                   | FP16/BF16 training with loss scaling                  | How half-precision training works                       |

## Learning Path

These scripts can be studied in any order, but this sequence builds concepts incrementally:

```
microrope.py          → How position gets encoded through rotation matrices
microattention.py     → How attention actually works (all variants)
microkv.py            → Why LLM inference is memory-bound
micropaged.py         → How vLLM manages KV-cache memory with paging
microflash.py         → How attention gets fast (tiling + online softmax)
microcheckpoint.py    → How to train deeper models by recomputing activations
microparallel.py      → How models get split across devices
microquant.py         → How models get compressed (INT8/INT4)
microssm.py           → How Mamba models bypass attention entirely
microbeam.py          → How decoding strategies shape output quality
```
