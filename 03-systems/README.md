# Systems & Inference

The engineering that makes models fast, small, and deployable. These scripts demystify the optimizations that turn research prototypes into production systems.

## Planned Scripts

| Script | Algorithm | Status |
|---|---|---|
| `microattention.py` | Attention variants compendium (MHA, GQA, MQA, sliding window) | Planned |
| `microkv.py` | KV-cache mechanics (with vs. without, paged attention) | Planned |
| `microquant.py` | Weight quantization (INT8, INT4, per-channel vs. per-tensor) | Planned |
| `microflash.py` | Flash Attention algorithmic simulation (tiling, online softmax) | Planned |
| `microbeam.py` | Decoding strategies (greedy, top-k, top-p, beam, speculative) | Planned |

### Forward-Pass Scripts

`microattention.py` and `microflash.py` are **forward-pass comparisons** — they do not train models. This is an intentional exception to the train+infer rule: the pedagogical value is in comparing implementations side-by-side, not in demonstrating training.

### Algorithmic Simulations

`microflash.py` is an **algorithmic simulation** of Flash Attention. Pure Python will be slower than standard attention. The script demonstrates *what* Flash Attention does (tiled computation, online softmax), not *why* it's fast in practice (GPU memory hierarchy optimization). Comments make this distinction explicit.

## Future Candidates

| Algorithm | What It Would Teach | Notes |
|---|---|---|
| **Speculative Decoding (standalone)** | Draft-verify paradigm in depth | Currently part of microbeam; could be its own deep-dive |
| **Model Parallelism** | Tensor parallelism, pipeline parallelism concepts | Algorithmic simulation of distributed inference |
| **Continuous Batching** | Dynamic batching for throughput optimization | The technique behind vLLM's performance |
| **Prefix Caching** | Sharing KV-cache across requests with common prefixes | Extension of microkv concepts |
| **Activation Checkpointing** | Trading compute for memory during training | Gradient checkpointing from scratch |
| **Mixed Precision** | FP16/BF16 training with loss scaling | How half-precision training works |

## Learning Path

These scripts can be studied in any order, but this sequence builds concepts incrementally:

```
microattention.py   → How attention actually works (all variants)
microkv.py          → Why LLM inference is memory-bound
microflash.py       → How attention gets fast (tiling + online softmax)
microquant.py       → How models get compressed (INT8/INT4)
microbeam.py        → How decoding strategies shape output quality
```
