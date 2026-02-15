# Foundations

Core algorithms that form the building blocks of modern AI systems. These are the primitives — if you understand these, everything else is composition.

## Scripts

| Script              | Algorithm                                                                 | Run Time | Status   |
| ------------------- | ------------------------------------------------------------------------- | -------- | -------- |
| `microbert.py`      | Bidirectional transformer encoder (BERT) with masked language modeling    | 4m 34s   | Complete |
| `microconv.py`      | Convolutional Neural Network — kernels, pooling, and feature maps         | 0m 31s   | Complete |
| `microdiffusion.py` | Denoising diffusion on 2D point clouds                                    | 0m 41s   | Complete |
| `microembedding.py` | Contrastive embedding learning (InfoNCE)                                  | 0m 44s   | Complete |
| `microgan.py`       | Generative Adversarial Network — generator vs. discriminator minimax game | 2m 02s   | Complete |
| `microgpt.py`       | Autoregressive language model (GPT) with scalar autograd                  | 1m 41s   | Complete |
| `microoptimizer.py` | Optimizer comparison — SGD vs. Momentum vs. RMSProp vs. Adam              | 0m 34s   | Complete |
| `microrag.py`       | Retrieval-Augmented Generation (BM25 + MLP)                               | 12m 30s  | Complete |
| `micrornn.py`       | Vanilla RNN vs. GRU — vanishing gradients and gating                      | 18m 30s  | Complete |
| `microtokenizer.py` | Byte-Pair Encoding (BPE) tokenization                                     | 0m 12s   | Complete |
| `microvae.py`       | Variational Autoencoder with reparameterization trick                     | 1m 31s   | Complete |

## Test Results

Measured on Apple M-series, Python 3.12. Times are wall-clock.

| Script              | Status | Time    |
| ------------------- | ------ | ------- |
| `microbert.py`      | Pass   | 4m 34s  |
| `microconv.py`      | Pass   | 0m 31s  |
| `microdiffusion.py` | Pass   | 0m 41s  |
| `microembedding.py` | Pass   | 0m 44s  |
| `microgan.py`       | Pass   | 2m 02s  |
| `microgpt.py`       | Pass   | 1m 41s  |
| `microoptimizer.py` | Pass   | 0m 34s  |
| `microrag.py`       | Pass   | 12m 30s |
| `micrornn.py`       | Pass   | 18m 30s |
| `microtokenizer.py` | Pass   | 0m 12s  |
| `microvae.py`       | Pass   | 1m 31s  |

## Future Candidates

These algorithms are strong candidates for future addition. Each would need to meet the project constraints (single file, zero dependencies, trains and infers, under 10 minutes on CPU).

| Algorithm    | What It Would Teach                                 | Notes                                                          |
| ------------ | --------------------------------------------------- | -------------------------------------------------------------- |
| **LSTM**     | Long Short-Term Memory gating (3 gates vs. GRU's 2) | Could extend micrornn.py or be standalone                      |
| **Word2Vec** | Skip-gram with negative sampling                    | Classic embedding algorithm, simpler than contrastive learning |

## Learning Path

For a guided walkthrough of the foundations tier, follow this order:

```plaintext
microtokenizer.py   → How text becomes numbers
microembedding.py   → How meaning becomes geometry
microgpt.py         → How sequences become predictions
micrornn.py         → How sequences were modeled before attention
microconv.py        → How spatial features get extracted by sliding kernels
microbert.py        → How bidirectional context differs from autoregressive
microrag.py         → How retrieval augments generation
microoptimizer.py   → How optimizer choice shapes convergence
microgan.py         → How two networks learn by competing
microdiffusion.py   → How data emerges from noise
microvae.py         → How to learn compressed generative representations
```
