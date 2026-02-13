# Foundational

Core algorithms that form the building blocks of modern AI systems. These are the primitives — if you understand these, everything else is composition.

## Planned Scripts

| Script | Algorithm | Status |
|---|---|---|
| `microgpt.py` | Autoregressive language model (GPT) with scalar autograd | Planned |
| `micrornn.py` | Vanilla RNN vs. GRU — vanishing gradients and gating | Planned |
| `microtokenizer.py` | Byte-Pair Encoding (BPE) tokenization | Planned |
| `microembedding.py` | Contrastive embedding learning (InfoNCE) | Planned |
| `microrag.py` | Retrieval-Augmented Generation (BM25 + MLP) | Planned |
| `microdiffusion.py` | Denoising diffusion on 2D point clouds | Planned |
| `microvae.py` | Variational Autoencoder with reparameterization trick | Planned |

## Future Candidates

These algorithms are strong candidates for future addition. Each would need to meet the project constraints (single file, zero dependencies, trains and infers, under 7 minutes on CPU).

| Algorithm | What It Would Teach | Notes |
|---|---|---|
| **LSTM** | Long Short-Term Memory gating (3 gates vs. GRU's 2) | Could extend micrornn.py or be standalone |
| **GAN** | Generative Adversarial Networks on 2D data | Generator vs. discriminator dynamics, mode collapse, training instability |
| **Transformer Encoder (BERT-style)** | Masked language modeling, bidirectional attention | Contrasts with microgpt (decoder-only) |
| **ConvNet** | Convolution from scratch on tiny images | Kernel sliding, pooling, feature maps — the vision primitive |
| **Optimizer Comparison** | SGD vs. Momentum vs. Adam side-by-side | Convergence dynamics, adaptive learning rates |
| **Word2Vec** | Skip-gram with negative sampling | Classic embedding algorithm, simpler than contrastive learning |

## Learning Path

For a guided walkthrough of the foundational tier, follow this order:

```
microtokenizer.py   → How text becomes numbers
microembedding.py   → How meaning becomes geometry
microgpt.py         → How sequences become predictions
microrag.py         → How retrieval augments generation
micrornn.py         → How sequences were modeled before attention
microdiffusion.py   → How data emerges from noise
microvae.py         → How to learn compressed generative representations
```
