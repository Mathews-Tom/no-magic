# Algorithm Visualization Videos

Short (10–30s) animated explanations of the algorithms in `no-magic`, generated programmatically with [Manim](https://www.manim.community/).

## Structure

```
videos/
├── scenes/          ← Manim source files (version-controlled)
│   ├── base.py      ← Shared base class, color palette
│   ├── scene_microattention.py
│   ├── scene_microgpt.py
│   └── ...
├── previews/        ← GIF previews for README embedding
├── renders/         ← Full MP4s (local only, not committed)
├── render.sh        ← Batch render script
└── requirements.txt ← Pinned manim version
```

## Rendering

```bash
# Install dependencies (one-time)
pip install -r videos/requirements.txt

# Render all scenes (MP4 + GIF)
bash videos/render.sh

# Render a single scene
bash videos/render.sh microattention

# Preview GIFs only (fast)
bash videos/render.sh --preview-only

# Full MP4s only
bash videos/render.sh --full-only
```

## Contributing a Scene

1. Create `videos/scenes/scene_<algorithm>.py`
2. Inherit from `NoMagicScene` in `base.py`
3. Set `title_text` and `subtitle_text` class attributes
4. Implement `animate()` with the algorithm visualization
5. Test with `manim -ql videos/scenes/scene_<algorithm>.py <ClassName>`

Every scene follows the same structure: title card → setup → core animation → result → end card.

## Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Background | `#1a1a2e` | Dark navy (GitHub dark mode) |
| Primary | `#e94560` | Red/coral — highlights, emphasis |
| Blue | `#0f3460` | Deep blue — secondary elements |
| Green | `#16c79a` | Active/flowing states |
| Text | `#eaeaea` | Light gray labels |
| Grid | `#2a2a4a` | Subtle guides and gridlines |
| Yellow | `#f5c542` | Tertiary accent |
| Orange | `#e97d32` | Tertiary accent |
| Purple | `#9b59b6` | Tertiary accent |

## Video Index

### Tier 1 (complete)

| Algorithm | Scene | Duration | Status |
|-----------|-------|----------|--------|
| Attention Mechanism | `scene_microattention.py` | ~20s | Done |
| Autoregressive GPT | `scene_microgpt.py` | ~18s | Done |
| LoRA Fine-tuning | `scene_microlora.py` | ~19s | Done |
| Word Embeddings | `scene_microembedding.py` | ~18s | Done |
| DPO Alignment | `scene_microdpo.py` | ~17s | Done |
| RAG Pipeline | `scene_microrag.py` | ~17s | Done |
| Flash Attention | `scene_microflash.py` | ~19s | Done |
| Quantization | `scene_microquant.py` | ~15s | Done |
| Mixture of Experts | `scene_micromoe.py` | ~18s | Done |

### Tier 2 (complete)

| Algorithm | Scene | Duration | Status |
|-----------|-------|----------|--------|
| BPE Tokenizer | `scene_microtokenizer.py` | ~18s | Done |
| BERT (Bidirectional) | `scene_microbert.py` | ~17s | Done |
| KV-Cache | `scene_microkv.py` | ~16s | Done |
| Beam Search | `scene_microbeam.py` | ~16s | Done |
| RoPE (Rotary Position) | `scene_microrope.py` | ~17s | Done |
| PPO (RLHF) | `scene_microppo.py` | ~18s | Done |
| State Space Models | `scene_microssm.py` | ~18s | Done |

### Tier 3 (complete)

| Algorithm | Scene | Duration | Status |
|-----------|-------|----------|--------|
| Batch Normalization | `scene_microbatchnorm.py` | ~17s | Done |
| Convolutional Neural Net | `scene_microconv.py` | ~18s | Done |
| Denoising Diffusion | `scene_microdiffusion.py` | ~17s | Done |
| GAN (Adversarial) | `scene_microgan.py` | ~19s | Done |
| Optimizer Comparison | `scene_microoptimizer.py` | ~18s | Done |
| RNN vs GRU | `scene_micrornn.py` | ~17s | Done |
| Variational Autoencoder | `scene_microvae.py` | ~17s | Done |
| Dropout | `scene_microdropout.py` | ~18s | Done |
| GRPO | `scene_microgrpo.py` | ~16s | Done |
| QLoRA | `scene_microqlora.py` | ~17s | Done |
| REINFORCE | `scene_microreinforce.py` | ~16s | Done |
| Activation Checkpointing | `scene_microcheckpoint.py` | ~17s | Done |
| PagedAttention | `scene_micropaged.py` | ~17s | Done |
| Model Parallelism | `scene_microparallel.py` | ~18s | Done |
