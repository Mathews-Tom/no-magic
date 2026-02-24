# Alignment & Training Techniques

Methods for steering, fine-tuning, and aligning models after pretraining. These are the techniques that turn a base model into something useful.

## Scripts

Measured on Apple M-series, Python 3.12. Times are wall-clock.

| Script              | Algorithm                                                             | Time   | Status | Video                                             |
| ------------------- | --------------------------------------------------------------------- | ------ | ------ | ------------------------------------------------- |
| `microbatchnorm.py` | Batch Normalization — internal covariate shift and running statistics | 0m 34s | Pass   | ![Preview](../videos/previews/microbatchnorm.gif) |
| `microdpo.py`       | Direct Preference Optimization                                        | 2m 42s | Pass   | ![Preview](../videos/previews/microdpo.gif)       |
| `microdropout.py`   | Dropout, weight decay, and early stopping as regularization           | 3m 21s | Pass   | ![Preview](../videos/previews/microdropout.gif)   |
| `microgrpo.py`      | Group Relative Policy Optimization (DeepSeek's RLHF simplification)   | 0m 23s | Pass   | ![Preview](../videos/previews/microgrpo.gif)      |
| `microlora.py`      | Low-Rank Adaptation (LoRA) fine-tuning                                | 2m 32s | Pass   | ![Preview](../videos/previews/microlora.gif)      |
| `micromoe.py`       | Mixture of Experts with sparse routing (hybrid autograd)              | 0m 06s | Pass   | ![Preview](../videos/previews/micromoe.gif)       |
| `microppo.py`       | Proximal Policy Optimization for RLHF (hybrid autograd)               | 0m 34s | Pass   | ![Preview](../videos/previews/microppo.gif)       |
| `microqlora.py`     | QLoRA — fine-tuning 4-bit quantized models with LoRA adapters         | 2m 27s | Pass   | ![Preview](../videos/previews/microqlora.gif)     |
| `microreinforce.py` | REINFORCE — vanilla policy gradient with baseline                     | 5m 39s | Pass   | ![Preview](../videos/previews/microreinforce.gif) |

### Hybrid Autograd Scripts

`microppo.py` and `micromoe.py` use a **hybrid autograd approach** to meet runtime constraints:

- **microppo:** Policy model uses scalar autograd (`Value` class). Reward model and value function use plain float arrays with manual gradients — they're trained separately before the PPO loop.
- **micromoe:** Router uses scalar autograd. Expert MLPs use plain float arrays — the routing decision is the novel mechanism, not the expert forward pass.

See `docs/autograd-interface.md` for the canonical interface and `docs/implementation.md` for per-script details.

## Future Candidates

| Algorithm                    | What It Would Teach                       | Notes                                   |
| ---------------------------- | ----------------------------------------- | --------------------------------------- |
| **Learning Rate Scheduling** | Warmup, cosine decay, step decay          | How schedule choice affects convergence |
| **Knowledge Distillation**   | Training small models to mimic large ones | Compression via soft targets            |

## Learning Path

These scripts build on the foundations tier. Recommended order:

```
microbatchnorm.py   → How normalizing activations stabilizes training
microdropout.py     → How regularization prevents overfitting
microlora.py        → How fine-tuning works efficiently (1% of parameters)
microqlora.py       → How quantization combines with LoRA for memory efficiency
microreinforce.py   → How policy gradients turn rewards into learning signals
microdpo.py         → How preference alignment works (without reward model)
microppo.py         → How RLHF works (the full reward → policy loop)
microgrpo.py        → How DeepSeek simplified RLHF with group-relative rewards
micromoe.py         → How sparse routing scales model capacity
```
