# Alignment & Training Techniques

Methods for steering, fine-tuning, and aligning models after pretraining. These are the techniques that turn a base model into something useful.

## Planned Scripts

| Script | Algorithm | Status |
|---|---|---|
| `microlora.py` | Low-Rank Adaptation (LoRA) fine-tuning | Planned |
| `microdpo.py` | Direct Preference Optimization | Planned |
| `microppo.py` | Proximal Policy Optimization for RLHF (hybrid autograd) | Planned |
| `micromoe.py` | Mixture of Experts with sparse routing (hybrid autograd) | Planned |

### Hybrid Autograd Scripts

`microppo.py` and `micromoe.py` use a **hybrid autograd approach** to meet runtime constraints:

- **microppo:** Policy model uses scalar autograd (`Value` class). Reward model and value function use plain float arrays with manual gradients — they're trained separately before the PPO loop.
- **micromoe:** Router uses scalar autograd. Expert MLPs use plain float arrays — the routing decision is the novel mechanism, not the expert forward pass.

See `docs/autograd-interface.md` for the canonical interface and `docs/implementation.md` for per-script details.

## Future Candidates

| Algorithm | What It Would Teach | Notes |
|---|---|---|
| **REINFORCE** | Vanilla policy gradient with baseline | Simpler RL alternative, foundation for understanding PPO |
| **Dropout / Regularization** | Why random neuron deactivation prevents overfitting | Could cover dropout, weight decay, and early stopping in one file |
| **Batch Normalization** | Internal covariate shift, running statistics | The technique that made deep networks trainable |
| **Learning Rate Scheduling** | Warmup, cosine decay, step decay | How schedule choice affects convergence |
| **Knowledge Distillation** | Training small models to mimic large ones | Compression via soft targets |

## Learning Path

These scripts build on the foundational tier. Recommended order:

```
microlora.py   → How fine-tuning works efficiently (1% of parameters)
microdpo.py    → How preference alignment works (without reward model)
microppo.py    → How RLHF works (the full reward → policy loop)
micromoe.py    → How sparse routing scales model capacity
```
