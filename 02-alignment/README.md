# Alignment & Training Techniques

Methods for steering, fine-tuning, and aligning models after pretraining. These are the techniques that turn a base model into something useful.

## Scripts

| Script              | Algorithm                                                             | Run Time | Status   |
| ------------------- | --------------------------------------------------------------------- | -------- | -------- |
| `microbatchnorm.py` | Batch Normalization — internal covariate shift and running statistics | 0m 34s   | Complete |
| `microdpo.py`       | Direct Preference Optimization                                        | 2m 42s   | Complete |
| `microdropout.py`   | Dropout, weight decay, and early stopping as regularization           | 3m 21s   | Complete |
| `microgrpo.py`      | Group Relative Policy Optimization (DeepSeek's RLHF simplification)   | 0m 23s   | Complete |
| `microlora.py`      | Low-Rank Adaptation (LoRA) fine-tuning                                | 2m 32s   | Complete |
| `micromoe.py`       | Mixture of Experts with sparse routing (hybrid autograd)              | 0m 06s   | Complete |
| `microppo.py`       | Proximal Policy Optimization for RLHF (hybrid autograd)               | 0m 34s   | Complete |
| `microqlora.py`     | QLoRA — fine-tuning 4-bit quantized models with LoRA adapters         | 2m 27s   | Complete |
| `microreinforce.py` | REINFORCE — vanilla policy gradient with baseline                     | 5m 39s   | Complete |

### Hybrid Autograd Scripts

`microppo.py` and `micromoe.py` use a **hybrid autograd approach** to meet runtime constraints:

- **microppo:** Policy model uses scalar autograd (`Value` class). Reward model and value function use plain float arrays with manual gradients — they're trained separately before the PPO loop.
- **micromoe:** Router uses scalar autograd. Expert MLPs use plain float arrays — the routing decision is the novel mechanism, not the expert forward pass.

See `docs/autograd-interface.md` for the canonical interface and `docs/implementation.md` for per-script details.

## Test Results

Measured on Apple M-series, Python 3.12. Times are wall-clock.

| Script              | Status | Time   |
| ------------------- | ------ | ------ |
| `microbatchnorm.py` | Pass   | 0m 34s |
| `microdpo.py`       | Pass   | 2m 42s |
| `microdropout.py`   | Pass   | 3m 21s |
| `microgrpo.py`      | Pass   | 0m 23s |
| `microlora.py`      | Pass   | 2m 32s |
| `micromoe.py`       | Pass   | 0m 06s |
| `microppo.py`       | Pass   | 0m 34s |
| `microqlora.py`     | Pass   | 2m 27s |
| `microreinforce.py` | Pass   | 5m 39s |

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
