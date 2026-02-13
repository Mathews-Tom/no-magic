# no-magic

**Because `model.fit()` isn't an explanation.**

---

## What This Is

`no-magic` is a curated collection of single-file, dependency-free Python implementations of the algorithms that power modern AI. Each script is a complete, runnable program that trains a model from scratch and performs inference — no frameworks, no abstractions, no hidden complexity.

Every script in this repository is an **executable proof** that these algorithms are simpler than the industry makes them seem. The goal is not to replace PyTorch or TensorFlow — it's to make you dangerous enough to understand what they're doing underneath.

## Philosophy

Modern ML education has a gap. There are thousands of tutorials that teach you to call library functions, and there are academic papers full of notation. What's missing is the middle layer: **the algorithm itself, expressed as readable code**.

This project follows a strict set of constraints:

- **One file, one algorithm.** Every script is completely self-contained. No imports from local modules, no `utils.py`, no shared libraries.
- **Zero external dependencies.** Only Python's standard library. If it needs `pip install`, it doesn't belong here.
- **Train and infer.** Every script includes both the learning loop and generation/prediction. You see the full lifecycle.
- **Runs in minutes on a CPU.** No GPU required. No cloud credits. Every script completes on a laptop in reasonable time.
- **Comments are mandatory, not decorative.** Every script must be readable as a guided walkthrough of the algorithm. We are not optimizing for line count — we are optimizing for understanding. See `CONTRIBUTING.md` for the full commenting standard.

## Who This Is For

- **ML engineers** who use frameworks daily but want to understand the internals they rely on.
- **Students** transitioning from theory to practice who want to see algorithms as working code, not just equations.
- **Career switchers** entering ML who need intuition for what's actually happening when they call high-level APIs.
- **Researchers** who want minimal reference implementations to prototype ideas without framework overhead.
- **Anyone** who has ever stared at a library call and thought: _"but what is it actually doing?"_

This is not a beginner's introduction to programming. You should be comfortable reading Python and have at least a surface-level familiarity with ML concepts. The scripts will give you the depth.

## What You'll Find Here

The repository is organized into three tiers based on conceptual dependency:

### 01 — Foundations

Core algorithms that form the building blocks of modern AI systems. These are the primitives — if you understand these, everything else is composition.

See [`01-foundations/README.md`](01-foundations/README.md) for the full algorithm list and roadmap.

### 02 — Alignment & Training Techniques

Methods for steering, fine-tuning, and aligning models after pretraining. These are the techniques that turn a base model into something useful.

See [`02-alignment/README.md`](02-alignment/README.md) for the full algorithm list and roadmap.

### 03 — Systems & Inference

The engineering that makes models fast, small, and deployable. These scripts demystify the optimizations that turn research prototypes into production systems.

See [`03-systems/README.md`](03-systems/README.md) for the full algorithm list and roadmap.

## How to Use This Repo

```bash
# Clone the repository
git clone https://github.com/your-username/no-magic.git
cd no-magic

# Pick any script and run it
python 01-foundations/microgpt.py
```

That's it. No virtual environments, no dependency installation, no configuration. Each script will download any small datasets it needs on first run.

### Minimum Requirements

- Python 3.10+
- 8 GB RAM
- Any modern CPU (2019-era or newer)

### Suggested Learning Path

If you're working through the scripts systematically, this order builds concepts incrementally:

```plaintext
microtokenizer.py     → How text becomes numbers
microembedding.py     → How meaning becomes geometry
microgpt.py        → How sequences become predictions
microrag.py           → How retrieval augments generation
microattention.py     → How attention actually works (all variants)
microlora.py          → How fine-tuning works efficiently
microdpo.py           → How preference alignment works
microquant.py         → How models get compressed
microflash.py         → How attention gets fast
```

For comprehensive coverage of all algorithms, see `docs/implementation.md`.

## Inspiration & Attribution

This project is directly inspired by [Andrej Karpathy's](https://github.com/karpathy) extraordinary work on minimal implementations — particularly [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), and the `microgpt.py` script that demonstrated the entire GPT algorithm in a single dependency-free Python file.

Karpathy proved that there's enormous demand for "the algorithm, naked." `no-magic` extends that philosophy across the full landscape of modern AI/ML.

## Contributing

Contributions are welcome, but the constraints are non-negotiable. See `CONTRIBUTING.md` for the full guidelines. The short version:

- One file. Zero dependencies. Trains and infers.
- If your PR adds a `requirements.txt`, it will be closed.
- Quality over quantity. Each script should be the **best possible** minimal implementation of its algorithm.

## License

MIT — use these however you want. Learn from them, teach with them, build on them.

---

_The constraint is the product. Everything else is just efficiency._
