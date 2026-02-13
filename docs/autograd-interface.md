# Canonical Autograd Interface

This document defines the scalar autograd `Value` class interface that all scripts must implement. The interface ensures consistency across the 8+ scripts that use scalar automatic differentiation while allowing per-script extensions.

## Why This Exists

Each script is self-contained (no shared imports), so the `Value` class is reimplemented in every script that needs it. Without a canonical interface:

- Implementations drift (one supports `sigmoid`, another doesn't)
- Numerical stability patterns are applied inconsistently
- Readers who skip the autograd section in later scripts miss per-script differences

This spec defines the **minimum** interface. Scripts may add operations (documented via the autograd callout pattern below).

---

## Required Operations

Every `Value` class must support these operations:

### Arithmetic

| Operation      | Python Method         | Notes                                             |
| -------------- | --------------------- | ------------------------------------------------- |
| Addition       | `__add__`, `__radd__` | `Value + Value`, `Value + float`, `float + Value` |
| Multiplication | `__mul__`, `__rmul__` | `Value * Value`, `Value * float`, `float * Value` |
| Negation       | `__neg__`             | `-Value` (implemented as `self * -1`)             |
| Subtraction    | `__sub__`, `__rsub__` | Via `__add__` and `__neg__`                       |
| Division       | `__truediv__`         | Via `__mul__` and `__pow__(-1)`                   |
| Power          | `__pow__`             | `Value ** int` or `Value ** float`                |

### Activations

| Function | Signature              | Backward                                        |
| -------- | ---------------------- | ----------------------------------------------- |
| `tanh()` | `self.tanh() -> Value` | `grad * (1 - out**2)`                           |
| `exp()`  | `self.exp() -> Value`  | `grad * out`                                    |
| `relu()` | `self.relu() -> Value` | `grad * (1 if self.data > 0 else 0)`            |
| `log()`  | `self.log() -> Value`  | `grad / self.data` (clamp `self.data >= 1e-10`) |

### Backward Pass

```python
def backward(self):
    """Compute gradients via reverse-mode autodiff (topological sort)."""
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

### Gradient Management

```python
# Before each training step, zero all gradients
for p in params:
    p.grad = 0.0
```

---

## Per-Script Extensions

Scripts that need additional operations beyond the base set must:

1. Implement the extension in the `Value` class
2. Document it with the autograd callout pattern (see below)

### Known Extensions by Script

| Script         | Additional Operations | Why Needed                                     |
| -------------- | --------------------- | ---------------------------------------------- |
| `micrornn.py`  | `sigmoid()`           | GRU gating: `z_t = sigmoid(...)`               |
| `microlora.py` | (none beyond base)    | Uses base set                                  |
| `microdpo.py`  | `log()`               | Log-probability ratios in DPO loss             |
| `microppo.py`  | `log()`, `clip()`     | PPO ratio clipping, log-probs                  |
| `micromoe.py`  | (router only)         | Router uses base set; experts are plain floats |

### Autograd Callout Pattern

Every script using the `Value` class must include this block immediately after the class definition:

```python
# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following additions/modifications:
# - sigmoid(): Required for GRU gating (z_t and r_t computations)
# - [list any other additions]
# Base operations (add, mul, tanh, exp, relu, pow, backward) are identical
# to the canonical spec.
```

For scripts with NO additions:

```python
# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.
```

---

## Numerical Stability Patterns

These patterns are **mandatory** in every script that uses them. Each must be accompanied by a comment explaining the numerical reasoning.

### Stable Softmax

```python
def softmax(logits):
    # Numerically stable softmax: subtract max before exp to prevent overflow.
    # softmax is translation-invariant: softmax(x) = softmax(x - c) for any c.
    # Without this, exp(x) overflows for x > 709 (Python's math.exp limit).
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]
```

### Clipped Log-Probability

```python
def safe_log(x):
    # Prevent log(0) which returns -inf and breaks gradient computation.
    # Clipping to 1e-10 gives log(1e-10) ≈ -23, which is finite and
    # preserves gradient information for near-zero probabilities.
    #
    # Critical: we build the log node manually with x as its child so gradients
    # flow back through the computation graph. Using Value(clamped).log() would
    # create a disconnected node, severing the gradient path entirely.
    clamped = max(x.data, 1e-10)
    return Value(math.log(clamped), (x,), (1.0 / clamped,))
```

### Adam Epsilon

```python
def adam_step(param, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # eps prevents division by zero when v (second moment) is near zero.
    # Standard value: 1e-8 (matches PyTorch/TensorFlow defaults).
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    param.data -= lr * m / (v ** 0.5 + eps)
```

### KL Divergence Clamping (microvae only)

```python
def kl_divergence(mean, log_var):
    # Clamp log_var to [-5, 5] to prevent exp(log_var) explosion.
    # exp(5) = 148 (reasonable variance); exp(10) = 22,026 (KL blows up).
    clamped_lv = max(min(log_var.data, 5.0), -5.0)
    return Value(0.5) * (Value(1.0) + Value(clamped_lv) - mean * mean - Value(clamped_lv).exp())
```

---

## Test Vectors

Use these to verify your `Value` class implementation produces correct gradients:

### Test 1: Simple chain

```python
a = Value(2.0)
b = Value(3.0)
c = a * b + b  # c = 2*3 + 3 = 9
c.backward()
assert a.grad == 3.0   # dc/da = b = 3
assert b.grad == 3.0   # dc/db = a + 1 = 3
```

### Test 2: Tanh gradient

```python
x = Value(0.5)
y = x.tanh()  # y = tanh(0.5) = 0.4621
y.backward()
# dy/dx = 1 - tanh(0.5)^2 = 1 - 0.2135 = 0.7865
assert abs(x.grad - 0.7865) < 0.001
```

### Test 3: Reuse (gradient accumulation)

```python
a = Value(2.0)
b = a + a  # a is used twice
b.backward()
assert a.grad == 2.0  # db/da = 1 + 1 = 2
```

### Test 4: Softmax stability

```python
logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
probs = softmax(logits)
# Should NOT overflow. Expected: ~[0.09, 0.24, 0.67]
assert all(0 < p.data < 1 for p in probs)
assert abs(sum(p.data for p in probs) - 1.0) < 1e-6
```

---

## Implementation Notes

- **Python object overhead:** Each `Value` stores `.data` (float), `.grad` (float), `._backward` (closure), `._prev` (set). Approximate memory: ~100 bytes per Value.
- **Parameter budget:** The 7-minute runtime constraint effectively limits total model parameters to ~5,000 Value objects per script. Scripts exceeding this (microppo, micromoe) use hybrid autograd.
- **Determinism:** `set` iteration order varies across Python sessions due to hash randomization. For strict reproducibility, document `PYTHONHASHSEED=0` in script headers. For pedagogical purposes, slight numerical variation across runs is acceptable.
- **Gradient zeroing:** Must happen before every `backward()` call. Forgetting this is the most common autograd bug — gradients accumulate across training steps otherwise.
