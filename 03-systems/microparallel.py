"""
How models that exceed single-device memory get distributed — tensor parallelism, pipeline
parallelism, and communication costs, demonstrated end-to-end on a 4-layer MLP.
"""
# Reference: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models
# Using Model Parallelism" (2019). https://arxiv.org/abs/1909.08053
# Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline
# Parallelism" (2019). https://arxiv.org/abs/1811.06965

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# 4-layer MLP for 2D classification. 4 layers is the minimum where pipeline
# splitting (2+2) is meaningful and tensor splitting demonstrates all-reduce cost.
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 2
N_LAYERS = 4

LEARNING_RATE = 0.1
NUM_STEPS = 300
BATCH_SIZE = 16
N_SAMPLES = 200
N_MICRO_BATCHES = 2  # splits each batch to reduce pipeline bubble

# Signpost: production models use HIDDEN_DIM in the thousands (GPT-3: 12288) across
# hundreds of GPUs. Our 16-wide hidden dim preserves the algorithmic structure —
# splitting, all-reducing, pipeline staging — while keeping runtime under a minute.


# === SCALAR AUTOGRAD ENGINE ===

# Each strategy builds its computation graph from these same primitives — the
# parallelism is in how weights are partitioned and how values move between "devices."

class Value:
    """Scalar with reverse-mode automatic differentiation."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * (other ** -1)
    def __rtruediv__(self, other: float) -> Value: return other * (self ** -1)

    def relu(self) -> Value:
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def exp(self) -> Value:
        e = math.exp(min(self.data, 80.0))  # clamp prevents overflow
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        clamped = max(self.data, 1e-12)
        return Value(math.log(clamped), (self,), (1.0 / clamped,))

    def backward(self) -> None:
        """Reverse-mode autodiff: topological sort then chain-rule propagation."""
        topo: list[Value] = []
        visited: set[int] = set()
        def build_topo(v: Value) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad  # chain rule: dL/dchild += dL/dv * dv/dchild


# === DATASET: CONCENTRIC RINGS ===

# Two concentric rings in 2D — non-linearly separable (no straight line can divide
# an inner circle from an outer ring). Requires combining multiple ReLU regions.

def make_rings_data(n: int) -> tuple[list[list[float]], list[int]]:
    """Generate concentric rings (class 0 = inner, class 1 = outer)."""
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n // 2):
        # Inner ring: radius ~0.3
        a = random.uniform(0, 2 * math.pi)
        r = 0.3 + random.gauss(0, 0.08)
        xs.append([r * math.cos(a), r * math.sin(a)])
        ys.append(0)
        # Outer ring: radius ~0.8
        a = random.uniform(0, 2 * math.pi)
        r = 0.8 + random.gauss(0, 0.08)
        xs.append([r * math.cos(a), r * math.sin(a)])
        ys.append(1)
    return xs, ys


# === WEIGHT INITIALIZATION AND PRIMITIVES ===

# Kaiming init: std = sqrt(2 / fan_in) keeps activation variance roughly constant
# across layers, preventing vanishing/exploding gradients in the 4-layer network.

def make_weights(rows: int, cols: int) -> list[list[Value]]:
    std = math.sqrt(2.0 / rows)
    return [[Value(random.gauss(0, std)) for _ in range(cols)] for _ in range(rows)]

def make_bias(size: int) -> list[Value]:
    return [Value(0.0) for _ in range(size)]

def linear_forward(x: list[Value], W: list[list[Value]], b: list[Value]) -> list[Value]:
    """y = x @ W + b. The fundamental building block all strategies share."""
    result = []
    for j in range(len(b)):
        acc = b[j]
        for i in range(len(x)):
            acc = acc + x[i] * W[i][j]  # dot product: sum_i(x_i * W_ij)
        result.append(acc)
    return result

def relu_forward(x: list[Value]) -> list[Value]:
    return [v.relu() for v in x]

def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = exps[0]
    for e in exps[1:]:
        total = total + e
    return [e / total for e in exps]

def cross_entropy_loss(probs: list[Value], target: int) -> Value:
    return -probs[target].log()


# === COMMUNICATION TRACKER ===

# Core instrumentation making parallelism costs visible. In real distributed
# training, communication is the bottleneck — compute is fast, interconnect
# bandwidth (NVLink, InfiniBand) is limited.

class CommTracker:
    """Track inter-device communication: rounds and floats transferred."""
    def __init__(self) -> None:
        self.rounds = 0
        self.floats_transferred = 0

    def transfer(self, n_floats: int) -> None:
        self.rounds += 1
        self.floats_transferred += n_floats


# === DEVICE ABSTRACTION ===

# A "device" is a dict holding weights. In real systems, each device is a GPU
# with its own HBM. Moving data between dicts is trivial in Python, but in
# production it requires NCCL all-reduce over NVLink and dominates training time.

def make_layer_dims() -> list[int]:
    return [INPUT_DIM] + [HIDDEN_DIM] * (N_LAYERS - 1) + [OUTPUT_DIM]


# === STRATEGY 1: SINGLE-DEVICE BASELINE ===

# All weights on one device. No communication. The reference for measuring
# the cost of parallelism.

def init_single_device() -> tuple[dict, list]:
    device: dict = {"weights": {}}
    params: list = []
    dims = make_layer_dims()
    for i in range(N_LAYERS):
        W = make_weights(dims[i], dims[i + 1])
        b = make_bias(dims[i + 1])
        device["weights"][f"W{i}"] = W
        device["weights"][f"b{i}"] = b
        params.extend([v for row in W for v in row])
        params.extend(b)
    return device, params

def forward_single(device: dict, x: list[Value]) -> list[Value]:
    h = x
    for i in range(N_LAYERS):
        h = linear_forward(h, device["weights"][f"W{i}"], device["weights"][f"b{i}"])
        if i < N_LAYERS - 1:  # ReLU on hidden layers, raw logits on output
            h = relu_forward(h)
    return h


# === STRATEGY 2: TENSOR PARALLELISM ===

# Split each layer's weight matrix column-wise across 2 devices.
#
# Math: W = [W_0 | W_1] (column split)
#   y_0 = x @ W_0  (device 0 computes left half)
#   y_1 = x @ W_1  (device 1 computes right half)
#   y   = [y_0 | y_1] (all-gather reconstructs full vector)
#
# Each layer requires an all-gather so both devices have the full activation
# vector for the next layer's input. This is the communication cost.
#
# Signpost: Megatron-LM splits column-wise for the first MLP linear and row-wise
# for the second, avoiding one all-reduce per pair. Our simpler approach requires
# an all-gather after every layer but is easier to understand.

def init_tensor_parallel() -> tuple[list[dict], list]:
    dev0: dict = {"weights": {}}
    dev1: dict = {"weights": {}}
    all_params: list = []
    dims = make_layer_dims()
    for i in range(N_LAYERS):
        W_full = make_weights(dims[i], dims[i + 1])
        b_full = make_bias(dims[i + 1])
        half = dims[i + 1] // 2
        # Column split: device 0 gets [0..half), device 1 gets [half..out)
        dev0["weights"][f"W{i}"] = [[W_full[r][c] for c in range(half)] for r in range(dims[i])]
        dev1["weights"][f"W{i}"] = [[W_full[r][c] for c in range(half, dims[i+1])] for r in range(dims[i])]
        dev0["weights"][f"b{i}"] = b_full[:half]
        dev1["weights"][f"b{i}"] = b_full[half:]
        for row in W_full:
            all_params.extend(row)
        all_params.extend(b_full)
    return [dev0, dev1], all_params

def forward_tensor_parallel(
    devices: list[dict], x: list[Value], comm: CommTracker
) -> list[Value]:
    """Each layer: both devices compute half the columns, all-gather to reconstruct."""
    h = x
    dev0, dev1 = devices
    for i in range(N_LAYERS):
        partial0 = linear_forward(h, dev0["weights"][f"W{i}"], dev0["weights"][f"b{i}"])
        partial1 = linear_forward(h, dev1["weights"][f"W{i}"], dev1["weights"][f"b{i}"])
        # All-gather: each device sends its half to the other. In NCCL this is one
        # collective op. Cost: half_dim floats sent by each device = half_dim * 2 total.
        comm.transfer(len(partial0) * 2)
        h = partial0 + partial1  # list concatenation reconstructs full vector
        if i < N_LAYERS - 1:
            h = relu_forward(h)
    return h


# === STRATEGY 3: PIPELINE PARALLELISM ===

# Layers 0-1 on device 0, layers 2-3 on device 1. Activations flow forward
# (dev 0 → dev 1); gradients flow backward (dev 1 → dev 0).
#
# The fundamental problem: pipeline bubble. While dev 0 computes forward for
# layers 0-1, dev 1 is idle. While dev 1 computes backward for layers 2-3,
# dev 0 is idle.
#
# GPipe's solution: split the batch into micro-batches. Dev 0 forwards
# micro-batch 0, then immediately starts micro-batch 1. Dev 1 can begin
# micro-batch 0's forward while dev 0 works on micro-batch 1. More
# micro-batches → less idle time.

def init_pipeline_parallel() -> tuple[list[dict], list]:
    dev0: dict = {"weights": {}}
    dev1: dict = {"weights": {}}
    all_params: list = []
    dims = make_layer_dims()
    for i in range(N_LAYERS):
        W = make_weights(dims[i], dims[i + 1])
        b = make_bias(dims[i + 1])
        dev = dev0 if i < N_LAYERS // 2 else dev1
        dev["weights"][f"W{i}"] = W
        dev["weights"][f"b{i}"] = b
        for row in W:
            all_params.extend(row)
        all_params.extend(b)
    return [dev0, dev1], all_params

def forward_pipeline_stage(device: dict, x: list[Value], layer_range: range) -> list[Value]:
    """Forward through a contiguous subset of layers (one pipeline stage)."""
    h = x
    for i in layer_range:
        h = linear_forward(h, device["weights"][f"W{i}"], device["weights"][f"b{i}"])
        if i < N_LAYERS - 1:
            h = relu_forward(h)
    return h

def forward_pipeline(
    devices: list[dict], x: list[Value], comm: CommTracker
) -> list[Value]:
    """Stage 0 (layers 0-1) → send activations → stage 1 (layers 2-3)."""
    h = forward_pipeline_stage(devices[0], x, range(0, N_LAYERS // 2))
    comm.transfer(len(h))  # activation transfer at stage boundary
    return forward_pipeline_stage(devices[1], h, range(N_LAYERS // 2, N_LAYERS))


# === TRAINING LOOP ===

def train_step(params: list, loss: Value) -> None:
    """Backward + SGD update. SGD (not Adam) keeps memory tracking honest —
    the parallelism comparison is about communication, not optimizer state."""
    for p in params:
        p.grad = 0.0
    loss.backward()
    for p in params:
        p.data -= LEARNING_RATE * p.grad

def compute_batch_loss(forward_fn, batch_x: list[list[float]], batch_y: list[int]) -> Value:
    total_loss = Value(0.0)
    for x_raw, y in zip(batch_x, batch_y):
        x = [Value(v) for v in x_raw]
        probs = softmax(forward_fn(x))
        total_loss = total_loss + cross_entropy_loss(probs, y)
    return total_loss * (1.0 / len(batch_x))

def evaluate(forward_fn, xs: list[list[float]], ys: list[int]) -> float:
    correct = sum(1 for x_raw, y in zip(xs, ys)
                  if (0 if forward_fn([Value(v) for v in x_raw])[0].data >
                      forward_fn([Value(v) for v in x_raw])[1].data else 1) == y)
    return correct / len(ys)

def evaluate_fast(forward_fn, xs: list[list[float]], ys: list[int]) -> float:
    """Evaluate accuracy without double forward pass."""
    correct = 0
    for x_raw, y in zip(xs, ys):
        logits = forward_fn([Value(v) for v in x_raw])
        if (0 if logits[0].data > logits[1].data else 1) == y:
            correct += 1
    return correct / len(ys)

def train_strategy(
    name: str, forward_fn, params: list,
    xs: list[list[float]], ys: list[int], comm: CommTracker,
) -> tuple[float, float]:
    """Train for NUM_STEPS, return (final_accuracy, elapsed_seconds)."""
    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
    n = len(xs)
    t0 = time.time()
    for step in range(NUM_STEPS):
        indices = [random.randint(0, n - 1) for _ in range(BATCH_SIZE)]
        loss = compute_batch_loss(forward_fn, [xs[i] for i in indices], [ys[i] for i in indices])
        train_step(params, loss)
        if step % 50 == 0 or step == NUM_STEPS - 1:
            acc = evaluate_fast(forward_fn, xs, ys)
            print(f"  step {step:4d} | loss={loss.data:.4f} | acc={acc:.1%}"
                  f" | comm_rounds={comm.rounds}, floats={comm.floats_transferred}")
    elapsed = time.time() - t0
    final_acc = evaluate_fast(forward_fn, xs, ys)
    print(f"  Final accuracy: {final_acc:.1%} ({elapsed:.2f}s)")
    return final_acc, elapsed


# === PIPELINE PARALLEL WITH MICRO-BATCHING ===

# Instead of one forward-backward on the full batch, split into M micro-batches,
# accumulate gradients, then apply one update. Mathematically equivalent to
# full-batch but pipeline utilization improves from 50% to M/(M+1).

def train_pipeline_microbatch(
    devices: list[dict], params: list,
    xs: list[list[float]], ys: list[int],
    comm: CommTracker, n_micro: int,
) -> tuple[float, float]:
    print(f"\n{'=' * 60}\nTraining: Pipeline Parallel ({n_micro} micro-batches)\n{'=' * 60}")
    n = len(xs)
    t0 = time.time()
    dev0, dev1 = devices
    for step in range(NUM_STEPS):
        indices = [random.randint(0, n - 1) for _ in range(BATCH_SIZE)]
        bx = [xs[i] for i in indices]
        by = [ys[i] for i in indices]
        for p in params:
            p.grad = 0.0
        micro_size = BATCH_SIZE // n_micro
        total_loss_val = 0.0
        # In a real system, micro-batch k+1's stage-0 overlaps with micro-batch k's stage-1.
        # We simulate the gradient accumulation semantics, not the overlap timing.
        for mb in range(n_micro):
            mb_x = bx[mb * micro_size : (mb + 1) * micro_size]
            mb_y = by[mb * micro_size : (mb + 1) * micro_size]
            micro_loss = Value(0.0)
            for x_raw, y in zip(mb_x, mb_y):
                x = [Value(v) for v in x_raw]
                h = forward_pipeline_stage(dev0, x, range(0, N_LAYERS // 2))
                comm.transfer(len(h))  # activation transfer: dev 0 → dev 1
                logits = forward_pipeline_stage(dev1, h, range(N_LAYERS // 2, N_LAYERS))
                micro_loss = micro_loss + cross_entropy_loss(softmax(logits), y)
            micro_loss = micro_loss * (1.0 / len(mb_x))
            micro_loss.backward()  # accumulates into .grad (doesn't overwrite)
            total_loss_val += micro_loss.data
            comm.transfer(HIDDEN_DIM)  # gradient transfer: dev 1 → dev 0
        # Average accumulated gradients, then update
        for p in params:
            p.data -= LEARNING_RATE * p.grad / n_micro
        if step % 50 == 0 or step == NUM_STEPS - 1:
            fwd = lambda x: forward_pipeline(devices, x, CommTracker())
            acc = evaluate_fast(fwd, xs, ys)
            print(f"  step {step:4d} | loss={total_loss_val / n_micro:.4f} | acc={acc:.1%}"
                  f" | comm_rounds={comm.rounds}, floats={comm.floats_transferred}")
    elapsed = time.time() - t0
    fwd_eval = lambda x: forward_pipeline(devices, x, CommTracker())
    final_acc = evaluate_fast(fwd_eval, xs, ys)
    print(f"  Final accuracy: {final_acc:.1%} ({elapsed:.2f}s)")
    return final_acc, elapsed


# === PIPELINE BUBBLE ANALYSIS ===

# Bubble fraction = (K-1) / (K-1+M), where K=stages, M=micro-batches.
# Intuition: the first stage must finish one micro-batch before stage 2 can start.
# More micro-batches amortize this startup cost. In the limit (M → inf), bubble vanishes.

def compute_bubble_fraction(n_stages: int, n_micro: int) -> float:
    return (n_stages - 1) / (n_stages - 1 + n_micro)

def print_pipeline_schedule(n_stages: int, n_micro: int) -> None:
    """ASCII visualization: F=forward, B=backward, --=idle (the bubble)."""
    print(f"\n--- {n_stages} stages, {n_micro} micro-batches ---")
    total_fwd = n_stages + n_micro - 1
    total_slots = total_fwd * 2
    grid: list[list[str]] = [["    "] * total_slots for _ in range(n_stages)]
    # Forward: stage s processes micro-batch m at time s + m
    for m in range(n_micro):
        for s in range(n_stages):
            grid[s][s + m] = f"F{m:>2} "
    # Backward: mirror of forward, offset after all forwards complete
    for m in range(n_micro):
        for s in range(n_stages):
            grid[s][total_fwd + (n_stages - 1 - s) + m] = f"B{m:>2} "
    header = "Time -> " + "".join(f"{t:>4}" for t in range(total_slots))
    print(header)
    for s in range(n_stages):
        cells = "".join(f"[{grid[s][t]}]" if grid[s][t].strip() else "[ -- ]"
                        for t in range(total_slots))
        print(f"Dev {s}:  {cells}")
    print(f"Bubble: {compute_bubble_fraction(n_stages, n_micro):.0%}")


# === MAIN: RUN ALL STRATEGIES AND COMPARE ===

def main() -> None:
    print("=" * 60)
    print("  MODEL PARALLELISM SIMULATION")
    print("  4-layer MLP on concentric rings, 3 strategies compared")
    print("=" * 60)

    xs, ys = make_rings_data(N_SAMPLES)
    print(f"\nDataset: {N_SAMPLES} points (concentric rings), {OUTPUT_DIM} classes")
    print(f"Model: {N_LAYERS}-layer MLP, hidden_dim={HIDDEN_DIM}")
    print(f"Training: {NUM_STEPS} steps, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

    # Reset seed before each strategy so weight initialization is identical.
    # Any accuracy difference comes from parallelism mechanics (float ordering),
    # not from different random initializations.

    # --- Strategy 1: Single Device ---
    random.seed(42)
    xs, ys = make_rings_data(N_SAMPLES)
    random.seed(100)
    device_single, params_single = init_single_device()
    print(f"Parameters: {len(params_single)}")
    comm_none = CommTracker()
    fwd_single = lambda x: forward_single(device_single, x)
    acc_single, time_single = train_strategy(
        "Single Device (baseline)", fwd_single, params_single, xs, ys, comm_none)

    # --- Strategy 2: Tensor Parallel ---
    random.seed(100)
    devices_tp, params_tp = init_tensor_parallel()
    comm_tp = CommTracker()
    fwd_tp = lambda x: forward_tensor_parallel(devices_tp, x, comm_tp)
    acc_tp, time_tp = train_strategy(
        "Tensor Parallel (2 devices)", fwd_tp, params_tp, xs, ys, comm_tp)

    # --- Strategy 3: Pipeline Parallel ---
    random.seed(100)
    devices_pp, params_pp = init_pipeline_parallel()
    comm_pp = CommTracker()
    fwd_pp = lambda x: forward_pipeline(devices_pp, x, comm_pp)
    acc_pp, time_pp = train_strategy(
        "Pipeline Parallel (2 devices)", fwd_pp, params_pp, xs, ys, comm_pp)

    # --- Strategy 4: Pipeline + Micro-batching ---
    random.seed(100)
    devices_mb, params_mb = init_pipeline_parallel()
    comm_mb = CommTracker()
    acc_mb, time_mb = train_pipeline_microbatch(
        devices_mb, params_mb, xs, ys, comm_mb, N_MICRO_BATCHES)

    # === PIPELINE BUBBLE VISUALIZATION ===
    print(f"\n{'=' * 60}\n  PIPELINE BUBBLE ANALYSIS\n{'=' * 60}")
    # Varying micro-batch counts show how micro-batching reduces the bubble
    print_pipeline_schedule(2, 1)   # 50% bubble
    print_pipeline_schedule(2, 2)   # 33% bubble
    print_pipeline_schedule(2, 4)   # 20% bubble
    # Signpost: PipeDream overlaps forward of micro-batch k+1 with backward of
    # micro-batch k, further reducing the bubble. We show the simpler GPipe
    # schedule where all forwards complete before any backward starts.

    # === COMPARISON TABLE ===
    bubble_pp = compute_bubble_fraction(2, 1)
    bubble_mb = compute_bubble_fraction(2, N_MICRO_BATCHES)
    print(f"\n{'=' * 60}\n  COMPARISON SUMMARY\n{'=' * 60}")
    print(f"\n{'Strategy':<35} {'Dev':>3} {'Comm':>7} {'Floats':>8} "
          f"{'Bubble':>6} {'Acc':>6} {'Time':>6}")
    print("-" * 73)
    rows = [
        ("Single device",       1, 0,              0,                       0.0,      acc_single, time_single),
        ("Tensor parallel",     2, comm_tp.rounds,  comm_tp.floats_transferred, 0.0,  acc_tp,     time_tp),
        ("Pipeline parallel",   2, comm_pp.rounds,  comm_pp.floats_transferred, bubble_pp, acc_pp, time_pp),
        (f"Pipeline + {N_MICRO_BATCHES} micro-batches", 2, comm_mb.rounds,
         comm_mb.floats_transferred, bubble_mb, acc_mb, time_mb),
    ]
    for name, dev, comm_r, floats, bub, acc, t in rows:
        print(f"{name:<35} {dev:>3} {comm_r:>7} {floats:>8} "
              f"{bub:>5.0%} {acc:>5.1%} {t:>5.1f}s")

    # === KEY TAKEAWAYS ===
    print(f"""
{'=' * 60}
  KEY TAKEAWAYS
{'=' * 60}

1. TENSOR PARALLELISM splits each layer across devices.
   - All-reduce at every layer: {comm_tp.floats_transferred} floats transferred.
   - Communication scales with layers x steps x batch_size.
   - Accuracy matches baseline (same math, different execution order).
   - Best for: layers too wide for one device's memory.

2. PIPELINE PARALLELISM splits layers across devices.
   - Communication only at stage boundaries: {comm_pp.floats_transferred} floats.
   - Pipeline bubble wastes {bubble_pp:.0%} compute (1 micro-batch).
   - Micro-batching reduces bubble to {bubble_mb:.0%} ({N_MICRO_BATCHES} micro-batches).
   - Best for: models too deep for one device's memory.

3. COMMUNICATION IS THE BOTTLENECK.
   - Tensor parallel needs ~{comm_tp.floats_transferred / max(comm_pp.floats_transferred, 1):.0f}x more data transfer than pipeline.
   - At scale, interconnect bandwidth (NVLink: 900 GB/s, InfiniBand: 400 Gb/s)
     determines how many devices you can use before communication dominates.

Signpost: modern large-model training uses 3D parallelism — data parallelism
(replicate model across groups), tensor parallelism (split within a node's
GPUs), pipeline parallelism (split across nodes). Designing the optimal
partition is itself an active research problem (Alpa, FlexFlow).
""")


if __name__ == "__main__":
    main()
