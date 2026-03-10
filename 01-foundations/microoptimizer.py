"""
Why Adam converges when SGD stalls — momentum, adaptive learning rates, and the geometry of
loss landscapes, demonstrated head-to-head.
"""
# Reference: Adam optimizer from Kingma & Ba (2015). This script trains identical character
# bigram models with SGD, Momentum, RMSProp, and Adam to show why adaptive methods dominate.
# Extension: learning rate warmup + cosine decay (Loshchilov & Hutter, 2016).

# === TRADEOFFS ===
# + Adam converges faster than SGD on most tasks via adaptive per-parameter learning rates
# + Momentum-based methods escape shallow local minima that trap vanilla SGD
# + Learning rate schedules (warmup + decay) improve final convergence quality
# - Adam uses 3x memory of SGD (stores m and v per parameter)
# - Adaptive methods can generalize worse than well-tuned SGD on some tasks
# - More hyperparameters to tune (beta1, beta2, epsilon, schedule shape)
# WHEN TO USE: Default to Adam/AdamW for most deep learning tasks, especially
#   transformers. Switch to SGD+momentum only if Adam overfits or memory is tight.
# WHEN NOT TO: Extremely memory-constrained training, or convex optimization
#   problems where SGD with a proper schedule converges optimally.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — deliberately simple so optimizer differences are visible
N_EMBD = 8           # embedding dimension (small to keep scalar autograd tractable)
VOCAB_SIZE = 0       # set after loading data (number of unique characters + BOS)

# Training parameters — shared across all optimizer runs
NUM_STEPS = 300      # training iterations per optimizer
BATCH_SIZE = 4       # names sampled per step — small because scalar autograd builds a
                     # full computation graph per Value operation; batch_size=4 with
                     # ~5 bigrams per name ≈ 20 forward passes per step, each creating
                     # thousands of Value nodes. Larger batches are cost-prohibitive.

# Optimizer-specific hyperparameters
SGD_LR = 0.05              # SGD needs a larger learning rate to make progress
MOMENTUM_LR = 0.05         # momentum accumulates velocity, so smaller lr is stable
MOMENTUM_BETA = 0.9        # exponential decay rate for velocity (standard choice)
RMSPROP_LR = 0.01          # adaptive lr means we can start smaller
RMSPROP_BETA = 0.99        # decay rate for squared gradient average
RMSPROP_EPS = 1e-8         # prevents division by zero in denominator
ADAM_LR = 0.01             # Adam's bias correction allows moderate lr
ADAM_BETA1 = 0.9           # first moment decay (momentum component)
ADAM_BETA2 = 0.999         # second moment decay (RMSProp component)
ADAM_EPS = 1e-8            # numerical stability in denominator

# Signpost: these hyperparameters are tuned for this specific problem scale. In production,
# Adam(lr=3e-4, beta1=0.9, beta2=0.999) is the standard starting point for most deep learning.

# Extension parameters — warmup + cosine decay applied to Adam
WARMUP_STEPS = 20          # linearly ramp lr from 0 to peak over this many steps
COSINE_LR = 0.01           # peak learning rate after warmup

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    return docs


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (∂out/∂input) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule.
        """
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

        # Seed: gradient of loss with respect to itself is 1
        self.grad = 1.0

        # Reverse topological order: gradients flow backward from output to inputs
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: ∂Loss/∂child += ∂Loss/∂v * ∂v/∂child
                child.grad += local_grad * v.grad


# === BIGRAM MODEL ===
# A character bigram predicts the next character given only the current character.
# Architecture: embed(char) → linear → softmax → next char distribution.
# This is deliberately simple — the optimizer comparison is the focus, not model
# sophistication. All four optimizers train this exact same architecture.

def make_params(vocab_size: int, embd_dim: int) -> list[list[Value]]:
    """Initialize model parameters: embedding matrix [vocab_size, embd_dim] and
    output projection [vocab_size, embd_dim].

    Returns a flat structure: [embedding_matrix, projection_matrix] where each
    is a list of rows, each row a list of Value scalars.

    Weight initialization uses Gaussian noise scaled by 1/sqrt(embd_dim) — the
    Xavier/Glorot heuristic that keeps activation variance roughly constant
    across layers. Critical for gradient flow in deeper models, helpful here
    for consistent starting conditions across optimizer runs.
    """
    std = 1.0 / math.sqrt(embd_dim)
    embedding = [[Value(random.gauss(0, std)) for _ in range(embd_dim)]
                 for _ in range(vocab_size)]
    projection = [[Value(random.gauss(0, std)) for _ in range(embd_dim)]
                  for _ in range(vocab_size)]
    return [embedding, projection]


def clone_params(params: list[list[list[Value]]]) -> list[list[list[Value]]]:
    """Deep copy model parameters so each optimizer starts from identical weights.

    Without cloning, all optimizers would share the same Value objects, meaning
    one optimizer's gradient updates would corrupt another's starting point.
    """
    return [
        [[Value(v.data) for v in row] for row in matrix]
        for matrix in params
    ]


def flatten_params(params: list[list[list[Value]]]) -> list[Value]:
    """Flatten nested parameter structure into a single list for optimizer bookkeeping."""
    return [v for matrix in params for row in matrix for v in row]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Subtract max(logits) before exp() to prevent overflow. Without this,
    large logits cause exp() to return inf, breaking the computation.
    Math: softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clamped logarithm to prevent log(0) = -inf.

    Keeps prob as a child node so gradients flow back through the computation graph.
    d(log(x))/dx = 1/x, evaluated at the clamped value for stability.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def bigram_loss(params: list[list[list[Value]]], data: list[list[int]]) -> Value:
    """Compute cross-entropy loss over a mini-batch of tokenized names.

    For each consecutive character pair (context → target), the model:
    1. Looks up the context character's embedding vector
    2. Projects it to vocabulary-sized logits via dot product with projection matrix
    3. Applies softmax to get a probability distribution
    4. Computes -log(p(target)) as the loss for this pair

    The total loss is averaged over all bigrams in the batch, making it
    scale-invariant to batch size and sequence length.
    """
    embedding, projection = params
    total_loss = Value(0.0)
    count = 0

    for token_seq in data:
        for i in range(len(token_seq) - 1):
            context = token_seq[i]
            target = token_seq[i + 1]

            # Forward pass: embed → project → softmax → loss
            emb = embedding[context]
            # Logits: each output neuron computes dot(projection_row, embedding)
            logits = [sum(projection[j][k] * emb[k] for k in range(len(emb)))
                      for j in range(len(projection))]
            probs = softmax(logits)

            # Cross-entropy loss for this bigram: -log(p(target))
            total_loss = total_loss + (-safe_log(probs[target]))
            count += 1

    # Average over all bigrams — ensures loss is comparable across different batch sizes
    return total_loss / count


# === OPTIMIZER IMPLEMENTATIONS ===
# Each optimizer takes the same interface: a list of Value parameters and their gradients.
# The key insight is that they all compute parameter updates using the same gradient
# information, but accumulate and scale it differently.

def step_sgd(
    param_list: list[Value],
    learning_rate: float,
    state: dict,
) -> None:
    """Vanilla stochastic gradient descent.

    Update rule: θ = θ - lr * ∇L

    The simplest possible optimizer: move each parameter in the direction opposite
    to its gradient, scaled by the learning rate. No memory of past gradients.

    Problem: all parameters get the same learning rate regardless of gradient
    history. Parameters with consistently large gradients may overshoot, while
    those with small gradients barely move. Also, SGD gets stuck in saddle points
    because it has no momentum to carry it through flat regions.
    """
    for param in param_list:
        param.data -= learning_rate * param.grad
        param.grad = 0.0


def step_momentum(
    param_list: list[Value],
    learning_rate: float,
    state: dict,
) -> None:
    """SGD with momentum — adds a velocity term that accumulates past gradients.

    Update rule:
        v = β*v + ∇L           (accumulate velocity)
        θ = θ - lr * v          (update parameters using velocity)

    Momentum acts like a ball rolling downhill: it accelerates through consistent
    gradient directions and dampens oscillation in directions where gradients
    alternate sign. β controls how much past gradients influence the current step.
    At β=0.9, the effective window is ~10 past gradients (1/(1-β)).

    This helps escape saddle points and shallow local minima where vanilla SGD stalls.
    """
    if 'velocity' not in state:
        state['velocity'] = [0.0] * len(param_list)

    velocity = state['velocity']
    for i, param in enumerate(param_list):
        # v = β*v + ∇L
        velocity[i] = MOMENTUM_BETA * velocity[i] + param.grad
        # θ = θ - lr * v
        param.data -= learning_rate * velocity[i]
        param.grad = 0.0


def step_rmsprop(
    param_list: list[Value],
    learning_rate: float,
    state: dict,
) -> None:
    """RMSProp — adapts the learning rate per-parameter using squared gradient history.

    Update rule:
        s = β*s + (1-β)*∇L²     (running average of squared gradients)
        θ = θ - lr * ∇L/√(s+ε)  (scale update by inverse RMS of gradient history)

    The key insight: dividing by √s normalizes each parameter's update by the
    typical magnitude of its gradient. Parameters with historically large gradients
    get smaller effective learning rates (preventing overshooting), while parameters
    with small gradients get larger effective rates (accelerating learning).

    Signpost: RMSProp was proposed by Hinton in an unpublished lecture. It fixes
    AdaGrad's problem of monotonically decreasing learning rates by using an
    exponential moving average instead of a cumulative sum.
    """
    if 'sq_avg' not in state:
        state['sq_avg'] = [0.0] * len(param_list)

    sq_avg = state['sq_avg']
    for i, param in enumerate(param_list):
        # s = β*s + (1-β)*∇L²
        sq_avg[i] = RMSPROP_BETA * sq_avg[i] + (1 - RMSPROP_BETA) * param.grad ** 2
        # θ = θ - lr * ∇L / √(s + ε)
        param.data -= learning_rate * param.grad / (math.sqrt(sq_avg[i]) + RMSPROP_EPS)
        param.grad = 0.0


def step_adam(
    param_list: list[Value],
    learning_rate: float,
    state: dict,
) -> None:
    """Adam — combines momentum (first moment) with RMSProp (second moment) plus
    bias correction.

    Update rule:
        m = β1*m + (1-β1)*∇L        (first moment: momentum/mean of gradients)
        v = β2*v + (1-β2)*∇L²       (second moment: uncentered variance of gradients)
        m̂ = m / (1-β1^t)            (bias correction for first moment)
        v̂ = v / (1-β2^t)            (bias correction for second moment)
        θ = θ - lr * m̂ / √(v̂ + ε)  (parameter update)

    Why bias correction matters: m and v are initialized to 0. In early steps,
    they're biased toward zero because the exponential moving average hasn't
    had time to "warm up". Without correction, early updates would be much
    too small (m ≈ 0) or misgauged (v ≈ 0). The correction factor 1/(1-β^t)
    compensates — it's large when t is small and approaches 1 as t grows.

    Adam's dominance in practice comes from combining the best of both worlds:
    momentum provides noise-averaged gradient direction, while adaptive scaling
    provides per-parameter step sizes.
    """
    if 'step_count' not in state:
        state['step_count'] = 0
        state['m'] = [0.0] * len(param_list)
        state['v'] = [0.0] * len(param_list)

    state['step_count'] += 1
    t = state['step_count']
    m_state = state['m']
    v_state = state['v']

    for i, param in enumerate(param_list):
        # m = β1*m + (1-β1)*∇L
        m_state[i] = ADAM_BETA1 * m_state[i] + (1 - ADAM_BETA1) * param.grad
        # v = β2*v + (1-β2)*∇L²
        v_state[i] = ADAM_BETA2 * v_state[i] + (1 - ADAM_BETA2) * param.grad ** 2

        # Bias correction: divide by (1 - β^t)
        m_hat = m_state[i] / (1 - ADAM_BETA1 ** t)
        v_hat = v_state[i] / (1 - ADAM_BETA2 ** t)

        # θ = θ - lr * m̂ / √(v̂ + ε)
        param.data -= learning_rate * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
        param.grad = 0.0


# === TRAINING LOOP ===
# Train the same bigram model architecture with each optimizer independently.
# Each run starts from identical initial weights (via clone_params) so differences
# in convergence are purely due to the optimizer, not initialization luck.

def train_optimizer(
    optimizer_name: str,
    step_fn,
    learning_rate: float,
    params: list[list[list[Value]]],
    batches: list[list[list[int]]],
    num_steps: int,
    lr_schedule_fn=None,
) -> tuple[list[float], float]:
    """Train a bigram model using a specific optimizer and return loss history.

    Args:
        optimizer_name: display name for logging
        step_fn: optimizer step function (step_sgd, step_momentum, etc.)
        learning_rate: base learning rate
        params: model parameters [embedding, projection]
        batches: pre-generated mini-batches of tokenized names
        num_steps: number of training iterations
        lr_schedule_fn: optional function(step, num_steps) -> lr multiplier

    Returns:
        (loss_history, elapsed_seconds)
    """
    param_list = flatten_params(params)
    state: dict = {}
    loss_history: list[float] = []

    start_time = time.time()

    for step in range(num_steps):
        batch = batches[step % len(batches)]

        # Compute loss and gradients
        loss = bigram_loss(params, batch)
        loss.backward()

        # Determine effective learning rate (with optional schedule)
        effective_lr = learning_rate
        if lr_schedule_fn is not None:
            effective_lr = lr_schedule_fn(step, num_steps)

        # Apply optimizer step
        step_fn(param_list, effective_lr, state)

        loss_history.append(loss.data)

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  [{optimizer_name:>20s}] step {step + 1:>4}/{num_steps} | "
                  f"loss: {loss.data:.4f} | lr: {effective_lr:.6f}")

    elapsed = time.time() - start_time
    return loss_history, elapsed


# === EXTENSION: LEARNING RATE SCHEDULING ===
# Warmup + cosine decay is the standard schedule in modern deep learning (GPT, LLaMA,
# BERT all use variants of this). The intuition:
#
# Warmup phase (first N steps): linearly ramp lr from 0 to peak. Early in training,
# the loss landscape is chaotic and gradients are unreliable. Large learning rates
# cause divergence. Warmup lets the optimizer "feel out" the landscape before
# committing to large steps.
#
# Cosine decay phase (remaining steps): smoothly anneal lr following a cosine curve
# from peak to 0. As the model converges, the loss landscape becomes more curved
# (higher curvature near minima). Smaller learning rates prevent overshooting the
# narrow valley of the minimum.
#
# Math:
#   if step < warmup_steps:
#       lr = peak_lr * step / warmup_steps                (linear warmup)
#   else:
#       progress = (step - warmup_steps) / (total - warmup_steps)
#       lr = peak_lr * 0.5 * (1 + cos(π * progress))     (cosine decay)

def cosine_schedule(step: int, num_steps: int) -> float:
    """Compute learning rate with linear warmup followed by cosine decay.

    Returns the actual learning rate (not a multiplier), matching Adam's expected lr range.
    """
    if step < WARMUP_STEPS:
        # Linear warmup: lr grows from 0 to COSINE_LR over WARMUP_STEPS
        return COSINE_LR * (step + 1) / WARMUP_STEPS
    else:
        # Cosine decay: lr decreases from COSINE_LR to 0 following cos curve
        progress = (step - WARMUP_STEPS) / max(1, num_steps - WARMUP_STEPS)
        return COSINE_LR * 0.5 * (1 + math.cos(math.pi * progress))


# === COMPARISON AND RESULTS ===


def run_optimizer_comparison(
    sgd_lr: float, momentum_lr: float, rmsprop_lr: float,
    adam_lr: float, cosine_lr: float, num_steps: int,
) -> None:
    """Run all optimizers with the given learning rates and step count."""
    global VOCAB_SIZE, NUM_STEPS, COSINE_LR

    NUM_STEPS = num_steps
    COSINE_LR = cosine_lr

    # -- Load and prepare data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    # Build vocabulary from unique characters
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} names")
    print(f"Vocabulary: {VOCAB_SIZE} tokens ({len(unique_chars)} chars + BOS)\n")

    # Tokenize all names: [BOS, char_0, char_1, ..., char_n, BOS]
    tokenized = [[BOS] + [unique_chars.index(ch) for ch in name] + [BOS] for name in docs]

    # Pre-generate mini-batches so all optimizers see the same data in the same order.
    # This eliminates data ordering as a confound — any performance difference is
    # purely due to the optimizer.
    num_batches = (num_steps // 1) + 1
    batches: list[list[list[int]]] = []
    for b in range(num_batches):
        start = (b * BATCH_SIZE) % len(tokenized)
        batch = [tokenized[(start + j) % len(tokenized)] for j in range(BATCH_SIZE)]
        batches.append(batch)

    # Initialize base model parameters (shared starting point via cloning)
    random.seed(42)  # reset seed so weight initialization is identical
    base_params = make_params(VOCAB_SIZE, N_EMBD)

    param_count = sum(len(row) for matrix in base_params for row in matrix)
    print(f"Model parameters: {param_count:,}")
    print(f"Training: {num_steps} steps, batch size {BATCH_SIZE}\n")

    # -- Train with each optimizer --
    optimizers = [
        ("SGD",             step_sgd,      sgd_lr,      None),
        ("SGD + Momentum",  step_momentum, momentum_lr, None),
        ("RMSProp",         step_rmsprop,  rmsprop_lr,  None),
        ("Adam",            step_adam,      adam_lr,     None),
        ("Adam + Schedule", step_adam,      cosine_lr,   cosine_schedule),
    ]

    results: list[tuple[str, list[float], float]] = []

    for name, step_fn, lr, schedule_fn in optimizers:
        print(f"--- {name} (lr={lr}) ---")

        # Clone parameters so this optimizer starts from identical weights
        random.seed(42)  # ensure any randomness in training is controlled
        params_copy = clone_params(base_params)

        loss_history, elapsed = train_optimizer(
            name, step_fn, lr, params_copy, batches, num_steps,
            lr_schedule_fn=schedule_fn,
        )
        results.append((name, loss_history, elapsed))
        print(f"  Final loss: {loss_history[-1]:.4f} | Time: {elapsed:.1f}s\n")

    # -- Comparison table --
    # Find the step where each optimizer first drops below a loss threshold.
    # This measures convergence speed: fewer steps = faster convergence.
    loss_threshold = 3.0

    print("=" * 76)
    print(f"{'Optimizer':<20s} {'Final Loss':>12s} {'Steps to <' + str(loss_threshold):>16s} "
          f"{'Time (s)':>10s} {'Best Loss':>12s}")
    print("-" * 76)

    for name, loss_history, elapsed in results:
        final_loss = loss_history[-1]
        best_loss = min(loss_history)

        # Find first step below threshold
        steps_to_threshold = "never"
        for step_idx, loss_val in enumerate(loss_history):
            if loss_val < loss_threshold:
                steps_to_threshold = str(step_idx + 1)
                break

        print(f"{name:<20s} {final_loss:>12.4f} {steps_to_threshold:>16s} "
              f"{elapsed:>10.1f} {best_loss:>12.4f}")

    print("=" * 76)

    # -- Key takeaways --
    print("\nKey observations:")
    print("  - SGD converges slowly and may stall at a higher loss plateau")
    print("  - Momentum accelerates convergence by accumulating gradient direction")
    print("  - RMSProp adapts per-parameter rates, handling uneven gradient scales")
    print("  - Adam combines both benefits with bias correction for stable early training")
    print("  - LR scheduling (warmup + cosine decay) further improves Adam's convergence")

    # -- Inference: generate names with the best model (Adam + Schedule) --
    # Use the last trained model (Adam + Schedule) for generation
    print("\n--- Generating names with Adam + Schedule model ---\n")

    # Retrain Adam + Schedule to get its final parameters (since we need the actual params)
    random.seed(42)
    best_params = clone_params(base_params)
    best_param_list = flatten_params(best_params)
    adam_state: dict = {}

    for step in range(num_steps):
        batch = batches[step % len(batches)]
        loss = bigram_loss(best_params, batch)
        loss.backward()
        lr_val = cosine_schedule(step, num_steps)
        step_adam(best_param_list, lr_val, adam_state)

    # Generate 10 names via autoregressive sampling
    embedding, projection = best_params
    temperature = 0.8

    for sample_idx in range(10):
        token_id = BOS
        generated: list[str] = []

        for _ in range(20):  # max name length
            # Forward pass: embed → project → softmax
            emb = embedding[token_id]
            logits_data = [
                sum(projection[j][k].data * emb[k].data for k in range(N_EMBD))
                for j in range(VOCAB_SIZE)
            ]

            # Temperature-scaled softmax (forward-only, no autograd needed for inference)
            max_logit = max(logits_data)
            exp_vals = [math.exp((v - max_logit) / temperature) for v in logits_data]
            total = sum(exp_vals)
            probs = [e / total for e in exp_vals]

            # Sample next token
            token_id = random.choices(range(VOCAB_SIZE), weights=probs)[0]

            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")


# === INTERACTIVE MODE ===
# Optional functionality: allows parameter exploration without editing the script.
# Activated only via --interactive flag; default behavior is unchanged.

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizer comparison: SGD, Momentum, RMSProp, Adam head-to-head"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive mode to modify learning rates and re-run comparison"
    )
    return parser.parse_args()


def interactive_loop() -> None:
    """Interactive parameter exploration mode."""
    print("\n=== INTERACTIVE MODE ===")
    print("Modify optimizer learning rates and training steps, then re-run.")
    print("Type 'quit' to exit.\n")

    params = {
        'sgd_lr': SGD_LR,
        'momentum_lr': MOMENTUM_LR,
        'rmsprop_lr': RMSPROP_LR,
        'adam_lr': ADAM_LR,
        'cosine_lr': COSINE_LR,
        'num_steps': NUM_STEPS,
    }

    while True:
        print("Current parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")

        user_input = input(
            "\nParameter to change (or 'run' to execute, 'quit' to exit): "
        ).strip().lower()

        if user_input == 'quit':
            break
        elif user_input == 'run':
            run_optimizer_comparison(
                params['sgd_lr'], params['momentum_lr'], params['rmsprop_lr'],
                params['adam_lr'], params['cosine_lr'], params['num_steps']
            )
        elif '=' in user_input:
            key, _, val = user_input.partition('=')
            key = key.strip()
            val = val.strip()
            if key not in params:
                print(f"Unknown parameter: {key}")
                print(f"Available: {', '.join(params)}")
                continue
            try:
                if key == 'num_steps':
                    params[key] = int(val)
                else:
                    params[key] = float(val)
            except ValueError:
                print(f"Invalid value: {val}")
        else:
            print("Enter 'parameter=value', 'run', or 'quit'.")


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        interactive_loop()
    else:
        # === DEFAULT BEHAVIOR (unchanged) ===
        run_optimizer_comparison(
            SGD_LR, MOMENTUM_LR, RMSPROP_LR, ADAM_LR, COSINE_LR, NUM_STEPS
        )
