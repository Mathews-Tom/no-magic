"""
Adam converges faster than SGD because it maintains per-parameter learning rates that adapt
to gradient history -- but SGD with the right schedule can match or beat Adam's final loss.
"""
# Reference: Adam from Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015).
# SGD convergence theory from Robbins & Monro (1951). This script trains the same character
# bigram model with both optimizers on identical data to isolate the optimizer's effect
# on convergence speed, final loss, and generalization.

# === TRADEOFFS ===
# + Adam converges faster via adaptive per-parameter learning rates (momentum + RMS scaling)
# + Adam is less sensitive to learning rate choice -- works well across a wide range
# + SGD with momentum can generalize better: the "noise" in fixed-rate updates acts as
#   implicit regularization (Smith & Le, 2018)
# - Adam uses 3x memory of SGD (stores first and second moment estimates per parameter)
# - Adam can converge to sharper minima that generalize worse on some tasks
# - SGD requires careful learning rate tuning and scheduling to compete
# WHEN TO USE: Adam for prototyping and most tasks. SGD+momentum for final training
#   of large models where generalization matters (vision models, LLMs at scale).
# WHEN NOT TO: Don't use vanilla SGD without momentum -- it's strictly dominated.
#   Don't use Adam when memory is extremely tight (3x overhead matters at scale).

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_EMBD = 8           # embedding dimension -- small for scalar autograd tractability
NUM_STEPS = 400      # training iterations per optimizer
BATCH_SIZE = 4       # names per mini-batch

# SGD hyperparameters
SGD_LR = 0.05        # larger lr needed because SGD has no adaptive scaling
SGD_MOMENTUM = 0.9   # exponential decay for velocity accumulation

# Adam hyperparameters
ADAM_LR = 0.01       # smaller lr works because Adam scales per-parameter
ADAM_BETA1 = 0.9     # first moment decay (momentum component)
ADAM_BETA2 = 0.999   # second moment decay (RMS component)
ADAM_EPS = 1e-8      # numerical stability in denominator

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: these hyperparameters are tuned for this specific problem scale.
# In production, Adam(lr=3e-4) and SGD(lr=0.1, momentum=0.9) are standard starting points.


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

    Tracks computational history for gradient computation via the chain rule.
    Each forward operation stores its local derivative; backward() replays
    the graph in reverse topological order.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
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
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        """Reverse-mode automatic differentiation via topological sort."""
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
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# === BIGRAM MODEL ===
# A character bigram predicts the next character from the current one.
# Architecture: embed(char) -> dot product with projection matrix -> softmax.
# Deliberately simple so optimizer differences dominate model capacity effects.

def make_params(vocab_size: int, embd_dim: int) -> list[list[list[Value]]]:
    """Initialize embedding and projection matrices.

    Xavier scaling (1/sqrt(fan_in)) keeps activation variance stable across layers.
    Both optimizers start from identical weights via clone_params.
    """
    std = 1.0 / math.sqrt(embd_dim)
    embedding = [[Value(random.gauss(0, std)) for _ in range(embd_dim)]
                 for _ in range(vocab_size)]
    projection = [[Value(random.gauss(0, std)) for _ in range(embd_dim)]
                  for _ in range(vocab_size)]
    return [embedding, projection]


def clone_params(params: list[list[list[Value]]]) -> list[list[list[Value]]]:
    """Deep copy so each optimizer starts from identical weights."""
    return [
        [[Value(v.data) for v in row] for row in matrix]
        for matrix in params
    ]


def flatten_params(params: list[list[list[Value]]]) -> list[Value]:
    """Flatten nested parameters into a single list for optimizer bookkeeping."""
    return [v for matrix in params for row in matrix for v in row]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clamped logarithm to prevent log(0) = -inf."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def bigram_loss(params: list[list[list[Value]]], data: list[list[int]]) -> Value:
    """Cross-entropy loss over a mini-batch of tokenized names.

    For each consecutive (context, target) pair: embed context, project to logits,
    softmax, compute -log(p(target)). Averaged over all bigrams.
    """
    embedding, projection = params
    total_loss = Value(0.0)
    count = 0

    for token_seq in data:
        for i in range(len(token_seq) - 1):
            context = token_seq[i]
            target = token_seq[i + 1]

            emb = embedding[context]
            logits = [sum(projection[j][k] * emb[k] for k in range(len(emb)))
                      for j in range(len(projection))]
            probs = softmax(logits)
            total_loss = total_loss + (-safe_log(probs[target]))
            count += 1

    return total_loss / count


# === OPTIMIZER IMPLEMENTATIONS ===

def step_sgd_momentum(
    param_list: list[Value], learning_rate: float, state: dict
) -> None:
    """SGD with momentum: v = beta*v + grad; theta -= lr * v

    Momentum accumulates gradient direction over time. The velocity v acts like
    a ball rolling downhill: consistent gradient directions accelerate, while
    oscillating directions cancel out. This smooths the optimization trajectory.

    The effective step size grows when gradients are consistent (up to lr/(1-beta)
    in the steady state), helping escape shallow local minima and saddle points.
    """
    if 'velocity' not in state:
        state['velocity'] = [0.0] * len(param_list)

    velocity = state['velocity']
    for i, param in enumerate(param_list):
        velocity[i] = SGD_MOMENTUM * velocity[i] + param.grad
        param.data -= learning_rate * velocity[i]
        param.grad = 0.0


def step_adam(
    param_list: list[Value], learning_rate: float, state: dict
) -> None:
    """Adam: combines momentum (first moment) with adaptive scaling (second moment).

    Update rule:
        m = beta1*m + (1-beta1)*grad        (exponential moving average of gradients)
        v = beta2*v + (1-beta2)*grad^2       (exponential moving average of squared gradients)
        m_hat = m / (1-beta1^t)              (bias correction -- critical for early steps)
        v_hat = v / (1-beta2^t)              (bias correction)
        theta -= lr * m_hat / sqrt(v_hat + eps)

    Why this works: m_hat provides noise-averaged gradient direction (like momentum),
    while sqrt(v_hat) provides per-parameter scaling. Parameters with large historical
    gradients get smaller effective learning rates (preventing overshooting), and
    parameters with small gradients get larger rates (accelerating learning).

    The bias correction factor 1/(1-beta^t) is essential: without it, the first
    few updates use near-zero moment estimates, making early training sluggish.
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
        m_state[i] = ADAM_BETA1 * m_state[i] + (1 - ADAM_BETA1) * param.grad
        v_state[i] = ADAM_BETA2 * v_state[i] + (1 - ADAM_BETA2) * param.grad ** 2

        m_hat = m_state[i] / (1 - ADAM_BETA1 ** t)
        v_hat = v_state[i] / (1 - ADAM_BETA2 ** t)

        param.data -= learning_rate * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
        param.grad = 0.0


# === TRAINING LOOP ===

def train_optimizer(
    optimizer_name: str, step_fn, learning_rate: float,
    params: list[list[list[Value]]], batches: list[list[list[int]]],
    num_steps: int
) -> tuple[list[float], float, list[float]]:
    """Train a bigram model with a specific optimizer, tracking loss and gradient norms.

    Returns:
        (loss_history, elapsed_seconds, gradient_norm_history)

    Gradient norms reveal optimizer dynamics: Adam should show stable, moderate norms
    because adaptive scaling prevents both vanishing and exploding updates. SGD norms
    fluctuate more because all parameters share the same learning rate.
    """
    param_list = flatten_params(params)
    state: dict = {}
    loss_history: list[float] = []
    grad_norm_history: list[float] = []

    start_time = time.time()

    for step in range(num_steps):
        batch = batches[step % len(batches)]
        loss = bigram_loss(params, batch)
        loss.backward()

        # Compute gradient norm before the optimizer step modifies gradients
        # ||grad||_2 = sqrt(sum_i grad_i^2)
        grad_norm = math.sqrt(sum(p.grad ** 2 for p in param_list))
        grad_norm_history.append(grad_norm)

        step_fn(param_list, learning_rate, state)
        loss_history.append(loss.data)

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  [{optimizer_name:>16s}] step {step + 1:>4}/{num_steps} | "
                  f"loss: {loss.data:.4f} | grad_norm: {grad_norm:.4f}")

    elapsed = time.time() - start_time
    return loss_history, elapsed, grad_norm_history


# === EFFECTIVE LEARNING RATE ANALYSIS ===

def compute_effective_lr(state: dict, param_list: list[Value]) -> list[float]:
    """Compute Adam's effective learning rate for each parameter.

    Adam's actual step size for parameter i at step t is:
        effective_lr_i = lr * |m_hat_i| / sqrt(v_hat_i + eps)

    This shows how Adam adapts: parameters with large gradient variance get
    smaller effective rates, while rarely-updated parameters get larger rates.
    SGD's effective rate is constant (lr) for all parameters.
    """
    if 'step_count' not in state or state['step_count'] == 0:
        return [ADAM_LR] * len(param_list)

    t = state['step_count']
    effective_lrs = []

    for i in range(len(param_list)):
        m_hat = state['m'][i] / (1 - ADAM_BETA1 ** t)
        v_hat = state['v'][i] / (1 - ADAM_BETA2 ** t)

        # Effective lr = lr * |m_hat| / (sqrt(v_hat) + eps)
        # This represents the actual step size for this parameter
        if abs(m_hat) > 1e-15:
            eff = ADAM_LR * abs(m_hat) / (math.sqrt(v_hat) + ADAM_EPS)
        else:
            eff = 0.0
        effective_lrs.append(eff)

    return effective_lrs


# === GENERALIZATION TEST ===

def eval_loss(params: list[list[list[Value]]], data: list[list[int]]) -> float:
    """Compute loss on held-out data without gradient tracking.

    Forward-only pass using raw floats for speed. This measures generalization:
    how well the model predicts unseen data. If Adam overfits (memorizes training
    data) while SGD generalizes better, the validation loss will show it.
    """
    embedding, projection = params
    total_loss = 0.0
    count = 0

    for token_seq in data:
        for i in range(len(token_seq) - 1):
            context = token_seq[i]
            target = token_seq[i + 1]

            # Forward pass with raw floats (no autograd overhead)
            emb = [v.data for v in embedding[context]]
            logits = [sum(projection[j][k].data * emb[k] for k in range(len(emb)))
                      for j in range(len(projection))]

            # Stable softmax
            max_val = max(logits)
            exp_vals = [math.exp(v - max_val) for v in logits]
            total = sum(exp_vals)
            probs = [e / total for e in exp_vals]

            total_loss += -math.log(max(probs[target], 1e-10))
            count += 1

    return total_loss / count if count > 0 else float('inf')


# === MAIN ===

if __name__ == "__main__":
    print("=" * 70)
    print("ADAM VS SGD: Optimizer Convergence Comparison")
    print("=" * 70)
    print()

    # -- Load and prepare data --
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} names, vocab size: {VOCAB_SIZE}")

    # Split into train and validation for generalization test
    split = int(len(docs) * 0.9)
    train_docs = docs[:split]
    val_docs = docs[split:]

    # Tokenize
    tokenized_train = [[BOS] + [unique_chars.index(ch) for ch in name] + [BOS]
                       for name in train_docs]
    tokenized_val = [[BOS] + [unique_chars.index(ch) for ch in name] + [BOS]
                     for name in val_docs]

    # Pre-generate mini-batches: identical data ordering for both optimizers
    # eliminates data order as a confound
    num_batches = NUM_STEPS + 1
    batches: list[list[list[int]]] = []
    for b in range(num_batches):
        start = (b * BATCH_SIZE) % len(tokenized_train)
        batch = [tokenized_train[(start + j) % len(tokenized_train)] for j in range(BATCH_SIZE)]
        batches.append(batch)

    # Initialize base model parameters
    random.seed(42)
    base_params = make_params(VOCAB_SIZE, N_EMBD)
    param_count = sum(len(row) for matrix in base_params for row in matrix)
    print(f"Model parameters: {param_count:,}")
    print(f"Training: {NUM_STEPS} steps, batch size {BATCH_SIZE}")
    print(f"Train/val split: {len(train_docs)}/{len(val_docs)}\n")

    # -- Train with SGD+Momentum --
    print(f"--- SGD + Momentum (lr={SGD_LR}, momentum={SGD_MOMENTUM}) ---")
    random.seed(42)
    sgd_params = clone_params(base_params)
    sgd_losses, sgd_time, sgd_grad_norms = train_optimizer(
        "SGD+Momentum", step_sgd_momentum, SGD_LR,
        sgd_params, batches, NUM_STEPS
    )
    sgd_val_loss = eval_loss(sgd_params, tokenized_val[:200])
    print(f"  Final train loss: {sgd_losses[-1]:.4f} | Val loss: {sgd_val_loss:.4f} | "
          f"Time: {sgd_time:.1f}s\n")

    # -- Train with Adam --
    print(f"--- Adam (lr={ADAM_LR}, beta1={ADAM_BETA1}, beta2={ADAM_BETA2}) ---")
    random.seed(42)
    adam_params = clone_params(base_params)
    adam_losses, adam_time, adam_grad_norms = train_optimizer(
        "Adam", step_adam, ADAM_LR,
        adam_params, batches, NUM_STEPS
    )
    adam_val_loss = eval_loss(adam_params, tokenized_val[:200])
    print(f"  Final train loss: {adam_losses[-1]:.4f} | Val loss: {adam_val_loss:.4f} | "
          f"Time: {adam_time:.1f}s\n")

    # === CONVERGENCE COMPARISON ===
    print("=" * 70)
    print("CONVERGENCE CURVES (every 50 steps)")
    print("=" * 70)
    print(f"{'Step':>6} | {'SGD+Momentum':>14} | {'Adam':>14} | {'Delta':>10}")
    print("-" * 52)

    for step in range(0, NUM_STEPS, 50):
        window = 10
        lo = max(0, step - window)
        hi = min(NUM_STEPS, step + window)
        sgd_avg = sum(sgd_losses[lo:hi]) / max(1, hi - lo)
        adam_avg = sum(adam_losses[lo:hi]) / max(1, hi - lo)
        delta = sgd_avg - adam_avg
        marker = "<-- Adam wins" if delta > 0.05 else ("<-- SGD wins" if delta < -0.05 else "")
        print(f"{step + 1:>6} | {sgd_avg:>14.4f} | {adam_avg:>14.4f} | {delta:>+10.4f}  {marker}")

    print()

    # === GRADIENT NORM ANALYSIS ===
    # Gradient norms reveal the stability of each optimizer's trajectory.
    # Adam's adaptive scaling should produce more stable (lower variance) norms.
    print("=" * 70)
    print("GRADIENT NORM STATISTICS")
    print("=" * 70)

    def norm_stats(norms: list[float]) -> tuple[float, float, float]:
        avg = sum(norms) / len(norms)
        variance = sum((n - avg) ** 2 for n in norms) / len(norms)
        std = math.sqrt(variance)
        return avg, std, max(norms)

    sgd_avg_norm, sgd_std_norm, sgd_max_norm = norm_stats(sgd_grad_norms)
    adam_avg_norm, adam_std_norm, adam_max_norm = norm_stats(adam_grad_norms)

    print(f"{'Metric':<24} | {'SGD+Momentum':>14} | {'Adam':>14}")
    print("-" * 58)
    print(f"{'Mean gradient norm':<24} | {sgd_avg_norm:>14.4f} | {adam_avg_norm:>14.4f}")
    print(f"{'Std gradient norm':<24} | {sgd_std_norm:>14.4f} | {adam_std_norm:>14.4f}")
    print(f"{'Max gradient norm':<24} | {sgd_max_norm:>14.4f} | {adam_max_norm:>14.4f}")
    print()

    print("Interpretation: Adam's gradient norms are typically more stable (lower std)")
    print("because the adaptive scaling in the denominator normalizes gradient magnitudes.")
    print("SGD+Momentum passes raw gradients through, so norms fluctuate more with")
    print("the stochasticity of mini-batch sampling.\n")

    # === FINAL COMPARISON TABLE ===
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    sgd_final = sum(sgd_losses[-50:]) / 50
    adam_final = sum(adam_losses[-50:]) / 50
    sgd_best = min(sgd_losses)
    adam_best = min(adam_losses)

    # Steps to reach threshold
    threshold = 3.0
    sgd_steps_str = "never"
    adam_steps_str = "never"
    for i, v in enumerate(sgd_losses):
        if v < threshold:
            sgd_steps_str = str(i + 1)
            break
    for i, v in enumerate(adam_losses):
        if v < threshold:
            adam_steps_str = str(i + 1)
            break

    print(f"{'Metric':<28} | {'SGD+Momentum':>14} | {'Adam':>14}")
    print("-" * 62)
    print(f"{'Final Train Loss (avg 50)':<28} | {sgd_final:>14.4f} | {adam_final:>14.4f}")
    print(f"{'Best Train Loss':<28} | {sgd_best:>14.4f} | {adam_best:>14.4f}")
    print(f"{'Validation Loss':<28} | {sgd_val_loss:>14.4f} | {adam_val_loss:>14.4f}")
    print(f"{'Steps to loss < ' + str(threshold):<28} | {sgd_steps_str:>14} | {adam_steps_str:>14}")
    print(f"{'Memory (moment buffers)':<28} | {'1x (velocity)':>14} | {'2x (m + v)':>14}")
    print(f"{'Training time (s)':<28} | {sgd_time:>14.1f} | {adam_time:>14.1f}")
    print("=" * 70)

    # === GENERALIZATION ANALYSIS ===
    print()
    print("Generalization gap (val_loss - train_loss):")
    sgd_gap = sgd_val_loss - sgd_final
    adam_gap = adam_val_loss - adam_final
    print(f"  SGD+Momentum: {sgd_gap:+.4f}")
    print(f"  Adam:         {adam_gap:+.4f}")

    if sgd_gap < adam_gap:
        print("  -> SGD+Momentum generalizes better (smaller gap)")
    elif adam_gap < sgd_gap:
        print("  -> Adam generalizes better (smaller gap)")
    else:
        print("  -> Similar generalization")

    # === INFERENCE ===
    print("\n--- Generating names with Adam model ---\n")

    embedding, projection = adam_params
    temperature = 0.8

    for sample_idx in range(10):
        token_id = BOS
        generated: list[str] = []

        for _ in range(20):
            emb = [v.data for v in embedding[token_id]]
            logits_data = [
                sum(projection[j][k].data * emb[k] for k in range(N_EMBD))
                for j in range(VOCAB_SIZE)
            ]

            max_logit = max(logits_data)
            exp_vals = [math.exp((v - max_logit) / temperature) for v in logits_data]
            total = sum(exp_vals)
            probs = [e / total for e in exp_vals]

            token_id = random.choices(range(VOCAB_SIZE), weights=probs)[0]
            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")

    # === KEY TAKEAWAYS ===
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("  1. Adam converges faster because per-parameter adaptive scaling")
    print("     automatically handles parameters with different gradient magnitudes.")
    print("  2. SGD+Momentum is simpler and uses less memory (1 buffer vs 2).")
    print("  3. The generalization gap can favor SGD: the fixed learning rate's")
    print("     stochastic noise acts as implicit regularization, pushing the model")
    print("     toward flatter minima that generalize better (Smith & Le, 2018).")
    print("  4. In practice: use Adam for prototyping and most tasks, SGD+momentum")
    print("     for large-scale training where generalization is critical.")
