"""
Before attention conquered everything -- how sequences were modeled with recurrence, and why
gating was the breakthrough that made RNNs actually work.
"""
# Reference: Vanilla RNN dates to the 1980s (Rumelhart et al.). GRU (Gated Recurrent Unit)
# introduced by Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for
# Statistical Machine Translation" (2014). This implementation demonstrates both side-by-side
# on the same character-level language modeling task to show why gating matters.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_HIDDEN = 32       # hidden state dimension (compact for 7-minute runtime)
SEQ_LEN = 16        # maximum sequence length
LEARNING_RATE = 0.1   # SGD learning rate — 10x higher than microgpt's Adam because plain
                      # SGD needs much larger steps to compensate for lack of adaptive rates
NUM_STEPS = 3000    # training steps per model (3000 vanilla RNN, 3000 GRU)
TRAIN_SIZE = 200    # small training subset so each name is seen ~15x in 3000 steps

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~800 parameters per model (vanilla RNN and GRU have similar sizes).
# Production RNNs had millions. The architecture is correct; this is a toy scale
# for pedagogical clarity.


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

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # scalar float value
        self.grad = 0.0           # accumulated gradient (∂Loss/∂self)
        self._children = children # parent Values in the computation graph
        self._local_grads = local_grads  # ∂self/∂child for each child

    # Arithmetic operations
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

    # Activation functions
    def tanh(self):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def sigmoid(self):
        # sigmoid(x) = 1 / (1 + exp(-x))
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        # This is the gating activation for GRUs: values in [0,1] act as "forget"
        # and "update" weights. When sigmoid(x) ≈ 0, the gate blocks information;
        # when ≈ 1, the gate passes information through.
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        # We assume input is already clamped (see safe_log below)
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule. For a composite function
        f(g(h(x))), the chain rule says df/dx = (df/dg) * (dg/dh) * (dh/dx).
        The topological sort ensures we compute df/dg before we need it for df/dh.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
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


# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following addition:
# - sigmoid(): Required for GRU gating (z_t and r_t computations)
# Base operations (add, mul, tanh, exp, relu, pow, backward) are identical
# to the canonical spec.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    Standard deviation of 0.08 is chosen empirically for this tiny model.
    Larger models typically use std = 1/sqrt(d_in) (Xavier/Glorot initialization).
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_vanilla_rnn_params():
    """Initialize parameters for vanilla RNN.

    Vanilla RNN update rule:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y

    Returns state_dict with weight matrices.
    """
    params = {}
    params['W_xh'] = make_matrix(N_HIDDEN, VOCAB_SIZE)  # input-to-hidden
    params['W_hh'] = make_matrix(N_HIDDEN, N_HIDDEN)    # hidden-to-hidden (recurrent)
    params['b_h'] = [Value(0.0) for _ in range(N_HIDDEN)]  # hidden bias

    params['W_hy'] = make_matrix(VOCAB_SIZE, N_HIDDEN)  # hidden-to-output
    params['b_y'] = [Value(0.0) for _ in range(VOCAB_SIZE)]  # output bias

    return params


def init_gru_params():
    """Initialize parameters for GRU.

    GRU update rules:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})           # update gate
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})           # reset gate
        h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate       # interpolate

    The GRU uses 3 weight matrices per gate type (z, r, h), doubling the
    parameter count vs vanilla RNN, but the gating mechanism is what makes
    the difference, not the parameter count.
    """
    params = {}
    # Update gate
    params['W_xz'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hz'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # Reset gate
    params['W_xr'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hr'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # Candidate hidden state
    params['W_xh'] = make_matrix(N_HIDDEN, VOCAB_SIZE)
    params['W_hh'] = make_matrix(N_HIDDEN, N_HIDDEN)

    # Output projection (shared with vanilla RNN structure)
    params['W_hy'] = make_matrix(VOCAB_SIZE, N_HIDDEN)
    params['b_y'] = [Value(0.0) for _ in range(VOCAB_SIZE)]

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]], b: list[Value] = None) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x + b (bias optional).

    For weight matrix W with shape [n_out, n_in] and input vector x with
    shape [n_in], computes output y with shape [n_out].
    """
    y = [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]
    if b is not None:
        y = [y_i + b_i for y_i, b_i in zip(y, b)]
    return y


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Softmax is translation-invariant: softmax(x) = softmax(x - c) for any c.
    We subtract max(x) before exp() to prevent overflow.
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm for numerical stability in loss computation.

    Prevents log(0) which returns -inf and breaks gradient backpropagation.
    Critical: we must keep `prob` as a child node so gradients flow back through
    the computation graph.
    """
    clamped = max(prob.data, 1e-10)
    # Build the log node manually with prob as its child, preserving the graph.
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === VANILLA RNN FORWARD PASS ===

def vanilla_rnn_forward(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """Single-step vanilla RNN forward pass.

    Math: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
          y_t = W_hy @ h_t + b_y

    The recurrent connection (W_hh @ h_{t-1}) is what makes this "recurrent" --
    the hidden state carries information from previous timesteps. However,
    backpropagating through this recurrence causes gradients to be repeatedly
    multiplied by W_hh, which leads to exponential decay (vanishing gradients)
    or explosion depending on the spectral radius of W_hh.

    Returns: (logits, new_hidden_state)
    """
    # Compute new hidden state
    h_input = linear(x, params['W_xh'])
    h_recurrent = linear(h_prev, params['W_hh'])
    h_combined = [h_i + h_r + params['b_h'][i] for i, (h_i, h_r) in enumerate(zip(h_input, h_recurrent))]
    h = [h_i.tanh() for h_i in h_combined]

    # Compute output logits
    logits = linear(h, params['W_hy'], params['b_y'])

    return logits, h


# === GRU FORWARD PASS ===

def gru_forward(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """Single-step GRU forward pass.

    Math:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})           # update gate
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})           # reset gate
        h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate

    The update gate z_t acts as a "gradient highway": when z_t ≈ 0, h_t = h_{t-1}
    (we keep the old hidden state), so dh_t/dh_{t-1} = 1. This identity gradient
    flow prevents vanishing gradients -- the derivative doesn't get multiplied by
    weight matrices, it just passes through. This is the core insight of gating.

    The reset gate r_t controls how much past information is used when computing
    the candidate hidden state. When r_t ≈ 0, the network ignores h_{t-1} and
    starts fresh from the input x_t.

    Returns: (logits, new_hidden_state)
    """
    # Update gate: controls how much of the new state vs old state to use
    z_input = linear(x, params['W_xz'])
    z_recurrent = linear(h_prev, params['W_hz'])
    z = [(z_i + z_r).sigmoid() for z_i, z_r in zip(z_input, z_recurrent)]

    # Reset gate: controls how much of the previous state to use for candidate
    r_input = linear(x, params['W_xr'])
    r_recurrent = linear(h_prev, params['W_hr'])
    r = [(r_i + r_r).sigmoid() for r_i, r_r in zip(r_input, r_recurrent)]

    # Candidate hidden state: computed using reset-gated previous state
    # The element-wise multiplication (r_t * h_{t-1}) "resets" components of the
    # hidden state that are irrelevant for the current input.
    h_input = linear(x, params['W_xh'])
    h_reset = [r_i * h_i for r_i, h_i in zip(r, h_prev)]
    h_recurrent = linear(h_reset, params['W_hh'])
    h_candidate = [(h_i + h_r).tanh() for h_i, h_r in zip(h_input, h_recurrent)]

    # Interpolate between old state and candidate: z_t controls the blend
    # When z_t = 0: h_t = h_{t-1} (keep old state, "forget" new input)
    # When z_t = 1: h_t = h_candidate (fully update with new input)
    # This linear interpolation creates the gradient highway: dh_t/dh_{t-1}
    # includes a (1 - z_t) term that bypasses the weight matrices.
    h = [(1 - z_i) * h_prev_i + z_i * h_cand_i
         for z_i, h_prev_i, h_cand_i in zip(z, h_prev, h_candidate)]

    # Compute output logits
    logits = linear(h, params['W_hy'], params['b_y'])

    return logits, h


# === TRAINING FUNCTION ===

def train_rnn(
    docs: list[str],
    unique_chars: list[str],
    forward_fn,
    params: dict,
    model_name: str
) -> tuple[float, list[float]]:
    """Train an RNN model (vanilla or GRU) and track gradient norms.

    Args:
        docs: Training documents (names)
        unique_chars: Vocabulary (character list)
        forward_fn: vanilla_rnn_forward or gru_forward
        params: Model parameters (state_dict)
        model_name: "Vanilla RNN" or "GRU" (for logging)

    Returns:
        (final_loss, gradient_norms_per_timestep)
    """
    BOS = len(unique_chars)
    VOCAB_SIZE_LOCAL = len(unique_chars) + 1

    # Flatten all parameters into a single list for optimizer
    param_list = []
    for key, val in params.items():
        if isinstance(val, list) and isinstance(val[0], Value):
            param_list.extend(val)
        elif isinstance(val, list) and isinstance(val[0], list):
            for row in val:
                param_list.extend(row)

    print(f"Training {model_name}...")
    print(f"Parameters: {len(param_list):,}")

    final_loss_value = 0.0

    for step in range(NUM_STEPS):
        # Cycle through dataset
        doc = docs[step % len(docs)]

        # Tokenize: [BOS, char_0, char_1, ..., char_n, BOS]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]

        # Truncate to sequence length
        seq_len = min(SEQ_LEN, len(tokens) - 1)

        # Initialize hidden state to zeros
        h = [Value(0.0) for _ in range(N_HIDDEN)]

        # Forward pass through sequence
        losses = []
        for pos in range(seq_len):
            # One-hot encode input token
            x_onehot = [Value(1.0 if i == tokens[pos] else 0.0) for i in range(VOCAB_SIZE_LOCAL)]

            # Single timestep forward
            logits, h = forward_fn(x_onehot, h, params)

            # Compute loss
            probs = softmax(logits)
            target = tokens[pos + 1]
            loss_t = -safe_log(probs[target])
            losses.append(loss_t)

        # Average loss over sequence
        loss = (1.0 / seq_len) * sum(losses)

        # Backward pass (Backpropagation Through Time - BPTT)
        loss.backward()

        # SGD update
        for param in param_list:
            param.data -= LEARNING_RATE * param.grad
            param.grad = 0.0

        final_loss_value = loss.data

        # Print progress
        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    print(f"{model_name} training complete. Final loss: {final_loss_value:.4f}\n")

    # === GRADIENT NORM TRACKING ===
    # Measure gradient norms across a LONG sequence to demonstrate vanishing gradients.
    # Short names (~6 chars) don't show dramatic gradient decay because there aren't
    # enough timesteps for exponential decay to accumulate. We concatenate multiple
    # names to create a sequence of length SEQ_LEN for a clearer demonstration.
    print(f"Measuring gradient norms for {model_name}...")

    # Build a long token sequence by concatenating names until we reach SEQ_LEN
    long_tokens = [BOS]
    for doc in docs:
        long_tokens.extend([unique_chars.index(ch) for ch in doc])
        long_tokens.append(BOS)
        if len(long_tokens) > SEQ_LEN:
            break
    seq_len = min(SEQ_LEN, len(long_tokens) - 1)

    # Forward pass through the long sequence
    h = [Value(0.0) for _ in range(N_HIDDEN)]
    hidden_states = []

    for pos in range(seq_len):
        x_onehot = [Value(1.0 if i == long_tokens[pos] else 0.0) for i in range(VOCAB_SIZE_LOCAL)]
        logits, h = forward_fn(x_onehot, h, params)
        hidden_states.append(h)

    # Compute loss only at the final timestep: gradient must flow ALL the way back
    # through seq_len timesteps. This is the scenario where vanishing gradients
    # are most severe — the loss signal must traverse the entire sequence.
    probs = softmax(logits)
    target = long_tokens[seq_len]
    loss = -safe_log(probs[target])

    # Backward pass
    loss.backward()

    # Compute L2 norm of gradient for each hidden state
    # ||dL/dh_t|| = sqrt(sum_i (dL/dh_t[i])^2)
    # For vanilla RNN: expect exponential decay as t decreases (further from loss)
    # For GRU: expect more uniform norms due to gradient highways through gates
    gradient_norms = []
    for h_t in hidden_states:
        norm_sq = sum(h_i.grad ** 2 for h_i in h_t)
        norm = math.sqrt(norm_sq)
        gradient_norms.append(norm)

    # Print gradient norms
    print(f"Gradient norms per timestep (sequence length {seq_len}):")
    for t, norm in enumerate(gradient_norms):
        bar = "#" * min(50, int(norm * 100))
        print(f"  t={t:>2}: ||dL/dh_t|| = {norm:.6f}  {bar}")

    # Compute ratio: first / last (measures how much gradient decays going backward)
    # < 1 means gradients vanish going backward through time (first < last)
    # The more this ratio approaches 0, the worse the vanishing gradient problem.
    if gradient_norms[-1] > 1e-10:
        ratio = gradient_norms[0] / gradient_norms[-1]
    else:
        ratio = 0.0
    print(f"Gradient norm ratio (first/last): {ratio:.6f}")
    print(f"  (< 0.01 = severe vanishing, > 0.1 = gradient highway active)\n")

    return final_loss_value, gradient_norms


# === INFERENCE FUNCTION ===

def generate_names(
    params: dict,
    forward_fn,
    unique_chars: list[str],
    num_samples: int = 10,
    model_name: str = "Model"
) -> list[str]:
    """Generate names from a trained RNN model."""
    BOS = len(unique_chars)
    VOCAB_SIZE_LOCAL = len(unique_chars) + 1

    print(f"Generating {num_samples} samples from {model_name}:")

    samples = []
    for _ in range(num_samples):
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        token_id = BOS
        generated = []

        for pos in range(SEQ_LEN):
            # One-hot encode current token
            x_onehot = [Value(1.0 if i == token_id else 0.0) for i in range(VOCAB_SIZE_LOCAL)]

            # Forward
            logits, h = forward_fn(x_onehot, h, params)

            # Sample from probabilities
            probs = softmax(logits)
            token_id = random.choices(
                range(VOCAB_SIZE_LOCAL),
                weights=[p.data for p in probs]
            )[0]

            # Stop if BOS (end-of-sequence)
            if token_id == BOS:
                break

            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        samples.append(name)
        print(f"  {''.join(generated)}")

    print()
    return samples


# === MAIN ===

if __name__ == "__main__":
    # -- Load and prepare data --
    print("Loading data...")
    all_docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(all_docs)

    # Use a small training subset so each name is seen multiple times in 500 steps.
    # With 200 names and 500 steps, each name is seen ~2.5 times — enough to learn
    # character-level patterns without requiring thousands of gradient steps.
    docs = all_docs[:TRAIN_SIZE]

    # Build vocabulary from all names (so we don't miss any characters)
    unique_chars = sorted(set(''.join(all_docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(all_docs)} documents, training on {len(docs)}")
    print(f"Vocabulary size: {VOCAB_SIZE} (characters + BOS token)\n")

    # === TRAIN VANILLA RNN ===
    vanilla_params = init_vanilla_rnn_params()
    vanilla_loss, vanilla_grad_norms = train_rnn(
        docs, unique_chars, vanilla_rnn_forward, vanilla_params, "Vanilla RNN"
    )

    # === TRAIN GRU ===
    gru_params = init_gru_params()
    gru_loss, gru_grad_norms = train_rnn(
        docs, unique_chars, gru_forward, gru_params, "GRU"
    )

    # === COMPARISON TABLE ===
    print("=" * 70)
    print("COMPARISON: Vanilla RNN vs GRU")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Vanilla RNN':<15} | {'GRU':<15}")
    print("-" * 70)
    print(f"{'Final Loss':<30} | {vanilla_loss:<15.4f} | {gru_loss:<15.4f}")

    # Gradient norm ratios: first/last measures backward gradient decay
    # Lower ratio = worse vanishing gradients (gradient decays more going backward)
    vanilla_ratio = vanilla_grad_norms[0] / vanilla_grad_norms[-1] if vanilla_grad_norms[-1] > 1e-10 else 0.0
    gru_ratio = gru_grad_norms[0] / gru_grad_norms[-1] if gru_grad_norms[-1] > 1e-10 else 0.0

    print(f"{'Gradient Norm Ratio':<30} | {vanilla_ratio:<15.6f} | {gru_ratio:<15.6f}")
    print(f"{'(first/last, higher=better)':<30} |                 |                ")
    print("-" * 70)

    # Why the difference matters
    print("\nWhy the gradient norm ratio matters:")
    print("  Vanilla RNN: Gradient norms decay exponentially due to repeated")
    print("               multiplication by W_hh. Spectral radius < 1 causes")
    print("               gradients to vanish as they propagate backward through time.")
    print("  GRU:         Update gate creates 'gradient highways' where dh_t/dh_{t-1} ≈ 1")
    print("               when z_t ≈ 0. This identity connection bypasses weight matrices,")
    print("               preserving gradient magnitude across long sequences.\n")

    # === INFERENCE ===
    print("=" * 70)
    print("GENERATED SAMPLES")
    print("=" * 70)
    print()

    vanilla_samples = generate_names(
        vanilla_params, vanilla_rnn_forward, unique_chars, num_samples=10, model_name="Vanilla RNN"
    )

    gru_samples = generate_names(
        gru_params, gru_forward, unique_chars, num_samples=10, model_name="GRU"
    )

    # === HISTORICAL CONTEXT ===
    print("=" * 70)
    print("HISTORICAL ARC")
    print("=" * 70)
    print("  1990s:  Vanilla RNNs introduced — theoretically powerful, but gradients")
    print("          vanish in practice, limiting them to short sequences (~10 steps).")
    print("  1997:   LSTM (Long Short-Term Memory) introduced gating to solve the")
    print("          vanishing gradient problem. Became the standard for sequence modeling.")
    print("  2014:   GRU (Gated Recurrent Unit) simplified LSTM's 3 gates to 2, achieving")
    print("          similar performance with fewer parameters and faster training.")
    print("  2017:   Transformers (Attention Is All You Need) replaced recurrence entirely,")
    print("          using attention for O(1) path length between any two positions.")
    print("  Today:  RNNs are largely historical, but the gating principle (learned routing")
    print("          of gradients) lives on in modern architectures like state-space models.")
