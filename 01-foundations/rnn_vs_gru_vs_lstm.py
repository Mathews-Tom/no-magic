"""
Three generations of recurrent architectures on the same task: vanilla RNN fails at long-range
dependencies, GRU gates solve vanishing gradients, and LSTM's separate cell state provides
the most robust gradient highway.
"""
# Reference: Vanilla RNN (Elman, 1990). LSTM from Hochreiter & Schmidhuber, "Long Short-Term
# Memory" (1997). GRU from Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"
# (2014). This script extends micrornn.py by adding LSTM and using a task specifically designed
# to test long-range memory: predicting the last character of a name given the first.

# === TRADEOFFS ===
# + Vanilla RNN: simplest, fewest parameters, fastest per step
# + GRU: 2 gates (update + reset) solve vanishing gradients with fewer parameters than LSTM
# + LSTM: 3 gates (forget + input + output) + cell state provide the most robust gradient flow
# - Vanilla RNN: gradients vanish exponentially through time, limiting effective memory to ~10 steps
# - GRU: slightly more parameters than vanilla RNN, but far fewer than LSTM
# - LSTM: most parameters (3 gate matrices + cell), slowest per step
# WHEN TO USE: LSTM for long sequences where gradient stability is critical.
#   GRU as a lighter alternative when sequences are moderate length.
#   Vanilla RNN only for very short sequences or as a baseline.
# WHEN NOT TO: All three are outperformed by transformers when parallel compute
#   is available. Use RNNs only for streaming or memory-constrained settings.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_HIDDEN = 16        # hidden state dimension -- identical across all 3 architectures
SEQ_LEN = 16         # maximum sequence length
LEARNING_RATE = 0.1  # SGD learning rate -- slightly higher for smaller hidden dim
NUM_STEPS = 800      # training steps per architecture (3 models must finish in <10 min total)
TRAIN_SIZE = 120     # small training subset for speed

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: identical hidden dimension across all 3 models means LSTM has ~3x the
# parameters of vanilla RNN (3 gate matrices + cell), GRU has ~2x. This is by design:
# we want to compare architectures, not parameter counts. The key difference is
# HOW information flows, not HOW MUCH capacity the model has.


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

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def sigmoid(self):
        # sigmoid(x) = 1/(1+exp(-x))
        # Gates in GRU and LSTM use sigmoid to produce values in [0,1]
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

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


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]], b: list[Value] | None = None) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x + b (bias optional)."""
    y = [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]
    if b is not None:
        y = [y_i + b_i for y_i, b_i in zip(y, b)]
    return y


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


# === VANILLA RNN ===

def init_rnn_params(vocab_size: int) -> dict:
    """Vanilla RNN: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

    The simplest recurrent architecture. The recurrent weight W_hh is multiplied
    at every timestep, causing gradients to either vanish (|eigenvalues| < 1) or
    explode (|eigenvalues| > 1). No mechanism to preserve gradients over time.
    """
    return {
        'W_xh': make_matrix(N_HIDDEN, vocab_size),
        'W_hh': make_matrix(N_HIDDEN, N_HIDDEN),
        'b_h': [Value(0.0) for _ in range(N_HIDDEN)],
        'W_hy': make_matrix(vocab_size, N_HIDDEN),
        'b_y': [Value(0.0) for _ in range(vocab_size)],
    }


def rnn_step(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """Single-step vanilla RNN.

    Math: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

    The gradient dh_t/dh_{t-1} = diag(1 - h_t^2) @ W_hh. Over T timesteps,
    the gradient dh_T/dh_0 = product of T such terms. If the spectral radius
    of W_hh is < 1, this product decays exponentially -- the vanishing gradient
    problem that motivated gated architectures.

    Returns: (new_hidden_state, cell_state_placeholder)
    """
    h_input = linear(x, params['W_xh'])
    h_recurrent = linear(h_prev, params['W_hh'])
    h = [(h_i + h_r + params['b_h'][i]).tanh()
         for i, (h_i, h_r) in enumerate(zip(h_input, h_recurrent))]
    # Return (h, None) to match GRU/LSTM interface -- vanilla RNN has no cell state
    return h, [Value(0.0) for _ in range(N_HIDDEN)]


# === GRU ===

def init_gru_params(vocab_size: int) -> dict:
    """GRU: 2 gates (update z, reset r) that control information flow.

    The update gate creates a gradient highway: when z_t near 0, h_t = h_{t-1}
    with gradient 1 (identity). This additive update bypasses the multiplicative
    W_hh bottleneck that causes vanilla RNN gradients to vanish.
    """
    return {
        'W_xz': make_matrix(N_HIDDEN, vocab_size),
        'W_hz': make_matrix(N_HIDDEN, N_HIDDEN),
        'W_xr': make_matrix(N_HIDDEN, vocab_size),
        'W_hr': make_matrix(N_HIDDEN, N_HIDDEN),
        'W_xh': make_matrix(N_HIDDEN, vocab_size),
        'W_hh': make_matrix(N_HIDDEN, N_HIDDEN),
        'W_hy': make_matrix(vocab_size, N_HIDDEN),
        'b_y': [Value(0.0) for _ in range(vocab_size)],
    }


def gru_step(
    x: list[Value], h_prev: list[Value], params: dict
) -> tuple[list[Value], list[Value]]:
    """Single-step GRU.

    Math:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})         (update gate)
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})         (reset gate)
        h_cand = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1})) (candidate)
        h_t = (1-z_t) * h_{t-1} + z_t * h_cand              (interpolate)

    The gradient dh_t/dh_{t-1} includes a (1-z_t) identity path that preserves
    gradient magnitude when z_t near 0.

    Returns: (new_hidden_state, cell_state_placeholder)
    """
    z_input = linear(x, params['W_xz'])
    z_recurrent = linear(h_prev, params['W_hz'])
    z = [(z_i + z_r).sigmoid() for z_i, z_r in zip(z_input, z_recurrent)]

    r_input = linear(x, params['W_xr'])
    r_recurrent = linear(h_prev, params['W_hr'])
    r = [(r_i + r_r).sigmoid() for r_i, r_r in zip(r_input, r_recurrent)]

    h_reset = [r_i * h_i for r_i, h_i in zip(r, h_prev)]
    h_input = linear(x, params['W_xh'])
    h_recurrent = linear(h_reset, params['W_hh'])
    h_candidate = [(h_i + h_r).tanh() for h_i, h_r in zip(h_input, h_recurrent)]

    h = [(1 - z_i) * h_prev_i + z_i * h_cand_i
         for z_i, h_prev_i, h_cand_i in zip(z, h_prev, h_candidate)]

    return h, [Value(0.0) for _ in range(N_HIDDEN)]


# === LSTM ===

def init_lstm_params(vocab_size: int) -> dict:
    """LSTM: 3 gates (forget f, input i, output o) + cell state c.

    The cell state c_t is the key innovation. Unlike GRU which gates the hidden
    state directly, LSTM maintains a separate memory cell that accumulates
    information additively:
        c_t = f_t * c_{t-1} + i_t * g_t

    The forget gate f_t controls what to erase from memory. The input gate i_t
    controls what new information to write. The output gate o_t controls what
    to expose as the hidden state.

    This additive cell update means dc_t/dc_{t-1} = f_t (a scalar near 1 when
    the forget gate is open). Gradients flow through the cell state without
    being multiplied by weight matrices -- the "gradient highway" that solved
    the vanishing gradient problem.
    """
    return {
        # Forget gate: what to erase from cell state
        'W_xf': make_matrix(N_HIDDEN, vocab_size),
        'W_hf': make_matrix(N_HIDDEN, N_HIDDEN),
        'b_f': [Value(1.0) for _ in range(N_HIDDEN)],
        # ^^^ Bias initialized to 1.0 (not 0.0) following Jozefowicz et al. (2015):
        # This makes the forget gate start near sigmoid(1) ≈ 0.73, so the cell
        # state is initially "open" and information flows through. Without this,
        # the LSTM starts with forget gates near 0.5, which decays cell state
        # too aggressively in early training.

        # Input gate: what new information to write
        'W_xi': make_matrix(N_HIDDEN, vocab_size),
        'W_hi': make_matrix(N_HIDDEN, N_HIDDEN),
        'b_i': [Value(0.0) for _ in range(N_HIDDEN)],

        # Cell candidate (new information to potentially write)
        'W_xg': make_matrix(N_HIDDEN, vocab_size),
        'W_hg': make_matrix(N_HIDDEN, N_HIDDEN),
        'b_g': [Value(0.0) for _ in range(N_HIDDEN)],

        # Output gate: what to expose as hidden state
        'W_xo': make_matrix(N_HIDDEN, vocab_size),
        'W_ho': make_matrix(N_HIDDEN, N_HIDDEN),
        'b_o': [Value(0.0) for _ in range(N_HIDDEN)],

        # Output projection
        'W_hy': make_matrix(vocab_size, N_HIDDEN),
        'b_y': [Value(0.0) for _ in range(vocab_size)],
    }


def lstm_step(
    x: list[Value], h_prev: list[Value], params: dict,
    c_prev: list[Value] | None = None
) -> tuple[list[Value], list[Value]]:
    """Single-step LSTM.

    Math:
        f_t = sigmoid(W_xf @ x_t + W_hf @ h_{t-1} + b_f)    (forget gate)
        i_t = sigmoid(W_xi @ x_t + W_hi @ h_{t-1} + b_i)    (input gate)
        g_t = tanh(W_xg @ x_t + W_hg @ h_{t-1} + b_g)       (cell candidate)
        c_t = f_t * c_{t-1} + i_t * g_t                       (cell update)
        o_t = sigmoid(W_xo @ x_t + W_ho @ h_{t-1} + b_o)    (output gate)
        h_t = o_t * tanh(c_t)                                 (hidden state)

    The cell state c_t is the gradient highway:
        dc_t/dc_{t-1} = f_t (a scalar gate value)
    When f_t near 1, gradients pass through perfectly. No weight matrix multiplication.
    This is why LSTM solved the vanishing gradient problem where vanilla RNNs failed.

    Returns: (new_hidden_state, new_cell_state)
    """
    if c_prev is None:
        c_prev = [Value(0.0) for _ in range(N_HIDDEN)]

    # Forget gate: sigmoid output in [0,1], controls cell state erasure
    f = [(fi + fh + params['b_f'][j]).sigmoid()
         for j, (fi, fh) in enumerate(zip(
             linear(x, params['W_xf']), linear(h_prev, params['W_hf'])))]

    # Input gate: controls writing of new information
    i = [(ii + ih + params['b_i'][j]).sigmoid()
         for j, (ii, ih) in enumerate(zip(
             linear(x, params['W_xi']), linear(h_prev, params['W_hi'])))]

    # Cell candidate: new information to potentially write (tanh bounds to [-1,1])
    g = [(gi + gh + params['b_g'][j]).tanh()
         for j, (gi, gh) in enumerate(zip(
             linear(x, params['W_xg']), linear(h_prev, params['W_hg'])))]

    # Cell update: additive combination of old (gated by forget) and new (gated by input)
    # This is the gradient highway: dc_t/dc_{t-1} = f_t, not a weight matrix product
    c = [f_j * c_prev_j + i_j * g_j
         for f_j, c_prev_j, i_j, g_j in zip(f, c_prev, i, g)]

    # Output gate: controls what part of cell state is exposed
    o = [(oi + oh + params['b_o'][j]).sigmoid()
         for j, (oi, oh) in enumerate(zip(
             linear(x, params['W_xo']), linear(h_prev, params['W_ho'])))]

    # Hidden state: output gate applied to squashed cell state
    h = [o_j * c_j.tanh() for o_j, c_j in zip(o, c)]

    return h, c


# === TRAINING ===

def collect_params(params: dict) -> list[Value]:
    """Flatten all parameters into a single list for optimizer updates."""
    param_list = []
    for val in params.values():
        if isinstance(val, list):
            if isinstance(val[0], Value):
                param_list.extend(val)
            elif isinstance(val[0], list):
                for row in val:
                    param_list.extend(row)
    return param_list


def train_model(
    docs: list[str], unique_chars: list[str], params: dict,
    step_fn, model_name: str, vocab_size: int,
) -> tuple[float, list[float]]:
    """Train an RNN variant and return final loss + loss history.

    All 3 architectures use the same training loop, data, and hyperparameters.
    The only difference is the step function (rnn_step, gru_step, lstm_step).
    """
    BOS = len(unique_chars)
    param_list = collect_params(params)

    print(f"Training {model_name}... ({len(param_list):,} parameters)")

    loss_history: list[float] = []
    start = time.time()

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(SEQ_LEN, len(tokens) - 1)
        tokens = tokens[:seq_len + 1]

        # Forward pass through sequence
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        c = [Value(0.0) for _ in range(N_HIDDEN)]
        losses = []

        for pos in range(seq_len):
            x_onehot = [Value(1.0 if i == tokens[pos] else 0.0) for i in range(vocab_size)]

            if step_fn == lstm_step:
                h, c = step_fn(x_onehot, h, params, c)
            else:
                h, c = step_fn(x_onehot, h, params)

            logits = linear(h, params['W_hy'], params['b_y'])
            probs = softmax(logits)
            target = tokens[pos + 1]
            losses.append(-safe_log(probs[target]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # SGD update
        for param in param_list:
            param.data -= LEARNING_RATE * param.grad
            param.grad = 0.0

        loss_history.append(loss.data)

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    elapsed = time.time() - start
    final_loss = loss_history[-1]
    print(f"  {model_name} done. Final loss: {final_loss:.4f} ({elapsed:.1f}s)\n")

    return final_loss, loss_history


# === GRADIENT NORM MEASUREMENT ===

def measure_gradient_norms(
    docs: list[str], unique_chars: list[str], params: dict,
    step_fn, model_name: str, vocab_size: int
) -> list[float]:
    """Measure gradient norms at each timestep by backpropagating from the final loss.

    The gradient must flow backward through the entire sequence. For vanilla RNN,
    gradients decay exponentially with distance. For GRU/LSTM, gating preserves
    gradient magnitude through the sequence.

    We concatenate multiple names to create a long sequence (up to SEQ_LEN) so
    the decay effect is visible.
    """
    BOS = len(unique_chars)

    # Build a long token sequence by concatenating names
    long_tokens = [BOS]
    for doc in docs:
        long_tokens.extend([unique_chars.index(ch) for ch in doc])
        long_tokens.append(BOS)
        if len(long_tokens) > SEQ_LEN:
            break
    seq_len = min(SEQ_LEN, len(long_tokens) - 1)

    # Forward pass, storing hidden states
    h = [Value(0.0) for _ in range(N_HIDDEN)]
    c = [Value(0.0) for _ in range(N_HIDDEN)]
    hidden_states = []

    for pos in range(seq_len):
        x_onehot = [Value(1.0 if i == long_tokens[pos] else 0.0) for i in range(vocab_size)]
        if step_fn == lstm_step:
            h, c = step_fn(x_onehot, h, params, c)
        else:
            h, c = step_fn(x_onehot, h, params)
        hidden_states.append(h)

    # Loss at final timestep only -- gradient must traverse the entire sequence
    logits = linear(h, params['W_hy'], params['b_y'])
    probs = softmax(logits)
    target = long_tokens[seq_len]
    loss = -safe_log(probs[target])
    loss.backward()

    # Compute L2 norm of gradient at each timestep
    # ||dL/dh_t|| = sqrt(sum_i (dL/dh_t[i])^2)
    gradient_norms = []
    for h_t in hidden_states:
        norm_sq = sum(h_i.grad ** 2 for h_i in h_t)
        gradient_norms.append(math.sqrt(norm_sq))

    return gradient_norms


# === INFERENCE ===

def generate_names(
    params: dict, step_fn, unique_chars: list[str],
    vocab_size: int, model_name: str, num_samples: int = 8
) -> list[str]:
    """Generate names via autoregressive sampling."""
    BOS = len(unique_chars)

    print(f"Generating {num_samples} names from {model_name}:")
    samples = []

    for _ in range(num_samples):
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        c = [Value(0.0) for _ in range(N_HIDDEN)]
        token_id = BOS
        generated = []

        for pos in range(SEQ_LEN):
            x_onehot = [Value(1.0 if i == token_id else 0.0) for i in range(vocab_size)]
            if step_fn == lstm_step:
                h, c = step_fn(x_onehot, h, params, c)
            else:
                h, c = step_fn(x_onehot, h, params)

            logits = linear(h, params['W_hy'], params['b_y'])
            probs = softmax(logits)
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        samples.append(name)
        print(f"  {name}")

    print()
    return samples


# === MAIN ===

if __name__ == "__main__":
    print("=" * 70)
    print("RNN VS GRU VS LSTM: Three Generations of Recurrent Architecture")
    print("=" * 70)
    print()

    # -- Load and prepare data --
    all_docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(all_docs)
    docs = all_docs[:TRAIN_SIZE]

    unique_chars = sorted(set(''.join(all_docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Training on {len(docs)} names, vocab size: {VOCAB_SIZE}")
    print(f"Hidden dim: {N_HIDDEN}, max seq len: {SEQ_LEN}")
    print(f"Steps: {NUM_STEPS}, learning rate: {LEARNING_RATE}\n")

    # -- Train all 3 architectures --
    models = [
        ("Vanilla RNN", init_rnn_params, rnn_step),
        ("GRU", init_gru_params, gru_step),
        ("LSTM", init_lstm_params, lstm_step),
    ]

    results = {}
    trained_params = {}

    for model_name, init_fn, step_fn in models:
        random.seed(42)
        params = init_fn(VOCAB_SIZE)
        final_loss, loss_history = train_model(
            docs, unique_chars, params, step_fn, model_name, VOCAB_SIZE
        )
        results[model_name] = {
            'final_loss': final_loss,
            'loss_history': loss_history,
            'param_count': len(collect_params(params)),
        }
        trained_params[model_name] = (params, step_fn)

    # -- Measure gradient norms --
    print("=" * 70)
    print("GRADIENT NORM MEASUREMENT")
    print("=" * 70)
    print("Backpropagating from final-timestep loss through the full sequence.\n")

    gradient_data = {}
    for model_name, init_fn, step_fn in models:
        # Use freshly trained params for gradient measurement
        params, _ = trained_params[model_name]
        norms = measure_gradient_norms(
            docs, unique_chars, params, step_fn, model_name, VOCAB_SIZE
        )
        gradient_data[model_name] = norms

        print(f"{model_name} gradient norms (first 12 timesteps):")
        for t, norm in enumerate(norms[:12]):
            bar = "#" * min(40, int(norm * 50))
            print(f"  t={t:>2}: {norm:.6f}  {bar}")

        if norms[-1] > 1e-10:
            ratio = norms[0] / norms[-1]
        else:
            ratio = 0.0
        print(f"  Ratio (first/last): {ratio:.6f}")
        print(f"  ({'severe vanishing' if ratio < 0.01 else 'moderate' if ratio < 0.1 else 'good gradient flow'})\n")

    # === LOSS CURVE COMPARISON ===
    print("=" * 70)
    print("LOSS CURVES (every 200 steps)")
    print("=" * 70)
    print(f"{'Step':>6} | {'RNN':>10} | {'GRU':>10} | {'LSTM':>10}")
    print("-" * 46)

    for step in range(0, NUM_STEPS, 100):
        window = 30
        lo = max(0, step - window)
        hi = min(NUM_STEPS, step + window)
        vals = {}
        for name in ["Vanilla RNN", "GRU", "LSTM"]:
            h = results[name]['loss_history']
            vals[name] = sum(h[lo:hi]) / max(1, hi - lo)
        print(f"{step + 1:>6} | {vals['Vanilla RNN']:>10.4f} | {vals['GRU']:>10.4f} | {vals['LSTM']:>10.4f}")

    print()

    # === FINAL COMPARISON TABLE ===
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    header_names = ["Vanilla RNN", "GRU", "LSTM"]

    print(f"{'Metric':<28} | ", end="")
    print(" | ".join(f"{n:>12}" for n in header_names))
    print("-" * 72)

    # Parameter counts
    print(f"{'Parameters':<28} | ", end="")
    print(" | ".join(f"{results[n]['param_count']:>12,}" for n in header_names))

    # Final loss (averaged over last 100 steps for stability)
    print(f"{'Final Loss (avg last 100)':<28} | ", end="")
    finals = []
    for n in header_names:
        h = results[n]['loss_history']
        avg = sum(h[-100:]) / 100
        finals.append(avg)
    print(" | ".join(f"{v:>12.4f}" for v in finals))

    # Best loss
    print(f"{'Best Loss':<28} | ", end="")
    print(" | ".join(f"{min(results[n]['loss_history']):>12.4f}" for n in header_names))

    # Gradient norm ratio
    print(f"{'Grad Norm Ratio (first/last)':<28} | ", end="")
    ratios = []
    for n in header_names:
        norms = gradient_data[n]
        if norms[-1] > 1e-10:
            r = norms[0] / norms[-1]
        else:
            r = 0.0
        ratios.append(r)
    print(" | ".join(f"{r:>12.6f}" for r in ratios))

    # Steps to threshold
    threshold = 2.8
    print(f"{'Steps to loss < ' + str(threshold):<28} | ", end="")
    steps_strs = []
    for n in header_names:
        found = "never"
        for i, v in enumerate(results[n]['loss_history']):
            if v < threshold:
                found = str(i + 1)
                break
        steps_strs.append(found)
    print(" | ".join(f"{s:>12}" for s in steps_strs))

    print("=" * 72)

    # === INFERENCE ===
    print()
    for model_name, (params, step_fn) in trained_params.items():
        generate_names(params, step_fn, unique_chars, VOCAB_SIZE, model_name)

    # === ARCHITECTURAL INSIGHT ===
    print("=" * 70)
    print("ARCHITECTURAL INSIGHT: THE CELL STATE AS GRADIENT HIGHWAY")
    print("=" * 70)
    print()
    print("  Vanilla RNN gradient path through T timesteps:")
    print("    dL/dh_0 = dL/dh_T * prod_{t=1}^{T} diag(1-h_t^2) @ W_hh")
    print("    -> T matrix multiplications cause exponential decay/explosion")
    print()
    print("  GRU gradient path:")
    print("    dL/dh_0 includes a (1-z_t) identity shortcut at each step")
    print("    -> When z_t near 0, gradients pass through without W_hh multiplication")
    print("    -> But z_t and h_t share the same state, coupling gating and memory")
    print()
    print("  LSTM gradient path:")
    print("    dL/dc_0 = dL/dc_T * prod_{t=1}^{T} f_t")
    print("    -> Only element-wise multiplication by forget gate (no weight matrices)")
    print("    -> Cell state c and hidden state h are SEPARATE: the cell is a pure")
    print("       memory register, h is a gated view of it. This separation gives")
    print("       LSTM the cleanest gradient highway of the three architectures.")
    print()
    print("  The key insight: LSTM's cell state acts like a conveyor belt. Information")
    print("  written at timestep 0 can survive to timestep T if the forget gate stays")
    print("  open (f_t near 1). Gradients ride the same conveyor belt backward.")
    print("  This is why LSTM was the dominant sequence model for 20 years (1997-2017)")
    print("  until transformers replaced recurrence with attention entirely.")
