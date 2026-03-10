"""
Attention is not just helpful -- it is the mechanism that lets a model look back across an
entire sequence instead of compressing everything into a fixed-size hidden state.
"""
# Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and
# Translate" (2015) introduced attention for seq2seq. This script trains two identical RNN
# architectures on character-level name prediction -- one with additive attention over hidden
# states, one without -- to demonstrate where attention provides measurable gains.

# === TRADEOFFS ===
# + Attention lets the model directly access any previous timestep's representation
# + Longer sequences benefit disproportionately: attention bypasses the information bottleneck
# + Gradient flow improves: loss connects directly to distant hidden states via attention
# - O(T^2) cost per sequence position (T = sequence length) vs O(T) for plain RNN
# - Attention adds parameters (query/key projections) and memory (storing all hidden states)
# - For very short sequences (<5 tokens), the overhead outweighs the benefit
# WHEN TO USE: Sequence tasks where distant tokens influence predictions (translation,
#   summarization, long names). Essentially all modern sequence models use attention.
# WHEN NOT TO: Extremely resource-constrained streaming where O(T^2) is prohibitive
#   and sequences are short enough that a hidden state suffices.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_HIDDEN = 12        # hidden state dimension -- very small to keep attention tractable
                     # with scalar autograd (attention is O(T^2 * N_HIDDEN) per step)
SEQ_LEN = 12         # maximum sequence length -- shorter to limit attention cost
LEARNING_RATE = 0.1  # SGD learning rate -- slightly higher for smaller model
NUM_STEPS = 600      # training steps per model (2 models must finish in <10 min total)
TRAIN_SIZE = 100     # small training subset for speed

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~1,500 parameters per model. The attention model adds ~600 parameters
# for query/key projections. Production attention models have billions of parameters,
# but the mechanism is identical.


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
    its local derivative (dout/dinput), then backward() replays the computation
    graph in reverse topological order, accumulating gradients.
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
        # sigmoid(x) = 1 / (1 + exp(-x))
        # Used for gating in GRU and attention energy normalization
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
        """Compute gradients via reverse-mode automatic differentiation."""
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
    """Initialize a weight matrix with Gaussian noise.

    Xavier-like scaling: std ~ 1/sqrt(fan_in) keeps activation variance stable.
    We use a fixed 0.08 for this toy scale.
    """
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


# === MODEL: GRU WITHOUT ATTENTION (BASELINE) ===

def init_gru_params(vocab_size: int) -> dict:
    """GRU parameters: update gate, reset gate, candidate, and output projection.

    This is the baseline model -- it compresses the entire sequence history into
    a single hidden vector h_t. The final h_t must encode everything needed to
    predict the next character, regardless of sequence length.
    """
    params = {}
    params['W_xz'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hz'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['W_xr'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hr'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['W_xh'] = make_matrix(N_HIDDEN, vocab_size)
    params['W_hh'] = make_matrix(N_HIDDEN, N_HIDDEN)
    params['W_hy'] = make_matrix(vocab_size, N_HIDDEN)
    params['b_y'] = [Value(0.0) for _ in range(vocab_size)]
    return params


def gru_step(
    x: list[Value], h_prev: list[Value], params: dict
) -> list[Value]:
    """Single-step GRU: returns new hidden state.

    Math:
        z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1})
        r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1})
        h_candidate = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate
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

    # Interpolation: gradient highway when z_t near 0
    h = [(1 - z_i) * h_prev_i + z_i * h_cand_i
         for z_i, h_prev_i, h_cand_i in zip(z, h_prev, h_candidate)]
    return h


def forward_no_attention(
    tokens: list[int], params: dict, vocab_size: int
) -> tuple[list[list[Value]], list[list[Value]]]:
    """Run GRU over a token sequence, predict next token at each step using only h_t.

    Returns (all_logits, all_hidden_states) where each logits[t] is the prediction
    for tokens[t+1] based solely on the hidden state at time t.
    """
    h = [Value(0.0) for _ in range(N_HIDDEN)]
    all_logits = []
    all_hidden = []

    for pos in range(len(tokens) - 1):
        x_onehot = [Value(1.0 if i == tokens[pos] else 0.0) for i in range(vocab_size)]
        h = gru_step(x_onehot, h, params)
        all_hidden.append(h)
        logits = linear(h, params['W_hy'], params['b_y'])
        all_logits.append(logits)

    return all_logits, all_hidden


# === MODEL: GRU WITH ADDITIVE ATTENTION ===

def init_attention_gru_params(vocab_size: int) -> dict:
    """GRU + attention parameters.

    The attention mechanism adds:
    - W_query: projects current hidden state to query space
    - W_key: projects each past hidden state to key space
    - v_attn: weight vector that scores query-key alignment

    This is Bahdanau-style additive attention:
        score(h_t, h_j) = v^T * tanh(W_query @ h_t + W_key @ h_j)
        alpha_j = softmax(scores)_j
        context = sum_j alpha_j * h_j

    The context vector is concatenated with h_t for the output projection,
    giving the model direct access to any past hidden state.
    """
    params = init_gru_params(vocab_size)

    # Attention-specific parameters
    # Use a smaller attention dimension (half of hidden) to keep scalar autograd
    # tractable -- the full attention computation is O(T * attn_dim) per timestep,
    # and we do this T times per sequence, so total is O(T^2 * attn_dim).
    attn_dim = N_HIDDEN // 2
    params['attn_dim'] = attn_dim  # store for use in attend()
    params['W_query'] = make_matrix(attn_dim, N_HIDDEN)
    params['W_key'] = make_matrix(attn_dim, N_HIDDEN)
    params['v_attn'] = [Value(random.gauss(0, 0.08)) for _ in range(attn_dim)]

    # Output projection takes concatenated [h_t; context] instead of just h_t
    # So it maps from 2*N_HIDDEN to vocab_size
    params['W_hy_attn'] = make_matrix(vocab_size, 2 * N_HIDDEN)
    params['b_y_attn'] = [Value(0.0) for _ in range(vocab_size)]

    return params


def attend(
    h_current: list[Value], hidden_states: list[list[Value]], params: dict
) -> list[Value]:
    """Compute attention context vector over all past hidden states.

    Bahdanau additive attention:
        e_j = v^T * tanh(W_query @ h_current + W_key @ h_j)
        alpha = softmax(e)
        context = sum_j alpha_j * h_j

    Why additive (not dot-product)? At this scale, additive attention is more
    expressive per parameter because it applies a nonlinearity (tanh) to the
    query-key interaction. Dot-product attention is faster at scale due to
    matrix multiplication hardware, but the quality difference is minimal.
    """
    if not hidden_states:
        # No past states to attend to -- return zero context
        return [Value(0.0) for _ in range(N_HIDDEN)]

    query = linear(h_current, params['W_query'])

    # Compute alignment scores for each past hidden state
    scores = []
    for h_j in hidden_states:
        key = linear(h_j, params['W_key'])
        # Additive combination followed by tanh nonlinearity
        combined = [(q + k).tanh() for q, k in zip(query, key)]
        # Score = v^T @ combined (scalar)
        score = sum(v * c for v, c in zip(params['v_attn'], combined))
        scores.append(score)

    # Softmax over alignment scores to get attention weights
    alpha = softmax(scores)

    # Weighted sum of past hidden states
    context = [Value(0.0) for _ in range(N_HIDDEN)]
    for a_j, h_j in zip(alpha, hidden_states):
        context = [c + a_j * h_i for c, h_i in zip(context, h_j)]

    return context


def forward_with_attention(
    tokens: list[int], params: dict, vocab_size: int
) -> tuple[list[list[Value]], list[list[Value]]]:
    """Run GRU with attention over a token sequence.

    At each timestep, the model:
    1. Updates the GRU hidden state (same as baseline)
    2. Attends over ALL previous hidden states to form a context vector
    3. Concatenates [h_t; context] and projects to logits

    The attention mechanism gives the model a direct path to any past timestep,
    bypassing the information bottleneck of the fixed-size hidden state.
    """
    h = [Value(0.0) for _ in range(N_HIDDEN)]
    all_logits = []
    all_hidden = []

    for pos in range(len(tokens) - 1):
        x_onehot = [Value(1.0 if i == tokens[pos] else 0.0) for i in range(vocab_size)]
        h = gru_step(x_onehot, h, params)
        all_hidden.append(h)

        # Attend over all previous hidden states (not including current)
        # This mirrors how decoder attention works in seq2seq: the current
        # state queries past encoder/decoder states for relevant context
        context = attend(h, all_hidden[:-1], params)

        # Concatenate hidden state with context vector
        h_context = h + context  # list concatenation: [h_t; context]

        # Project concatenated vector to logits
        logits = linear(h_context, params['W_hy_attn'], params['b_y_attn'])
        all_logits.append(logits)

    return all_logits, all_hidden


# === TRAINING ===

def collect_params(params: dict) -> list[Value]:
    """Flatten all parameters into a single list for optimizer updates."""
    param_list = []
    for val in params.values():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], Value):
                param_list.extend(val)
            elif isinstance(val[0], list):
                for row in val:
                    param_list.extend(row)
    return param_list


def train_model(
    docs: list[str], unique_chars: list[str], params: dict,
    forward_fn, model_name: str, vocab_size: int
) -> tuple[list[float], dict[str, list[float]]]:
    """Train a model and track loss by sequence length.

    Returns:
        (loss_history, per_length_losses) where per_length_losses maps
        length bucket strings to lists of losses for sequences of that length.
    """
    BOS = len(unique_chars)
    param_list = collect_params(params)

    print(f"Training {model_name}... ({len(param_list):,} parameters)")

    loss_history: list[float] = []
    # Track losses bucketed by sequence length to show where attention helps most
    per_length_losses: dict[str, list[float]] = {"short(2-4)": [], "medium(5-7)": [], "long(8+)": []}

    start = time.time()

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(SEQ_LEN, len(tokens) - 1)
        tokens = tokens[:seq_len + 1]

        all_logits, _ = forward_fn(tokens, params, vocab_size)

        # Compute average cross-entropy loss over the sequence
        total_loss = Value(0.0)
        for t, logits in enumerate(all_logits):
            probs = softmax(logits)
            target = tokens[t + 1]
            total_loss = total_loss + (-safe_log(probs[target]))

        loss = total_loss / len(all_logits)
        loss.backward()

        # SGD update
        for param in param_list:
            param.data -= LEARNING_RATE * param.grad
            param.grad = 0.0

        loss_history.append(loss.data)

        # Bucket by original name length (before BOS tokens)
        name_len = len(doc)
        if name_len <= 4:
            per_length_losses["short(2-4)"].append(loss.data)
        elif name_len <= 7:
            per_length_losses["medium(5-7)"].append(loss.data)
        else:
            per_length_losses["long(8+)"].append(loss.data)

        if (step + 1) % 150 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    elapsed = time.time() - start
    final_loss = loss_history[-1]
    print(f"  {model_name} done. Final loss: {final_loss:.4f} ({elapsed:.1f}s)\n")

    return loss_history, per_length_losses


# === INFERENCE ===

def generate_names(
    params: dict, forward_fn, unique_chars: list[str],
    vocab_size: int, model_name: str, num_samples: int = 8
) -> list[str]:
    """Generate names via autoregressive sampling."""
    BOS = len(unique_chars)
    print(f"Generating {num_samples} names from {model_name}:")

    samples = []
    for _ in range(num_samples):
        h = [Value(0.0) for _ in range(N_HIDDEN)]
        token_id = BOS
        generated = []
        all_hidden: list[list[Value]] = []

        for pos in range(SEQ_LEN):
            x_onehot = [Value(1.0 if i == token_id else 0.0) for i in range(vocab_size)]
            h = gru_step(x_onehot, h, params)
            all_hidden.append(h)

            # Use attention for the attention model, plain projection otherwise
            if 'W_hy_attn' in params:
                context = attend(h, all_hidden[:-1], params)
                h_context = h + context
                logits = linear(h_context, params['W_hy_attn'], params['b_y_attn'])
            else:
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
    print("ATTENTION VS NO ATTENTION: Character-Level Name Prediction")
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

    # -- Train baseline (GRU without attention) --
    random.seed(42)
    baseline_params = init_gru_params(VOCAB_SIZE)
    baseline_losses, baseline_per_len = train_model(
        docs, unique_chars, baseline_params,
        forward_no_attention, "GRU (no attention)", VOCAB_SIZE
    )

    # -- Train attention model (GRU with attention) --
    random.seed(42)
    attn_params = init_attention_gru_params(VOCAB_SIZE)
    attn_losses, attn_per_len = train_model(
        docs, unique_chars, attn_params,
        forward_with_attention, "GRU + Attention", VOCAB_SIZE
    )

    # === LOSS CURVE COMPARISON ===
    print("=" * 70)
    print("LOSS CURVES (every 150 steps)")
    print("=" * 70)
    print(f"{'Step':>6} | {'No Attention':>14} | {'With Attention':>14} | {'Delta':>10}")
    print("-" * 52)

    for step in range(0, NUM_STEPS, 75):
        # Average over a small window to smooth noise
        window = 20
        lo = max(0, step - window)
        hi = min(NUM_STEPS, step + window)
        bl = sum(baseline_losses[lo:hi]) / max(1, hi - lo)
        at = sum(attn_losses[lo:hi]) / max(1, hi - lo)
        delta = bl - at
        marker = "<-- attn wins" if delta > 0.05 else ""
        print(f"{step + 1:>6} | {bl:>14.4f} | {at:>14.4f} | {delta:>+10.4f}  {marker}")

    print()

    # === PER-LENGTH COMPARISON ===
    # This is where attention's advantage becomes clearest: longer sequences
    # require the model to remember earlier characters, which the fixed hidden
    # state struggles with but attention handles directly.
    print("=" * 70)
    print("PER-SEQUENCE-LENGTH LOSS (averaged over last 500 steps)")
    print("=" * 70)
    print(f"{'Length Bucket':<16} | {'No Attention':>14} | {'With Attention':>14} | {'Delta':>10}")
    print("-" * 62)

    for bucket in ["short(2-4)", "medium(5-7)", "long(8+)"]:
        bl_vals = baseline_per_len[bucket]
        at_vals = attn_per_len[bucket]

        # Use only losses from the last portion of training (after convergence)
        cutoff_bl = max(0, len(bl_vals) - 200)
        cutoff_at = max(0, len(at_vals) - 200)

        bl_avg = sum(bl_vals[cutoff_bl:]) / max(1, len(bl_vals) - cutoff_bl) if bl_vals else float('nan')
        at_avg = sum(at_vals[cutoff_at:]) / max(1, len(at_vals) - cutoff_at) if at_vals else float('nan')

        delta = bl_avg - at_avg
        print(f"{bucket:<16} | {bl_avg:>14.4f} | {at_avg:>14.4f} | {delta:>+10.4f}")

    print()
    print("Key insight: attention's advantage grows with sequence length because")
    print("the no-attention model must compress ALL history into a fixed-size vector,")
    print("while attention can directly reference any past timestep.")

    # === FINAL COMPARISON TABLE ===
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    bl_final = sum(baseline_losses[-100:]) / 100
    at_final = sum(attn_losses[-100:]) / 100
    bl_best = min(baseline_losses)
    at_best = min(attn_losses)

    bl_param_count = len(collect_params(baseline_params))
    at_param_count = len(collect_params(attn_params))

    print(f"{'Metric':<30} | {'No Attention':>14} | {'With Attention':>14}")
    print("-" * 64)
    print(f"{'Parameters':<30} | {bl_param_count:>14,} | {at_param_count:>14,}")
    print(f"{'Final Loss (avg last 100)':<30} | {bl_final:>14.4f} | {at_final:>14.4f}")
    print(f"{'Best Loss':<30} | {bl_best:>14.4f} | {at_best:>14.4f}")

    # Steps to reach a threshold
    threshold = 2.8
    bl_steps = "never"
    at_steps = "never"
    for i, v in enumerate(baseline_losses):
        if v < threshold:
            bl_steps = str(i + 1)
            break
    for i, v in enumerate(attn_losses):
        if v < threshold:
            at_steps = str(i + 1)
            break

    print(f"{'Steps to loss < ' + str(threshold):<30} | {bl_steps:>14} | {at_steps:>14}")
    print("=" * 70)

    # === INFERENCE ===
    print()
    generate_names(
        baseline_params, forward_no_attention, unique_chars,
        VOCAB_SIZE, "GRU (no attention)"
    )

    generate_names(
        attn_params, forward_with_attention, unique_chars,
        VOCAB_SIZE, "GRU + Attention"
    )

    # === WHY ATTENTION WORKS ===
    print("=" * 70)
    print("WHY ATTENTION WORKS")
    print("=" * 70)
    print("  Without attention: The hidden state h_t is a fixed-size vector that must")
    print("  encode the ENTIRE sequence history. For long names, early characters get")
    print("  progressively overwritten by later ones (the information bottleneck).")
    print()
    print("  With attention: The model stores all past hidden states and computes a")
    print("  weighted combination at each step. To predict the ending of 'christopher',")
    print("  the model can directly attend to 'c-h-r-i-s' rather than hoping h_t")
    print("  still encodes those characters 10 steps later.")
    print()
    print("  The cost: O(T^2) attention computation vs O(T) for plain RNN.")
    print("  The benefit: direct information access regardless of distance.")
    print("  This tradeoff overwhelmingly favors attention for sequences > ~5 tokens,")
    print("  which is why transformers (all-attention, no recurrence) won.")
