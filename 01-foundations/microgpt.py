"""
The autoregressive language model from first principles: GPT learns to predict the next
character in a sequence using nothing but matrix multiplication, attention, and gradient descent.
"""
# Reference: This implementation follows the GPT-2 architecture (Radford et al., 2019)
# with pedagogical simplifications: RMSNorm instead of LayerNorm, ReLU instead of GELU,
# no bias terms. Algorithmic flow inspired by Karpathy's microgpt.py but rewritten from
# scratch with comprehensive commenting for educational clarity.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture
N_EMBD = 16         # embedding dimension (d_model in Transformer papers)
N_HEAD = 4          # number of attention heads
N_LAYER = 1         # number of transformer blocks
BLOCK_SIZE = 16     # context window size (maximum sequence length)
HEAD_DIM = N_EMBD // N_HEAD  # dimension per attention head (16/4 = 4)

# Training parameters
LEARNING_RATE = 0.01  # Adam base learning rate
BETA1 = 0.85          # Adam first moment decay
BETA2 = 0.99          # Adam second moment decay
EPS_ADAM = 1e-8       # Adam epsilon (prevents division by zero)
NUM_STEPS = 1000      # total training steps

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~4,200 parameters total. Production GPTs have billions. The architecture
# is identical (attention is attention), but this toy scale lets us train on CPU in
# minutes rather than weeks on GPU clusters.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        # Each line is a document (name). Strip whitespace and filter empty lines.
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
        # This is the canonical activation -- microgpt uses ReLU, but tanh is part
        # of the standard interface for scripts that need it (micrornn, microlora).
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

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
        # ReLU is dead simple: max(0, x). Gradient is 1 for positive inputs, 0 otherwise.
        # Modern transformers often use GELU, but ReLU is easier to understand and
        # produces qualitatively similar results.
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


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    Standard deviation of 0.08 is chosen empirically for this tiny model --
    larger models typically use std = 1/sqrt(d_in) (Xavier/Glorot initialization)
    to keep activations from exploding or vanishing through deep layers. With
    only 1 layer, the initialization is less critical.
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters():
    """Initialize all model parameters: embeddings, attention, and MLP weights.

    Returns a dict keyed by human-readable names. This is the "state_dict" --
    the complete specification of the trained model. Save this dict and you've
    saved the model.
    """
    params = {}

    # Token and position embeddings
    # wte: [vocab_size, n_embd] - maps token IDs to vectors
    # wpe: [block_size, n_embd] - maps positions (0..15) to vectors
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    # Per-layer weights (we only have 1 layer, but the pattern generalizes)
    for layer_idx in range(N_LAYER):
        # Attention weights (Q, K, V projections and output projection)
        # All are square [n_embd, n_embd] matrices
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)

        # MLP weights (2-layer feedforward network with expansion factor 4)
        # fc1: [n_embd, 4*n_embd] - expand, fc2: [4*n_embd, n_embd] - contract
        # The 4x expansion is a GPT convention -- gives the MLP more capacity to
        # process the attention output without increasing the residual stream width.
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    # Language model head: projects final hidden states to vocabulary logits
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x (no bias).

    For weight matrix W with shape [n_out, n_in] and input vector x with
    shape [n_in], computes output y with shape [n_out] where each element
    y[i] = sum_j W[i,j] * x[j]. This is the fundamental operation of neural
    networks: every layer is just linear() followed by a nonlinearity.
    """
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Softmax is translation-invariant: softmax(x) = softmax(x - c) for any c.
    We subtract max(x) before exp() to prevent overflow. Without this, large
    logits (>700) cause exp() to return inf, breaking the computation.

    Math: softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    Stable: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square normalization: scale vector to unit RMS magnitude.

    RMSNorm is LayerNorm without mean centering or learned affine parameters.
    Fewer ops, fewer parameters, and empirically works just as well (used in
    LLaMA, Gemma, and other recent architectures).

    Math: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    The epsilon (1e-5) prevents division by zero when x is all zeros.
    """
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm for numerical stability in loss computation.

    Prevents log(0) which returns -inf and breaks gradient backpropagation.
    Clamping to 1e-10 gives log(1e-10) ≈ -23, which is finite and preserves
    gradient information. Without this, a single zero probability (which can
    happen early in training) kills the entire gradient.

    Critical: we must keep `prob` as a child node so gradients flow back through
    the computation graph. Creating a disconnected Value(clamped) would sever the
    gradient path and prevent the model from learning.
    """
    clamped = max(prob.data, 1e-10)
    # Build the log node manually with prob as its child, preserving the graph.
    # d(log(x))/dx = 1/x, evaluated at the clamped value for stability.
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === GPT FORWARD PASS ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict,
) -> list[Value]:
    """Single-token forward pass through the GPT model.

    This function processes ONE token at position `pos_id` and returns logits
    over the vocabulary. The keys and values lists accumulate the KV cache --
    a running history of all past tokens' key/value projections, which lets us
    implement causal attention without an explicit mask matrix.

    Args:
        token_id: Integer in [0, vocab_size-1] identifying the input token
        pos_id: Integer in [0, block_size-1] indicating position in sequence
        keys: KV cache for keys, shape [n_layer][seq_len][n_embd]
        values: KV cache for values, shape [n_layer][seq_len][n_embd]
        params: Model weight matrices

    Returns:
        Logits (unnormalized log-probabilities) over vocabulary, length vocab_size
    """
    # -- Embedding layer --
    # Look up learned vectors for this token and position, then add them.
    # This is the GPT input representation: tok_emb encodes "what" (the token),
    # pos_emb encodes "where" (position in sequence).
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    # Normalize embeddings before feeding to transformer blocks
    x = rmsnorm(x)

    # -- Transformer layers --
    for layer_idx in range(N_LAYER):
        # Residual connection pattern: x_new = x + f(x)
        # This "highway" lets gradients flow directly backward through the model
        # without passing through attention or MLP, preventing vanishing gradients.
        x_residual = x

        # Pre-norm: normalize before attention (modern architectures do this rather
        # than post-norm because it stabilizes training in deep models)
        x = rmsnorm(x)

        # -- Multi-head self-attention --
        # Project input to queries, keys, values
        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v = linear(x, params[f'layer{layer_idx}.attn_wv'])

        # Append k, v to cache for this layer. This builds the KV cache incrementally:
        # at position t, keys[layer_idx] contains [k_0, k_1, ..., k_t].
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        # Process each attention head independently, then concatenate outputs
        x_attn = []
        for head in range(N_HEAD):
            head_start = head * HEAD_DIM

            # Slice out this head's portion of the q/k/v vectors
            q_head = q[head_start : head_start + HEAD_DIM]
            k_head = [k_t[head_start : head_start + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[head_start : head_start + HEAD_DIM] for v_t in values[layer_idx]]

            # Compute attention scores: how much should we attend to each past token?
            # Formula: score(q, k_t) = (q · k_t) / sqrt(d_head)
            # The sqrt(d_head) scaling prevents scores from growing too large as
            # dimensionality increases (which would make softmax saturate).
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]

            # Convert scores to probabilities via softmax
            attn_weights = softmax(attn_logits)

            # Weighted sum of values: output[j] = sum_t attn_weights[t] * v[t][j]
            # This is the "attention" mechanism: we look at all past tokens (via their
            # value vectors) and weight each by its relevance (attention weight).
            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]

            x_attn.extend(head_output)

        # Signpost: Why KV caching provides causal masking without an explicit mask --
        # At position t, keys[layer_idx] only contains keys for positions 0..t, so the
        # attention scores loop (range(len(k_head))) naturally excludes future tokens.
        # This incremental construction is equivalent to applying a lower-triangular mask
        # in a batch setting, but more efficient for autoregressive generation.

        # Project concatenated head outputs back to residual dimension
        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])

        # First residual connection (around attention)
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        # -- MLP (feedforward network) --
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])  # expand
        x = [xi.relu() for xi in x]                         # nonlinearity
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])  # contract

        # Second residual connection (around MLP)
        x = [a + b for a, b in zip(x, x_residual)]

    # -- Output layer --
    # Project final hidden state to vocabulary logits
    logits = linear(x, params['lm_head'])
    return logits


# === TRAINING LOOP ===

if __name__ == "__main__":
    # -- Prepare vocabulary and data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    # Build vocabulary from unique characters in the corpus
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)  # Beginning-of-sequence token (appended to char set)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary size: {VOCAB_SIZE} (characters + BOS token)")

    # Initialize parameters after we know vocab size
    params = init_parameters()

    # Flatten all parameters into a single list for optimizer bookkeeping
    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,}\n")

    # -- Initialize Adam optimizer state --
    # m: first moment (momentum), v: second moment (variance)
    # These are per-parameter running averages that help Adam adapt the learning
    # rate individually for each weight based on gradient history.
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    # -- Training --
    print("Training...")
    for step in range(NUM_STEPS):
        # Cycle through the dataset (with shuffling, this is essentially SGD)
        doc = docs[step % len(docs)]

        # Tokenize: convert document to integer sequence with BOS markers
        # Format: [BOS, char_0, char_1, ..., char_n, BOS]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]

        # Truncate to block_size (context window limit)
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # Initialize KV cache for this sequence (fresh for each document)
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]

        # Compute loss across the sequence (cross-entropy at each position)
        losses = []
        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            # Forward pass
            logits = gpt_forward(input_token, pos, keys, values, params)

            # Convert logits to probabilities
            probs = softmax(logits)

            # Negative log-likelihood loss: -log(p(target))
            # This is the cross-entropy loss for classification. We want the model
            # to assign high probability to the actual next token.
            loss_t = -safe_log(probs[target_token])
            losses.append(loss_t)

        # Average loss over the sequence (makes loss scale-invariant to doc length)
        loss = (1.0 / seq_len) * sum(losses)

        # -- Backward pass --
        loss.backward()

        # -- Adam optimizer step --
        # Linear learning rate decay: lr_t = lr_0 * (1 - t/T)
        # This "learning rate warmdown" prevents overshooting as the loss landscape
        # sharpens near the optimum. Without decay, the fixed step size can cause
        # the optimizer to bounce around the minimum rather than converging.
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, param in enumerate(param_list):
            # Adam update rule:
            # m_t = β1*m_{t-1} + (1-β1)*g_t         (momentum)
            # v_t = β2*v_{t-1} + (1-β2)*g_t^2       (variance)
            # θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + ε)
            m[i] = BETA1 * m[i] + (1 - BETA1) * param.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * param.grad ** 2

            # Bias correction: m and v are biased toward zero in early steps because
            # they're initialized to 0. Dividing by (1 - β^t) corrects for this.
            # Without bias correction, early updates would be too small.
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))

            # Parameter update
            # epsilon (1e-8) prevents division by zero when v_hat is tiny
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

            # Zero gradient for next iteration
            param.grad = 0.0

        # Print progress
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    print(f"\nTraining complete. Final loss: {loss.data:.4f}\n")

    # === INFERENCE ===
    # Generate new samples from the trained model using temperature-scaled sampling.
    # Temperature controls randomness: lower = more deterministic, higher = more random.
    TEMPERATURE = 0.5
    NUM_SAMPLES = 20

    print(f"Generating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        # Fresh KV cache for each sample
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]

        # Start with BOS token
        token_id = BOS
        generated = []

        for pos in range(BLOCK_SIZE):
            # Forward pass
            logits = gpt_forward(token_id, pos, keys, values, params)

            # Temperature scaling: divide logits by temperature before softmax
            # This sharpens (T < 1) or flattens (T > 1) the probability distribution.
            # Lower temperature makes the model more confident (picks high-prob tokens),
            # higher temperature makes it more exploratory (samples more uniformly).
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            # Sample next token from the probability distribution
            # random.choices uses the probabilities as sampling weights
            token_id = random.choices(
                range(VOCAB_SIZE),
                weights=[p.data for p in probs]
            )[0]

            # Stop if we hit BOS (end-of-sequence marker)
            if token_id == BOS:
                break

            generated.append(unique_chars[token_id])

        # Print the generated name
        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")
