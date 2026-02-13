"""
The full RLHF loop: pretrain a language model, train a reward model on human preferences,
then optimize the policy with Proximal Policy Optimization -- all in one file, from scratch.
"""
# Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
# https://arxiv.org/abs/1707.06347
# Also: Ouyang et al., "Training language models to follow instructions with human
# feedback" (InstructGPT, 2022). https://arxiv.org/abs/2203.02155
# Architecture reuses the microgpt pattern (Radford et al., 2019) with a smaller model
# (n_embd=8) to accommodate the three-model RLHF pipeline within runtime constraints.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Policy model architecture -- smaller than microgpt to fit the three-model pipeline
# within the 7-minute runtime budget. The architecture is identical (attention, MLP,
# residual connections); only the dimensions are reduced.
N_EMBD = 8          # embedding dimension (vs. 16 in microgpt)
N_HEAD = 2          # attention heads (vs. 4)
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 12     # context window (vs. 16) -- shorter because names are typically 3-8 chars
HEAD_DIM = N_EMBD // N_HEAD  # 4 dimensions per head

# Pretraining parameters
PRETRAIN_LR = 0.01
PRETRAIN_STEPS = 500
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# Reward model parameters
REWARD_HIDDEN = 32       # hidden layer width
REWARD_LR = 0.01         # SGD learning rate
REWARD_STEPS = 400        # training iterations
REWARD_MARGIN = 1.0       # ranking loss margin -- forces clear separation between preferred/rejected

# PPO parameters
PPO_CLIP_EPS = 0.2       # clipping epsilon -- limits how far the policy ratio can deviate
KL_COEFF = 0.5           # KL penalty coefficient -- higher than typical (0.01-0.1) because our
                         # tiny model and synthetic reward are prone to mode collapse
PPO_STEPS = 100          # number of PPO optimization steps (fewer to prevent over-optimization)
BATCH_SIZE = 4           # completions generated per PPO step
MAX_GEN_LEN = 8          # max generation length
MIN_GEN_LEN = 2          # minimum generation length -- penalize degenerate empty outputs
PPO_LR = 0.0005          # lower than pretraining to prevent catastrophic forgetting
VALUE_LR = 0.01          # value function learning rate

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# IMPLEMENTATION NOTE: The reward model and value function use plain floats (not
# autograd Value objects) for runtime tractability. The policy model uses scalar
# autograd because PPO gradients must flow through the policy's generation process.
# Production RLHF (InstructGPT, ChatGPT) vectorizes all three models on GPUs;
# we split the approach to stay within pure-Python runtime constraints while
# preserving the complete PPO algorithm.


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

    Every forward operation records its local derivative (dout/dinput). backward()
    replays the computation graph in reverse topological order, accumulating gradients
    via the chain rule: dLoss/dx = sum over paths (product of local gradients along path).
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * (other ** -1)
    def __rtruediv__(self, other: float) -> Value: return other * (self ** -1)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def clip(self, low: float, high: float) -> Value:
        """Clip value to [low, high] range. Gradient passes through when in range, zero
        when clipped. Required for PPO ratio clipping: the clipped surrogate objective
        prevents catastrophically large policy updates by clamping the probability ratio.
        When the ratio is within [1-eps, 1+eps], gradients flow normally. When clipped,
        the gradient is zeroed -- this is the "proximal" constraint in PPO."""
        clamped = max(low, min(high, self.data))
        grad = 1.0 if low < self.data < high else 0.0
        return Value(clamped, (self,), (grad,))

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort of the computation graph."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with the following addition:
# - clip(): Required for PPO ratio clipping (clamps value, passes gradient when in range)
# See docs/autograd-interface.md for the full canonical interface.


# === PARAMETER INITIALIZATION (POLICY MODEL -- VALUE CLASS) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize weight matrix ~ N(0, std). Small std prevents activation explosion
    in the tiny model where Xavier scaling (1/sqrt(d_in)) would give std=0.35."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_policy_params(vocab_size: int) -> dict[str, list[list[Value]]]:
    """Initialize all policy model parameters: embeddings, attention, MLP, LM head."""
    params: dict[str, list[list[Value]]] = {}

    params['wte'] = make_matrix(vocab_size, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        # MLP: expand 4x then contract (standard GPT feedforward)
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    params['lm_head'] = make_matrix(vocab_size, N_EMBD)

    return params


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Collect all Value objects from a parameter dict into a flat list."""
    return [p for matrix in params.values() for row in matrix for p in row]


# === CORE OPERATIONS (POLICY MODEL -- VALUE CLASS) ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x. Shape: [n_out, n_in] @ [n_in] -> [n_out]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Stable softmax: subtract max before exp to prevent overflow.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS normalization: x / sqrt(mean(x^2) + eps). Simpler than LayerNorm."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped log for numerical stability. Prevents log(0) = -inf. Builds the node
    manually with prob as its child so gradients flow through the graph."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === POLICY MODEL FORWARD PASS ===

def policy_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """Single-token forward pass through the policy GPT. Returns logits over vocabulary.

    Structurally identical to microgpt's forward pass but with smaller dimensions
    (n_embd=8, n_head=2). The KV cache builds incrementally so causal masking is
    implicit: at position t, only keys/values for positions 0..t exist.
    """
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v = linear(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attn: list[Value] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM

            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

            # Scaled dot-product attention: score = (q . k) / sqrt(d_head)
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            attn_weights = softmax(attn_logits)

            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_output)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        # MLP: expand, ReLU, contract
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === REWARD MODEL (PLAIN FLOATS -- NO AUTOGRAD) ===
# The reward model is a simple MLP that scores a character sequence. It uses plain
# float arrays because it is trained independently before the PPO loop (pairwise
# ranking loss with manual SGD), so autograd overhead is unnecessary. Production
# reward models are large transformers; this MLP captures the same preference-learning
# mechanism at toy scale.

def init_reward_model(d_in: int) -> dict[str, list[list[float]]]:
    """Initialize reward model: 2-layer MLP mapping feature vector to scalar."""
    params: dict[str, list[list[float]]] = {}
    # Hidden layer: d_in -> REWARD_HIDDEN
    params['w1'] = [[random.gauss(0, 0.1) for _ in range(d_in)] for _ in range(REWARD_HIDDEN)]
    params['b1'] = [[0.0] for _ in range(REWARD_HIDDEN)]
    # Output layer: REWARD_HIDDEN -> 1
    params['w2'] = [[random.gauss(0, 0.1) for _ in range(REWARD_HIDDEN)]]
    params['b2'] = [[0.0]]
    return params


def reward_forward(features: list[float], params: dict[str, list[list[float]]]) -> float:
    """Forward pass through the reward MLP. Returns a scalar reward score.

    Architecture: input -> linear -> ReLU -> linear -> scalar
    The input is a feature vector encoding character frequencies and sequence length.
    This loses ordering information but captures the preference signal (name length,
    character distribution) with minimal parameters.
    """
    # Hidden layer with ReLU
    hidden = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features[j] for j in range(len(features))) + params['b1'][i][0]
        hidden.append(max(0.0, h))  # ReLU

    # Output layer: scalar
    score = sum(params['w2'][0][j] * hidden[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]
    return score


def sequence_to_features(token_ids: list[int], vocab_size: int) -> list[float]:
    """Convert token ID sequence to a feature vector for the reward/value models.

    Features: normalized character counts (vocab_size dims) + normalized length (1 dim).
    The length feature is critical: our preference signal is based on name length (4-7 chars
    preferred), so the reward model needs explicit access to this. We normalize counts by
    a fixed constant (not sequence length) so that longer sequences have larger feature
    magnitudes, preserving the length signal in the character features too.
    """
    features = [0.0] * (vocab_size + 1)  # +1 for length feature
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            features[tid] += 1.0
    # Scale character counts to roughly [0, 1] per-character range
    for i in range(vocab_size):
        features[i] /= 10.0
    # Length feature: normalized to [0, 1] (max practical name ~10 chars)
    features[vocab_size] = len(token_ids) / 10.0
    return features


def reward_backward(
    features_chosen: list[float],
    features_rejected: list[float],
    params: dict[str, list[list[float]]],
    lr: float,
) -> float:
    """One step of pairwise ranking loss with manual SGD.

    Math: loss = max(0, margin - (reward_chosen - reward_rejected))
    This hinge loss pushes the reward model to score chosen sequences higher than
    rejected sequences by at least `margin`. It is the same objective used to train
    reward models in InstructGPT, simplified from the full Bradley-Terry framework.

    Gradients are computed manually via the chain rule because the reward model
    uses plain floats (not autograd Value objects).
    """
    d_in = len(features_chosen)

    # Forward pass for chosen
    hidden_c = []
    pre_relu_c = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features_chosen[j] for j in range(d_in)) + params['b1'][i][0]
        pre_relu_c.append(h)
        hidden_c.append(max(0.0, h))
    score_c = sum(params['w2'][0][j] * hidden_c[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]

    # Forward pass for rejected
    hidden_r = []
    pre_relu_r = []
    for i in range(REWARD_HIDDEN):
        h = sum(params['w1'][i][j] * features_rejected[j] for j in range(d_in)) + params['b1'][i][0]
        pre_relu_r.append(h)
        hidden_r.append(max(0.0, h))
    score_r = sum(params['w2'][0][j] * hidden_r[j] for j in range(REWARD_HIDDEN)) + params['b2'][0][0]

    # Hinge loss: max(0, margin - (score_chosen - score_rejected))
    diff = score_c - score_r
    loss = max(0.0, REWARD_MARGIN - diff)

    if loss <= 0.0:
        # Margin satisfied -- no gradient, no update needed
        return loss

    # Backward pass: d_loss/d_diff = -1 (since loss = margin - diff when active)
    # d_diff/d_score_c = 1, d_diff/d_score_r = -1
    d_score_c = -1.0  # d_loss/d_score_c
    d_score_r = 1.0   # d_loss/d_score_r

    # Gradient through output layer
    d_hidden_c = [params['w2'][0][j] * d_score_c for j in range(REWARD_HIDDEN)]
    d_hidden_r = [params['w2'][0][j] * d_score_r for j in range(REWARD_HIDDEN)]

    # Update output layer weights
    for j in range(REWARD_HIDDEN):
        params['w2'][0][j] -= lr * (hidden_c[j] * d_score_c + hidden_r[j] * d_score_r)
    params['b2'][0][0] -= lr * (d_score_c + d_score_r)

    # Gradient through ReLU and hidden layer
    for i in range(REWARD_HIDDEN):
        relu_grad_c = 1.0 if pre_relu_c[i] > 0 else 0.0
        relu_grad_r = 1.0 if pre_relu_r[i] > 0 else 0.0
        d_pre_c = d_hidden_c[i] * relu_grad_c
        d_pre_r = d_hidden_r[i] * relu_grad_r

        for j in range(d_in):
            params['w1'][i][j] -= lr * (d_pre_c * features_chosen[j] + d_pre_r * features_rejected[j])
        params['b1'][i][0] -= lr * (d_pre_c + d_pre_r)

    return loss


def score_completion(token_ids: list[int], vocab_size: int,
                     reward_params: dict[str, list[list[float]]]) -> float:
    """Score a completion using the reward model, with normalization.

    The raw reward model output is centered around zero with unknown scale. We apply
    a simple normalization: shift so that the mean reward on training-distribution
    sequences is roughly zero, and scale to unit variance. This is precomputed once
    after reward model training (see calibration step in Phase 2).

    Signpost: Production RLHF systems maintain running statistics of reward model
    outputs and normalize rewards online. Our static calibration captures the same
    mechanism with less runtime overhead.
    """
    features = sequence_to_features(token_ids, vocab_size)
    return reward_forward(features, reward_params)


# === VALUE FUNCTION (PLAIN FLOATS -- NO AUTOGRAD) ===
# The value function predicts the expected reward for a given sequence. It serves as a
# baseline in the advantage computation: advantage = reward - value_baseline.
# Without this baseline, the policy gradient has high variance (every reward signal
# is treated as equally informative), making PPO optimization unstable.
# Signpost: Production RLHF uses GAE (Generalized Advantage Estimation) with a deeper
# value network. Our single linear layer captures the core variance-reduction mechanism.

def init_value_function(d_in: int) -> dict[str, list[float]]:
    """Initialize value function: linear layer mapping features to scalar."""
    params: dict[str, list[float]] = {}
    params['w'] = [random.gauss(0, 0.01) for _ in range(d_in)]
    params['b'] = [0.0]
    return params


def value_forward(features: list[float], params: dict[str, list[float]]) -> float:
    """Value function forward pass: simple dot product + bias."""
    return sum(params['w'][j] * features[j] for j in range(len(features))) + params['b'][0]


def value_update(
    features: list[float],
    target: float,
    params: dict[str, list[float]],
    lr: float,
) -> float:
    """Update value function with MSE loss: (predicted - target)^2.

    Manual SGD: d_loss/d_w[j] = 2 * (pred - target) * features[j]
    """
    pred = value_forward(features, params)
    error = pred - target
    mse = error ** 2

    # SGD update with gradient clipping for stability
    grad_scale = min(1.0, 1.0 / (abs(error) + 1e-8))
    for j in range(len(features)):
        params['w'][j] -= lr * 2.0 * error * features[j] * grad_scale
    params['b'][0] -= lr * 2.0 * error * grad_scale

    return mse


# === GENERATION AND LOG-PROBABILITY UTILITIES ===

def generate_completion(
    params: dict[str, list[list[Value]]],
    bos: int,
    vocab_size: int,
    max_len: int,
    temperature: float = 0.8,
) -> list[int]:
    """Generate a token sequence from the policy model using temperature sampling.

    Returns the generated token IDs (excluding the BOS prefix). This function does NOT
    build an autograd graph -- it uses .data for sampling, which is correct because
    we only need the generated tokens, not gradients through the generation process.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    token_id = bos
    generated: list[int] = []

    for pos in range(max_len):
        logits = policy_forward(token_id, pos, keys, vals, params)
        # Temperature sampling: lower T = more greedy, higher T = more exploration
        scaled = [logit / temperature for logit in logits]
        probs = softmax(scaled)
        token_id = random.choices(
            range(vocab_size), weights=[p.data for p in probs]
        )[0]
        if token_id == bos:
            break
        generated.append(token_id)

    return generated


def compute_log_probs_detached(
    token_ids: list[int],
    bos: int,
    params: dict[str, list[list[Value]]],
) -> float:
    """Compute total log-probability of a sequence under the policy (no autograd graph).

    Used to store "old" log-probs before each PPO update. These become the denominator
    in the importance sampling ratio: ratio = pi_new(a|s) / pi_old(a|s).
    Using .data avoids building an autograd graph, since we only need the scalar values.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    full_seq = [bos] + token_ids
    total_logp = 0.0

    for pos in range(len(token_ids)):
        logits = policy_forward(full_seq[pos], pos, keys, vals, params)
        # Stable log-softmax: log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x_j - max(x))))
        logit_data = [l.data for l in logits]
        max_l = max(logit_data)
        exp_sum = sum(math.exp(l - max_l) for l in logit_data)
        log_prob = logit_data[full_seq[pos + 1]] - max_l - math.log(exp_sum)
        total_logp += log_prob

    return total_logp


def compute_log_probs_autograd(
    token_ids: list[int],
    bos: int,
    params: dict[str, list[list[Value]]],
) -> Value:
    """Compute total log-probability of a sequence WITH autograd graph.

    This is the expensive version used inside the PPO update. The autograd graph must
    be built because PPO needs gradients of the surrogate objective with respect to
    the policy parameters. The ratio exp(log_pi_new - log_pi_old) flows through this
    computation.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    full_seq = [bos] + token_ids
    total_logp: Value = Value(0.0)

    for pos in range(len(token_ids)):
        logits = policy_forward(full_seq[pos], pos, keys, vals, params)
        probs = softmax(logits)
        total_logp = total_logp + safe_log(probs[full_seq[pos + 1]])

    return total_logp


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    # -- Prepare vocabulary and data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # === Phase 1: Pretraining Policy Model ===
    print("\n=== Phase 1: Pretraining Policy Model ===")
    policy_params = init_policy_params(VOCAB_SIZE)
    policy_param_list = flatten_params(policy_params)
    print(f"Policy parameters: {len(policy_param_list):,} (Value class autograd)")

    # Adam optimizer state for policy pretraining
    m_pre = [0.0] * len(policy_param_list)
    v_pre = [0.0] * len(policy_param_list)

    for step in range(PRETRAIN_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses: list[Value] = []
        for pos in range(seq_len):
            logits = policy_forward(tokens[pos], pos, keys, vals, policy_params)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # Adam with linear LR decay
        lr_t = PRETRAIN_LR * (1 - step / PRETRAIN_STEPS)
        for i, p in enumerate(policy_param_list):
            m_pre[i] = BETA1 * m_pre[i] + (1 - BETA1) * p.grad
            v_pre[i] = BETA2 * v_pre[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_pre[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_pre[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{PRETRAIN_STEPS} | loss: {loss.data:.4f}")

    print(f"Pretraining complete. Final loss: {loss.data:.4f}")

    # Store reference policy parameters for KL penalty. The reference model is the policy
    # at the end of pretraining -- a frozen snapshot. In production RLHF, this would be
    # a separate copy of the model. Here we store the parameter values as plain floats
    # and temporarily swap them in when computing reference log-probs.
    ref_param_data: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        ref_param_data[key] = [[v.data for v in row] for row in matrix]

    def compute_ref_log_probs(token_ids: list[int]) -> float:
        """Compute log-probs under the frozen reference policy.

        Temporarily swaps policy parameters to reference values, computes log-probs
        (detached -- no autograd), then restores current parameters. This avoids
        storing a second full model. Production RLHF keeps both models in memory;
        we trade compute for memory since scalar autograd objects are expensive.
        """
        # Save current params, load reference params
        current_data: dict[str, list[list[float]]] = {}
        for key, matrix in policy_params.items():
            current_data[key] = [[v.data for v in row] for row in matrix]
            for r, row in enumerate(matrix):
                for c, v in enumerate(row):
                    v.data = ref_param_data[key][r][c]

        logp = compute_log_probs_detached(token_ids, BOS, policy_params)

        # Restore current params
        for key, matrix in policy_params.items():
            for r, row in enumerate(matrix):
                for c, v in enumerate(row):
                    v.data = current_data[key][r][c]

        return logp

    # === Phase 2: Training Reward Model ===
    print("\n=== Phase 2: Training Reward Model ===")

    d_features = VOCAB_SIZE + 1  # character features + length feature
    reward_params = init_reward_model(d_features)
    reward_param_count = REWARD_HIDDEN * d_features + REWARD_HIDDEN + REWARD_HIDDEN + 1
    print(f"Reward model parameters: {reward_param_count} (plain floats)")

    # Create synthetic preference pairs from names.txt.
    # "Chosen" = names with 4-7 characters (well-formed, pronounceable names).
    # "Rejected" = names with 1-3 characters (too short) or 8+ characters (too long).
    # This synthetic signal mimics a human annotator who prefers moderate-length names.
    # The boundary at 3/8 (rather than 2/10) gives the reward model clear signal about
    # the preferred range. Real RLHF collects human comparisons; we use this heuristic
    # to demonstrate the same algorithmic pipeline.
    chosen_names: list[str] = []
    rejected_names: list[str] = []
    for name in docs:
        if 4 <= len(name) <= 7:
            chosen_names.append(name)
        elif len(name) <= 3 or len(name) >= 8:
            rejected_names.append(name)

    random.shuffle(chosen_names)
    random.shuffle(rejected_names)

    # Create preference pairs by zipping chosen and rejected lists
    n_pairs = min(200, len(chosen_names), len(rejected_names))
    preference_pairs: list[tuple[str, str]] = [
        (chosen_names[i], rejected_names[i]) for i in range(n_pairs)
    ]
    print(f"Created {n_pairs} preference pairs")

    # Hold out 20% for evaluation
    split = int(0.8 * n_pairs)
    train_pairs = preference_pairs[:split]
    eval_pairs = preference_pairs[split:]

    for step in range(REWARD_STEPS):
        pair = train_pairs[step % len(train_pairs)]
        chosen_tokens = [unique_chars.index(ch) for ch in pair[0]]
        rejected_tokens = [unique_chars.index(ch) for ch in pair[1]]

        feat_c = sequence_to_features(chosen_tokens, VOCAB_SIZE)
        feat_r = sequence_to_features(rejected_tokens, VOCAB_SIZE)

        rloss = reward_backward(feat_c, feat_r, reward_params, REWARD_LR)

        if (step + 1) % 100 == 0 or step == 0:
            correct = 0
            for ep in eval_pairs:
                c_tok = [unique_chars.index(ch) for ch in ep[0]]
                r_tok = [unique_chars.index(ch) for ch in ep[1]]
                sc = reward_forward(sequence_to_features(c_tok, VOCAB_SIZE), reward_params)
                sr = reward_forward(sequence_to_features(r_tok, VOCAB_SIZE), reward_params)
                if sc > sr:
                    correct += 1
            acc = 100.0 * correct / len(eval_pairs)
            print(f"  step {step + 1:>4}/{REWARD_STEPS} | ranking_loss: {rloss:.4f} | accuracy: {acc:.1f}%")

    # Final reward model accuracy
    correct = 0
    for ep in eval_pairs:
        c_tok = [unique_chars.index(ch) for ch in ep[0]]
        r_tok = [unique_chars.index(ch) for ch in ep[1]]
        sc = reward_forward(sequence_to_features(c_tok, VOCAB_SIZE), reward_params)
        sr = reward_forward(sequence_to_features(r_tok, VOCAB_SIZE), reward_params)
        if sc > sr:
            correct += 1
    final_acc = 100.0 * correct / len(eval_pairs)
    print(f"Reward model accuracy: {final_acc:.1f}%")

    # Calibrate reward model: compute mean and std on a sample of training-distribution
    # sequences so we can normalize rewards during PPO. Without normalization, the
    # absolute reward scale is arbitrary, making the advantage signal hard to interpret
    # and the KL penalty coefficient hard to tune.
    cal_rewards: list[float] = []
    for name in docs[:200]:
        tok = [unique_chars.index(ch) for ch in name]
        cal_rewards.append(score_completion(tok, VOCAB_SIZE, reward_params))
    reward_mean = sum(cal_rewards) / len(cal_rewards)
    reward_var = sum((r - reward_mean) ** 2 for r in cal_rewards) / len(cal_rewards)
    reward_std = max(math.sqrt(reward_var), 1e-4)

    def normalized_reward(token_ids: list[int]) -> float:
        """Return reward combining the learned reward model with length shaping.

        The normalized reward model score captures character-level preferences learned
        from pairwise comparisons. The length shaping term provides an explicit signal
        for the preferred name length (4-7 chars) because the MLP reward model can
        have blind spots for out-of-distribution lengths (empty, 1-char sequences)
        that never appeared in its training data.

        Signpost: Production RLHF systems also use reward shaping (format penalties,
        safety classifiers, length bonuses). Our length shaping serves the same role.
        """
        n = len(token_ids)
        if n < MIN_GEN_LEN:
            return -3.0

        raw = score_completion(token_ids, VOCAB_SIZE, reward_params)
        norm = (raw - reward_mean) / reward_std

        # Length shaping: bonus for the preferred range, mild penalty outside it
        if 4 <= n <= 7:
            length_bonus = 1.0
        elif n == 3:
            length_bonus = 0.0
        else:
            length_bonus = -0.5

        return norm + length_bonus

    # === Phase 3: PPO Training ===
    print("\n=== Phase 3: PPO Training ===")

    value_params = init_value_function(d_features)
    value_param_count = d_features + 1
    print(f"Value function parameters: {value_param_count} (plain floats)")
    print(f"PPO clip epsilon: {PPO_CLIP_EPS} | KL coefficient: {KL_COEFF} (squared penalty)")

    # Fresh Adam state for PPO fine-tuning (do not carry over pretraining momentum,
    # since the objective has changed from language modeling to reward maximization)
    m_ppo = [0.0] * len(policy_param_list)
    v_ppo = [0.0] * len(policy_param_list)

    # Store pretrained model state for "before PPO" comparison
    pretrained_param_data: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        pretrained_param_data[key] = [[v.data for v in row] for row in matrix]

    # Track metrics across PPO steps for the final summary
    all_rewards: list[float] = []
    all_kl: list[float] = []

    for step in range(PPO_STEPS):
        # --- Step 1: Generate a batch of completions from current policy ---
        batch_tokens: list[list[int]] = []
        batch_rewards: list[float] = []
        batch_old_logps: list[float] = []
        batch_ref_logps: list[float] = []
        batch_features: list[list[float]] = []

        for _ in range(BATCH_SIZE):
            gen_tokens = generate_completion(
                policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.8
            )
            # Ensure non-empty completion (degenerate empty sequences give no gradient signal)
            if not gen_tokens:
                gen_tokens = [random.randint(0, VOCAB_SIZE - 2)]

            batch_tokens.append(gen_tokens)

            # --- Step 2: Score with normalized reward model ---
            reward = normalized_reward(gen_tokens)
            batch_rewards.append(reward)
            features = sequence_to_features(gen_tokens, VOCAB_SIZE)
            batch_features.append(features)

            # --- Step 5: Store old log-probs (before parameter update) ---
            old_logp = compute_log_probs_detached(gen_tokens, BOS, policy_params)
            batch_old_logps.append(old_logp)

            # Reference log-probs for KL penalty
            ref_logp = compute_ref_log_probs(gen_tokens)
            batch_ref_logps.append(ref_logp)

        # --- Step 3: Compute value baselines and advantages ---
        batch_advantages: list[float] = []
        for i in range(BATCH_SIZE):
            val = value_forward(batch_features[i], value_params)
            # Advantage = reward - baseline. This centers the reward signal so the policy
            # gradient has lower variance: actions that are better than average get positive
            # advantage (reinforced), worse than average get negative (discouraged).
            # Without the baseline, all rewards > 0 would reinforce all actions equally.
            batch_advantages.append(batch_rewards[i] - val)

        # --- Step 6: PPO update (through autograd) ---
        # The PPO clipped surrogate objective prevents catastrophically large updates.
        # Math: L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        # where ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
        # and A is the advantage.
        #
        # Intuition: vanilla policy gradient uses ratio * A, which can produce huge
        # updates when the ratio is large (policy changed a lot). PPO clips the ratio
        # to [1-eps, 1+eps], bounding the step size. This is the "proximal" constraint:
        # stay close to the old policy.
        total_ppo_loss = Value(0.0)
        total_kl = 0.0

        for i in range(BATCH_SIZE):
            # Compute current log-prob WITH autograd (expensive but necessary)
            current_logp = compute_log_probs_autograd(batch_tokens[i], BOS, policy_params)

            # Importance sampling ratio: how much has the policy changed for this completion?
            # ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
            log_ratio = current_logp - batch_old_logps[i]
            ratio = log_ratio.exp()

            # Clipped surrogate objective
            adv = batch_advantages[i]
            surr1 = ratio * adv                                                    # unclipped
            surr2 = ratio.clip(1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv    # clipped

            # Take the minimum: this is conservative. When advantage > 0 (good action),
            # we don't let the ratio exceed 1+eps (prevent over-reinforcing). When
            # advantage < 0 (bad action), we don't let the ratio go below 1-eps
            # (prevent over-penalizing). Either way, the update is bounded.
            if surr1.data < surr2.data:
                ppo_obj = surr1
            else:
                ppo_obj = surr2

            # KL penalty: discourages the policy from drifting too far from the reference.
            # Without this, the policy would collapse to a degenerate distribution that
            # maximizes the (imperfect) reward model -- "reward hacking". The KL term
            # preserves the reference model's language modeling quality.
            #
            # We use a squared log-ratio penalty: 0.5 * (log_pi - log_pi_ref)^2
            # Unlike the raw log-ratio (which can be negative for individual samples),
            # the squared form is always >= 0 and penalizes divergence in both directions.
            # This is equivalent to the Schulman (2020) "KL penalty" variant.
            kl_per_sample = current_logp.data - batch_ref_logps[i]
            total_kl += abs(kl_per_sample)

            # Squared KL penalty through autograd so gradients flow back to policy
            log_diff = current_logp - batch_ref_logps[i]
            kl_penalty = KL_COEFF * log_diff * log_diff * 0.5

            # Total loss: negate PPO objective (we minimize loss, PPO maximizes objective)
            # plus KL penalty term (always positive, pushes policy toward reference)
            sample_loss = -ppo_obj + kl_penalty
            total_ppo_loss = total_ppo_loss + sample_loss

        # Average over batch
        ppo_loss = total_ppo_loss * (1.0 / BATCH_SIZE)
        avg_kl = total_kl / BATCH_SIZE
        avg_reward = sum(batch_rewards) / BATCH_SIZE

        # Backward pass and Adam update on policy parameters only
        ppo_loss.backward()

        lr_t = PPO_LR * (1 - step / PPO_STEPS)
        for i, p in enumerate(policy_param_list):
            # Gradient clipping: prevent exploding gradients from the multi-step computation
            # graph (policy forward pass runs multiple times per PPO step across the batch).
            grad = max(-1.0, min(1.0, p.grad))
            m_ppo[i] = BETA1 * m_ppo[i] + (1 - BETA1) * grad
            v_ppo[i] = BETA2 * v_ppo[i] + (1 - BETA2) * grad ** 2
            m_hat = m_ppo[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_ppo[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        # --- Step 7: Update value function with MSE loss ---
        for i in range(BATCH_SIZE):
            value_update(batch_features[i], batch_rewards[i], value_params, VALUE_LR)

        all_rewards.append(avg_reward)
        all_kl.append(avg_kl)

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{PPO_STEPS} | reward: {avg_reward:.2f} | "
                  f"kl_div: {avg_kl:.2f} | ppo_loss: {ppo_loss.data:.4f}")

    # === Results ===
    print("\n=== Results ===")

    # Generate from pretrained model (before PPO) by temporarily restoring weights
    current_data_backup: dict[str, list[list[float]]] = {}
    for key, matrix in policy_params.items():
        current_data_backup[key] = [[v.data for v in row] for row in matrix]
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                v.data = pretrained_param_data[key][r][c]

    print("Generating from PRETRAINED model (before PPO):")
    pre_rewards: list[float] = []
    pre_lengths: list[int] = []
    for i in range(10):
        gen = generate_completion(policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.5)
        name = ''.join(unique_chars[t] for t in gen if t < len(unique_chars))
        shaped_r = normalized_reward(gen)
        pre_rewards.append(shaped_r)
        pre_lengths.append(len(gen))
        print(f"  {i + 1:>2}. {name:10s} (reward: {shaped_r:+.2f}, len: {len(gen)})")

    # Restore PPO-trained weights
    for key, matrix in policy_params.items():
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                v.data = current_data_backup[key][r][c]

    print("\nGenerating from PPO-ALIGNED model:")
    post_rewards: list[float] = []
    post_lengths: list[int] = []
    for i in range(10):
        gen = generate_completion(policy_params, BOS, VOCAB_SIZE, MAX_GEN_LEN, temperature=0.5)
        name = ''.join(unique_chars[t] for t in gen if t < len(unique_chars))
        shaped_r = normalized_reward(gen)
        post_rewards.append(shaped_r)
        post_lengths.append(len(gen))
        print(f"  {i + 1:>2}. {name:10s} (reward: {shaped_r:+.2f}, len: {len(gen)})")

    avg_pre = sum(pre_rewards) / len(pre_rewards)
    avg_post = sum(post_rewards) / len(post_rewards)
    avg_pre_len = sum(pre_lengths) / len(pre_lengths)
    avg_post_len = sum(post_lengths) / len(post_lengths)
    avg_kl_final = sum(all_kl[-20:]) / len(all_kl[-20:]) if all_kl else 0.0

    print(f"\nAverage reward -- Before PPO: {avg_pre:+.2f} | After PPO: {avg_post:+.2f}")
    print(f"Average length -- Before PPO: {avg_pre_len:.1f} | After PPO: {avg_post_len:.1f}")
    print(f"Average KL divergence from reference: {avg_kl_final:.2f}")

    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.1f}s")
