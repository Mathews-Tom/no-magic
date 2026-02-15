"""
How DeepSeek simplified RLHF — group-based reward normalization that eliminates the value
function entirely.
"""
# Reference: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
# Open Language Models" (2024). https://arxiv.org/abs/2402.03300
# Also: DeepSeek-R1 (2025), which scaled GRPO to reasoning tasks.
# Architecture reuses the microgpt pattern (Radford et al., 2019) with smaller dimensions
# (n_embd=8) to keep the generate-then-score loop within runtime constraints.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Policy model architecture — smaller than microgpt because GRPO generates multiple
# completions per prompt (the "group"), multiplying the forward pass count.
N_EMBD = 8          # embedding dimension (vs. 16 in microgpt)
N_HEAD = 2          # attention heads
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 12     # context window
HEAD_DIM = N_EMBD // N_HEAD  # 4

# Pretraining parameters
PRETRAIN_LR = 0.01
PRETRAIN_STEPS = 500
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# GRPO parameters
GROUP_SIZE = 4       # number of completions generated per prompt (the "group")
GRPO_STEPS = 80      # number of GRPO optimization steps
GRPO_LR = 0.001      # lower than pretraining to prevent catastrophic forgetting
MAX_GEN_LEN = 8      # maximum generation length
MIN_GEN_LEN = 2      # minimum generation length
KL_COEFF = 0.3       # KL penalty against reference model
# KL_COEFF is higher than typical production values (0.01-0.1) because our tiny model
# with synthetic rewards is prone to mode collapse without strong regularization.

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: Production GRPO (DeepSeek-R1) uses group sizes of 16-64, trains billions of
# parameters, and scores completions with outcome-based reward models on math/code tasks.
# The algorithm is identical at any scale — generate a group, normalize rewards within
# the group, update the policy with the normalized advantages.


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
    """A scalar value with reverse-mode automatic differentiation."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
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

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# Follows the canonical interface exactly. GRPO's novelty is algorithmic (group-relative
# normalization), not operational — same autograd ops as microgpt/microdpo.
# See docs/autograd-interface.md for the full specification.


# === CORE OPERATIONS ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x (no bias)."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square normalization."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm for numerical stability."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === MODEL DEFINITION ===

def init_parameters(vocab_size: int) -> dict:
    """Initialize GPT-style model parameters."""
    params = {}
    params['wte'] = make_matrix(vocab_size, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    params['lm_head'] = make_matrix(vocab_size, N_EMBD)
    return params


def get_param_list(params: dict) -> list[Value]:
    """Flatten all parameter matrices into a single list."""
    return [p for matrix in params.values() for row in matrix for p in row]


def gpt_forward(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict,
) -> list[Value]:
    """Single-token forward pass through the GPT model."""
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

        x_attn = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_h = q[hs: hs + HEAD_DIM]
            k_h = [kt[hs: hs + HEAD_DIM] for kt in keys[layer_idx]]
            v_h = [vt[hs: hs + HEAD_DIM] for vt in values[layer_idx]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === GENERATION ===

def generate_sequence(
    params: dict,
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    max_len: int = MAX_GEN_LEN,
    temperature: float = 0.8,
) -> tuple[list[int], list[Value]]:
    """Generate a sequence and return (token_ids, log_probs_at_each_step).

    Returns the token sequence AND the log-probabilities of each chosen action.
    GRPO needs log π(a_t|s_t) to compute the policy gradient — storing them during
    generation avoids a costly second forward pass.
    """
    keys = [[] for _ in range(N_LAYER)]
    values_cache = [[] for _ in range(N_LAYER)]

    token_id = bos
    generated_tokens: list[int] = []
    log_probs: list[Value] = []

    for pos in range(max_len):
        logits = gpt_forward(token_id, pos, keys, values_cache, params)
        scaled_logits = [logit / temperature for logit in logits]
        probs = softmax(scaled_logits)

        # Sample next token
        weights = [p.data for p in probs]
        token_id = random.choices(range(vocab_size), weights=weights)[0]

        # Store log-probability of the chosen action (needed for policy gradient)
        log_prob = safe_log(probs[token_id])
        log_probs.append(log_prob)

        if token_id == bos:
            break
        generated_tokens.append(token_id)

    return generated_tokens, log_probs


def compute_log_prob_sequence(
    tokens: list[int],
    params: dict,
    bos: int,
) -> float:
    """Compute log P(sequence) under the model. Uses plain floats (no autograd).

    Used to compute log-probs under the reference model, which doesn't need gradients.
    """
    keys = [[] for _ in range(N_LAYER)]
    values_cache = [[] for _ in range(N_LAYER)]

    full_seq = [bos] + tokens + [bos]
    total_log_prob = 0.0

    for pos in range(len(full_seq) - 1):
        logits = gpt_forward(full_seq[pos], pos, keys, values_cache, params)

        # Compute softmax in plain floats for speed
        logit_data = [l.data for l in logits]
        max_logit = max(logit_data)
        exp_vals = [math.exp(l - max_logit) for l in logit_data]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]

        target = full_seq[pos + 1]
        total_log_prob += math.log(max(probs[target], 1e-10))

    return total_log_prob


# === REWARD FUNCTION ===

def compute_reward(tokens: list[int], unique_chars: list[str]) -> float:
    """Score a generated sequence using simple, interpretable rules.

    Same preference signal as microdpo: prefer longer, more varied names.
    This creates a three-way comparison:
    - DPO: offline, uses preference PAIRS (chosen vs rejected)
    - PPO: online RL, uses scalar REWARDS + value function baseline
    - GRPO: online RL, uses scalar REWARDS + group-relative normalization (no value function)

    The reward function is deliberately simple so the alignment effect is interpretable.
    Production GRPO uses outcome-based rewards (math correctness, code execution results).
    """
    name = ''.join(unique_chars[t] for t in tokens if t < len(unique_chars))

    reward = 0.0

    # Length reward: longer names are preferred (matches DPO's length preference signal)
    if len(name) >= 4:
        reward += 1.0
    if len(name) >= 6:
        reward += 1.0

    # Diversity: names with more unique characters are preferred
    if len(name) > 0:
        unique_ratio = len(set(name)) / len(name)
        reward += unique_ratio

    # Vowel presence: names with vowels sound more natural
    vowels = set('aeiou')
    if any(ch in vowels for ch in name):
        reward += 0.5

    # Penalty for degenerate outputs
    if len(name) < MIN_GEN_LEN:
        reward -= 2.0

    return reward


# === GRPO ALGORITHM ===

def grpo_step(
    params: dict,
    ref_log_probs_fn,
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    param_list: list[Value],
    m_state: list[float],
    v_state: list[float],
    step: int,
) -> tuple[float, float, float]:
    """Execute one GRPO optimization step.

    THE CORE GRPO IDEA:
    Instead of training a value function V(s) to estimate expected future reward (like PPO),
    GRPO normalizes rewards WITHIN a group of completions for the same prompt. The advantage
    of completion i is: A_i = (R_i - mean(R_group)) / std(R_group).

    Why this works: the group mean acts as a prompt-specific baseline (like V(s) in PPO,
    but computed from actual samples rather than a learned estimate). The std normalization
    ensures advantages are on a consistent scale regardless of absolute reward magnitude.

    What's eliminated: the entire value function network (V(s)), its training loop, and
    the generalized advantage estimation (GAE) computation. GRPO trades off sample efficiency
    (needs GROUP_SIZE completions per prompt) for simplicity (no value function to train).

    Returns: (mean_reward, policy_loss, kl_divergence)
    """
    # -- Step 1: Generate a GROUP of completions --
    # All completions start from the same initial state (BOS token).
    # In production GRPO, you'd sample diverse prompts; with character-level names
    # and BOS as the only prompt, the group diversity comes from stochastic sampling.
    group_tokens: list[list[int]] = []
    group_log_probs: list[list[Value]] = []
    group_rewards: list[float] = []

    for _ in range(GROUP_SIZE):
        tokens, log_probs = generate_sequence(
            params, unique_chars, bos, vocab_size
        )
        reward = compute_reward(tokens, unique_chars)

        group_tokens.append(tokens)
        group_log_probs.append(log_probs)
        group_rewards.append(reward)

    # -- Step 2: Compute group-relative advantages --
    # This is GRPO's key innovation over PPO:
    #   A_i = (R_i - μ_group) / σ_group
    #
    # The group mean μ replaces PPO's learned value function V(s).
    # The group std σ normalizes scale, making the gradient magnitude independent
    # of absolute reward values (important when rewards vary across prompts/tasks).
    mean_reward = sum(group_rewards) / len(group_rewards)
    variance = sum((r - mean_reward) ** 2 for r in group_rewards) / max(len(group_rewards) - 1, 1)
    std_reward = max(math.sqrt(variance), 1e-8)  # floor to prevent div by zero

    advantages = [(r - mean_reward) / std_reward for r in group_rewards]

    # -- Step 3: Compute GRPO policy gradient loss --
    # L_GRPO = -Σ_i Σ_t A_i * log π_θ(a_t|s_t) + β * KL(π_θ || π_ref)
    #
    # This is a weighted version of REINFORCE where the weights (advantages)
    # are normalized within the group rather than using a learned baseline.
    total_policy_loss = Value(0.0)
    total_kl = 0.0
    num_tokens_total = 0

    for i in range(GROUP_SIZE):
        if not group_log_probs[i]:
            continue

        advantage = advantages[i]

        # Policy gradient: weight each log-prob by the group-relative advantage
        for log_prob in group_log_probs[i]:
            # -advantage * log_prob: negative because we MAXIMIZE expected reward
            # but autograd MINIMIZES the loss
            total_policy_loss = total_policy_loss + (-advantage * log_prob)
            num_tokens_total += 1

        # KL divergence penalty against reference model
        # Prevents the policy from drifting too far from the pretrained distribution.
        # Without KL penalty, GRPO would collapse to outputting whatever maximizes
        # the reward function, losing language quality (mode collapse).
        ref_log_prob = ref_log_probs_fn(group_tokens[i])
        policy_log_prob = sum(lp.data for lp in group_log_probs[i])
        kl_div = policy_log_prob - ref_log_prob
        total_kl += kl_div

    # Average over tokens and add KL penalty
    if num_tokens_total > 0:
        avg_policy_loss = total_policy_loss / num_tokens_total
    else:
        avg_policy_loss = total_policy_loss

    avg_kl = total_kl / max(GROUP_SIZE, 1)
    # Add KL penalty as a scalar (no autograd needed — it modulates the learning rate)
    # In practice, the KL term can also be computed through autograd, but the scalar
    # version is simpler and equally effective for small models.

    # -- Step 4: Backward pass and parameter update --
    avg_policy_loss.backward()

    lr_t = GRPO_LR * (1 - step / GRPO_STEPS)

    for idx, param in enumerate(param_list):
        # Adam update
        m_state[idx] = BETA1 * m_state[idx] + (1 - BETA1) * param.grad
        v_state[idx] = BETA2 * v_state[idx] + (1 - BETA2) * param.grad ** 2

        m_hat = m_state[idx] / (1 - BETA1 ** (step + 1))
        v_hat = v_state[idx] / (1 - BETA2 ** (step + 1))

        # Apply KL penalty as effective learning rate modulation
        effective_lr = lr_t / (1 + KL_COEFF * max(avg_kl, 0))
        param.data -= effective_lr * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        param.grad = 0.0

    return mean_reward, avg_policy_loss.data, avg_kl


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary: {VOCAB_SIZE} tokens ({len(unique_chars)} chars + BOS)")

    # === PHASE 1: PRETRAIN BASE MODEL ===
    # Same pretraining as microdpo/microppo: standard next-token prediction on names.txt
    print(f"\n{'=' * 60}")
    print("PHASE 1: Pretraining base model")
    print(f"{'=' * 60}")

    params = init_parameters(VOCAB_SIZE)
    param_list = get_param_list(params)
    print(f"Parameters: {len(param_list):,}")

    m_adam = [0.0] * len(param_list)
    v_adam = [0.0] * len(param_list)

    for step in range(PRETRAIN_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, vals, params)
            probs = softmax(logits)
            loss_t = -safe_log(probs[tokens[pos + 1]])
            losses.append(loss_t)

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = PRETRAIN_LR * (1 - step / PRETRAIN_STEPS)
        for i, param in enumerate(param_list):
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * param.grad
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * param.grad ** 2
            m_hat = m_adam[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_adam[i] / (1 - BETA2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{PRETRAIN_STEPS} | loss: {loss.data:.4f}")

    pretrain_time = time.time() - start_time
    print(f"\nPretraining complete ({pretrain_time:.1f}s). Final loss: {loss.data:.4f}")

    # === SNAPSHOT REFERENCE MODEL ===
    # Store a frozen copy of pretrained weights. The KL penalty keeps the GRPO-optimized
    # policy from straying too far from this reference — identical to DPO/PPO's reference model.
    ref_params = {}
    for key, matrix in params.items():
        ref_params[key] = [[Value(v.data) for v in row] for row in matrix]

    def ref_log_probs(tokens: list[int]) -> float:
        """Compute log P(tokens) under the frozen reference model."""
        return compute_log_prob_sequence(tokens, ref_params, BOS)

    # Generate samples from base model (before GRPO)
    print("\nBase model samples:")
    base_rewards = []
    for i in range(10):
        tokens, _ = generate_sequence(params, unique_chars, BOS, VOCAB_SIZE)
        name = ''.join(unique_chars[t] for t in tokens if t < len(unique_chars))
        reward = compute_reward(tokens, unique_chars)
        base_rewards.append(reward)
        print(f"  {i + 1:>2}. {name:<12} reward: {reward:.2f}")

    base_mean_reward = sum(base_rewards) / len(base_rewards)
    print(f"\nBase model mean reward: {base_mean_reward:.3f}")

    # === PHASE 2: GRPO ALIGNMENT ===
    print(f"\n{'=' * 60}")
    print("PHASE 2: GRPO alignment")
    print(f"{'=' * 60}")
    print(f"  Group size: {GROUP_SIZE} (completions per prompt)")
    print(f"  KL coefficient: {KL_COEFF}")
    print(f"  Steps: {GRPO_STEPS}")

    # Signpost: GRPO vs PPO vs DPO comparison:
    # - DPO: Offline. Needs preference PAIRS. No generation during training. No reward model.
    # - PPO: Online. Needs scalar REWARDS. Generates + scores. Learns a VALUE FUNCTION for baseline.
    # - GRPO: Online. Needs scalar REWARDS. Generates + scores. Uses GROUP NORMALIZATION for baseline.
    # GRPO eliminates PPO's value function by computing baselines from the group statistics.
    # The tradeoff: GRPO needs GROUP_SIZE forward passes per step (vs. 1 for PPO), but
    # saves the entire value network's parameters and training cost.

    # Reset optimizer for GRPO phase
    m_adam = [0.0] * len(param_list)
    v_adam = [0.0] * len(param_list)

    grpo_start = time.time()

    for step in range(GRPO_STEPS):
        mean_reward, policy_loss, kl_div = grpo_step(
            params, ref_log_probs, unique_chars, BOS, VOCAB_SIZE,
            param_list, m_adam, v_adam, step
        )

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  step {step + 1:>3}/{GRPO_STEPS} | reward: {mean_reward:>6.3f}"
                  f" | loss: {policy_loss:>8.4f} | KL: {kl_div:>6.3f}")

    grpo_time = time.time() - grpo_start
    print(f"\nGRPO complete ({grpo_time:.1f}s)")

    # === INFERENCE: ALIGNED MODEL SAMPLES ===
    print(f"\n{'=' * 60}")
    print("RESULTS: GRPO-aligned model samples")
    print(f"{'=' * 60}")

    aligned_rewards = []
    for i in range(10):
        tokens, _ = generate_sequence(params, unique_chars, BOS, VOCAB_SIZE)
        name = ''.join(unique_chars[t] for t in tokens if t < len(unique_chars))
        reward = compute_reward(tokens, unique_chars)
        aligned_rewards.append(reward)
        print(f"  {i + 1:>2}. {name:<12} reward: {reward:.2f}")

    aligned_mean_reward = sum(aligned_rewards) / len(aligned_rewards)
    print(f"\nAligned model mean reward: {aligned_mean_reward:.3f}")

    # === COMPARISON TABLE ===
    print(f"\n{'=' * 60}")
    print("GRPO ALIGNMENT SUMMARY")
    print(f"{'=' * 60}")

    improvement = aligned_mean_reward - base_mean_reward
    print(f"""
  Metric                    | Base Model    | After GRPO
  ──────────────────────────┼───────────────┼──────────────
  Mean reward               | {base_mean_reward:>10.3f}    | {aligned_mean_reward:>10.3f}
  Reward improvement        |      —        | {improvement:>+10.3f}
  Parameters                | {len(param_list):>10,}    | {len(param_list):>10,} (same)
  Value function params     |      —        |          0 (eliminated)

  GRPO configuration:
    Group size:             {GROUP_SIZE}
    KL coefficient:         {KL_COEFF}
    Optimization steps:     {GRPO_STEPS}

  Total runtime: {time.time() - start_time:.1f}s
""")

    # === GRPO vs DPO vs PPO ===
    print(f"{'=' * 60}")
    print("ALIGNMENT METHOD COMPARISON")
    print(f"{'=' * 60}")
    print("""
  Method | Training Signal   | Baseline       | Online? | Value Function?
  ───────┼───────────────────┼────────────────┼─────────┼────────────────
  DPO    | Preference pairs  | Reference model| No      | No
  PPO    | Scalar rewards    | Learned V(s)   | Yes     | Yes (full network)
  GRPO   | Scalar rewards    | Group mean     | Yes     | No (eliminated)

  GRPO's key insight: instead of training a separate neural network to estimate
  V(s), generate GROUP_SIZE completions and use their mean reward as the baseline.
  This trades off sample efficiency (more forward passes per step) for simplicity
  (no value function to train, tune, or debug).

  When GRPO wins: tasks where rewards are cheap to compute (math verification,
  code execution, rule-based scoring). The extra forward passes cost less than
  training and maintaining a value function.

  When PPO wins: tasks where each reward evaluation is expensive (human feedback,
  long-running simulations). PPO's value function amortizes the cost across steps.
""")
