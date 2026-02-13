"""
Beyond greedy: six decoding strategies for language model text generation, from deterministic
argmax to speculative decoding with a draft-verify two-model pipeline.
"""
# Reference: Leviathan et al., "Fast Inference from Transformers via Speculative
# Decoding" (2023). https://arxiv.org/abs/2211.17192
# Also: Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019).
# https://arxiv.org/abs/1904.09751 (nucleus/top-p sampling)

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

# Target model (larger, ~4,200 params) and draft model (smaller, ~1,300 params).
# Both share vocabulary and block_size — required for speculative decoding since
# the draft model must produce tokens the target model can verify.
TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER = 16, 4, 1
DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER = 8, 2, 1
BLOCK_SIZE = 16

# Training
LEARNING_RATE, BETA1, BETA2, EPS_ADAM = 0.01, 0.85, 0.99, 1e-8
TARGET_STEPS, DRAFT_STEPS = 700, 500

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: production speculative decoding pairs a 70B target with a 7B draft.
# Our 4,200 / 1,300 param ratio preserves the algorithmic structure. Real speedups
# come from GPU parallelism during the verify pass — here we measure acceptance
# rate, which is the hardware-independent metric that matters.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput), then backward() replays the graph in
    reverse topological order, accumulating gradients.
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
        # d(x^n)/dx = n * x^(n-1)
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

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort then chain rule."""
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
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# Autograd is used only for training. All decoding strategies use plain float
# forward passes for inference speed.
# See docs/autograd-interface.md for the full specification.


# === TRAINING HELPERS (Value-based) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]

def linear_v(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]

def softmax_v(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm_v(x: list[Value]) -> list[Value]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def safe_log(prob: Value) -> Value:
    """Clipped log: clamp to 1e-10 to prevent log(0). Node keeps prob as child
    so gradients flow back through the computation graph."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))

def gpt_forward_train(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
    n_embd: int, n_head: int, n_layer: int, head_dim: int,
) -> list[Value]:
    """Single-token GPT forward pass for training, parameterized by model config."""
    x = [t + p for t, p in zip(params['wte'][token_id], params['wpe'][pos_id])]
    x = rmsnorm_v(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm_v(x)
        q = linear_v(x, params[f'l{li}.wq'])
        k = linear_v(x, params[f'l{li}.wk'])
        v = linear_v(x, params[f'l{li}.wv'])
        keys[li].append(k); values[li].append(v)
        # Multi-head attention with incremental KV construction (implicit causal mask)
        x_attn: list[Value] = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [kt[hs:hs + head_dim] for kt in keys[li]]
            v_h = [vt[hs:hs + head_dim] for vt in values[li]]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                      for t in range(len(k_h))]
            w = softmax_v(scores)
            x_attn.extend([sum(w[t] * v_h[t][j] for t in range(len(v_h)))
                           for j in range(head_dim)])
        x = linear_v(x_attn, params[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm_v(x)
        x = linear_v(x, params[f'l{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear_v(x, params[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_v(x, params['lm_head'])


# === PLAIN-FLOAT INFERENCE ===
# After training, weights are extracted as plain floats. All six decoding
# strategies operate here — no autograd overhead, enabling clean comparison.

def extract(w: list[list[Value]]) -> list[list[float]]:
    return [[v.data for v in row] for row in w]

def linear_f(x: list[float], w: list[list[float]]) -> list[float]:
    return [sum(w[i][j] * x[j] for j in range(len(x))) for i in range(len(w))]

def softmax_f(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm_f(x: list[float]) -> list[float]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    return [xi * (mean_sq + 1e-5) ** -0.5 for xi in x]

# Model config dict: n_embd, n_head, n_layer, head_dim, vocab_size, bos
Cfg = dict

def forward_float(tok: int, pos: int, kv: list[dict[str, list[list[float]]]],
                  wf: dict[str, list[list[float]]], c: Cfg) -> list[float]:
    """GPT forward pass using plain floats. Processes one token, appends K/V to
    kv cache, returns logits. Shared by all decoding strategies — they differ
    only in how they select the next token from these logits."""
    x = [wf['wte'][tok][j] + wf['wpe'][pos][j] for j in range(c['n_embd'])]
    x = rmsnorm_f(x)
    for li in range(c['n_layer']):
        x_res = x[:]
        x = rmsnorm_f(x)
        q = linear_f(x, wf[f'l{li}.wq'])
        k = linear_f(x, wf[f'l{li}.wk'])
        v = linear_f(x, wf[f'l{li}.wv'])
        kv[li]['k'].append(k); kv[li]['v'].append(v)
        head_cat: list[float] = []
        clen = len(kv[li]['k'])
        hd = c['head_dim']
        for h in range(c['n_head']):
            hs = h * hd
            q_h = q[hs:hs + hd]
            scores = [sum(q_h[j] * kv[li]['k'][t][hs + j] for j in range(hd)) / (hd ** 0.5)
                      for t in range(clen)]
            w = softmax_f(scores)
            for j in range(hd):
                head_cat.append(sum(w[t] * kv[li]['v'][t][hs + j] for t in range(clen)))
        x = linear_f(head_cat, wf[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x[:]
        x = rmsnorm_f(x)
        x = linear_f(x, wf[f'l{li}.fc1'])
        x = [max(0.0, v) for v in x]
        x = linear_f(x, wf[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_f(x, wf['lm_head'])

def make_kv(c: Cfg) -> list[dict[str, list[list[float]]]]:
    return [{'k': [], 'v': []} for _ in range(c['n_layer'])]

def clone_kv(cache: list[dict[str, list[list[float]]]]) -> list[dict[str, list[list[float]]]]:
    """Deep copy KV cache so beam branches don't share mutable state."""
    return [{'k': [r[:] for r in l['k']], 'v': [r[:] for r in l['v']]} for l in cache]

def feed_prompt(toks: list[int], wf: dict[str, list[list[float]]],
                c: Cfg) -> tuple[list[dict[str, list[list[float]]]], list[float]]:
    """Feed prompt through the model, returning (kv_cache, last_logits)."""
    kv = make_kv(c)
    logits: list[float] = []
    for i, t in enumerate(toks):
        logits = forward_float(t, i, kv, wf, c)
    return kv, logits


# === DECODING STRATEGIES ===
# Each strategy takes a prompt, weights, and config, returns generated tokens
# plus total log-probability. They differ ONLY in token selection.

def decode_greedy(prompt: list[int], wf: dict, c: Cfg,
                  max_len: int = 12) -> tuple[list[int], float]:
    """Always pick the highest-probability token. Deterministic.

    Simple but suboptimal: commits to the locally best choice at each step,
    which can miss globally better sequences. Greedy decoding is optimal only
    when the model is perfectly calibrated (it never is).
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        tok = max(range(c['vocab_size']), key=lambda i: probs[i])
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_temperature(prompt: list[int], wf: dict, c: Cfg,
                       max_len: int = 12, temperature: float = 0.8) -> tuple[list[int], float]:
    """Scale logits by temperature before sampling.

    Temperature reshapes the probability distribution without changing its
    ranking. T < 1 sharpens (more deterministic), T > 1 flattens (more random).
    The math: softmax(logits/T) concentrates mass on the mode as T -> 0
    and approaches uniform as T -> inf.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f([l / temperature for l in logits])
        tok = random.choices(range(c['vocab_size']), weights=probs)[0]
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_top_k(prompt: list[int], wf: dict, c: Cfg,
                 max_len: int = 12, k: int = 5) -> tuple[list[int], float]:
    """Only consider the k most likely tokens, zero out the rest.

    Prevents sampling from the long tail of unlikely tokens. The cutoff is
    fixed regardless of the model's confidence — this rigidity is top-k's
    weakness compared to top-p.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_set = set(ranked[:k])
        filt = [probs[i] if i in top_set else 0.0 for i in range(len(probs))]
        total = sum(filt)
        filt = [p / total for p in filt]
        tok = random.choices(range(c['vocab_size']), weights=filt)[0]
        if tok == c['bos']: break
        # Log-prob from the ORIGINAL distribution — measures model confidence
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_top_p(prompt: list[int], wf: dict, c: Cfg,
                 max_len: int = 12, p: float = 0.9) -> tuple[list[int], float]:
    """Include tokens until cumulative probability exceeds p (nucleus sampling).

    Adaptive: for confident predictions (one token at 95%), only that token
    is considered. For uncertain predictions, many tokens enter the nucleus.
    This adaptivity is why top-p often outperforms fixed top-k — the model's
    own confidence determines the effective vocabulary size at each step.
    """
    kv, logits = feed_prompt(prompt, wf, c)
    gen: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        pos = len(prompt) + len(gen)
        if pos >= BLOCK_SIZE: break
        probs = softmax_f(logits)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        nucleus: set[int] = set()
        for idx in ranked:
            nucleus.add(idx)
            cumsum += probs[idx]
            if cumsum >= p: break
        filt = [probs[i] if i in nucleus else 0.0 for i in range(len(probs))]
        total = sum(filt)
        filt = [pr / total for pr in filt]
        tok = random.choices(range(c['vocab_size']), weights=filt)[0]
        if tok == c['bos']: break
        lp += math.log(max(probs[tok], 1e-10))
        gen.append(tok)
        logits = forward_float(tok, pos, kv, wf, c)
    return gen, lp


def decode_beam(prompt: list[int], wf: dict, c: Cfg,
                max_len: int = 12, beam_width: int = 3) -> tuple[list[int], float]:
    """Maintain top-B candidate sequences, expand and prune at each step.

    Finds higher log-probability sequences than greedy by exploring multiple
    paths simultaneously. Beam search is NOT sampling — it is a deterministic
    search algorithm. Two runs with the same input produce identical output.
    The key tradeoff: beam_width * cost_per_step compute for potentially much
    better global solutions. Used heavily in machine translation.
    """
    # Each beam: (cumulative_log_prob, generated_tokens, kv_cache, pending_logits)
    init_kv, init_logits = feed_prompt(prompt, wf, c)
    beams: list[tuple[float, list[int], list[dict[str, list[list[float]]]], list[float]]] = [
        (0.0, [], clone_kv(init_kv), init_logits)
    ]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        candidates: list[tuple[float, list[int], list[dict[str, list[list[float]]]], list[float]]] = []
        for blp, btoks, bkv, blogits in beams:
            pos = len(prompt) + len(btoks)
            if pos >= BLOCK_SIZE:
                completed.append((blp, btoks))
                continue
            probs = softmax_f(blogits)
            ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            for idx in ranked[:beam_width]:
                token_lp = math.log(max(probs[idx], 1e-10))
                if idx == c['bos']:
                    completed.append((blp + token_lp, btoks))
                    continue
                # Each expansion gets its own KV cache copy (beams diverge)
                new_kv = clone_kv(bkv)
                new_logits = forward_float(idx, pos, new_kv, wf, c)
                candidates.append((blp + token_lp, btoks + [idx], new_kv, new_logits))
        if not candidates: break
        # Prune: keep only top beam_width by cumulative log-prob
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    all_results = completed + [(lp, toks) for lp, toks, _, _ in beams]
    if not all_results: return [], 0.0
    best = max(all_results, key=lambda r: r[0])
    return best[1], best[0]


def decode_speculative(
    prompt: list[int], t_wf: dict, d_wf: dict, tc: Cfg, dc: Cfg,
    max_len: int = 12, draft_k: int = 4,
) -> tuple[list[int], float, int, int]:
    """Draft model generates k tokens, target model verifies.

    The key insight: verifying k tokens with the target model costs roughly
    the same as generating 1 token (on a GPU, k forward passes batch into one).
    If the draft tokens match the target's distribution, we get ~k tokens per
    target verification — a significant speedup.

    Acceptance (Leviathan et al.): accept each draft token with probability
    min(1, p_target/p_draft). On rejection, resample from max(0, p_target -
    p_draft) and discard subsequent drafts. This is lossless: the output
    distribution exactly matches the target model.

    Returns: (tokens, log_prob, total_proposed, total_accepted)
    """
    t_kv, t_logits = feed_prompt(prompt, t_wf, tc)
    d_kv, d_logits = feed_prompt(prompt, d_wf, dc)
    gen: list[int] = []
    lp = 0.0
    total_proposed = 0
    total_accepted = 0

    while len(gen) < max_len:
        cur = len(prompt) + len(gen)
        remaining = min(draft_k, max_len - len(gen))
        if cur >= BLOCK_SIZE or remaining <= 0: break

        # Phase 1: Draft model proposes k tokens greedily (fast, small model)
        draft_toks: list[int] = []
        draft_probs: list[list[float]] = []
        tmp_d_kv = clone_kv(d_kv)
        tmp_d_logits = d_logits[:]
        for di in range(remaining):
            pos = cur + di
            if pos >= BLOCK_SIZE: break
            dp = softmax_f(tmp_d_logits)
            draft_probs.append(dp)
            dtok = max(range(dc['vocab_size']), key=lambda i: dp[i])
            if dtok == dc['bos']: break
            draft_toks.append(dtok)
            tmp_d_logits = forward_float(dtok, pos, tmp_d_kv, d_wf, dc)

        if not draft_toks:
            # Draft produced BOS — fall back to one target greedy step
            tp = softmax_f(t_logits)
            ttok = max(range(tc['vocab_size']), key=lambda i: tp[i])
            if ttok == tc['bos']: break
            lp += math.log(max(tp[ttok], 1e-10))
            gen.append(ttok)
            t_logits = forward_float(ttok, cur, t_kv, t_wf, tc)
            d_logits = forward_float(ttok, cur, d_kv, d_wf, dc)
            continue

        total_proposed += len(draft_toks)

        # Phase 2: Target model verifies each draft token
        # On GPU this would be one batched forward pass. The acceptance logic is
        # identical to the parallel version regardless of serial/parallel execution.
        accepted: list[int] = []
        tmp_t_kv = clone_kv(t_kv)
        tmp_t_logits = t_logits[:]

        for vi in range(len(draft_toks)):
            tp = softmax_f(tmp_t_logits)
            dp = draft_probs[vi]
            dtok = draft_toks[vi]
            # Rejection sampling: accept with p = min(1, p_target/p_draft)
            ratio = min(1.0, tp[dtok] / max(dp[dtok], 1e-10))
            if random.random() < ratio:
                accepted.append(dtok)
                lp += math.log(max(tp[dtok], 1e-10))
                tmp_t_logits = forward_float(dtok, cur + vi, tmp_t_kv, t_wf, tc)
            else:
                # Reject: resample from max(0, p_target - p_draft)
                adj = [max(0.0, tp[j] - dp[j]) for j in range(len(tp))]
                adj_s = sum(adj)
                if adj_s > 0:
                    adj = [a / adj_s for a in adj]
                    rtok = random.choices(range(tc['vocab_size']), weights=adj)[0]
                else:
                    rtok = random.choices(range(tc['vocab_size']), weights=tp)[0]
                if rtok != tc['bos']:
                    accepted.append(rtok)
                    lp += math.log(max(tp[rtok], 1e-10))
                    forward_float(rtok, cur + vi, tmp_t_kv, t_wf, tc)
                break  # Discard all remaining draft tokens after rejection

        total_accepted += len(accepted)

        # Commit accepted tokens to both real KV caches
        for ai, atok in enumerate(accepted):
            t_logits = forward_float(atok, cur + ai, t_kv, t_wf, tc)
            d_logits = forward_float(atok, cur + ai, d_kv, d_wf, dc)
            gen.append(atok)

        if not accepted:
            tp = softmax_f(t_logits)
            ttok = max(range(tc['vocab_size']), key=lambda i: tp[i])
            if ttok == tc['bos']: break
            lp += math.log(max(tp[ttok], 1e-10))
            gen.append(ttok)
            t_logits = forward_float(ttok, cur, t_kv, t_wf, tc)
            d_logits = forward_float(ttok, cur, d_kv, d_wf, dc)

    return gen, lp, total_proposed, total_accepted


# === MODEL INIT AND TRAINING ===

def init_params(vocab_size: int, n_embd: int, n_head: int,
                n_layer: int) -> dict[str, list[list[Value]]]:
    """Initialize all GPT parameters for the given dimensions."""
    p: dict[str, list[list[Value]]] = {}
    p['wte'] = make_matrix(vocab_size, n_embd)
    p['wpe'] = make_matrix(BLOCK_SIZE, n_embd)
    for li in range(n_layer):
        p[f'l{li}.wq'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wk'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wv'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.wo'] = make_matrix(n_embd, n_embd)
        p[f'l{li}.fc1'] = make_matrix(4 * n_embd, n_embd)
        p[f'l{li}.fc2'] = make_matrix(n_embd, 4 * n_embd)
    p['lm_head'] = make_matrix(vocab_size, n_embd)
    return p


def train_model(docs: list[str], chars: list[str], bos: int, vocab_size: int,
                params: dict[str, list[list[Value]]], n_embd: int, n_head: int,
                n_layer: int, head_dim: int, num_steps: int) -> None:
    """Train a GPT model with Adam optimizer and linear LR decay."""
    plist = [p for w in params.values() for row in w for p in row]
    m_s = [0.0] * len(plist)
    v_s = [0.0] * len(plist)
    print(f"Parameters: {len(plist):,}")

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        toks = [bos] + [chars.index(ch) for ch in doc] + [bos]
        sl = min(BLOCK_SIZE, len(toks) - 1)
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        losses: list[Value] = []
        for pos in range(sl):
            logits = gpt_forward_train(toks[pos], pos, keys, vals, params,
                                       n_embd, n_head, n_layer, head_dim)
            probs = softmax_v(logits)
            losses.append(-safe_log(probs[toks[pos + 1]]))
        loss = (1.0 / sl) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / num_steps)
        for i, p in enumerate(plist):
            m_s[i] = BETA1 * m_s[i] + (1 - BETA1) * p.grad
            v_s[i] = BETA2 * v_s[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_s[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_s[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{num_steps} | loss: {loss.data:.4f}")
    print(f"  Final loss: {loss.data:.4f}\n")


# === MAIN ===

if __name__ == "__main__":
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Loaded {len(docs)} documents, vocab size: {VOCAB_SIZE}\n")

    # Train target model (larger)
    print(f"=== Training Target Model (n_embd={TARGET_N_EMBD}, n_layer={TARGET_N_LAYER}) ===")
    target_params = init_params(VOCAB_SIZE, TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER)
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, target_params,
                TARGET_N_EMBD, TARGET_N_HEAD, TARGET_N_LAYER,
                TARGET_N_EMBD // TARGET_N_HEAD, TARGET_STEPS)
    print(f"Target model trained in {time.time() - t0:.1f}s")

    # Train draft model (smaller)
    print(f"\n=== Training Draft Model (n_embd={DRAFT_N_EMBD}, n_layer={DRAFT_N_LAYER}) ===")
    draft_params = init_params(VOCAB_SIZE, DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER)
    t0 = time.time()
    train_model(docs, unique_chars, BOS, VOCAB_SIZE, draft_params,
                DRAFT_N_EMBD, DRAFT_N_HEAD, DRAFT_N_LAYER,
                DRAFT_N_EMBD // DRAFT_N_HEAD, DRAFT_STEPS)
    print(f"Draft model trained in {time.time() - t0:.1f}s")

    # Extract weights as plain floats for inference
    twf = {k: extract(v) for k, v in target_params.items()}
    dwf = {k: extract(v) for k, v in draft_params.items()}
    tc: Cfg = {'n_embd': TARGET_N_EMBD, 'n_head': TARGET_N_HEAD,
               'n_layer': TARGET_N_LAYER, 'head_dim': TARGET_N_EMBD // TARGET_N_HEAD,
               'vocab_size': VOCAB_SIZE, 'bos': BOS}
    dc: Cfg = {'n_embd': DRAFT_N_EMBD, 'n_head': DRAFT_N_HEAD,
               'n_layer': DRAFT_N_LAYER, 'head_dim': DRAFT_N_EMBD // DRAFT_N_HEAD,
               'vocab_size': VOCAB_SIZE, 'bos': BOS}

    def tok2str(toks: list[int]) -> str:
        return ''.join(unique_chars[t] if t != BOS else '' for t in toks)

    # === DECODING STRATEGIES COMPARISON ===
    print("\n=== Decoding Strategies Comparison ===")
    prompts = [("a", [BOS, unique_chars.index('a')]),
               ("m", [BOS, unique_chars.index('m')])]

    for label, ptoks in prompts:
        print(f'\nPrompt: "{label}" (BOS + \'{label}\')\n')
        print(f"{'Strategy':<22} {'Output':<16} {'Log-Prob':>10} {'Tokens/Step':>12}")
        print("-" * 62)

        g, glp = decode_greedy(ptoks, twf, tc)
        print(f"{'Greedy':<22} {tok2str(g):<16} {glp:>10.2f} {'1.0':>12}")

        t, tlp = decode_temperature(ptoks, twf, tc, temperature=0.8)
        print(f"{'Temperature (0.8)':<22} {tok2str(t):<16} {tlp:>10.2f} {'1.0':>12}")

        k, klp = decode_top_k(ptoks, twf, tc, k=5)
        print(f"{'Top-k (k=5)':<22} {tok2str(k):<16} {klp:>10.2f} {'1.0':>12}")

        p, plp = decode_top_p(ptoks, twf, tc, p=0.9)
        print(f"{'Top-p (p=0.9)':<22} {tok2str(p):<16} {plp:>10.2f} {'1.0':>12}")

        b, blp = decode_beam(ptoks, twf, tc, beam_width=3)
        print(f"{'Beam (width=3)':<22} {tok2str(b):<16} {blp:>10.2f} {'1.0':>12}")

        s, slp, sp, sa = decode_speculative(ptoks, twf, dwf, tc, dc, draft_k=4)
        tps = sa / max(1, sp / 4) if sp > 0 else 1.0
        print(f"{'Speculative (k=4)':<22} {tok2str(s):<16} {slp:>10.2f} {tps:>12.1f}")

    # === DIVERSITY ANALYSIS ===
    # Deterministic strategies (greedy, beam) produce the same output for the same
    # prompt. Stochastic strategies (temperature, top-k, top-p) produce diversity —
    # essential for creative applications, undesirable for factual ones.
    print("\n=== Diversity Analysis ===")
    print("Generated 20 names with each strategy:\n")
    n_samp = 20
    seeds = list("abcdefghijklmnopqrst")
    strats = [
        ("Greedy", lambda pt: decode_greedy(pt, twf, tc)),
        ("Temperature (0.8)", lambda pt: decode_temperature(pt, twf, tc, temperature=0.8)),
        ("Top-k (k=5)", lambda pt: decode_top_k(pt, twf, tc, k=5)),
        ("Top-p (p=0.9)", lambda pt: decode_top_p(pt, twf, tc, p=0.9)),
        ("Beam (width=3)", lambda pt: decode_beam(pt, twf, tc, beam_width=3)),
    ]
    print(f"{'Strategy':<22} {'Unique Names':>13} {'Avg Length':>11} {'Avg Log-Prob':>13}")
    print("-" * 62)
    for sname, sfn in strats:
        names: list[str] = []
        lps: list[float] = []
        for i in range(n_samp):
            pt = [BOS, unique_chars.index(seeds[i])]
            toks, lp = sfn(pt)
            names.append(tok2str(toks))
            lps.append(lp)
        print(f"{sname:<22} {len(set(names)):>13} "
              f"{sum(len(n) for n in names) / n_samp:>11.1f} "
              f"{sum(lps) / n_samp:>13.2f}")

    # === SPECULATIVE DECODING STATS ===
    print("\n=== Speculative Decoding Stats ===")
    tot_prop = tot_acc = 0
    for i in range(n_samp):
        pt = [BOS, unique_chars.index(seeds[i])]
        _, _, prop, acc = decode_speculative(pt, twf, dwf, tc, dc, draft_k=4)
        tot_prop += prop
        tot_acc += acc

    acc_rate = 100 * tot_acc / max(tot_prop, 1)
    n_rounds = tot_prop / 4
    toks_per_round = tot_acc / max(n_rounds, 1)
    print(f"Draft tokens proposed per step: 4")
    print(f"Total proposed: {tot_prop} | Total accepted: {tot_acc}")
    print(f"Average acceptance rate: {acc_rate:.1f}%")
    print(f"Average tokens accepted per target verify pass: {toks_per_round:.1f}")
    # Signpost: in production with a well-matched draft model, acceptance rates of
    # 70-90% are common. The real GPU speedup comes from batching the k verification
    # forward passes into a single kernel launch — our scalar Python cannot show that
    # parallelism, but the acceptance rate is the hardware-independent metric.
