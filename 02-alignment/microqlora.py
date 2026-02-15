"""
How to fine-tune a 4-bit quantized model with full-precision LoRA adapters — the technique
that brought LLM fine-tuning to consumer GPUs.
"""
# Reference: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models"
# (2023). https://arxiv.org/abs/2305.14314
# Combines microquant.py's quantization with microlora.py's low-rank adaptation.
# Architecture: microgpt pattern (Radford et al., 2019) with pedagogical simplifications.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — identical to microgpt/microlora for cross-script comparison
N_EMBD = 16         # embedding dimension (d_model)
N_HEAD = 4          # number of attention heads
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 16     # context window length
HEAD_DIM = N_EMBD // N_HEAD  # 4

# LoRA hyperparameters
LORA_RANK = 2       # rank of adaptation matrices (r << d_model)
# Same rank as microlora — the adaptation mechanism is identical, the difference
# is that base weights are quantized rather than full-precision.

# Quantization parameters
QUANT_BITS = 4      # INT4 quantization for base weights
BLOCK_SIZE_QUANT = 8  # quantization block size (weights per group)
# NF4 quantizes in blocks — each block of BLOCK_SIZE_QUANT weights shares a single
# scale factor. Smaller blocks = better accuracy but more overhead.

# Training — base model pretraining (full precision)
BASE_LR = 0.01
BASE_STEPS = 800

# Training — QLoRA fine-tuning
QLORA_LR = 0.01
QLORA_STEPS = 500

# Shared optimizer constants
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: At toy scale (16x16 weight matrices), INT4 quantization barely matters —
# the memory savings are invisible. At production scale (4096x4096 and larger), INT4
# cuts memory by 8x (32-bit → 4-bit), making 7B models fit on a single 16GB GPU.
# The algorithm is identical; the savings are proportional to model size.


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
# Follows the canonical interface exactly. QLoRA's gradients flow through the LoRA
# adapters (full-precision Value objects) while base weights are quantized integers
# with no autograd. The dequantize operation creates Value objects from quantized data,
# allowing gradients to flow through the LoRA path but NOT through the base weights.
# See docs/autograd-interface.md for the full specification.


# === QUANTIZATION ENGINE ===

def compute_nf4_levels() -> list[float]:
    """Compute the 16 NormalFloat4 quantization levels.

    NF4 is the quantization scheme from the QLoRA paper. Instead of uniformly spacing
    INT4 levels across [-max, +max], NF4 places levels at quantiles of a normal
    distribution — more levels near zero (where most weights cluster) and fewer at
    the extremes.

    Why this matters: neural network weights follow an approximately normal distribution.
    Uniform quantization wastes half its levels on extreme values that few weights use.
    NF4 concentrates precision where the density is highest, minimizing quantization error.

    The 16 levels (-8..+7 mapped to float) are precomputed from N(0,1) quantiles.
    """
    # Compute quantiles of the standard normal distribution.
    # For 4-bit quantization, we have 16 levels (0..15).
    # Each level maps to the midpoint of its quantile range.
    num_levels = 2 ** QUANT_BITS  # 16

    # Use the inverse CDF (quantile function) of the standard normal.
    # We approximate it using the Beasley-Springer-Moro algorithm.
    def norm_quantile(p: float) -> float:
        """Approximate inverse normal CDF (percent-point function)."""
        # Rational approximation for 0.5 < p < 1.0
        # For p < 0.5, use symmetry: Φ^-1(p) = -Φ^-1(1-p)
        if p < 0.5:
            return -norm_quantile(1 - p)
        if p >= 1.0:
            return 4.0
        if p <= 0.0:
            return -4.0
        # Abramowitz and Stegun approximation 26.2.23
        t = math.sqrt(-2.0 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    # Place levels at quantile midpoints: level i corresponds to the midpoint
    # between the (i/16)-th and ((i+1)/16)-th quantiles of N(0,1).
    levels = []
    for i in range(num_levels):
        # Quantile midpoint for bin i
        p_lo = i / num_levels
        p_hi = (i + 1) / num_levels
        p_mid = (p_lo + p_hi) / 2
        levels.append(norm_quantile(p_mid))

    return levels


# Precompute NF4 levels (these are constant across all quantization operations)
NF4_LEVELS = compute_nf4_levels()


def quantize_block_nf4(weights: list[float]) -> tuple[list[int], float]:
    """Quantize a block of float weights to NF4 integers.

    The quantization process:
    1. Compute the absolute maximum of the block (the scale factor)
    2. Normalize weights to [-1, 1] by dividing by absmax
    3. Map each normalized weight to the nearest NF4 level
    4. Store the integer index (0..15) and the scale factor

    Returns: (quantized_indices, scale)
    """
    if not weights:
        return [], 1.0

    # Scale factor: maps the weight range to [-1, 1]
    absmax = max(abs(w) for w in weights)
    scale = absmax if absmax > 0 else 1.0

    # Quantize: find the nearest NF4 level for each normalized weight
    indices = []
    for w in weights:
        normalized = w / scale
        # Find the closest NF4 level
        best_idx = 0
        best_dist = abs(normalized - NF4_LEVELS[0])
        for idx in range(1, len(NF4_LEVELS)):
            dist = abs(normalized - NF4_LEVELS[idx])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        indices.append(best_idx)

    return indices, scale


def dequantize_block_nf4(indices: list[int], scale: float) -> list[float]:
    """Reconstruct float weights from NF4 indices and scale factor.

    dequant(idx) = NF4_LEVELS[idx] * scale

    The reconstruction is lossy — each weight is rounded to its nearest NF4 level.
    The quantization error is the difference between the original and reconstructed weight.
    """
    return [NF4_LEVELS[idx] * scale for idx in indices]


def quantize_matrix(matrix: list[list[float]]) -> list[list[tuple[list[int], float]]]:
    """Quantize a weight matrix using block-wise NF4.

    Each row is split into blocks of BLOCK_SIZE_QUANT weights. Each block gets its
    own scale factor. Smaller blocks = more scale factors = more overhead but better
    accuracy. Production QLoRA uses block sizes of 64.

    Returns a structure where each element is (quantized_indices, scale) per block.
    """
    quantized_rows = []
    for row in matrix:
        blocks = []
        for start in range(0, len(row), BLOCK_SIZE_QUANT):
            block = row[start: start + BLOCK_SIZE_QUANT]
            indices, scale = quantize_block_nf4(block)
            blocks.append((indices, scale))
        quantized_rows.append(blocks)
    return quantized_rows


def dequantize_matrix_to_values(
    quantized: list[list[tuple[list[int], float]]]
) -> list[list[Value]]:
    """Dequantize a matrix back to Value objects for forward pass computation.

    THE KEY QLORA PATTERN: During the forward pass, quantized base weights are
    dequantized on-the-fly to full-precision Values. These Values participate in
    the forward computation but their gradients are NOT accumulated (we never update
    the base weights). Only the LoRA adapter weights receive gradient updates.

    This dequantize-during-forward pattern is what makes QLoRA work: the model
    operates at full precision during computation (preserving accuracy) while
    storing weights at 4-bit precision (saving memory). The LoRA adapters provide
    the trainable pathway.
    """
    rows = []
    for quant_row in quantized:
        row_values = []
        for indices, scale in quant_row:
            floats = dequantize_block_nf4(indices, scale)
            # These Values are created fresh each forward pass — no persistent gradient
            row_values.extend([Value(f) for f in floats])
        rows.append(row_values)
    return rows


# === DOUBLE QUANTIZATION ===

def double_quantize_scales(scales: list[float], bits: int = 8) -> tuple[list[int], float]:
    """Quantize the scale factors themselves — the 'double' in double quantization.

    In standard block quantization, each block stores a FP32 scale factor (4 bytes).
    With many small blocks, these scale factors add up. Double quantization compresses
    them to INT8, reducing the overhead from 4 bytes to 1 byte per block.

    At our toy scale this saves negligible memory, but at production scale with millions
    of blocks, double quantization saves ~0.37 bits per parameter (from the QLoRA paper).
    """
    if not scales:
        return [], 1.0

    num_levels = 2 ** bits  # 256 for INT8
    absmax = max(abs(s) for s in scales)
    meta_scale = absmax if absmax > 0 else 1.0

    quantized = []
    for s in scales:
        # Map to [0, num_levels-1] range
        normalized = s / meta_scale
        idx = int(round((normalized + 1) * (num_levels - 1) / 2))
        idx = max(0, min(num_levels - 1, idx))
        quantized.append(idx)

    return quantized, meta_scale


# === CORE OPERATIONS ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_lora_pair(d_out: int, d_in: int, rank: int) -> tuple[list[list[Value]], list[list[Value]]]:
    """Initialize a LoRA adapter pair (B, A) with proper scaling.

    LoRA decomposes a weight update ΔW into two low-rank matrices: ΔW = B @ A
    where A: [rank, d_in] and B: [d_out, rank].

    B is initialized to zero so the model starts identical to the base model.
    A is initialized with small random values to break symmetry.
    """
    lora_B = [[Value(random.gauss(0, 0.01)) for _ in range(d_in)] for _ in range(rank)]
    lora_A = [[Value(0.0) for _ in range(rank)] for _ in range(d_out)]
    return lora_B, lora_A


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x (no bias)."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def qlora_linear(
    x: list[Value],
    quantized_base: list[list[tuple[list[int], float]]],
    lora_B: list[list[Value]],
    lora_A: list[list[Value]],
) -> list[Value]:
    """QLoRA forward: dequantized base weights + full-precision LoRA adapter.

    y = dequant(W_base) @ x + (A @ (B @ x)) * (α/r)

    This is the core QLoRA operation. The base weight matrix is stored in INT4 and
    dequantized to float on-the-fly. The LoRA adapter (B @ A) is full precision and
    receives all gradient updates during training.

    The dequantized base weights introduce quantization noise but no trainable parameters.
    The LoRA adapter compensates for quantization error AND adapts to the new task.
    """
    # Dequantize base weights to Values (fresh each pass, no persistent gradient)
    base_w = dequantize_matrix_to_values(quantized_base)
    base_out = linear(x, base_w)

    # LoRA path: B projects down to rank, A projects back up
    # B: [rank, d_in] @ x: [d_in] → [rank]
    lora_hidden = linear(x, lora_B)
    # A: [d_out, rank] @ hidden: [rank] → [d_out]
    lora_out = linear(lora_hidden, lora_A)

    # Sum base and adapter outputs
    return [b + l for b, l in zip(base_out, lora_out)]


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
    """Initialize full-precision model parameters (for pretraining)."""
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


def init_qlora_adapters() -> dict:
    """Initialize LoRA adapters for Q and V projections (same as microlora)."""
    lora = {}
    for layer_idx in range(N_LAYER):
        lora_q_B, lora_q_A = make_lora_pair(N_EMBD, N_EMBD, LORA_RANK)
        lora_v_B, lora_v_A = make_lora_pair(N_EMBD, N_EMBD, LORA_RANK)
        lora[f'layer{layer_idx}.lora_q_B'] = lora_q_B
        lora[f'layer{layer_idx}.lora_q_A'] = lora_q_A
        lora[f'layer{layer_idx}.lora_v_B'] = lora_v_B
        lora[f'layer{layer_idx}.lora_v_A'] = lora_v_A
    return lora


# === GPT FORWARD PASS ===

def gpt_forward_full(
    token_id: int, pos_id: int,
    keys: list, values: list,
    params: dict,
) -> list[Value]:
    """Standard full-precision forward pass (for pretraining)."""
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


def gpt_forward_qlora(
    token_id: int, pos_id: int,
    keys: list, values: list,
    quant_params: dict,
    lora: dict,
    embed_params: dict,
) -> list[Value]:
    """QLoRA forward pass: quantized base + full-precision LoRA adapters.

    Base weights (attention, MLP) are stored in NF4 and dequantized during forward.
    Embeddings and LM head remain full-precision (they're lookup tables, not matmuls).
    LoRA adapters on Q and V projections receive all gradient updates.

    This is the core QLoRA pattern:
    1. Embeddings: full precision (lookup, no quantization benefit)
    2. Attention Q: dequant(W_q) @ x + LoRA_q(x)  ← trainable via LoRA
    3. Attention K: dequant(W_k) @ x                ← frozen, quantized
    4. Attention V: dequant(W_v) @ x + LoRA_v(x)  ← trainable via LoRA
    5. Attention O: dequant(W_o) @ x                ← frozen, quantized
    6. MLP fc1/fc2: dequant(W) @ x                  ← frozen, quantized
    7. LM head: full precision (output projection)
    """
    tok_emb = embed_params['wte'][token_id]
    pos_emb = embed_params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        # Q and V use QLoRA: dequantized base + LoRA adapter
        q = qlora_linear(
            x,
            quant_params[f'layer{layer_idx}.attn_wq'],
            lora[f'layer{layer_idx}.lora_q_B'],
            lora[f'layer{layer_idx}.lora_q_A'],
        )
        v_proj = qlora_linear(
            x,
            quant_params[f'layer{layer_idx}.attn_wv'],
            lora[f'layer{layer_idx}.lora_v_B'],
            lora[f'layer{layer_idx}.lora_v_A'],
        )

        # K and O: dequantized base only (no LoRA)
        k_base = dequantize_matrix_to_values(quant_params[f'layer{layer_idx}.attn_wk'])
        k = linear(x, k_base)
        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

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

        o_base = dequantize_matrix_to_values(quant_params[f'layer{layer_idx}.attn_wo'])
        x = linear(x_attn, o_base)
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        x = rmsnorm(x)
        fc1_base = dequantize_matrix_to_values(quant_params[f'layer{layer_idx}.mlp_fc1'])
        x = linear(x, fc1_base)
        x = [xi.relu() for xi in x]
        fc2_base = dequantize_matrix_to_values(quant_params[f'layer{layer_idx}.mlp_fc2'])
        x = linear(x, fc2_base)
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, embed_params['lm_head'])


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    # Split data: first 80% for pretraining, last 20% for QLoRA fine-tuning
    # This mirrors microlora's split — pretrain on general data, adapt to specific subset
    split_point = int(len(docs) * 0.8)
    pretrain_docs = docs[:split_point]
    finetune_docs = docs[split_point:]

    print(f"Vocabulary: {VOCAB_SIZE} tokens ({len(unique_chars)} chars + BOS)")
    print(f"Pretrain: {len(pretrain_docs)} names | Fine-tune: {len(finetune_docs)} names")

    # === PHASE 1: PRETRAIN BASE MODEL (full precision) ===
    print(f"\n{'=' * 60}")
    print("PHASE 1: Pretraining base model (full precision)")
    print(f"{'=' * 60}")

    params = init_parameters(VOCAB_SIZE)
    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,} (all FP32)")

    m_adam = [0.0] * len(param_list)
    v_adam = [0.0] * len(param_list)

    for step in range(BASE_STEPS):
        doc = pretrain_docs[step % len(pretrain_docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            logits = gpt_forward_full(tokens[pos], pos, keys, vals, params)
            probs = softmax(logits)
            loss_t = -safe_log(probs[tokens[pos + 1]])
            losses.append(loss_t)

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = BASE_LR * (1 - step / BASE_STEPS)
        for i, p in enumerate(param_list):
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * p.grad
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_adam[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{BASE_STEPS} | loss: {loss.data:.4f}")

    pretrain_time = time.time() - start_time
    print(f"\nPretraining complete ({pretrain_time:.1f}s). Final loss: {loss.data:.4f}")

    # === PHASE 2: QUANTIZE BASE MODEL ===
    print(f"\n{'=' * 60}")
    print("PHASE 2: Quantizing base weights to NF4 (4-bit)")
    print(f"{'=' * 60}")

    # Extract float weights and quantize attention/MLP matrices
    # Embeddings and LM head stay full-precision (they're lookup tables)
    quant_params = {}
    embed_params = {'wte': params['wte'], 'wpe': params['wpe'], 'lm_head': params['lm_head']}

    total_weights = 0
    total_quant_error = 0.0

    for key, matrix in params.items():
        if key in ('wte', 'wpe', 'lm_head'):
            continue  # skip embeddings

        # Extract float values
        float_matrix = [[v.data for v in row] for row in matrix]

        # Quantize to NF4
        quantized = quantize_matrix(float_matrix)
        quant_params[key] = quantized

        # Measure quantization error
        for row_idx, row in enumerate(float_matrix):
            for block_idx, (indices, scale) in enumerate(quantized[row_idx]):
                start = block_idx * BLOCK_SIZE_QUANT
                dequantized = dequantize_block_nf4(indices, scale)
                for j, (orig, deq) in enumerate(zip(row[start:start + len(dequantized)], dequantized)):
                    total_quant_error += (orig - deq) ** 2
                    total_weights += 1

    rmse = math.sqrt(total_quant_error / max(total_weights, 1))
    print(f"  Quantized {total_weights} weights to NF4")
    print(f"  Quantization RMSE: {rmse:.6f}")
    print(f"  Block size: {BLOCK_SIZE_QUANT} weights per scale factor")

    # Double quantization of scale factors
    all_scales = []
    for key, quant_matrix in quant_params.items():
        for quant_row in quant_matrix:
            for _, scale in quant_row:
                all_scales.append(scale)

    dq_indices, dq_meta_scale = double_quantize_scales(all_scales)
    print(f"  Scale factors: {len(all_scales)} (double-quantized to INT8)")

    # Memory comparison
    fp32_bytes = total_weights * 4  # 4 bytes per float32
    nf4_bytes = total_weights * 0.5  # 0.5 bytes per 4-bit value
    scale_bytes = len(all_scales) * 4  # 4 bytes per scale (before double quant)
    dq_scale_bytes = len(all_scales) * 1 + 4  # 1 byte per INT8 + 4 bytes meta scale

    print(f"\n  Memory comparison (attention + MLP weights only):")
    print(f"    FP32:           {fp32_bytes:>6} bytes")
    print(f"    NF4:            {nf4_bytes:>6.0f} bytes + {scale_bytes} scale bytes"
          f" = {nf4_bytes + scale_bytes:.0f} bytes")
    print(f"    NF4 + double Q: {nf4_bytes:>6.0f} bytes + {dq_scale_bytes} scale bytes"
          f" = {nf4_bytes + dq_scale_bytes:.0f} bytes")
    print(f"    Compression:    {fp32_bytes / (nf4_bytes + dq_scale_bytes):.1f}x")

    # Signpost: At toy scale the compression ratio is modest (~3-4x) because scale factor
    # overhead dominates. At production scale with large weight matrices and block_size=64,
    # the ratio approaches the theoretical 8x (32-bit → 4-bit).

    # === PHASE 3: QLORA FINE-TUNING ===
    print(f"\n{'=' * 60}")
    print("PHASE 3: QLoRA fine-tuning (INT4 base + FP32 LoRA adapters)")
    print(f"{'=' * 60}")

    # Initialize LoRA adapters (full precision, trainable)
    lora = init_qlora_adapters()
    lora_param_list = [p for matrix in lora.values() for row in matrix for p in row]
    print(f"  Base model weights: {total_weights} (frozen, NF4)")
    print(f"  LoRA adapter params: {len(lora_param_list)} (trainable, FP32)")
    print(f"  Trainable ratio: {len(lora_param_list) / max(total_weights, 1):.1%}")

    # Embedding params also need to be in the trainable list for gradient zeroing
    embed_param_list = [p for key in ('wte', 'wpe', 'lm_head')
                        for row in embed_params[key] for p in row]

    # Adam state for LoRA params only (base weights are frozen)
    m_lora = [0.0] * len(lora_param_list)
    v_lora = [0.0] * len(lora_param_list)

    qlora_start = time.time()

    for step in range(QLORA_STEPS):
        doc = finetune_docs[step % len(finetune_docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            logits = gpt_forward_qlora(
                tokens[pos], pos, keys, vals, quant_params, lora, embed_params
            )
            probs = softmax(logits)
            loss_t = -safe_log(probs[tokens[pos + 1]])
            losses.append(loss_t)

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # Update ONLY LoRA parameters (base weights stay frozen and quantized)
        lr_t = QLORA_LR * (1 - step / QLORA_STEPS)
        for i, p in enumerate(lora_param_list):
            m_lora[i] = BETA1 * m_lora[i] + (1 - BETA1) * p.grad
            v_lora[i] = BETA2 * v_lora[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_lora[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_lora[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        # Zero gradients on embedding params (they accumulate but we don't update them
        # through the LoRA path — in production QLoRA, embeddings can optionally be trained)
        for p in embed_param_list:
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{QLORA_STEPS} | loss: {loss.data:.4f}")

    qlora_time = time.time() - qlora_start
    print(f"\nQLoRA fine-tuning complete ({qlora_time:.1f}s). Final loss: {loss.data:.4f}")

    # === INFERENCE ===
    print(f"\n{'=' * 60}")
    print("INFERENCE: QLoRA-adapted model")
    print(f"{'=' * 60}")

    TEMPERATURE = 0.5
    NUM_SAMPLES = 15

    print(f"\nGenerating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        token_id = BOS
        generated = []

        for pos in range(BLOCK_SIZE):
            logits = gpt_forward_qlora(
                token_id, pos, keys, vals, quant_params, lora, embed_params
            )
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            token_id = random.choices(
                range(VOCAB_SIZE),
                weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")

    # === SUMMARY ===
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("QLORA SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  Phase                  | Time      | Details
  ───────────────────────┼───────────┼──────────────────────────────
  Pretrain (FP32)        | {pretrain_time:>6.1f}s   | {BASE_STEPS} steps, all {len(param_list):,} params
  Quantize (FP32→NF4)   | <0.1s     | {total_weights} weights, RMSE={rmse:.6f}
  QLoRA fine-tune        | {qlora_time:>6.1f}s   | {QLORA_STEPS} steps, {len(lora_param_list)} LoRA params only
  Total                  | {total_time:>6.1f}s   |

  What QLoRA achieves:
    1. Base weights compressed to 4-bit ({fp32_bytes} → {nf4_bytes + dq_scale_bytes:.0f} bytes)
    2. Fine-tuning trains only {len(lora_param_list)} params ({len(lora_param_list) / max(total_weights, 1):.1%} of base)
    3. Full-precision computation preserved (dequantize during forward)
    4. Gradients flow through LoRA adapters only (base stays frozen)

  Cross-reference:
    - microlora.py:  FP32 base + LoRA adapters (no quantization)
    - microquant.py: Quantized model, no fine-tuning (post-training only)
    - microqlora.py: Quantized base + LoRA fine-tuning (this script, best of both)
""")
