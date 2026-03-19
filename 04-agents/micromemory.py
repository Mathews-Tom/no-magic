"""
Memory-augmented neural networks from first principles: a Neural Turing Machine learns to
read and write to external memory via differentiable addressing — bridging static retrieval
and dynamic agent memory using only gradient descent.
"""
# Reference: Graves, Wayne, and Danihelka, "Neural Turing Machines" (2014).
# Addressing mechanisms inspired by Graves et al., "Hybrid computing using a neural network
# with dynamic external memory" (2016, Differentiable Neural Computer).

# === TRADEOFFS ===
# + Decouples memory capacity from network size (add more slots, not more weights)
# + Differentiable end-to-end: the network LEARNS where to read and write
# + Solves algorithmic tasks (copy, sort, recall) that fixed-size networks cannot
# - Slow: every time step computes attention over the full memory matrix
# - Addressing is soft (weighted average), not hard (single slot) — blurry reads
# - Scalar autograd makes this pedagogically clear but computationally expensive
# WHEN TO USE: Tasks requiring variable-length working memory, sequence-to-sequence
#   problems with algorithmic structure, or any setting where the model must store
#   and retrieve information across many time steps.
# WHEN NOT TO: Fixed-length classification, tasks with no memory requirements,
#   or production systems where transformers with KV-cache are faster.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === HYPERPARAMETERS ===

# Memory geometry
MEMORY_SLOTS = 6        # N: number of addressable memory locations
MEMORY_DIM = 6          # M: width of each memory slot (vector dimension)

# Controller network
INPUT_DIM = 3           # dimension of input vectors in the copy task
HIDDEN_DIM = 24         # controller MLP hidden layer size
OUTPUT_DIM = INPUT_DIM  # output matches input for the copy task

# Addressing parameters
SHIFT_RANGE = 1         # circular shift: -1, 0, +1 (3 shift positions)
NUM_SHIFT = 2 * SHIFT_RANGE + 1  # = 3

# Training
LEARNING_RATE = 0.03
MOMENTUM = 0.9          # SGD momentum coefficient
NUM_EPISODES = 1500     # training episodes
MIN_SEQ_LEN = 2         # minimum copy sequence length
MAX_SEQ_LEN = 3         # maximum copy sequence length during training
GRAD_CLIP = 10.0        # gradient clipping threshold

# Signpost: These dimensions are tiny (6 slots x 6-wide memory, 3-dim vectors).
# The original NTM paper used 128 slots x 20-wide with 8-dim input vectors and
# trained for 100K+ episodes with RMSProp on tensor-level autograd. We shrink
# everything so training completes in minutes on CPU with scalar autograd.


# === AUTOGRAD ENGINE (Value class) ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1). Exponent must be a Python number, not Value.
        return Value(
            self.data ** exponent, (self,),
            (exponent * self.data ** (exponent - 1),)
        )

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
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def exp(self):
        # d(e^x)/dx = e^x. Clamp to prevent overflow.
        clamped = max(-20.0, min(20.0, self.data))
        e = math.exp(clamped)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x. Clamp to avoid log(0).
        clamped = max(1e-10, self.data)
        return Value(math.log(clamped), (self,), (1.0 / clamped,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,))

    def sigmoid(self):
        # sigmoid(x) = 1 / (1 + exp(-x))
        # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        clamped = max(-20.0, min(20.0, self.data))
        s = 1.0 / (1.0 + math.exp(-clamped))
        return Value(s, (self,), (s * (1.0 - s),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation.

        Builds topological ordering of the computation graph, then propagates
        gradients backward using the chain rule: dL/dx = sum(dL/dy * dy/dx)
        for all outputs y that depend on x.
        """
        topo = []
        visited = set()

        def build_topo(v):
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


# === VECTOR OPERATIONS ===

# These operate on lists of Value objects — our "tensors" are just Python lists.
# Each function is a building block used by the addressing and controller logic.

def dot_product(a: list[Value], b: list[Value]) -> Value:
    """Dot product of two vectors: sum(a_i * b_i)."""
    result = a[0] * b[0]
    for i in range(1, len(a)):
        result = result + a[i] * b[i]
    return result


def vector_norm(v: list[Value]) -> Value:
    """L2 norm: sqrt(sum(v_i^2)). Epsilon prevents division by zero."""
    sq_sum = v[0] * v[0]
    for i in range(1, len(v)):
        sq_sum = sq_sum + v[i] * v[i]
    return (sq_sum + 1e-8) ** 0.5


def cosine_similarity(a: list[Value], b: list[Value]) -> Value:
    """Cosine similarity: dot(a,b) / (||a|| * ||b||).

    Math-to-code:
        cos(a, b) = (a . b) / (||a||_2 * ||b||_2)

    This measures directional alignment, ignoring magnitude. Two vectors pointing
    the same direction have similarity 1, orthogonal vectors have 0, opposite
    vectors have -1. Content-based addressing uses this to find the memory row
    most similar to a query key — the same mechanism as attention in transformers.
    """
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: exp(x_i - max) / sum(exp(x_j - max)).

    Math-to-code:
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Subtracting max(x) before exp prevents overflow while producing identical
    results (the constant cancels in the ratio). The output sums to 1 and can
    be interpreted as a probability distribution over memory slots.
    """
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = exps[0]
    for i in range(1, len(exps)):
        total = total + exps[i]
    return [e / total for e in exps]


def make_vector(dim: int, std: float = 0.1) -> list[Value]:
    """Initialize a random vector of Value objects."""
    return [Value(random.gauss(0, std)) for _ in range(dim)]


def make_matrix(rows: int, cols: int, std: float = 0.1) -> list[list[Value]]:
    """Initialize a random matrix of Value objects."""
    return [make_vector(cols, std) for _ in range(rows)]


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x (no bias).

    w has shape [n_out, n_in], x has shape [n_in], output has shape [n_out].
    """
    return [dot_product(w_row, x) for w_row in w]


def linear_with_bias(
    x: list[Value], w: list[list[Value]], b: list[Value]
) -> list[Value]:
    """y = W @ x + b."""
    out = linear(x, w)
    return [out[i] + b[i] for i in range(len(out))]


def zero_vector(dim: int) -> list[Value]:
    """A vector of zeros (used for initial memory and state)."""
    return [Value(0.0) for _ in range(dim)]


# === MEMORY MATRIX ===

# The memory is a 2D grid of Value objects: M[slot][dim].
# Think of it as a tiny RAM: each row is an addressable "register" that holds
# a vector. The controller can read from and write to any combination of rows
# using soft (weighted) addressing — unlike a real computer's hard (one-hot)
# addressing, every read and write touches all rows, just with different weights.
#
# This is the key insight that makes NTMs differentiable: hard addressing
# (select row 7) has zero gradient with respect to "which row." Soft addressing
# (read 0.8 from row 7, 0.1 from row 8, 0.1 from row 6) has smooth gradients
# that tell the network "you should have read a bit more from row 8."

def init_memory(num_slots: int, slot_dim: int) -> list[list[Value]]:
    """Initialize memory to small random values.

    Signpost: The original NTM paper initializes memory to a learned bias vector.
    We use small random values to break symmetry — if all slots start identical,
    content-based addressing cannot distinguish them on the first step.
    """
    return [
        [Value(random.gauss(0, 0.01)) for _ in range(slot_dim)]
        for _ in range(num_slots)
    ]


# === CONTENT-BASED ADDRESSING ===

# Content-based addressing IS attention over memory. Given a query key k, we
# compute how similar k is to each memory row, then use softmax to get a
# probability distribution over rows. This is exactly Q*K^T/sqrt(d) from
# transformers, except:
#   - Q is the key vector from the controller (1 query, not a sequence)
#   - K and V are the memory rows (persistent storage, not the input)
#   - We use cosine similarity with a sharpness parameter beta instead of
#     scaled dot-product
#
# Self-attention is content-based addressing where the memory IS the input
# sequence. The NTM separates memory from input, allowing persistent storage
# across time steps. This separation is what makes MemGPT, LangChain's
# ConversationBufferMemory, and ChatGPT's Memory feature possible.

def content_addressing(
    key: list[Value],
    memory: list[list[Value]],
    beta: Value,
) -> list[Value]:
    """Compute content-based address weights over memory.

    Math-to-code:
        w_i^c = softmax(beta * cosine(key, M_i))

    beta (sharpness) controls how peaked the distribution is:
        beta -> 0: uniform weights (read everything equally, ignore content)
        beta -> inf: one-hot on the most similar row (precise lookup)

    The network learns to modulate beta based on the task: imprecise reads
    during exploration, sharp reads when it knows exactly what it wants.
    """
    num_slots = len(memory)
    similarities = []
    for i in range(num_slots):
        sim = cosine_similarity(key, memory[i])
        similarities.append(beta * sim)
    return softmax(similarities)


# === LOCATION-BASED ADDRESSING ===

# Content addressing finds memory by WHAT it contains. Location addressing finds
# memory by WHERE it is. The NTM combines both: content addressing proposes
# "I want the row most similar to this key," then location addressing adjusts
# "actually, shift one position to the right."
#
# Why both? The copy task requires sequential access: read slot 0, then slot 1,
# then slot 2. Content addressing alone can't express "the next slot" — it can
# only say "the slot containing X." Location addressing adds the concept of
# relative position.

def interpolate(
    content_weights: list[Value],
    previous_weights: list[Value],
    gate: Value,
) -> list[Value]:
    """Interpolation gate: blend content-based and previous weights.

    Math-to-code:
        w_i^g = g * w_i^c + (1 - g) * w_i^{t-1}

    g = 1: fully trust content addressing (look up by content)
    g = 0: fully trust previous weights (stay where you were)

    The gate lets the controller choose between "find something specific" (high g)
    and "keep reading from the same place" (low g). Sequential reads use low g
    combined with shift to step through memory one slot at a time.
    """
    result = []
    for i in range(len(content_weights)):
        w = gate * content_weights[i] + (Value(1.0) - gate) * previous_weights[i]
        result.append(w)
    return result


def circular_convolve(
    weights: list[Value], shift_kernel: list[Value]
) -> list[Value]:
    """Circular convolution: shift the weight distribution over memory.

    Math-to-code:
        w_shifted[i] = sum_j w[j] * s[(i - j) mod N]

    where s is the shift kernel (3 values: shift left, stay, shift right).

    This implements relative addressing: if the current attention is focused on
    slot 5 and the shift kernel says "shift right by 1," the new attention moves
    to slot 6. The "circular" part means shifting past the last slot wraps to
    the first — memory is a ring buffer.

    Signpost: We implement this as explicit modular indexing. The original NTM
    paper describes it as circular convolution, which could use FFT for large
    memories. With 8 slots, direct computation is faster.
    """
    num_slots = len(weights)
    num_shifts = len(shift_kernel)
    half = num_shifts // 2  # = 1 for shift range of [-1, 0, +1]
    result = []
    for i in range(num_slots):
        val = Value(0.0)
        # shift_kernel[0] = shift left (slot i-1 contributes to slot i)
        # shift_kernel[1] = stay       (slot i contributes to slot i)
        # shift_kernel[2] = shift right (slot i+1 contributes to slot i)
        for s in range(num_shifts):
            j = (i - s + half) % num_slots
            val = val + weights[j] * shift_kernel[s]
        result.append(val)
    return result


def sharpen(weights: list[Value], gamma: float) -> list[Value]:
    """Sharpen the address distribution by raising to a power and renormalizing.

    Math-to-code:
        w_i^sharp = w_i^gamma / sum_j(w_j^gamma)

    gamma >= 1. Higher gamma concentrates the distribution: if weights are
    [0.4, 0.3, 0.2, 0.1] and gamma=2, the squared weights become
    [0.16, 0.09, 0.04, 0.01] which after renormalization is
    [0.53, 0.30, 0.13, 0.03] — sharper peaks.

    Why sharpen? Circular convolution can blur the address distribution.
    Shifting a peaked distribution creates a spread. Sharpening counteracts
    this blur, keeping addresses precise enough for single-slot reads/writes.
    """
    # gamma is a plain float (>= 1) to keep computation graph shallow
    powered = [w ** gamma for w in weights]
    total = powered[0]
    for i in range(1, len(powered)):
        total = total + powered[i]
    return [p / (total + 1e-10) for p in powered]


def full_addressing(
    key: list[Value],
    beta: Value,
    gate: Value,
    shift_kernel: list[Value],
    gamma: float,
    memory: list[list[Value]],
    previous_weights: list[Value],
) -> list[Value]:
    """Complete NTM addressing pipeline: content -> interpolate -> shift -> sharpen.

    This 50-line addressing mechanism is the algorithmic core of MemGPT's archival
    memory, LangChain's ConversationBufferMemory (the non-trivial version), and
    ChatGPT's Memory feature. Those systems implement the same read/write-with-
    attention pattern, just with transformer-scale models instead of scalar autograd.

    The four stages form a pipeline:
    1. Content addressing: WHAT to access (cosine similarity lookup)
    2. Interpolation: HOW MUCH to trust content vs. location (gate)
    3. Shift: WHERE to move relative to current position (convolution)
    4. Sharpen: HOW PRECISE the final address should be (power + renorm)
    """
    # Stage 1: Content-based lookup
    content_weights = content_addressing(key, memory, beta)

    # Stage 2: Blend with previous time step's weights
    gated_weights = interpolate(content_weights, previous_weights, gate)

    # Stage 3: Circular shift (relative positioning)
    shifted_weights = circular_convolve(gated_weights, shift_kernel)

    # Stage 4: Sharpen to counteract convolution blur
    final_weights = sharpen(shifted_weights, gamma)

    return final_weights


# === READ HEAD ===

def read_memory(
    memory: list[list[Value]], weights: list[Value]
) -> list[Value]:
    """Read from memory using soft addressing: weighted sum of all rows.

    Math-to-code:
        r = sum_i(w_i * M_i)

    where w_i is the attention weight for slot i and M_i is that slot's vector.
    This is identical to the "value" computation in transformer attention:
    output = attention_weights @ V. The difference is that V here is the
    persistent memory, not a projection of the current input.
    """
    num_slots = len(memory)
    dim = len(memory[0])
    result = zero_vector(dim)
    for i in range(num_slots):
        for d in range(dim):
            result[d] = result[d] + weights[i] * memory[i][d]
    return result


# === WRITE HEAD ===

# Writing uses an erase-then-add protocol inspired by how LSTM gates work.
# First, erase: selectively zero out parts of addressed memory slots.
# Then, add: write new content into those (now partially empty) slots.
#
# Why erase-then-add instead of just overwrite? Overwrite (M = new_value) would
# destroy ALL content in a slot. Erase-then-add lets the network keep some old
# content while adding new content — it can append, modify, or fully replace
# depending on the erase and add vectors it produces.

def write_memory(
    memory: list[list[Value]],
    weights: list[Value],
    erase_vector: list[Value],
    add_vector: list[Value],
) -> list[list[Value]]:
    """Write to memory using erase-then-add protocol.

    Math-to-code:
        Erase:  M_i = M_i * (1 - w_i * e)     for each slot i, element-wise
        Add:    M_i = M_i + w_i * a             for each slot i, element-wise

    where w_i is the write weight for slot i, e is the erase vector (values
    in [0,1] from sigmoid), and a is the add vector.

    When w_i = 1 and e = [1,1,...,1]: the slot is fully erased then overwritten.
    When w_i = 0: the slot is untouched (0 erase, 0 add).
    When e = [0,0,...,0]: nothing is erased, only added (accumulate mode).
    """
    num_slots = len(memory)
    dim = len(memory[0])
    new_memory = []
    for i in range(num_slots):
        new_row = []
        for d in range(dim):
            # Erase: multiply by (1 - w_i * e_d)
            erased = memory[i][d] * (Value(1.0) - weights[i] * erase_vector[d])
            # Add: add w_i * a_d
            written = erased + weights[i] * add_vector[d]
            new_row.append(written)
        new_memory.append(new_row)
    return new_memory


# === CONTROLLER NETWORK ===

# The controller is a small MLP that acts as the NTM's "CPU." It takes the
# current input and previous read vector, then outputs everything needed
# for one read-write cycle:
#   - Key vector (for content addressing)
#   - Beta (sharpness scalar)
#   - Gate (interpolation scalar)
#   - Shift kernel (3 values: left, stay, right)
#   - Erase vector (what to erase from memory)
#   - Add vector (what to write to memory)
#   - Output prediction
#
# All of these are differentiable functions of the input, so the network
# learns WHEN and WHERE to read/write purely from the training signal.
# This is why differentiable addressing matters: the network discovers its
# own memory access patterns via gradient descent.

def softplus(x: Value) -> Value:
    """softplus(x) = log(1 + exp(x)). Smooth approximation to ReLU.

    Used to produce positive scalars (beta) without hard constraints.
    Unlike ReLU, softplus is never exactly zero, so gradients always flow.
    """
    if x.data > 20.0:
        return x
    return (Value(1.0) + x.exp()).log()


def init_controller() -> dict:
    """Initialize controller parameters.

    Architecture: (input + read_vector) -> hidden -> {addressing params, output}

    The controller input is the concatenation of the current input vector and
    the previous read vector. This gives the controller access to both new
    information and what it previously retrieved from memory.
    """
    controller_input_dim = INPUT_DIM + MEMORY_DIM
    params = {}

    # Xavier-like init: std = 1/sqrt(fan_in)
    std_hidden = 1.0 / math.sqrt(controller_input_dim)
    params['w_hidden'] = make_matrix(HIDDEN_DIM, controller_input_dim, std_hidden)
    params['b_hidden'] = [Value(0.0) for _ in range(HIDDEN_DIM)]

    std_out = 1.0 / math.sqrt(HIDDEN_DIM)

    # --- Read head parameters ---
    params['w_rkey'] = make_matrix(MEMORY_DIM, HIDDEN_DIM, std_out)
    params['b_rkey'] = [Value(0.0) for _ in range(MEMORY_DIM)]

    params['w_rbeta'] = make_matrix(1, HIDDEN_DIM, std_out)
    params['b_rbeta'] = [Value(0.0)]

    params['w_rgate'] = make_matrix(1, HIDDEN_DIM, std_out)
    params['b_rgate'] = [Value(0.0)]

    params['w_rshift'] = make_matrix(NUM_SHIFT, HIDDEN_DIM, std_out)
    params['b_rshift'] = [Value(0.0) for _ in range(NUM_SHIFT)]

    # --- Write head parameters (separate addressing from read) ---
    params['w_wkey'] = make_matrix(MEMORY_DIM, HIDDEN_DIM, std_out)
    params['b_wkey'] = [Value(0.0) for _ in range(MEMORY_DIM)]

    params['w_wbeta'] = make_matrix(1, HIDDEN_DIM, std_out)
    params['b_wbeta'] = [Value(0.0)]

    params['w_wgate'] = make_matrix(1, HIDDEN_DIM, std_out)
    params['b_wgate'] = [Value(0.0)]

    params['w_wshift'] = make_matrix(NUM_SHIFT, HIDDEN_DIM, std_out)
    params['b_wshift'] = [Value(0.0) for _ in range(NUM_SHIFT)]

    # Write content vectors
    params['w_erase'] = make_matrix(MEMORY_DIM, HIDDEN_DIM, std_out)
    params['b_erase'] = [Value(0.0) for _ in range(MEMORY_DIM)]

    params['w_add'] = make_matrix(MEMORY_DIM, HIDDEN_DIM, std_out)
    params['b_add'] = [Value(0.0) for _ in range(MEMORY_DIM)]

    # Output: takes hidden + read_vec concatenated
    output_input_dim = HIDDEN_DIM + MEMORY_DIM
    std_output = 1.0 / math.sqrt(output_input_dim)
    params['w_output'] = make_matrix(OUTPUT_DIM, output_input_dim, std_output)
    params['b_output'] = [Value(0.0) for _ in range(OUTPUT_DIM)]

    return params


# Signpost: gamma (sharpening exponent) is fixed at 1.5 rather than learned.
# Making gamma a learned Value adds another deep branch to the computation
# graph (exp(gamma * log(w)) for each weight), which causes gradient instability
# with scalar autograd. The original NTM learns gamma, but with 128-dim vectors
# and matrix-level autograd, not scalar-level. Fixed gamma still demonstrates
# the sharpening concept; the key learned parameters are the key vectors and beta.
GAMMA_FIXED = 1.5


def controller_step(
    input_vec: list[Value],
    prev_read: list[Value],
    params: dict,
    prev_read_weights: list[Value],
    prev_write_weights: list[Value],
    memory: list[list[Value]],
) -> tuple:
    """Run one step of the controller: input -> addressing params + output.

    Returns (output, read_vec, read_weights, write_weights, new_memory).
    """
    # Concatenate input and previous read vector
    controller_input = input_vec + prev_read

    # Hidden layer with tanh activation
    hidden = linear_with_bias(
        controller_input, params['w_hidden'], params['b_hidden']
    )
    hidden = [h.tanh() for h in hidden]

    # --- Read head addressing ---

    # Key: what to look for in memory (unbounded vector)
    read_key = linear_with_bias(hidden, params['w_rkey'], params['b_rkey'])

    # Beta: sharpness of content-based addressing (positive scalar)
    # softplus ensures beta > 0; +1 ensures beta >= 1
    rbeta_raw = linear_with_bias(hidden, params['w_rbeta'], params['b_rbeta'])
    read_beta = softplus(rbeta_raw[0]) + Value(1.0)

    # Gate: interpolation between content and previous weights [0, 1]
    rgate_raw = linear_with_bias(hidden, params['w_rgate'], params['b_rgate'])
    read_gate = rgate_raw[0].sigmoid()

    # Shift kernel: distribution over {-1, 0, +1} via softmax
    rshift_raw = linear_with_bias(
        hidden, params['w_rshift'], params['b_rshift']
    )
    read_shift = softmax(rshift_raw)

    # Full addressing pipeline for read head
    read_weights = full_addressing(
        read_key, read_beta, read_gate, read_shift, GAMMA_FIXED,
        memory, prev_read_weights,
    )

    # Read from memory
    read_vec = read_memory(memory, read_weights)

    # --- Write head addressing ---
    # Separate addressing pipeline: the write head must target different slots
    # than the read head. During input phase, write head stores vectors while
    # read head is idle; during output phase, read head retrieves while write
    # head is idle.

    write_key = linear_with_bias(hidden, params['w_wkey'], params['b_wkey'])

    wbeta_raw = linear_with_bias(hidden, params['w_wbeta'], params['b_wbeta'])
    write_beta = softplus(wbeta_raw[0]) + Value(1.0)

    wgate_raw = linear_with_bias(hidden, params['w_wgate'], params['b_wgate'])
    write_gate = wgate_raw[0].sigmoid()

    wshift_raw = linear_with_bias(
        hidden, params['w_wshift'], params['b_wshift']
    )
    write_shift = softmax(wshift_raw)

    write_weights = full_addressing(
        write_key, write_beta, write_gate, write_shift, GAMMA_FIXED,
        memory, prev_write_weights,
    )

    # Erase vector: sigmoid ensures values in [0, 1] (fraction to erase)
    erase_raw = linear_with_bias(hidden, params['w_erase'], params['b_erase'])
    erase_vector = [e.sigmoid() for e in erase_raw]

    # Add vector: unbounded (what to write into memory)
    add_vector = linear_with_bias(hidden, params['w_add'], params['b_add'])

    # Write to memory
    new_memory = write_memory(memory, write_weights, erase_vector, add_vector)

    # --- Output ---
    # The output combines the controller's hidden state with what was just read
    # from memory. The read vector carries the retrieved information; the hidden
    # state provides context about what phase we're in (input vs output).
    output_input = hidden + read_vec
    output = linear_with_bias(
        output_input, params['w_output'], params['b_output']
    )

    return output, read_vec, read_weights, write_weights, new_memory


# === TRAINING (COPY TASK) ===

# The copy task is THE canonical benchmark for memory-augmented networks.
# It tests whether the network can:
#   1. Write a sequence of vectors to memory (store)
#   2. Read them back in the correct order (retrieve)
#
# Input format: [x1, x2, ..., xN, delimiter, zero, zero, ..., zero]
# Target:       [0,  0,  ..., 0,   0,        x1,   x2,   ..., xN  ]
#
# The delimiter signals "stop writing, start reading." The network must learn:
#   - During input phase: write each vector to a different memory slot
#   - At the delimiter: switch from writing to reading
#   - During output phase: read memory slots in the same order they were written
#
# Why this is the right benchmark: it requires both precise writing (each vector
# goes to a unique slot) and ordered reading (the shift mechanism must step
# through slots sequentially). A network that can copy has learned the basic
# primitives of a general-purpose memory system.

def generate_copy_sequence(seq_len: int, dim: int) -> list[list[float]]:
    """Generate a random sequence of binary vectors for the copy task.

    Binary vectors (0/1) make the task easier to learn and visualize.
    """
    return [
        [float(random.randint(0, 1)) for _ in range(dim)]
        for _ in range(seq_len)
    ]


def build_copy_episode(
    seq_len: int, dim: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Build input and target sequences for one copy episode.

    Input:  [x1, x2, ..., xN, delimiter, 0, 0, ..., 0]  (2N+1 steps)
    Target: [0,  0,  ..., 0,  0,         x1, x2, ..., xN]

    The zeros in the target during the input phase mean "no output expected."
    The delimiter is a vector of 0.5s (distinct from binary data vectors).
    """
    sequence = generate_copy_sequence(seq_len, dim)

    # Input: the data sequence, then delimiter, then zeros
    delimiter = [0.5] * dim
    zero_input = [0.0] * dim
    inputs = sequence + [delimiter] + [zero_input] * seq_len

    # Target: zeros during input phase + delimiter, then the sequence
    zero_target = [0.0] * dim
    targets = [zero_target] * (seq_len + 1) + sequence

    return inputs, targets


def collect_parameters(params: dict) -> list[Value]:
    """Flatten all controller parameters into a single list for SGD."""
    all_params = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, list):
            for item in val:
                if isinstance(item, list):
                    all_params.extend(item)
                elif isinstance(item, Value):
                    all_params.append(item)
        elif isinstance(val, Value):
            all_params.append(val)
    return all_params


def zero_grads(params_list: list[Value]) -> None:
    """Reset all gradients to zero before backward pass."""
    for p in params_list:
        p.grad = 0.0


def clip_gradients(params_list: list[Value], max_norm: float) -> None:
    """Clip gradients by global norm to prevent exploding gradients.

    Compute the L2 norm of all gradients concatenated into one vector.
    If it exceeds max_norm, scale all gradients down proportionally.
    This is critical for memory networks where gradients flow through many
    time steps and can grow exponentially.
    """
    total_norm_sq = sum(p.grad ** 2 for p in params_list)
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for p in params_list:
            p.grad *= scale


def compute_bit_accuracy(
    outputs: list[list[Value]], targets: list[list[float]],
    output_start: int, total_steps: int,
) -> float:
    """Compute bit-level accuracy for the output phase only."""
    correct = 0
    total_bits = 0
    for t in range(output_start, total_steps):
        for d in range(OUTPUT_DIM):
            predicted = 1.0 if outputs[t][d].data > 0.5 else 0.0
            actual = targets[t][d]
            if predicted == actual:
                correct += 1
            total_bits += 1
    return correct / total_bits if total_bits > 0 else 0.0


def train() -> tuple[dict, list[float]]:
    """Train the NTM on the copy task.

    Returns the trained controller parameters and loss history.
    """
    print("=" * 60)
    print("MEMORY-AUGMENTED NEURAL NETWORK (Neural Turing Machine)")
    print("=" * 60)
    print()
    print(f"Memory: {MEMORY_SLOTS} slots x {MEMORY_DIM} dim")
    print(f"Input/output dim: {INPUT_DIM}")
    print(f"Controller hidden: {HIDDEN_DIM}")
    print(f"Copy sequences: length {MIN_SEQ_LEN}-{MAX_SEQ_LEN}")
    print(f"Training episodes: {NUM_EPISODES}")
    print()

    params = init_controller()
    params_list = collect_parameters(params)
    num_params = len(params_list)
    print(f"Total parameters: {num_params}")
    print()

    print("=== TRAINING (COPY TASK) ===")
    print()
    print("Task: see [x1, x2, ..., xN, delim], then reproduce [x1, ..., xN]")
    print("The network must learn to write each vector to memory, then read")
    print("them back in order.")
    print()

    # Momentum buffer: one velocity per parameter, initially zero.
    # SGD with momentum: v = mu*v - lr*grad; p += v
    # Momentum smooths gradients across episodes, which is critical for NTMs
    # because each episode has different sequence length and content, causing
    # high gradient variance. Without momentum, the model oscillates.
    velocity = [0.0] * len(params_list)

    start_time = time.time()
    loss_history = []

    for episode in range(NUM_EPISODES):
        # Random sequence length
        seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
        inputs, targets = build_copy_episode(seq_len, INPUT_DIM)
        total_steps = len(inputs)
        output_start = seq_len + 1

        # Fresh memory and initial states for each episode
        memory = init_memory(MEMORY_SLOTS, MEMORY_DIM)
        prev_read = zero_vector(MEMORY_DIM)

        # Initial uniform addressing weights
        uniform_w = 1.0 / MEMORY_SLOTS
        prev_read_weights = [Value(uniform_w) for _ in range(MEMORY_SLOTS)]
        prev_write_weights = [Value(uniform_w) for _ in range(MEMORY_SLOTS)]

        zero_grads(params_list)

        # Forward pass: run controller for all time steps
        total_loss = Value(0.0)
        outputs = []

        for t in range(total_steps):
            input_vec = [Value(v) for v in inputs[t]]

            output, prev_read, prev_read_weights, prev_write_weights, memory = (
                controller_step(
                    input_vec, prev_read, params,
                    prev_read_weights, prev_write_weights, memory,
                )
            )
            outputs.append(output)

            # Binary cross-entropy loss ONLY during output phase.
            # During input phase, the network should be quiet (targets are 0),
            # but we don't penalize it — this lets gradients focus entirely on
            # the memory read-back, which is the hard part of the task.
            # BCE is better than MSE for binary targets: it penalizes confident
            # wrong predictions much more heavily (log(0.01) >> (0.99-0)^2).
            if t >= output_start:
                for d in range(OUTPUT_DIM):
                    target_val = targets[t][d]
                    pred = output[d].sigmoid()
                    if target_val > 0.5:
                        total_loss = total_loss - pred.log()
                    else:
                        total_loss = total_loss - (Value(1.0) - pred).log()

        # Average loss over output phase only
        num_elements = seq_len * OUTPUT_DIM
        avg_loss = total_loss / Value(num_elements)

        # Backward pass
        avg_loss.backward()

        # Gradient clipping
        clip_gradients(params_list, GRAD_CLIP)

        # SGD with momentum update. Linear LR decay to help convergence.
        lr = LEARNING_RATE * (1.0 - episode / NUM_EPISODES)
        for i, p in enumerate(params_list):
            velocity[i] = MOMENTUM * velocity[i] - lr * p.grad
            p.data += velocity[i]

        loss_val = avg_loss.data
        loss_history.append(loss_val)

        if (episode + 1) % 75 == 0 or episode == 0:
            elapsed = time.time() - start_time
            # Use sigmoid for accuracy since we use BCE loss
            acc_outputs = []
            for t in range(total_steps):
                acc_outputs.append(
                    [o.sigmoid() for o in outputs[t]]
                )
            accuracy = compute_bit_accuracy(
                acc_outputs, targets, output_start, total_steps,
            )
            print(
                f"  Episode {episode + 1:4d}/{NUM_EPISODES} | "
                f"loss={loss_val:.4f} | "
                f"accuracy={accuracy:.1%} | "
                f"seq_len={seq_len} | "
                f"{elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print()
    print(f"Training complete in {total_time:.1f}s")
    # Signpost: The original NTM paper trained for 100K+ episodes with RMSProp
    # on tensor-level autograd, achieving near-perfect copy accuracy. Our 1500
    # episodes with scalar autograd produce partial convergence — the model
    # learns to write to memory and produce outputs that correlate with the
    # targets, but addressing weights remain soft (not sharply peaked). This
    # is expected: the NTM addressing pipeline creates computation graphs
    # thousands of nodes deep, and scalar backprop through all of them is
    # inherently slow. The architecture and algorithm are correct; more
    # episodes with a better optimizer (Adam/RMSProp) would close the gap.
    print()

    return params, loss_history


# === INFERENCE (MEMORY VISUALIZATION) ===

def format_vector_short(vec: list[float]) -> str:
    """Format a vector as a compact binary-ish string."""
    return "".join(str(int(round(v))) for v in vec)


def run_inference(params: dict) -> None:
    """Demonstrate the trained NTM on copy tasks and visualize memory access."""

    print("=== INFERENCE (MEMORY VISUALIZATION) ===")
    print()

    for test_idx, seq_len in enumerate([2, 3, 4]):
        label = "in-distribution" if seq_len <= MAX_SEQ_LEN else "generalization"
        print(f"--- Test {test_idx + 1}: Copy length {seq_len} ({label}) ---")
        print()

        inputs, targets = build_copy_episode(seq_len, INPUT_DIM)
        total_steps = len(inputs)
        output_start = seq_len + 1

        # Initialize fresh memory and state
        memory = init_memory(MEMORY_SLOTS, MEMORY_DIM)
        prev_read = zero_vector(MEMORY_DIM)
        uniform_w = 1.0 / MEMORY_SLOTS
        prev_read_weights = [Value(uniform_w) for _ in range(MEMORY_SLOTS)]
        prev_write_weights = [Value(uniform_w) for _ in range(MEMORY_SLOTS)]

        all_read_weights = []
        all_write_weights = []
        all_outputs = []

        for t in range(total_steps):
            input_vec = [Value(v) for v in inputs[t]]
            output, prev_read, prev_read_weights, prev_write_weights, memory = (
                controller_step(
                    input_vec, prev_read, params,
                    prev_read_weights, prev_write_weights, memory,
                )
            )
            all_read_weights.append([w.data for w in prev_read_weights])
            all_write_weights.append([w.data for w in prev_write_weights])
            # Apply sigmoid to output (matching BCE training)
            all_outputs.append([o.sigmoid().data for o in output])

        # Show input sequence
        print("  Input sequence:")
        for i in range(seq_len):
            bits = format_vector_short(inputs[i])
            print(f"    t={i}: {bits}")
        print(f"    t={seq_len}: {'delim':>5}")
        print()

        # Show output vs target during the output phase
        print("  Output phase (target vs predicted):")
        correct_vecs = 0
        total_vecs = 0
        for t in range(output_start, total_steps):
            target_bits = format_vector_short(targets[t])
            pred_bits = "".join(
                str(1 if all_outputs[t][d] > 0.5 else 0)
                for d in range(OUTPUT_DIM)
            )
            match = "OK" if target_bits == pred_bits else "XX"
            if target_bits == pred_bits:
                correct_vecs += 1
            total_vecs += 1
            # Also show raw sigmoid values for insight
            raw = " ".join(f"{all_outputs[t][d]:.2f}" for d in range(OUTPUT_DIM))
            print(f"    t={t}: target={target_bits}  pred={pred_bits}  "
                  f"[{match}]  raw=({raw})")

        vec_acc = correct_vecs / total_vecs if total_vecs > 0 else 0.0
        print(f"  Vector accuracy: {correct_vecs}/{total_vecs} ({vec_acc:.0%})")
        print()

        # Visualize write addressing weights over time (input phase)
        print("  Write attention during input phase (which slot gets written):")
        print(f"  {'step':<6} {'input':<8} ", end="")
        for s in range(MEMORY_SLOTS):
            print(f"s{s:<3}", end="")
        print()
        print(f"  {'-' * (14 + MEMORY_SLOTS * 4)}")
        for t in range(seq_len + 1):
            if t < seq_len:
                phase_label = format_vector_short(inputs[t])
            else:
                phase_label = "delim"
            print(f"  t={t:<3} {phase_label:<8} ", end="")
            for s in range(MEMORY_SLOTS):
                w = all_write_weights[t][s]
                # Use block chars for visual weight indication
                if w > 0.3:
                    ch = "###"
                elif w > 0.15:
                    ch = "## "
                elif w > 0.08:
                    ch = "#  "
                else:
                    ch = ".  "
                print(f"{ch} ", end="")
            print()
        print()

        # Visualize read addressing weights over time (output phase)
        print("  Read attention during output phase (which slot gets read):")
        print(f"  {'step':<6} {'output':<8} ", end="")
        for s in range(MEMORY_SLOTS):
            print(f"s{s:<3}", end="")
        print()
        print(f"  {'-' * (14 + MEMORY_SLOTS * 4)}")
        for t in range(output_start, total_steps):
            pred_bits = "".join(
                str(1 if all_outputs[t][d] > 0.5 else 0)
                for d in range(OUTPUT_DIM)
            )
            print(f"  t={t:<3} {pred_bits:<8} ", end="")
            for s in range(MEMORY_SLOTS):
                w = all_read_weights[t][s]
                if w > 0.3:
                    ch = "###"
                elif w > 0.15:
                    ch = "## "
                elif w > 0.08:
                    ch = "#  "
                else:
                    ch = ".  "
                print(f"{ch} ", end="")
            print()
        print()

        # Show memory contents after processing (top slots by write weight)
        print("  Memory contents after sequence (first 4 dims per slot):")
        for i in range(min(6, MEMORY_SLOTS)):
            vals = [memory[i][d].data for d in range(min(4, MEMORY_DIM))]
            val_str = " ".join(f"{v:+.2f}" for v in vals)
            # Max write weight this slot received during input phase
            max_w = max(all_write_weights[t][i] for t in range(seq_len))
            print(f"    slot {i}: [{val_str}]  max_write_w={max_w:.3f}")
        print()

    # Summary
    print("=== HOW THIS CONNECTS TO MODERN AI MEMORY ===")
    print()
    print("This NTM implements the core primitive: differentiable read/write.")
    print("Modern systems build on the same foundation:")
    print()
    print("  NTM (this script)         -> Learned addressing via backprop")
    print("  Transformer self-attention -> Content addressing where memory")
    print("                               = input sequence")
    print("  MemGPT                    -> NTM-style read/write over LLM context")
    print("  ChatGPT Memory            -> Persistent key-value store with")
    print("                               learned retrieval (content addressing)")
    print("  RAG (microrag.py)         -> Read-only memory (no write head)")
    print()
    print("The progression: static retrieval (RAG) -> learned read/write (NTM)")
    print("-> persistent agent memory (MemGPT) -> user-facing memory (ChatGPT).")
    print()


# === MAIN ===

def main() -> None:
    """Run the full NTM demonstration: train on copy task, then visualize."""
    params, loss_history = train()
    run_inference(params)

    # Show loss curve summary
    print("=== LOSS CURVE ===")
    print()
    num_bins = 10
    bin_size = len(loss_history) // num_bins
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size
        avg = sum(loss_history[start:end]) / bin_size
        bar_len = int(avg * 40)
        bar_len = min(bar_len, 50)
        bar = '#' * bar_len
        print(f"  Episodes {start + 1:4d}-{end:4d}: {avg:.4f} {bar}")
    print()

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
