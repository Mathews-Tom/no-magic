"""
Every attention mechanism that matters, side by side: how MHA, GQA, MQA, and sliding
window trade off memory, compute, and representational power on the same input.
"""
# Reference: Vaswani et al., "Attention Is All You Need" (2017) for scaled dot-product
# and multi-head attention. Shazeer, "Fast Transformer Decoding" (2019) for multi-query.
# Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
# Multi-Head Checkpoints" (2023). Beltagy et al., "Longformer" (2020) for sliding window.

# === TRADEOFFS ===
# + MHA: maximum representational power with independent head projections
# + GQA: reduces KV-cache memory by sharing key/value heads across query groups
# + MQA: minimal KV-cache (single KV head) for maximum inference throughput
# - MHA: KV-cache scales linearly with head count (memory bottleneck at long contexts)
# - MQA: quality degrades from extreme KV sharing (fewer distinct attention patterns)
# - Sliding window: loses global context beyond the window boundary
# WHEN TO USE: MHA for training quality, GQA for balanced serving, MQA for
#   extreme throughput, sliding window for very long sequences.
# WHEN NOT TO: MQA when quality is non-negotiable, sliding window when global
#   context is required, MHA when KV-cache memory is the deployment bottleneck.

from __future__ import annotations

import math
import random
import time

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

SEQ_LEN = 32
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # = 16
N_KV_HEADS_GQA = 2
WINDOW_SIZE = 8

# Signpost: production transformers use d_model=4096+, 32-128 heads, seq_len=8192+.
# These toy dimensions preserve every algorithmic detail while staying fast.


# === HELPER FUNCTIONS ===
# Matrix operations on plain Python lists-of-lists -- the linear algebra
# primitives that attention is built from.

def rand_matrix(rows: int, cols: int) -> list[list[float]]:
    """Random matrix with Xavier-like 1/sqrt(cols) scaling to prevent softmax saturation."""
    s = 1.0 / math.sqrt(cols)
    return [[random.gauss(0, s) for _ in range(cols)] for _ in range(rows)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """A[m,k] @ B[k,n] -> C[m,n]. The dominant cost in every attention variant."""
    k = len(a[0])
    n = len(b[0])
    # Pre-transpose B for row-contiguous inner loop access
    bt = [[b[r][c] for r in range(k)] for c in range(n)]
    return [[sum(a[i][p] * bt[j][p] for p in range(k)) for j in range(n)]
            for i in range(len(a))]


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


def softmax_row(row: list[float]) -> list[float]:
    """Stable softmax: subtract max to prevent exp() overflow.
    exp(x-c)/sum(exp(x_j-c)) = exp(x)/sum(exp(x_j)) for any constant c."""
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def flatten(m: list[list[float]]) -> list[float]:
    return [v for row in m for v in row]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Directional agreement between vectors, ignoring magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def avg_head_weights(
    w: list[list[float]], head_dim: int, n_heads: int, n_kv_heads: int
) -> list[list[float]]:
    """Derive reduced KV weights by averaging MHA head groups (Ainslie et al. 2023).
    For GQA/MQA conversion from MHA, mean-pool KV columns within each group."""
    gs = n_heads // n_kv_heads
    d = len(w)
    kv_dim = n_kv_heads * head_dim
    result = [[0.0] * kv_dim for _ in range(d)]
    for r in range(d):
        for g in range(n_kv_heads):
            for c in range(head_dim):
                result[r][g * head_dim + c] = sum(
                    w[r][(g * gs + h) * head_dim + c] for h in range(gs)
                ) / gs
    return result


# === ATTENTION VARIANTS ===

def vanilla_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> list[list[float]]:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V

    The sqrt(d_k) scaling is critical: without it, dot products grow proportional
    to d_k, pushing softmax into saturation (near-one-hot outputs). Dividing by
    sqrt(d_k) keeps variance at ~1.0 so softmax produces useful distributions."""
    scale = 1.0 / math.sqrt(len(q[0]))
    # scores[i][j] = how much position i attends to position j
    scores = [[v * scale for v in row] for row in matmul(q, transpose(k))]
    weights = [softmax_row(row) for row in scores]
    return matmul(weights, v)


def multi_head_attention(
    x: list[list[float]], w_q: list[list[float]], w_k: list[list[float]],
    w_v: list[list[float]], w_o: list[list[float]], n_heads: int,
) -> list[list[float]]:
    """MHA(X) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)

    Why multiple heads? Each head learns a different notion of "relevance" --
    syntactic, semantic, positional. Total attention FLOPs equal single-head;
    we just partition d_model across heads. The benefit is representational."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads
    q_full, k_full, v_full = matmul(x, w_q), matmul(x, w_k), matmul(x, w_v)

    heads = []
    for h in range(n_heads):
        lo, hi = h * hd, (h + 1) * hd
        heads.append(vanilla_attention(
            [r[lo:hi] for r in q_full], [r[lo:hi] for r in k_full],
            [r[lo:hi] for r in v_full]))

    # Concatenate heads, then project. W_O is where cross-head mixing happens.
    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def grouped_query_attention(
    x: list[list[float]], w_q: list[list[float]], w_k_r: list[list[float]],
    w_v_r: list[list[float]], w_o: list[list[float]],
    n_heads: int, n_kv_heads: int,
) -> list[list[float]]:
    """GQA: partition query heads into groups sharing KV projections.
    With 4 query heads and 2 KV heads, heads 0-1 share one KV pair and 2-3 another.

    Why this works: in trained MHA models, KV representations across heads are
    highly correlated. GQA exploits this redundancy -- sharing KV heads loses
    little quality while cutting KV cache memory proportionally.
    LLaMA 2 70B uses GQA with 8 KV heads for 64 query heads."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads
    gs = n_heads // n_kv_heads  # query heads per KV group

    q_full = matmul(x, w_q)
    k_r, v_r = matmul(x, w_k_r), matmul(x, w_v_r)

    heads = []
    for h in range(n_heads):
        q_lo, q_hi = h * hd, (h + 1) * hd
        g = h // gs  # KV group index
        kv_lo, kv_hi = g * hd, (g + 1) * hd
        heads.append(vanilla_attention(
            [r[q_lo:q_hi] for r in q_full], [r[kv_lo:kv_hi] for r in k_r],
            [r[kv_lo:kv_hi] for r in v_r]))

    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def multi_query_attention(
    x: list[list[float]], w_q: list[list[float]], w_k_s: list[list[float]],
    w_v_s: list[list[float]], w_o: list[list[float]], n_heads: int,
) -> list[list[float]]:
    """MQA: all query heads share a single KV head (GQA with n_kv=1).

    During autoregressive decoding, KV cache scales as O(layers*heads*seq*hd).
    MQA reduces this to O(layers*seq*hd) -- a factor-of-heads reduction.
    PaLM, Falcon, and StarCoder use MQA for this memory saving.
    Tradeoff: all heads see one KV view, reducing representational diversity."""
    seq_len = len(x)
    hd = len(x[0]) // n_heads

    q_full = matmul(x, w_q)
    k_s, v_s = matmul(x, w_k_s), matmul(x, w_v_s)  # [seq_len, head_dim]

    heads = []
    for h in range(n_heads):
        lo, hi = h * hd, (h + 1) * hd
        # Every head reuses the same K, V -- the entire MQA idea
        heads.append(vanilla_attention([r[lo:hi] for r in q_full], k_s, v_s))

    concat = [[c for h in heads for c in h[i]] for i in range(seq_len)]
    return matmul(concat, w_o)


def sliding_window_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]],
    window_size: int,
) -> list[list[float]]:
    """Each position attends only to its w nearest predecessors: O(n*w) not O(n^2).

    Why local attention works: most relevant context is nearby. Full attention
    over 100K tokens wastes compute on distant, irrelevant positions.

    Signpost: Mistral 7B uses sliding window (w=4096) interleaved with full
    attention layers. Longformer adds sparse global tokens. Pure local here."""
    d_k = len(q[0])
    scale = 1.0 / math.sqrt(d_k)
    output = []
    for i in range(len(q)):
        start = max(0, i - window_size + 1)
        # Scores computed only within window -- positions outside are never
        # evaluated, not just masked. This is the compute saving.
        scores = [sum(q[i][d] * k[j][d] for d in range(d_k)) * scale
                  for j in range(start, i + 1)]
        weights = softmax_row(scores)
        row = [0.0] * d_k
        for idx, j in enumerate(range(start, i + 1)):
            for d in range(d_k):
                row[d] += weights[idx] * v[j][d]
        output.append(row)
    return output


# === FLOP AND MEMORY ANALYSIS ===
# FLOPs: multiply-add = 2 FLOPs. Memory: peak floats for scores / KV cache.
#
# Core formulas:
#   Vanilla:  4*n^2*d  (QK^T + attn@V, no projections)
#   MHA:      8*n*d^2 + 4*n^2*d  (projections + attention)
#   GQA:      saves on KV projection (2*n*d*(nkv*hd) instead of 2*n*d*d for K,V)
#   MQA:      minimal KV projection (2*n*d*hd for K,V)
#   Window:   8*n*d^2 + 4*n*w*d  (projections + local attention)

def compute_analysis(
    n: int, d: int, h: int, hd: int, nkv: int, w: int
) -> list[tuple[str, int, int]]:
    """Return (name, flops, memory) for each variant."""
    return [
        ("Vanilla (single-head)",
         4 * n * n * d,
         n * n),  # full n x n score matrix

        (f"Multi-Head ({h} heads)",
         8 * n * d * d + 4 * n * n * d,
         n * n),  # sequential: one head's scores at a time

        (f"GQA ({h}q, {nkv}kv heads)",
         2*n*d*d + 2*2*n*d*(nkv*hd) + 2*n*d*d + 4*n*n*d,
         2 * nkv * n * hd),  # KV cache: the primary saving

        (f"MQA ({h}q, 1kv head)",
         2*n*d*d + 2*2*n*d*hd + 2*n*d*d + 4*n*n*d,
         2 * n * hd),  # single KV head

        (f"Sliding Window (w={w})",
         8 * n * d * d + 4 * n * w * d,
         n * w),  # w scores per position instead of n
    ]


# === MAIN: RUN ALL VARIANTS AND COMPARE ===


def run_attention_comparison(
    seq_len: int, d_model: int, n_heads: int, n_kv_heads_gqa: int, window_size: int
) -> None:
    """Run all attention variants with the given parameters and print comparison."""
    head_dim = d_model // n_heads

    print("=== Attention Variants Comparison ===\n")
    print(f"Config: seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, "
          f"head_dim={head_dim}, n_kv_heads_gqa={n_kv_heads_gqa}, "
          f"window_size={window_size}\n")

    random.seed(42)
    x = rand_matrix(seq_len, d_model)
    w_q = rand_matrix(d_model, d_model)
    w_k = rand_matrix(d_model, d_model)
    w_v = rand_matrix(d_model, d_model)
    w_o = rand_matrix(d_model, d_model)

    # Derive reduced KV weights by averaging MHA head groups (Ainslie et al. 2023).
    # This mirrors how GQA models are initialized from MHA checkpoints, ensuring
    # GQA/MQA outputs approximate MHA rather than being unrelated random projections.
    w_k_gqa = avg_head_weights(w_k, head_dim, n_heads, n_kv_heads_gqa)
    w_v_gqa = avg_head_weights(w_v, head_dim, n_heads, n_kv_heads_gqa)
    w_k_mqa = avg_head_weights(w_k, head_dim, n_heads, 1)
    w_v_mqa = avg_head_weights(w_v, head_dim, n_heads, 1)

    results: dict[str, tuple[list[list[float]], float]] = {}

    def run(name: str, fn, *args) -> None:
        t0 = time.time()
        out = fn(*args)
        results[name] = (out, (time.time() - t0) * 1000)

    run("Vanilla (single-head)", vanilla_attention, x, x, x)
    run(f"Multi-Head ({n_heads} heads)",
        multi_head_attention, x, w_q, w_k, w_v, w_o, n_heads)
    run(f"GQA ({n_heads}q, {n_kv_heads_gqa}kv heads)",
        grouped_query_attention, x, w_q, w_k_gqa, w_v_gqa, w_o,
        n_heads, n_kv_heads_gqa)
    run(f"MQA ({n_heads}q, 1kv head)",
        multi_query_attention, x, w_q, w_k_mqa, w_v_mqa, w_o, n_heads)

    # Sliding window: project with full MHA weights, then restrict attention to
    # a local window. This isolates the windowing effect from projection differences.
    q_p, k_p, v_p = matmul(x, w_q), matmul(x, w_k), matmul(x, w_v)
    t0 = time.time()
    sw = sliding_window_attention(q_p, k_p, v_p, window_size)
    sw_out = matmul(sw, w_o)
    results[f"Sliding Window (w={window_size})"] = (sw_out, (time.time() - t0) * 1000)

    # Validate: no NaN or Inf in any output
    all_valid = True
    for name, (out, _) in results.items():
        flat = flatten(out)
        if any(math.isnan(v) or math.isinf(v) for v in flat):
            print(f"  WARNING: {name} has numerical issues")
            all_valid = False
    print(f"Numerical validity: {'all outputs clean' if all_valid else 'ISSUES DETECTED'}\n")

    # Cosine similarity to MHA -- measures how much information each variant
    # preserves relative to full multi-head attention
    mha_key = f"Multi-Head ({n_heads} heads)"
    mha_flat = flatten(results[mha_key][0])
    sims = {n: cosine_sim(flatten(o), mha_flat) for n, (o, _) in results.items()}

    # Analytical cost model
    analysis = compute_analysis(seq_len, d_model, n_heads, head_dim,
                                n_kv_heads_gqa, window_size)

    # --- Print comparison table ---
    hdr = (f"{'Variant':<28} {'FLOPs':>12} {'Memory':>10} "
           f"{'Cos Sim':>10} {'Time(ms)':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, flops, mem in analysis:
        cs = sims.get(name, 0.0)
        ms = results[name][1] if name in results else 0.0
        print(f"{name:<28} {flops:>12,} {mem:>10,} {cs:>10.4f} {ms:>10.2f}")

    # --- Takeaways ---
    print("\n=== Key Takeaways ===\n")
    print("1. MHA and vanilla share attention FLOPs (4*n^2*d). MHA adds projection")
    print("   cost (8*n*d^2) but gains multiple learned attention patterns.")
    print(f"2. GQA cuts KV memory {n_heads // n_kv_heads_gqa}x "
          f"({n_heads}->{n_kv_heads_gqa} KV heads), output stays close to MHA.")
    print(f"3. MQA cuts KV memory {n_heads}x ({n_heads}->1 KV head) -- max savings,")
    print("   more quality loss because all heads share one KV view.")
    print(f"4. Sliding window (w={window_size}) makes attention O(n*w) not O(n^2),")
    print(f"   {seq_len // window_size}x cheaper at seq_len={seq_len}. "
          "Works when locality dominates.")
    print("\nProduction systems compose these: Mistral uses sliding window + GQA,")
    print("LLaMA 2 uses GQA, PaLM/Falcon use MQA. Choose based on whether your")
    print("bottleneck is compute (window), memory (MQA/GQA), or neither (full MHA).")


# === INTERACTIVE MODE ===
# Optional functionality: allows parameter exploration without editing the script.
# Activated only via --interactive flag; default behavior is unchanged.

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MHA, GQA, MQA, and sliding window attention variants"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive mode to modify parameters and re-run comparisons"
    )
    return parser.parse_args()


def interactive_loop() -> None:
    """Interactive parameter exploration mode."""
    print("\n=== INTERACTIVE MODE ===")
    print("Modify parameters and re-run the attention comparison.")
    print("Type 'quit' to exit.\n")

    params = {
        'n_heads': N_HEADS,
        'seq_len': SEQ_LEN,
        'd_model': D_MODEL,
        'n_kv_heads_gqa': N_KV_HEADS_GQA,
        'window_size': WINDOW_SIZE,
    }

    while True:
        print("Current parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        print(f"  head_dim = {params['d_model'] // params['n_heads']}  (derived)")

        user_input = input(
            "\nParameter to change (or 'run' to execute, 'quit' to exit): "
        ).strip().lower()

        if user_input == 'quit':
            break
        elif user_input == 'run':
            # Validate d_model is divisible by n_heads
            if params['d_model'] % params['n_heads'] != 0:
                print(f"ERROR: d_model ({params['d_model']}) must be divisible "
                      f"by n_heads ({params['n_heads']})")
                continue
            if params['n_heads'] % params['n_kv_heads_gqa'] != 0:
                print(f"ERROR: n_heads ({params['n_heads']}) must be divisible "
                      f"by n_kv_heads_gqa ({params['n_kv_heads_gqa']})")
                continue
            run_attention_comparison(
                params['seq_len'], params['d_model'], params['n_heads'],
                params['n_kv_heads_gqa'], params['window_size']
            )
        elif '=' in user_input:
            key, _, val = user_input.partition('=')
            key = key.strip()
            val = val.strip()
            if key not in params:
                print(f"Unknown parameter: {key}")
                print(f"Available: {', '.join(params)}")
                continue
            try:
                params[key] = int(val)
            except ValueError:
                print(f"Invalid integer: {val}")
        else:
            print("Enter 'parameter=value', 'run', or 'quit'.")


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        interactive_loop()
    else:
        # === DEFAULT BEHAVIOR (unchanged) ===
        run_attention_comparison(SEQ_LEN, D_MODEL, N_HEADS, N_KV_HEADS_GQA, WINDOW_SIZE)
