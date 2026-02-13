"""
How meaning becomes geometry — training vectors where distance equals similarity,
using only character n-grams and contrastive loss.
"""
# Contrastive embedding learning (InfoNCE loss) transforms sparse high-dimensional
# character n-gram features into dense low-dimensional vectors where cosine similarity
# reflects semantic similarity. Inspired by SimCLR and sentence-transformers, but
# simplified to a linear projection without the deep network machinery.

from __future__ import annotations

import math
import os
import random
import urllib.request
from collections import Counter

random.seed(42)


# === CONSTANTS ===

EMBEDDING_DIM = 32  # Target embedding dimension (sparse n-grams → dense vectors)
LEARNING_RATE = 0.05  # SGD learning rate (Adam adds overhead without much benefit here)
TEMPERATURE = 0.1  # InfoNCE temperature: lower = sharper similarity distribution
NUM_EPOCHS = 30  # Enough to see clear separation between positive/random pairs
BATCH_SIZE = 64
MAX_VOCAB = 500  # Cap n-gram vocabulary to most frequent entries for speed
TRAIN_SIZE = 5000  # Subset of names for training (full set is 32K — too slow for demo)

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: production embedding models (sentence-transformers, CLIP) use deep
# transformer encoders with 12+ layers. This linear projection demonstrates the
# core contrastive learning mechanism without the architectural complexity.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download dataset if not cached, return list of names."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


# === FEATURE EXTRACTION ===

def extract_ngrams(text: str) -> list[str]:
    """Extract character bigrams and trigrams from text.

    Why n-grams: they capture local phonetic patterns better than individual
    characters. "anna" and "anne" share bigrams "an", "nn" and trigram "ann",
    so their n-gram vectors will have high overlap even though they differ by
    one character. This is what makes n-grams sensitive to pronunciation similarity.
    """
    # Pad with boundary markers to capture start/end patterns
    padded = f"^{text}$"
    bigrams = [padded[i:i+2] for i in range(len(padded) - 1)]
    trigrams = [padded[i:i+3] for i in range(len(padded) - 2)]
    return bigrams + trigrams


def build_ngram_vocab(names: list[str], max_vocab: int) -> dict[str, int]:
    """Build vocabulary mapping most frequent n-grams to indices.

    Capping the vocabulary serves two purposes: (1) performance — the gradient
    loop is O(non_zero_ngrams * embedding_dim), and (2) quality — rare n-grams
    seen once or twice add noise without learning useful patterns.
    """
    counts: Counter[str] = Counter()
    for name in names:
        counts.update(extract_ngrams(name))

    # Keep the top max_vocab most frequent n-grams
    most_common = counts.most_common(max_vocab)
    return {ngram: idx for idx, (ngram, _) in enumerate(most_common)}


def encode_ngrams_sparse(text: str, vocab: dict[str, int]) -> dict[int, float]:
    """Convert text to sparse n-gram count dict (index → count).

    Returns only non-zero entries. This is critical for performance: names have
    ~10-15 n-grams out of a vocab of 500, so sparse representation skips 97%
    of the computation in gradient and encoder loops.
    """
    sparse: dict[int, float] = {}
    for ngram in extract_ngrams(text):
        if ngram in vocab:
            idx = vocab[ngram]
            sparse[idx] = sparse.get(idx, 0.0) + 1.0
    return sparse


# === AUGMENTATION ===

def augment(name: str) -> str:
    """Create positive pair by random character deletion or swap.

    Why augmentation: forces the encoder to learn invariances to small changes.
    If "anna" and "ana" map to similar embeddings, the model has learned that
    character deletion preserves identity — this is the contrastive learning
    principle that similar inputs should have similar representations.
    """
    if len(name) <= 2:
        return name  # too short to augment safely

    if random.random() < 0.5:
        # Delete one random character
        idx = random.randint(0, len(name) - 1)
        return name[:idx] + name[idx + 1:]
    else:
        # Swap two adjacent characters
        idx = random.randint(0, len(name) - 2)
        chars = list(name)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)


# === ENCODER ===

def l2_normalize(vec: list[float]) -> list[float]:
    """Normalize vector to unit length.

    Why L2 normalization: constrains embeddings to the unit hypersphere. After
    normalization, cosine similarity = dot product, which simplifies the math
    and makes the embedding space isotropic (all directions have equal variance).
    This is standard practice in contrastive learning (SimCLR, CLIP).
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


def encode_sparse_raw(
    sparse_ngrams: dict[int, float], W: list[list[float]]
) -> list[float]:
    """Project sparse n-gram features to embedding space WITHOUT normalization.

    Math: z = W @ x (raw, unnormalized embedding)
    Used in training where we need to backpropagate through normalization.
    """
    embedding = [0.0] * len(W)
    for i in range(len(W)):
        total = 0.0
        for j, count in sparse_ngrams.items():
            total += W[i][j] * count
        embedding[i] = total
    return embedding


def encode_sparse(
    sparse_ngrams: dict[int, float], W: list[list[float]]
) -> list[float]:
    """Project sparse n-gram features to embedding space and normalize.

    Math: emb = normalize(W @ x)
    Sparse version: only sums over non-zero entries in x, which is 10-15
    n-grams instead of the full 500-entry vocabulary.
    """
    return l2_normalize(encode_sparse_raw(sparse_ngrams, W))


def grad_through_norm(
    raw_emb: list[float], grad_normalized: list[float]
) -> list[float]:
    """Backpropagate gradient through L2 normalization.

    If z = raw_emb and e = z/||z|| (the normalized embedding), then:
        d(L)/d(z_i) = (g_i - e_i * dot(g, e)) / ||z||

    The normalization Jacobian projects out the radial component of the
    gradient, leaving only the tangential direction on the unit sphere.
    Without this projection, gradients can push all embeddings in the same
    radial direction, causing "representation collapse" — the most common
    failure mode in contrastive learning.
    """
    norm = math.sqrt(sum(x * x for x in raw_emb))
    if norm < 1e-10:
        return list(grad_normalized)
    e = [x / norm for x in raw_emb]
    g_dot_e = sum(g * ei for g, ei in zip(grad_normalized, e))
    return [(g - ei * g_dot_e) / norm for g, ei in zip(grad_normalized, e)]


# === SIMILARITY ===

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two L2-normalized vectors.

    Since vectors are L2-normalized, cosine similarity = dot product.
    Range: [-1, 1] where 1 = identical direction, -1 = opposite, 0 = orthogonal.
    """
    return sum(a[i] * b[i] for i in range(len(a)))


# === INFONCE LOSS ===

def infonce_loss_and_grads(
    anchor_embs: list[list[float]],
    positive_embs: list[list[float]],
    temperature: float,
) -> tuple[float, list[list[float]], list[list[float]]]:
    """Compute InfoNCE (NT-Xent) loss and embedding-space gradients.

    For each (anchor, positive) pair in the batch, the loss encourages high
    similarity to the positive and low similarity to all negatives (other
    samples in the batch).

    Math (for anchor i):
        sim_pos = cos(anchor_i, positive_i) / tau
        sim_neg_j = cos(anchor_i, anchor_j) / tau   for j != i
        loss_i = -log(exp(sim_pos) / (exp(sim_pos) + sum_j exp(sim_neg_j)))

    Why temperature: controls sharpness of the similarity distribution. Low tau
    (e.g. 0.1) makes the loss focus on hard negatives. tau=0.1 is standard in SimCLR.

    Returns: (avg_loss, anchor_grads, positive_grads)
    """
    bs = len(anchor_embs)
    total_loss = 0.0

    anchor_grads = [[0.0] * EMBEDDING_DIM for _ in range(bs)]
    positive_grads = [[0.0] * EMBEDDING_DIM for _ in range(bs)]

    for i in range(bs):
        # Similarity to positive pair
        sim_pos = cosine_similarity(anchor_embs[i], positive_embs[i]) / temperature

        # Similarities to all negatives (other anchors in batch)
        sim_negs = []
        for j in range(bs):
            if j != i:
                sim_negs.append(
                    cosine_similarity(anchor_embs[i], anchor_embs[j]) / temperature
                )

        # Log-sum-exp trick for numerical stability (subtract max before exp)
        max_sim = max([sim_pos] + sim_negs)
        exp_pos = math.exp(sim_pos - max_sim)
        exp_negs = [math.exp(s - max_sim) for s in sim_negs]
        denom = exp_pos + sum(exp_negs)

        # Loss: -log(softmax probability of positive pair)
        total_loss += -math.log(max(exp_pos / denom, 1e-10))

        # Gradient of loss w.r.t. anchor embedding:
        # d(loss)/d(anchor_i) = (1/tau) * (sum_j p_j * anchor_j - positive_i)
        # where p_j = exp(sim_neg_j) / denom is the softmax probability

        # Positive contribution: pushes anchor toward positive
        p_pos = exp_pos / denom
        for d in range(EMBEDDING_DIM):
            anchor_grads[i][d] += (p_pos - 1.0) / temperature * positive_embs[i][d]
            positive_grads[i][d] += (p_pos - 1.0) / temperature * anchor_embs[i][d]

        # Negative contributions: pushes anchor away from negatives
        neg_idx = 0
        for j in range(bs):
            if j == i:
                continue
            p_neg = exp_negs[neg_idx] / denom
            for d in range(EMBEDDING_DIM):
                anchor_grads[i][d] += p_neg / temperature * anchor_embs[j][d]
            neg_idx += 1

    return total_loss / bs, anchor_grads, positive_grads


# === TRAINING ===

def train(
    names: list[str],
    vocab: dict[str, int],
    W: list[list[float]],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train embedding model with SGD.

    Signpost: production systems use Adam with learning rate warmup. SGD is
    sufficient here because the model is a single linear layer — there's no
    depth to cause gradient scale issues across layers.
    """
    vocab_size = len(vocab)

    for epoch in range(num_epochs):
        epoch_names = names[:]
        random.shuffle(epoch_names)

        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(epoch_names), batch_size):
            batch = epoch_names[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            # Encode anchors and positives (sparse n-grams → dense embeddings)
            # Store both raw (pre-normalization) and normalized embeddings:
            # raw embeddings are needed for the normalization Jacobian in backprop
            anchor_sparse = []
            positive_sparse = []
            anchor_raw = []
            positive_raw = []
            anchor_embs = []
            positive_embs = []

            for name in batch:
                a_sp = encode_ngrams_sparse(name, vocab)
                anchor_sparse.append(a_sp)
                a_raw = encode_sparse_raw(a_sp, W)
                anchor_raw.append(a_raw)
                anchor_embs.append(l2_normalize(a_raw))

                p_sp = encode_ngrams_sparse(augment(name), vocab)
                positive_sparse.append(p_sp)
                p_raw = encode_sparse_raw(p_sp, W)
                positive_raw.append(p_raw)
                positive_embs.append(l2_normalize(p_raw))

            # Compute loss and gradients w.r.t. NORMALIZED embeddings
            loss, a_grads, p_grads = infonce_loss_and_grads(
                anchor_embs, positive_embs, TEMPERATURE
            )
            epoch_loss += loss
            num_batches += 1

            # Backpropagate gradients to W using SPARSE computation.
            # Chain rule: d(L)/d(W) = d(L)/d(emb_norm) * d(emb_norm)/d(emb_raw) * d(emb_raw)/d(W)
            # The normalization Jacobian (middle term) projects out the radial
            # gradient component, preventing representation collapse.
            grad_W = [[0.0] * vocab_size for _ in range(EMBEDDING_DIM)]

            for b_idx in range(len(batch)):
                # Transform gradients through normalization Jacobian
                a_grad_raw = grad_through_norm(anchor_raw[b_idx], a_grads[b_idx])
                p_grad_raw = grad_through_norm(positive_raw[b_idx], p_grads[b_idx])

                for j, count in anchor_sparse[b_idx].items():
                    for i in range(EMBEDDING_DIM):
                        grad_W[i][j] += a_grad_raw[i] * count

                for j, count in positive_sparse[b_idx].items():
                    for i in range(EMBEDDING_DIM):
                        grad_W[i][j] += p_grad_raw[i] * count

            # SGD update (only for entries with non-zero gradients)
            scale = learning_rate / len(batch)
            for i in range(EMBEDDING_DIM):
                for j in range(vocab_size):
                    if grad_W[i][j] != 0.0:
                        W[i][j] -= scale * grad_W[i][j]

        avg_loss = epoch_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3}/{num_epochs}  loss={avg_loss:.4f}")


# === INFERENCE ===

def find_nearest_neighbors(
    query: str,
    candidates: list[str],
    vocab: dict[str, int],
    W: list[list[float]],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Find k nearest neighbors by cosine similarity in embedding space."""
    q_emb = encode_sparse(encode_ngrams_sparse(query, vocab), W)

    similarities = []
    for candidate in candidates:
        if candidate == query:
            continue
        c_emb = encode_sparse(encode_ngrams_sparse(candidate, vocab), W)
        sim = cosine_similarity(q_emb, c_emb)
        similarities.append((candidate, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# === MAIN ===

if __name__ == "__main__":
    # Load data
    all_names = load_data(DATA_URL, DATA_FILE)
    print(f"Loaded {len(all_names):,} names")

    # Use a training subset for speed; keep all names for nearest-neighbor search
    train_names = all_names[:TRAIN_SIZE]
    print(f"Training on {len(train_names):,} names\n")

    # Build n-gram vocabulary from training set (capped at MAX_VOCAB)
    print("Building n-gram vocabulary...")
    vocab = build_ngram_vocab(train_names, MAX_VOCAB)
    print(f"Vocabulary: {len(vocab)} n-grams (top {MAX_VOCAB} most frequent)\n")

    # Initialize projection matrix W: [embedding_dim × vocab_size]
    W = [
        [random.gauss(0, 0.01) for _ in range(len(vocab))]
        for _ in range(EMBEDDING_DIM)
    ]
    num_params = EMBEDDING_DIM * len(vocab)
    print(f"Model: linear projection ({EMBEDDING_DIM} x {len(vocab)} = {num_params:,} params)\n")

    # Train
    print(f"Training (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, temp={TEMPERATURE})...")
    train(train_names, vocab, W, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    print()

    # === EVALUATION ===

    # Positive pairs: similar-sounding names (should have high similarity)
    positive_pairs = [
        ("anna", "anne"), ("john", "jon"), ("elizabeth", "elisabeth"),
        ("michael", "michelle"), ("alexander", "alexandra"),
    ]

    # Random pairs: dissimilar names (should have low similarity)
    random_pairs = [
        ("anna", "zachary"), ("john", "penelope"), ("elizabeth", "bob"),
        ("michael", "quinn"), ("alexander", "ivy"),
    ]

    print("Positive pairs (should be similar):")
    pos_sims = []
    for name1, name2 in positive_pairs:
        e1 = encode_sparse(encode_ngrams_sparse(name1, vocab), W)
        e2 = encode_sparse(encode_ngrams_sparse(name2, vocab), W)
        sim = cosine_similarity(e1, e2)
        pos_sims.append(sim)
        print(f"  {name1:<12} <-> {name2:<12}  sim={sim:>6.3f}")

    print("\nRandom pairs (should be dissimilar):")
    rand_sims = []
    for name1, name2 in random_pairs:
        e1 = encode_sparse(encode_ngrams_sparse(name1, vocab), W)
        e2 = encode_sparse(encode_ngrams_sparse(name2, vocab), W)
        sim = cosine_similarity(e1, e2)
        rand_sims.append(sim)
        print(f"  {name1:<12} <-> {name2:<12}  sim={sim:>6.3f}")

    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_rand = sum(rand_sims) / len(rand_sims)
    print(f"\nAverage positive pair similarity: {avg_pos:.3f}")
    print(f"Average random pair similarity:   {avg_rand:.3f}")

    # Nearest neighbor retrieval demo
    # Search over a larger pool for more interesting results
    search_pool = all_names[:10000]
    query_names = ["anna", "john", "elizabeth", "michael"]
    print("\nNearest neighbor retrieval:")
    for query in query_names:
        neighbors = find_nearest_neighbors(query, search_pool, vocab, W, k=5)
        neighbor_str = ", ".join(f"{n} ({s:.2f})" for n, s in neighbors)
        print(f"  {query:<12} -> {neighbor_str}")
