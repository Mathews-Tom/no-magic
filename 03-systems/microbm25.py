"""
The evolution of text retrieval scoring: from raw term frequency through TF-IDF to BM25,
showing how each refinement fixes a specific flaw in its predecessor.
"""
# Reference: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and
# Beyond" (2009). Original BM25: Robertson et al., TREC-3 (1994).
# TF-IDF origins: Sparck Jones, "A statistical interpretation of term specificity" (1972).

# === TRADEOFFS ===
# + No training required — purely statistical, deterministic scoring
# + Extremely fast: sparse term matching is O(|query| × avg_doc_length)
# + Battle-tested: default ranking function in Elasticsearch, Lucene, Solr
# - Bag-of-words: ignores word order and semantic meaning entirely
# - Cannot handle synonyms or paraphrases (exact lexical matching only)
# - k1/b parameters need corpus-specific tuning for optimal results
# WHEN TO USE: First-stage retrieval in search systems, keyword search,
#   document ranking when you need interpretable, fast scoring.
# WHEN NOT TO: Semantic search where meaning matters more than keywords,
#   multilingual search, or queries requiring reasoning over content.

from __future__ import annotations

import math
import random
import string

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

# BM25 standard parameters from Robertson et al. These values are the de facto
# defaults across Elasticsearch, Lucene, and most IR libraries. They were empirically
# tuned on TREC collections in the 1990s and remain surprisingly robust.
K1_DEFAULT = 1.2   # TF saturation — how fast term frequency returns diminish
B_DEFAULT = 0.75   # length normalization — 0.75 is a moderate penalty for long docs

# Parameter exploration ranges
K1_VALUES = [0.0, 0.5, 1.2, 2.0, 5.0]  # from binary to near-raw-TF
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # from no normalization to full normalization

# Number of top results to display per query
TOP_K = 5


# === SYNTHETIC CORPUS ===
# BM25 needs multi-word documents with overlapping vocabulary to demonstrate meaningfully.
# A synthetic corpus lets us control document length variation, term overlap, and topic
# distribution — exactly the factors that differentiate TF, TF-IDF, and BM25.

def build_corpus() -> tuple[list[str], list[str]]:
    """Build a synthetic corpus of short documents about ML/CS topics.

    Returns (documents, queries) where documents have deliberate properties:
    - Varying lengths (short vs long) to expose length bias in raw TF
    - Repeated terms in some docs to expose unbounded TF in TF-IDF
    - Common terms across many docs to demonstrate IDF's value
    - Topic clusters to show meaningful retrieval
    """
    documents = [
        # --- Neural network cluster ---
        "neural networks learn by adjusting weights through backpropagation gradient descent",
        "deep neural networks stack many layers to learn hierarchical representations of data",
        ("training deep neural networks requires large datasets and careful tuning of the "
         "learning rate and other hyperparameters for the neural network to converge"),
        "convolutional neural networks apply filters to detect spatial patterns in images",
        "recurrent neural networks process sequences by maintaining hidden state across time steps",

        # --- Transformer / attention cluster ---
        "transformers use self attention to model long range dependencies in sequences",
        "attention mechanisms compute weighted sums where weights reflect token relevance",
        ("the transformer architecture replaced recurrent models because attention allows "
         "parallel processing of all positions in the sequence simultaneously"),
        "multi head attention lets the model attend to different representation subspaces",

        # --- Search / retrieval cluster ---
        "search engines rank documents by relevance using scoring functions like bm25",
        "information retrieval systems index documents and match query terms efficiently",
        ("inverted indexes map each term to the list of documents containing it enabling "
         "fast lookup during query processing in search systems"),
        "relevance ranking combines term frequency and inverse document frequency for scoring",

        # --- Optimization cluster ---
        "gradient descent minimizes loss by following the negative gradient direction",
        "stochastic gradient descent samples mini batches to approximate the full gradient",
        "adam optimizer combines momentum and adaptive learning rates for faster convergence",
        "learning rate schedules reduce the step size over training to fine tune convergence",

        # --- Deliberately long document (tests length normalization) ---
        ("machine learning is a broad field that encompasses supervised learning unsupervised "
         "learning and reinforcement learning where supervised learning uses labeled data to "
         "train models that predict outputs from inputs and unsupervised learning discovers "
         "hidden patterns in data without labels and reinforcement learning trains agents to "
         "maximize cumulative reward through trial and error in an environment and all of "
         "these approaches can use neural networks as the underlying model architecture"),

        # --- Deliberately short documents (tests length normalization from the other side) ---
        "gradient descent optimization",
        "neural network training",
        "attention mechanism overview",

        # --- Document with heavy term repetition (tests TF saturation) ---
        ("search search search retrieval retrieval retrieval ranking ranking ranking "
         "search retrieval ranking search retrieval ranking documents documents"),
    ]

    queries = [
        "neural network training",
        "attention mechanism in transformers",
        "search ranking documents",
        "gradient descent optimization",
        "learning rate convergence",
    ]

    return documents, queries


# === TOKENIZATION ===
# Minimal whitespace tokenizer with punctuation stripping and lowercasing.
# Production systems use subword tokenization (BPE, WordPiece) but for BM25
# the standard practice is word-level tokens — BM25 was designed for this.

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    # Remove punctuation — in production you'd handle hyphens, apostrophes,
    # and Unicode normalization more carefully
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# === RAW TERM FREQUENCY SCORING ===
# The simplest retrieval scoring: count how many times each query term appears
# in the document. Sum across all query terms.
#
# Fatal flaw: longer documents accumulate more term counts simply by having more
# words. A 1000-word document mentioning "neural" 5 times beats a 20-word
# document about neural networks — even though the short doc is more focused.

def raw_tf_score(query_terms: list[str], doc_tokens: list[str]) -> float:
    """Score = Σ count(term, doc) for each query term."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    score = 0.0
    for term in query_terms:
        score += doc_counts.get(term, 0)
    return score


def raw_tf_score_breakdown(
    query_terms: list[str], doc_tokens: list[str]
) -> dict[str, float]:
    """Per-term score breakdown for raw TF."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    breakdown: dict[str, float] = {}
    for term in query_terms:
        breakdown[term] = float(doc_counts.get(term, 0))
    return breakdown


# === TF-IDF SCORING ===
# Improvement over raw TF: multiply by inverse document frequency to downweight
# terms that appear in many documents (common words carry less signal).
#
# Math-to-code mapping:
#   TF(t, d)  = log(1 + count(t, d))     — log-sublinear TF
#   IDF(t)    = log(N / df(t))            — classic IDF
#   Score     = Σ TF(t, d) × IDF(t)
#
# Why log(1 + tf) instead of raw tf? Diminishing returns — a term appearing 20
# times isn't 20x more relevant than once. But this is still unbounded: log(1 + 1000)
# is much larger than log(1 + 1), so extreme repetition still dominates.
#
# Why IDF alone isn't enough: IDF handles common-vs-rare but doesn't address
# document length. A 500-word document still has more chances to contain query terms.

def compute_idf_classic(
    corpus_tokens: list[list[str]], num_docs: int
) -> dict[str, float]:
    """IDF(t) = log(N / df(t)) where df(t) = number of docs containing term t.

    Classic IDF can produce very large values for rare terms and is undefined
    for terms not in the corpus (df=0). BM25's IDF fixes both issues.
    """
    df: dict[str, int] = {}
    for doc_tokens in corpus_tokens:
        seen: set[str] = set()
        for token in doc_tokens:
            if token not in seen:
                df[token] = df.get(token, 0) + 1
                seen.add(token)

    idf: dict[str, float] = {}
    for term, freq in df.items():
        # Guard against log(0) — in practice df > 0 for terms in the corpus
        idf[term] = math.log(num_docs / freq)
    return idf


def tfidf_score(
    query_terms: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
) -> float:
    """Score = Σ log(1 + tf(t, d)) × IDF(t) for each query term."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = doc_counts.get(term, 0)
        if tf == 0:
            continue
        # Log-sublinear TF: softens the impact of high frequency but doesn't bound it
        score += math.log(1 + tf) * idf.get(term, 0.0)
    return score


def tfidf_score_breakdown(
    query_terms: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
) -> dict[str, float]:
    """Per-term score breakdown for TF-IDF."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    breakdown: dict[str, float] = {}
    for term in query_terms:
        tf = doc_counts.get(term, 0)
        if tf == 0:
            breakdown[term] = 0.0
        else:
            breakdown[term] = math.log(1 + tf) * idf.get(term, 0.0)
    return breakdown


# === BM25 SCORING ===
# The culmination: fixes both TF-IDF flaws simultaneously.
#
# Math-to-code mapping:
#   IDF(t)     = log((N - df + 0.5) / (df + 0.5) + 1)
#   TF_sat(t,d) = (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl/avgdl))
#   BM25(q, d) = Σ_{t ∈ q} IDF(t) × TF_sat(t, d)
#
# Key insight 1 — TF saturation:
#   As tf → ∞, TF_sat → (k1 + 1). The score has a ceiling.
#   k1 controls how fast we approach it:
#     k1 = 0  → binary model (term present or not, frequency ignored)
#     k1 = 1.2 → standard: moderate saturation, first few occurrences matter most
#     k1 → ∞  → raw TF (no saturation, back to the original problem)
#
# Key insight 2 — length normalization via b:
#   The denominator includes k1 × (1 - b + b × dl/avgdl).
#   b = 0  → no length normalization (treats all doc lengths equally)
#   b = 0.75 → moderate: long docs penalized but not crippled
#   b = 1  → full normalization: score scales inversely with doc length
#   The "1 - b + b × dl/avgdl" term is a weighted interpolation between
#   1 (no normalization) and dl/avgdl (full normalization).
#
# Key insight 3 — IDF smoothing:
#   BM25's IDF adds 0.5 to numerator and denominator. This prevents:
#   - Negative IDF for terms appearing in more than half the corpus
#   - Division by zero for terms not in any document
#   The +1 outside the log ensures IDF is always non-negative.

def compute_idf_bm25(
    corpus_tokens: list[list[str]], num_docs: int
) -> dict[str, float]:
    """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1).

    Unlike classic IDF which can produce negative values for terms in >50%
    of documents, BM25's formulation is always non-negative. The +0.5
    smoothing (Laplace-like) also prevents extreme values for very rare terms.
    """
    df: dict[str, int] = {}
    for doc_tokens in corpus_tokens:
        seen: set[str] = set()
        for token in doc_tokens:
            if token not in seen:
                df[token] = df.get(token, 0) + 1
                seen.add(token)

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
    return idf


def bm25_score(
    query_terms: list[str],
    doc_tokens: list[str],
    doc_length: int,
    avg_doc_length: float,
    idf: dict[str, float],
    k1: float = K1_DEFAULT,
    b: float = B_DEFAULT,
) -> float:
    """BM25(q, d) = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × dl/avgdl))."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    # Length normalization factor, precomputed for the document.
    # When dl = avgdl, norm = 1 (average-length docs are unaffected).
    # When dl > avgdl, norm > 1 (long docs are penalized: TF needs to be higher to score).
    # When dl < avgdl, norm < 1 (short docs get a boost: even modest TF scores well).
    norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0

    score = 0.0
    for term in query_terms:
        tf = doc_counts.get(term, 0)
        if tf == 0:
            continue
        # Saturating TF: approaches (k1 + 1) as tf → ∞
        # For k1=1.2: tf=1 → 1.09, tf=5 → 1.71, tf=100 → 2.18, limit → 2.2
        tf_saturated = (tf * (k1 + 1)) / (tf + k1 * norm)
        score += idf.get(term, 0.0) * tf_saturated
    return score


def bm25_score_breakdown(
    query_terms: list[str],
    doc_tokens: list[str],
    doc_length: int,
    avg_doc_length: float,
    idf: dict[str, float],
    k1: float = K1_DEFAULT,
    b: float = B_DEFAULT,
) -> dict[str, float]:
    """Per-term BM25 score breakdown showing each query term's contribution."""
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0

    breakdown: dict[str, float] = {}
    for term in query_terms:
        tf = doc_counts.get(term, 0)
        if tf == 0:
            breakdown[term] = 0.0
        else:
            tf_saturated = (tf * (k1 + 1)) / (tf + k1 * norm)
            breakdown[term] = idf.get(term, 0.0) * tf_saturated
    return breakdown


# === TF SATURATION CURVE ===
# Demonstrates the core mathematical difference between TF-IDF and BM25.
# For a fixed document of average length, shows how score grows with term frequency.

def print_saturation_curve(k1: float = K1_DEFAULT, b: float = B_DEFAULT) -> None:
    """Print TF saturation comparison: log(1+tf) vs BM25's bounded formula."""
    print("\n" + "=" * 70)
    print("TF SATURATION CURVE: TF-IDF vs BM25")
    print(f"(k1={k1}, b={b}, document at average length so dl/avgdl=1.0)")
    print("=" * 70)
    print(f"{'tf':>4}  {'log(1+tf)':>10}  {'BM25 TF':>10}  {'BM25 limit':>12}")
    print("-" * 42)

    # BM25 TF limit as tf → ∞ is (k1 + 1)
    bm25_limit = k1 + 1
    # norm = 1 when dl = avgdl and b doesn't matter
    norm = 1.0

    for tf in [1, 2, 3, 5, 10, 20, 50, 100, 500]:
        log_tf = math.log(1 + tf)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * norm)
        print(f"{tf:>4}  {log_tf:>10.3f}  {bm25_tf:>10.3f}  {bm25_limit:>10.3f}")

    # The takeaway: log(1+tf) keeps growing forever. BM25's TF plateaus.
    # At tf=500, log(1+500)=6.22 while BM25 TF is ~2.20 (for k1=1.2).
    # This means a document spamming a term 500 times only scores ~2x a doc
    # with that term once, not 6x — a much more reasonable assumption about relevance.


# === PARAMETER EXPLORATION ===
# Vary k1 and b to show their independent effects on ranking.

def explore_k1_effect(
    query: str,
    corpus_tokens: list[list[str]],
    doc_lengths: list[int],
    avg_doc_length: float,
    idf: dict[str, float],
    documents: list[str],
) -> None:
    """Show how k1 changes rankings by controlling TF saturation speed."""
    print("\n" + "=" * 70)
    print(f"K1 EXPLORATION (b={B_DEFAULT} fixed)")
    print(f"Query: \"{query}\"")
    print("=" * 70)
    # k1=0 is a pure binary model: only presence/absence matters
    # k1→∞ approaches raw TF: repetition dominates
    print("k1=0: binary (term present or not)")
    print("k1=1.2: standard (moderate saturation)")
    print("k1→∞: approaches raw TF (repetition dominates)\n")

    query_terms = tokenize(query)

    for k1 in K1_VALUES:
        scores: list[tuple[int, float]] = []
        for doc_id in range(len(corpus_tokens)):
            s = bm25_score(
                query_terms, corpus_tokens[doc_id], doc_lengths[doc_id],
                avg_doc_length, idf, k1=k1, b=B_DEFAULT,
            )
            scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"  k1={k1:<4}  top-3: ", end="")
        for rank, (doc_id, s) in enumerate(scores[:3]):
            # Truncate doc text for display
            snippet = documents[doc_id][:50]
            print(f"[{doc_id}]{s:.2f}", end="  " if rank < 2 else "")
        print()


def explore_b_effect(
    query: str,
    corpus_tokens: list[list[str]],
    doc_lengths: list[int],
    avg_doc_length: float,
    idf: dict[str, float],
    documents: list[str],
) -> None:
    """Show how b changes rankings by controlling length normalization."""
    print("\n" + "=" * 70)
    print(f"B EXPLORATION (k1={K1_DEFAULT} fixed)")
    print(f"Query: \"{query}\"")
    print("=" * 70)
    print("b=0: no length normalization (long docs have advantage)")
    print("b=0.75: standard (moderate length penalty)")
    print("b=1: full normalization (short docs strongly favored)\n")

    query_terms = tokenize(query)

    # Show document lengths for context
    print("  Document lengths: ", end="")
    for doc_id in range(len(corpus_tokens)):
        print(f"[{doc_id}]={doc_lengths[doc_id]}", end=" ")
    print(f"\n  Average length: {avg_doc_length:.1f}\n")

    for b in B_VALUES:
        scores: list[tuple[int, float]] = []
        for doc_id in range(len(corpus_tokens)):
            s = bm25_score(
                query_terms, corpus_tokens[doc_id], doc_lengths[doc_id],
                avg_doc_length, idf, k1=K1_DEFAULT, b=b,
            )
            scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"  b={b:<4}  top-3: ", end="")
        for rank, (doc_id, s) in enumerate(scores[:3]):
            print(f"[{doc_id}]{s:.2f}", end="  " if rank < 2 else "")
        print()


# === THREE-WAY COMPARISON ===
# Score every document with all three methods and show how rankings differ.

def compare_methods(
    queries: list[str],
    documents: list[str],
    corpus_tokens: list[list[str]],
    doc_lengths: list[int],
    avg_doc_length: float,
    idf_classic: dict[str, float],
    idf_bm25: dict[str, float],
) -> None:
    """Run each query against all three scoring methods and compare top results."""
    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON: Raw TF vs TF-IDF vs BM25")
    print("=" * 70)

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)

        query_terms = tokenize(query)

        # Score all documents with each method
        tf_scores: list[tuple[int, float]] = []
        tfidf_scores: list[tuple[int, float]] = []
        bm25_scores: list[tuple[int, float]] = []

        for doc_id in range(len(documents)):
            tf_s = raw_tf_score(query_terms, corpus_tokens[doc_id])
            tfidf_s = tfidf_score(query_terms, corpus_tokens[doc_id], idf_classic)
            bm25_s = bm25_score(
                query_terms, corpus_tokens[doc_id], doc_lengths[doc_id],
                avg_doc_length, idf_bm25,
            )
            tf_scores.append((doc_id, tf_s))
            tfidf_scores.append((doc_id, tfidf_s))
            bm25_scores.append((doc_id, bm25_s))

        tf_scores.sort(key=lambda x: x[1], reverse=True)
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_scores.sort(key=lambda x: x[1], reverse=True)

        # Show top results side by side
        print(f"  {'Rank':<6} {'Raw TF':^24} {'TF-IDF':^24} {'BM25':^24}")
        for rank in range(min(TOP_K, len(documents))):
            tf_id, tf_s = tf_scores[rank]
            tfidf_id, tfidf_s = tfidf_scores[rank]
            bm25_id, bm25_s = bm25_scores[rank]
            print(
                f"  {rank+1:<6} "
                f"doc[{tf_id:>2}] {tf_s:>6.2f}       "
                f"doc[{tfidf_id:>2}] {tfidf_s:>6.2f}       "
                f"doc[{bm25_id:>2}] {bm25_s:>6.2f}"
            )

        # Highlight ranking disagreements — these are the educational moments
        tf_top = tf_scores[0][0]
        tfidf_top = tfidf_scores[0][0]
        bm25_top = bm25_scores[0][0]
        if tf_top == tfidf_top == bm25_top:
            print(f"  → All methods agree: doc[{bm25_top}]")
        else:
            if tf_top != bm25_top:
                print(f"  → Raw TF picks doc[{tf_top}], BM25 picks doc[{bm25_top}]")
            if tfidf_top != bm25_top:
                print(f"  → TF-IDF picks doc[{tfidf_top}], BM25 picks doc[{bm25_top}]")


# === DETAILED SCORE BREAKDOWN ===
# For one query, show exactly how each term contributes to the final score.
# This is the "open the hood" moment that makes the algorithm click.

def print_score_breakdown(
    query: str,
    documents: list[str],
    corpus_tokens: list[list[str]],
    doc_lengths: list[int],
    avg_doc_length: float,
    idf_classic: dict[str, float],
    idf_bm25: dict[str, float],
) -> None:
    """For the top BM25 result, show per-term contribution from all three methods."""
    print("\n" + "=" * 70)
    print("DETAILED SCORE BREAKDOWN")
    print(f"Query: \"{query}\"")
    print("=" * 70)

    query_terms = tokenize(query)

    # Find top BM25 document
    best_id = -1
    best_score = -1.0
    for doc_id in range(len(documents)):
        s = bm25_score(
            query_terms, corpus_tokens[doc_id], doc_lengths[doc_id],
            avg_doc_length, idf_bm25,
        )
        if s > best_score:
            best_score = s
            best_id = doc_id

    print(f"\nTop BM25 result: doc[{best_id}] (score={best_score:.4f})")
    print(f"Document: \"{documents[best_id]}\"")
    print(f"Length: {doc_lengths[best_id]} tokens (avg={avg_doc_length:.1f})")

    # Per-term breakdowns
    tf_bd = raw_tf_score_breakdown(query_terms, corpus_tokens[best_id])
    tfidf_bd = tfidf_score_breakdown(query_terms, corpus_tokens[best_id], idf_classic)
    bm25_bd = bm25_score_breakdown(
        query_terms, corpus_tokens[best_id], doc_lengths[best_id],
        avg_doc_length, idf_bm25,
    )

    print(f"\n  {'Term':<16} {'Raw TF':>8} {'TF-IDF':>8} {'BM25':>8}   "
          f"{'IDF(classic)':>12} {'IDF(BM25)':>10}")
    print("  " + "-" * 68)
    for term in query_terms:
        idf_c = idf_classic.get(term, 0.0)
        idf_b = idf_bm25.get(term, 0.0)
        print(
            f"  {term:<16} {tf_bd[term]:>8.3f} {tfidf_bd[term]:>8.3f} "
            f"{bm25_bd[term]:>8.3f}   {idf_c:>12.3f} {idf_b:>10.3f}"
        )

    total_tf = sum(tf_bd.values())
    total_tfidf = sum(tfidf_bd.values())
    total_bm25 = sum(bm25_bd.values())
    print("  " + "-" * 68)
    print(f"  {'TOTAL':<16} {total_tf:>8.3f} {total_tfidf:>8.3f} {total_bm25:>8.3f}")


# === IDF COMPARISON TABLE ===
# Show how classic and BM25 IDF diverge for terms at different document frequencies.

def print_idf_comparison(
    idf_classic: dict[str, float],
    idf_bm25: dict[str, float],
    corpus_tokens: list[list[str]],
    num_docs: int,
) -> None:
    """Compare classic and BM25 IDF for terms at various document frequencies."""
    print("\n" + "=" * 70)
    print("IDF COMPARISON: Classic vs BM25")
    print(f"Corpus: {num_docs} documents")
    print("=" * 70)

    # Compute document frequency for each term
    df: dict[str, int] = {}
    for doc_tokens in corpus_tokens:
        seen: set[str] = set()
        for token in doc_tokens:
            if token not in seen:
                df[token] = df.get(token, 0) + 1
                seen.add(token)

    # Sort terms by DF to show the spectrum from rare to common
    terms_by_df = sorted(df.items(), key=lambda x: x[1])

    # Sample terms at different DF levels
    sampled: list[tuple[str, int]] = []
    seen_dfs: set[int] = set()
    for term, freq in terms_by_df:
        if freq not in seen_dfs:
            sampled.append((term, freq))
            seen_dfs.add(freq)

    print(f"\n  {'Term':<20} {'df':>4} {'df/N':>6} {'IDF(classic)':>13} {'IDF(BM25)':>10}")
    print("  " + "-" * 57)
    for term, freq in sampled:
        idf_c = idf_classic.get(term, 0.0)
        idf_b = idf_bm25.get(term, 0.0)
        ratio = freq / num_docs
        print(f"  {term:<20} {freq:>4} {ratio:>6.2f} {idf_c:>13.3f} {idf_b:>10.3f}")

    # Educational note: classic IDF goes negative when df > N/2 (term in >50% of docs).
    # BM25's formulation with +0.5 and outer +1 keeps IDF non-negative, which makes
    # more intuitive sense — a term shouldn't *hurt* a document's score.
    print("\n  Note: Classic IDF = log(N/df) goes negative when df > N/2.")
    print("  BM25 IDF = log((N-df+0.5)/(df+0.5)+1) is always non-negative.")


# === MAIN ===

def main() -> None:
    """Run the full TF → TF-IDF → BM25 evolution demonstration."""
    print("=" * 70)
    print("MICROBM25: The Evolution of Text Retrieval Scoring")
    print("Raw Term Frequency → TF-IDF → BM25")
    print("=" * 70)

    # --- Build corpus ---
    documents, queries = build_corpus()
    corpus_tokens = [tokenize(doc) for doc in documents]
    doc_lengths = [len(tokens) for tokens in corpus_tokens]
    num_docs = len(documents)
    avg_doc_length = sum(doc_lengths) / num_docs

    print(f"\nCorpus: {num_docs} documents")
    print(f"Average document length: {avg_doc_length:.1f} tokens")
    print(f"Length range: {min(doc_lengths)}–{max(doc_lengths)} tokens")
    print(f"Vocabulary size: {len(set(t for tokens in corpus_tokens for t in tokens))} unique terms")

    # --- Precompute IDF tables ---
    # Two separate IDF computations: classic for TF-IDF, BM25-style for BM25.
    # Comparing them side by side shows why BM25's smoothed IDF is superior.
    idf_classic = compute_idf_classic(corpus_tokens, num_docs)
    idf_bm25 = compute_idf_bm25(corpus_tokens, num_docs)

    # --- Print corpus overview ---
    print("\nDocuments:")
    for doc_id, doc in enumerate(documents):
        snippet = doc[:70] + ("..." if len(doc) > 70 else "")
        print(f"  [{doc_id:>2}] ({doc_lengths[doc_id]:>3} tokens) {snippet}")

    # --- TF saturation curve ---
    # The single most important visualization: shows WHY BM25 bounds term frequency
    print_saturation_curve()

    # --- IDF comparison ---
    print_idf_comparison(idf_classic, idf_bm25, corpus_tokens, num_docs)

    # --- Three-way comparison ---
    compare_methods(
        queries, documents, corpus_tokens, doc_lengths,
        avg_doc_length, idf_classic, idf_bm25,
    )

    # --- Detailed breakdown for one query ---
    # Pick the query most likely to show interesting differences
    print_score_breakdown(
        "search ranking documents", documents, corpus_tokens,
        doc_lengths, avg_doc_length, idf_classic, idf_bm25,
    )

    # --- Parameter exploration ---
    # Show how k1 and b independently affect rankings
    explore_k1_effect(
        "neural network training", corpus_tokens, doc_lengths,
        avg_doc_length, idf_bm25, documents,
    )
    explore_b_effect(
        "neural network training", corpus_tokens, doc_lengths,
        avg_doc_length, idf_bm25, documents,
    )

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Why BM25 Wins")
    print("=" * 70)
    print("""
  Raw TF:  count(term, doc)
    Problem: longer documents always score higher (more words = more counts)

  TF-IDF:  log(1 + tf) × log(N / df)
    Fixes:  IDF downweights common terms
    Problem: TF is still unbounded — extreme repetition dominates

  BM25:    IDF_smooth × (tf × (k1+1)) / (tf + k1 × (1-b+b×dl/avgdl))
    Fixes:  Saturating TF (bounded by k1+1)
            Document length normalization (controlled by b)
            Smooth IDF (always non-negative)
    Result: The standard scoring function in production search engines.
""")


if __name__ == "__main__":
    main()
