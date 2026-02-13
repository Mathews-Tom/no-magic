"""
How retrieval augments generation -- the simplest system that actually works, with BM25
search and a character-level MLP in pure Python.
"""
# Reference: RAG architecture inspired by "Retrieval-Augmented Generation for
# Knowledge-Intensive NLP Tasks" (Lewis et al., 2020), BM25 scoring from Robertson
# and Zaragoza (2009). Implementation rewritten from scratch for educational clarity.

from __future__ import annotations

import math
import random
import string

random.seed(42)


# === CONSTANTS ===

LEARNING_RATE = 0.01
HIDDEN_DIM = 64  # hidden layer size for MLP
NUM_EPOCHS = 300
TOP_K = 3  # retrieve top 3 documents
BATCH_SIZE = 5

# BM25 hyperparameters (standard values from information retrieval literature)
K1 = 1.2  # term frequency saturation parameter
B = 0.75  # document length normalization parameter

CHAR_VOCAB = list(string.ascii_lowercase + " .,")  # character vocabulary
VOCAB_SIZE = len(CHAR_VOCAB)


# === SYNTHETIC KNOWLEDGE BASE ===

def generate_knowledge_base() -> tuple[list[str], list[tuple[str, str]]]:
    """Generate 100 synthetic factual paragraphs and 20 test queries.

    We use templates + data tables to create verifiable factual knowledge about
    cities, countries, populations, and geography. This ensures deterministic,
    reproducible data without external downloads or API calls.

    Returns: (documents, test_queries) where test_queries are (query, expected_doc_index)
    """
    # Data tables -- these are the "ground truth" facts
    cities = [
        ("Paris", "France", "2.1 million", "Seine"),
        ("London", "United Kingdom", "8.9 million", "Thames"),
        ("Berlin", "Germany", "3.8 million", "Spree"),
        ("Madrid", "Spain", "3.3 million", "Manzanares"),
        ("Rome", "Italy", "2.8 million", "Tiber"),
        ("Tokyo", "Japan", "14 million", "Sumida"),
        ("Beijing", "China", "21 million", "Yongding"),
        ("Delhi", "India", "16 million", "Yamuna"),
        ("Cairo", "Egypt", "9.5 million", "Nile"),
        ("Lagos", "Nigeria", "14 million", "Lagos Lagoon"),
    ]

    mountains = [
        ("Everest", "Nepal", "8849 meters"),
        ("K2", "Pakistan", "8611 meters"),
        ("Kilimanjaro", "Tanzania", "5895 meters"),
        ("Mont Blanc", "France", "4808 meters"),
        ("Denali", "United States", "6190 meters"),
    ]

    # Generate city paragraphs
    documents = []
    for city, country, pop, river in cities:
        doc = (
            f"{city} is the capital of {country}. "
            f"It has a population of approximately {pop}. "
            f"The {river} river flows through the city."
        )
        documents.append(doc.lower())

    # Generate mountain paragraphs
    for mountain, country, height in mountains:
        doc = (
            f"{mountain} is located in {country}. "
            f"The mountain has a height of {height}. "
            f"It is a popular destination for climbers."
        )
        documents.append(doc.lower())

    # Generate additional filler documents (continent facts, simple statements)
    continents = [
        "africa is the second largest continent by area.",
        "asia is the most populous continent in the world.",
        "europe has diverse cultures and languages.",
        "north america includes canada, united states, and mexico.",
        "south america is home to the amazon rainforest.",
    ]
    documents.extend(continents)

    # Add more diverse factual statements to reach 100 documents
    for i in range(80):
        # Recombine facts with slight variations to create more documents
        if i % 4 == 0:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"The population of {city} is about {pop}. It is in {country}."
        elif i % 4 == 1:
            mountain, country, height = mountains[i % len(mountains)]
            doc = f"{mountain} stands at {height} in {country}."
        elif i % 4 == 2:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"The {river} river is a major waterway in {city}, {country}."
        else:
            city, country, pop, river = cities[i % len(cities)]
            doc = f"{city} is a major city with population {pop}."
        documents.append(doc.lower())

    # Generate test queries with known correct answers (document index)
    test_queries = [
        ("population of paris", 0),  # Paris doc
        ("seine river", 0),  # Paris doc mentions Seine
        ("tokyo population", 5),  # Tokyo doc
        ("everest height", 10),  # Everest doc
        ("capital of germany", 2),  # Berlin doc
        ("nile river", 8),  # Cairo doc mentions Nile
        ("kilimanjaro tanzania", 12),  # Kilimanjaro doc
        ("thames river london", 1),  # London doc
        ("mont blanc france", 13),  # Mont Blanc doc
        ("beijing china", 6),  # Beijing doc
    ]

    return documents, test_queries


# === TOKENIZATION ===

def tokenize(text: str) -> list[str]:
    """Simple word-level tokenization: lowercase, strip punctuation, split on spaces.

    Signpost: production RAG systems use learned subword tokenizers (BPE, SentencePiece).
    Word-level tokenization is sufficient here for demonstrating retrieval mechanics --
    the focus is on BM25 scoring and context injection, not tokenization quality.
    """
    # Remove punctuation and split into words
    words = []
    word = []
    for char in text.lower():
        if char.isalpha() or char.isdigit():
            word.append(char)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    return words


# === BM25 INDEX ===

class BM25Index:
    """BM25 scoring for document retrieval.

    BM25 improves on TF-IDF with two key insights:
    1. TF saturation: 10 occurrences isn't 10x more relevant than 1 occurrence.
       The formula uses (tf * (k1 + 1)) / (tf + k1) which saturates as tf → ∞.
    2. Document length normalization: long documents aren't inherently more relevant.
       The normalization term (1 - b + b * dl/avgdl) penalizes long docs.

    Math-to-code mapping:
      idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
      tf_score(term, doc) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
      BM25(query, doc) = Σ_{term in query} idf(term) * tf_score(term, doc)

    where:
      N = total documents
      df = number of docs containing term
      tf = term frequency in document
      dl = document length (word count)
      avgdl = average document length across corpus
      k1 = TF saturation parameter (1.2 standard)
      b = length normalization parameter (0.75 standard)
    """

    def __init__(self, documents: list[str], k1: float = K1, b: float = B):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)  # total number of documents

        # Tokenize all documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Build inverted index: term -> list of (doc_id, term_frequency)
        # This is the core data structure for efficient retrieval -- for each term,
        # we precompute which documents contain it and how often. At query time we
        # only score documents that share at least one term with the query.
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(self.doc_tokens):
            term_counts: dict[str, int] = {}
            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, count))

        # Precompute IDF scores for all terms
        # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1) where df = document frequency
        # Why add 0.5? Smoothing to prevent division by zero and reduce impact of rare terms.
        # Why the +1 outside? Ensures IDF is always positive (log(x) < 0 for x < 1).
        self.idf: dict[str, float] = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)  # document frequency = number of docs containing term
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: int) -> float:
        """Compute BM25 score for a query against a specific document."""
        query_terms = tokenize(query)
        score = 0.0

        dl = self.doc_lengths[doc_id]  # document length
        # Document length normalization factor: penalizes long docs but not linearly
        norm = 1 - self.b + self.b * (dl / self.avgdl)

        # Count term frequencies in document
        doc_term_counts: dict[str, int] = {}
        for term in self.doc_tokens[doc_id]:
            doc_term_counts[term] = doc_term_counts.get(term, 0) + 1

        for term in query_terms:
            if term not in self.idf:
                continue  # term not in corpus, contributes 0 to score
            tf = doc_term_counts.get(term, 0)
            if tf == 0:
                continue  # term not in this document

            # TF saturation: (tf * (k1 + 1)) / (tf + k1 * norm)
            # As tf → ∞, this approaches (k1 + 1) / k1 ≈ 1.83 (for k1=1.2).
            # This prevents term frequency from dominating the score.
            tf_score = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            score += self.idf[term] * tf_score

        return score

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """Retrieve top-k documents for a query, ranked by BM25 score.

        Returns: list of (doc_id, score) tuples sorted by descending score.
        """
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in range(self.N)]
        # Sort by score descending, take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# === CHARACTER-LEVEL MLP GENERATOR ===

def char_to_index(char: str) -> int:
    """Map character to index in vocabulary."""
    if char in CHAR_VOCAB:
        return CHAR_VOCAB.index(char)
    return CHAR_VOCAB.index(" ")  # fallback to space for unknown chars


def index_to_char(idx: int) -> str:
    """Map index to character."""
    return CHAR_VOCAB[idx]


def one_hot(idx: int, size: int) -> list[float]:
    """Create one-hot encoded vector."""
    vec = [0.0] * size
    vec[idx] = 1.0
    return vec


class MLP:
    """Character-level MLP generator with concatenated query + context input.

    Architecture:
      input (query_chars + context_chars) → hidden (ReLU) → output (softmax over chars)

    The key RAG mechanism: by concatenating retrieved context with the query, the MLP
    can condition its predictions on retrieved facts. This is the minimum architecture
    that meaningfully demonstrates RAG -- the model actually uses retrieved information
    rather than just ignoring it.

    Signpost: production RAG uses transformer generators (GPT, LLaMA). We use an MLP
    to keep the focus on the retrieval mechanism and context injection pattern.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization: scale weights by 1/sqrt(fan_in) for stable gradients
        # Why Xavier? Maintains variance of activations across layers, preventing
        # gradients from vanishing or exploding early in training.
        scale_1 = (2.0 / input_dim) ** 0.5
        scale_2 = (2.0 / hidden_dim) ** 0.5

        self.W1 = [[random.gauss(0, scale_1) for _ in range(input_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        self.W2 = [[random.gauss(0, scale_2) for _ in range(hidden_dim)]
                   for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

    def forward(self, x: list[float]) -> tuple[list[float], dict]:
        """Forward pass: input → hidden (ReLU) → output (softmax).

        Returns: (output_probs, cache) where cache stores intermediate values for backward.
        """
        # Hidden layer: h = ReLU(W1 @ x + b1)
        hidden = []
        for i in range(self.hidden_dim):
            activation = self.b1[i]
            for j in range(self.input_dim):
                activation += self.W1[i][j] * x[j]
            hidden.append(max(0.0, activation))  # ReLU

        # Output layer: o = W2 @ h + b2
        logits = []
        for i in range(self.output_dim):
            activation = self.b2[i]
            for j in range(self.hidden_dim):
                activation += self.W2[i][j] * hidden[j]
            logits.append(activation)

        # Stable softmax: exp(x - max(x)) prevents overflow
        # Without this, large logits cause exp() to overflow to inf, breaking gradients.
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # Cache for backward pass
        cache = {"x": x, "hidden": hidden, "logits": logits, "probs": probs}
        return probs, cache

    def backward(
        self, target_idx: int, cache: dict, learning_rate: float
    ) -> float:
        """Backward pass: compute gradients and update weights.

        Cross-entropy loss: L = -log(p[target_idx])
        Gradient of cross-entropy + softmax has a clean form: dL/do_i = p_i - 1[i == target]

        Returns: loss value
        """
        x = cache["x"]
        hidden = cache["hidden"]
        probs = cache["probs"]

        # Clip probability to prevent log(0) = -inf
        loss = -math.log(max(probs[target_idx], 1e-10))

        # Gradient of loss w.r.t. output logits: p - y (where y is one-hot target)
        dlogits = list(probs)
        dlogits[target_idx] -= 1.0

        # Gradient w.r.t. W2 and b2
        dW2 = [[0.0] * self.hidden_dim for _ in range(self.output_dim)]
        db2 = [0.0] * self.output_dim
        for i in range(self.output_dim):
            db2[i] = dlogits[i]
            for j in range(self.hidden_dim):
                dW2[i][j] = dlogits[i] * hidden[j]

        # Backprop through hidden layer
        dhidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for i in range(self.output_dim):
                dhidden[j] += dlogits[i] * self.W2[i][j]
            # ReLU gradient: 0 if hidden[j] <= 0, else pass through
            if hidden[j] <= 0:
                dhidden[j] = 0.0

        # Gradient w.r.t. W1 and b1
        dW1 = [[0.0] * self.input_dim for _ in range(self.hidden_dim)]
        db1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            db1[i] = dhidden[i]
            for j in range(self.input_dim):
                dW1[i][j] = dhidden[i] * x[j]

        # Update weights with SGD: w = w - lr * dw
        for i in range(self.output_dim):
            self.b2[i] -= learning_rate * db2[i]
            for j in range(self.hidden_dim):
                self.W2[i][j] -= learning_rate * dW2[i][j]

        for i in range(self.hidden_dim):
            self.b1[i] -= learning_rate * db1[i]
            for j in range(self.input_dim):
                self.W1[i][j] -= learning_rate * dW1[i][j]

        return loss

    def generate(self, input_text: str, max_length: int = 50) -> str:
        """Generate text character-by-character given an input context.

        The input_text contains both the query and retrieved context (concatenated).
        The model uses this full context to predict the next character at each step.
        """
        # Start with input context as the seed
        current_text = input_text
        for _ in range(max_length):
            # Encode recent context (last 100 chars to keep input size manageable)
            context = current_text[-100:]
            x = []
            for char in context:
                idx = char_to_index(char)
                x.extend(one_hot(idx, VOCAB_SIZE))
            # Pad to fixed input size if needed
            while len(x) < self.input_dim:
                x.append(0.0)
            x = x[:self.input_dim]  # truncate if too long

            # Generate next character
            probs, _ = self.forward(x)
            next_idx = probs.index(max(probs))  # greedy sampling
            next_char = index_to_char(next_idx)

            # Stop at period (simple generation termination)
            if next_char == ".":
                current_text += next_char
                break
            current_text += next_char

        return current_text


# === TRAINING LOOP ===

def train_rag(
    documents: list[str],
    bm25: BM25Index,
    mlp: MLP,
    num_epochs: int,
    learning_rate: float
):
    """Train the MLP on (query, context, answer) triples from the knowledge base.

    Training process:
    1. Sample a random document as ground truth
    2. Extract a query from the document (first few words)
    3. Retrieve context using BM25
    4. Concatenate query + retrieved context
    5. Train MLP to predict next character in the ground truth answer

    This teaches the model to use retrieved context to generate accurate completions.
    """
    print("Training RAG model...\n")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_samples = 0

        for _ in range(BATCH_SIZE):
            # Sample a random document as the ground truth
            doc_idx = random.randint(0, len(documents) - 1)
            doc = documents[doc_idx]

            # Create a query from the first few words of the document
            # This simulates a user asking about the topic in the document
            words = tokenize(doc)
            if len(words) < 3:
                continue
            query_words = words[:min(3, len(words))]
            query = " ".join(query_words)

            # Retrieve context using BM25
            retrieved = bm25.retrieve(query, top_k=TOP_K)
            context = " ".join([documents[doc_id] for doc_id, _ in retrieved[:2]])

            # Concatenate query + context as model input
            # This is the core RAG mechanism: the model sees both the query and
            # retrieved facts, enabling it to condition its predictions on external knowledge.
            input_text = query + " " + context

            # Target: the full ground truth document
            # The model learns to complete from query+context to the full factual answer
            target = doc

            # Train on each character in the target
            for i in range(min(20, len(target))):  # limit to first 20 chars for speed
                # Encode input context — use the LAST 100 chars (sliding window)
                # so the model sees updated context as target chars are appended.
                # This matches the inference-time behavior in generate().
                x = []
                for char in input_text[-100:]:
                    idx = char_to_index(char)
                    x.extend(one_hot(idx, VOCAB_SIZE))
                # Pad to fixed size
                while len(x) < mlp.input_dim:
                    x.append(0.0)
                x = x[:mlp.input_dim]

                # Target character
                target_idx = char_to_index(target[i])

                # Forward + backward
                _, cache = mlp.forward(x)
                loss = mlp.backward(target_idx, cache, learning_rate)
                epoch_loss += loss
                num_samples += 1

                # Update input_text to include predicted character
                input_text += target[i]

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {avg_loss:.4f}")

    print()


# === INFERENCE DEMO ===

def demo_retrieval_comparison(
    queries: list[str],
    documents: list[str],
    bm25: BM25Index,
    mlp: MLP
):
    """Demonstrate generation WITH and WITHOUT retrieval for comparison.

    This shows the core RAG value proposition: retrieved context improves generation
    quality by providing factual grounding. Without retrieval, the model must rely
    entirely on its parametric knowledge (learned weights), which is prone to
    hallucination on factual queries.
    """
    print("=== RETRIEVAL COMPARISON ===\n")

    for query in queries:
        print(f"Query: '{query}'")

        # WITH retrieval: BM25 → retrieve context → MLP generates
        retrieved = bm25.retrieve(query, top_k=TOP_K)
        print(f"Retrieved docs (top {TOP_K}):")
        for doc_id, score in retrieved:
            print(f"  [{doc_id}] score={score:.2f}: {documents[doc_id][:60]}...")

        context = " ".join([documents[doc_id] for doc_id, _ in retrieved[:2]])
        input_with_context = query + " " + context
        generation_with = mlp.generate(input_with_context, max_length=40)

        # WITHOUT retrieval: empty context → MLP generates
        # The model receives only the query, no external facts to condition on
        input_without_context = query + " "
        generation_without = mlp.generate(input_without_context, max_length=40)

        print(f"WITH retrieval:    {generation_with}")
        print(f"WITHOUT retrieval: {generation_without}")
        print()


# === MAIN ===

if __name__ == "__main__":
    # Generate synthetic knowledge base
    print("Generating synthetic knowledge base...")
    documents, test_queries = generate_knowledge_base()
    print(f"Created {len(documents)} documents\n")

    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Index(documents, k1=K1, b=B)
    print(f"Indexed {bm25.N} documents, {len(bm25.idf)} unique terms\n")

    # Test retrieval accuracy on known queries.
    # Since the knowledge base has multiple documents per topic (e.g., Paris appears
    # in city paragraphs, population docs, and river docs), BM25 may return a more
    # specific document about the same entity. We measure accuracy by checking whether
    # the query's key terms appear in the retrieved document — this tests whether BM25
    # finds relevant content, not whether it picks a specific document index.
    print("=== RETRIEVAL ACCURACY TEST ===")
    correct = 0
    for query, expected_doc_idx in test_queries:
        retrieved = bm25.retrieve(query, top_k=1)
        if not retrieved:
            print(f"  MISS: '{query}' -> no results")
            continue

        retrieved_idx = retrieved[0][0]
        retrieved_terms = set(tokenize(documents[retrieved_idx]))
        query_terms = set(tokenize(query))

        # A retrieval is correct if the returned document contains ≥50% of query terms.
        # This measures topical relevance: "seine river" → doc mentioning seine and river.
        query_hits = sum(1 for t in query_terms if t in retrieved_terms)
        if query_hits >= max(len(query_terms) * 0.5, 1):
            correct += 1
            print(f"  HIT:  '{query}' -> [{retrieved_idx}] {documents[retrieved_idx][:50]}...")
        else:
            print(
                f"  MISS: '{query}' -> [{retrieved_idx}] {documents[retrieved_idx][:50]}..."
            )
    accuracy = 100 * correct / len(test_queries)
    print(f"Retrieval accuracy: {correct}/{len(test_queries)} = {accuracy:.1f}%\n")

    # Initialize MLP generator
    # Input dimension: concatenated query + context (each ~100 chars, one-hot encoded)
    # We use a fixed input window to keep dimensions manageable
    input_dim = 100 * VOCAB_SIZE  # 100 characters, one-hot encoded
    mlp = MLP(input_dim, HIDDEN_DIM, VOCAB_SIZE)
    print(f"Initialized MLP: {input_dim} -> {HIDDEN_DIM} -> {VOCAB_SIZE}")
    total_params = (
        len(mlp.W1) * len(mlp.W1[0]) + len(mlp.b1) +
        len(mlp.W2) * len(mlp.W2[0]) + len(mlp.b2)
    )
    print(f"Total parameters: {total_params:,}\n")

    # Train the RAG model
    train_rag(documents, bm25, mlp, NUM_EPOCHS, LEARNING_RATE)

    # Demo: compare generation with and without retrieval
    demo_queries = [
        "population of paris",
        "seine river",
        "everest height",
        "capital of germany",
    ]
    demo_retrieval_comparison(demo_queries, documents, bm25, mlp)

    print("RAG demonstration complete.")
