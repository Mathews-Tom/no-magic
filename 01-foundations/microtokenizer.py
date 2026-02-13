"""
How text becomes numbers -- the compression algorithm hiding inside every LLM.
Byte-Pair Encoding learns a vocabulary by iteratively merging the most frequent
adjacent token pairs, then encodes new text by replaying those merges in priority order.
"""
# Reference: Philip Gage, "A New Algorithm for Data Compression" (1994).
# GPT-2's byte-level BPE variant (Radford et al., 2019) starts from raw bytes
# rather than characters -- that's the version implemented here.

from __future__ import annotations

import os
import random
import urllib.request
from collections import Counter

random.seed(42)  # repo convention; BPE itself is fully deterministic


# === CONSTANTS ===

NUM_MERGES = 256  # Final vocab = 256 byte tokens + 256 merges = 512 tokens.
# Signpost: production tokenizers (GPT-2, GPT-4) use 50K+ merges trained on
# hundreds of gigabytes. 256 merges on 18KB is a toy, but the algorithm is identical.

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"


# === DATA LOADING ===

def load_data(url: str, filename: str) -> bytes:
    """Download dataset if not cached, return raw bytes."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "rb") as f:
        return f.read()


# === BPE TRAINING ===

def get_pair_counts(token_ids: list[int]) -> Counter:
    """Count frequency of every adjacent token pair.

    For sequence s = [s_0, s_1, ..., s_n], we count all (s_i, s_{i+1}) pairs.
    Example: [a, b, c, b, c] -> {(a,b): 1, (b,c): 2, (c,b): 1}.
    This is the core statistic BPE uses to decide what to merge next.
    """
    # zip(ids, ids[1:]) pairs each element with its right neighbor -- O(n).
    return Counter(zip(token_ids, token_ids[1:]))


def apply_merge(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id` in a single left-to-right pass.

    Overlapping pairs resolve left-to-right: in [a, a, a] merging (a,a) produces
    [new, a], not [a, new]. This matches the standard BPE convention and ensures
    the merge operation is deterministic regardless of pair overlap patterns.
    """
    # Signpost: this O(n) scan runs once per merge, giving O(n * M) total training
    # cost for M merges. Production implementations (SentencePiece, tiktoken) use
    # priority queues for O(n log n) total, but the output is identical.
    merged = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # consumed both tokens in the pair
        else:
            merged.append(token_ids[i])
            i += 1
    return merged


def train_bpe(
    token_ids: list[int], num_merges: int
) -> list[tuple[tuple[int, int], int]]:
    """Learn BPE merge rules by greedily merging the most frequent adjacent pair.

    Each merge absorbs the single most redundant pair in the corpus -- a greedy
    compression step that naturally discovers morphological units ("an" + "a",
    "el" + "la") without any linguistic rules. The merge table is ordered by
    priority: merge 0 was most frequent in the original corpus, merge 1 most
    frequent after merge 0, and so on. This ordering is critical for encoding.

    Returns: ordered list of (pair, new_id) tuples where new_id = 256 + merge_index.
    """
    ids = list(token_ids)  # work on a copy
    merges: list[tuple[tuple[int, int], int]] = []

    for i in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            # Entire corpus collapsed to a single token (or is empty). Rare in
            # practice, but correct to handle: no more pairs means no more merges.
            break

        # The pair with the highest count gets merged next.
        pair = max(counts, key=counts.get)  # type: ignore[arg-type]
        new_id = 256 + i  # byte IDs 0-255 reserved; merges start at 256

        ids = apply_merge(ids, pair, new_id)
        merges.append((pair, new_id))

        if (i + 1) % 32 == 0 or i == 0:
            a, b = pair
            print(
                f"  merge {i + 1:>3}/{num_merges}: "
                f"({a:>3}, {b:>3}) -> {new_id:>3}  "
                f"freq={counts[pair]:>5}  corpus_len={len(ids)}"
            )

    return merges


# === ENCODING & DECODING ===

def build_vocab(merges: list[tuple[tuple[int, int], int]]) -> dict[int, bytes]:
    """Build token ID -> bytes lookup table.

    Base vocabulary: 256 entries mapping each byte value to its single-byte string.
    Each merge extends the table: vocab[new_id] = vocab[a] + vocab[b].
    This recursive expansion means decoding is just a table lookup -- no merge
    replay needed, and round-trip correctness is guaranteed by construction.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges:
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: list[tuple[tuple[int, int], int]]) -> list[int]:
    """Encode a string to BPE token IDs by replaying merges in priority order.

    Critical: merges are applied in the order they were learned (priority order),
    NOT by re-counting frequencies on the new text. Priority order ensures
    deterministic tokenization -- the same string always produces the same token
    sequence, regardless of what other text the tokenizer was trained on.
    Re-counting frequencies would make the output dependent on the input batch,
    breaking the contract that tokenization is a pure function of the input string.
    """
    # Signpost: this O(n * M) naive encoding checks every merge against the full
    # sequence. Production tokenizers (tiktoken, HuggingFace) use trie structures
    # for O(n) encoding, but produce identical output.
    token_ids = list(text.encode("utf-8"))
    for pair, new_id in merges:
        token_ids = apply_merge(token_ids, pair, new_id)
    return token_ids


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode token IDs back to a string via byte lookup and UTF-8 decoding.

    Every token maps to a definite byte sequence through the vocab table, so
    decode(encode(text)) == text is guaranteed for any valid UTF-8 input.
    Decoding is trivially simple by design -- all the complexity lives in encoding.
    """
    raw_bytes = b"".join(vocab[tid] for tid in token_ids)
    return raw_bytes.decode("utf-8")


# === INFERENCE DEMO ===

if __name__ == "__main__":
    # -- Load and prepare data --
    raw = load_data(DATA_URL, DATA_FILE)
    corpus_ids = list(raw)

    # Starting from raw bytes means every possible input is representable --
    # there are no "unknown token" problems. This is the key insight of byte-level
    # BPE: the base vocabulary covers all of Unicode (via UTF-8 byte sequences)
    # without needing a character-level vocabulary for every writing system.
    print(f"Corpus: {len(raw):,} bytes, base vocab: 256 byte tokens")
    print(f"Training {NUM_MERGES} merges (final vocab: {256 + NUM_MERGES} tokens)\n")

    # -- Train --
    print("Training BPE...")
    merges = train_bpe(corpus_ids, NUM_MERGES)
    vocab = build_vocab(merges)
    print(f"\nTraining complete: {len(merges)} merges learned\n")

    # -- Round-trip tests --
    # Verify encode-decode identity on diverse inputs: common name, uncommon name,
    # hyphenated, apostrophe, empty string, single character.
    test_strings = ["Emma", "Xiomara", "Mary-Jane", "O'Brien", "", "Z"]
    print("Round-trip tests:")
    all_pass = True
    for s in test_strings:
        encoded = encode(s, merges)
        decoded = decode(encoded, vocab)
        status = "PASS" if decoded == s else "FAIL"
        if status == "FAIL":
            all_pass = False
        display = f'"{s}"' if s else '""'
        print(f"  [{status}] {display:<14} -> {len(encoded):>2} tokens -> {decoded!r}")
    print()

    # -- Compression ratio --
    # compression_ratio = len(original_bytes) / len(bpe_tokens)
    # Each BPE token represents `ratio` bytes on average. Higher is better --
    # it means the tokenizer discovered more compressible structure.
    corpus_text = raw.decode("utf-8")
    corpus_encoded = encode(corpus_text, merges)
    ratio = len(raw) / len(corpus_encoded)
    print(
        f"Compression: {len(raw):,} bytes -> {len(corpus_encoded):,} tokens "
        f"(ratio: {ratio:.2f}x)\n"
    )

    # -- Top 20 merges --
    print("Top 20 merges (earliest = highest priority):")
    for i, ((a, b), new_id) in enumerate(merges[:20]):
        a_str = vocab[a].decode("utf-8", errors="replace")
        b_str = vocab[b].decode("utf-8", errors="replace")
        merged_str = vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {i + 1:>2}. {a_str!r:>6} + {b_str!r:<6} -> {merged_str!r}")
    print()

    # -- Tokenization example --
    example = "Elizabeth"
    example_tokens = encode(example, merges)
    pieces = [vocab[tid].decode("utf-8", errors="replace") for tid in example_tokens]
    print(f'Tokenization example: "{example}"')
    print(f"  Bytes:  {list(example.encode('utf-8'))}")
    print(f"  Tokens: {example_tokens}")
    print(f"  Pieces: {pieces}")
