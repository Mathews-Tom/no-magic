"""
How vLLM serves thousands of concurrent requests -- paged memory allocation for KV-caches
and the OS principles behind scalable LLM serving.
"""
# Reference: Kwon et al., "Efficient Memory Management for Large Language Model Serving
# with PagedAttention" (2023). https://arxiv.org/abs/2309.06180
# Also: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative
# Models" (2022) for continuous batching.

from __future__ import annotations

import math
import random
import time

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

HEAD_DIM = 8       # dimensionality of each key/value vector
N_HEADS = 2        # number of attention heads

# Page geometry -- the core knob. Smaller pages = less internal fragmentation but more
# metadata overhead. vLLM uses block_size=16 by default; we use 4 for visible behavior.
PAGE_BLOCK_SIZE = 4     # KV positions stored per physical page
NUM_PHYSICAL_PAGES = 16 # total physical memory budget (the "RAM" of our system)

# Naive allocator pre-reserves this many positions per request, regardless of actual length.
MAX_SEQ_LEN = 20

# Simulation parameters
NUM_REQUESTS = 8
MAX_GEN_LEN = 12

# Signpost: production vLLM manages thousands of physical blocks across 80GB GPU memory
# with block_size=16 and head_dim=128. Our toy numbers preserve every algorithmic detail
# while keeping the simulation readable in a terminal.


# === HELPER FUNCTIONS ===
# Plain Python vector operations for attention computation. No matrices needed --
# PagedAttention operates on individual query vectors against scattered KV pages.

def rand_vec(dim: int) -> list[float]:
    """Random vector with 1/sqrt(dim) scaling to keep dot products O(1)."""
    s = 1.0 / math.sqrt(dim)
    return [random.gauss(0.0, s) for _ in range(dim)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def softmax(scores: list[float]) -> list[float]:
    """Stable softmax: subtract max to prevent exp() overflow.
    softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))"""
    if not scores:
        return []
    mx = max(scores)
    exps = [math.exp(x - mx) for x in scores]
    s = sum(exps)
    return [e / s for e in exps]


# === PHYSICAL MEMORY POOL ===
# Intuition: just like OS physical memory is divided into fixed-size frames (typically 4KB),
# GPU memory for KV-cache is divided into fixed-size blocks. The indirection through page
# tables is what enables efficient, non-contiguous allocation.
PhysicalPage = list[tuple[list[float], list[float]]]  # (key_vec, value_vec) per position


# === NAIVE KV-CACHE ALLOCATOR (BASELINE) ===
# Pre-allocates max_seq_len worth of contiguous storage per request (pre-vLLM approach).
# Waste sources: internal fragmentation (short seqs don't fill) + reservation fragmentation
# (memory reserved for max_seq_len blocks other requests).

class NaiveAllocator:
    """Pre-allocates contiguous KV storage at maximum possible length per request.
    A request generating only 5 tokens still holds 20 slots, blocking other requests."""

    def __init__(self, max_seq_len: int, total_budget: int) -> None:
        self.max_seq_len = max_seq_len
        self.total_budget = total_budget
        self.allocated: dict[str, int] = {}  # rid -> reserved slots
        self.used: dict[str, int] = {}       # rid -> filled slots
        self.peak_allocated = 0

    def allocate(self, rid: str) -> bool:
        """Reserve max_seq_len slots. Returns False if budget exhausted."""
        if sum(self.allocated.values()) + self.max_seq_len > self.total_budget:
            return False
        self.allocated[rid] = self.max_seq_len
        self.used[rid] = 0
        self.peak_allocated = max(self.peak_allocated, sum(self.allocated.values()))
        return True

    def append_token(self, rid: str) -> bool:
        if self.used.get(rid, 0) >= self.allocated.get(rid, 0):
            return False
        self.used[rid] += 1
        return True

    def free(self, rid: str) -> None:
        self.allocated.pop(rid, None)
        self.used.pop(rid, None)

    def utilization(self) -> float:
        a = sum(self.allocated.values())
        return sum(self.used.values()) / a if a > 0 else 0.0


# === PAGED KV-CACHE ALLOCATOR ===
# Each request has a block table mapping logical -> physical pages.
#
# Math-to-code (identical to OS virtual memory translation):
#   KV-cache side:              OS side:
#   logical_page = pos // BS    virtual_page = addr // PAGE_SIZE
#   offset = pos % BS           offset = addr % PAGE_SIZE
#   phys = block_table[page]    frame = page_table[virtual_page]
#   data = memory[phys][offset] byte = ram[frame * PAGE_SIZE + offset]

class PagedAllocator:
    """On-demand page allocation for KV-cache, mirroring OS virtual memory.

    Key insight: a request generating 5 tokens only consumes ceil(5/4) = 2 pages,
    not the 5 pages (20 slots) that naive allocation would reserve."""

    def __init__(self, num_pages: int, block_size: int) -> None:
        self.num_pages = num_pages
        self.block_size = block_size
        self.physical_memory: list[PhysicalPage] = [[] for _ in range(num_pages)]
        self.free_list: list[int] = list(range(num_pages))  # like OS free frame list
        self.block_tables: dict[str, list[int]] = {}        # the "page tables"
        self.seq_lens: dict[str, int] = {}
        self.peak_pages_used = 0
        # Preemption: saved state for paused requests (like swap space)
        self.preempted: dict[str, tuple[int, list[PhysicalPage]]] = {}

    def _alloc_page(self) -> int | None:
        if not self.free_list:
            return None
        idx = self.free_list.pop()
        self.peak_pages_used = max(self.peak_pages_used, self.pages_used())
        return idx

    def _free_page(self, idx: int) -> None:
        self.physical_memory[idx] = []
        self.free_list.append(idx)

    def allocate_request(self, rid: str) -> bool:
        """Empty block table; pages allocated lazily on first append_token."""
        if rid in self.block_tables:
            return False
        self.block_tables[rid] = []
        self.seq_lens[rid] = 0
        return True

    def append_token(self, rid: str, k: list[float], v: list[float]) -> bool:
        """Append KV data. Allocates new page if current one is full ("page fault").
        Returns False on OOM -- caller must preempt or reject."""
        if rid not in self.block_tables:
            return False
        seq_len = self.seq_lens[rid]
        logical_page = seq_len // self.block_size
        if logical_page >= len(self.block_tables[rid]):
            phys = self._alloc_page()
            if phys is None:
                return False
            self.block_tables[rid].append(phys)
        # Production vLLM stores heads in separate pools for coalescing
        self.physical_memory[self.block_tables[rid][logical_page]].append((k, v))
        self.seq_lens[rid] = seq_len + 1
        return True

    def free_request(self, rid: str) -> int:
        pages = self.block_tables.pop(rid, [])
        self.seq_lens.pop(rid, None)
        for p in pages:
            self._free_page(p)
        return len(pages)

    def preempt(self, rid: str) -> bool:
        """Save KV data and free pages -- like OS swapping a process to disk.
        vLLM uses recomputation-based preemption (cheaper); we copy for clarity."""
        if rid not in self.block_tables:
            return False
        pages = self.block_tables.pop(rid)
        saved = [list(self.physical_memory[p]) for p in pages]
        for p in pages:
            self._free_page(p)
        self.preempted[rid] = (self.seq_lens.pop(rid, 0), saved)
        return True

    def resume(self, rid: str) -> bool:
        """Allocate new pages and restore saved KV data."""
        if rid not in self.preempted:
            return False
        seq_len, saved = self.preempted.pop(rid)
        new_pages: list[int] = []
        for page_data in saved:
            phys = self._alloc_page()
            if phys is None:  # still OOM -- re-shelve
                for p in new_pages:
                    self._free_page(p)
                self.preempted[rid] = (seq_len, saved)
                return False
            self.physical_memory[phys] = list(page_data)
            new_pages.append(phys)
        self.block_tables[rid] = new_pages
        self.seq_lens[rid] = seq_len
        return True

    def pages_used(self) -> int:
        return self.num_pages - len(self.free_list)

    def slots_allocated(self) -> int:
        return self.pages_used() * self.block_size

    def utilization(self) -> float:
        a = self.slots_allocated()
        return sum(self.seq_lens.values()) / a if a > 0 else 0.0


# === COPY-ON-WRITE FOR BEAM SEARCH ===
# Multiple beams share the same prefix pages. Only when a beam diverges (writes a
# different token) do we copy the page.
#
# Intuition: fork() in Unix uses COW for process memory. Two child processes share
# the parent's pages until one writes, triggering a copy of just that page.

class CopyOnWriteManager:
    """Reference-counted page sharing for beam search."""

    def __init__(self, allocator: PagedAllocator) -> None:
        self.allocator = allocator
        self.ref_counts: dict[int, int] = {}  # physical_page -> ref count

    def fork(self, src: str, dst: str) -> bool:
        """Fork block table (like Unix fork). Shared pages get ref_count++."""
        if src not in self.allocator.block_tables:
            return False
        pages = self.allocator.block_tables[src]
        self.allocator.block_tables[dst] = list(pages)
        self.allocator.seq_lens[dst] = self.allocator.seq_lens[src]
        for p in pages:
            self.ref_counts[p] = self.ref_counts.get(p, 1) + 1
        return True

    def cow(self, rid: str, logical_page: int) -> int | None:
        """Copy shared page before writing. No-op if sole owner (ref_count==1).
        Returns physical page index to write to."""
        bt = self.allocator.block_tables.get(rid)
        if bt is None or logical_page >= len(bt):
            return None
        old = bt[logical_page]
        rc = self.ref_counts.get(old, 1)
        if rc <= 1:
            return old  # sole owner, write in place
        # Shared -- must copy (the COW trigger)
        new = self.allocator._alloc_page()
        if new is None:
            return None
        self.allocator.physical_memory[new] = list(self.allocator.physical_memory[old])
        bt[logical_page] = new
        self.ref_counts[old] = rc - 1
        self.ref_counts[new] = 1
        return new


# === PAGED ATTENTION COMPUTATION ===
# attention(q, K, V) = softmax(qK^T / sqrt(d)) V
# Identical math to standard attention; only difference is K,V rows live in
# non-contiguous physical pages, gathered through block table indirection.

def paged_attention(
    query: list[float], block_table: list[int],
    phys_mem: list[PhysicalPage], seq_len: int, block_size: int,
) -> list[float]:
    """Attention against paged KV-cache. Real vLLM fuses gather+attention in a
    single CUDA kernel for memory bandwidth efficiency."""
    d = len(query)
    scale = 1.0 / math.sqrt(d)

    # Gather K,V from scattered pages -- the core PagedAttention operation
    keys: list[list[float]] = []
    vals: list[list[float]] = []
    for pos in range(seq_len):
        phys_page = block_table[pos // block_size]
        k, v = phys_mem[phys_page][pos % block_size]
        keys.append(k)
        vals.append(v)

    # Standard scaled dot-product: scores_i = q . k_i / sqrt(d)
    scores = [dot(query, k) * scale for k in keys]
    weights = softmax(scores)
    out = [0.0] * d
    for w, v in zip(weights, vals):
        out = vec_add(out, vec_scale(v, w))
    return out


def contiguous_attention(
    query: list[float], keys: list[list[float]], vals: list[list[float]],
) -> list[float]:
    """Standard attention with contiguous KV arrays. Correctness reference."""
    d = len(query)
    scale = 1.0 / math.sqrt(d)
    weights = softmax([dot(query, k) * scale for k in keys])
    out = [0.0] * d
    for w, v in zip(weights, vals):
        out = vec_add(out, vec_scale(v, w))
    return out


# === CORRECTNESS VERIFICATION ===
# Proves paged attention = contiguous attention to floating-point precision.
# The block table is transparent indirection -- math doesn't know K,V are scattered.

def verify_correctness() -> None:
    print("=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("Paged attention must match contiguous attention exactly.")
    print("=" * 60)

    alloc = PagedAllocator(NUM_PHYSICAL_PAGES, PAGE_BLOCK_SIZE)
    alloc.allocate_request("v")
    all_k: list[list[float]] = []
    all_v: list[list[float]] = []
    seq_len = 13  # non-multiple of PAGE_BLOCK_SIZE to test remainder handling

    for _ in range(seq_len):
        k, v = rand_vec(HEAD_DIM), rand_vec(HEAD_DIM)
        all_k.append(k)
        all_v.append(v)
        alloc.append_token("v", k, v)

    q = rand_vec(HEAD_DIM)
    paged_out = paged_attention(q, alloc.block_tables["v"], alloc.physical_memory,
                                seq_len, PAGE_BLOCK_SIZE)
    contig_out = contiguous_attention(q, all_k, all_v)
    diff = max(abs(a - b) for a, b in zip(paged_out, contig_out))

    print(f"\n  Sequence length: {seq_len} ({math.ceil(seq_len / PAGE_BLOCK_SIZE)} pages)")
    print(f"  Max abs diff:    {diff:.2e}")
    print(f"  Result:          {'PASS' if diff < 1e-10 else 'FAIL'}")
    alloc.free_request("v")
    print()


# === SERVING SIMULATION ===
# Multiple concurrent requests with different lengths. The naive allocator reserves
# MAX_SEQ_LEN per request; the paged allocator grows incrementally. This is where
# the 2-4x throughput advantage of vLLM becomes visible.

def simulate_serving() -> None:
    print("=" * 60)
    print("SERVING SIMULATION")
    print(f"  {NUM_PHYSICAL_PAGES} pages x {PAGE_BLOCK_SIZE} slots = "
          f"{NUM_PHYSICAL_PAGES * PAGE_BLOCK_SIZE} total | "
          f"naive reserves {MAX_SEQ_LEN}/request")
    print("=" * 60)

    reqs: list[tuple[str, int, int, int]] = []  # (id, arrive, prompt, gen)
    for i in range(NUM_REQUESTS):
        reqs.append((chr(65 + i), i * 2, random.randint(2, 6), random.randint(3, MAX_GEN_LEN)))
    total_steps = max(a + p + g for _, a, p, g in reqs) + 1

    naive = NaiveAllocator(MAX_SEQ_LEN, NUM_PHYSICAL_PAGES * PAGE_BLOCK_SIZE)
    n_active: dict[str, int] = {}
    n_done: list[str] = []
    n_rejected: list[str] = []
    n_util: list[float] = []

    paged = PagedAllocator(NUM_PHYSICAL_PAGES, PAGE_BLOCK_SIZE)
    p_active: dict[str, int] = {}
    p_done: list[str] = []
    p_rejected: list[str] = []
    p_util: list[float] = []
    p_preempted: list[str] = []

    print("\n--- TIMELINE ---\n")
    for step in range(total_steps):
        ev: list[str] = []
        # Arrivals
        for rid, arrive, prompt, gen in reqs:
            if arrive != step:
                continue
            total = prompt + gen
            if naive.allocate(rid):
                n_active[rid] = total
            else:
                n_rejected.append(rid)
                ev.append(f"[naive] {rid} REJECTED")
            if paged.allocate_request(rid):
                p_active[rid] = total
            else:
                p_rejected.append(rid)

        # Resume preempted if space available
        for pid in list(p_preempted):
            if paged.resume(pid):
                p_preempted.remove(pid)
                for rid, _, prompt, gen in reqs:
                    if rid == pid:
                        p_active[pid] = max(0, prompt + gen - paged.seq_lens.get(pid, 0))
                ev.append(f"[paged] {pid} RESUMED")

        # Process tokens -- one per active request per step
        for rid in list(n_active):
            naive.append_token(rid)
            n_active[rid] -= 1
            if n_active[rid] <= 0:
                naive.free(rid)
                del n_active[rid]
                n_done.append(rid)
                ev.append(f"[naive] {rid} done")

        for rid in list(p_active):
            k, v = rand_vec(HEAD_DIM), rand_vec(HEAD_DIM)
            ok = paged.append_token(rid, k, v)
            if not ok:
                victims = [r for r in p_active if r != rid]
                if victims:
                    victim = victims[-1]
                    paged.preempt(victim)
                    p_preempted.append(victim)
                    del p_active[victim]
                    ev.append(f"[paged] PREEMPT {victim}")
                    ok = paged.append_token(rid, k, v)
            if ok:
                p_active[rid] -= 1
            if p_active.get(rid, 1) <= 0:
                freed = paged.free_request(rid)
                del p_active[rid]
                p_done.append(rid)
                ev.append(f"[paged] {rid} done (freed {freed} pg)")

        n_util.append(naive.utilization())
        p_util.append(paged.utilization())
        if ev:
            for e in ev:
                print(f"  Step {step:2d}: {e}")

    # Results
    avg_n = sum(n_util) / len(n_util) if n_util else 0
    avg_p = sum(p_util) / len(p_util) if p_util else 0

    print("\n" + "=" * 60)
    print("MEMORY UTILIZATION COMPARISON")
    print("=" * 60)
    print(f"\n  Naive:  peak={naive.peak_allocated} slots | "
          f"avg util={avg_n * 100:.1f}% | done={len(n_done)} rejected={len(n_rejected)}")
    print(f"  Paged:  peak={paged.peak_pages_used}/{NUM_PHYSICAL_PAGES} pages | "
          f"avg util={avg_p * 100:.1f}% | done={len(p_done)} rejected={len(p_rejected)}")
    if avg_n > 0:
        print(f"\n  Utilization improvement: {(avg_p - avg_n) / avg_n * 100:+.1f}%")
    print(f"  Naive rejected {len(n_rejected)} requests; paged served all {len(p_done)}.")


# === COPY-ON-WRITE BEAM SEARCH DEMO ===
# Beams share prefix pages and only copy when they diverge. Without COW, beam_width=4
# with 8-token prefix needs 4*2=8 pages. With COW: 2 shared + copies only at divergence.

def demo_cow() -> None:
    print("\n" + "=" * 60)
    print("COPY-ON-WRITE FOR BEAM SEARCH")
    print("=" * 60)

    alloc = PagedAllocator(NUM_PHYSICAL_PAGES, PAGE_BLOCK_SIZE)
    cow = CopyOnWriteManager(alloc)
    beam_width = 4

    # Build shared prefix
    alloc.allocate_request("b0")
    prefix_len = 8
    for _ in range(prefix_len):
        alloc.append_token("b0", rand_vec(HEAD_DIM), rand_vec(HEAD_DIM))

    prefix_pages = len(alloc.block_tables["b0"])
    print(f"\n  Prefix: {prefix_len} positions in {prefix_pages} pages")

    # Fork beams -- zero-copy, just share page references
    for i in range(1, beam_width):
        cow.fork("b0", f"b{i}")
    print(f"  After {beam_width} forks: {alloc.pages_used()} pages (shared, no copies)")
    print(f"  Without COW would need: {prefix_pages * beam_width} pages")

    # Each beam diverges -- COW triggers on shared last page
    for i in range(beam_width):
        cow.cow(f"b{i}", len(alloc.block_tables[f"b{i}"]) - 1)
        alloc.append_token(f"b{i}", rand_vec(HEAD_DIM), rand_vec(HEAD_DIM))

    pages_with_cow = alloc.pages_used()
    # Without COW: each beam needs full prefix copy + new page for divergent token
    pages_without = prefix_pages * beam_width + beam_width
    saved = pages_without - pages_with_cow

    print(f"  After divergence:    {pages_with_cow} pages (COW) vs {pages_without} (naive)")
    print(f"  Pages saved:         {saved} ({saved / pages_without * 100:.0f}%)")

    for i in range(beam_width):
        alloc.free_request(f"b{i}")

    print(f"\n  Longer shared prefixes = bigger savings. At prefix=1024 with beam=4,")
    print(f"  COW avoids duplicating ~3072 KV positions worth of pages.")


# === CONTINUOUS BATCHING DEMO ===
# Paging enables fine-grained memory release, synergistic with Orca-style continuous
# batching: new requests start as soon as pages free, no waiting for the full batch.

def demo_continuous_batching() -> None:
    print("\n" + "=" * 60)
    print("CONTINUOUS BATCHING")
    print("=" * 60)

    alloc = PagedAllocator(NUM_PHYSICAL_PAGES, PAGE_BLOCK_SIZE)
    b1 = [("S1", 5), ("S2", 3), ("S3", 7)]
    b2 = [("S4", 4), ("S5", 6)]
    max_b1 = max(t for _, t in b1)
    static = max_b1 + max(t for _, t in b2)
    print(f"\n  Static: batch 2 waits {max_b1} steps -> total {static}")

    active: dict[str, int] = {}
    done_at: dict[str, int] = {}
    for rid, toks in b1 + b2:
        alloc.allocate_request(rid)
        active[rid] = toks

    for step in range(30):
        if not active:
            break
        for rid in list(active):
            alloc.append_token(rid, rand_vec(HEAD_DIM), rand_vec(HEAD_DIM))
            active[rid] -= 1
            if active[rid] <= 0:
                freed = alloc.free_request(rid)
                del active[rid]
                done_at[rid] = step
                print(f"    Step {step}: {rid} done (freed {freed} pages)")

    cont = max(done_at.values())
    print(f"\n  Static: {static} | Continuous: {cont} | "
          f"Saved: {(static - cont) / static * 100:.0f}%")


# === INTERNAL FRAGMENTATION ANALYSIS ===
# The cost of paging: last page may be partially filled. But max waste per request
# is PAGE_BLOCK_SIZE-1 slots (3), vs MAX_SEQ_LEN-1 (19) for naive.

def analyze_fragmentation() -> None:
    print("\n" + "=" * 60)
    print("INTERNAL FRAGMENTATION ANALYSIS")
    print("=" * 60)
    print(f"\n  {'Seq':>4} {'Pg':>3} {'PgW':>4} {'PgF%':>6} {'NvW':>4} {'NvF%':>6}")
    print("  " + "-" * 30)
    for sl in [1, 4, 5, 8, 9, 13, 20]:
        pg = math.ceil(sl / PAGE_BLOCK_SIZE)
        pw = pg * PAGE_BLOCK_SIZE - sl
        nw = MAX_SEQ_LEN - sl
        print(f"  {sl:>4} {pg:>3} {pw:>4} {pw / (pg * PAGE_BLOCK_SIZE) * 100:>5.1f}% "
              f"{nw:>4} {nw / MAX_SEQ_LEN * 100:>5.1f}%")
    print(f"\n  Worst case: paged={PAGE_BLOCK_SIZE - 1} slots | "
          f"naive={MAX_SEQ_LEN - 1} slots | "
          f"naive is {(MAX_SEQ_LEN - 1) // (PAGE_BLOCK_SIZE - 1)}x worse")


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("  PAGED ATTENTION")
    print("  OS virtual memory principles applied to KV-cache management")
    print("=" * 60)
    print(f"\n  HEAD_DIM={HEAD_DIM}  N_HEADS={N_HEADS}  "
          f"PAGE_BLOCK_SIZE={PAGE_BLOCK_SIZE}  PAGES={NUM_PHYSICAL_PAGES}")
    print(f"  Total capacity: {NUM_PHYSICAL_PAGES * PAGE_BLOCK_SIZE} slots\n")

    t0 = time.time()
    verify_correctness()
    simulate_serving()
    demo_cow()
    demo_continuous_batching()
    analyze_fragmentation()
    print(f"\nTotal runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
