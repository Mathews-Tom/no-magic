"""
Monte Carlo Tree Search from first principles: the algorithm behind AlphaGo's game play —
using random simulations to build a search tree that balances exploration and exploitation.
"""
# Reference: Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
# (2006). Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016).

# === TRADEOFFS ===
# + Anytime algorithm: returns best-so-far at any point (more time = better results)
# + No domain-specific evaluation function needed (uses random rollouts)
# + Naturally balances exploration vs exploitation via UCB1
# - Random rollouts give noisy value estimates (high variance)
# - Exponential branching: struggles with very high branching factors
# - No learning between games (each search starts from scratch without neural guidance)
# WHEN TO USE: Game playing, planning problems with discrete actions,
#   any domain where you can simulate outcomes cheaply.
# WHEN NOT TO: Continuous action spaces, real-time decisions with no
#   simulation budget, or problems where rollouts are expensive.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# UCB1 exploration constant. sqrt(2) is theoretically optimal for rewards in [0,1]
# per the UCB1 theorem (Auer et al., 2002). Higher values explore more aggressively;
# lower values exploit known-good branches sooner.
EXPLORATION_CONSTANT = math.sqrt(2)

# Number of MCTS simulations (select → expand → rollout → backprop cycles) per move.
# More simulations = better play but slower. 1000 is overkill for Tic-Tac-Toe
# (game tree has ~255K nodes), but it demonstrates the algorithm clearly.
SIMULATIONS_PER_MOVE = 1000

# Demo configuration
NUM_GAMES_VS_RANDOM = 100   # games to estimate MCTS win rate vs random
NUM_GAMES_VS_MCTS = 100     # games to estimate draw rate in MCTS vs MCTS


# === GAME ENVIRONMENT: TIC-TAC-TOE ===

# Signpost: Tic-Tac-Toe is intentionally trivial. The point isn't the game — it's the
# search algorithm. Production MCTS (AlphaGo, MuZero) handles games with 10^170 states.
# The API below (legal_moves, make_move, is_terminal, get_winner) is the same interface
# you'd implement for Chess, Go, or any discrete game.

# Board representation: flat list of 9 cells. 0 = empty, 1 = player X, -1 = player O.
# Index mapping:
#   0 | 1 | 2
#   ---------
#   3 | 4 | 5
#   ---------
#   6 | 7 | 8

# All lines that constitute a win (rows, columns, diagonals)
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def make_board() -> list[int]:
    """Return an empty board. Player 1 (X) moves first."""
    return [0] * 9


def get_current_player(board: list[int]) -> int:
    """Determine whose turn it is from the board state.

    X (1) always goes first, so if the number of filled cells is even,
    it's X's turn; otherwise it's O's turn.
    """
    filled = sum(1 for cell in board if cell != 0)
    return 1 if filled % 2 == 0 else -1


def get_legal_moves(board: list[int]) -> list[int]:
    """Return indices of empty cells."""
    return [i for i in range(9) if board[i] == 0]


def make_move(board: list[int], action: int, player: int) -> list[int]:
    """Return a new board with the player's move applied. Does not mutate input."""
    new_board = board[:]
    new_board[action] = player
    return new_board


def get_winner(board: list[int]) -> int | None:
    """Check all win lines. Returns 1 (X wins), -1 (O wins), or None (no winner yet)."""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return None


def is_terminal(board: list[int]) -> bool:
    """Game is over if someone won or the board is full (draw)."""
    if get_winner(board) is not None:
        return True
    return len(get_legal_moves(board)) == 0


def board_to_string(board: list[int]) -> str:
    """Pretty-print the board for display."""
    symbols = {0: ".", 1: "X", -1: "O"}
    rows = []
    for r in range(3):
        row = " ".join(symbols[board[r * 3 + c]] for c in range(3))
        rows.append(row)
    return "\n".join(rows)


# === MCTS NODE ===

# The tree node tracks statistics accumulated across many simulations.
# Each node corresponds to a game state reached by a specific action from its parent.

class MCTSNode:
    """A node in the Monte Carlo search tree.

    Attributes:
        board: the game state at this node
        player: whose turn it is to move FROM this state
        parent: the node that led here (None for root)
        action: the action taken from parent to reach this node
        children: dict mapping action → child MCTSNode
        visit_count: N(s) — how many simulations passed through this node
        total_value: W(s) — cumulative reward from simulations through this node

    The value estimate Q(s) = W(s) / N(s) represents how promising this state is,
    averaged over all rollouts that visited it.
    """

    def __init__(
        self,
        board: list[int],
        player: int,
        parent: MCTSNode | None = None,
        action: int | None = None,
    ) -> None:
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        # Cache legal moves once — they never change for a given board state
        self._untried_actions = get_legal_moves(board)

    def is_fully_expanded(self) -> bool:
        """All legal moves have a corresponding child node."""
        return len(self._untried_actions) == 0

    def is_leaf(self) -> bool:
        """No children have been created yet (or this is a terminal state)."""
        return len(self.children) == 0

    def mean_value(self) -> float:
        """Q(s) = W(s) / N(s). The average value from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


# === MCTS PHASE 1: SELECTION ===

# Walk down the tree from root, choosing the child with the highest UCB1 score at
# each level, until we reach a node that hasn't been fully expanded or is terminal.
#
# UCB1 formula:  UCB1(s, a) = Q(s,a) + c * sqrt( ln(N(parent)) / N(child) )
#                              ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                           exploitation              exploration bonus
#
# Intuition: Q(s,a) favors moves that have worked well (exploitation). The square root
# term favors moves that haven't been tried much (exploration). As N(child) grows, the
# exploration bonus shrinks — the node has been "explored enough." The ln(N(parent))
# in the numerator ensures even well-explored nodes get revisited occasionally,
# because the parent's count keeps growing from other branches.
#
# This is the same explore/exploit tradeoff as the multi-armed bandit problem.
# UCB1 is provably optimal for bandits; MCTS applies it recursively down the tree.

def ucb1_score(parent: MCTSNode, child: MCTSNode, exploration_c: float) -> float:
    """Compute UCB1 score for a child node.

    Math-to-code:
        UCB1 = Q(s,a) + c * sqrt(ln(N_parent) / N_child)
        Q(s,a) = child.total_value / child.visit_count   (mean value)
        N_parent = parent.visit_count
        N_child = child.visit_count
        c = exploration_c

    The value is negated because the child's value is from the opponent's perspective.
    If the child reports a high win rate, that's BAD for the parent (the current player).
    """
    if child.visit_count == 0:
        # Unvisited children get infinite priority — must try everything at least once
        return float("inf")

    # Negate child value: child's wins are parent's losses
    # This is the key insight for adversarial search — what's good for the opponent
    # is bad for the current player.
    exploitation = -child.mean_value()

    # Exploration bonus: decreases as child is visited more, increases as parent is visited more
    exploration = exploration_c * math.sqrt(math.log(parent.visit_count) / child.visit_count)

    return exploitation + exploration


def select(node: MCTSNode, exploration_c: float) -> MCTSNode:
    """Phase 1: Walk down the tree using UCB1 until reaching an expandable or terminal node.

    Returns the node where expansion should happen. This is the "tree policy" —
    it determines how we navigate the known part of the search tree.
    """
    while not is_terminal(node.board) and node.is_fully_expanded():
        # Pick the child with the highest UCB1 score
        best_action = max(
            node.children,
            key=lambda a: ucb1_score(node, node.children[a], exploration_c),
        )
        node = node.children[best_action]
    return node


# === MCTS PHASE 2: EXPANSION ===

# Once selection reaches a node with untried actions, pick one and create a new child.
# This grows the tree by one node per simulation. Over many simulations, the tree
# expands toward the most promising branches (because selection steers toward them).

def expand(node: MCTSNode) -> MCTSNode:
    """Phase 2: Add a new child node for one untried action.

    Picks a random untried action (not the "best" one — we don't know which is best yet,
    that's what the simulations will determine). Creates the resulting board state and
    attaches a new child node.
    """
    # Pick a random untried action and remove it from the list
    action = node._untried_actions.pop(random.randrange(len(node._untried_actions)))

    # Create the new board state by applying the action
    child_board = make_move(node.board, action, node.player)
    child_player = -node.player  # other player's turn

    # Create and attach the child node
    child = MCTSNode(child_board, child_player, parent=node, action=action)
    node.children[action] = child

    return child


# === MCTS PHASE 3: SIMULATION (ROLLOUT) ===

# From the newly expanded node, play random moves until the game ends. The result
# tells us something about how good the position is. One rollout is noisy (random play
# is weak), but the law of large numbers means many rollouts average out to a reliable
# estimate of the position's true value.
#
# Signpost: This is where AlphaGo diverges from vanilla MCTS. Instead of random rollouts,
# AlphaGo uses a neural network to estimate the value of a position directly.
# This is far more accurate but requires training data and compute for the network.
# The pure MCTS approach here works for simple games; neural guidance is essential
# for games like Go where random play is nearly meaningless.

def simulate(board: list[int], player: int) -> int:
    """Phase 3: Play out the game randomly from the given position.

    Returns the winner (1 for X, -1 for O, 0 for draw). The randomness is the price
    we pay for not having domain knowledge — but it's a price that shrinks with more
    simulations (central limit theorem).
    """
    current_board = board[:]
    current_player = player

    while not is_terminal(current_board):
        legal = get_legal_moves(current_board)
        action = random.choice(legal)
        current_board = make_move(current_board, action, current_player)
        current_player = -current_player

    winner = get_winner(current_board)
    return winner if winner is not None else 0


# === MCTS PHASE 4: BACKPROPAGATION ===

# After a rollout produces a result, propagate it up the tree from the expanded node
# to the root. Every node on the path updates its visit count and total value.
#
# The value stored at each node is from that node's player's perspective. If player X
# won the rollout (result = 1), then nodes where it's X's turn get +1, and nodes where
# it's O's turn get -1. This ensures mean_value() correctly represents how good the
# position is for the player who's about to move.

def backpropagate(node: MCTSNode, result: int) -> None:
    """Phase 4: Update statistics from the expanded node back to root.

    Math-to-code:
        N(s) += 1           →  node.visit_count += 1
        W(s) += reward       →  node.total_value += value

    The reward is relative to the node's player: +1 if they won, -1 if they lost, 0 draw.
    """
    while node is not None:
        node.visit_count += 1
        # Value from this node's player's perspective
        node.total_value += result * node.player
        node = node.parent


# === MCTS SEARCH ===

# The full MCTS algorithm: run many simulations of select → expand → simulate → backprop,
# then pick the action whose root child has the most visits. We use visit count (not mean
# value) to choose the final action because it's more robust — a high visit count means
# the algorithm was consistently drawn to that branch, while a high mean value could come
# from a small sample.

def mcts_search(
    board: list[int],
    player: int,
    num_simulations: int = SIMULATIONS_PER_MOVE,
    exploration_c: float = EXPLORATION_CONSTANT,
) -> tuple[int, MCTSNode]:
    """Run MCTS from the given position and return (best_action, root_node).

    The root_node is returned for inspection (visualizing the search tree).
    """
    root = MCTSNode(board, player)

    for _ in range(num_simulations):
        # Phase 1: Selection — walk down tree using UCB1
        node = select(root, exploration_c)

        # Phase 2: Expansion — add a new child if the node isn't terminal
        if not is_terminal(node.board) and not node.is_fully_expanded():
            node = expand(node)

        # Phase 3: Simulation — random rollout from the new/selected node
        result = simulate(node.board, node.player)

        # Phase 4: Backpropagation — update statistics up to root
        backpropagate(node, result)

    # Choose action with most visits (robust child selection)
    # Signpost: Some implementations use max mean value for the final choice.
    # Most visits is standard because it's less susceptible to noise from
    # a few lucky rollouts on a rarely-visited branch.
    best_action = max(
        root.children,
        key=lambda a: root.children[a].visit_count,
    )

    return best_action, root


# === PLAYERS ===

def mcts_player(board: list[int], player: int) -> int:
    """MCTS-based player. Returns the chosen action."""
    action, _ = mcts_search(board, player)
    return action


def random_player(board: list[int], player: int) -> int:
    """Uniformly random player. The baseline opponent."""
    return random.choice(get_legal_moves(board))


# === GAME RUNNER ===

def play_game(
    player_x_fn: callable,
    player_o_fn: callable,
    verbose: bool = False,
) -> int:
    """Play a full game between two player functions. Returns winner (1, -1, or 0 for draw)."""
    board = make_board()

    if verbose:
        print("Starting position:")
        print(board_to_string(board))
        print()

    while not is_terminal(board):
        current_player = get_current_player(board)
        player_fn = player_x_fn if current_player == 1 else player_o_fn
        action = player_fn(board, current_player)
        board = make_move(board, action, current_player)

        if verbose:
            symbol = "X" if current_player == 1 else "O"
            row, col = divmod(action, 3)
            print(f"{symbol} plays position {action} (row {row}, col {col}):")
            print(board_to_string(board))
            print()

    winner = get_winner(board)
    if verbose:
        if winner is None:
            print("Result: Draw")
        else:
            print(f"Result: {'X' if winner == 1 else 'O'} wins")

    return winner if winner is not None else 0


# === TREE VISUALIZATION ===

def print_tree(node: MCTSNode, max_depth: int = 2, indent: int = 0) -> None:
    """Print the search tree showing visit counts and value estimates.

    Displays the tree structure that MCTS built, revealing where it focused
    its computational budget. High visit counts indicate moves the algorithm
    considers promising (or needs to investigate further).
    """
    prefix = "  " * indent
    if node.action is not None:
        pos_row, pos_col = divmod(node.action, 3)
        symbol = "X" if node.player == -1 else "O"  # player who MADE the move (parent's player)
        q_value = -node.mean_value()  # from parent's perspective
        print(
            f"{prefix}Action {node.action} ({symbol} at row {pos_row}, col {pos_col}): "
            f"visits={node.visit_count}, Q={q_value:+.3f}"
        )
    else:
        print(f"{prefix}Root: visits={node.visit_count}")

    if indent < max_depth:
        # Sort children by visit count (most visited first) for readable output
        sorted_children = sorted(
            node.children.values(),
            key=lambda c: c.visit_count,
            reverse=True,
        )
        for child in sorted_children:
            print_tree(child, max_depth, indent + 1)


def print_root_analysis(root: MCTSNode) -> None:
    """Print a summary of the root's children — the move candidates and their statistics."""
    print(f"Root visits: {root.visit_count}")
    print(f"Board state:")
    print(board_to_string(root.board))
    print()
    print(f"{'Action':<8} {'Visits':<8} {'Win Rate':<10} {'UCB1':<10}")
    print("-" * 36)

    sorted_children = sorted(
        root.children.items(),
        key=lambda pair: pair[1].visit_count,
        reverse=True,
    )
    for action, child in sorted_children:
        row, col = divmod(action, 3)
        win_rate = -child.mean_value()  # from root player's perspective
        ucb = ucb1_score(root, child, EXPLORATION_CONSTANT)
        print(f"({row},{col})    {child.visit_count:<8} {win_rate:<+10.3f} {ucb:<10.3f}")
    print()


# === EXPLORATION CONSTANT ANALYSIS ===

def analyze_exploration_constant() -> None:
    """Demonstrate how the exploration constant c affects MCTS behavior.

    c = 0    → pure exploitation: always picks the move with highest current win rate
    c = √2   → theoretically optimal balance for rewards in [0,1]
    c → ∞    → pure exploration: visits all moves roughly equally

    In practice, the optimal c depends on the domain. Games with many traps
    (like Go) benefit from higher exploration; simpler games converge faster
    with lower c.
    """
    print("=== EXPLORATION CONSTANT ANALYSIS ===")
    print()
    print("Running MCTS from empty board with different exploration constants...")
    print("Higher c → more exploration (visits spread across moves)")
    print("Lower c  → more exploitation (visits concentrated on best moves)")
    print()

    board = make_board()
    player = 1  # X moves first

    for c_value in [0.0, 0.5, EXPLORATION_CONSTANT, 3.0, 5.0]:
        _, root = mcts_search(board, player, num_simulations=500, exploration_c=c_value)

        # Collect visit counts for root children to show distribution
        visits = []
        for action in sorted(root.children):
            child = root.children[action]
            visits.append(child.visit_count)

        # Coefficient of variation measures how spread out the visits are
        mean_v = sum(visits) / len(visits)
        variance = sum((v - mean_v) ** 2 for v in visits) / len(visits)
        std_v = math.sqrt(variance)
        cv = std_v / mean_v if mean_v > 0 else 0.0

        visit_str = ", ".join(f"{v:3d}" for v in visits)
        c_label = f"c={c_value:.1f}"
        if abs(c_value - EXPLORATION_CONSTANT) < 0.01:
            c_label += " (√2)"
        print(f"  {c_label:<12} visits=[{visit_str}]  CV={cv:.2f}")

    print()
    print("CV (coefficient of variation): lower = more uniform, higher = more concentrated")
    print()


# === DEMO ===

def main() -> None:
    """Run the full MCTS demonstration."""
    start_time = time.time()

    print("=" * 60)
    print("MONTE CARLO TREE SEARCH — No-Magic Implementation")
    print("=" * 60)
    print()

    # --- Demo 1: MCTS vs Random ---
    print("=== DEMO 1: MCTS (X) vs RANDOM (O) ===")
    print(f"Playing {NUM_GAMES_VS_RANDOM} games with {SIMULATIONS_PER_MOVE} simulations/move...")
    print()

    wins = {1: 0, -1: 0, 0: 0}
    demo_start = time.time()

    for i in range(NUM_GAMES_VS_RANDOM):
        result = play_game(mcts_player, random_player)
        wins[result] += 1
        if (i + 1) % 20 == 0:
            elapsed = time.time() - demo_start
            print(f"  Game {i + 1}/{NUM_GAMES_VS_RANDOM} — "
                  f"X wins: {wins[1]}, O wins: {wins[-1]}, Draws: {wins[0]} "
                  f"({elapsed:.1f}s)")

    x_win_pct = wins[1] / NUM_GAMES_VS_RANDOM * 100
    o_win_pct = wins[-1] / NUM_GAMES_VS_RANDOM * 100
    draw_pct = wins[0] / NUM_GAMES_VS_RANDOM * 100

    print()
    print(f"Results: X (MCTS) wins {x_win_pct:.0f}%, O (random) wins {o_win_pct:.0f}%, "
          f"draws {draw_pct:.0f}%")

    # MCTS should dominate random play. Tic-Tac-Toe is simple enough that MCTS with
    # 1000 simulations plays near-perfectly.
    if x_win_pct >= 90:
        print("MCTS achieves >90% win rate against random — search works.")
    print()

    # --- Demo 2: MCTS vs MCTS ---
    print("=== DEMO 2: MCTS (X) vs MCTS (O) ===")
    print(f"Playing {NUM_GAMES_VS_MCTS} games...")
    # Use fewer simulations to keep runtime reasonable
    reduced_sims = 200

    def mcts_x(board: list[int], player: int) -> int:
        action, _ = mcts_search(board, player, num_simulations=reduced_sims)
        return action

    def mcts_o(board: list[int], player: int) -> int:
        action, _ = mcts_search(board, player, num_simulations=reduced_sims)
        return action

    wins2 = {1: 0, -1: 0, 0: 0}
    demo2_start = time.time()

    for i in range(NUM_GAMES_VS_MCTS):
        result = play_game(mcts_x, mcts_o)
        wins2[result] += 1
        if (i + 1) % 20 == 0:
            elapsed = time.time() - demo2_start
            print(f"  Game {i + 1}/{NUM_GAMES_VS_MCTS} — "
                  f"X wins: {wins2[1]}, O wins: {wins2[-1]}, Draws: {wins2[0]} "
                  f"({elapsed:.1f}s)")

    x_pct = wins2[1] / NUM_GAMES_VS_MCTS * 100
    o_pct = wins2[-1] / NUM_GAMES_VS_MCTS * 100
    d_pct = wins2[0] / NUM_GAMES_VS_MCTS * 100

    print()
    print(f"Results: X wins {x_pct:.0f}%, O wins {o_pct:.0f}%, draws {d_pct:.0f}%")

    # Tic-Tac-Toe is a solved game (optimal play = draw). Two competent MCTS agents
    # should draw most games. Some wins may occur due to the stochastic nature of
    # rollouts — with finite simulations, MCTS isn't perfect.
    print("Two MCTS agents should mostly draw — Tic-Tac-Toe is solved with perfect play.")
    print()

    # --- Demo 3: Sample game with verbose output ---
    print("=== DEMO 3: SAMPLE GAME (MCTS vs RANDOM, VERBOSE) ===")
    print()
    play_game(mcts_player, random_player, verbose=True)
    print()

    # --- Demo 4: Search tree visualization ---
    print("=== DEMO 4: SEARCH TREE ANALYSIS ===")
    print()
    print("Running MCTS from empty board (X to move)...")
    _, root = mcts_search(make_board(), 1, num_simulations=SIMULATIONS_PER_MOVE)
    print()

    print("--- Root children (move candidates) ---")
    print_root_analysis(root)

    # Intuition: The center (position 4) and corners (0, 2, 6, 8) should get more
    # visits than edges (1, 3, 5, 7) because they're stronger opening moves.
    # MCTS discovers this purely through simulation — no human knowledge encoded.
    print("Note: MCTS discovers that center and corners are strong openings")
    print("purely through random simulation — no game knowledge encoded.")
    print()

    print("--- Search tree (depth 2) ---")
    print_tree(root, max_depth=2)
    print()

    # --- Demo 5: Exploration constant analysis ---
    analyze_exploration_constant()

    # --- Demo 6: Connection to AlphaGo ---
    print("=== CONNECTION TO ALPHAGO ===")
    print()
    print("This implementation uses random rollouts for position evaluation.")
    print("AlphaGo replaced random rollouts with two neural networks:")
    print("  1. Policy network: guides which moves to explore (replaces random expansion)")
    print("  2. Value network: estimates position value (replaces random rollout)")
    print()
    print("The MCTS framework (select/expand/simulate/backprop) stays the same.")
    print("Neural guidance turns MCTS from 'adequate at simple games' into")
    print("'superhuman at Go' — the tree search structure amplifies the networks.")
    print()

    # --- Comparison to minimax ---
    print("=== MCTS vs MINIMAX ===")
    print()
    print("Minimax: explores the ENTIRE game tree uniformly to a fixed depth.")
    print("  Pro: optimal with perfect evaluation. Con: exponential in branching factor.")
    print()
    print("MCTS: focuses compute on PROMISING branches via UCB1 selection.")
    print("  Pro: scales to huge branching factors (Go: ~250). Con: stochastic, not optimal.")
    print()
    print("Minimax works when you can search the full tree (Chess endgames, Tic-Tac-Toe).")
    print("MCTS works when you can't (Go, large-scale planning, real-time games).")
    print()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
