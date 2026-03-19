# MicroMCTS Challenges

Test your understanding of Monte Carlo Tree Search by predicting what happens in these scenarios. Try to work out the answer before revealing it.

---

### Challenge 1: UCB1 Value Negation

**Setup:** `ucb1_score` (lines 192-217) computes `exploit = -child.total_value / child.visit_count` (line 212). The negative sign is explained in the comment on lines 205-211. The game is a two-player zero-sum game where player 1 wins = +1.0 and player 2 wins = -1.0.

**Question:** From the perspective of the current node (player 1 to move), a child node has `total_value = -3.0` and `visit_count = 5`. What is the exploit term? Why does negating the child's value give the correct exploitation signal from the parent's perspective?

<details>
<summary>Reveal Answer</summary>

**Answer:** `exploit = -(-3.0) / 5 = 3.0 / 5 = 0.6`. This says "from player 1's perspective, this move leads to winning 60% of simulated games." Without negation, `exploit = -3.0 / 5 = -0.6` would say "this is a losing move" — exactly backwards.

**Why:** In a two-player zero-sum game, values are stored from the perspective of the player who just moved. When player 1 moves to a child node, the child node stores simulation results from player 2's turn. A child with `total_value = -3.0` means that in those simulations, player 2's outcomes summed to -3.0 — but player 2's loss is player 1's gain. The backpropagation in `backpropagate` (lines 306-319) alternates sign as it walks up the tree: `node.total_value += value; value = -value` (lines 315-316). This ensures each node's `total_value` reflects the cumulative outcome from the perspective of the player who is NOT about to move at that node. When the parent selects among children, it correctly negates the child value to convert from "child's player's perspective" back to "parent's player's perspective."

**Script reference:** `04-agents/micromcts.py`, lines 192-217 (ucb1_score with negation), lines 205-211 (negation rationale comment), lines 306-319 (backpropagate alternating signs), lines 312-316 (value negation in backprop loop)

</details>

---

### Challenge 2: Visit Count vs Mean Value for Final Action

**Setup:** `mcts_search` (lines 330-365) selects the final action by `best_child = max(root.children, key=lambda c: c.visit_count)` (lines 360-363). The comment on lines 357-362 explains why visit count is preferred over mean value.

**Question:** Suppose after 100 simulations, child A has `visit_count=80, total_value=48.0` (mean 0.60) and child B has `visit_count=20, total_value=14.0` (mean 0.70). UCB1 explored B less. Which child is selected, and why is the higher-mean child not necessarily better?

<details>
<summary>Reveal Answer</summary>

**Answer:** Child A is selected (80 visits > 20 visits). Child B's higher mean (0.70 vs 0.60) is based on only 20 samples and has high variance in the estimate. After 100 total simulations, UCB1 has already judged that child A deserves 4x more exploration — this is evidence that A is the better move, not just a statistical artifact.

**Why:** The UCB1 algorithm is designed to balance exploration and exploitation throughout the search budget. If child B were truly better than A, UCB1's exploration bonus would have directed more simulations to B to verify this — but instead it kept returning to A. By the end of the search, visit counts reflect the algorithm's confidence: high visit count means UCB1 repeatedly found it worth exploring that subtree. Mean value from few samples has high variance; mean value from many samples has low variance. `max(visit_count)` is equivalent to "most robustly explored" — a robust choice that is less likely to be a statistical fluke. The comment on line 358 states this explicitly: "most-visited child (robust under simulation noise)." This is in contrast to the exploration phase where UCB1 deliberately visits less-explored children to potentially find better moves.

**Script reference:** `04-agents/micromcts.py`, lines 330-365 (mcts_search), lines 357-363 (final child selection by visit count), lines 358-362 (robust selection comment), lines 192-217 (ucb1_score during search uses mean + exploration bonus)

</details>

---

### Challenge 3: Exploration Constant Sensitivity

**Setup:** `EXPLORATION_CONSTANT = math.sqrt(2)` (line 34). The UCB1 formula is `score = exploit + c * sqrt(log(parent.visit_count) / child.visit_count)` (line 215). `analyze_exploration_constant` (lines 475-519) tests multiple values of `c` and reports the coefficient of variation (CV) of visit counts as a measure of how evenly the tree is explored.

**Question:** With `c = 0.0` (pure exploitation), what does `ucb1_score` reduce to? After 100 simulations starting from a fresh tree, where do all simulations go? Contrast with `c = 10.0` (heavy exploration).

<details>
<summary>Reveal Answer</summary>

**Answer:** With `c = 0.0`: `score = exploit = -child.total_value / child.visit_count` (pure mean value). The first simulation from root visits a random unvisited child (all have visit_count=0, handled by the `if child.visit_count == 0` early return on line 209 returning infinity). After the first rollout assigns initial values, subsequent simulations always pick the child with the highest mean value — pure greedy. All 100 simulations likely funnel into 1-2 children, ignoring potentially better unexplored moves. With `c = 10.0`: the exploration term `10 * sqrt(log(N) / n_i)` dominates for all children with small visit counts. Simulations distribute nearly uniformly across all children until visit counts equalize, spending most of the budget exploring rather than exploiting the best-seen moves.

**Why:** The exploration constant `c = sqrt(2)` is theoretically optimal for the UCB1 bandit problem when rewards are bounded in `[0, 1]` and you want to minimize cumulative regret. In tree search, rewards come from game outcomes (±1), so the theoretical justification applies. With `c = 0`, the algorithm is greedy and susceptible to early bad luck: if a good move happens to lose its first two simulations due to random play, it may never be re-explored. With very large `c`, the algorithm wastes its budget exploring obviously bad moves rather than refining estimates of good ones. The `analyze_exploration_constant` function in the script demonstrates this empirically — low CV (even distribution) with high `c`, high CV (concentrated) with low `c`.

**Script reference:** `04-agents/micromcts.py`, lines 34 (EXPLORATION_CONSTANT), lines 192-217 (ucb1_score formula), lines 207-210 (unvisited child infinity handling), lines 475-519 (analyze_exploration_constant comparing CV at different c values)

</details>

---

### Challenge 4: Simulation Depth and Random Playout Bias

**Setup:** `simulate` (lines 276-293) plays out a game randomly from the expanded node until terminal: `action = random.choice(legal_actions)` (line 284). The comment on lines 278-282 explains this is "fast rollout" using pure random play. `backpropagate` (lines 306-319) then propagates the terminal value back up.

**Question:** In a complex game where random play performs very poorly (most random games end in a loss for the first player), how does this bias the MCTS estimates? What alternative to random simulation would produce more accurate value estimates at the cost of more computation?

<details>
<summary>Reveal Answer</summary>

**Answer:** Biased random play gives systematically pessimistic value estimates for positions where good play is required. UCB1 still finds relatively better moves (because all moves are evaluated with the same biased simulator), but the absolute values are unreliable. A position that wins with probability 0.9 under optimal play might estimate at 0.4 under random play — MCTS can still prefer it over a position at 0.2, but the values are not probabilities.

**Why:** Random simulation is unbiased only if the game outcome under random play correlates with outcome under optimal play. In Go and complex strategy games, this correlation is low — random play leads to incoherent sequences that don't resemble how the game would actually unfold. This is why AlphaGo replaced random rollouts with a learned value network: instead of simulating to terminal, the value network predicts `V(s)` for the expanded node directly. This requires a trained neural network but eliminates the random-play bias entirely. The simulation depth also matters: `simulate` here plays until terminal, but "shallow" simulations that run for only a few moves and then use a heuristic evaluation can be more accurate than deep random play in some games. The comment on line 281 acknowledges this limitation and notes that production MCTS systems (AlphaZero, MuZero) use learned value functions instead.

**Script reference:** `04-agents/micromcts.py`, lines 276-293 (simulate with random rollout), lines 278-282 (random playout comment and limitation), lines 306-319 (backpropagate propagating simulation value), lines 330-365 (mcts_search outer loop)

</details>
