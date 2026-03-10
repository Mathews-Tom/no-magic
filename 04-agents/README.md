# 04 — Agents & Planning

Algorithms that make decisions through search, simulation, and interaction with environments. These go beyond pattern recognition into active decision-making — the agent doesn't just classify or generate, it reasons about consequences and chooses actions.

## Scripts

Measured on Apple M-series, Python 3.12. Times are wall-clock.

| Script | Algorithm | Key Concept | Time |
|--------|-----------|-------------|------|
| `micromcts.py` | Monte Carlo Tree Search | UCB1 exploration + random rollouts | ~90s |

## What Connects These Scripts

These scripts implement algorithms that plan ahead by simulating possible futures. Where foundations scripts learn static mappings (input → output), agent scripts learn dynamic strategies (state → action → next state → ...). The common thread is **search**: exploring a space of possible action sequences to find ones that lead to good outcomes.

Key ideas shared across agent algorithms:

- **Exploration vs exploitation** — balancing known-good actions against untried ones
- **Simulation** — using a model (or random play) to estimate the value of future states
- **Credit assignment** — figuring out which past actions led to current rewards
- **Anytime computation** — returning a better answer the longer you let the algorithm run

## Future Candidates

| Algorithm | What It Would Teach | Notes |
|-----------|---------------------|-------|
| **Q-Learning** | Tabular reinforcement learning, Bellman equation | Classic RL, pairs well with MCTS |
| **Policy Gradient (REINFORCE)** | Gradient-based policy optimization | Bridges to PPO in 02-alignment |
| **Minimax + Alpha-Beta** | Adversarial search with pruning | Contrast with MCTS approach |
| **Multi-Armed Bandit** | Pure exploration/exploitation | UCB1 foundation without tree search |
