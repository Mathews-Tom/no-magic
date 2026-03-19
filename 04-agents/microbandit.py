"""
Multi-armed bandits from first principles: three strategies — Epsilon-Greedy, UCB1, and
Thompson Sampling — compete on a synthetic Bernoulli bandit, revealing how uncertainty-aware
exploration beats naive randomness.
"""
# Reference: Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002)
# for UCB1. Thompson, "On the Likelihood that One Unknown Probability Exceeds Another in
# View of the Evidence of Two Samples" (1933) for Thompson Sampling.

# === TRADEOFFS ===
# + Bandits formalize the exploration/exploitation tradeoff with provable regret bounds
# + Thompson Sampling is Bayes-optimal and empirically dominates in practice
# + UCB1 gives deterministic, worst-case guarantees (O(K ln T) regret)
# - Assumes stationary reward distributions (non-stationary requires sliding windows)
# - Standard bandits ignore context (contextual bandits extend to features)
# - Regret bounds are asymptotic; finite-time performance varies
# WHEN TO USE: A/B testing, ad selection, clinical trials, any sequential decision
#   where you learn from outcomes and want to minimize wasted pulls.
# WHEN NOT TO: Environments where rewards change over time (use discounted/sliding-window
#   bandits), or where arm features matter (use contextual bandits / LinUCB).

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

NUM_ARMS = 10             # number of slot machines (actions)
NUM_ROUNDS = 10000        # how many pulls each strategy gets
EPSILON = 0.1             # exploration rate for epsilon-greedy
UCB_EXPLORATION_C = 2.0   # exploration constant inside the sqrt for UCB1

# In production, bandits power A/B testing (Optimizely), recommendation systems (Spotify),
# and LLM routing (deciding which model to call). The same math applies at every scale.


# === BANDIT ENVIRONMENT ===

# Each arm is a Bernoulli bandit: pulling it returns 1 with probability p_k, 0 otherwise.
# The agent does not know the true probabilities — it must learn them from experience.
# This is the simplest nontrivial reward model, but the strategies generalize to
# continuous rewards (Gaussian bandits) with minor modifications.

def make_bandit(num_arms: int) -> list[float]:
    """Generate true reward probabilities for each arm.

    Returns a list of num_arms floats in [0, 1]. These are the hidden ground truth
    that the agent tries to discover through interaction.
    """
    return [random.random() for _ in range(num_arms)]


def pull_arm(true_probabilities: list[float], arm: int) -> int:
    """Pull an arm and observe a Bernoulli reward.

    Returns 1 with probability true_probabilities[arm], 0 otherwise.
    This is the ONLY information the agent receives — it never sees the true probability.
    """
    return 1 if random.random() < true_probabilities[arm] else 0


def optimal_arm(true_probabilities: list[float]) -> tuple[int, float]:
    """Return the index and probability of the best arm.

    The optimal strategy (if you knew the probabilities) is to always pull this arm.
    Regret measures how far each strategy falls short of this oracle baseline.
    """
    best_arm = max(range(len(true_probabilities)), key=lambda i: true_probabilities[i])
    return best_arm, true_probabilities[best_arm]


# === EPSILON-GREEDY STRATEGY ===

# The simplest exploration strategy: with probability epsilon, pull a random arm
# (explore); otherwise pull the arm with the highest observed mean reward (exploit).
#
# Why epsilon-greedy is suboptimal: the exploration rate is fixed. Early on, when you
# know nothing, epsilon = 0.1 explores too LITTLE (you need more information). Late in
# the game, when you've identified the best arm, epsilon = 0.1 explores too MUCH (you're
# wasting 10% of pulls on arms you already know are bad). UCB1 and Thompson Sampling
# both solve this by exploring proportionally to uncertainty, which naturally decreases.

def epsilon_greedy_select(
    counts: list[int],
    total_rewards: list[float],
    epsilon: float,
) -> int:
    """Select an arm using the epsilon-greedy policy.

    With probability epsilon: pick uniformly at random (explore).
    With probability 1 - epsilon: pick the arm with the highest mean reward (exploit).

    Args:
        counts: number of times each arm has been pulled
        total_rewards: cumulative reward from each arm
        epsilon: exploration probability
    """
    num_arms = len(counts)

    if random.random() < epsilon:
        # Explore: uniform random
        return random.randrange(num_arms)

    # Exploit: pick the arm with the highest empirical mean
    # Arms that haven't been pulled get mean = 0, which biases against unexplored arms.
    # This is a known weakness — epsilon-greedy can get stuck if the first pull of the
    # best arm happens to return 0.
    best_arm = 0
    best_mean = -1.0
    for arm in range(num_arms):
        mean = total_rewards[arm] / counts[arm] if counts[arm] > 0 else 0.0
        if mean > best_mean:
            best_mean = mean
            best_arm = arm
    return best_arm


# === UCB1 STRATEGY ===

# Upper Confidence Bound: pull the arm with the highest optimistic estimate.
# Instead of exploring randomly, UCB1 explores arms it's uncertain about.
#
# UCB1 formula:
#   arm* = argmax_k [ x_bar_k + c * sqrt( ln(t) / n_k ) ]
#                      -------   --------------------------
#                      exploit         exploration bonus
#
# Math-to-code mapping:
#   x_bar_k  = total_rewards[k] / counts[k]    (empirical mean of arm k)
#   t        = total_pulls                       (total pulls across all arms)
#   n_k      = counts[k]                         (pulls of arm k)
#   c        = exploration constant               (sqrt(2) is theoretically optimal)
#
# Why the exploration term shrinks over time: n_k grows linearly with pulls of arm k,
# but ln(t) grows logarithmically with total pulls. So the ratio ln(t)/n_k → 0 as
# arm k is pulled more, meaning the confidence interval tightens. Arms that haven't
# been pulled much retain a large exploration bonus, ensuring they get revisited.
# This is the SAME formula that MCTS uses at each tree node (see micromcts.py) —
# MCTS is essentially UCB1 applied recursively down a search tree.

def ucb1_select(counts: list[int], total_rewards: list[float], total_pulls: int) -> int:
    """Select an arm using the UCB1 policy.

    Any arm with zero pulls gets infinite priority (must try everything once).
    After that, each arm's score = empirical mean + exploration bonus.

    Args:
        counts: number of times each arm has been pulled
        total_rewards: cumulative reward from each arm
        total_pulls: total number of pulls across all arms so far
    """
    num_arms = len(counts)

    # Phase 1: Pull each arm once before computing UCB scores.
    # UCB1 requires at least one observation per arm; ln(t)/0 is undefined.
    for arm in range(num_arms):
        if counts[arm] == 0:
            return arm

    # Phase 2: All arms have been pulled at least once. Compute UCB1 scores.
    best_arm = 0
    best_score = -1.0
    log_total = math.log(total_pulls)

    for arm in range(num_arms):
        empirical_mean = total_rewards[arm] / counts[arm]

        # Exploration bonus: sqrt(c * ln(t) / n_k)
        # Higher when arm k has been pulled less (relative to total pulls).
        # The ln ensures the bonus doesn't vanish too fast — even well-explored
        # arms get a slight nudge as total_pulls grows.
        exploration_bonus = math.sqrt(UCB_EXPLORATION_C * log_total / counts[arm])

        score = empirical_mean + exploration_bonus
        if score > best_score:
            best_score = score
            best_arm = arm

    return best_arm


# === THOMPSON SAMPLING STRATEGY ===

# Bayesian approach: maintain a probability distribution (posterior) over each arm's
# true reward rate. Sample from each posterior, pull the arm with the highest sample.
#
# For Bernoulli bandits, the conjugate prior is the Beta distribution:
#   Prior: Beta(alpha=1, beta=1) = Uniform(0, 1)    (no information)
#   After observing a success:  alpha += 1
#   After observing a failure:  beta  += 1
#   Posterior: Beta(alpha, beta)
#
# Why Thompson Sampling naturally balances exploration and exploitation:
# - An arm with few observations has a WIDE posterior (high variance). Samples from it
#   will sometimes be very high, triggering exploration of that arm.
# - An arm with many observations has a NARROW posterior (low variance). Its samples
#   cluster near the true mean. If the true mean is high, it gets exploited; if low,
#   it's ignored.
# - Uncertainty IS the exploration mechanism. No explicit exploration parameter needed.
#
# Thompson Sampling is what most modern bandit deployments use — it empirically
# outperforms UCB1 despite lacking UCB1's theoretical worst-case regret bound.
# Its Bayesian foundation also makes it easy to incorporate prior knowledge.

def thompson_select(alphas: list[float], betas: list[float]) -> int:
    """Select an arm using Thompson Sampling.

    Sample from each arm's Beta posterior, pull the arm with the highest sample.
    random.betavariate(a, b) is Python stdlib — no external dependencies needed.

    Math-to-code:
        For each arm k:
            theta_k ~ Beta(alpha_k, beta_k)
        arm* = argmax_k theta_k

    Args:
        alphas: Beta posterior alpha parameter per arm (successes + 1)
        betas: Beta posterior beta parameter per arm (failures + 1)
    """
    num_arms = len(alphas)
    best_arm = 0
    best_sample = -1.0

    for arm in range(num_arms):
        # Sample from the posterior belief about this arm's reward probability.
        # Early on (alpha=1, beta=1), this is uniform — maximum uncertainty.
        # After many observations, the sample concentrates near the true mean.
        sample = random.betavariate(alphas[arm], betas[arm])
        if sample > best_sample:
            best_sample = sample
            best_arm = arm

    return best_arm


# === TRAINING (BELIEF UPDATING) ===

# "Training" for bandits is the act of pulling arms and updating beliefs. There's no
# separate training/inference split like in supervised learning — the agent learns and
# acts simultaneously. This online learning property is what makes bandits practical
# for real-time decision-making.

def run_epsilon_greedy(
    true_probabilities: list[float],
    num_rounds: int,
    epsilon: float,
) -> tuple[list[float], list[int], list[float]]:
    """Run epsilon-greedy for num_rounds pulls.

    Returns:
        cumulative_regret: running sum of per-round regret at each timestep
        counts: final pull counts per arm
        total_rewards: final cumulative rewards per arm
    """
    num_arms = len(true_probabilities)
    _, optimal_reward = optimal_arm(true_probabilities)

    counts = [0] * num_arms
    total_rewards = [0.0] * num_arms
    cumulative_regret: list[float] = []
    running_regret = 0.0

    # Pull each arm once to initialize estimates
    for arm in range(num_arms):
        reward = pull_arm(true_probabilities, arm)
        counts[arm] += 1
        total_rewards[arm] += reward
        # Regret: difference between optimal arm's probability and pulled arm's probability
        # We use expected regret (based on true probabilities), not realized regret
        running_regret += optimal_reward - true_probabilities[arm]
        cumulative_regret.append(running_regret)

    for _ in range(num_arms, num_rounds):
        arm = epsilon_greedy_select(counts, total_rewards, epsilon)
        reward = pull_arm(true_probabilities, arm)
        counts[arm] += 1
        total_rewards[arm] += reward
        running_regret += optimal_reward - true_probabilities[arm]
        cumulative_regret.append(running_regret)

    return cumulative_regret, counts, total_rewards


def run_ucb1(
    true_probabilities: list[float],
    num_rounds: int,
) -> tuple[list[float], list[int], list[float]]:
    """Run UCB1 for num_rounds pulls.

    Returns:
        cumulative_regret: running sum of per-round regret at each timestep
        counts: final pull counts per arm
        total_rewards: final cumulative rewards per arm
    """
    num_arms = len(true_probabilities)
    _, optimal_reward = optimal_arm(true_probabilities)

    counts = [0] * num_arms
    total_rewards = [0.0] * num_arms
    cumulative_regret: list[float] = []
    running_regret = 0.0

    for round_num in range(num_rounds):
        total_pulls = round_num + 1
        arm = ucb1_select(counts, total_rewards, total_pulls)
        reward = pull_arm(true_probabilities, arm)
        counts[arm] += 1
        total_rewards[arm] += reward
        running_regret += optimal_reward - true_probabilities[arm]
        cumulative_regret.append(running_regret)

    return cumulative_regret, counts, total_rewards


def run_thompson(
    true_probabilities: list[float],
    num_rounds: int,
) -> tuple[list[float], list[int], list[float]]:
    """Run Thompson Sampling for num_rounds pulls.

    Returns:
        cumulative_regret: running sum of per-round regret at each timestep
        counts: final pull counts per arm
        total_rewards: final cumulative rewards per arm
    """
    num_arms = len(true_probabilities)
    _, optimal_reward = optimal_arm(true_probabilities)

    # Beta posterior parameters: start with Beta(1, 1) = Uniform prior (no prior knowledge)
    alphas = [1.0] * num_arms
    betas = [1.0] * num_arms
    counts = [0] * num_arms
    total_rewards = [0.0] * num_arms
    cumulative_regret: list[float] = []
    running_regret = 0.0

    for _ in range(num_rounds):
        arm = thompson_select(alphas, betas)
        reward = pull_arm(true_probabilities, arm)

        # Bayesian update: conjugate posterior update for Bernoulli likelihood + Beta prior
        # Posterior after observing reward r: Beta(alpha + r, beta + 1 - r)
        alphas[arm] += reward
        betas[arm] += 1 - reward

        counts[arm] += 1
        total_rewards[arm] += reward
        running_regret += optimal_reward - true_probabilities[arm]
        cumulative_regret.append(running_regret)

    return cumulative_regret, counts, total_rewards


# === RESULTS & ANALYSIS ===

def print_comparison_table(
    strategies: list[tuple[str, list[float], list[int], list[float]]],
    true_probabilities: list[float],
) -> None:
    """Print a summary table comparing all strategies.

    Shows cumulative regret, which arm each strategy converged to, and whether
    it found the true optimal arm.
    """
    best_arm_idx, best_arm_prob = optimal_arm(true_probabilities)

    print(f"{'Strategy':<20} {'Regret':>10} {'Best Arm':>10} {'Optimal?':>10}")
    print("-" * 52)

    for name, regret_curve, counts, _ in strategies:
        final_regret = regret_curve[-1]
        chosen_arm = max(range(len(counts)), key=lambda i: counts[i])
        is_optimal = chosen_arm == best_arm_idx
        marker = "YES" if is_optimal else "no"
        print(
            f"{name:<20} {final_regret:>10.1f} {chosen_arm:>10d} {marker:>10}"
        )

    print()
    print(f"Optimal arm: {best_arm_idx} (p = {best_arm_prob:.4f})")
    print(f"Lower regret = better. An oracle that always pulls the optimal arm has 0 regret.")


def print_arm_distribution(
    name: str,
    counts: list[int],
    total_rewards: list[float],
    true_probabilities: list[float],
) -> None:
    """Print how many times each arm was pulled and the estimated vs true probability."""
    num_arms = len(counts)
    total = sum(counts)
    best_arm_idx, _ = optimal_arm(true_probabilities)

    print(f"\n  {name} — arm pull distribution:")
    print(f"  {'Arm':>4} {'Pulls':>7} {'%':>7} {'Est p':>8} {'True p':>8} {'':>4}")
    print(f"  {'-' * 40}")

    for arm in range(num_arms):
        pull_pct = counts[arm] / total * 100
        est_prob = total_rewards[arm] / counts[arm] if counts[arm] > 0 else 0.0
        true_prob = true_probabilities[arm]
        marker = " <-- optimal" if arm == best_arm_idx else ""
        print(
            f"  {arm:>4d} {counts[arm]:>7d} {pull_pct:>6.1f}% "
            f"{est_prob:>8.4f} {true_prob:>8.4f}{marker}"
        )


def print_regret_ascii(
    strategies: list[tuple[str, list[float], list[int], list[float]]],
    num_rounds: int,
) -> None:
    """Print an ASCII visualization of cumulative regret over time.

    Samples the regret curve at regular intervals and draws a simple text chart.
    This is the key diagnostic: sublinear regret means the strategy is learning;
    linear regret means it's not converging to the optimal arm.
    """
    chart_width = 60
    chart_height = 15
    num_samples = chart_width

    # Sample regret curves at regular intervals
    sample_indices = [
        int(i * (num_rounds - 1) / (num_samples - 1)) for i in range(num_samples)
    ]

    sampled: list[tuple[str, list[float]]] = []
    for name, regret_curve, _, _ in strategies:
        points = [regret_curve[i] for i in sample_indices]
        sampled.append((name, points))

    # Find global max regret for y-axis scaling
    max_regret = max(max(pts) for _, pts in sampled)
    if max_regret == 0:
        max_regret = 1.0  # avoid division by zero

    # Build the chart row by row, top to bottom
    symbols = ["E", "U", "T"]  # Epsilon-greedy, UCB1, Thompson
    print(f"\n  Cumulative Regret over {num_rounds} rounds")
    print(f"  (E = Epsilon-Greedy, U = UCB1, T = Thompson Sampling)")
    print()

    for row in range(chart_height, -1, -1):
        threshold = max_regret * row / chart_height
        # Y-axis label
        if row == chart_height:
            label = f"{max_regret:>7.0f}"
        elif row == chart_height // 2:
            label = f"{max_regret / 2:>7.0f}"
        elif row == 0:
            label = f"{'0':>7}"
        else:
            label = "       "

        line = label + " |"
        for col in range(num_samples):
            cell_chars = []
            for strat_idx, (_, points) in enumerate(sampled):
                if points[col] >= threshold and (
                    row == 0 or points[col] < max_regret * (row + 1) / chart_height
                ):
                    cell_chars.append(symbols[strat_idx])

            if cell_chars:
                # If multiple strategies overlap, show all symbols
                line += cell_chars[0]
            else:
                line += " "
        print(f"  {line}")

    # X-axis
    print(f"  {'':>7} +{'-' * num_samples}")
    print(f"  {'':>7}  {'0':<{num_samples // 2}}{num_rounds}")
    print()

    # Legend with final regret values
    for strat_idx, (name, points) in enumerate(sampled):
        print(f"  {symbols[strat_idx]} = {name:<20} (final regret: {points[-1]:.1f})")
    print()


def print_regret_curve_compact(
    strategies: list[tuple[str, list[float], list[int], list[float]]],
    num_rounds: int,
) -> None:
    """Print regret at milestone intervals as a compact table.

    Complements the ASCII chart with exact numbers at key checkpoints.
    Sublinear growth (regret increases slower over time) indicates convergence.
    """
    milestones = [100, 500, 1000, 2000, 5000, num_rounds]
    milestones = [m for m in milestones if m <= num_rounds]

    print(f"  {'Strategy':<20}", end="")
    for m in milestones:
        print(f" {'t=' + str(m):>8}", end="")
    print()
    print(f"  {'-' * (20 + 9 * len(milestones))}")

    for name, regret_curve, _, _ in strategies:
        print(f"  {name:<20}", end="")
        for m in milestones:
            print(f" {regret_curve[m - 1]:>8.1f}", end="")
        print()
    print()


# === INFERENCE (FINAL POLICIES) ===

def demonstrate_final_policies(
    true_probabilities: list[float],
    eg_counts: list[int],
    eg_rewards: list[float],
    ucb_counts: list[int],
    ucb_rewards: list[float],
    ts_alphas: list[float],
    ts_betas: list[float],
) -> None:
    """Show what each strategy would do after learning — its final policy.

    This is the "inference" phase: given everything the agent has learned,
    which arm does it choose, and how confident is it?
    """
    num_arms = len(true_probabilities)
    best_arm_idx, best_arm_prob = optimal_arm(true_probabilities)

    print("After learning, each strategy's recommended action:\n")

    # Epsilon-Greedy: picks the arm with the highest empirical mean (ignoring epsilon)
    eg_means = [
        eg_rewards[k] / eg_counts[k] if eg_counts[k] > 0 else 0.0
        for k in range(num_arms)
    ]
    eg_best = max(range(num_arms), key=lambda k: eg_means[k])
    print(f"  Epsilon-Greedy: pull arm {eg_best} "
          f"(estimated p = {eg_means[eg_best]:.4f}, true p = {true_probabilities[eg_best]:.4f})")

    # UCB1: at convergence, the arm with the most pulls is the recommendation
    ucb_best = max(range(num_arms), key=lambda k: ucb_counts[k])
    ucb_mean = ucb_rewards[ucb_best] / ucb_counts[ucb_best] if ucb_counts[ucb_best] > 0 else 0.0
    print(f"  UCB1:           pull arm {ucb_best} "
          f"(estimated p = {ucb_mean:.4f}, true p = {true_probabilities[ucb_best]:.4f})")

    # Thompson Sampling: sample from posteriors and pick the highest
    # At convergence, the posterior of the best arm is tightly concentrated
    # around the true probability, so it wins the sample contest almost always.
    ts_means = [ts_alphas[k] / (ts_alphas[k] + ts_betas[k]) for k in range(num_arms)]
    ts_best = max(range(num_arms), key=lambda k: ts_means[k])
    ts_confidence = ts_alphas[ts_best] + ts_betas[ts_best] - 2  # total observations
    print(f"  Thompson:       pull arm {ts_best} "
          f"(posterior mean = {ts_means[ts_best]:.4f}, "
          f"true p = {true_probabilities[ts_best]:.4f}, "
          f"observations = {ts_confidence:.0f})")

    print(f"\n  Oracle:         pull arm {best_arm_idx} "
          f"(true p = {best_arm_prob:.4f})")

    # Show posterior uncertainty for Thompson Sampling's top 3 arms
    print("\n  Thompson Sampling posterior summary (top 3 arms by mean):")
    sorted_arms = sorted(range(num_arms), key=lambda k: ts_means[k], reverse=True)
    for arm in sorted_arms[:3]:
        alpha = ts_alphas[arm]
        beta = ts_betas[arm]
        mean = alpha / (alpha + beta)
        # Variance of Beta(a,b) = ab / ((a+b)^2 (a+b+1))
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)
        print(
            f"    Arm {arm}: Beta({alpha:.0f}, {beta:.0f}) "
            f"-> mean = {mean:.4f}, std = {std:.4f}"
        )
        # Intuition: the standard deviation shows remaining uncertainty. An arm with
        # std < 0.01 means the agent is very confident about its reward probability.
        # This is why Thompson Sampling stops exploring good arms — the posterior
        # narrows, so samples cluster near the mean instead of occasionally spiking high.


# === DEMO ===

def main() -> None:
    """Run the full multi-armed bandit demonstration."""
    start_time = time.time()

    print("=" * 60)
    print("MULTI-ARMED BANDITS — No-Magic Implementation")
    print("=" * 60)
    print()

    # --- Setup ---
    print("=== BANDIT SETUP ===")
    print()
    true_probs = make_bandit(NUM_ARMS)
    best_idx, best_prob = optimal_arm(true_probs)

    print(f"Generated {NUM_ARMS} Bernoulli arms with hidden reward probabilities:")
    for arm in range(NUM_ARMS):
        marker = " <-- optimal" if arm == best_idx else ""
        print(f"  Arm {arm}: p = {true_probs[arm]:.4f}{marker}")
    print()
    print(f"Optimal arm: {best_idx} (p = {best_prob:.4f})")
    print(f"Each strategy gets {NUM_ROUNDS} pulls to discover this.\n")

    # --- Run all strategies ---
    print("=== TRAINING (BELIEF UPDATING) ===")
    print()

    print(f"Running Epsilon-Greedy (epsilon={EPSILON})...")
    eg_start = time.time()
    eg_regret, eg_counts, eg_rewards = run_epsilon_greedy(
        true_probs, NUM_ROUNDS, EPSILON
    )
    print(f"  Done in {time.time() - eg_start:.2f}s")

    print(f"Running UCB1 (c={UCB_EXPLORATION_C})...")
    ucb_start = time.time()
    ucb_regret, ucb_counts, ucb_rewards = run_ucb1(true_probs, NUM_ROUNDS)
    print(f"  Done in {time.time() - ucb_start:.2f}s")

    print("Running Thompson Sampling...")
    ts_start = time.time()
    ts_regret, ts_counts, ts_rewards = run_thompson(true_probs, NUM_ROUNDS)
    print(f"  Done in {time.time() - ts_start:.2f}s")

    # Recover the Thompson Sampling posteriors for the inference demo
    # Re-derive from counts and rewards (alpha = 1 + successes, beta = 1 + failures)
    ts_alphas = [1.0 + ts_rewards[k] for k in range(NUM_ARMS)]
    ts_betas = [1.0 + ts_counts[k] - ts_rewards[k] for k in range(NUM_ARMS)]

    print()

    # --- Results ---
    strategies: list[tuple[str, list[float], list[int], list[float]]] = [
        ("Epsilon-Greedy", eg_regret, eg_counts, eg_rewards),
        ("UCB1", ucb_regret, ucb_counts, ucb_rewards),
        ("Thompson Sampling", ts_regret, ts_counts, ts_rewards),
    ]

    print("=== RESULTS & ANALYSIS ===")
    print()

    print_comparison_table(strategies, true_probs)
    print()

    # Regret milestones
    print("--- Cumulative regret at milestones ---")
    print_regret_curve_compact(strategies, NUM_ROUNDS)

    # ASCII regret visualization
    print("--- Regret curves (ASCII) ---")
    print_regret_ascii(strategies, NUM_ROUNDS)

    # Arm pull distributions
    print("--- Arm pull distributions ---")
    for name, _, counts, rewards in strategies:
        print_arm_distribution(name, counts, rewards, true_probs)
    print()

    # --- Inference ---
    print("=== INFERENCE (FINAL POLICIES) ===")
    print()
    demonstrate_final_policies(
        true_probs,
        eg_counts, eg_rewards,
        ucb_counts, ucb_rewards,
        ts_alphas, ts_betas,
    )

    # --- Key takeaways ---
    print()
    print("=== KEY INSIGHTS ===")
    print()
    print("1. Epsilon-Greedy has LINEAR regret: it wastes a fixed fraction of pulls")
    print("   exploring arms it already knows are bad. Regret ~ epsilon * T.")
    print()
    print("2. UCB1 has LOGARITHMIC regret: O(K ln T). The exploration bonus shrinks")
    print("   as arms are pulled, so it naturally stops exploring bad arms.")
    print()
    print("3. Thompson Sampling achieves the Lai-Robbins lower bound empirically,")
    print("   matching or beating UCB1 in practice. Its Bayesian uncertainty")
    print("   tracking is more informative than UCB1's confidence intervals.")
    print()
    print("4. All three converge to the optimal arm, but at different rates.")
    print("   The regret gap between them grows with more rounds — Thompson")
    print("   and UCB1 pull further ahead because their exploration is targeted.")
    print()
    print("5. Connection to MCTS (micromcts.py): MCTS applies UCB1 at each node")
    print("   of a search tree. Each node is a multi-armed bandit where the arms")
    print("   are the child actions. The same explore/exploit math scales from")
    print("   single-step decisions (bandits) to sequential planning (MCTS).")
    print()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
