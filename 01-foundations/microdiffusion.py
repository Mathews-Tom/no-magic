"""
How images emerge from noise -- the denoising diffusion algorithm behind Stable Diffusion,
demonstrated on a 2D spiral. Train a model to predict noise, then iteratively remove it to
generate new samples from pure randomness.
"""
# Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
# https://arxiv.org/abs/2006.11239
# This 2D implementation preserves the exact DDPM algorithm used in Stable Diffusion,
# scaled down from billion-param U-Nets on images to ~1000-param MLPs on point clouds.

from __future__ import annotations

import math
import random

random.seed(42)


# === CONSTANTS ===

# Diffusion process hyperparameters
T = 100  # Number of diffusion timesteps -- production models use 1000, but 100 is
# enough to demonstrate the algorithm without excessive runtime
BETA_START = 0.0001  # Initial noise level (very small)
BETA_END = 0.02  # Final noise level (moderate corruption)
# Why linear schedule: simplest option; cosine schedules improve quality but add
# complexity. The linear schedule is enough to teach the core algorithm.

# Model architecture
HIDDEN_DIM = 64  # MLP hidden layer size (~1000 total params)
TIME_EMB_DIM = 32  # Sinusoidal timestep embedding dimension

# Training
NUM_EPOCHS = 8000  # Training iterations (each processes one random (x0, t) pair)
# 8000 updates for 6400 params ≈ 1.25 updates/param — sufficient for a smooth
# 2D spiral. Production DDPM trains for millions of updates.
LEARNING_RATE = 0.001  # Adam learning rate
NUM_SAMPLES = 800  # Number of training data points (2D spiral)

# Inference
NUM_GENERATED = 500  # Number of samples to generate for statistics


# === SYNTHETIC DATA GENERATION ===

def generate_spiral(num_points: int) -> list[tuple[float, float]]:
    """Generate a 2D spiral point cloud for training.

    The spiral grows linearly in radius as angle increases, creating a non-trivial
    distribution that's easy to verify visually via statistics (mean near origin,
    bounded variance). This tests whether the model learns structure, not just
    memorizes Gaussian blobs.
    """
    points = []
    for i in range(num_points):
        # Parametric spiral: r = theta / (2*pi) for one revolution per unit radius
        theta = (i / num_points) * 4 * math.pi  # 2 full revolutions
        r = theta / (2 * math.pi)

        # Convert to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        # Add small Gaussian noise to make it realistic (not a perfect curve)
        x += random.gauss(0, 0.05)
        y += random.gauss(0, 0.05)

        points.append((x, y))

    return points


# === NOISE SCHEDULE ===

def compute_noise_schedule(t_steps: int, beta_start: float, beta_end: float):
    """Precompute noise schedule coefficients for all timesteps.

    The forward diffusion process adds Gaussian noise at each timestep according to:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

    We precompute alpha_bar_t = prod(1 - beta_i for i in 1..t) to enable direct
    noising to arbitrary timesteps without sequential application:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    This closed-form jump is what makes diffusion models practical to train --
    we can sample any timestep in O(1) rather than stepping through all t-1 prior
    timesteps.

    Returns:
        betas: noise variance at each timestep (length T)
        alphas: 1 - beta at each timestep (length T)
        alpha_bars: cumulative product of alphas (length T)
        sqrt_alpha_bars: precomputed sqrt for noising (length T)
        sqrt_one_minus_alpha_bars: precomputed sqrt for noise coefficient (length T)
    """
    # Linear interpolation from beta_start to beta_end
    betas = [beta_start + (beta_end - beta_start) * t / (t_steps - 1)
             for t in range(t_steps)]

    alphas = [1.0 - b for b in betas]

    # Cumulative product: alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
    alpha_bars = []
    product = 1.0
    for alpha in alphas:
        product *= alpha
        alpha_bars.append(product)

    # Precompute square roots for forward process formula
    sqrt_alpha_bars = [math.sqrt(ab) for ab in alpha_bars]
    sqrt_one_minus_alpha_bars = [math.sqrt(1.0 - ab) for ab in alpha_bars]

    return betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars


# === TIMESTEP EMBEDDING ===

def sinusoidal_embedding(t: int, dim: int) -> list[float]:
    """Encode timestep t as a vector using sinusoidal positional encoding.

    For dimension i:
        emb[2*i]   = sin(t / 10000^(2*i/dim))
        emb[2*i+1] = cos(t / 10000^(2*i/dim))

    Why sinusoidal: provides a unique representation for each timestep with smooth
    interpolation between adjacent steps. Lower frequency components (early dims)
    change slowly with t, higher frequencies (later dims) change rapidly -- this
    multi-scale encoding helps the model distinguish nearby timesteps.

    Same embedding used for positional encoding in Transformers (Vaswani et al., 2017).
    """
    embedding = []
    for i in range(dim // 2):
        freq = 1.0 / (10000.0 ** (2 * i / dim))
        embedding.append(math.sin(t * freq))
        embedding.append(math.cos(t * freq))
    return embedding


# === NEURAL NETWORK (MANUAL IMPLEMENTATION) ===

def relu(x: float) -> float:
    """ReLU activation: max(0, x)."""
    return max(0.0, x)


def initialize_weights(input_dim: int, output_dim: int) -> list[list[float]]:
    """Initialize weight matrix with Xavier/Glorot uniform initialization.

    Scale = sqrt(6 / (input_dim + output_dim)) ensures variance is approximately
    preserved through layers, preventing gradients from vanishing or exploding
    during early training.
    """
    scale = math.sqrt(6.0 / (input_dim + output_dim))
    return [[random.uniform(-scale, scale) for _ in range(output_dim)]
            for _ in range(input_dim)]


def initialize_bias(dim: int) -> list[float]:
    """Initialize bias vector to zeros."""
    return [0.0 for _ in range(dim)]


class DenoisingMLP:
    """Small MLP that predicts noise given (noisy_data, timestep).

    Architecture:
        Input: [x_noisy (2D), t_embedding (TIME_EMB_DIM)] -> concat to (2+TIME_EMB_DIM)D
        Hidden1: (2+TIME_EMB_DIM) -> HIDDEN_DIM, ReLU
        Hidden2: HIDDEN_DIM -> HIDDEN_DIM, ReLU
        Output: HIDDEN_DIM -> 2 (predicted noise, no activation)

    In production diffusion models (Stable Diffusion), this MLP is replaced by a
    U-Net with billions of parameters, attention layers, and skip connections.
    But the training objective is identical: given x_t and t, predict epsilon.
    """

    def __init__(self):
        input_dim = 2 + TIME_EMB_DIM

        # Layer 1: input -> hidden
        self.w1 = initialize_weights(input_dim, HIDDEN_DIM)
        self.b1 = initialize_bias(HIDDEN_DIM)

        # Layer 2: hidden -> hidden
        self.w2 = initialize_weights(HIDDEN_DIM, HIDDEN_DIM)
        self.b2 = initialize_bias(HIDDEN_DIM)

        # Layer 3: hidden -> output (2D noise)
        self.w3 = initialize_weights(HIDDEN_DIM, 2)
        self.b3 = initialize_bias(2)

        # Adam optimizer state (first and second moments)
        self.m = {'w1': [[0.0]*HIDDEN_DIM for _ in range(input_dim)],
                  'b1': [0.0]*HIDDEN_DIM,
                  'w2': [[0.0]*HIDDEN_DIM for _ in range(HIDDEN_DIM)],
                  'b2': [0.0]*HIDDEN_DIM,
                  'w3': [[0.0]*2 for _ in range(HIDDEN_DIM)],
                  'b3': [0.0]*2}

        self.v = {'w1': [[0.0]*HIDDEN_DIM for _ in range(input_dim)],
                  'b1': [0.0]*HIDDEN_DIM,
                  'w2': [[0.0]*HIDDEN_DIM for _ in range(HIDDEN_DIM)],
                  'b2': [0.0]*HIDDEN_DIM,
                  'w3': [[0.0]*2 for _ in range(HIDDEN_DIM)],
                  'b3': [0.0]*2}

        self.step = 0  # Adam timestep counter

    def forward(self, x_noisy: tuple[float, float], t: int) -> tuple[float, float]:
        """Forward pass: (noisy_point, timestep) -> predicted_noise.

        Returns the intermediate activations for backprop.
        """
        # Concatenate noisy data and timestep embedding
        t_emb = sinusoidal_embedding(t, TIME_EMB_DIM)
        input_vec = [x_noisy[0], x_noisy[1]] + t_emb

        # Layer 1
        h1 = [sum(input_vec[i] * self.w1[i][j] for i in range(len(input_vec))) + self.b1[j]
              for j in range(HIDDEN_DIM)]
        h1_relu = [relu(h) for h in h1]

        # Layer 2
        h2 = [sum(h1_relu[i] * self.w2[i][j] for i in range(HIDDEN_DIM)) + self.b2[j]
              for j in range(HIDDEN_DIM)]
        h2_relu = [relu(h) for h in h2]

        # Layer 3 (output, no activation)
        output = [sum(h2_relu[i] * self.w3[i][j] for i in range(HIDDEN_DIM)) + self.b3[j]
                  for j in range(2)]

        # Cache for backprop
        self.cache = {
            'input': input_vec,
            'h1': h1,
            'h1_relu': h1_relu,
            'h2': h2,
            'h2_relu': h2_relu,
            'output': output
        }

        return tuple(output)

    def backward_and_update(self, grad_output: tuple[float, float], lr: float):
        """Backpropagate MSE gradient and update weights with Adam.

        Manual gradient computation through all layers. In production, this is
        handled by autograd frameworks (PyTorch, JAX), but implementing it manually
        reveals the mechanics.
        """
        # Gradient at output layer
        grad_out = list(grad_output)

        # Backprop through layer 3 (linear, no activation)
        grad_w3 = [[self.cache['h2_relu'][i] * grad_out[j] for j in range(2)]
                   for i in range(HIDDEN_DIM)]
        grad_b3 = grad_out
        grad_h2_relu = [sum(self.w3[i][j] * grad_out[j] for j in range(2))
                        for i in range(HIDDEN_DIM)]

        # Backprop through ReLU (derivative is 0 if input <= 0, else 1)
        grad_h2 = [grad_h2_relu[i] if self.cache['h2'][i] > 0 else 0.0
                   for i in range(HIDDEN_DIM)]

        # Backprop through layer 2
        grad_w2 = [[self.cache['h1_relu'][i] * grad_h2[j] for j in range(HIDDEN_DIM)]
                   for i in range(HIDDEN_DIM)]
        grad_b2 = grad_h2
        grad_h1_relu = [sum(self.w2[i][j] * grad_h2[j] for j in range(HIDDEN_DIM))
                        for i in range(HIDDEN_DIM)]

        # Backprop through ReLU
        grad_h1 = [grad_h1_relu[i] if self.cache['h1'][i] > 0 else 0.0
                   for i in range(HIDDEN_DIM)]

        # Backprop through layer 1
        input_dim = len(self.cache['input'])
        grad_w1 = [[self.cache['input'][i] * grad_h1[j] for j in range(HIDDEN_DIM)]
                   for i in range(input_dim)]
        grad_b1 = grad_h1

        # Adam update
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Update each parameter with Adam
        def adam_update(param, grad, m, v):
            """Apply Adam update rule to a single parameter array."""
            # First moment (exponential moving average of gradients)
            for i in range(len(param)):
                if isinstance(param[i], list):
                    for j in range(len(param[i])):
                        m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]
                        v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] ** 2
                        m_hat = m[i][j] / (1 - beta1 ** self.step)
                        v_hat = v[i][j] / (1 - beta2 ** self.step)
                        param[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                else:
                    m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                    m_hat = m[i] / (1 - beta1 ** self.step)
                    v_hat = v[i] / (1 - beta2 ** self.step)
                    param[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        adam_update(self.w1, grad_w1, self.m['w1'], self.v['w1'])
        adam_update(self.b1, grad_b1, self.m['b1'], self.v['b1'])
        adam_update(self.w2, grad_w2, self.m['w2'], self.v['w2'])
        adam_update(self.b2, grad_b2, self.m['b2'], self.v['b2'])
        adam_update(self.w3, grad_w3, self.m['w3'], self.v['w3'])
        adam_update(self.b3, grad_b3, self.m['b3'], self.v['b3'])


# === FORWARD DIFFUSION PROCESS ===

def add_noise(x0: tuple[float, float], t: int,
              sqrt_alpha_bars: list[float],
              sqrt_one_minus_alpha_bars: list[float]) -> tuple[tuple[float, float],
                                                                tuple[float, float]]:
    """Add noise to clean data point x0 at timestep t.

    Math-to-code mapping:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Where epsilon ~ N(0, I) is the noise we sample.

    Why this formula works: the forward process is a Markov chain that gradually
    converts any data distribution into a standard Gaussian. After T steps,
    x_T ≈ N(0, I) regardless of x_0. The sqrt coefficients ensure variance is
    preserved: Var(x_t) = alpha_bar_t + (1 - alpha_bar_t) = 1.

    Returns:
        x_t: noised data point
        epsilon: the noise that was added (ground truth for training)
    """
    # Sample noise from standard Gaussian
    epsilon = (random.gauss(0, 1), random.gauss(0, 1))

    # Apply closed-form noising formula
    coeff_signal = sqrt_alpha_bars[t]
    coeff_noise = sqrt_one_minus_alpha_bars[t]

    x_t = (coeff_signal * x0[0] + coeff_noise * epsilon[0],
           coeff_signal * x0[1] + coeff_noise * epsilon[1])

    return x_t, epsilon


# === TRAINING ===

def train(data: list[tuple[float, float]], model: DenoisingMLP,
          betas: list[float], alphas: list[float], alpha_bars: list[float],
          sqrt_alpha_bars: list[float], sqrt_one_minus_alpha_bars: list[float],
          num_epochs: int, lr: float):
    """Train the denoising model to predict noise.

    Training loop:
        1. Sample random data point x_0 from training set
        2. Sample random timestep t from [0, T-1]
        3. Sample noise epsilon ~ N(0, I)
        4. Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        5. Predict epsilon_pred = model(x_t, t)
        6. Loss = MSE(epsilon_pred, epsilon)
        7. Backprop and update weights

    Why predict noise instead of clean data: empirically, predicting the noise
    epsilon is easier to learn than predicting x_0. Intuitively, the noise is
    simpler (zero-mean Gaussian) than the data (complex spiral structure).

    Why MSE loss: the variational lower bound derivation of DDPM (Ho et al., 2020)
    shows that minimizing KL divergence between learned and true reverse process
    reduces to MSE between predicted and actual noise. The MSE is the correct
    loss for maximum likelihood training.
    """
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Random training sample
        x0 = random.choice(data)

        # Random timestep (0 to T-1)
        t = random.randint(0, T - 1)

        # Add noise using forward process
        x_t, epsilon_true = add_noise(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)

        # Predict noise
        epsilon_pred = model.forward(x_t, t)

        # MSE loss
        loss = ((epsilon_pred[0] - epsilon_true[0]) ** 2 +
                (epsilon_pred[1] - epsilon_true[1]) ** 2) / 2

        # Gradient of MSE: d/d(pred) [(pred - true)^2 / 2] = (pred - true)
        grad_loss = (epsilon_pred[0] - epsilon_true[0],
                     epsilon_pred[1] - epsilon_true[1])

        # Backprop and update
        model.backward_and_update(grad_loss, lr)

        # Print progress
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>5}/{num_epochs}  Loss: {loss:.6f}")


# === SAMPLING (REVERSE PROCESS) ===

def sample(model: DenoisingMLP, betas: list[float], alphas: list[float],
           alpha_bars: list[float]) -> tuple[float, float]:
    """Generate a new 2D point by iteratively denoising pure noise.

    Reverse process (sampling):
        Start with x_T ~ N(0, I)
        For t = T-1, T-2, ..., 0:
            epsilon_pred = model(x_t, t)
            x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred)
                      + sigma_t * z
            where z ~ N(0, I) for t > 0, z = 0 for t = 0

    Math-to-code mapping (from DDPM paper, Equation 11):
        mean = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred)
        variance = sigma_t^2 = beta_t (simplified; see signpost below)
        x_{t-1} ~ N(mean, variance)

    Signpost: the full DDPM reverse variance is more complex (interpolates between
    beta_t and a different formula). We use sigma_t = sqrt(beta_t) for simplicity.
    This produces slightly higher variance samples but preserves the core algorithm.
    """
    # Start from pure noise
    x = (random.gauss(0, 1), random.gauss(0, 1))

    # Iteratively denoise from t = T-1 down to t = 0
    for t in range(T - 1, -1, -1):
        # Predict noise at current timestep
        epsilon_pred = model.forward(x, t)

        # Compute mean of p(x_{t-1} | x_t)
        coeff = 1.0 / math.sqrt(alphas[t])
        noise_coeff = betas[t] / math.sqrt(1.0 - alpha_bars[t])

        mean_x = coeff * (x[0] - noise_coeff * epsilon_pred[0])
        mean_y = coeff * (x[1] - noise_coeff * epsilon_pred[1])

        # Add noise (except at t=0, final step is deterministic)
        if t > 0:
            sigma = math.sqrt(betas[t])
            z = (random.gauss(0, 1), random.gauss(0, 1))
            x = (mean_x + sigma * z[0], mean_y + sigma * z[1])
        else:
            x = (mean_x, mean_y)

    return x


# === STATISTICS ===

def compute_statistics(points: list[tuple[float, float]]) -> dict[str, float]:
    """Compute mean and standard deviation of 2D point cloud."""
    n = len(points)

    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n

    var_x = sum((p[0] - mean_x) ** 2 for p in points) / n
    var_y = sum((p[1] - mean_y) ** 2 for p in points) / n

    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y
    }


# === MAIN ===

if __name__ == "__main__":
    print("=" * 70)
    print("DENOISING DIFFUSION ON 2D SPIRAL")
    print("=" * 70)
    print()

    # Generate training data
    print("Generating training data...")
    data = generate_spiral(NUM_SAMPLES)
    train_stats = compute_statistics(data)
    print(f"  Training set: {NUM_SAMPLES} points")
    print(f"  Mean: ({train_stats['mean_x']:.4f}, {train_stats['mean_y']:.4f})")
    print(f"  Std:  ({train_stats['std_x']:.4f}, {train_stats['std_y']:.4f})")
    print()

    # Precompute noise schedule
    print("Computing noise schedule...")
    betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = \
        compute_noise_schedule(T, BETA_START, BETA_END)
    print(f"  Timesteps: {T}")
    print(f"  Beta range: [{BETA_START:.6f}, {BETA_END:.6f}]")
    print(f"  Alpha_bar at T-1: {alpha_bars[-1]:.6f}")
    print()

    # Initialize model
    print("Initializing denoising model...")
    model = DenoisingMLP()
    print(f"  Architecture: (2+{TIME_EMB_DIM}) -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> 2")
    print(f"  Parameters: ~{(2 + TIME_EMB_DIM) * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM * 2}")
    print()

    # Train
    train(data, model, betas, alphas, alpha_bars, sqrt_alpha_bars,
          sqrt_one_minus_alpha_bars, NUM_EPOCHS, LEARNING_RATE)
    print()

    # Generate samples
    print(f"Generating {NUM_GENERATED} samples from trained model...")
    generated = [sample(model, betas, alphas, alpha_bars)
                 for _ in range(NUM_GENERATED)]
    gen_stats = compute_statistics(generated)
    print(f"  Generated set: {NUM_GENERATED} points")
    print(f"  Mean: ({gen_stats['mean_x']:.4f}, {gen_stats['mean_y']:.4f})")
    print(f"  Std:  ({gen_stats['std_x']:.4f}, {gen_stats['std_y']:.4f})")
    print()

    # Compare distributions
    print("Distribution comparison:")
    print(f"  Training mean: ({train_stats['mean_x']:.4f}, {train_stats['mean_y']:.4f})")
    print(f"  Generated mean: ({gen_stats['mean_x']:.4f}, {gen_stats['mean_y']:.4f})")
    print()
    print(f"  Training std: ({train_stats['std_x']:.4f}, {train_stats['std_y']:.4f})")
    print(f"  Generated std: ({gen_stats['std_x']:.4f}, {gen_stats['std_y']:.4f})")
    print()

    # Quality metrics: compare generated vs training distributions
    # For means: use absolute difference normalized by training std (z-score)
    # because percentage difference is unstable when the mean is near zero.
    # For stds: percentage difference is appropriate since std is always positive.
    mean_x_zscore = abs(gen_stats['mean_x'] - train_stats['mean_x']) / train_stats['std_x']
    mean_y_zscore = abs(gen_stats['mean_y'] - train_stats['mean_y']) / train_stats['std_y']
    std_x_diff = abs(gen_stats['std_x'] - train_stats['std_x']) / train_stats['std_x'] * 100
    std_y_diff = abs(gen_stats['std_y'] - train_stats['std_y']) / train_stats['std_y'] * 100

    print("Quality metrics:")
    print(f"  Mean shift (in std units):  X={mean_x_zscore:.2f}σ  Y={mean_y_zscore:.2f}σ")
    print(f"  Std deviation difference:   X={std_x_diff:.1f}%  Y={std_y_diff:.1f}%")
    print()

    # Success: mean shift < 0.5σ and std within 20%
    success = mean_x_zscore < 0.5 and mean_y_zscore < 0.5 and std_x_diff < 20 and std_y_diff < 20
    if success:
        print("SUCCESS: Generated distribution matches training distribution.")
    else:
        print("PARTIAL: Generated distribution differs from training (may need more epochs).")
    print()

    print("=" * 70)
    print("ALGORITHM COMPLETE")
    print("=" * 70)
    print()
    print("What just happened:")
    print("  1. Generated a 2D spiral (non-trivial distribution)")
    print("  2. Trained a tiny MLP to predict noise at each diffusion timestep")
    print("  3. Sampled new points by starting from random noise and iteratively")
    print("     removing predicted noise for T steps")
    print()
    print("Mapping to image diffusion (Stable Diffusion, DALL-E):")
    print("  - 2D coordinates (x, y) -> RGB pixel values (R, G, B)")
    print("  - ~1000-param MLP -> ~1 billion-param U-Net with attention")
    print("  - 800 training points -> hundreds of millions of images")
    print("  - Gaussian noise on (x,y) -> Gaussian noise on (R,G,B)")
    print()
    print("The algorithm is identical. The scale is different.")
    print("This is how all modern image generation models work.")
