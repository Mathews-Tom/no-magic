"""
How to learn a compressed, generative representation of data — the reparameterization
trick demystified, in pure Python with zero dependencies.
"""
# Reference: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013).
# https://arxiv.org/abs/1312.6114
# The reparameterization trick (z = μ + σ * ε) is the core contribution that makes
# VAEs trainable — before this, sampling operations blocked gradient flow.

from __future__ import annotations

import math
import random

random.seed(42)


# === CONSTANTS ===

LATENT_DIM = 2          # Size of latent space z. 2D for easy interpretation.
HIDDEN_DIM = 16         # Hidden layer size for encoder and decoder MLPs.
LEARNING_RATE = 0.001   # Adam learning rate.
BETA = 1.0              # KL weight in ELBO. β=1 is standard VAE, β>1 encourages disentanglement.
NUM_EPOCHS = 1000       # Training iterations.
BATCH_SIZE = 16         # Minibatch size for stochastic gradient descent.

# Signpost: production VAEs use convolutional encoders/decoders for images. This MLP on
# 2D data demonstrates the same principles (ELBO, reparameterization, latent interpolation)
# at 1% of the complexity. The algorithm is identical — only the encoder/decoder architecture
# changes when scaling to pixels.


# === SYNTHETIC DATA GENERATION ===

def generate_data(n_points: int = 800) -> list[list[float]]:
    """Generate a mixture of 2D Gaussians for training.

    We create 4 clusters at different positions so the VAE has interesting structure
    to learn. A single Gaussian would be trivial (the VAE would learn the mean/variance
    directly). Multiple modes force the latent space to organize meaningfully.
    """
    # Four cluster centers in 2D space, arranged in a rough square.
    centers = [
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
    ]
    variance = 0.3  # Small variance so clusters are distinct but not separated.

    data = []
    for _ in range(n_points):
        # Randomly select a cluster, then sample from N(center, variance).
        center = random.choice(centers)
        x = random.gauss(center[0], math.sqrt(variance))
        y = random.gauss(center[1], math.sqrt(variance))
        data.append([x, y])

    return data


# === MLP UTILITIES ===
# We use plain float arrays (not the Value autograd class) because VAE training with
# scalar autograd hits the 7-minute runtime limit. Manual gradient computation keeps
# the core VAE algorithm visible while meeting runtime constraints.

def matrix_multiply(a: list[list[float]], b: list[float]) -> list[float]:
    """Multiply matrix a (m×n) by vector b (n,) to get vector (m,)."""
    return [sum(a[i][j] * b[j] for j in range(len(b))) for i in range(len(a))]


def relu(x: list[float]) -> list[float]:
    """ReLU activation: max(0, x) element-wise."""
    return [max(0.0, val) for val in x]


def relu_grad(x: list[float]) -> list[float]:
    """Gradient of ReLU: 1 if x > 0, else 0."""
    return [1.0 if val > 0 else 0.0 for val in x]


def add_bias(x: list[float], b: list[float]) -> list[float]:
    """Add bias vector b to x element-wise."""
    return [x[i] + b[i] for i in range(len(x))]


def init_weights(rows: int, cols: int) -> list[list[float]]:
    """Initialize weights using Xavier/Glorot initialization.

    Scale by sqrt(2 / (rows + cols)) for stable gradients. Without this,
    deep networks suffer from vanishing/exploding activations.
    """
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def init_bias(size: int) -> list[float]:
    """Initialize bias to zeros (standard practice)."""
    return [0.0 for _ in range(size)]


# === ENCODER MLP ===

def encoder_forward(
    x: list[float],
    w1: list[list[float]],
    b1: list[float],
    w_mean: list[list[float]],
    b_mean: list[float],
    w_logvar: list[list[float]],
    b_logvar: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Encoder: input (2D) → hidden (ReLU) → (mean, log_var) in latent space.

    Why two output heads (mean and log_var)? The encoder parameterizes the approximate
    posterior q(z|x) as a Gaussian. Instead of directly outputting variance σ²,
    we output log(σ²) because:
    - Variance must be positive, but the network output is unconstrained
    - Optimizing log_var avoids numerical issues (exp is always positive)
    - Standard parameterization in variational inference

    Returns: (hidden_state, mean, log_var) where mean and log_var both have shape (latent_dim,)
    """
    # Input → hidden layer with ReLU activation
    hidden = relu(add_bias(matrix_multiply(w1, x), b1))

    # Hidden → mean of latent distribution (unconstrained)
    mean = add_bias(matrix_multiply(w_mean, hidden), b_mean)

    # Hidden → log variance of latent distribution (unconstrained)
    # We'll exponentiate this when computing the reparameterization trick, so
    # log_var can be any real number, and exp(0.5 * log_var) = σ will be positive.
    log_var = add_bias(matrix_multiply(w_logvar, hidden), b_logvar)

    return hidden, mean, log_var


# === REPARAMETERIZATION TRICK ===
# This is the core pedagogical point of the script. Everything else is machinery;
# this single function is what makes VAEs trainable.

def reparameterize(mean: list[float], log_var: list[float]) -> list[float]:
    """Sample z from q(z|x) via the reparameterization trick.

    THE CORE INSIGHT — why this works:

    We want to sample z ~ N(μ, σ²) where μ = encoder_mean(x) and σ² = exp(encoder_log_var(x)).
    Naively, we'd write:
        z = random.gauss(mean, sigma)

    But this breaks gradient flow. The randomness blocks backpropagation — gradients
    can't flow through a sampling operation because the derivative of "sample a random
    number" is undefined.

    The reparameterization trick solves this:
        ε ~ N(0,1)           # sample from standard normal (no parameters)
        σ = exp(0.5 * log_var)    # deterministic function of log_var
        z = μ + σ * ε        # deterministic function of μ, log_var, and external ε

    Now the randomness (ε) is external to the computation graph. Gradients flow through
    μ and log_var (which are deterministic network outputs), but not through ε. This
    makes the sampling operation differentiable.

    Math-to-code mapping:
        μ: mean (encoder output)
        log(σ²): log_var (encoder output)
        σ: exp(0.5 * log_var)
        ε: epsilon (sampled externally)
        z: mean + sigma * epsilon

    Before Kingma & Welling (2013), people used REINFORCE-style gradient estimators
    which have much higher variance and require many more samples. The reparameterization
    trick is what made VAEs practical.
    """
    epsilon = [random.gauss(0, 1) for _ in range(len(mean))]

    # σ = exp(0.5 * log_var). We use 0.5 * log_var instead of log_var because
    # log_var = log(σ²), so 0.5 * log_var = log(σ).
    sigma = [math.exp(0.5 * lv) for lv in log_var]

    # z = μ + σ * ε
    z = [mean[i] + sigma[i] * epsilon[i] for i in range(len(mean))]

    return z


# === DECODER MLP ===

def decoder_forward(
    z: list[float],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> tuple[list[float], list[float]]:
    """Decoder: latent z → hidden (ReLU) → reconstructed output (2D).

    Returns: (hidden_state, output) where output is the reconstructed 2D point.
    """
    # Latent → hidden layer with ReLU activation
    hidden = relu(add_bias(matrix_multiply(w1, z), b1))

    # Hidden → output (2D reconstructed point, no activation)
    # We don't apply an activation because the data is unconstrained (can be negative).
    output = add_bias(matrix_multiply(w2, hidden), b2)

    return hidden, output


# === ELBO LOSS ===

def compute_loss(
    x: list[float],
    mean: list[float],
    log_var: list[float],
    x_recon: list[float],
    beta: float,
) -> tuple[float, float, float]:
    """Compute the Evidence Lower Bound (ELBO) loss.

    ELBO = reconstruction_loss + β * KL_divergence

    WHY THIS LOSS FUNCTION:
    VAEs maximize the log-likelihood log p(x) of the data. We can't compute this directly,
    so we maximize a lower bound (ELBO) instead. Maximizing ELBO ≈ maximizing log p(x).

    The ELBO decomposes into two terms:
    1. Reconstruction loss: how well the decoder reconstructs x from z
       We use MSE (mean squared error): ||x - decoder(z)||²
    2. KL divergence: how different q(z|x) is from the prior p(z) = N(0,I)
       This regularizes the latent space to be smooth and continuous.

    Why KL divergence? It forces the latent space to have nice properties:
    - Mean near 0, variance near 1 (matching the prior)
    - Smooth transitions between nearby z values
    - We can sample from N(0,I) at inference time and decode to generate new data

    Without KL regularization, the encoder would learn arbitrary, discontinuous
    mappings (e.g., cluster 1 → z=[100,0], cluster 2 → z=[-50,200]) and the decoder
    would overfit. The latent space would be useless for generation because random
    samples from N(0,1) would decode to garbage.
    """
    # Reconstruction loss: MSE between input and reconstructed output
    reconstruction_loss = sum((x[i] - x_recon[i]) ** 2 for i in range(len(x)))

    # KL divergence KL(q(z|x) || p(z)) for diagonal Gaussians.
    # When both q and p are Gaussian, KL has a closed form (no sampling needed):
    #   KL(N(μ, σ²) || N(0,I)) = 0.5 * sum(1 + log(σ²) - μ² - σ²)
    #                           = 0.5 * sum(1 + log_var - mean² - exp(log_var))
    #
    # Math-to-code mapping:
    #   μ: mean
    #   σ²: exp(log_var)
    #   log(σ²): log_var
    #
    # Why this has a closed form: both distributions are Gaussian, and the KL between
    # two Gaussians is analytic (no integrals to compute).
    #
    # KL clamping: we clamp log_var to [-5, 5] to prevent exp(log_var) explosion.
    # exp(5) = 148 (reasonable variance); exp(10) = 22,026 (KL blows up and gradients
    # vanish). Without clamping, the encoder can output extreme log_var values that
    # cause numerical instability.
    kl_loss = 0.0
    for i in range(len(mean)):
        # Clamp log_var to prevent numerical explosion
        clamped_lv = max(min(log_var[i], 5.0), -5.0)
        kl_loss += 1.0 + clamped_lv - mean[i] ** 2 - math.exp(clamped_lv)
    kl_loss = -0.5 * kl_loss  # negative because we derived the formula with a minus sign

    # Total ELBO loss (we minimize negative ELBO, which is equivalent to maximizing ELBO)
    # β-weighting: β=1 is standard VAE. β>1 (β-VAE) encourages disentangled representations
    # by penalizing KL more heavily, trading off reconstruction quality for latent space
    # structure. β<1 emphasizes reconstruction at the cost of a messier latent space.
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# === MANUAL GRADIENT COMPUTATION ===

def backward_and_update(
    x: list[float],
    mean: list[float],
    log_var: list[float],
    z: list[float],
    x_recon: list[float],
    enc_hidden: list[float],
    dec_hidden: list[float],
    # Encoder weights
    enc_w1: list[list[float]],
    enc_b1: list[float],
    enc_w_mean: list[list[float]],
    enc_b_mean: list[float],
    enc_w_logvar: list[list[float]],
    enc_b_logvar: list[float],
    # Decoder weights
    dec_w1: list[list[float]],
    dec_b1: list[float],
    dec_w2: list[list[float]],
    dec_b2: list[float],
    # Adam moments
    m_enc_w1: list[list[float]],
    v_enc_w1: list[list[float]],
    m_enc_b1: list[float],
    v_enc_b1: list[float],
    m_enc_w_mean: list[list[float]],
    v_enc_w_mean: list[list[float]],
    m_enc_b_mean: list[float],
    v_enc_b_mean: list[float],
    m_enc_w_logvar: list[list[float]],
    v_enc_w_logvar: list[list[float]],
    m_enc_b_logvar: list[float],
    v_enc_b_logvar: list[float],
    m_dec_w1: list[list[float]],
    v_dec_w1: list[list[float]],
    m_dec_b1: list[float],
    v_dec_b1: list[float],
    m_dec_w2: list[list[float]],
    v_dec_w2: list[list[float]],
    m_dec_b2: list[float],
    v_dec_b2: list[float],
    lr: float,
    beta: float,
) -> None:
    """Compute gradients and update parameters using Adam optimizer.

    This function is intentionally long — it shows the full gradient flow from
    reconstruction loss and KL divergence back through the decoder, reparameterization,
    and encoder. The reparameterization trick gradient is the key insight.
    """
    # --- Gradient of reconstruction loss w.r.t. reconstructed output ---
    # d(MSE)/d(x_recon) = 2 * (x_recon - x)
    grad_recon = [2.0 * (x_recon[i] - x[i]) for i in range(len(x))]

    # --- Backprop through decoder ---
    # Decoder output layer: x_recon = dec_w2 @ dec_hidden + dec_b2
    grad_dec_b2 = grad_recon[:]
    grad_dec_w2 = [[grad_recon[i] * dec_hidden[j] for j in range(len(dec_hidden))]
                   for i in range(len(grad_recon))]
    grad_dec_hidden = [sum(dec_w2[i][j] * grad_recon[i] for i in range(len(grad_recon)))
                       for j in range(len(dec_hidden))]

    # Decoder hidden layer: dec_hidden = ReLU(dec_w1 @ z + dec_b1)
    grad_dec_hidden = [grad_dec_hidden[i] * relu_grad([dec_hidden[i]])[0]
                       for i in range(len(grad_dec_hidden))]
    grad_dec_b1 = grad_dec_hidden[:]
    grad_dec_w1 = [[grad_dec_hidden[i] * z[j] for j in range(len(z))]
                   for i in range(len(grad_dec_hidden))]
    grad_z_recon = [sum(dec_w1[i][j] * grad_dec_hidden[i] for i in range(len(grad_dec_hidden)))
                    for j in range(len(z))]

    # --- Gradient of KL divergence w.r.t. mean and log_var ---
    # KL = -0.5 * sum(1 + log_var - mean² - exp(log_var))
    # d(KL)/d(mean) = -0.5 * (-2 * mean) = mean
    # d(KL)/d(log_var) = -0.5 * (1 - exp(log_var))
    grad_mean_kl = [beta * mean[i] for i in range(len(mean))]
    grad_logvar_kl = [beta * -0.5 * (1.0 - math.exp(max(min(log_var[i], 5.0), -5.0)))
                      for i in range(len(log_var))]

    # --- Gradient through reparameterization trick ---
    # z = mean + exp(0.5 * log_var) * epsilon
    # d(loss)/d(mean) = d(loss)/d(z) * d(z)/d(mean) + d(KL)/d(mean)
    #                 = d(loss)/d(z) * 1 + d(KL)/d(mean)
    # d(loss)/d(log_var) = d(loss)/d(z) * d(z)/d(log_var) + d(KL)/d(log_var)
    #                    = d(loss)/d(z) * (0.5 * exp(0.5*log_var) * epsilon) + d(KL)/d(log_var)
    epsilon = [(z[i] - mean[i]) / (math.exp(0.5 * log_var[i]) + 1e-10) for i in range(len(z))]

    grad_mean = [grad_z_recon[i] + grad_mean_kl[i] for i in range(len(mean))]
    grad_logvar = [grad_z_recon[i] * 0.5 * math.exp(0.5 * log_var[i]) * epsilon[i] + grad_logvar_kl[i]
                   for i in range(len(log_var))]

    # --- Backprop through encoder ---
    # Encoder mean head: mean = enc_w_mean @ enc_hidden + enc_b_mean
    grad_enc_b_mean = grad_mean[:]
    grad_enc_w_mean = [[grad_mean[i] * enc_hidden[j] for j in range(len(enc_hidden))]
                       for i in range(len(grad_mean))]
    grad_enc_hidden_mean = [sum(enc_w_mean[i][j] * grad_mean[i] for i in range(len(grad_mean)))
                            for j in range(len(enc_hidden))]

    # Encoder log_var head: log_var = enc_w_logvar @ enc_hidden + enc_b_logvar
    grad_enc_b_logvar = grad_logvar[:]
    grad_enc_w_logvar = [[grad_logvar[i] * enc_hidden[j] for j in range(len(enc_hidden))]
                         for i in range(len(grad_logvar))]
    grad_enc_hidden_logvar = [sum(enc_w_logvar[i][j] * grad_logvar[i] for i in range(len(grad_logvar)))
                              for j in range(len(enc_hidden))]

    # Combine gradients from both heads
    grad_enc_hidden = [grad_enc_hidden_mean[i] + grad_enc_hidden_logvar[i]
                       for i in range(len(enc_hidden))]

    # Encoder hidden layer: enc_hidden = ReLU(enc_w1 @ x + enc_b1)
    grad_enc_hidden = [grad_enc_hidden[i] * relu_grad([enc_hidden[i]])[0]
                       for i in range(len(grad_enc_hidden))]
    grad_enc_b1 = grad_enc_hidden[:]
    grad_enc_w1 = [[grad_enc_hidden[i] * x[j] for j in range(len(x))]
                   for i in range(len(grad_enc_hidden))]

    # --- Adam update ---
    # Adam: adaptive learning rate per parameter using first and second moment estimates.
    # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    # θ_t = θ_{t-1} - α * m_t / (sqrt(v_t) + ε)
    #
    # ε prevents division by zero when v (second moment) is near zero.
    # Standard hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8 (matches PyTorch/TensorFlow).
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Helper to update a single parameter with Adam
    def adam_update(param, grad, m, v):
        for i in range(len(param)):
            if isinstance(param[i], list):  # weight matrix
                for j in range(len(param[i])):
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] ** 2
                    param[i][j] -= lr * m[i][j] / (math.sqrt(v[i][j]) + eps)
            else:  # bias vector
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                param[i] -= lr * m[i] / (math.sqrt(v[i]) + eps)

    # Update encoder parameters
    adam_update(enc_w1, grad_enc_w1, m_enc_w1, v_enc_w1)
    adam_update(enc_b1, grad_enc_b1, m_enc_b1, v_enc_b1)
    adam_update(enc_w_mean, grad_enc_w_mean, m_enc_w_mean, v_enc_w_mean)
    adam_update(enc_b_mean, grad_enc_b_mean, m_enc_b_mean, v_enc_b_mean)
    adam_update(enc_w_logvar, grad_enc_w_logvar, m_enc_w_logvar, v_enc_w_logvar)
    adam_update(enc_b_logvar, grad_enc_b_logvar, m_enc_b_logvar, v_enc_b_logvar)

    # Update decoder parameters
    adam_update(dec_w1, grad_dec_w1, m_dec_w1, v_dec_w1)
    adam_update(dec_b1, grad_dec_b1, m_dec_b1, v_dec_b1)
    adam_update(dec_w2, grad_dec_w2, m_dec_w2, v_dec_w2)
    adam_update(dec_b2, grad_dec_b2, m_dec_b2, v_dec_b2)


# === TRAINING LOOP ===

if __name__ == "__main__":
    print("Generating synthetic 2D data (mixture of 4 Gaussians)...")
    data = generate_data()
    print(f"Generated {len(data)} 2D points\n")

    # Initialize encoder weights
    enc_w1 = init_weights(HIDDEN_DIM, 2)          # 2D input → hidden
    enc_b1 = init_bias(HIDDEN_DIM)
    enc_w_mean = init_weights(LATENT_DIM, HIDDEN_DIM)  # hidden → mean
    enc_b_mean = init_bias(LATENT_DIM)
    enc_w_logvar = init_weights(LATENT_DIM, HIDDEN_DIM)  # hidden → log_var
    enc_b_logvar = init_bias(LATENT_DIM)

    # Initialize decoder weights
    dec_w1 = init_weights(HIDDEN_DIM, LATENT_DIM)  # latent → hidden
    dec_b1 = init_bias(HIDDEN_DIM)
    dec_w2 = init_weights(2, HIDDEN_DIM)          # hidden → 2D output
    dec_b2 = init_bias(2)

    # Initialize Adam moment buffers (all zeros)
    def init_moments_like(shape):
        if isinstance(shape[0], list):  # matrix
            return [[0.0 for _ in range(len(shape[0]))] for _ in range(len(shape))]
        else:  # vector
            return [0.0 for _ in range(len(shape))]

    m_enc_w1, v_enc_w1 = init_moments_like(enc_w1), init_moments_like(enc_w1)
    m_enc_b1, v_enc_b1 = init_moments_like(enc_b1), init_moments_like(enc_b1)
    m_enc_w_mean, v_enc_w_mean = init_moments_like(enc_w_mean), init_moments_like(enc_w_mean)
    m_enc_b_mean, v_enc_b_mean = init_moments_like(enc_b_mean), init_moments_like(enc_b_mean)
    m_enc_w_logvar, v_enc_w_logvar = init_moments_like(enc_w_logvar), init_moments_like(enc_w_logvar)
    m_enc_b_logvar, v_enc_b_logvar = init_moments_like(enc_b_logvar), init_moments_like(enc_b_logvar)

    m_dec_w1, v_dec_w1 = init_moments_like(dec_w1), init_moments_like(dec_w1)
    m_dec_b1, v_dec_b1 = init_moments_like(dec_b1), init_moments_like(dec_b1)
    m_dec_w2, v_dec_w2 = init_moments_like(dec_w2), init_moments_like(dec_w2)
    m_dec_b2, v_dec_b2 = init_moments_like(dec_b2), init_moments_like(dec_b2)

    print("Training VAE...")
    print(f"{'Epoch':<8} {'Total Loss':<12} {'Recon Loss':<12} {'KL Loss':<12}")
    print("-" * 48)

    for epoch in range(NUM_EPOCHS):
        # Shuffle data for stochastic gradient descent
        random.shuffle(data)

        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0

        # Process data in minibatches
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]

            batch_total_loss = 0.0
            batch_recon_loss = 0.0
            batch_kl_loss = 0.0

            for x in batch:
                # Forward pass
                enc_hidden, mean, log_var = encoder_forward(
                    x, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
                )
                z = reparameterize(mean, log_var)
                dec_hidden, x_recon = decoder_forward(z, dec_w1, dec_b1, dec_w2, dec_b2)

                # Compute loss
                total_loss, recon_loss, kl_loss = compute_loss(x, mean, log_var, x_recon, BETA)

                batch_total_loss += total_loss
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss

                # Backward pass and update
                backward_and_update(
                    x, mean, log_var, z, x_recon, enc_hidden, dec_hidden,
                    enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar,
                    dec_w1, dec_b1, dec_w2, dec_b2,
                    m_enc_w1, v_enc_w1, m_enc_b1, v_enc_b1,
                    m_enc_w_mean, v_enc_w_mean, m_enc_b_mean, v_enc_b_mean,
                    m_enc_w_logvar, v_enc_w_logvar, m_enc_b_logvar, v_enc_b_logvar,
                    m_dec_w1, v_dec_w1, m_dec_b1, v_dec_b1,
                    m_dec_w2, v_dec_w2, m_dec_b2, v_dec_b2,
                    LEARNING_RATE, BETA,
                )

            # Average loss over batch
            batch_total_loss /= len(batch)
            batch_recon_loss /= len(batch)
            batch_kl_loss /= len(batch)

            epoch_total_loss += batch_total_loss
            epoch_recon_loss += batch_recon_loss
            epoch_kl_loss += batch_kl_loss

        # Average loss over all batches
        num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
        epoch_total_loss /= num_batches
        epoch_recon_loss /= num_batches
        epoch_kl_loss /= num_batches

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"{epoch + 1:<8} {epoch_total_loss:<12.4f} {epoch_recon_loss:<12.4f} {epoch_kl_loss:<12.4f}")

    print("\nTraining complete\n")

    # === INFERENCE DEMO ===

    print("=" * 60)
    print("INFERENCE: Latent Space Interpolation")
    print("=" * 60)
    print("Encode two data points, interpolate in latent space, decode.\n")

    # Pick two points from different clusters
    point_a = data[0]      # likely from one cluster
    point_b = data[200]    # likely from a different cluster

    # Encode both points
    _, mean_a, log_var_a = encoder_forward(
        point_a, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
    )
    _, mean_b, log_var_b = encoder_forward(
        point_b, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
    )

    print(f"Point A: {[round(v, 3) for v in point_a]}")
    print(f"  → Latent mean: {[round(v, 3) for v in mean_a]}")
    print(f"Point B: {[round(v, 3) for v in point_b]}")
    print(f"  → Latent mean: {[round(v, 3) for v in mean_b]}\n")

    print("Interpolation (5 steps from A to B):")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Linearly interpolate in latent space
        z_interp = [mean_a[i] * (1 - alpha) + mean_b[i] * alpha for i in range(LATENT_DIM)]

        # Decode the interpolated latent point
        _, x_interp = decoder_forward(z_interp, dec_w1, dec_b1, dec_w2, dec_b2)

        print(f"  α={alpha:.2f}: z={[round(v, 3) for v in z_interp]} → x={[round(v, 3) for v in x_interp]}")

    print()

    print("=" * 60)
    print("INFERENCE: Prior Sampling (Generation)")
    print("=" * 60)
    print("Sample z ~ N(0,1), decode to generate new data points.\n")

    generated_points = []
    for _ in range(10):
        # Sample from the prior N(0,1)
        z_sample = [random.gauss(0, 1) for _ in range(LATENT_DIM)]

        # Decode to generate a new 2D point
        _, x_gen = decoder_forward(z_sample, dec_w1, dec_b1, dec_w2, dec_b2)

        generated_points.append(x_gen)

    print("10 generated points:")
    for i, point in enumerate(generated_points):
        print(f"  {i + 1}. {[round(v, 3) for v in point]}")

    print()

    print("=" * 60)
    print("INFERENCE: Reconstruction Quality")
    print("=" * 60)
    print("Encode training points, decode them, compare original vs reconstructed.\n")

    print("Original → Reconstructed (5 samples):")
    for i in range(5):
        x_orig = data[i * 100]  # sample every 100th point

        # Encode and decode
        _, mean, log_var = encoder_forward(
            x_orig, enc_w1, enc_b1, enc_w_mean, enc_b_mean, enc_w_logvar, enc_b_logvar
        )
        z = mean  # use mean (no sampling) for reconstruction quality check
        _, x_rec = decoder_forward(z, dec_w1, dec_b1, dec_w2, dec_b2)

        # Compute reconstruction error
        error = math.sqrt(sum((x_orig[j] - x_rec[j]) ** 2 for j in range(len(x_orig))))

        print(f"  {[round(v, 3) for v in x_orig]} → {[round(v, 3) for v in x_rec]} (error: {error:.4f})")

    print()
    print("VAE training and inference complete.")
