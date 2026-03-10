"""
Residual networks from first principles: skip connections as gradient highways — proving
that F(x) + x preserves gradient flow where F(x) alone would vanish.
"""
# Reference: He et al., "Deep Residual Learning for Image Recognition" (2015)
# https://arxiv.org/abs/1512.03385

# === TRADEOFFS ===
# + Skip connections prevent vanishing gradients in deep networks
# + Residual learning: easier to learn F(x)=0 than H(x)=x directly
# + Enables training of much deeper networks (100+ layers)
# - Skip connections add complexity to gradient computation graph
# - Identity shortcuts require matching dimensions (or projection)
# - Diminishing returns: ultra-deep resnets don't always beat moderately deep ones
# WHEN TO USE: Any deep network architecture where vanishing gradients
#   are a concern — image classification, feature extraction, backbone networks.
# WHEN NOT TO: Shallow networks (< 5 layers) where gradients flow fine,
#   or sequence tasks where attention-based architectures dominate.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Image and dataset
IMAGE_SIZE = 4            # 4x4 images — tiny to keep pure-Python convolutions tractable
NUM_CHANNELS_IN = 1       # grayscale input
NUM_CLASSES = 4           # four synthetic pattern types
TRAIN_SAMPLES = 80        # per class (320 total)
TEST_SAMPLES = 20         # per class (80 total)
NOISE_PROB = 0.08         # pixel flip probability for dataset variety

# Convolution architecture
KERNEL_SIZE = 3           # 3x3 kernels throughout
NUM_CHANNELS = 4          # feature channels inside residual blocks
PADDING = 1               # same-padding: output spatial dims = input spatial dims

# Training
LEARNING_RATE = 0.01
NUM_STEPS = 200           # training iterations (each over a mini-batch)
BATCH_SIZE = 8

# Signpost: Production ResNets use 64-2048 channels, 224x224 images, and 50-152 layers.
# This toy version (4x4 images, 4 channels, 2 residual blocks) exists to demonstrate
# the gradient-preserving mechanics of skip connections, not to achieve SOTA accuracy.
# Tensor libraries would do the convolutions ~10,000x faster via BLAS/cuDNN.


# === SYNTHETIC DATASET ===

# Four pattern classes on 4x4 grids. Patterns are designed so a 3x3 conv can distinguish them:
# class 0 — horizontal stripe (row of 1s)
# class 1 — vertical stripe (column of 1s)
# class 2 — checkerboard (alternating 0/1)
# class 3 — center blob (2x2 block in the middle)

def make_horizontal(noise: float = NOISE_PROB) -> list[list[float]]:
    """4x4 image with one horizontal stripe."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    row = random.randint(0, IMAGE_SIZE - 1)
    for c in range(IMAGE_SIZE):
        img[row][c] = 1.0
    _apply_noise(img, noise)
    return img


def make_vertical(noise: float = NOISE_PROB) -> list[list[float]]:
    """4x4 image with one vertical stripe."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    col = random.randint(0, IMAGE_SIZE - 1)
    for r in range(IMAGE_SIZE):
        img[r][col] = 1.0
    _apply_noise(img, noise)
    return img


def make_checkerboard(noise: float = NOISE_PROB) -> list[list[float]]:
    """4x4 checkerboard pattern with random phase."""
    phase = random.randint(0, 1)
    img = [
        [float((r + c + phase) % 2) for c in range(IMAGE_SIZE)]
        for r in range(IMAGE_SIZE)
    ]
    _apply_noise(img, noise)
    return img


def make_center_blob(noise: float = NOISE_PROB) -> list[list[float]]:
    """4x4 image with a 2x2 block in the center."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    # Place 2x2 block near center (offset by 0 or 1 for variety)
    sr = 1 + random.randint(0, 1)
    sc = 1 + random.randint(0, 1)
    for r in range(sr, min(sr + 2, IMAGE_SIZE)):
        for c in range(sc, min(sc + 2, IMAGE_SIZE)):
            img[r][c] = 1.0
    _apply_noise(img, noise)
    return img


def _apply_noise(img: list[list[float]], prob: float) -> None:
    """Flip pixels with given probability to add variety."""
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if random.random() < prob:
                img[r][c] = 1.0 - img[r][c]


GENERATORS = [make_horizontal, make_vertical, make_checkerboard, make_center_blob]
CLASS_NAMES = ["horizontal", "vertical", "checker", "blob"]


def generate_dataset(
    samples_per_class: int,
) -> tuple[list[list[list[float]]], list[int]]:
    """Generate a balanced dataset of synthetic 4x4 images and labels."""
    images: list[list[list[float]]] = []
    labels: list[int] = []
    for class_id, gen_fn in enumerate(GENERATORS):
        for _ in range(samples_per_class):
            images.append(gen_fn())
            labels.append(class_id)
    combined = list(zip(images, labels))
    random.shuffle(combined)
    imgs, lbls = zip(*combined)
    return list(imgs), list(lbls)


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput), then backward() replays the computation
    graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        # This is the CRITICAL operation for residual connections: when we compute
        # F(x) + x, addition distributes gradients equally to both branches.
        # The gradient of the skip branch is always 1, preventing vanishing.
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        # d(x^n)/dx = n * x^(n-1)
        return Value(
            self.data ** exponent, (self,),
            (exponent * self.data ** (exponent - 1),)
        )

    def __neg__(self) -> Value:
        return self * -1.0

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) * (self ** -1.0)

    def relu(self) -> Value:
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def exp(self) -> Value:
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self) -> None:
        """Compute gradients via reverse-mode autodiff.

        Builds topological ordering, then propagates gradients backward using
        the chain rule: dL/dx = sum(dL/dy * dy/dx) for each output y of x.
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# === CONVOLUTION OPERATIONS ===

# Same-padded convolution keeps spatial dimensions constant, which is essential for
# residual connections: F(x) + x requires F(x) and x to have identical shapes.
# We pad with zeros so a 4x4 input with a 3x3 kernel produces a 4x4 output.

def conv2d_same(
    image: list[list[Value]],
    kernel: list[list[Value]],
    bias: Value,
) -> list[list[Value]]:
    """Apply a 3x3 convolution with same-padding (output size = input size).

    Math: out[i,j] = bias + sum_{m,n} kernel[m,n] * padded_image[i+m, j+n]
    where padding = 1 on each side ensures output spatial dims match input.

    Args:
        image: [H, W] grid of Value objects
        kernel: [3, 3] grid of learned Value weights
        bias: scalar Value bias added to each output position

    Returns:
        [H, W] grid of Value objects
    """
    h = len(image)
    w = len(image[0])
    k = len(kernel)
    pad = k // 2  # padding = 1 for 3x3 kernel

    output: list[list[Value]] = []
    for i in range(h):
        row: list[Value] = []
        for j in range(w):
            val = bias
            for m in range(k):
                for n in range(k):
                    # Map output position (i,j) + kernel offset (m,n) to input position
                    ii = i + m - pad
                    jj = j + n - pad
                    if 0 <= ii < h and 0 <= jj < w:
                        val = val + kernel[m][n] * image[ii][jj]
                    # Out-of-bounds positions contribute zero (zero-padding)
            row.append(val)
        output.append(row)
    return output


def conv1x1(
    feature_maps: list[list[list[Value]]],
    weights: list[list[Value]],
    biases: list[Value],
) -> list[list[list[Value]]]:
    """1x1 convolution: linear combination across channels at each spatial position.

    Math: out[c_out][i][j] = bias[c_out] + sum_{c_in} W[c_out][c_in] * in[c_in][i][j]

    1x1 convolutions serve as "projection shortcuts" in ResNets: when the number of
    channels changes between the input and output of a residual block, we can't do
    F(x) + x directly because the dimensions don't match. A 1x1 conv projects x
    into the right channel count so the addition is valid.
    """
    c_out = len(weights)
    h = len(feature_maps[0])
    w = len(feature_maps[0][0])
    c_in = len(feature_maps)

    output: list[list[list[Value]]] = []
    for co in range(c_out):
        channel: list[list[Value]] = []
        for i in range(h):
            row: list[Value] = []
            for j in range(w):
                val = biases[co]
                for ci in range(c_in):
                    val = val + weights[co][ci] * feature_maps[ci][i][j]
                row.append(val)
            channel.append(row)
        output.append(channel)
    return output


# === BATCH NORMALIZATION ===

# BatchNorm stabilizes training by normalizing activations to zero mean and unit variance
# within each mini-batch. This prevents the "internal covariate shift" problem where
# the distribution of layer inputs changes as earlier layers update, forcing later layers
# to constantly readapt.
#
# Math: BN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
# where mu, sigma^2 are computed over the spatial dimensions of the current batch,
# and gamma, beta are learned affine parameters.
#
# Signpost: We use a simplified per-channel normalization over spatial positions within
# a single sample (instance norm style) rather than true batch statistics, because
# accumulating batch stats through scalar autograd would be prohibitively slow.
# The educational point — normalizing activations stabilizes gradient flow — still holds.

def instance_norm(
    feature_map: list[list[Value]],
    gamma: Value,
    beta: Value,
    eps: float = 1e-5,
) -> list[list[Value]]:
    """Normalize a single channel's feature map to zero mean and unit variance.

    Math: y = gamma * (x - mean) / sqrt(var + eps) + beta
    This is instance normalization (per-sample, per-channel), a simplification of
    batch normalization that avoids accumulating statistics across the batch.
    """
    h = len(feature_map)
    w = len(feature_map[0])
    n = h * w

    # Compute mean (forward-only, not through autograd — treating stats as constants)
    # Signpost: True batch norm computes gradients through the mean/variance computation.
    # We detach the stats to keep the graph small. This sacrifices some gradient accuracy
    # but makes training tractable with scalar autograd.
    mu = sum(v.data for row in feature_map for v in row) / n
    var = sum((v.data - mu) ** 2 for row in feature_map for v in row) / n
    inv_std = 1.0 / math.sqrt(var + eps)

    output: list[list[Value]] = []
    for i in range(h):
        row: list[Value] = []
        for j in range(w):
            # Normalize, then apply learned scale (gamma) and shift (beta)
            normalized = (feature_map[i][j] - mu) * inv_std
            row.append(gamma * normalized + beta)
        output.append(row)
    return output


# === RESIDUAL BLOCK ===

# The core insight of ResNets: instead of learning H(x) directly, learn the residual
# F(x) = H(x) - x, so the output is F(x) + x. This is easier to optimize because:
#
# 1. If the optimal mapping is close to identity, the network only needs to push
#    F(x) → 0 (all weights → 0), which is much easier than learning H(x) = x
#    from scratch with random weights.
#
# 2. The gradient of F(x) + x with respect to x is:
#    ∂(F(x) + x)/∂x = F'(x) + 1
#    The "+1" term means gradients ALWAYS have magnitude ≥ 1 flowing through the
#    skip connection, even if F'(x) ≈ 0. This is the "gradient highway" — gradients
#    can flow directly from loss to early layers without being multiplied by small
#    numbers at each layer (which causes vanishing gradients in plain networks).
#
# 3. In a deep network with L residual blocks, the gradient from block L to block 1
#    includes a direct path: 1 * 1 * ... * 1 = 1. Without skip connections, the
#    gradient must pass through L matrix multiplications, each potentially < 1,
#    giving a product that shrinks exponentially: 0.5^L → 0 for large L.

def init_residual_block(
    channels_in: int,
    channels_out: int,
    block_name: str,
) -> dict:
    """Initialize parameters for one residual block.

    Architecture: Conv3x3 → InstanceNorm → ReLU → Conv3x3 → InstanceNorm → (+skip) → ReLU

    When channels_in != channels_out, a 1x1 projection shortcut matches dimensions.
    This is the "projection shortcut" from He et al. (option B). The alternative
    (option A) zero-pads the extra channels, but projection is more common in practice.
    """
    params: dict = {}
    # Kaiming init: std = sqrt(2 / fan_in) for ReLU networks
    std = math.sqrt(2.0 / (channels_in * KERNEL_SIZE * KERNEL_SIZE))

    # First conv: channels_in 3x3 kernels → channels_out feature maps
    for co in range(channels_out):
        for ci in range(channels_in):
            params[f'{block_name}_conv1_k{co}_{ci}'] = [
                [Value(random.gauss(0, std)) for _ in range(KERNEL_SIZE)]
                for _ in range(KERNEL_SIZE)
            ]
        params[f'{block_name}_conv1_b{co}'] = Value(0.0)

    # First instance norm: per-channel gamma and beta
    for c in range(channels_out):
        params[f'{block_name}_bn1_gamma{c}'] = Value(1.0)
        params[f'{block_name}_bn1_beta{c}'] = Value(0.0)

    # Second conv: channels_out 3x3 kernels → channels_out feature maps
    std2 = math.sqrt(2.0 / (channels_out * KERNEL_SIZE * KERNEL_SIZE))
    for co in range(channels_out):
        for ci in range(channels_out):
            params[f'{block_name}_conv2_k{co}_{ci}'] = [
                [Value(random.gauss(0, std2)) for _ in range(KERNEL_SIZE)]
                for _ in range(KERNEL_SIZE)
            ]
        params[f'{block_name}_conv2_b{co}'] = Value(0.0)

    # Second instance norm
    for c in range(channels_out):
        params[f'{block_name}_bn2_gamma{c}'] = Value(1.0)
        params[f'{block_name}_bn2_beta{c}'] = Value(0.0)

    # Projection shortcut (1x1 conv) when channel dimensions differ
    if channels_in != channels_out:
        proj_std = math.sqrt(2.0 / channels_in)
        params[f'{block_name}_proj_w'] = [
            [Value(random.gauss(0, proj_std)) for _ in range(channels_in)]
            for _ in range(channels_out)
        ]
        params[f'{block_name}_proj_b'] = [Value(0.0) for _ in range(channels_out)]

    return params


def multi_channel_conv(
    feature_maps: list[list[list[Value]]],
    params: dict,
    block_name: str,
    conv_name: str,
    channels_out: int,
) -> list[list[list[Value]]]:
    """Apply multi-channel convolution: each output channel sums over all input channels.

    Math: out[co][i][j] = bias[co] + sum_{ci} conv2d(in[ci], kernel[co][ci])[i][j]
    This generalizes single-channel conv: each output feature map is a learned linear
    combination of all input feature maps, convolved with independent 3x3 kernels.
    """
    channels_in = len(feature_maps)
    h = len(feature_maps[0])
    w = len(feature_maps[0][0])

    output: list[list[list[Value]]] = []
    for co in range(channels_out):
        # Start with bias, then accumulate contributions from each input channel
        channel = [
            [params[f'{block_name}_{conv_name}_b{co}'] for _ in range(w)]
            for _ in range(h)
        ]
        for ci in range(channels_in):
            kernel = params[f'{block_name}_{conv_name}_k{co}_{ci}']
            conv_out = conv2d_same(feature_maps[ci], kernel, Value(0.0))
            for i in range(h):
                for j in range(w):
                    channel[i][j] = channel[i][j] + conv_out[i][j]
        output.append(channel)
    return output


def forward_residual_block(
    x: list[list[list[Value]]],
    params: dict,
    block_name: str,
    channels_in: int,
    channels_out: int,
    use_skip: bool = True,
) -> list[list[list[Value]]]:
    """Forward pass through one residual block.

    Architecture: x → Conv → BN → ReLU → Conv → BN → (+shortcut) → ReLU
    The shortcut is either identity (same channels) or 1x1 projection (different channels).

    Args:
        x: input feature maps [channels_in, H, W]
        use_skip: if False, skip connection is disabled (for gradient comparison)
    """
    h = len(x[0])
    w = len(x[0][0])

    # --- Branch F(x): the "residual" path ---
    # First conv + norm + activation
    out = multi_channel_conv(x, params, block_name, 'conv1', channels_out)
    for c in range(channels_out):
        out[c] = instance_norm(
            out[c],
            params[f'{block_name}_bn1_gamma{c}'],
            params[f'{block_name}_bn1_beta{c}'],
        )
        out[c] = [[v.relu() for v in row] for row in out[c]]

    # Second conv + norm (no ReLU yet — applied after the addition)
    out = multi_channel_conv(out, params, block_name, 'conv2', channels_out)
    for c in range(channels_out):
        out[c] = instance_norm(
            out[c],
            params[f'{block_name}_bn2_gamma{c}'],
            params[f'{block_name}_bn2_beta{c}'],
        )

    # --- Skip connection: the "identity" or "projection" path ---
    # F(x) + x: this single addition is what makes ResNets work.
    # Without it (use_skip=False), this is just a plain deep network.
    if use_skip:
        if channels_in != channels_out:
            # Projection shortcut: 1x1 conv to match channel dimensions
            # He et al. call this "option B" — slightly more parameters but
            # consistently better than zero-padding (option A).
            shortcut = conv1x1(
                x,
                params[f'{block_name}_proj_w'],
                params[f'{block_name}_proj_b'],
            )
        else:
            shortcut = x

        # The residual addition: out = F(x) + x
        # Gradient: ∂out/∂x = ∂F(x)/∂x + I  (identity matrix)
        # The I term guarantees gradient magnitude ≥ 1 through this block.
        for c in range(channels_out):
            for i in range(h):
                for j in range(w):
                    out[c][i][j] = out[c][i][j] + shortcut[c][i][j]

    # Final ReLU after the addition (He et al. "post-activation" variant)
    for c in range(channels_out):
        out[c] = [[v.relu() for v in row] for row in out[c]]

    return out


# === FULL NETWORK ===

# Architecture: Input(4x4x1) → ResBlock1(1→4ch) → ResBlock2(4→4ch) → GlobalAvgPool → Linear → Softmax
#
# This is a minimal ResNet with 2 residual blocks. The first block uses a projection shortcut
# (1→4 channels), the second uses identity shortcuts (4→4 channels).
#
# Signpost: ResNet-18 uses [2,2,2,2] blocks with [64,128,256,512] channels. ResNet-50
# uses bottleneck blocks with 1x1→3x3→1x1 conv sequences. Our 2-block, 4-channel
# version is ~1000x smaller but demonstrates the same gradient flow mechanics.

def init_all_parameters() -> dict:
    """Initialize parameters for the full ResNet."""
    params: dict = {}

    # Block 1: 1 input channel → NUM_CHANNELS output channels (needs projection)
    params.update(init_residual_block(1, NUM_CHANNELS, 'block1'))

    # Block 2: NUM_CHANNELS → NUM_CHANNELS (identity shortcut)
    params.update(init_residual_block(NUM_CHANNELS, NUM_CHANNELS, 'block2'))

    # Classification head: global average pooling reduces spatial dims to 1x1,
    # so the linear layer input size = NUM_CHANNELS
    linear_std = math.sqrt(2.0 / NUM_CHANNELS)
    params['cls_w'] = [
        [Value(random.gauss(0, linear_std)) for _ in range(NUM_CHANNELS)]
        for _ in range(NUM_CLASSES)
    ]
    params['cls_b'] = [Value(0.0) for _ in range(NUM_CLASSES)]

    return params


def global_avg_pool(feature_maps: list[list[list[Value]]]) -> list[Value]:
    """Reduce each channel to a single scalar by averaging over spatial dimensions.

    Math: out[c] = (1/HW) * sum_{i,j} feature_maps[c][i][j]

    Global average pooling (Lin et al., 2013) replaces the flattening + large FC layer
    used in earlier CNNs (VGG, AlexNet). Benefits: (1) no learned parameters to overfit,
    (2) enforces correspondence between feature maps and categories, (3) more robust
    to spatial translations. ResNets use this exclusively.
    """
    pooled: list[Value] = []
    for channel in feature_maps:
        h = len(channel)
        w = len(channel[0])
        n = h * w
        total = channel[0][0]
        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    continue
                total = total + channel[i][j]
        pooled.append(total * (1.0 / n))
    return pooled


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow.

    Math: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = exp_vals[0]
    for e in exp_vals[1:]:
        total = total + e
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clamped log for numerical stability in cross-entropy loss."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def forward(
    image_data: list[list[float]],
    params: dict,
    use_skip: bool = True,
) -> list[Value]:
    """Full ResNet forward pass: image → 2 residual blocks → pool → classify.

    Args:
        image_data: raw 4x4 image as floats
        params: model parameters
        use_skip: whether skip connections are active (False for comparison)

    Returns:
        logits: list of NUM_CLASSES Values
    """
    # Wrap raw pixels as Value objects — this is the entry point to the autograd graph
    image = [[Value(pixel) for pixel in row] for row in image_data]

    # Reshape to [1, H, W] (single input channel)
    x: list[list[list[Value]]] = [image]

    # Residual block 1: 1 → NUM_CHANNELS channels (with projection shortcut)
    x = forward_residual_block(x, params, 'block1', 1, NUM_CHANNELS, use_skip)

    # Residual block 2: NUM_CHANNELS → NUM_CHANNELS (identity shortcut)
    x = forward_residual_block(x, params, 'block2', NUM_CHANNELS, NUM_CHANNELS, use_skip)

    # Global average pooling: [NUM_CHANNELS, 4, 4] → [NUM_CHANNELS]
    pooled = global_avg_pool(x)

    # Linear classifier: logit_c = sum_j W[c,j] * pooled[j] + bias[c]
    logits: list[Value] = []
    for c in range(NUM_CLASSES):
        val = params['cls_b'][c]
        for j in range(NUM_CHANNELS):
            val = val + params['cls_w'][c][j] * pooled[j]
        logits.append(val)

    return logits


def compute_loss(logits: list[Value], target: int) -> Value:
    """Cross-entropy loss: L = -log(softmax(logits)[target])."""
    probs = softmax(logits)
    return -safe_log(probs[target])


# === OPTIMIZER ===

def collect_params(params: dict) -> list[Value]:
    """Flatten all parameter Values into a single list for optimization."""
    all_params: list[Value] = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, Value):
            all_params.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, Value):
                    all_params.append(item)
                elif isinstance(item, list):
                    for v in item:
                        if isinstance(v, Value):
                            all_params.append(v)
                        elif isinstance(v, list):
                            for vv in v:
                                if isinstance(vv, Value):
                                    all_params.append(vv)
    return all_params


def sgd_step(param_list: list[Value], learning_rate: float) -> None:
    """Vanilla SGD update: w = w - lr * grad.

    SGD is simpler than Adam and sufficient for this demonstration. The focus is on
    gradient flow mechanics (skip vs no-skip), not optimizer sophistication.
    """
    for p in param_list:
        p.data -= learning_rate * p.grad


def zero_grad(param_list: list[Value]) -> None:
    """Reset all gradients to zero before the next backward pass.

    Gradients accumulate by default (+=), so we must zero them between iterations.
    Forgetting this is a classic bug — gradients grow without bound.
    """
    for p in param_list:
        p.grad = 0.0


# === GRADIENT ANALYSIS ===

# The key experiment: train the same architecture WITH and WITHOUT skip connections,
# then compare gradient norms per layer. Without skip connections, gradients in early
# layers should be orders of magnitude smaller (vanishing gradient problem).

def compute_gradient_norms(params: dict, prefix: str) -> float:
    """Compute L2 norm of gradients for all parameters matching a prefix.

    ||grad||_2 = sqrt(sum(grad_i^2))
    A small norm means gradients are vanishing — the layer is learning very slowly.
    """
    total = 0.0
    count = 0
    for key in params:
        if key.startswith(prefix) and 'conv' in key:
            val = params[key]
            if isinstance(val, list):
                for row in val:
                    if isinstance(row, list):
                        for v in row:
                            if isinstance(v, Value):
                                total += v.grad ** 2
                                count += 1
                    elif isinstance(row, Value):
                        total += row.grad ** 2
                        count += 1
    return math.sqrt(total) if total > 0 else 0.0


# === TRAINING LOOP ===

def train_model(
    train_images: list[list[list[float]]],
    train_labels: list[int],
    use_skip: bool,
    label: str,
) -> tuple[dict, list[float], list[float], list[float]]:
    """Train the ResNet and record loss + gradient norms per layer.

    Returns:
        params: trained model parameters
        losses: loss history
        block1_norms: gradient norm history for block 1 (early layer)
        block2_norms: gradient norm history for block 2 (later layer)
    """
    # Re-seed so both runs (skip/no-skip) start from identical weights
    random.seed(42)
    params = init_all_parameters()
    param_list = collect_params(params)

    losses: list[float] = []
    block1_norms: list[float] = []
    block2_norms: list[float] = []
    n = len(train_images)

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    for step in range(NUM_STEPS):
        # Sample a mini-batch
        indices = [random.randint(0, n - 1) for _ in range(BATCH_SIZE)]

        zero_grad(param_list)

        batch_loss = 0.0
        for idx in indices:
            logits = forward(train_images[idx], params, use_skip=use_skip)
            loss = compute_loss(logits, train_labels[idx])
            loss.backward()
            batch_loss += loss.data

        # Average gradients over the batch
        for p in param_list:
            p.grad /= BATCH_SIZE

        avg_loss = batch_loss / BATCH_SIZE
        losses.append(avg_loss)

        # Record gradient norms to compare skip vs no-skip
        b1_norm = compute_gradient_norms(params, 'block1')
        b2_norm = compute_gradient_norms(params, 'block2')
        block1_norms.append(b1_norm)
        block2_norms.append(b2_norm)

        sgd_step(param_list, LEARNING_RATE)

        if step % 50 == 0 or step == NUM_STEPS - 1:
            print(
                f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                f"Block1 grad: {b1_norm:.6f} | Block2 grad: {b2_norm:.6f}"
            )

    return params, losses, block1_norms, block2_norms


def evaluate(
    images: list[list[list[float]]],
    labels: list[int],
    params: dict,
    use_skip: bool,
) -> float:
    """Compute classification accuracy on a dataset."""
    correct = 0
    for img, label in zip(images, labels):
        logits = forward(img, params, use_skip=use_skip)
        pred = max(range(NUM_CLASSES), key=lambda c: logits[c].data)
        if pred == label:
            correct += 1
    return correct / len(images)


# === MAIN ===

if __name__ == "__main__":
    start_time = time.time()

    # Generate datasets
    print("Generating synthetic 4x4 image dataset...")
    train_images, train_labels = generate_dataset(TRAIN_SAMPLES)
    test_images, test_labels = generate_dataset(TEST_SAMPLES)
    print(f"  Training: {len(train_images)} images ({TRAIN_SAMPLES} per class)")
    print(f"  Test:     {len(test_images)} images ({TEST_SAMPLES} per class)")

    # --- Experiment 1: Train WITH skip connections ---
    params_skip, losses_skip, b1_skip, b2_skip = train_model(
        train_images, train_labels, use_skip=True, label="ResNet (with skip connections)"
    )

    # --- Experiment 2: Train WITHOUT skip connections ---
    # Same architecture, same initialization, same data — only difference is F(x)+x vs F(x)
    params_noskip, losses_noskip, b1_noskip, b2_noskip = train_model(
        train_images, train_labels, use_skip=False, label="Plain network (no skip connections)"
    )

    # === EVALUATION ===
    print(f"\n{'='*60}")
    print("Classification Results")
    print(f"{'='*60}")

    # Evaluate both models on test set
    acc_skip = evaluate(test_images, test_labels, params_skip, use_skip=True)
    acc_noskip = evaluate(test_images, test_labels, params_noskip, use_skip=False)

    print(f"  ResNet (skip)    — Test accuracy: {acc_skip:.1%}")
    print(f"  Plain (no skip)  — Test accuracy: {acc_noskip:.1%}")

    # === GRADIENT COMPARISON ===
    # This is the key educational output: skip connections preserve gradient magnitude
    # in early layers, while plain networks suffer from vanishing gradients.

    print(f"\n{'='*60}")
    print("Gradient Flow Comparison (avg norm over last 50 steps)")
    print(f"{'='*60}")
    print(f"{'Layer':<20} {'With Skip':>15} {'Without Skip':>15} {'Ratio':>10}")
    print(f"{'-'*60}")

    # Average gradient norms over the last 50 training steps for stable comparison
    last_n = min(50, NUM_STEPS)

    avg_b1_skip = sum(b1_skip[-last_n:]) / last_n
    avg_b2_skip = sum(b2_skip[-last_n:]) / last_n
    avg_b1_noskip = sum(b1_noskip[-last_n:]) / last_n
    avg_b2_noskip = sum(b2_noskip[-last_n:]) / last_n

    ratio_b1 = avg_b1_skip / avg_b1_noskip if avg_b1_noskip > 1e-12 else float('inf')
    ratio_b2 = avg_b2_skip / avg_b2_noskip if avg_b2_noskip > 1e-12 else float('inf')

    print(f"{'Block 1 (early)':<20} {avg_b1_skip:>15.6f} {avg_b1_noskip:>15.6f} {ratio_b1:>9.1f}x")
    print(f"{'Block 2 (later)':<20} {avg_b2_skip:>15.6f} {avg_b2_noskip:>15.6f} {ratio_b2:>9.1f}x")

    print(f"\n  Interpretation:")
    print(f"  - Block 1 is the EARLY layer (closest to input, farthest from loss).")
    print(f"  - Without skip connections, Block 1 gradients are smaller because they")
    print(f"    must pass through more multiplicative layers to reach the loss.")
    print(f"  - Skip connections provide a 'gradient highway' that bypasses these")
    print(f"    multiplications: d(F(x)+x)/dx = F'(x) + 1, so the +1 term ensures")
    print(f"    gradients never vanish completely through the skip path.")

    # The degradation problem: deeper plain networks can perform WORSE than shallow ones.
    # This is counterintuitive — a deeper network should be at least as good as a shallow
    # one because it could learn the identity for extra layers. But optimization difficulty
    # (vanishing gradients) prevents this in practice. ResNets solve this by making the
    # identity mapping the default (F(x)=0 is easy to learn), so extra layers don't hurt.
    if acc_skip > acc_noskip:
        print(f"\n  The ResNet outperforms the plain network ({acc_skip:.1%} vs {acc_noskip:.1%}),")
        print(f"  demonstrating that skip connections improve optimization even for")
        print(f"  this shallow 2-block network.")
    else:
        print(f"\n  Both models achieve similar accuracy ({acc_skip:.1%} vs {acc_noskip:.1%}).")
        print(f"  With only 2 blocks, vanishing gradients are not catastrophic — the")
        print(f"  gradient norm difference is more revealing than the accuracy gap.")
        print(f"  The benefit grows dramatically with network depth (50+ layers).")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s")
