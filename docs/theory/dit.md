# Diffusion Transformers (DiT)

A **Diffusion Transformer (DiT)** replaces the traditional U-Net backbone in diffusion models with a standard Transformer, showing that the Transformer architecture alone is enough to achieve state-of-the-art image generation.

## Background: How Diffusion Models Work

Diffusion models learn to generate images through a two-phase process:

| Phase | What happens |
|---|---|
| **Forward (noising)** | Gradually add Gaussian noise to a real image over T timesteps until it becomes pure noise |
| **Reverse (denoising)** | A neural network learns to reverse each step, gradually recovering a clean image from noise |

The neural network receives a noisy image and a **timestep t** (telling it how much noise is present) and predicts the noise to remove. Traditionally, this network is a **U-Net**. DiT asks: what if we use a **Transformer** instead?

## Why Replace the U-Net?

| U-Net limitation | DiT advantage |
|---|---|
| CNN-based with fixed inductive biases | Transformer is architecture-agnostic and scales predictably |
| Scaling behaviour is not well studied | Clear scaling laws: more parameters and compute = better FID |
| Hybrid attention bolted onto convolutions | Attention is the entire architecture, not an add-on |

## How DiT Works

### Step 1: Operate in Latent Space

DiT does **not** work on raw pixels. A pretrained VAE encoder first compresses the image into a smaller latent representation (following the Latent Diffusion / Stable Diffusion approach):

```
Image (256x256x3)
    |
  [VAE Encoder]
    |
Latent (32x32x4)
```

### Step 2: Patchify the Latent

The latent is divided into non-overlapping patches (like ViT), then linearly embedded into tokens with added positional embeddings:

```
Latent (32x32x4)
    |
  [Split into p x p patches]
    |
Sequence of tokens (e.g. 256 tokens for p=2)
```

Smaller patch size = more tokens = more compute but finer detail.

### Step 3: Transformer Blocks with Conditioning

Each DiT block is a standard Transformer block (self-attention + MLP), but it needs to know **two things** beyond the image content:

| Conditioning signal | Purpose |
|---|---|
| **Timestep, t** | Tells the model the current noise level |
| **Class label, c** | Tells the model what class to generate (e.g. "dog", "car") |

These are injected via **adaLN-Zero** (adaptive Layer Norm), which is the key architectural innovation. Instead of using cross-attention or concatenation, the timestep and class embeddings directly modulate the scale and shift parameters of each layer norm:

```
t + c
  |
[MLP]
  |
scale, shift, gate parameters
  |
LayerNorm(x) * scale + shift
```

The "Zero" in adaLN-Zero means the gate parameters are **initialised to zero**, so each DiT block initially acts as an identity function. This stabilises early training.

### Step 4: Predict the Noise

The final layer decodes the token sequence back to the original latent shape and predicts the noise to be removed at this timestep.

## The Key Finding: Scaling Laws

The most important result from the DiT paper is that **Transformer scaling laws apply directly to image generation**:

| Model | Parameters | Patch size | FID (lower = better) |
|---|---|---|---|
| DiT-S/8 | 33M | 8 | 68.4 |
| DiT-B/4 | 130M | 4 | 43.5 |
| DiT-L/4 | 458M | 4 | 23.3 |
| DiT-XL/2 | 675M | 2 | 2.27 |

Larger model + smaller patches = consistently better generation quality. This predictable scaling is what made DiT influential and led to its adoption in systems like Stable Diffusion 3 and SORA.

## DiT vs. U-Net Diffusion

| | U-Net | DiT |
|---|---|---|
| Core architecture | CNN + skip connections + attention layers | Pure Transformer |
| Conditioning method | Cross-attention, concatenation | adaLN-Zero |
| Scaling behaviour | Unclear, architecture-dependent | Predictable: more compute = better results |
| Inductive bias | Strong spatial bias from convolutions | Minimal, learned from data |
| Adopted by | Stable Diffusion 1.x/2.x, DALL-E 2 | Stable Diffusion 3, SORA |