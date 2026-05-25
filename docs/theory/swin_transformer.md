# Swin Transformer

The **Shifted Window (Swin) Transformer** is a hierarchical vision backbone that addresses the computational bottleneck of applying self-attention to high-resolution images, while preserving the ability to capture both fine-grained local features and broad global context.

## The Problem with Standard ViT

| Issue | Detail |
|---|---|
| Large patches miss detail | Standard ViT uses 16×16 patches which is too coarse for dense tasks like detection or segmentation |
| Pixel-level patches are too expensive | A 1920×1080 image would have 2+ million tokens; attention cost scales quadratically |

---

## How Swin Solves It

Swin takes a **hierarchical** approach which is to start small and then merge gradually and restricts self-attention to **local windows** instead of the full image.

### Step 1: Small Patches

Images are split into **4×4 patches** (much finer than ViT's 16×16). Each patch has 3 RGB channels, giving an initial feature dimension of `4 × 4 × 3 = 48`.

### Step 2: Local Window Self-Attention

Instead of attending globally, each token only attends to its **nearest m × m neighbours** (typically m = 7). This makes attention cost linear in image size rather than quadratic.

```
Image → [Window partition] → Self-attention within each window → Output
```

### Step 3: Shifted Windows

To allow information to flow *between* windows across layers, Swin **shifts the window grid** between consecutive layers similar to strided convolution:

```
Layer 1:  windows at positions (0, 0), (0, 7), (7, 0), ...
Layer 2:  windows shifted by (3, 3) → different tokens can now communicate
```

This means **layer 1 can communicate with layer 2**, and so on across the depth of the network.

### Step 4: Patch Merging

After each stage, a **Merging Layer** concatenates 2×2 neighbouring patches and projects them through a linear layer, halving the spatial resolution while doubling the feature dimension, exactly like strided convolution in a CNN:

```
Stage 1: 56×56 patches, dim=96
Stage 2: 28×28 patches, dim=192
Stage 3: 14×14 patches, dim=384
Stage 4:  7×7  patches, dim=768
```

---

## Swin vs. ViT

| | ViT | Swin |
|---|---|---|
| Patch size | 16×16 (fixed) | 4×4, merges hierarchically |
| Attention scope | Global (all patches) | Local windows (shifted each layer) |
| Computation | Quadratic in image size | Linear in image size |
| Feature hierarchy | None | Yes, like a CNN's multi-scale features |
| Best for | Image classification | Detection, segmentation, classification |
