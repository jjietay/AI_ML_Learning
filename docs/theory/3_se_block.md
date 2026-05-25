# Squeeze-and-Excitation (SE) Block

An **SE Block** is a lightweight plug-in module that recalibrates a CNN's feature maps by learning *which channels are most important* for a given input. Rather than treating all channels equally, it dynamically re-weights them.

---

## Introduction

After a CNN processes an image, it produces a stack of feature maps (channel), one per filter (kernel). By default, the CNN treats **all feature maps as equally important**, regardless of the input.

The SE Block changes this: it looks at the overall response of each channel and produces a set of **importance scores** (one per channel) that are multiplied back into the feature maps. Channels deemed important are amplified and less useful ones are suppressed.

---

## Background: CNN Assumptions

SE Blocks are designed to augment CNNs, so we need to understand what CNNs assume:

| Assumption | Meaning |
|---|---|
| **Local patterns** | Meaningful features (edges, eyes) are made of nearby pixels, a small filter (e.g. 3×3) is sufficient |
| **Translation equivariance** | A vertical edge looks the same wherever it appears, the same filter weights are shared across all positions |

These assumptions drastically reduce parameters, but they also mean the CNN has **no mechanism to adapt channel importance per input**. SE Blocks allows us to add importance to channels.

---

## How It Works

An SE Block consists of three steps: **Squeeze → Excitation → Scale**.

### 1. Squeeze

Compress each 2D feature map down to a **single number** using Global Average Pooling:

```
Input:  (H × W × C)  feature maps
Output: (C,)          one summary value per channel
```

This gives a compact descriptor of what each channel "detected" across the entire spatial extent.

### 2. Excitation

Pass the `C`-dimensional vector through a **two-layer fully-connected network** with a bottleneck:

```
(C,)  →  FC + ReLU  →  (C/r,)  →  FC + Sigmoid  →  (C,) importance scores
```

- **Bottleneck** of reduction ratio `r` (typically 16) forces the network to learn compressed, generalised channel relationships
- **Sigmoid** output where each score lies in `(0, 1)` which acts as a soft gate per channel

### 3. Scale

Multiply each channel's feature map element-wise by its importance score:

```
Output = Input  ×  importance scores    (broadcast across spatial dims)
```

- Score ≈ 1 means channel passes through strongly
- Score ≈ 0 means channel is suppressed

---

## Worked Example (4 Channels)

After Squeeze, the summary vector is:

```
s = [0.9,  0.1,  0.7,  0.3]
```

**Layer 1 — FC + ReLU (C=4 --> C/r=2):**

```
W₁ = [[ 0.8, -0.5,  0.3,  0.1],
      [-0.2,  0.9, -0.4,  0.6]]

W₁ · s = [0.91, -0.19]   →  After ReLU: h = [0.91, 0.0]
```

**Layer 2 — FC + Sigmoid (C/r=2 --> C=4):**

```
W₂ = [[ 0.6,  0.3],
      [ 0.1,  0.8],
      [ 0.9, -0.2],
      [-0.1,  0.5]]

W₂ · h = [0.55, 0.09, 0.82, -0.09]   →  After Sigmoid: [0.63, 0.52, 0.69, 0.48]
```

Channel 3 (score **0.69**) is amplified most whereas Channel 4 (score **0.48**) is suppressed most.

---

## Why the Bottleneck?

Without a bottleneck, the excitation would need a `C × C` FC layer:

| Approach | Parameters (C=256, r=16) |
|---|---|
| Single FC (256 → 256) | 65,536 |
| Bottleneck (256 → 16 → 256) | **8,192** — 8× fewer |

The bottleneck forces the network to learn **generalised** channel relationships rather than memorising, and dramatically cuts parameter count. This also reduces overfitting.

---

## SE Block vs. Standard CNN

| | Standard CNN | CNN + SE Block |
|---|---|---|
| Channel weighting | Equal for all inputs | Dynamic (depends on the input) |
| Extra parameters | — | Small (two FC layers in bottleneck) |
| Representation | Static | Content-aware |
