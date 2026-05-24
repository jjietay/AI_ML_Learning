# Squeeze and Excitation Block (SE)

## CNNs: Two Key Structural Assumptions

Before understanding SE Blocks, it helps to know what CNNs assume about images:

| Assumption | Meaning |
|---|---|
| **Local patterns** | Meaningful features (edges, eyes) are made of nearby pixels — a small filter (e.g. 3×3) is sufficient |
| **Translation equivariance** | A vertical edge looks the same wherever it appears — the same filter weights are shared across all positions |

These two assumptions drastically reduce parameters compared to a fully-connected network applied directly to pixels.


## Introduction

- after a CNN processes an image, it produces a stack of feature maps, one map per filter (kernel)
- CNN treats all feature maps as **equally important**
- this is where the **Squeeze and Excitation Block** comes in
- it is an attachment used to bias certain convolutional channels (filters/kernels)
- meaning it decides which channels we should trust more than the other

## How it works

- a typical SE block contains of 3 steps: Squeeze, Excitation, Scale

### Squeeze

- each feature map is a 2D spatial grid (e.g. 7x7 pixels)
- Squeeze performs **Global Average Pooling** which takes each channel and compresses it down to a single number by averaging all the pixels of that particular channel

### Excitation

- assuming we have a total of X number of channels, we have X numbers of max pools
- the vector of X numbers is passed through a **2-layer fully connected network** with a bottleneck (such as shrinking it to smaller than X, and then expand back to X)
- this is then followed by a sigmoid function
- output is a vector of X values, each between 0 and 1, representing them as *importance scores per channel*
- the bottleneck forces the network to learn compressed generalised relationships between channels rather than memorising

### Scale

- each channel's feature map is multiplied by its importance score
- channels with a score near 1 passes through strongly while those channels with scores near 0 gets suppressed


### Worked Example (4 Channels)

After Squeeze, say you have:

```text
s = [0.9,  0.1,  0.7,  0.3]
```

**Layer 1 (W₁, shape [2×4]):**

```text
W₁ = [[ 0.8, -0.5,  0.3,  0.1],
      [-0.2,  0.9, -0.4,  0.6]]

W₁ · s = [0.91, -0.19]

After ReLU: h = [0.91, 0.0]
```

**Layer 2 (W₂, shape [4×2]):**

```text
W₂ = [[ 0.6,  0.3],
      [ 0.1,  0.8],
      [ 0.9, -0.2],
      [-0.1,  0.5]]

W₂ · h = [0.55, 0.09, 0.82, -0.09]

After Sigmoid: scores = [0.63, 0.52, 0.69, 0.48]
```

Channel 3 (score 0.69) is amplified most; Channel 4 (score 0.48) is suppressed most.

---

### Full SE Block Architecture

```text
Input Feature Maps (H × W × C)
        ↓
   [SQUEEZE] Global Average Pool → (1 × 1 × C) vector
        ↓
   [EXCITATION] FC → ReLU → FC → Sigmoid → C importance scores
        ↓
   [SCALE] Multiply scores back into original feature maps
        ↓
Output Feature Maps (H × W × C)   ← same shape, re-weighted
```

The output shape is identical to the input — SE Blocks are a **plug-in module** that can be inserted into any existing CNN architecture (e.g. ResNet) with minimal overhead.

---

### Why the Bottleneck in Excitation?

A single `[C × C]` FC layer would have C² parameters (e.g. 256² = 65,536). The bottleneck (shrink to r, expand back) gives only $2 \times C \times r$ parameters:

| Approach | Parameters (C=256, r=16) |
|---|---|
| Single FC layer (256→256) | 65,536 |
| Bottleneck (256→16→256) | 8,192 |

**8× fewer parameters** — reducing overfitting and forcing the network to learn compressed, generalised channel relationships rather than memorising noise.

---

### SE Blocks vs Standard CNN

| | Standard CNN Layer | CNN + SE Block |
|---|---|---|
| Channel weighting | Equal for all | Dynamic, input-dependent |
| Parameters added | — | Small (bottleneck FC) |
| Representation | Static | Content-aware |
| Analogy | Treating all instruments at equal volume | A mixing engineer adjusting the balance per song |