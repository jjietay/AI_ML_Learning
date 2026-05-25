# adaLN-Zero (Adaptive Layer Norm with Zero Initialisation)

**adaLN-Zero** is a conditioning mechanism used in Diffusion Transformers (DiT) that allows external signals like timestep and class label to modulate the behaviour of each Transformer block. It is the key architectural choice that makes DiT work.

## The Problem: How to Inject Conditioning

A diffusion model's denoising network needs to know **two things** beyond the noisy input:

| Signal | Why it matters |
|---|---|
| **Timestep t** | The noise level determines how aggressively to denoise |
| **Class label c** (or text embedding) | Guides what the model should generate |

The question is: how do you feed these signals into a Transformer? Several approaches exist, and adaLN-Zero turns out to be the best.

## Conditioning Approaches Compared

| Method | How it works | Drawback |
|---|---|---|
| **Concatenation** | Append conditioning tokens to the input sequence | Increases sequence length and compute cost |
| **Cross-attention** | Conditioning attends to input via separate attention layer | Adds parameters and architectural complexity |
| **Adaptive Layer Norm (adaLN)** | Conditioning modulates the scale and shift of layer norm | No extra tokens or attention layers needed |
| **adaLN-Zero** | adaLN + zero-initialised gating | Same efficiency as adaLN, but more stable training |

## How Standard Layer Norm Works

A quick refresher. Standard Layer Norm normalises each token's features to zero mean and unit variance, then applies learnable scale ($\gamma$) and shift ($\beta$) parameters:

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

Here $\gamma$ and $\beta$ are **fixed learned parameters**, the same for every input regardless of timestep or class.

## How adaLN Works

adaLN makes $\gamma$ and $\beta$ **input-dependent**. Instead of fixed parameters, they are predicted from the conditioning signal:

```
Timestep t + Class label c
        |
    [Embedding]
        |
      [MLP]
      /    \
  gamma    beta   (predicted per-sample)
     \    /
  LayerNorm(x) * gamma + beta
```

Now the normalisation behaviour **changes depending on the timestep and class**. At high noise levels the model can learn different scaling than at low noise levels. For different classes, it can emphasise different features.

## What adaLN-Zero Adds

adaLN-Zero extends adaLN with one critical addition: a **gating parameter** ($\alpha$) that is **initialised to zero**.

Each Transformer block has two sub-layers (self-attention and MLP). For each sub-layer, adaLN-Zero predicts **three** parameters from the conditioning:

| Parameter | Role |
|---|---|
| $\gamma$ | Scale for layer norm (before the sub-layer) |
| $\beta$ | Shift for layer norm (before the sub-layer) |
| $\alpha$ | Gate applied to the sub-layer's output (after attention or MLP) |

The full computation for one sub-layer:

```
Conditioning (t + c)
        |
      [MLP]
        |
  gamma, beta, alpha
        |
        v
x_norm = LayerNorm(x) * gamma + beta     [modulated normalisation]
         |
   [Self-Attention or MLP]
         |
   output = x + alpha * sublayer(x_norm)  [gated residual connection]
```

### Why Initialise to Zero?

When $\alpha = 0$ at the start of training:

$$
\text{output} = x + 0 \cdot \text{sublayer}(x_{\text{norm}}) = x
$$

Each Transformer block acts as an **identity function**, simply passing its input through unchanged. This means:

| Property | Benefit |
|---|---|
| The full DiT initially acts as identity | Stable training from the start, no exploding or collapsing outputs |
| Each block gradually learns to contribute | The network "grows" its complexity during training |
| Matches initialisation strategies from ResNets | Zero-init residual connections are a proven technique for deep networks |

Without zero initialisation, randomly initialised blocks can produce large, unstable outputs early in training, making diffusion model training harder to converge.

## adaLN-Zero in Context

A single DiT block with adaLN-Zero:

```
Input x
  |
  |--- adaLN(x, conditioning) --- [Self-Attention] --- * alpha_1 ---+
  |                                                                  |
  + <---------------------------------------------------------------+
  |
  |--- adaLN(x, conditioning) --- [MLP] --- * alpha_2 ---+
  |                                                       |
  + <----------------------------------------------------+
  |
Output
```

Six parameters are predicted per block from the conditioning: $\gamma_1, \beta_1, \alpha_1$ for the attention sub-layer and $\gamma_2, \beta_2, \alpha_2$ for the MLP sub-layer.

## Why adaLN-Zero Won

The DiT paper tested all four conditioning methods on the same model. Results on ImageNet 256x256:

| Method | FID (lower = better) |
|---|---|
| Cross-attention | 23.4 |
| Concatenation (in-context) | 42.2 |
| adaLN | 12.0 |
| **adaLN-Zero** | **2.27** |

adaLN-Zero significantly outperformed all alternatives while adding **zero extra parameters** to the attention mechanism itself. The only added cost is a small MLP per block to predict the six modulation parameters.