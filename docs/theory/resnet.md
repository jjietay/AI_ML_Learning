# ResNet (Residual Networks)

**ResNet** solved the fundamental problem that prevented neural networks from going deeper: the **degradation problem**. Its key innovation, the **residual connection** (skip connection), is now used in virtually every modern architecture including Transformers, U-Nets, and diffusion models.

## The Problem: Deeper is Not Always Better

VGG showed that depth improves performance. The natural next step was to keep adding layers. But experiments revealed something counterintuitive:

| Network | Training error |
|---|---|
| 20-layer CNN | Lower |
| 56-layer CNN | **Higher** |

The deeper network performed **worse**, not just on test data (which could be overfitting) but on **training data**. This is not a vanishing gradient problem. It is a **degradation problem**: the network struggles to learn the identity mapping needed when extra layers should simply pass information through unchanged.

## The Key Insight: Residual Learning

If a deeper network should at minimum perform as well as a shallower one, then the extra layers just need to learn the **identity function** (output = input). But learning identity through stacked conv layers is surprisingly hard.

ResNet reframes the problem. Instead of learning the full mapping $H(x)$, each block learns only the **residual** $F(x) = H(x) - x$:

$$
H(x) = F(x) + x
$$

```
        x
        |
   [Conv + BN + ReLU]
        |
   [Conv + BN]
        |
    F(x) + x  <--- skip connection adds input directly
        |
      [ReLU]
        |
      output
```

If the optimal behaviour is identity, the network just needs to push $F(x)$ toward **zero**, which is much easier than learning a full identity mapping through weight layers.

## The Residual Block

### Basic Block (ResNet-18, ResNet-34)

Two 3x3 convolutions with a skip connection:

```
Input x
   |
   |-------- skip connection --------|
   |                                 |
[3x3 Conv, BN, ReLU]                |
   |                                 |
[3x3 Conv, BN]                      |
   |                                 |
   +------------- add --------------+
   |
 [ReLU]
   |
Output
```

### Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)

For deeper networks, a three-layer bottleneck design reduces computation:

```
Input x (256 channels)
   |
   |-------------- skip connection --------------|
   |                                              |
[1x1 Conv, 64 channels]    <-- squeeze            |
   |                                              |
[3x3 Conv, 64 channels]    <-- process            |
   |                                              |
[1x1 Conv, 256 channels]   <-- expand             |
   |                                              |
   +------------------- add --------------------+
   |
Output (256 channels)
```

The 1x1 convolutions reduce and then restore the channel dimension, so the expensive 3x3 convolution operates on only 64 channels instead of 256.

| Block type | Layers per block | Parameters per block | Used in |
|---|---|---|---|
| Basic | 2 (3x3 + 3x3) | More per block | ResNet-18, 34 |
| Bottleneck | 3 (1x1 + 3x3 + 1x1) | Fewer per block | ResNet-50, 101, 152 |

## Handling Dimension Mismatches

The skip connection adds $x$ to $F(x)$, so their dimensions must match. When spatial dimensions or channel counts change between blocks, a **projection shortcut** (1x1 convolution with stride 2) is applied:

```
Input x (56x56, 64ch)
   |
   |------- [1x1 Conv, stride 2] -------|   <-- projection shortcut
   |         (28x28, 128ch)              |
   |                                     |
[3x3 Conv, stride 2, 128ch]             |
   |                                     |
[3x3 Conv, 128ch]                        |
   |                                     |
   +-------------- add ----------------+
   |
Output (28x28, 128ch)
```

## Architecture: ResNet-50

ResNet follows the same "halve spatial, double channels" pattern as VGG, but with residual blocks:

```
Input: 224x224x3

[7x7 Conv, 64, stride 2]    112x112x64
[MaxPool, stride 2]           56x56x64

Stage 1:  [Bottleneck, 64]   x 3     56x56x256
Stage 2:  [Bottleneck, 128]  x 4     28x28x512
Stage 3:  [Bottleneck, 256]  x 6     14x14x1024
Stage 4:  [Bottleneck, 512]  x 4      7x7x2048

[Global Average Pool]        1x1x2048
[FC, 1000]
[Softmax]
```

Global Average Pooling replaces VGG's massive FC layers, collapsing each 7x7 feature map into a single value. This reduces parameters dramatically.

## ResNet Variants

| Model | Stages (blocks) | Total layers | Parameters | Top-5 accuracy |
|---|---|---|---|---|
| ResNet-18 | 2+2+2+2 | 18 | 11.7M | 89.1% |
| ResNet-34 | 3+4+6+3 | 34 | 21.8M | 91.4% |
| ResNet-50 | 3+4+6+3 | 50 | 25.6M | 92.2% |
| ResNet-101 | 3+4+23+3 | 101 | 44.5M | 93.0% |
| ResNet-152 | 3+8+36+3 | 152 | 60.2M | 93.3% |

Compare with VGG-16: 138M parameters for 92.7% accuracy. ResNet-50 achieves **better accuracy with 5x fewer parameters**.

## Why Residual Connections Work

| Reason | Explanation |
|---|---|
| **Easy identity** | If a layer is unnecessary, weights can go to zero and the input passes through |
| **Gradient highway** | During backpropagation, gradients flow directly through the skip connection, avoiding vanishing gradients |
| **Ensemble effect** | A ResNet with $n$ blocks can be seen as an implicit ensemble of $2^n$ paths of different lengths |
| **Smooth loss landscape** | Skip connections make the optimisation surface smoother and easier to navigate |

### Gradient Flow Comparison

```
Without skip:  dL/dx = dL/dF * dF/dx            (can vanish through many layers)

With skip:     dL/dx = dL/dF * (dF/dx + 1)      (the +1 ensures gradient always flows)
```

The additive +1 term means the gradient **never fully vanishes**, regardless of depth.

## Impact and Legacy

The residual connection is arguably the single most influential architectural idea in deep learning. It appears in:

| Architecture | How it uses residual connections |
|---|---|
| **Transformer** | Every self-attention and MLP sub-layer has a skip connection |
| **U-Net** | Skip connections between encoder and decoder |
| **DenseNet** | Extends the idea by concatenating instead of adding |
| **BERT, GPT** | Transformer-based, so residual connections in every block |
| **Diffusion models** | ResNet blocks in U-Net, skip connections in DiT |

## VGG vs. ResNet

| | VGG-16 | ResNet-50 |
|---|---|---|
| Parameters | 138M | 25.6M |
| Depth | 16 layers | 50 layers |
| Skip connections | No | Yes |
| Can scale deeper | No (degrades) | Yes (tested to 1000+ layers) |
| FC layer parameters | 124M (90% of total) | Eliminated via Global Average Pooling |
| Top-5 accuracy | 92.7% | 92.2% (ResNet-152: 93.3%) |