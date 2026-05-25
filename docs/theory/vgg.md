# VGGNet

**VGGNet** demonstrated that network **depth** is a critical factor for performance in image recognition. Its key insight was radically simple: replace large convolutional filters with stacks of small 3x3 filters, allowing much deeper networks without increasing computational cost per layer.

## The Key Idea: Small Filters, Deeper Networks

Before VGG, architectures like AlexNet used large filters (11x11, 5x5) to capture spatial patterns. VGG asked: what if we use **only 3x3 filters** and just stack more layers?

| Approach | Filter size | Depth | Receptive field |
|---|---|---|---|
| AlexNet-style | 11x11, 5x5 | Shallow (5 conv layers) | Large per layer |
| VGG-style | 3x3 only | Deep (16-19 conv layers) | Built up gradually |

### Why 3x3 Works

Two stacked 3x3 conv layers have the **same receptive field** as one 5x5 layer. Three stacked 3x3 layers match a 7x7 layer. But the stacked version is better:

```
One 7x7 filter:     7 x 7 x C = 49C parameters
Three 3x3 filters:  3 x (3 x 3 x C) = 27C parameters
```

| Benefit | Why |
|---|---|
| **Fewer parameters** | 27C vs 49C for the same receptive field |
| **More non-linearity** | Each layer has its own ReLU, so three decision boundaries instead of one |
| **Better feature learning** | Forces the network to build features compositionally |

## Architecture: VGG-16

VGG-16 has 16 weight layers (13 convolutional + 3 fully connected). The architecture follows a strict, repeating pattern:

```
Input: 224x224x3 (RGB image)

Block 1:  [3x3 conv, 64]  x 2  +  MaxPool
Block 2:  [3x3 conv, 128] x 2  +  MaxPool
Block 3:  [3x3 conv, 256] x 3  +  MaxPool
Block 4:  [3x3 conv, 512] x 3  +  MaxPool
Block 5:  [3x3 conv, 512] x 3  +  MaxPool

Flatten

FC: 4096
FC: 4096
FC: 1000 (ImageNet classes)

Softmax
```

### Spatial Dimensions Through the Network

| Block | Output size | Channels |
|---|---|---|
| Input | 224 x 224 | 3 |
| Block 1 + Pool | 112 x 112 | 64 |
| Block 2 + Pool | 56 x 56 | 128 |
| Block 3 + Pool | 28 x 28 | 256 |
| Block 4 + Pool | 14 x 14 | 512 |
| Block 5 + Pool | 7 x 7 | 512 |
| Flatten | 25088 | - |
| FC layers | 4096 | - |
| Output | 1000 | - |

The pattern is consistent: **spatial dimensions halve, channels double** at each block. This became a standard design principle for later architectures.

## VGG Variants

| Model | Conv layers | FC layers | Total weight layers | Parameters |
|---|---|---|---|---|
| VGG-11 | 8 | 3 | 11 | 133M |
| VGG-13 | 10 | 3 | 13 | 133M |
| VGG-16 | 13 | 3 | 16 | 138M |
| VGG-19 | 16 | 3 | 19 | 144M |

The vast majority of parameters (about 124M of 138M in VGG-16) live in the **fully connected layers**, not the convolutional layers. This later motivated architectures like GoogLeNet and ResNet to eliminate or shrink FC layers.

## Why VGG Matters

VGG established principles that influenced nearly every architecture that followed:

| Principle | Impact |
|---|---|
| **Depth improves performance** | Directly motivated ResNet's push to 100+ layers |
| **Small filters are sufficient** | 3x3 became the standard conv filter size |
| **Uniform architecture** | Simple repeating blocks became a design pattern |
| **Halve spatial, double channels** | Adopted by ResNet, Swin, and many others |
| **Pretrained feature extractor** | VGG features are still used today for perceptual loss, style transfer, and transfer learning |

## Limitations

| Limitation | Detail |
|---|---|
| **Massive parameter count** | 138M parameters, most wasted in FC layers |
| **No skip connections** | Cannot train much deeper without vanishing gradients |
| **Slow inference** | Heavy computation compared to later efficient architectures |
| **No batch normalisation** | Predates BatchNorm, making training harder and slower |

## VGG vs. Later Architectures

| | VGG-16 | ResNet-50 | EfficientNet-B0 |
|---|---|---|---|
| Parameters | 138M | 25M | 5.3M |
| Layers | 16 | 50 | Compound scaled |
| Skip connections | No | Yes | Yes |
| Top-5 accuracy (ImageNet) | 92.7% | 93.3% | 93.3% |
| Key innovation | Depth with 3x3 filters | Residual connections | Architecture search |

ResNet achieves better accuracy with **5x fewer parameters** and **3x more layers**, precisely because skip connections solved the training difficulties VGG could not overcome.