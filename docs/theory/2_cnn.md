# Convolutional Neural Networks (CNNs)

## 1. Introduction

Images contain enormous redundancy. A photo of a cat has the same "catness" whether the cat is in the top-left or bottom-right of the frame. A fully connected (FC) network treating the image as a flat vector of pixels would need to learn "what a cat looks like" separately for every possible position. This is wasteful and doesn't generalize.

CNNs makes 2 assumptions (that is crucial to identify if its useful for a specific use case):

1. **Local patterns**
    meaningful visual features (an edge, a curve, an eye) are made of pixels that are physically close together in the image.

2. **Translation equivariance** 

A vertical edge in the top left corner of an image looks identical to a vertical edge in the bottom right corner. The pattern itself doesn't change based on where it appears.

## 2. Convolution

A **filter** (also called a **kernel**) is a small grid of learnable weights that are typically 3×3 or 5×5. To apply it to an image, you slide it across every position, and at each position you compute the elementwise product of the filter and the underlying image patch, then sum the result.

### Example

Suppose your filter is a 3×3 vertical edge detector:

```
-1   0   1
-1   0   1
-1   0   1
```

And you place it on this 3×3 image patch:

```
10  50   90
20  60  100
30  70  110
```

The output for this position is:

```
(-1×10 + 0×50 + 1×90)  =  80
(-1×20 + 0×60 + 1×100) =  80
(-1×30 + 0×70 + 1×110) =  80
───────────────────────────
                        240
```

A large positive output means "vertical edge with bright right side". This means that the pixels on the right of the patch are much brighter than pixels on the left. Slide this filter across every position of the image and you get a 2D map of vertical-edge responses across the entire image. **That output map is one channel of the layer's output.**

### Filter

When the input has multiple channels (e.g. 3 for RGB, or 64 for a deeper feature map), filters are 3D: shape `(k, k, C_in)`, where `C_in` matches the input channel count. The filter sees all input channels at once and linearly combines them.

If a conv layer has `C_out` filters, each one independently slides over the input and produces one 2D output map. Stacking those gives an output of shape `(H', W', C_out)`. The total trainable parameters in this layer:

```
params = k × k × C_in × C_out  (+ C_out biases, if used)
```

---

## 3. Channels, Tensors, and Shapes

A CNN's bread and butter is keeping track of tensor shapes. The two common axis-ordering conventions:

- **Channels-last (TensorFlow / Keras)**: `(B, H, W, C)`
- **Channels-first (PyTorch)**: `(B, C, H, W)`

Where `B` is batch size, `H` and `W` are spatial dimensions, and `C` is channels.

### Input Channels

- RGB image: `C = 3`
- Grayscale image or spectrogram: `C = 1`
- Multi-band or hyperspectral image: `C` can be much larger

### Output Channels

After a conv layer, each output channel is the response map of one filter. So if you want detectors for "edges, corners, color blobs, textures, …", you use one filter per pattern. Modern networks use anywhere from 32 to 2048 filters per layer.

Channels at deeper layers represent **increasingly abstract features**. Early layers learn edges and color blobs. Middle layers combine those into textures and parts (eyes, wheels). Deep layers combine parts into whole-object concepts.

---

## 4. Key Hyperparameters

Four hyperparameters fully control a conv layer's behavior:

| Hyperparameter | Effect |
|---|---|
| **Kernel size** (`k`) | How much spatial context each filter sees. 3×3 is by far the most common. |
| **Number of filters** (`C_out`) | How many output channels (i.e., feature detectors) the layer produces. |
| **Stride** (`s`) | How many positions the filter jumps between applications. Stride 1 = dense, stride 2 = halves spatial size. |
| **Padding** (`p`) | Zeros added around the input border so the filter can "hang off" the edge. |

### Output Shape Formula

For one spatial dimension:

```
out = floor((in + 2p − k) / s) + 1
```

Two common padding modes:

- **"Valid"** (`p = 0`) — filter cannot hang off the edge; output is smaller than input.
- **"Same"** — padding is set automatically so that `out = ceil(in / s)`. With stride 1, output spatial size equals input spatial size.

### Example Shape Walk

Starting from a 224×224 RGB image, input shape `(224, 224, 3)`:

```
Conv 3×3, 64 filters,  stride 1, same  →  (224, 224, 64)
Conv 3×3, 64 filters,  stride 2, same  →  (112, 112, 64)
Max pool 2×2,          stride 2        →  ( 56,  56, 64)
Conv 3×3, 128 filters, stride 1, same  →  ( 56,  56, 128)
```

The general pattern: each layer either preserves or downsamples spatial dimensions, while channel depth tends to grow.

---

## 5. Pooling

A **pooling layer** downsamples a feature map without learning anything. Two common types:

- **Max pooling** — Output is the maximum value in each window. Detects "is this feature present anywhere in this region?"
- **Average pooling** — Output is the mean of each window. Smoother summary.

A typical pool layer uses a 2×2 window with stride 2, halving both spatial dimensions. The channel dimension is untouched, and there are no learnable parameters.

**Global average pooling (GAP)** is a special case where the window covers the entire spatial extent, collapsing `(H, W, C)` down to `(C,)` — one number per channel. GAP is what the squeeze step of an SE block does. It's also commonly used right before the classification head in modern CNNs, replacing the parameter-heavy fully-connected layers of older designs.

---

## 6. Activation Functions

After each conv layer, an activation function adds nonlinearity. Without nonlinearity, stacking conv layers would be equivalent to a single linear transformation, and depth would buy you nothing.

- **ReLU** — `f(x) = max(0, x)`. Simple, fast, the default for most CNNs.
- **Leaky ReLU** — `f(x) = max(αx, x)` for small `α` (e.g. 0.01). Fixes "dying ReLU" where some neurons output zero forever.
- **GELU, SiLU/Swish** — Smoother variants used in newer architectures.

---

## 7. Batch Normalization

After conv (and before activation, typically), modern CNNs insert a **batch normalization (BN)** layer. BN normalizes each channel's activations to have zero mean and unit variance across the batch, then learns a per-channel scale and shift. This stabilizes training and lets you use higher learning rates.

The canonical building block in a modern CNN is the **conv → BN → ReLU** stack:

```
x  →  Conv  →  BatchNorm  →  ReLU  →  output
```

---

## 8. The Standard CNN Pattern

A typical CNN for image classification follows this overall pattern:

```
Input (H, W, 3)
   ↓
Stage 1: a few [Conv-BN-ReLU] blocks, then downsample
   ↓  (H/2, W/2, 64)
Stage 2: more blocks, more channels, downsample
   ↓  (H/4, W/4, 128)
Stage 3: more blocks, more channels, downsample
   ↓  (H/8, W/8, 256)
Stage 4: more blocks, more channels, downsample
   ↓  (H/16, W/16, 512)
   ↓
Global Average Pool  →  (512,)
   ↓
Fully Connected      →  (num_classes,)
   ↓
Softmax              →  probabilities
```

The principle is: **spatial dimensions shrink while channel depth grows**. Information is being compressed in space and expanded in feature richness. By the final stage, each of the 512 channels represents a complex high-level pattern — "dog face seen from the front," "wheel from the side" — and there's not much spatial layout left to preserve.

---

## 9. Feature Hierarchy

What each layer learns is determined entirely by training. But empirically, the same hierarchy emerges in almost every CNN trained on natural images:

| Depth | What the channels represent |
|---|---|
| Layer 1–2 | Oriented edges, color blobs, simple gradients |
| Layer 3–4 | Corners, T-junctions, simple textures (stripes, dots) |
| Layer 5–8 | Object parts — wheels, eyes, fur patches, text-like shapes |
| Layer 9+ | Whole-object detectors — specific breeds, models, scenes |

This hierarchy is why CNNs trained on large datasets like ImageNet **transfer so well** to other tasks. The early layers learn generic visual primitives that any image task can reuse.

---

## 10. Receptive Field

The **receptive field** of a neuron is the region of the original input that can influence its value. A neuron in the first conv layer with a 3×3 kernel has a receptive field of 3×3. Stack two such layers and the receptive field grows to 5×5 (each second-layer neuron sees a 3×3 window of first-layer outputs, each of which sees a 3×3 image patch). Stack more and it keeps growing. Add stride or pooling and it grows much faster.

By the final layer of a deep network, individual neurons effectively "see" the entire input image. This is why deep networks can detect whole objects: by the time information has flowed through enough layers, each neuron's receptive field is large enough to span an entire object.

---

## 11. Residual Connections

Very deep networks suffer from **vanishing gradients** — the signal traveling backward through many layers gets attenuated, and early layers stop learning. The fix introduced by ResNet is the **residual connection**: each block computes `F(x) + x` instead of just `F(x)`. The identity path lets gradients flow backward unobstructed.

```
x  →  [Conv-BN-ReLU-Conv-BN]  →  +  →  ReLU  →  out
└─────────────────────────────────┘
           (identity skip)
```

Residual blocks unlocked depths of 50, 100, even 1000 layers. Almost every modern CNN architecture uses them.

---

## 12. Squeeze-and-Excitation: Channel Attention

The **SE block** is a lightweight module that recalibrates channel-wise responses based on the input itself.

### Three Steps

1. **Squeeze** — Global average pool across spatial dims. Shape `(H, W, C) → (C,)`. Produces one summary number per channel.
2. **Excitation** — Two fully connected layers with a bottleneck (`C → C/r → C`), ending in sigmoid. Produces a vector of `C` weights, each in `(0, 1)`.
3. **Scale** — Broadcast the weight vector across all spatial positions and multiply elementwise with the original input. Shape stays `(H, W, C)`.

The block is typically inserted inside a residual branch, between the last conv and the identity addition:

```
x  →  [Conv-BN-ReLU-Conv-BN-SE]  →  +  →  ReLU  →  out
└─────────────────────────────────────┘
              (identity skip)
```

This is the simplest form of attention in vision — letting the network decide *per input* which channels matter, instead of relying on fixed weights.

---

## 13. A Concrete Architecture: ResNet-50

The canonical modern CNN. Input is `(224, 224, 3)`. The architecture:

| Stage | Operation | Output Shape |
|---|---|---|
| Stem | Conv 7×7, stride 2, 64 filters | `(112, 112, 64)` |
| Stem | Max pool 3×3, stride 2 | `(56, 56, 64)` |
| Stage 1 | 3× residual bottleneck blocks | `(56, 56, 256)` |
| Stage 2 | 4× residual bottleneck blocks, downsample first | `(28, 28, 512)` |
| Stage 3 | 6× residual bottleneck blocks, downsample first | `(14, 14, 1024)` |
| Stage 4 | 3× residual bottleneck blocks, downsample first | `(7, 7, 2048)` |
| Head | Global average pool | `(2048,)` |
| Head | Fully connected | `(num_classes,)` |

Each "bottleneck block" contains three convs (`1×1`, `3×3`, `1×1`) wrapped in a residual skip. The `1×1` convs squeeze and then expand the channel count around an expensive `3×3` conv, keeping compute manageable. Total parameters: ~25 million.

Add SE blocks inside each residual branch and you get **SE-ResNet-50**, which won the 2017 ImageNet competition.

---

## 14. Training Considerations

The essentials, briefly:

- **Loss function** — Cross-entropy loss for classification. MSE for regression-flavored tasks.
- **Optimizer** — SGD with momentum (0.9) or AdamW. Learning rate scheduling matters more than optimizer choice. Cosine decay and warmup are standard.
- **Data augmentation** — Random crops, horizontal flips, color jitter. Critical for generalization on small datasets. More aggressive augmentations (MixUp, CutMix, RandAugment) help on large ones.
- **Regularization** — Weight decay (L2 on weights), label smoothing, stochastic depth (drop residual blocks at random during training).
- **Transfer learning** — For most real-world tasks with limited data, start from a network pre-trained on ImageNet and fine-tune. Training from scratch makes sense only with millions of examples.

---

## 15. Where to Look Next

Some directions to explore once the basics feel solid:

- **Beyond classification** — Object detection (Faster R-CNN, YOLO), semantic segmentation (U-Net, DeepLab), instance segmentation (Mask R-CNN), keypoint detection.
- **Architecture refinements** — EfficientNet, RegNet, ConvNeXt — newer CNN designs that push the accuracy/efficiency frontier.
- **Vision transformers (ViT)** — A non-CNN approach using self-attention. Now competitive with or surpassing CNNs at scale, often used in hybrid form.
- **Efficient CNNs** — MobileNet, ShuffleNet, GhostNet — designed for phones and edge devices using tricks like depthwise separable convolutions.
- **Self-supervised pre-training** — MAE, DINO, SimCLR — learn good features from unlabeled images, then fine-tune on small labeled datasets.

The convolution itself isn't going anywhere. Most production vision systems still use CNNs or hybrid CNN-transformer architectures because they're efficient, well-understood, and easy to deploy.

---

## Quick Reference: Tensor Shape Cheat Sheet

For a conv layer with `C_in` input channels, `C_out` output channels, kernel `k×k`, stride `s`, and "same" padding:

```
Input shape:           (B, H,           W,           C_in)
Output shape:          (B, ceil(H/s),   ceil(W/s),   C_out)
Filter shape:          (k, k, C_in, C_out)
Trainable params:      k × k × C_in × C_out  (+ C_out biases)
FLOPs per forward:     ~ 2 × k × k × C_in × C_out × H_out × W_out × B
```

For a fully connected layer:

```
Input shape:           (B, in_features)
Output shape:          (B, out_features)
Trainable params:      in_features × out_features  (+ out_features biases)
```

For an SE block with `C` channels and reduction ratio `r`:

```
Input shape:           (B, H, W, C)
Squeeze output:        (B, C)
Excitation output:     (B, C)        — values in (0, 1)
Final output:          (B, H, W, C)  — same as input
Trainable params:      2 × C² / r  (the two FC layers)
```