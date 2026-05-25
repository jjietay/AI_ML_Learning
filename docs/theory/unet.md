# U-Net

**U-Net** is a convolutional neural network architecture originally designed for biomedical image segmentation. Its distinctive shape, a contracting encoder followed by an expanding decoder with skip connections between them, has made it the dominant backbone for diffusion models and any task requiring precise spatial output.

## The Problem

Standard CNNs (like VGG or ResNet) are built for classification: they progressively downsample an image into a single label. But tasks like segmentation, denoising, or image generation need to produce a **full-resolution output** with spatial detail preserved. Simply downsampling and then upsampling loses fine-grained information.

## How U-Net Solves It

U-Net uses a symmetric **encoder-decoder** structure with **skip connections** that shuttle high-resolution features directly from the encoder to the decoder.

### The Three Components

| Component | Role |
|---|---|
| **Encoder (contracting path)** | Downsamples the input, capturing *what* is in the image at increasing abstraction |
| **Decoder (expanding path)** | Upsamples back to full resolution, recovering *where* things are |
| **Skip connections** | Copy feature maps from encoder to decoder at each resolution level, preserving spatial detail |

### Architecture

```
Input (256x256)
    |
  [Conv + Conv + ReLU]  ----skip---->  [Concat + Conv + Conv + ReLU]
    |                                          |
  [MaxPool] (128x128)                    [UpConv] (128x128)
    |                                          |
  [Conv + Conv + ReLU]  ----skip---->  [Concat + Conv + Conv + ReLU]
    |                                          |
  [MaxPool] (64x64)                      [UpConv] (64x64)
    |                                          |
  [Conv + Conv + ReLU]  ----skip---->  [Concat + Conv + Conv + ReLU]
    |                                          |
  [MaxPool] (32x32)                      [UpConv] (32x32)
    |                                          |
  [Conv + Conv + ReLU]  ----skip---->  [Concat + Conv + Conv + ReLU]
    |                                          |
  [MaxPool] (16x16)                      [UpConv] (16x16)
    |                                          |
    +---------> [Bottleneck] <---------+
```

The shape forms a **U**, giving the architecture its name.

### Step-by-Step

**Encoder (left side going down):**

Each level applies two 3x3 convolutions (each followed by ReLU), then a 2x2 max pooling to halve the spatial dimensions. The number of feature channels doubles at each level.

```
256x256, 64 channels
128x128, 128 channels
 64x64,  256 channels
 32x32,  512 channels
 16x16, 1024 channels  (bottleneck)
```

**Bottleneck (bottom of the U):**

The deepest layer with the most channels and smallest spatial size. This captures the most abstract, high-level representation.

**Decoder (right side going up):**

Each level applies a 2x2 up-convolution (transposed convolution) to double the spatial dimensions, **concatenates** the corresponding encoder feature map via the skip connection, then applies two 3x3 convolutions.

```
 16x16, 1024 channels  (bottleneck)
 32x32,  512 channels
 64x64,  256 channels
128x128, 128 channels
256x256,  64 channels
```

**Final layer:** A 1x1 convolution maps the 64 channels to the desired number of output classes.

## Why Skip Connections Matter

Without skip connections, the decoder must reconstruct spatial detail entirely from the compressed bottleneck representation. This leads to blurry, imprecise outputs.

| Without skip connections | With skip connections |
|---|---|
| Decoder only sees the bottleneck | Decoder sees bottleneck + full-resolution features |
| Fine edges and boundaries are lost | Precise spatial detail is preserved |
| Blurry outputs | Sharp, accurate outputs |

The skip connections give the decoder access to the **exact features** the encoder saw at each resolution, so it does not have to "guess" where edges and details were.

## U-Net in Diffusion Models

In diffusion models, U-Net serves as the **denoising network**. It receives a noisy image and a timestep embedding, and predicts the noise to remove:

| Input | Output |
|---|---|
| Noisy image $x_t$ + timestep $t$ (+ optional conditioning like text) | Predicted noise $\epsilon$ |

The architecture is modified for this role:

| Modification | Purpose |
|---|---|
| **Self-attention layers** added at lower resolutions | Capture long-range dependencies between distant pixels |
| **Timestep embedding** injected via addition or adaptive norm | Tell the network the current noise level |
| **Cross-attention layers** for text conditioning | Allow text prompts to guide generation (as in Stable Diffusion) |
| **ResNet blocks** replace plain conv blocks | Improve gradient flow and training stability |

## Strengths and Limitations

| Strengths | Limitations |
|---|---|
| Excellent spatial precision from skip connections | Scaling behaviour is not well studied |
| Proven across segmentation, denoising, generation | Attention layers are bolted on rather than native |
| Relatively parameter-efficient | Fixed CNN inductive biases limit flexibility |
| Well-understood and widely implemented | Being replaced by Transformer backbones (DiT) in newer systems |