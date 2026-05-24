# Shifted Windows (Swin) Transformers

## Problem that Swin solves:

- ViT is good for generalization or looking at global features
- However, if we need to look at the closer up parts of the image, a standard ViT with 16*16 sized patches are too big
- If we shrink it such that every pixel is a patch, there will be too many patches and the computation cost will be high
- for a 1920px x 1080px image, there would be 2+ million tokens

## How swin transformers works:

- it first starts with small patches for the first transformer layer
- subsequently, merges them into bigger ones in the deeper transformer layers
- swin splits the image up into 4x4 patche of same 3 RGB Channels
- 4 x 4 x 8 feature dimensionality = 48 entries (patches are smaller compared to ViT)
- self-attention is not quadratically scaled to the sequence length
- it only occupies the nearest m neigherbours
- its called "Shifted Window based Self-attention"
- output is merged via a "Merging Layer"
- concatenates 2 by 2 neighbouring patches (not in the sequence)
- passed through a linear layer
- Attention window is shifted wrt the previous layer (similar to strided convolution)
- layer 1 can now communicate to layer 2 so on so forth
