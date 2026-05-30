'''
- 224 * 224 pixel image chopped up into 4x4 pixel patches -> each patch becomes one token
- 56x56 number of tokens in the full pixel image (56x56 token grid)
- each token contains 4x4x3=48 raw numbers
- a learned linear projection squashes those 48 numbers down into 96 numbers --> token
- cut 7x7 windows from the token grid (7x7 windows per token)
- attention runs inside each window
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinTransformer(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels=3,num_classes=1000, head_dim=32, window_size=7, downscaling_factors=(4,2,2,2), relative_pos_embedding=True):
        super().__init__()

        # Stage 1: Projects raw 3-channel image to hidden_dim (e.g., 3 -> 96), downscales spatial map by 4
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim,
                                layers=layers[0], downscaling_factor=downscaling_factors[0],
                                num_heads=heads[0], head_dim=head_dim, window_size=window_size,
                                relative_pos_embedding=relative_pos_embedding)
        
        # Stage 2: Conflates tokens (downscales spatial map by 2), doubles features (e.g., 96 -> 192)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2,
                                layers=layers[1], downscaling_factor=downscaling_factors[1],
                                num_heads=heads[1], head_dim=head_dim, window_size=window_size,
                                relative_pos_embedding=relative_pos_embedding)
        
        # Stage 3: Conflates tokens (downscales spatial map by 2), doubles features (e.g., 192 -> 384)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
                                layers=layers[2], downscaling_factor=downscaling_factors[2],
                                num_heads=heads[2], head_dim=head_dim, window_size=window_size,
                                relative_pos_embedding=relative_pos_embedding)
        
        # Stage 4: Conflates tokens (downscales spatial map by 2), doubles features (e.g., 384 -> 768)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
                                layers=layers[3], downscaling_factor=downscaling_factors[3],
                                num_heads=heads[3], head_dim=head_dim, window_size=window_size,
                                relative_pos_embedding=relative_pos_embedding)
        
        # Final classification network block
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes))
    
    def forward(self, img):
        '''
        input shape layout: (Batch_Size, Channels, Height, Width) -> (B, 3, 224, 224)
        '''
        x = self.stage1(img)    # Dimensional scale: (B, 96, 56, 56)
        x = self.stage2(x)      # Dimensional scale: (B, 192, 28, 28)
        x = self.stage3(x)      # Dimensional scale: (B, 384, 14, 14)
        x = self.stage4(x)      # Dimensional scale: (B, 768, 7, 7)
        
        # # Global Average Pooling: squashes 2D spatial maps (7x7) into a single 1D average summary vector
        x = x.mean(dim=[2,3])   # Flattened shape: (B, 768)
        return self.mlp_head(x) # Output logit shape: (B, num_classes)


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, 
                num_heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, "Stage layers must be an even number to alternate windows!"

        # 1. Downscaling & Channel Projection Layer
        if downscaling_factor == 4:
            # Stage 1 uses a raw image input projection (e.g., 4x4 patches)
            self.patch_partition = PatchMerging(
                patch_size=downscaling_factor, 
                in_channels=in_channels, 
                out_channels=hidden_dimension
            )
        else:
            # Stages 2, 3, and 4 downscale existing features by a factor of 2
            self.patch_partition = PatchMerging(
                patch_size=downscaling_factor, 
                in_channels=in_channels, 
                out_channels=hidden_dimension
            )

        # 2. Sequential Swin Transformer Blocks Layer Stack
        self.layers = nn.ModuleList([])
        for i in range(layers):
            # Crucial Swin Trick: Alternate shifting the window on every second block!
            should_shift = (i % 2 != 0)
            
            self.layers.append(
                SwinBlock(
                    dim=hidden_dimension, 
                    num_heads=num_heads, 
                    head_dim=head_dim,
                    window_size=window_size, 
                    shift=should_shift,
                    relative_pos_embedding=relative_pos_embedding
                )
            )

    def forward(self, x):
        # x starts with shape: (B, C, H, W)
        
        # Step A: Package pixels into patches, shift dimensions to channels-last for Transformers
        x = self.patch_partition(x) # Becomes: (B, H_new, W_new, C_new)
        
        # Step B: Pass sequentially through each Swin Block in this stage
        for layer in self.layers:
            x = layer(x)
            
        # Step C: Reshape back to channels-first format for standard CNN compatibility
        # (B, H_new, W_new, C_new) -> (B, C_new, H_new, W_new)
        return x.permute(0, 3, 1, 2)