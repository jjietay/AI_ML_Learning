import torch.nn as nn
import torch.nn.functional as F

'''
ResNet removes the vanishing gradient problem by:
    - introducing skip connections/residual connections
    - this creates a shortcut that lets gradients flow directly backwards without passing through every layer
'''


# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- RESNET WITHOUT SE BLOCK ---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)                           # Rescales output from convolution layer to have a mean of 0 and std of 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)                   

        # if channels differ, use 1x1 conv to match shape
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)         # Changes the channels from 32 to 64 after the 1x1 sized conv filter
        else:
            self.shortcut = nn.Identity()  # passthrough, do nothing


    def forward(self, x):
        main = F.relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(main))
        return F.relu(main + self.shortcut(x))  # add skip connection
    

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                                    nn.Conv2d(3,32,3,padding=1),    # input_channels=3, output_channels=32, size_of_filter=3x3
                                    nn.BatchNorm2d(32),             # normalize values such that mean=0, std=1, input is conv layer's output of 32 layers
                                    nn.ReLU(),                      # fix negative values to 0
                                    ResidualBlock(32, 32),          # input_channels is 32, output_channels is also 32, x just passes through shortcut path
                                    ResidualBlock(32, 64),          # input_channels is 32, output_channels is also 64, x required to go thru conv layer in shortcut path
                                    ResidualBlock(64, 64),          # input_channels is 64, output_channels is also 64, x just passes through shortcut path
                                    nn.AdaptiveAvgPool2d(1),        # 1 is the output size that we want, we get (C=64, H=1, W=1)
                                    nn.Flatten(),                   # from (batch, 64, 1, 1) to (batch, 64)
                                    nn.Linear(64, 10)               # turns 64 numbers into 10 scores
                                    )
    
    def forward(self, x):
        return self.network(x)
    


# ----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ RESNET WITH SE BLOCK ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)  # squeeze down
        self.fc2 = nn.Linear(channels // reduction, channels)  # excite back up
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # squeeze: (batch, channels, H, W) → (batch, channels)
        out = self.avgpool(x)
        out = out.squeeze(-1).squeeze(-1)   # (batch, 64, 1, 1) → (batch, 64, 1) → (batch, 64)
        # excite
        out = F.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        # scale: reshape back to (batch, channels, 1, 1) then multiply
        out = out.unsqueeze(-1).unsqueeze(-1) # (batch, 64) → (batch, 64, 1) → (batch, 64, 1, 1)
        return x * out


class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)                           # Rescales output from convolution layer to have a mean of 0 and std of 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)        
        self.se    = SEBlock(out_channels)      

        # if channels differ, use 1x1 conv to match shape
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)         # Changes the channels from 32 to 64 after the 1x1 sized conv filter
        else:
            self.shortcut = nn.Identity()  # passthrough, do nothing


    def forward(self, x):
        main = F.relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(main))
        main = self.se(main)
        return F.relu(main + self.shortcut(x))  # add skip connection


class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                                    nn.Conv2d(3,32,3,padding=1),    # input_channels=3, output_channels=32, size_of_filter=3x3
                                    nn.BatchNorm2d(32),             # normalize values such that mean=0, std=1, input is conv layer's output of 32 layers
                                    nn.ReLU(),                      # fix negative values to 0
                                    ResidualBlock(32, 32),          # input_channels is 32, output_channels is also 32, x just passes through shortcut path
                                    ResidualBlock(32, 64),          # input_channels is 32, output_channels is also 64, x required to go thru conv layer in shortcut path
                                    ResidualBlock(64, 64),          # input_channels is 64, output_channels is also 64, x just passes through shortcut path
                                    nn.AdaptiveAvgPool2d(1),        # 1 is the output size that we want, we get (C=64, H=1, W=1)
                                    nn.Flatten(),                   # from (batch, 64, 1, 1) to (batch, 64)
                                    nn.Linear(64, 10)               # turns 64 numbers into 10 scores
                                    )
    
    def forward(self, x):
        return self.network(x)







