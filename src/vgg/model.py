import torch.nn as nn
import torch.nn.functional as F

'''
VGGNet16 --> 16 layers: --> input is image of 224 x 224 x 3 (H,W,C)
    - very deep CNN for Large-scale image recognition
    - 2 conv layers (64 filters, size 3x3, stride and padding set to 1)
    - maxpool 2x2 pooling, s=2 (112 by 112 by 64)
    - 2 conv layers again (128 filters, 3x3, padding && stride == 1) ---> 112 x 112 x 128
    - another maxpool (2x2 and s=2) ---> 56x56x128
    - 3 conv layers (256 filters of size 3x3 with stride && padding == 1) ---> 56 x 56 x 256
    - maxpool (2x2, s=2)
    - 3 conv layers (512 filters of size 3x3, stride && padding == 1) ---> 28 x 28 x 512
    - maxpool (2x2, s=2) ---> 14 x 14 x 512
    - 3 conv layers (512 filters of size 3x3, stride && padding == 1) ---> 14 x 14 x 512
    - maxpool (2x2, s=2) ---> 7 x 7 x 512
    - flatten into 1D vector (25088)
    - ReLU dropout 4096
    - ReLU dropout 4096
    - softmax (final number of classes as output)
'''

class VGGBlock128And256(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        return x



class VGGBlock512(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool1 = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        return x
    


class VGG(nn.Module):
    def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                                        nn.Conv2d(3, 64, 3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        VGGBlock128And256(64, 256),
                                        VGGBlock512(256, 512),
                                        VGGBlock512(512, 512),
                                        nn.Flatten(),           # Flatten into a 1D vector (length of 25088)
                                        nn.Linear(25088, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 10)
                                        )
    def forward(self, x):
        return self.network(x)