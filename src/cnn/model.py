import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                                        nn.Conv2d(3, 32, 3, padding=1),    # in_channels=3, out_channels=32, size_of_sliding_filter=3x3
                                        nn.ReLU(),              # activation function to fix numbers that are -ve to 0, +ve as it is
                                        nn.MaxPool2d(2),        # W/2 & H/2 ---> 16 x 16
                                        nn.Dropout(0.25),
                                        nn.Conv2d(32, 64, 3, padding=1),   # in_channels=32, out_channels=64, size_of_sliding_filter=3x3
                                        nn.ReLU(),              # activation function ---> Rectified Linear Unit
                                        nn.MaxPool2d(2),        # W/2 & H/2 ---> 8 x 8
                                        nn.Dropout(0.25),
                                        nn.Flatten(),           # From (64, 8, 8) ---> into a 1D tensor = 64 * 8 * 8 = 4096
                                        nn.Linear(4096, 10)     # output 10 values (one score per class)
                                        )
    def forward(self, x):
        return self.network(x)