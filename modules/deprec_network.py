import torch
import torch.nn as nn
import torch.nn.functional as F

# Define ANN
class ANN(nn.Module):
    def __init__(self, n_inputs, n_pixels):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_pixels = n_pixels

        # Initialize layers
        self.lgn = nn.Conv2d(in_channels=n_inputs, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.v1_simple = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.v1_complex = nn.MaxPool2d(kernel_size=2, stride=2)
        self.v2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.v2_complex = nn.AdaptiveMaxPool2d((1)) 
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(8, n_pixels)

        # Activation functions
        self.activation = nn.ReLU()  

    def forward(self, x):
        # Feed-forward pass
        x = self.lgn(x)                 # LGN layer
        x = self.activation(x)

        x = self.v1_simple(x)           # V1 layer (simple cells)
        x = self.activation(x)
        x = self.v1_complex(x)          # V1 layer (complex cells)

        x = self.v2(x)                  # V2 layer
        x = self.activation(x)
        x = self.v2_complex(x)          # V2 complex layer

        x = self.flat(x)                # Flatten for fully connected layer
        x = self.fc1(x)                 # Fully connected decoding layer
        x = self.activation(x)

        return x
