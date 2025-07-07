import torch
import torch.nn as nn
import math

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = Sine(w0)
        self.init_weights(w0, is_first)

    def init_weights(self, w0, is_first):
        with torch.no_grad():
            if is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return self.activation(self.linear(input))

class Siren(nn.Module):
    def __init__(self, layer_sizes, w0=30.0, w0_initial=30.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            is_first = (i == 0)
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            w0_layer = w0_initial if is_first else w0
            layers.append(SirenLayer(in_size, out_size, w0=w0_layer, is_first=is_first))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))  # Final layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
