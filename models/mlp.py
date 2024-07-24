"""File with defined mlp class"""
import torch

class MLP(torch.nn.Module):
    """Class with simple MLP"""
    def __init__(self, hidden_size, num_layers, input_shape = 310, output_shape = 20):
        super().__init__()
        self.input = torch.nn.Linear(input_shape, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = torch.nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        """Forward passing"""
        x = torch.nn.functional.leaky_relu(self.input(x))
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return self.output(x)
