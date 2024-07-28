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


class SimpleTransformer(torch.nn.Module):
    """Class with simple transformer"""
    def __init__(self, input_dim=310, num_classes=20, num_heads=5, num_encoder_layers=2, dropout=0.1):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Forward passing"""
        x = x.unsqueeze(1)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
