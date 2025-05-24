import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter_mean


class MLPAggregator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, neighbor_agg):
        concat_features = torch.cat([x, neighbor_agg], dim=1)
        return self.mlp(concat_features)


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.agg = MLPAggregator(in_channels, out_channels, hidden_dim)

    def forward(self, x, edge_index):
        device = x.device
        row, col = edge_index.to(device)
        neighbor_mean = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        out = self.agg(x, neighbor_mean)

        return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout: float = 0.0):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(GraphSAGEConv(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
