from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def main():
    print("=== GNN: Karate Club with GCN (minimal demo) ===")

    dataset = KarateClub()
    data = dataset[0]
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    model = GCN(
        in_channels=data.x.size(-1),
        hidden_channels=16,
        out_channels=int(data.y.max().item()) + 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=-1)
                acc = (pred == data.y).float().mean()
            print(f"Epoch {epoch:2d}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        acc = (pred == data.y).float().mean()
    print(f"\nFinal accuracy (all nodes): {acc.item():.4f}")


if __name__ == "__main__":
    main()
