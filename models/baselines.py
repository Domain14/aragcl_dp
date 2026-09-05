"""
Non-contrastive baselines named in your proposal's evaluation protocol
(Section 3.6: "GCN, R-GCN, BGCN, GACL, GraphCL, GRACE") plus the two
trivial baselines you already have (Majority, Text-only).

GACL/GraphCL/GRACE live in aragcl_dp.py since they share the
contrastive pipeline. GCN, R-GCN and BiGCN are plain supervised models
and are implemented here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool


class MajorityBaseline(nn.Module):
    """No learning -- predicts the majority class seen in training."""

    def __init__(self):
        super().__init__()
        self.majority_class = 0

    def fit(self, y_train: torch.Tensor):
        self.majority_class = int(y_train.float().mean().round().item())

    def predict(self, n: int):
        return torch.full((n,), self.majority_class, dtype=torch.long)


class TextOnlyBaseline(nn.Module):
    """Ignores graph structure entirely -- classifies from the root
    post's text embedding only."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, root_feature: torch.Tensor):
        return self.net(root_feature)


class VanillaGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1])
                                     for i in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h, edge_index))
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(0, keepdim=True)
        return self.classifier(g)


class RGCNBaseline(nn.Module):
    """Relational GCN -- distinguishes edge TYPES (e.g. reply vs.
    repost/duplication), unlike plain GCN. Requires an `edge_type`
    tensor aligned with edge_index (0 = reply, 1 = duplication, by
    convention with data/duplication_graph.py's two edge lists)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 num_relations: int = 2, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([
            RGCNConv(dims[i], dims[i + 1], num_relations=num_relations)
            for i in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, batch=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h, edge_index, edge_type))
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(0, keepdim=True)
        return self.classifier(g)

    @staticmethod
    def build_edge_type(reply_edge_index, duplication_edge_index):
        """Helper to build the combined edge_index + edge_type pair
        directly from a DuplicationGraph's two edge lists."""
        n_reply = reply_edge_index.size(1)
        n_dup = duplication_edge_index.size(1)
        edge_index = torch.cat([reply_edge_index, duplication_edge_index], dim=1)
        edge_type = torch.cat([
            torch.zeros(n_reply, dtype=torch.long),
            torch.ones(n_dup, dtype=torch.long),
        ])
        return edge_index, edge_type


class BiGCN(nn.Module):
    """Bi-directional GCN (Bian et al. 2020): top-down (propagation)
    and bottom-up (dispersion) branches over the same tree, each
    re-injecting the root feature at every layer, concatenated before
    classification. Named as 'BGCN' in your baseline list."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.td_convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1])
                                        for i in range(num_layers)])
        self.bu_convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1])
                                        for i in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = dropout

    def _branch(self, convs, x, edge_index, root_mask, batch):
        h = x
        root_feat = x[root_mask]
        # broadcast each graph's root feature back onto every node in
        # that graph, per Bian et al.'s root-reinforcement mechanism.
        if batch is not None:
            root_per_node = root_feat[batch]
        else:
            root_per_node = root_feat.expand(x.size(0), -1)
        for i, conv in enumerate(convs):
            h = F.relu(conv(h, edge_index))
            h = h + root_per_node[:, :h.size(1)] if root_per_node.size(1) >= h.size(1) \
                else h
            if i < len(convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch) if batch is not None else h.mean(0, keepdim=True)

    def forward(self, x, td_edge_index, bu_edge_index, root_mask, batch=None):
        g_td = self._branch(self.td_convs, x, td_edge_index, root_mask, batch)
        g_bu = self._branch(self.bu_convs, x, bu_edge_index, root_mask, batch)
        g = torch.cat([g_td, g_bu], dim=1)
        return self.classifier(g)

    @staticmethod
    def bottom_up(reply_edge_index):
        """BiGCN's bottom-up graph is simply the reply tree with edges
        reversed."""
        return reply_edge_index.flip(0)
