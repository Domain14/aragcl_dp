"""
Augmented view generation (Section 3.5.4 / Eq 3.2-3.4, 3.5, 3.6).

Given a graph and a per-node importance score (from
augmentation.centrality), produces one augmented view by:
  1. dropping nodes with probability p^n_v
  2. dropping edges with probability p^e_uv (derived from an edge
     weight -- here, the min of the two endpoint importance scores,
     a common surrogate when explicit edge weights aren't available)
  3. masking node attributes with probability p^m_v
  4. (if timestamps are supplied) applying temporal jitter, Eq 3.5:
         t_v' = t_v + delta,  delta ~ U(-epsilon, epsilon)

This single function is deliberately shared across every contrastive
model (RAGCL, ARAGCL-DP, and the GraphCL/GRACE/GACL baselines) --
per your proposal's own "Consistency" requirement (Section 3.6): the
same backbone and augmentation *mechanism* is used everywhere, and
only the *strategy* argument changes which centrality function feeds
it. This is what makes the RQ2 comparison a controlled experiment
rather than a confound between architecture and augmentation.
"""
from dataclasses import dataclass
from typing import Optional
import torch

from .centrality import (
    degree_centrality,
    pagerank_centrality,
    duplication_aware_centrality,
    importance_scores_to_drop_probs,
    DuplicationCentralityConfig,
)

STRATEGIES = ("random", "degree", "pagerank", "duplication_aware", "ragcl", "dataset")


@dataclass
class ViewConfig:
    strategy: str = "duplication_aware"
    p_n: float = 0.2       # base node-drop rate
    p_e: float = 0.2       # base edge-drop rate
    p_m: float = 0.2       # base attribute-mask rate
    epsilon: float = 0.0   # temporal jitter range (0 = disabled)
    dup_cfg: DuplicationCentralityConfig = None
    centrality_key: str = "Degree"  # which metric to read for strategy="dataset"

    def __post_init__(self):
        if self.dup_cfg is None:
            self.dup_cfg = DuplicationCentralityConfig(p_base=self.p_n)


def _importance_scores(x, edge_index, num_nodes, cfg: ViewConfig,
                        duplication_freq=None, propagation_depth=None,
                        precomputed_centrality=None):
    if cfg.strategy == "random":
        # GraphCL-style: uniform augmentation, no centrality guidance.
        return torch.ones(num_nodes)
    if cfg.strategy == "degree":
        # GRACE/GCA-style baseline: degree-based importance.
        return degree_centrality(edge_index, num_nodes)
    if cfg.strategy == "pagerank":
        return pagerank_centrality(edge_index, num_nodes)
    if cfg.strategy == "dataset":
        # Uses centrality shipped WITH the dataset (e.g. the
        # Degree/Pagerank/Eigenvector/Betweenness arrays in the Weibo
        # JSON format -- see data/weibo_json_loader.py) instead of
        # recomputing it, so results match whatever graph the original
        # dataset authors used to compute it.
        if precomputed_centrality is None or cfg.centrality_key not in precomputed_centrality:
            raise ValueError(
                f"strategy='dataset' requires precomputed_centrality with key "
                f"'{cfg.centrality_key}' -- pass graph.precomputed_centrality "
                f"(see data/duplication_graph.py::DuplicationGraph) through."
            )
        return precomputed_centrality[cfg.centrality_key]
    if cfg.strategy in ("duplication_aware",):
        if duplication_freq is None or propagation_depth is None:
            raise ValueError(
                "duplication_aware strategy requires duplication_freq "
                "and propagation_depth tensors from the RQ1 graph "
                "construction step (data/duplication_graph.py)."
            )
        return duplication_aware_centrality(duplication_freq,
                                             propagation_depth, cfg.dup_cfg)
    if cfg.strategy == "ragcl":
        # Eq 3.2 base case: plain centrality (degree here as a stand-in
        # for phi_c) without the duplication extension -- this is your
        # RAGCL baseline, distinct from ARAGCL-DP.
        return degree_centrality(edge_index, num_nodes)
    raise ValueError(f"Unknown augmentation strategy: {cfg.strategy}")


def generate_view(x: torch.Tensor, edge_index: torch.Tensor,
                   cfg: ViewConfig,
                   duplication_freq: Optional[torch.Tensor] = None,
                   propagation_depth: Optional[torch.Tensor] = None,
                   root_mask: Optional[torch.Tensor] = None,
                   timestamps: Optional[torch.Tensor] = None,
                   precomputed_centrality: Optional[dict] = None):
    """
    Returns (x_aug, edge_index_aug, timestamps_aug).

    root_mask: bool tensor, True for the root/source post of each
    cascade. RAGCL's first augmentation principle is "exempt root
    nodes" -- roots are never dropped or masked regardless of score.

    precomputed_centrality: pass graph.precomputed_centrality when
    cfg.strategy == "dataset" (see data/weibo_json_loader.py).
    """
    num_nodes = x.size(0)
    scores = _importance_scores(x, edge_index, num_nodes, cfg,
                                 duplication_freq, propagation_depth,
                                 precomputed_centrality)

    node_drop_p = importance_scores_to_drop_probs(scores, cfg.p_n)
    attr_mask_p = importance_scores_to_drop_probs(scores, cfg.p_m)

    if root_mask is not None:
        node_drop_p = node_drop_p.masked_fill(root_mask, 0.0)
        attr_mask_p = attr_mask_p.masked_fill(root_mask, 0.0)

    keep_node = torch.bernoulli(1.0 - node_drop_p).bool()

    # Edge drop probability from endpoint importance (min-endpoint
    # surrogate for edge weight w_uv in Eq 3.3).
    src, dst = edge_index
    edge_weight = torch.minimum(scores[src], scores[dst])
    edge_drop_p = importance_scores_to_drop_probs(edge_weight, cfg.p_e)
    keep_edge = torch.bernoulli(1.0 - edge_drop_p).bool()
    # An edge only survives if both its endpoints survive too.
    keep_edge &= keep_node[src] & keep_node[dst]

    edge_index_aug = edge_index[:, keep_edge]

    x_aug = x.clone()
    mask = torch.bernoulli(attr_mask_p).bool()
    x_aug[mask] = 0.0
    # Dropped nodes get zeroed features rather than being physically
    # removed, so node indices stay stable across the batch/encoder.
    x_aug[~keep_node] = 0.0

    timestamps_aug = None
    if timestamps is not None and cfg.epsilon > 0:
        delta = (torch.rand_like(timestamps) * 2 - 1) * cfg.epsilon  # Eq 3.5
        timestamps_aug = timestamps + delta

    return x_aug, edge_index_aug, timestamps_aug


def generate_two_views(x, edge_index, cfg: ViewConfig, **kwargs):
    """Convenience wrapper: produces (view1, view2) for the contrastive
    objective (Eq 3.8), typically with the SAME strategy but independent
    stochastic draws."""
    v1 = generate_view(x, edge_index, cfg, **kwargs)
    v2 = generate_view(x, edge_index, cfg, **kwargs)
    return v1, v2
