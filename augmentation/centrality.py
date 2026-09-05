"""
Centrality-guided augmentation probabilities.

Implements Eq 3.2-3.4 (RAGCL-style node/edge/attribute augmentation,
Cui & Jia 2024) and Eq 3.6 (duplication-importance augmentation) from
the proposal, plus standard-centrality baselines (degree, PageRank)
for the RQ2 comparison your proposal commits to (Section 3.4, RQ2:
"benchmarking duplication-aware centrality against ... degree-based
dropout and PageRank-guided masking").

*** IMPORTANT: a formula note for your methodology write-up ***
As transcribed in the proposal, Eq 3.2 (node-drop probability) computes
    p_v = (log(phi_c(v)) - s_min) / (s_max - s_min) * p_n
This is INCREASING in centrality -- i.e. it would drop high-centrality
nodes MORE often. That contradicts the stated principle directly below
it ("Nodes with lower centrality are more likely to be dropped,
preserving influential nodes") and contradicts Eq 3.6, which correctly
uses (s_max - s_v) in the numerator so importance and drop probability
are inversely related. This is almost certainly a transcription error
introduced when the PDF/LaTeX was extracted (a common artifact:
operand order or a leading "1 -" gets lost). The implementation below
follows the STATED PRINCIPLE (and Eq 3.6's pattern) rather than the
literal Eq 3.2 as transcribed: importance-preserving, i.e. higher
centrality => lower drop probability. Verify this against the original
Cui & Jia (2024) equations before finalizing your thesis text, and
correct Eq 3.2 in your document accordingly.
"""
from dataclasses import dataclass
import math
import networkx as nx
import torch


def _minmax_importance_to_prob(scores: torch.Tensor, p_base: float,
                                eps: float = 1e-8) -> torch.Tensor:
    """
    Shared helper implementing the importance-preserving pattern used
    consistently across node-drop (Eq 3.2), edge-drop (Eq 3.3),
    attribute-mask (Eq 3.4) and duplication-importance (Eq 3.6):

        p_i = (s_max - s_i) / (s_max - s_min) * p_base

    i.e. higher score (more central / more important) => lower
    perturbation probability => preserved more aggressively.
    """
    s_max = scores.max()
    s_min = scores.min()
    denom = (s_max - s_min).clamp_min(eps)
    probs = (s_max - scores) / denom * p_base
    return probs.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Standard centrality (RQ2 comparison baselines)
# ---------------------------------------------------------------------------

def degree_centrality(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Standard degree-based centrality. This is the 'Standard Centrality
    Augmentation' baseline from your earlier sweep."""
    deg = torch.zeros(num_nodes)
    src, dst = edge_index
    deg.index_add_(0, src, torch.ones(src.size(0)))
    deg.index_add_(0, dst, torch.ones(dst.size(0)))
    return deg


def pagerank_centrality(edge_index: torch.Tensor, num_nodes: int,
                         alpha: float = 0.85) -> torch.Tensor:
    """PageRank centrality -- required RQ2 comparison point per the
    proposal (Section 3.4) that hasn't been implemented yet."""
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    g.add_edges_from(edges)
    try:
        pr = nx.pagerank(g, alpha=alpha)
    except nx.PowerIterationFailedConvergence:
        pr = {i: 1.0 / num_nodes for i in range(num_nodes)}
    scores = torch.tensor([pr.get(i, 1e-8) for i in range(num_nodes)],
                           dtype=torch.float)
    return scores


# ---------------------------------------------------------------------------
# Duplication-aware centrality (Eq 3.6, your proposed contribution)
# ---------------------------------------------------------------------------

@dataclass
class DuplicationCentralityConfig:
    depth_weight: float = 1.0        # weight on propagation depth
    duplication_weight: float = 1.0  # weight on duplication frequency
    p_base: float = 0.2              # base drop/mask rate (p_n, p_e, p_m)


def duplication_aware_centrality(
    duplication_freq: torch.Tensor,
    propagation_depth: torch.Tensor,
    cfg: DuplicationCentralityConfig,
) -> torch.Tensor:
    """
    Eq 3.6 style importance score, extended to fold in propagation depth
    alongside duplication frequency, matching the description you gave
    of 'incorporating duplication depth and propagation depth into
    centrality scoring'.

    duplication_freq[v]   = |D_v|, size of v's duplication set
                             (how many times v was reshared)
    propagation_depth[v]  = depth of v in the rumor propagation tree

    w_v is defined as a combination of both signals; log-compressed per
    Eq 3.6 (s_v = log(w_v)) to stabilize the min-max normalization
    against heavy-tailed duplication counts (a small number of posts
    go viral with very high reshare counts).

    NOTE: the exact combination rule below (weighted product) is a
    design choice -- the proposal names the two signals but does not
    give a closed-form combination formula. State this choice
    explicitly and justify/ablate it in your methodology chapter
    (this is exactly the RQ2 ablation already flagged as a next step:
    isolating duplication-depth-only vs propagation-depth-only vs
    combined).
    """
    w = (1.0 + cfg.duplication_weight * duplication_freq) * \
        (1.0 + cfg.depth_weight * propagation_depth)
    s = torch.log(w.clamp_min(1e-8))
    return s


def importance_scores_to_drop_probs(scores: torch.Tensor,
                                     p_base: float) -> torch.Tensor:
    """Public entry point wrapping the shared min-max transform (Eq 3.2-3.4,
    3.6 pattern) for any centrality score (degree, PageRank, or
    duplication-aware)."""
    return _minmax_importance_to_prob(scores, p_base)
