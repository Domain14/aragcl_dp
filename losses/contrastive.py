"""
Graph-level contrastive loss (Eq 3.8):

    L_contrastive = -log [ exp(sim(h1,h2)/tau) /
                            sum_{i=1}^{2N} 1[i != j] exp(sim(hi,hj)/tau) ]

Standard NT-Xent / InfoNCE over a batch of N graphs, each with two
augmented-view embeddings (h1, h2). Positives are the two views of the
same graph; all other 2N-2 embeddings in the batch are negatives.
"""
import torch
import torch.nn.functional as F


def nt_xent_loss(h1: torch.Tensor, h2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    h1, h2: [N, d] graph-level embeddings for view 1 / view 2 of N
    graphs in the batch (rows correspond -- h1[i] and h2[i] are the
    two views of the same graph).
    """
    n = h1.size(0)
    z = torch.cat([h1, h2], dim=0)               # [2N, d]
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau                          # [2N, 2N]

    mask_self = torch.eye(2 * n, dtype=torch.bool, device=h1.device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    # positive pair index for row i: i+n mod 2n
    pos_idx = torch.arange(2 * n, device=h1.device)
    pos_idx = (pos_idx + n) % (2 * n)

    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2 * n), pos_idx].mean()
    return loss


def joint_loss(sup_loss: torch.Tensor, unsup_loss: torch.Tensor,
               weight: float = 1.0) -> torch.Tensor:
    """Eq 3.1 / 3.9:  L = L_sup + lambda * L_unsup"""
    return sup_loss + weight * unsup_loss
