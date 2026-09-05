"""
ARAGCL-DP and its ablations/baselines, as ONE configurable pipeline.

Per your proposal's own "Consistency" protocol (Section 3.6: "The same
datasets, splits and backbone encoders are used across experiments to
isolate the impact of duplication-aware components"), the model below
is used for every entry in the required baseline list:

    view_strategy        use_duplication (encoder)   =>  model name
    ----------------------------------------------------------------
    "random"              False                       =>  GraphCL
    "degree"               False                       =>  GRACE/GCA
    "random" (adv. eps)    False                       =>  GACL*
    "ragcl" (centrality)   False                       =>  RAGCL (Cui & Jia 2024)
    "duplication_aware"    True                        =>  ARAGCL-DP (yours)

* GACL's defining feature is an ADVERSARIAL perturbation rather than a
  random one. A full adversarial implementation (e.g. FGSM/PGD on node
  features between the two forward passes) is a genuinely separate
  training loop, not a config flag -- see train/trainer.py's
  `adversarial_step` for a minimal FGSM version wired in as a
  best-effort stand-in. If your thesis needs a rigorous GACL
  comparison, budget time to verify this against the actual GACL paper
  rather than relying on this approximation.

Plain GCN/R-GCN/BiGCN (non-contrastive, supervised-only baselines) live
separately in models/baselines.py since they don't share this
two-view/contrastive structure at all.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn

from .encoders import DuplicationAwareEncoder
from augmentation.views import ViewConfig, generate_two_views
from losses.contrastive import nt_xent_loss, joint_loss


@dataclass
class ARAGCL_DP_Config:
    in_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    lam: float = 0.5                 # Eq 3.7 duplication influence weight
    use_duplication: bool = True     # False => Ablation 2 (see data/duplication_graph.py::build_static_graph for Ablation 1)
    backbone: str = "gcn"            # "gcn" or "gat", per the RQ1 plan's backbone requirement
    heads: int = 2                   # GAT attention heads (ignored for backbone="gcn")
    view_strategy: str = "duplication_aware"
    tau: float = 0.5                 # contrastive temperature
    unsup_weight: float = 1.0        # Eq 3.9 lambda / alpha
    num_classes: int = 2
    p_n: float = 0.2
    p_e: float = 0.2
    p_m: float = 0.2
    epsilon: float = 0.0             # temporal jitter (Eq 3.5), RQ2 sweep knob
    centrality_key: str = "Degree"   # which dataset-provided metric, for view_strategy="dataset"
    dropout: float = 0.3             # see models/encoders.py -- mitigates the overfitting seen on real data


class ARAGCL_DP(nn.Module):
    def __init__(self, cfg: ARAGCL_DP_Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = DuplicationAwareEncoder(
            cfg.in_dim, cfg.hidden_dim, cfg.num_layers,
            lam=cfg.lam, use_duplication=cfg.use_duplication,
            backbone=cfg.backbone, heads=cfg.heads, dropout=cfg.dropout,
        )
        self.projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def _view_cfg(self) -> ViewConfig:
        return ViewConfig(strategy=self.cfg.view_strategy, p_n=self.cfg.p_n,
                           p_e=self.cfg.p_e, p_m=self.cfg.p_m,
                           epsilon=self.cfg.epsilon,
                           centrality_key=self.cfg.centrality_key)

    def contrastive_forward(self, graph, batch=None):
        """graph: a DuplicationGraph (or batched equivalent) from
        data/duplication_graph.py. Returns (h1_graph, h2_graph) graph-level
        embeddings from the two augmented views, for Eq 3.8."""
        vcfg = self._view_cfg()
        (x1, ei1, _), (x2, ei2, _) = generate_two_views(
            graph.x, graph.reply_edge_index, vcfg,
            duplication_freq=graph.duplication_freq,
            propagation_depth=graph.propagation_depth,
            root_mask=graph.root_mask,
            timestamps=graph.timestamps,
            precomputed_centrality=graph.precomputed_centrality,
        )
        dup_ei = graph.duplication_edge_index if self.cfg.use_duplication else \
            torch.zeros((2, 0), dtype=torch.long)

        _, g1 = self.encoder(x1, ei1, dup_ei, batch=batch)
        _, g2 = self.encoder(x2, ei2, dup_ei, batch=batch)
        z1, z2 = self.projection(g1), self.projection(g2)
        return z1, z2

    def classify(self, graph, batch=None):
        """Clean (non-augmented) forward pass for supervised
        fine-tuning / inference."""
        _, g = self.encoder(graph.x, graph.reply_edge_index,
                             graph.duplication_edge_index, batch=batch)
        logits = self.classifier(g)
        return logits

    def pretrain_step(self, graph, batch=None):
        z1, z2 = self.contrastive_forward(graph, batch=batch)
        return nt_xent_loss(z1, z2, tau=self.cfg.tau)

    def finetune_step(self, graph, y, batch=None):
        logits = self.classify(graph, batch=batch)
        sup_loss = nn.functional.cross_entropy(logits, y)
        return sup_loss, logits


def make_ragcl(in_dim: int, **overrides) -> ARAGCL_DP:
    """RAGCL baseline (Cui & Jia 2024): centrality-guided augmentation,
    no duplication mechanism."""
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=False,
                            view_strategy="ragcl", **overrides)
    return ARAGCL_DP(cfg)


def make_graphcl(in_dim: int, **overrides) -> ARAGCL_DP:
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=False,
                            view_strategy="random", **overrides)
    return ARAGCL_DP(cfg)


def make_grace(in_dim: int, **overrides) -> ARAGCL_DP:
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=False,
                            view_strategy="degree", **overrides)
    return ARAGCL_DP(cfg)


def make_dataset_centrality(in_dim: int, centrality_key: str = "Degree",
                             **overrides) -> ARAGCL_DP:
    """RQ2 baseline using centrality shipped WITH the dataset (e.g.
    the Weibo JSON format's Degree/Pagerank/Eigenvector/Betweenness
    arrays -- data/weibo_json_loader.py) rather than recomputing it.
    Only usable with graphs that actually have
    graph.precomputed_centrality populated."""
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=False,
                            view_strategy="dataset",
                            centrality_key=centrality_key, **overrides)
    return ARAGCL_DP(cfg)


def make_aragcl_dp(in_dim: int, **overrides) -> ARAGCL_DP:
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=True,
                            view_strategy="duplication_aware", **overrides)
    return ARAGCL_DP(cfg)


# ---------------------------------------------------------------------------
# RQ1 plan, Step 5: Ablation & Sensitivity Testing
# ---------------------------------------------------------------------------
# Ablation 1 ("Static representation"): remove node duplication at the
#   GRAPH-CONSTRUCTION level. This model is architecturally identical to
#   ARAGCL-DP -- the difference is entirely in which DuplicationGraph
#   you feed it. Build the input with
#   data/duplication_graph.py::build_static_graph(posts) instead of
#   build_duplication_graph(posts), and use make_aragcl_dp() as normal.
#   (No separate factory function needed -- the ablation lives in the
#   data pipeline, not the model config.)
#
# Ablation 2 (remove temporal/duplication EDGES only, keep duplication
#   metadata for centrality scoring): same graph construction as the
#   full model (build_duplication_graph), but the ENCODER ignores
#   duplication_edge_index. This IS a model-config-level ablation:

def make_ablation2(in_dim: int, **overrides) -> ARAGCL_DP:
    """Ablation 2: duplication_freq/propagation_depth still feed RQ2's
    centrality-guided augmentation (view_strategy stays
    'duplication_aware'), but Eq 3.7's encoder no longer aggregates
    over duplication_edge_index -- isolates whether the gain comes from
    the augmentation signal, the encoder's structural term, or both."""
    cfg = ARAGCL_DP_Config(in_dim=in_dim, use_duplication=False,
                            view_strategy="duplication_aware", **overrides)
    return ARAGCL_DP(cfg)
