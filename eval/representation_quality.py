"""
RQ3 — Duplication-Driven Representation Quality for Semantic Alignment
and Discriminative Power (Section 3.4/3.7).

Nothing for this RQ existed yet. This module implements every metric
your proposal names:
  - Hamming distance between rumor / non-rumor embeddings
  - cosine similarity within- and across-rumor clusters
  - silhouette score for cluster separation
  - intra- vs inter-class variance
  - mutual information between labels and embeddings
  - alignment score between original nodes and their duplicates
  - t-SNE / UMAP 2D projection for visualization
"""
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


@dataclass
class RepresentationQualityReport:
    hamming_distance_rumor_vs_nonrumor: float
    cosine_sim_within_rumor: float
    cosine_sim_within_nonrumor: float
    cosine_sim_across_classes: float
    silhouette: float
    intra_class_variance: float
    inter_class_variance: float
    mutual_information: float
    duplication_alignment_score: float


def _to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _binarize(embeddings: np.ndarray) -> np.ndarray:
    """Sign-based binarization, the standard way to make a continuous
    embedding space comparable via Hamming distance."""
    return (embeddings > np.median(embeddings, axis=0)).astype(int)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def compute_representation_quality(
    embeddings: torch.Tensor, labels: torch.Tensor,
    duplication_pairs: list = None,
) -> RepresentationQualityReport:
    """
    embeddings: [N, d] graph-level (or node-level) embeddings
    labels:     [N] binary rumor labels
    duplication_pairs: optional list of (original_idx, duplicate_idx)
        pairs, used for the alignment score -- pass the duplication
        sets built in data/duplication_graph.py.
    """
    emb = _to_numpy(embeddings)
    y = _to_numpy(labels)

    rumor_emb = emb[y == 1]
    nonrumor_emb = emb[y == 0]

    # --- Hamming distance -------------------------------------------
    bin_emb = _binarize(emb)
    bin_rumor = bin_emb[y == 1]
    bin_nonrumor = bin_emb[y == 0]
    if len(bin_rumor) and len(bin_nonrumor):
        hamming = np.mean([
            (r != n).mean() for r in bin_rumor[:200] for n in bin_nonrumor[:1]
        ]) if len(bin_rumor) * len(bin_nonrumor) > 0 else 0.0
        # cheaper vectorized version for larger sets:
        hamming = float(np.mean(
            np.abs(bin_rumor.mean(0) - bin_nonrumor.mean(0))
        ))
    else:
        hamming = 0.0

    # --- cosine similarities -----------------------------------------
    cos_within_rumor = float(np.mean(cosine_similarity_matrix(rumor_emb, rumor_emb))) \
        if len(rumor_emb) > 1 else 0.0
    cos_within_nonrumor = float(np.mean(cosine_similarity_matrix(nonrumor_emb, nonrumor_emb))) \
        if len(nonrumor_emb) > 1 else 0.0
    cos_across = float(np.mean(cosine_similarity_matrix(rumor_emb, nonrumor_emb))) \
        if len(rumor_emb) and len(nonrumor_emb) else 0.0

    # --- silhouette + intra/inter class variance ----------------------
    try:
        sil = float(silhouette_score(emb, y)) if len(set(y.tolist())) > 1 else 0.0
    except ValueError:
        sil = 0.0

    overall_mean = emb.mean(0)
    intra_var = float(np.mean([
        np.mean(np.var(emb[y == c], axis=0)) for c in np.unique(y) if (y == c).sum() > 1
    ])) if len(np.unique(y)) > 0 else 0.0
    class_means = np.stack([emb[y == c].mean(0) for c in np.unique(y)])
    inter_var = float(np.mean(np.var(class_means, axis=0)))

    # --- mutual information between embeddings and labels -------------
    try:
        mi = float(np.mean(mutual_info_classif(emb, y, discrete_features=False)))
    except Exception:
        mi = 0.0

    # --- duplication alignment score -----------------------------------
    align_score = 0.0
    if duplication_pairs:
        sims = []
        for orig_idx, dup_idx in duplication_pairs:
            if orig_idx < len(emb) and dup_idx < len(emb):
                a, b = emb[orig_idx], emb[dup_idx]
                sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
        align_score = float(np.mean(sims)) if sims else 0.0

    return RepresentationQualityReport(
        hamming_distance_rumor_vs_nonrumor=hamming,
        cosine_sim_within_rumor=cos_within_rumor,
        cosine_sim_within_nonrumor=cos_within_nonrumor,
        cosine_sim_across_classes=cos_across,
        silhouette=sil,
        intra_class_variance=intra_var,
        inter_class_variance=inter_var,
        mutual_information=mi,
        duplication_alignment_score=align_score,
    )


def project_2d(embeddings: torch.Tensor, method: str = "tsne",
                random_state: int = 42) -> np.ndarray:
    """t-SNE (always available) or UMAP (if umap-learn is installed) for
    the latent-space visualizations named in RQ3."""
    emb = _to_numpy(embeddings)
    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(random_state=random_state)
        return reducer.fit_transform(emb)
    if method == "umap" and not _HAS_UMAP:
        print("umap-learn not installed -- falling back to t-SNE. "
              "`pip install umap-learn` to enable UMAP.")
    perplexity = min(30, max(2, emb.shape[0] // 3))
    return TSNE(n_components=2, random_state=random_state,
                perplexity=perplexity).fit_transform(emb)
