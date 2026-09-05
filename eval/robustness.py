"""
RQ2 robustness sweep + the still-pending truncated/early-cascade
evaluation.

Your earlier logs reported one aggregate number per augmentation
strategy. This module produces the full p_n x epsilon grid (needed to
show the gap between duplication-aware and standard centrality WIDENS
under perturbation, not just that it exists on average -- a much more
defensible robustness claim), plus the early-detection evaluation your
own proposal cites Thota et al. (2023) for and that you flagged as
"planned" but not yet run.
"""
from dataclasses import dataclass
from typing import Callable, List, Dict
import torch

from train.metrics import compute_metrics, Metrics


@dataclass
class SweepPoint:
    p_n: float
    epsilon: float
    metrics: Metrics


def robustness_sweep(
    model, forward_fn: Callable, val_batches,
    p_n_values: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
    epsilon_values: List[float] = (0.0, 0.25, 0.5, 1.0),
) -> List[SweepPoint]:
    """
    forward_fn(model, batch, p_n, epsilon) -> (logits, y)
        caller-supplied closure that re-generates an augmented view of
        each val batch at the given (p_n, epsilon) before running the
        model -- this is what actually tests robustness, as opposed to
        evaluating on the clean graph.
    """
    results = []
    for p_n in p_n_values:
        for eps in epsilon_values:
            all_true, all_pred = [], []
            with torch.no_grad():
                for batch in val_batches:
                    logits, y = forward_fn(model, batch, p_n, eps)
                    all_true.extend(y.tolist())
                    all_pred.extend(logits.argmax(dim=1).tolist())
            m = compute_metrics(all_true, all_pred)
            results.append(SweepPoint(p_n=p_n, epsilon=eps, metrics=m))
            print(f"p_n={p_n:.2f} eps={eps:.2f}  "
                  f"Acc={m.accuracy:.3f} F1={m.f1:.3f} Fβ={m.fbeta:.3f}")
    return results


def sweep_to_grid(results: List[SweepPoint], field: str = "f1"):
    """Reshape sweep results into a (p_n x epsilon) grid for a heatmap
    -- much more informative for the write-up than a single averaged
    robustness number."""
    p_ns = sorted(set(r.p_n for r in results))
    epsilons = sorted(set(r.epsilon for r in results))
    grid = [[next(getattr(r.metrics, field) for r in results
                  if r.p_n == p and r.epsilon == e)
             for e in epsilons] for p in p_ns]
    return p_ns, epsilons, grid


def truncate_cascade(reply_edge_index: torch.Tensor,
                      propagation_depth: torch.Tensor,
                      timestamps: torch.Tensor,
                      keep_ratio: float = None,
                      max_depth: int = None,
                      max_time: float = None) -> torch.Tensor:
    """
    Early/truncated-cascade evaluation (Section 3.4 RQ2 framing +
    Thota et al. 2023 early-detection precedent cited in your related
    work): returns a boolean node mask selecting only nodes that would
    have been observed early in the cascade's lifetime, under ONE of
    three truncation rules (pass exactly one):

      keep_ratio  -- keep the earliest `keep_ratio` fraction of nodes
                      by timestamp (e.g. 0.25 = first quarter of the
                      cascade's observed lifetime)
      max_depth   -- keep only nodes with propagation_depth <= max_depth
      max_time    -- keep only nodes with timestamp <= max_time
                      (absolute cutoff, e.g. "first 60 minutes")
    """
    n = timestamps.size(0)
    if keep_ratio is not None:
        order = timestamps.argsort()
        cutoff = order[: max(1, int(n * keep_ratio))]
        mask = torch.zeros(n, dtype=torch.bool)
        mask[cutoff] = True
        return mask
    if max_depth is not None:
        return propagation_depth <= max_depth
    if max_time is not None:
        return timestamps <= max_time
    raise ValueError("Specify exactly one of keep_ratio / max_depth / max_time")


def early_detection_curve(
    model, forward_fn: Callable, val_batches,
    keep_ratios: List[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> Dict[float, Metrics]:
    """
    forward_fn(model, batch, keep_ratio) -> (logits, y), applying
    truncate_cascade internally before the forward pass.

    Produces accuracy/F1 as a function of how much of the cascade has
    been observed -- the actual deliverable your proposal describes as
    'evaluate truncated cascades (early cascade evaluation)' and that
    was still marked pending.
    """
    curve = {}
    for ratio in keep_ratios:
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_batches:
                logits, y = forward_fn(model, batch, ratio)
                all_true.extend(y.tolist())
                all_pred.extend(logits.argmax(dim=1).tolist())
        m = compute_metrics(all_true, all_pred)
        curve[ratio] = m
        print(f"keep_ratio={ratio:.2f}  Acc={m.accuracy:.3f} F1={m.f1:.3f} Fβ={m.fbeta:.3f}")
    return curve
