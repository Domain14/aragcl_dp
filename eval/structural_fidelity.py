"""
Lens 1: Structural Fidelity Evaluation (RQ1 plan, Section 2/4 Step 2).

Compares TWO graph-construction pipelines against ground truth, per
cascade:
  - static graph  (data/duplication_graph.py::build_static_graph)
  - ARAGCL-DP graph (data/duplication_graph.py::build_duplication_graph)

For each, we ask: does the CONSTRUCTED graph let you recover the same
(Breadth, Depth) cell as the ground-truth cascade stats computed
straight from the raw posts?

The expected finding (and the actual claim RQ1 needs evidence for):
the static graph's duplication_freq is zero by construction, so
N_r recovered from the static graph is always 0 -- it can NEVER
correctly classify a cascade with real reshare activity (Non-Extensive
/ Extensive / Cascading), regardless of GNN sophistication downstream.
The ARAGCL-DP graph should recover N_r exactly by construction,
since duplication_freq is built directly from the same duplicate
groups the ground-truth stats use. This is a graph-construction-level
result, obtained BEFORE any classifier training -- exactly per your
plan's Lens 1 framing ("...before running classifier training").
"""
from dataclasses import dataclass
from typing import List

from data.duplication_graph import RawPost, build_static_graph, build_duplication_graph
from data.reshare_classifier import (
    ReshareClassifier, CascadeStats, build_grid, grid_to_table, IMPOSSIBLE_CELLS,
)


@dataclass
class FidelityResult:
    ground_truth_grid: dict
    static_grid: dict
    aragcl_dp_grid: dict
    static_preservation_rate: float   # fraction of cascades where static-derived cell == ground-truth cell
    aragcl_dp_preservation_rate: float
    n_cascades: int


def evaluate_structural_fidelity(cascades: List[List[RawPost]]) -> FidelityResult:
    gt_labels, static_labels, dp_labels = [], [], []
    static_match, dp_match = 0, 0

    for posts in cascades:
        gt_stats = ReshareClassifier.stats_from_raw_posts(posts)
        gt_label = ReshareClassifier.classify(gt_stats)
        gt_labels.append(gt_label)

        static_graph = build_static_graph(posts)
        static_stats = ReshareClassifier.stats_from_duplication_graph(static_graph)
        static_label = ReshareClassifier.classify(static_stats)
        static_labels.append(static_label)
        static_match += int(static_label == gt_label)

        dp_graph = build_duplication_graph(posts)
        dp_stats = ReshareClassifier.stats_from_duplication_graph(dp_graph)
        dp_label = ReshareClassifier.classify(dp_stats)
        dp_labels.append(dp_label)
        dp_match += int(dp_label == gt_label)

    n = len(cascades)
    return FidelityResult(
        ground_truth_grid=build_grid(gt_labels),
        static_grid=build_grid(static_labels),
        aragcl_dp_grid=build_grid(dp_labels),
        static_preservation_rate=static_match / n if n else 0.0,
        aragcl_dp_preservation_rate=dp_match / n if n else 0.0,
        n_cascades=n,
    )


def print_fidelity_report(result: FidelityResult):
    print(f"\n=== Lens 1: Structural Fidelity ({result.n_cascades} cascades) ===")
    print("\n-- Ground truth (raw posts) --")
    print(grid_to_table(result.ground_truth_grid))
    print("\n-- Static graph (no duplication) --")
    print(grid_to_table(result.static_grid))
    print(f"Preservation rate vs. ground truth: {result.static_preservation_rate:.3f}")
    print("\n-- ARAGCL-DP graph (with duplication) --")
    print(grid_to_table(result.aragcl_dp_grid))
    print(f"Preservation rate vs. ground truth: {result.aragcl_dp_preservation_rate:.3f}")
    print(f"\n({len(IMPOSSIBLE_CELLS)} grid cells are structurally impossible "
          f"by the plan's own Breadth definitions and are marked N/A, not 0 "
          f"-- see the note at the top of data/reshare_classifier.py)")
