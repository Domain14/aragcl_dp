"""
Classification metrics for rumor detection.

Implements the formulas exactly as written in Section 3.6.1 of the proposal
("Propagation Trees Are Shallow: Duplication-based Augmented Graph
Contrastive Learning for Rumor Detection"), so results reported by this
module can be cited directly against that section in the write-up.

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
    F_beta    = (1 + b^2) * P * R / (b^2 * P + R)

F_beta defaults to beta=2, which weights recall more heavily than
precision -- appropriate for rumor detection, where a missed rumor
(false negative) is generally costlier than a false alarm.
"""
from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    fbeta: float

    def as_row(self, name: str) -> str:
        return (f"{name:<40}{self.accuracy:<13.3f}{self.precision:<13.3f}"
                f"{self.recall:<13.3f}{self.f1:<13.3f}{self.fbeta:<13.3f}")

    @staticmethod
    def header() -> str:
        return (f"{'Model':<40}{'Accuracy':<13}{'Precision':<13}"
                f"{'Recall':<13}{'F1':<13}{'Fβ':<13}")


def compute_metrics(y_true, y_pred, beta: float = 2.0) -> Metrics:
    """
    y_true, y_pred: 1D int tensors/arrays of {0, 1}, where 1 = rumor.

    beta=2.0 by default per the proposal's stated rationale (recall
    should be weighted more heavily than precision for misinformation
    detection). Change beta at the call site if a different trade-off
    is required for a specific experiment.
    """
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    b2 = beta ** 2
    fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)
             if (b2 * precision + recall) > 0 else 0.0)

    return Metrics(accuracy, precision, recall, f1, fbeta)


def mean_std_table(runs: dict) -> str:
    """
    runs: {model_name: [Metrics, Metrics, ...]} across seeds.
    Produces a mean +/- std table -- this is the "multiple seeds" next
    step flagged earlier: every reported number should carry variance,
    not just a single-run point estimate.
    """
    import statistics as st
    lines = [Metrics.header()]
    for name, metric_list in runs.items():
        cols = []
        for field in ("accuracy", "precision", "recall", "f1", "fbeta"):
            vals = [getattr(m, field) for m in metric_list]
            mean = st.mean(vals)
            std = st.stdev(vals) if len(vals) > 1 else 0.0
            cols.append(f"{mean:.3f}±{std:.3f}")
        lines.append(f"{name:<40}" + "".join(f"{c:<15}" for c in cols))
    return "\n".join(lines)
