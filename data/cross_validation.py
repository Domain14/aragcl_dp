"""
Split protocol (RQ1 plan, Section 3): "A robust, stratified split
ratio with 5-fold cross-validation across multiple random seeds to
counteract dataset class imbalances."

This also directly implements the multi-seed mean+-std reporting
flagged as an outstanding item in train/metrics.py::mean_std_table --
use them together: run each (seed, fold) combination through your
training loop, collect a Metrics object per run, and pass the full
list into mean_std_table for the results chapter.
"""
from dataclasses import dataclass
from typing import List, Iterator, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class SplitPlan:
    seed: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def stratified_cv_splits(labels: List[int], n_folds: int = 5,
                          seeds: List[int] = (0, 1, 2)) -> Iterator[SplitPlan]:
    """
    Yields one SplitPlan per (seed, fold) combination -- e.g. with the
    default 5 folds x 3 seeds, 15 total train/val splits. This is what
    "5-fold cross-validation across multiple random seeds" means
    concretely: not just re-running the same 5 folds repeatedly, but
    re-SHUFFLING the fold assignment per seed (shuffle=True + distinct
    random_state per seed) so seeds genuinely vary which cascades land
    in which fold, not just model initialization.
    """
    y = np.asarray(labels)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield SplitPlan(seed=seed, fold=fold, train_idx=train_idx, val_idx=val_idx)


def summarize_split_plan(labels: List[int], n_folds: int = 5,
                          seeds: List[int] = (0, 1, 2)) -> str:
    """Sanity-check helper -- prints class balance per fold so you can
    confirm stratification actually held before spending compute on
    full training runs."""
    lines = [f"{'Seed':<6}{'Fold':<6}{'Train N':<10}{'Val N':<10}"
             f"{'Train %rumor':<15}{'Val %rumor':<12}"]
    y = np.asarray(labels)
    for plan in stratified_cv_splits(labels, n_folds, seeds):
        train_rate = y[plan.train_idx].mean()
        val_rate = y[plan.val_idx].mean()
        lines.append(f"{plan.seed:<6}{plan.fold:<6}{len(plan.train_idx):<10}"
                      f"{len(plan.val_idx):<10}{train_rate:<15.3f}{val_rate:<12.3f}")
    return "\n".join(lines)
