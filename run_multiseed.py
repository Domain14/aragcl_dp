"""
Multi-seed comparison across the FULL model suite (Majority through
Ablation 1) -- the defensible version of the single-run comparison
table in run_real_data.py.

Why this exists: your first real-data run showed BiGCN, ARAGCL-DP, and
Ablation 1 (no duplication at all) landing within ~0.01 F1 of each
other. On a SINGLE run, differences that small are not distinguishable
from noise -- you cannot claim "ARAGCL-DP beats BiGCN" or "duplication
helps" from one number each. This script runs every model across
multiple seeds (varying BOTH the train/val split and model
initialization per seed -- see load_and_prepare()'s seed parameter)
and reports mean +/- std, which is what you actually need to cite a
comparison in your thesis.

Run with:  python run_multiseed.py
Takes roughly (num_seeds x run_real_data.py's full model-suite time) --
this is the most expensive script in the project. Consider starting
with SEEDS = [0, 1, 2] (already the default) before scaling up.
"""
import random
import torch

from run_real_data import DATASET_DIR, load_and_prepare, train_all_models
from train.trainer import TrainConfig
from train.metrics import mean_std_table

SEEDS = [0, 1, 2, 3, 4]  # 5 seeds is a reasonable minimum for a defensible mean+-std;
                          # drop to [0, 1, 2] first if this is too slow to iterate on.


def main():
    all_results = {}  # model_name -> [Metrics, Metrics, ...] across seeds

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#' * 70}")
        print(f"# SEED {seed}  ({i + 1}/{len(SEEDS)})")
        print(f"{'#' * 70}")

        torch.manual_seed(seed)
        random.seed(seed)

        # verbose=False after the first seed -- the audit/split-balance
        # printouts don't depend on which seed you're on in any way
        # that needs re-reading every time; keeps the console readable
        # across 5 full runs instead of showing the same audit 5 times.
        data = load_and_prepare(DATASET_DIR, batch_size=8,
                                 verbose=(i == 0), seed=seed)
        cfg = TrainConfig(epochs=25, patience=6, pretrain_epochs=15)

        metrics, _ = train_all_models(data, data["feature_dim"], cfg, plot_path=None)

        for name, m in metrics.items():
            all_results.setdefault(name, []).append(m)

    print(f"\n{'=' * 70}")
    print(f"=== Multi-Seed Comparison ({len(SEEDS)} seeds: {SEEDS}) ===")
    print(f"{'=' * 70}")
    print(mean_std_table(all_results))

    print("\nRead this table, not any single-seed run, when writing comparative "
          "claims in your thesis (e.g. 'ARAGCL-DP outperforms BiGCN by X'). "
          "If two models' mean +/- std ranges overlap substantially, that "
          "difference is not yet a defensible claim -- consider a paired "
          "significance test (e.g. paired t-test across seeds on F1) before "
          "asserting one model is better than another.")


if __name__ == "__main__":
    main()
