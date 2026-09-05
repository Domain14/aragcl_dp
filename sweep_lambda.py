"""
Sweep lambda (Eq 3.7's duplication-influence weight) for ARAGCL-DP to
find whether tuning it closes the gap with BiGCN seen in the first
real-data run (BiGCN F1=0.773 vs ARAGCL-DP's untuned 0.727).

lambda controls how strongly duplicated-node signals are weighted
relative to ordinary reply-neighbor signals in the encoder:
    h_v^(l+1) = sigma(W^(l) * (sum_{u in N(v)} h_u + lambda * sum_{d in D_v} h_d))
lambda=0.5 was an arbitrary starting default, never tuned -- this
script trains ARAGCL-DP across a range of values and reports which
gives the best validation F1, using the SAME data split every time
(so differences are attributable to lambda, not to a different
random split).

Run with:  python sweep_lambda.py
Takes roughly (num_lambda_values x normal ARAGCL-DP training time) --
only trains ARAGCL-DP itself, not the full baseline suite, so it's
much faster than run_real_data.py end to end.
"""
import copy
import random
import torch

from run_real_data import load_and_prepare, DATASET_DIR
from models.aragcl_dp import ARAGCL_DP, ARAGCL_DP_Config
from train.trainer import TrainConfig, contrastive_pretrain, finetune
from train.metrics import compute_metrics, mean_std_table, Metrics

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
SEEDS = [0, 1, 2]  # multiple seeds per lambda -- a single-seed "best" value is
                    # not trustworthy on its own; report mean +/- std per lambda.


def evaluate_lambda(lam: float, seed: int, feature_dim: int,
                     data: dict, cfg: TrainConfig) -> Metrics:
    torch.manual_seed(seed)
    random.seed(seed)

    model_cfg = ARAGCL_DP_Config(in_dim=feature_dim, use_duplication=True,
                                  view_strategy="duplication_aware", lam=lam)
    model = ARAGCL_DP(model_cfg)
    model, _ = contrastive_pretrain(f"ARAGCL-DP (lambda={lam}, seed={seed})",
                                     model, data["pretrain_batches"], cfg)
    finetune(f"ARAGCL-DP (lambda={lam}, seed={seed})", model,
             data["train_batches"], data["val_batches"], cfg)

    # Final metrics on val using the RESTORED best checkpoint (see the
    # train/trainer.py fix -- finetune() now hands back the best-F1
    # epoch's weights, not the last epoch's).
    val_graph, val_batch_idx, val_y = data["val_graph"], data["val_batch_idx"], data["val_y"]
    with torch.no_grad():
        logits = model.classify(val_graph, batch=val_batch_idx)
    preds = logits.argmax(dim=1)
    return compute_metrics(val_y.tolist(), preds.tolist())


def main():
    cfg = TrainConfig(epochs=25, patience=6, pretrain_epochs=15)
    results = {}  # lambda -> [Metrics, Metrics, ...] across seeds

    for lam in LAMBDA_VALUES:
        results[f"lambda={lam}"] = []
        for seed in SEEDS:
            # Reload data PER SEED (varying the train/val split, not
            # just model init) -- consistent with run_multiseed.py.
            # Loading once and only varying model init (the original
            # version of this script) would understate real variance:
            # some of lambda's apparent effect could just be which
            # cascades landed in validation for a fixed split.
            data = load_and_prepare(DATASET_DIR, batch_size=8,
                                     verbose=(lam == LAMBDA_VALUES[0] and seed == SEEDS[0]),
                                     seed=seed)
            feature_dim = data["feature_dim"]
            m = evaluate_lambda(lam, seed, feature_dim, data, cfg)
            results[f"lambda={lam}"].append(m)
            print(f"lambda={lam} seed={seed}  F1={m.f1:.3f} Acc={m.accuracy:.3f}")

    print("\n=== Lambda Sweep Results (mean +/- std across seeds) ===")
    print(mean_std_table(results))

    best_lambda = max(results, key=lambda k: sum(m.f1 for m in results[k]) / len(results[k]))
    print(f"\nBest lambda by mean F1: {best_lambda}")
    print("Update ARAGCL_DP_Config's default `lam` in models/aragcl_dp.py "
          "to this value once confirmed, and re-run run_real_data.py for "
          "the full comparison against BiGCN with the tuned setting.")


if __name__ == "__main__":
    main()
