"""
Two-stage training harness (Section 3.6, "Training and Evaluation
Protocol"): (1) contrastive pretraining on augmented views, (2)
supervised finetuning of the pretrained encoder. Also supports
single-stage supervised-only training for the plain baselines
(GCN, R-GCN, BiGCN, Text-only, Majority) that don't have a
contrastive stage.

Logging format matches your existing notebook output exactly
(`Epoch N, Train Loss ..., Val Loss ..., Precision ..., Recall ...,
F1 ..., Fβ ...` then `Early stopping at epoch N (best F1=...)`) so
results drop straight into your existing comparison-table code.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import copy
import torch

from .metrics import compute_metrics, Metrics


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 5           # early stopping patience on val F1
    lr: float = 1e-3
    weight_decay: float = 5e-4
    pretrain_epochs: int = 20   # stage 1 (contrastive) -- ARAGCL-DP/RAGCL/GraphCL/GRACE only
    device: str = "cpu"


def _predict(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)


def supervised_train(model_name: str, forward_fn: Callable, model,
                      train_batches, val_batches, cfg: TrainConfig
                      ) -> List[Metrics]:
    """
    Generic single-stage supervised loop, used for GCN / R-GCN / BiGCN
    and for ARAGCL-DP's finetune stage.

    forward_fn(batch) -> (logits, y)   -- caller-supplied closure so
    this loop stays model-agnostic (different models need different
    forward signatures -- see run_all.py for the concrete closures).

    IMPORTANT: `model` must be the actual nn.Module (not just
    model.parameters()) -- this function checkpoints and restores the
    BEST validation-F1 state dict before returning. Previously this
    function tracked `best_f1` for the print statement and the early
    stop decision, but never actually saved or restored the
    corresponding weights -- so every "trained" model handed back to
    the caller was whatever the LAST epoch happened to produce (i.e.
    the overfit tail-end of training), not the best one. That silently
    undermined every result: robustness sweeps, RQ3 embeddings, and
    finetune() were all evaluating an overfit checkpoint instead of
    the one the early-stopping logic claimed to have selected.
    """
    print(f"\n=== Training {model_name} ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1, best_state, patience_left = -1.0, None, cfg.patience
    history = []

    for epoch in range(cfg.epochs):
        model_train_loss = 0.0
        n_batches = 0
        for batch in train_batches:
            optimizer.zero_grad()
            logits, y = forward_fn(batch)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            model_train_loss += loss.item()
            n_batches += 1
        train_loss = model_train_loss / max(n_batches, 1)

        val_loss, val_metrics = evaluate(forward_fn, val_batches)
        history.append(val_metrics)

        print(f"Epoch {epoch}, Train Loss {train_loss:.4f}, "
              f"Val Loss {val_loss:.4f}, Precision {val_metrics.precision:.3f}, "
              f"Recall {val_metrics.recall:.3f}, F1 {val_metrics.f1:.3f}, "
              f"Fβ {val_metrics.fbeta:.3f}")

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best F1={best_f1:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint (val F1={best_f1:.3f}) -- "
              f"returned model is NOT the last epoch's weights.")

    return history


def evaluate(forward_fn: Callable, batches):
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in batches:
            logits, y = forward_fn(batch)
            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
            n += 1
            all_true.extend(y.tolist())
            all_pred.extend(_predict(logits).tolist())
    metrics = compute_metrics(all_true, all_pred)
    return total_loss / max(n, 1), metrics


def contrastive_pretrain(model_name: str, model, train_batches, cfg: TrainConfig):
    """Stage 1 for ARAGCL-DP / RAGCL / GraphCL / GRACE: minimizes the
    contrastive loss only (Eq 3.8), no labels used."""
    print(f"\n=== Contrastive Pretraining: {model_name} ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    loss_history = []
    for epoch in range(cfg.pretrain_epochs):
        total = 0.0
        for graph, batch_idx in train_batches:
            optimizer.zero_grad()
            loss = model.pretrain_step(graph, batch=batch_idx)
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_loss = total / len(train_batches)
        loss_history.append(epoch_loss)
        print(f"Pretrain Epoch {epoch}, Contrastive Loss {epoch_loss:.4f}")
    return model, loss_history


def finetune(model_name: str, model, train_batches, val_batches, cfg: TrainConfig):
    """Stage 2: attach classifier, minimize supervised loss on the
    pretrained encoder (Eq 3.1/3.9 with unsup_weight effectively 0
    during this stage, since views aren't regenerated here -- if you
    want the JOINT loss during finetuning rather than a clean two-stage
    split, call model.finetune_step alongside model.pretrain_step
    inside one loop and combine with losses.contrastive.joint_loss
    instead of using this function)."""
    def forward_fn(batch):
        graph, batch_idx, y = batch
        logits = model.classify(graph, batch=batch_idx)
        return logits, y

    return supervised_train(model_name, forward_fn, model,
                             train_batches, val_batches, cfg)
