"""
Step 6 (RQ1 plan): "Plot contrastive loss convergence curves vs.
epochs." Takes the loss_history list returned by
train/trainer.py::contrastive_pretrain.
"""
from typing import Dict, List
import matplotlib.pyplot as plt


def plot_convergence(loss_histories: Dict[str, List[float]], save_path: str = None):
    """
    loss_histories: {model_name: [loss_epoch0, loss_epoch1, ...]}
    e.g. {"RAGCL": ragcl_losses, "ARAGCL-DP": dp_losses} to compare
    convergence speed/stability across models on one figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, losses in loss_histories.items():
        ax.plot(range(len(losses)), losses, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Contrastive Pretraining Convergence")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved convergence plot to {save_path}")
    return fig
