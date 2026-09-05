"""
Full pipeline on your REAL dataset (Weibo-style JSON cascades), not
synthetic data. This is what run_all.py always was NOT meant to be --
run_all.py is a smoke test on fabricated data and never touches your
real files, no matter what path you point elsewhere. This script is
the one that actually loads /storage/weibo1 (or wherever you point
DATASET_DIR) and trains every model on it.

Run with:  python run_real_data.py
Same requirements as run_all.py (pip install -r requirements.txt).
"""
import random
import torch
from sklearn.model_selection import train_test_split

from data.weibo_json_loader import load_dataset, attach_tfidf_features, audit_dataset
from data.duplication_graph import build_duplication_graph, build_static_graph, DuplicationGraph
from data.reshare_classifier import ReshareClassifier
from data.cross_validation import summarize_split_plan
from models.baselines import MajorityBaseline, TextOnlyBaseline, VanillaGCN, RGCNBaseline, BiGCN
from models.aragcl_dp import (make_ragcl, make_graphcl, make_grace, make_aragcl_dp,
                               make_ablation2, make_dataset_centrality)
from augmentation.views import ViewConfig, generate_view
from train.trainer import TrainConfig, supervised_train, contrastive_pretrain, finetune, evaluate
from train.metrics import compute_metrics
from eval.robustness import robustness_sweep, early_detection_curve, truncate_cascade
from eval.representation_quality import compute_representation_quality
from eval.structural_fidelity import evaluate_structural_fidelity, print_fidelity_report
from eval.convergence_plots import plot_convergence

DATASET_DIR = "/storage/weibo1"  # <-- point this at your actual data folder


def batch_graphs(graphs, labels_subset):
    xs, reply_eis, dup_eis, batch_idx = [], [], [], []
    node_offset = 0
    for i, g in enumerate(graphs):
        n = g.x.size(0)
        xs.append(g.x)
        reply_eis.append(g.reply_edge_index + node_offset)
        dup_eis.append(g.duplication_edge_index + node_offset)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n
    merged = DuplicationGraph(
        x=torch.cat(xs, dim=0),
        reply_edge_index=torch.cat(reply_eis, dim=1) if reply_eis else torch.zeros((2, 0), dtype=torch.long),
        duplication_edge_index=torch.cat(dup_eis, dim=1) if dup_eis else torch.zeros((2, 0), dtype=torch.long),
        timestamps=torch.cat([g.timestamps for g in graphs]),
        propagation_depth=torch.cat([g.propagation_depth for g in graphs]),
        duplication_freq=torch.cat([g.duplication_freq for g in graphs]),
        root_mask=torch.cat([g.root_mask for g in graphs]),
    )
    # Carry precomputed_centrality (Degree/Pagerank/etc.) through the
    # merge too -- DuplicationGraph's default is an empty dict, so
    # without this, every batched graph silently loses the centrality
    # attached before batching, even though each individual graph had
    # it. Only keys present on EVERY graph in this batch are kept, in
    # case any single file was missing centrality data.
    if graphs and graphs[0].precomputed_centrality:
        common_keys = set(graphs[0].precomputed_centrality.keys())
        for g in graphs[1:]:
            common_keys &= set(g.precomputed_centrality.keys())
        merged.precomputed_centrality = {
            key: torch.cat([g.precomputed_centrality[key] for g in graphs])
            for key in common_keys
        }
    return merged, torch.cat(batch_idx), labels_subset


def make_minibatches(graphs, labels, batch_size=8):
    chunks = []
    for i in range(0, len(graphs), batch_size):
        g_chunk = graphs[i:i + batch_size]
        y_chunk = torch.as_tensor(labels[i:i + batch_size], dtype=torch.long)
        merged, batch_idx, y = batch_graphs(g_chunk, y_chunk)
        chunks.append((merged, batch_idx, y))
    return chunks


def load_and_prepare(dataset_dir: str, batch_size: int = 8, verbose: bool = True,
                      seed: int = 0):
    """
    Everything from raw JSON files through ready-to-train batches, in
    one reusable place -- factored out of main() so other scripts
    (sweep_lambda.py, run_multiseed.py) don't have to duplicate ~50
    lines of loading/splitting/featurizing/graph-building code just to
    get a train/val batch set.

    seed controls the train/val SPLIT itself (random_state passed to
    train_test_split), not just model initialization. This matters for
    run_multiseed.py: if only the model's random seed varied while the
    split stayed fixed, the reported variance would understate real
    run-to-run variability -- some of a model's apparent strength or
    weakness can come from which cascades happened to land in
    validation, not just weight initialization.

    Returns a dict with everything callers need: train_batches,
    val_batches, pretrain_batches, feature_dim, train_labels, val_y,
    raw_cascades (for Lens 1), and the static (Ablation 1) batches too.
    """
    if verbose:
        print(f"=== Auditing {dataset_dir} ===")
        audit_dataset(dataset_dir)

    raw = load_dataset(dataset_dir)
    raw_cascades = [posts for posts, label, cen in raw]
    labels_list = [label for posts, label, cen in raw]
    centralities = [cen for posts, label, cen in raw]
    if verbose:
        print(f"\nLoaded {len(raw_cascades)} real cascades from {dataset_dir}")

    indices = list(range(len(raw_cascades)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels_list, random_state=seed)
    train_cascades = [raw_cascades[i] for i in train_idx]
    val_cascades = [raw_cascades[i] for i in val_idx]
    train_labels = [labels_list[i] for i in train_idx]
    val_labels = [labels_list[i] for i in val_idx]

    if verbose:
        print(f"Train: {len(train_cascades)}  Val: {len(val_cascades)}")
        print("\n=== Stratified split balance check ===")
        print(summarize_split_plan(train_labels, n_folds=min(5, len(train_cascades)), seeds=[seed]))

    vectorizer = attach_tfidf_features(train_cascades, other_cascade_sets=[val_cascades])
    feature_dim = train_cascades[0][0].feature.shape[0]
    if verbose:
        print(f"TF-IDF feature dim: {feature_dim} (vocab size {len(vectorizer.vocabulary_)})")

    train_graphs = [build_duplication_graph(p) for p in train_cascades]
    val_graphs = [build_duplication_graph(p) for p in val_cascades]
    for g, cen in zip(train_graphs, [centralities[i] for i in train_idx]):
        g.precomputed_centrality.update(cen)
    for g, cen in zip(val_graphs, [centralities[i] for i in val_idx]):
        g.precomputed_centrality.update(cen)

    val_graph, val_batch_idx, val_y = batch_graphs(val_graphs, torch.tensor(val_labels, dtype=torch.long))
    train_batches = make_minibatches(train_graphs, train_labels, batch_size=batch_size)
    val_batches = [(val_graph, val_batch_idx, val_y)]
    pretrain_batches = [(g, b_idx) for g, b_idx, _ in train_batches]

    static_train_graphs = [build_static_graph(p) for p in train_cascades]
    static_val_graphs = [build_static_graph(p) for p in val_cascades]
    static_train_batches = make_minibatches(static_train_graphs, train_labels, batch_size=batch_size)
    static_pretrain_batches = [(g, b_idx) for g, b_idx, _ in static_train_batches]
    static_val_graph, static_val_batch, static_val_y = batch_graphs(
        static_val_graphs, torch.tensor(val_labels, dtype=torch.long))

    return {
        "raw_cascades": raw_cascades,
        "train_cascades": train_cascades, "val_cascades": val_cascades,
        "train_labels": train_labels, "val_labels": val_labels,
        "feature_dim": feature_dim,
        "train_batches": train_batches, "val_batches": val_batches,
        "pretrain_batches": pretrain_batches,
        "val_graph": val_graph, "val_batch_idx": val_batch_idx, "val_y": val_y,
        "static_train_batches": static_train_batches,
        "static_pretrain_batches": static_pretrain_batches,
        "static_val_batches": [(static_val_graph, static_val_batch, static_val_y)],
    }


def train_all_models(data: dict, feature_dim: int, cfg: TrainConfig,
                      plot_path: str = None):
    """
    Trains the full model suite (Majority through Ablation 1) on one
    already-prepared dataset (from load_and_prepare()) and returns
    (metrics, trained_models):
      metrics        -- {model_name: Metrics} final val performance
                         using each model's BEST checkpoint (see the
                         train/trainer.py restore fix)
      trained_models -- {model_name: model object}, for callers that
                         need the actual model afterward (e.g. main()'s
                         robustness sweep / RQ3 section, which need the
                         trained ARAGCL-DP model specifically)

    Factored out of main() so run_multiseed.py can call this exact
    same training logic across multiple seeds without copy-pasting it
    -- any future change to how models are trained only needs to
    happen here, not in two places that can drift out of sync.
    """
    train_cascades, val_cascades = data["train_cascades"], data["val_cascades"]
    train_labels = data["train_labels"]
    train_batches, val_batches = data["train_batches"], data["val_batches"]
    pretrain_batches = data["pretrain_batches"]
    val_y = data["val_y"]

    metrics = {}
    trained_models = {}

    # --- Majority baseline ------------------------------------------------------
    maj = MajorityBaseline()
    maj.fit(torch.tensor(train_labels))
    preds = maj.predict(len(val_y))
    metrics["Majority"] = compute_metrics(val_y.tolist(), preds.tolist())
    print("\n=== Majority Baseline ===")
    print(metrics["Majority"])

    # --- Text-only baseline (TF-IDF on root post only) ---------------------------
    train_root_tfidf = torch.stack([posts[0].feature for posts in train_cascades])
    val_root_tfidf = torch.stack([posts[0].feature for posts in val_cascades])
    text_model = TextOnlyBaseline(in_dim=feature_dim)
    def text_forward(batch):
        x_tfidf, y = batch
        return text_model(x_tfidf), y
    supervised_train("Text-only (TF-IDF)", text_forward, text_model,
                      [(train_root_tfidf, torch.tensor(train_labels))],
                      [(val_root_tfidf, val_y)], cfg)
    with torch.no_grad():
        preds = text_model(val_root_tfidf).argmax(dim=1)
    metrics["Text-only"] = compute_metrics(val_y.tolist(), preds.tolist())
    trained_models["Text-only"] = text_model

    # --- Vanilla GCN --------------------------------------------------------------
    gcn = VanillaGCN(in_dim=feature_dim)
    def gcn_forward(batch):
        g, b_idx, y = batch
        return gcn(g.x, g.edge_index, batch=b_idx), y
    supervised_train("Vanilla GCN", gcn_forward, gcn, train_batches, val_batches, cfg)
    _, metrics["Vanilla GCN"] = evaluate(gcn_forward, val_batches)
    trained_models["Vanilla GCN"] = gcn

    # --- R-GCN ----------------------------------------------------------------------
    rgcn = RGCNBaseline(in_dim=feature_dim)
    def rgcn_forward(batch):
        g, b_idx, y = batch
        edge_index, edge_type = RGCNBaseline.build_edge_type(g.reply_edge_index, g.duplication_edge_index)
        return rgcn(g.x, edge_index, edge_type, batch=b_idx), y
    supervised_train("R-GCN", rgcn_forward, rgcn, train_batches, val_batches, cfg)
    _, metrics["R-GCN"] = evaluate(rgcn_forward, val_batches)
    trained_models["R-GCN"] = rgcn

    # --- BiGCN ------------------------------------------------------------------------
    bigcn = BiGCN(in_dim=feature_dim)
    def bigcn_forward(batch):
        g, b_idx, y = batch
        bu_ei = BiGCN.bottom_up(g.reply_edge_index)
        return bigcn(g.x, g.reply_edge_index, bu_ei, g.root_mask, batch=b_idx), y
    supervised_train("BiGCN", bigcn_forward, bigcn, train_batches, val_batches, cfg)
    _, metrics["BiGCN"] = evaluate(bigcn_forward, val_batches)
    trained_models["BiGCN"] = bigcn

    # --- RAGCL / GraphCL / GRACE / ARAGCL-DP (+ ablations, GAT, dataset-centrality) --
    loss_histories = {}
    for name, factory in [("RAGCL", make_ragcl), ("GraphCL", make_graphcl),
                           ("GRACE", make_grace), ("ARAGCL-DP", make_aragcl_dp),
                           ("ARAGCL-DP (Ablation 2, no dup edges)", make_ablation2),
                           ("ARAGCL-DP (GAT backbone)",
                            lambda in_dim: make_aragcl_dp(in_dim, backbone="gat")),
                           ("ARAGCL-DP (dataset centrality)",
                            lambda in_dim: make_dataset_centrality(in_dim, centrality_key="Degree"))]:
        model = factory(in_dim=feature_dim)
        model, loss_history = contrastive_pretrain(name, model, pretrain_batches, cfg)
        loss_histories[name] = loss_history
        finetune(name, model, train_batches, val_batches, cfg)
        with torch.no_grad():
            logits = model.classify(data["val_graph"], batch=data["val_batch_idx"])
        metrics[name] = compute_metrics(val_y.tolist(), logits.argmax(dim=1).tolist())
        trained_models[name] = model

    if plot_path:
        plot_convergence(loss_histories, save_path=plot_path)

    # --- Ablation 1: static graph construction -----------------------------------
    ablation1_model = make_aragcl_dp(in_dim=feature_dim)
    ablation1_model, _ = contrastive_pretrain(
        "ARAGCL-DP (Ablation 1, static graph)", ablation1_model,
        data["static_pretrain_batches"], cfg)
    finetune("ARAGCL-DP (Ablation 1, static graph)", ablation1_model,
             data["static_train_batches"], data["static_val_batches"], cfg)
    static_val_graph, static_val_batch, static_val_y = data["static_val_batches"][0]
    with torch.no_grad():
        logits = ablation1_model.classify(static_val_graph, batch=static_val_batch)
    metrics["ARAGCL-DP (Ablation 1, static graph)"] = compute_metrics(
        static_val_y.tolist(), logits.argmax(dim=1).tolist())
    trained_models["ARAGCL-DP (Ablation 1, static graph)"] = ablation1_model

    return metrics, trained_models


def main():
    torch.manual_seed(0)
    random.seed(0)

    data = load_and_prepare(DATASET_DIR, batch_size=8, verbose=True, seed=0)
    raw_cascades = data["raw_cascades"]
    val_graph, val_batch_idx, val_y = data["val_graph"], data["val_batch_idx"], data["val_y"]
    val_batches = data["val_batches"]

    # --- Lens 1: structural fidelity on REAL cascades ----------------------------
    fidelity = evaluate_structural_fidelity(raw_cascades)
    print_fidelity_report(fidelity)

    cfg = TrainConfig(epochs=25, patience=6, pretrain_epochs=15)

    metrics, trained_models = train_all_models(
        data, data["feature_dim"], cfg, plot_path="contrastive_convergence_real.png")

    print("\n=== Final Comparison (single seed=0 run -- see run_multiseed.py for a defensible multi-seed table) ===")
    for name, m in metrics.items():
        print(f"{name:<40}{m}")

    # --- RQ2: robustness sweep -----------------------------------------------------
    aragcl_dp_model = trained_models["ARAGCL-DP"]
    def robustness_forward(model, batch, p_n, eps):
        g, b_idx, y = batch
        vcfg = ViewConfig(strategy=model.cfg.view_strategy, p_n=p_n, p_e=p_n, p_m=p_n, epsilon=eps)
        x_aug, edge_index_aug, _ = generate_view(
            g.x, g.reply_edge_index, vcfg,
            duplication_freq=g.duplication_freq, propagation_depth=g.propagation_depth,
            root_mask=g.root_mask, timestamps=g.timestamps,
            precomputed_centrality=g.precomputed_centrality,
        )
        g_aug = DuplicationGraph(
            x=x_aug, reply_edge_index=edge_index_aug,
            duplication_edge_index=g.duplication_edge_index,
            timestamps=g.timestamps, propagation_depth=g.propagation_depth,
            duplication_freq=g.duplication_freq, root_mask=g.root_mask,
        )
        return model.classify(g_aug, batch=b_idx), y
    print("\n=== RQ2 Robustness Sweep: ARAGCL-DP ===")
    robustness_sweep(aragcl_dp_model, robustness_forward, val_batches,
                      p_n_values=[0.0, 0.2, 0.4], epsilon_values=[0.0, 0.5])

    # --- Early/truncated cascade evaluation ------------------------------------------
    def early_forward(model, batch, keep_ratio):
        g, b_idx, y = batch
        mask = truncate_cascade(g.reply_edge_index, g.propagation_depth, g.timestamps, keep_ratio=keep_ratio)
        x_trunc = g.x.clone()
        x_trunc[~mask] = 0.0
        g_trunc = DuplicationGraph(
            x=x_trunc, reply_edge_index=g.reply_edge_index,
            duplication_edge_index=g.duplication_edge_index,
            timestamps=g.timestamps, propagation_depth=g.propagation_depth,
            duplication_freq=g.duplication_freq, root_mask=g.root_mask,
        )
        return model.classify(g_trunc, batch=b_idx), y
    print("\n=== Early/Truncated Cascade Evaluation ===")
    early_detection_curve(aragcl_dp_model, early_forward, val_batches)

    # --- RQ3: representation quality -----------------------------------------------
    with torch.no_grad():
        _, graph_embeddings = aragcl_dp_model.encoder(
            val_graph.x, val_graph.reply_edge_index, val_graph.duplication_edge_index,
            batch=val_batch_idx)
    print("\n=== RQ3 Representation Quality ===")
    print(compute_representation_quality(graph_embeddings, val_y))

    print("\nReal-data run complete.")


if __name__ == "__main__":
    main()
