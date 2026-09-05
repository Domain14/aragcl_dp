"""
End-to-end smoke test on SYNTHETIC data.

Purpose: verify every module wires together correctly (graph
construction -> augmentation -> encoder -> contrastive loss ->
supervised finetune -> RQ2 sweep -> RQ3 representation quality)
BEFORE you plug in Twitter15/16, Weibo or DRWeibo. Getting a clean run
end-to-end here first will save you debugging time once real data
(with its own quirks -- missing timestamps, encoding issues, class
imbalance) is in the loop.

Run with:  python run_all.py
Requires:  pip install -r requirements.txt   (torch + torch_geometric
           are NOT available in the sandbox this was written in, so
           this script has been syntax-checked but not executed --
           run it in your own environment, e.g. the same
           /notebooks/ARGCL_DP setup your existing logs came from.)

Replace `make_synthetic_cascade` with your real Twitter15/16/Weibo/
DRWeibo loader (Section 3.5.2) once this smoke test passes -- every
downstream function expects exactly the RawPost / DuplicationGraph
interfaces defined in data/duplication_graph.py, so the rest of the
pipeline does not change.
"""
import random
import torch

from data.duplication_graph import RawPost, build_duplication_graph, build_static_graph, DuplicationGraph
from data.reshare_classifier import ReshareClassifier
from data.text_features import fit_tfidf, transform_tfidf, root_texts_from_cascades
from data.cross_validation import stratified_cv_splits, summarize_split_plan
from models.baselines import MajorityBaseline, TextOnlyBaseline, VanillaGCN, RGCNBaseline, BiGCN
from models.aragcl_dp import (make_ragcl, make_graphcl, make_grace, make_aragcl_dp,
                               make_ablation2, ARAGCL_DP_Config, ARAGCL_DP)
from train.trainer import TrainConfig, supervised_train, contrastive_pretrain, finetune
from train.metrics import Metrics, compute_metrics, mean_std_table
from eval.robustness import robustness_sweep, early_detection_curve, truncate_cascade
from eval.representation_quality import compute_representation_quality
from eval.structural_fidelity import evaluate_structural_fidelity, print_fidelity_report
from eval.convergence_plots import plot_convergence

FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Synthetic data generator -- STAND-IN for your real dataset loader
# ---------------------------------------------------------------------------
def make_synthetic_cascade(cascade_id: int, n_posts: int = 12,
                            duplication_rate: float = 0.3) -> list:
    """Builds a synthetic reply tree with some reshares (exact-text
    duplicates of earlier posts), mimicking the structure
    data/duplication_graph.py expects."""
    posts = []
    root = RawPost(post_id=0, parent_id=None, cascade_id=cascade_id,
                    timestamp=0.0, text=f"root_text_{cascade_id}",
                    feature=torch.randn(FEATURE_DIM))
    posts.append(root)
    for i in range(1, n_posts):
        parent_id = random.randint(0, i - 1)
        is_duplicate = random.random() < duplication_rate and i > 1
        if is_duplicate:
            source = random.choice(posts)
            text = source.text  # exact-text reshare
            feature = source.feature.clone()
        else:
            text = f"post_{cascade_id}_{i}"
            feature = torch.randn(FEATURE_DIM)
        posts.append(RawPost(
            post_id=i, parent_id=parent_id, cascade_id=cascade_id,
            timestamp=float(i) + random.random(), text=text, feature=feature,
        ))
    return posts


def make_dataset(n_cascades: int = 40):
    """Returns (raw_cascades, graphs, labels) -- raw_cascades is kept
    around (not just the constructed graphs) because Lens 1 structural
    fidelity, the TF-IDF text baseline, and stratified-CV splitting all
    need access to the original posts, not just one graph construction
    of them."""
    raw_cascades, graphs, labels = [], [], []
    for cid in range(n_cascades):
        posts = make_synthetic_cascade(cid, n_posts=random.randint(8, 20))
        raw_cascades.append(posts)
        g = build_duplication_graph(posts)
        graphs.append(g)
        # synthetic label: cascades with more duplication skew "rumor"
        labels.append(1 if g.duplication_freq.sum() > 3 else 0)
    return raw_cascades, graphs, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Minimal batching helpers (manual, so this has zero dependency on how
# your real data loader batches graphs -- swap for torch_geometric's
# Batch.from_data_list if you migrate DuplicationGraph to a proper
# torch_geometric.data.Data subclass later).
# ---------------------------------------------------------------------------
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
    return merged, torch.cat(batch_idx), labels_subset


def make_minibatches(graphs, labels, batch_size=8):
    """
    Splits graphs into several smaller batches instead of one giant
    batch containing the whole training set. With N graphs as ONE
    batch, `epochs=E` gives exactly E gradient steps total -- nowhere
    near enough for even a small GCN to escape predicting a constant
    class. With minibatches of size `batch_size`, each epoch does
    ceil(N/batch_size) gradient steps instead of 1, which is what
    actually lets the model start discriminating between classes
    rather than collapsing to "always predict the majority class"
    (which is what produced the flat, degenerate 0.750/0.000 results
    seen in earlier runs).
    """
    chunks = []
    for i in range(0, len(graphs), batch_size):
        g_chunk = graphs[i:i + batch_size]
        y_chunk = torch.as_tensor(labels[i:i + batch_size], dtype=torch.long)
        merged, batch_idx, y = batch_graphs(g_chunk, y_chunk)
        chunks.append((merged, batch_idx, y))
    return chunks


def main():
    torch.manual_seed(0)
    random.seed(0)

    raw_cascades, graphs, labels = make_dataset(n_cascades=40)
    n_train = 28
    train_graph, train_batch_idx, train_y = batch_graphs(graphs[:n_train], labels[:n_train])
    val_graph, val_batch_idx, val_y = batch_graphs(graphs[n_train:], labels[n_train:])

    # Mini-batched training set (several gradient steps/epoch instead of
    # one) -- see make_minibatches() docstring for why this matters.
    train_batches = make_minibatches(graphs[:n_train], labels[:n_train].tolist(), batch_size=8)
    val_batches = [(val_graph, val_batch_idx, val_y)]

    cfg = TrainConfig(epochs=25, patience=6, pretrain_epochs=15)  # enough to actually learn something on this toy data

    # --- RQ1 plan, Section 3: split protocol sanity check -----------------------
    print("\n=== Stratified 5-Fold CV Split Check (across seeds) ===")
    print(summarize_split_plan(labels.tolist(), n_folds=5, seeds=[0, 1, 2]))
    # NOTE: this smoke test still trains on ONE fixed 28/12 split below for
    # speed. For real experiments, loop over stratified_cv_splits(...) and
    # feed each (train_idx, val_idx) into batch_graphs(), then pass every
    # resulting Metrics object into train.metrics.mean_std_table().

    # --- RQ1 plan, Lens 1: Structural Fidelity (BEFORE any training) ------------
    fidelity = evaluate_structural_fidelity(raw_cascades)
    print_fidelity_report(fidelity)

    # --- Majority baseline ------------------------------------------------
    maj = MajorityBaseline()
    maj.fit(train_y)
    preds = maj.predict(len(val_y))
    print("\n=== Majority Baseline ===")
    print(compute_metrics(val_y.tolist(), preds.tolist()))

    # --- Text-only baseline (TF-IDF) ---------------------------------------------
    train_texts = root_texts_from_cascades(raw_cascades[:n_train])
    val_texts = root_texts_from_cascades(raw_cascades[n_train:])
    vectorizer = fit_tfidf(train_texts, max_features=128)
    train_tfidf = transform_tfidf(vectorizer, train_texts)
    val_tfidf = transform_tfidf(vectorizer, val_texts)
    text_model = TextOnlyBaseline(in_dim=train_tfidf.size(1))
    def text_forward(batch):
        x_tfidf, y = batch
        return text_model(x_tfidf), y
    supervised_train("Text-only (TF-IDF)", text_forward, text_model,
                      [(train_tfidf, train_y)], [(val_tfidf, val_y)], cfg)

    # --- Vanilla GCN --------------------------------------------------------
    gcn = VanillaGCN(in_dim=FEATURE_DIM)
    def gcn_forward(batch):
        g, b_idx, y = batch
        return gcn(g.x, g.edge_index, batch=b_idx), y
    supervised_train("Vanilla GCN", gcn_forward, gcn,
                      train_batches, val_batches, cfg)

    # --- R-GCN ----------------------------------------------------------------
    rgcn = RGCNBaseline(in_dim=FEATURE_DIM)
    def rgcn_forward(batch):
        g, b_idx, y = batch
        edge_index, edge_type = RGCNBaseline.build_edge_type(
            g.reply_edge_index, g.duplication_edge_index)
        return rgcn(g.x, edge_index, edge_type, batch=b_idx), y
    supervised_train("R-GCN", rgcn_forward, rgcn,
                      train_batches, val_batches, cfg)

    # --- BiGCN ------------------------------------------------------------------
    bigcn = BiGCN(in_dim=FEATURE_DIM)
    def bigcn_forward(batch):
        g, b_idx, y = batch
        bu_ei = BiGCN.bottom_up(g.reply_edge_index)
        return bigcn(g.x, g.reply_edge_index, bu_ei, g.root_mask, batch=b_idx), y
    supervised_train("BiGCN", bigcn_forward, bigcn,
                      train_batches, val_batches, cfg)

    # --- RAGCL / GraphCL / GRACE / ARAGCL-DP (two-stage) -----------------------
    # Includes both RQ1 Step 5 ablations:
    #   Ablation 2 (encoder ignores duplication edges, centrality signal kept)
    #     -> make_ablation2, added alongside the full model below.
    #   Ablation 1 (static graph construction) is exercised separately via
    #     build_static_graph() -- see the block right after this loop.
    # Also includes the GAT backbone variant (RQ1 plan's backbone requirement).
    pretrain_batches = [(g, b_idx) for g, b_idx, _ in train_batches]  # same minibatches, no labels needed
    loss_histories = {}
    trained_models = {}
    for name, factory in [("RAGCL", make_ragcl), ("GraphCL", make_graphcl),
                           ("GRACE", make_grace), ("ARAGCL-DP", make_aragcl_dp),
                           ("ARAGCL-DP (Ablation 2, no dup edges)", make_ablation2),
                           ("ARAGCL-DP (GAT backbone)",
                            lambda in_dim: make_aragcl_dp(in_dim, backbone="gat"))]:
        model = factory(in_dim=FEATURE_DIM)
        model, loss_history = contrastive_pretrain(name, model, pretrain_batches, cfg)
        loss_histories[name] = loss_history
        finetune(name, model, train_batches, val_batches, cfg)
        trained_models[name] = model  # keep the TRAINED model for reuse below,
                                       # rather than re-instantiating a fresh
                                       # (untrained, randomly-initialized) copy

    plot_convergence(loss_histories, save_path="contrastive_convergence.png")

    # --- Ablation 1: static graph construction (RQ1 plan Step 5) ----------------
    static_graphs = [build_static_graph(posts) for posts in raw_cascades]
    static_train_batches = make_minibatches(static_graphs[:n_train], labels[:n_train].tolist(), batch_size=8)
    static_pretrain_batches = [(g, b_idx) for g, b_idx, _ in static_train_batches]
    static_val_graph, static_val_batch, static_val_y = \
        batch_graphs(static_graphs[n_train:], labels[n_train:])
    ablation1_model = make_aragcl_dp(in_dim=FEATURE_DIM)  # same architecture
    ablation1_model, _ = contrastive_pretrain(
        "ARAGCL-DP (Ablation 1, static graph)", ablation1_model,
        static_pretrain_batches, cfg)
    finetune("ARAGCL-DP (Ablation 1, static graph)", ablation1_model,
             static_train_batches,
             [(static_val_graph, static_val_batch, static_val_y)], cfg)

    # --- RQ2: robustness sweep on ARAGCL-DP vs RAGCL ---------------------------
    aragcl_dp_model = trained_models["ARAGCL-DP"]  # reuse the TRAINED model,
                                                    # not a fresh untrained one
    from augmentation.views import ViewConfig, generate_view
    def robustness_forward(model, batch, p_n, eps):
        """Actually perturbs the graph at the given (p_n, eps) before
        classifying -- the earlier version of this closure set
        model.cfg.p_n/epsilon but then called model.classify() on the
        UNMODIFIED clean graph, which never reads those config fields.
        That made every sweep point identical regardless of p_n/eps
        (visible as flat Acc/F1 across the whole grid in a real run).
        This version builds the augmented view explicitly and
        classifies on THAT instead."""
        g, b_idx, y = batch
        vcfg = ViewConfig(strategy=model.cfg.view_strategy, p_n=p_n,
                           p_e=p_n, p_m=p_n, epsilon=eps)
        x_aug, edge_index_aug, _ = generate_view(
            g.x, g.reply_edge_index, vcfg,
            duplication_freq=g.duplication_freq,
            propagation_depth=g.propagation_depth,
            root_mask=g.root_mask, timestamps=g.timestamps,
            precomputed_centrality=g.precomputed_centrality,
        )
        g_aug = DuplicationGraph(
            x=x_aug, reply_edge_index=edge_index_aug,
            duplication_edge_index=g.duplication_edge_index,
            timestamps=g.timestamps, propagation_depth=g.propagation_depth,
            duplication_freq=g.duplication_freq, root_mask=g.root_mask,
        )
        logits = model.classify(g_aug, batch=b_idx)
        return logits, y
    print("\n=== RQ2 Robustness Sweep: ARAGCL-DP ===")
    robustness_sweep(aragcl_dp_model, robustness_forward, val_batches,
                      p_n_values=[0.0, 0.2, 0.4], epsilon_values=[0.0, 0.5])

    # --- Early/truncated cascade evaluation -------------------------------------
    def early_forward(model, batch, keep_ratio):
        g, b_idx, y = batch
        mask = truncate_cascade(g.reply_edge_index, g.propagation_depth,
                                 g.timestamps, keep_ratio=keep_ratio)
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

    # --- RQ3: representation quality --------------------------------------------
    with torch.no_grad():
        _, graph_embeddings = aragcl_dp_model.encoder(
            val_graph.x, val_graph.reply_edge_index,
            val_graph.duplication_edge_index, batch=val_batch_idx)
    print("\n=== RQ3 Representation Quality ===")
    report = compute_representation_quality(graph_embeddings, val_y)
    print(report)

    print("\nSmoke test complete -- if you see this, every module imports "
          "and runs end-to-end, including the 4x3 Lens 1 structural fidelity "
          "grid, both RQ1 ablations, the GAT backbone variant, the TF-IDF "
          "text baseline, and the stratified CV split check. Swap "
          "make_synthetic_cascade()/make_dataset() for your real "
          "Twitter15/16/Weibo/DRWeibo loader next.")


if __name__ == "__main__":
    main()
