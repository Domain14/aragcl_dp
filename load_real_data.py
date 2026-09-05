"""
Loads your real dataset (Weibo-style JSON cascades) instead of the
synthetic data run_all.py uses.

Place this file in the PROJECT ROOT, next to run_all.py (same folder
level) -- the imports below (`from data.weibo_json_loader import ...`)
resolve relative to that location, same as run_all.py's imports do.

Run with:
    python load_real_data.py
"""
from data.weibo_json_loader import audit_dataset, load_dataset, attach_tfidf_features
from data.duplication_graph import build_duplication_graph

DATASET_DIR = "datasets/weibo"  # <-- change this to wherever your .json files actually are

# 1. sanity-check the whole folder first (label balance, depth, centrality alignment)
audit_dataset(DATASET_DIR)

# 2. load every cascade in the folder
raw = load_dataset(DATASET_DIR)
cascades = [posts for posts, label, cen in raw]
labels = [label for posts, label, cen in raw]
print(f"\nLoaded {len(cascades)} cascades")

# 3. split (simple slice for a quick test; use data/cross_validation.py's
#    stratified_cv_splits for the real, multi-seed experiment)
n_train = int(0.8 * len(cascades))
train_cascades, val_cascades = cascades[:n_train], cascades[n_train:]
train_labels, val_labels = labels[:n_train], labels[n_train:]
print(f"Train: {len(train_cascades)}  Val: {len(val_cascades)}")

# 4. featurize (fits TF-IDF on train text only, fills .feature on both sets)
attach_tfidf_features(train_cascades, other_cascade_sets=[val_cascades])

# 5. build graphs, same construction run_all.py uses on synthetic data
train_graphs = [build_duplication_graph(p) for p in train_cascades]
val_graphs = [build_duplication_graph(p) for p in val_cascades]

print(f"\nBuilt {len(train_graphs)} train graphs and {len(val_graphs)} val graphs.")
print("Next step: feed these into the same batch_graphs()/training loop "
      "run_all.py uses -- copy that wiring (Majority/GCN/RAGCL/ARAGCL-DP "
      "training calls) into this script, replacing its synthetic "
      "graphs/labels with train_graphs/train_labels and "
      "val_graphs/val_labels from above.")
