# ARAGCL-DP — implementation scaffold

This is a working implementation of the framework described in
*"Propagation Trees Are Shallow: Duplication-based Augmented Graph
Contrastive Learning for Rumor Detection"* (Tomeng, 2026), built to
close the specific gaps identified between the proposal and what had
been implemented so far. Every file below is syntax-checked but has
**not been run against real data or torch** (this was built in a
sandbox without network access, so torch/torch_geometric couldn't be
installed) — run `run_all.py` in your own environment first, on the
included synthetic-data smoke test, before wiring in real datasets.

## File map (proposal section -> code)

| Proposal section / equation | File |
|---|---|
| §3.5.2–3.5.3, RQ1 graph construction (node duplication, `D_v` sets) | `data/duplication_graph.py` |
| Eq 3.2–3.4, augmentation probabilities from centrality | `augmentation/centrality.py`, `augmentation/views.py` |
| Eq 3.6, duplication-importance score | `augmentation/centrality.py::duplication_aware_centrality` |
| Eq 3.5, temporal jitter | `augmentation/views.py::generate_view` |
| Eq 3.7, duplication-aware encoder | `models/encoders.py::DuplicationAwareConv` |
| Eq 3.8, graph contrastive loss (NT-Xent) | `losses/contrastive.py` |
| Eq 3.1 / 3.9, joint loss | `losses/contrastive.py::joint_loss` |
| §3.6, baselines (GCN, R-GCN, BGCN) | `models/baselines.py` |
| §3.6, baselines (GACL, GraphCL, GRACE) | `models/aragcl_dp.py` (config variants) |
| §3.6.1, metrics (Accuracy/Precision/Recall/F1/Fβ) | `train/metrics.py` |
| §3.6, two-stage train/finetune protocol | `train/trainer.py` |
| RQ2, robustness sweep (p_n, ε) + early-cascade eval | `eval/robustness.py` |
| RQ3, representation quality / semantic alignment | `eval/representation_quality.py` |
| End-to-end wiring + synthetic-data smoke test | `run_all.py` |
| RQ1 plan §2/§4 Step 2, 4x3 Breadth x Depth grid (ReshareClassifier) | `data/reshare_classifier.py` |
| RQ1 plan §4 Step 2, Lens 1 structural fidelity (static vs. ARAGCL-DP graph construction) | `eval/structural_fidelity.py`, `data/duplication_graph.py::build_static_graph` |
| RQ1 plan §3, GCN **or** GAT backbone with duplication influence λ | `models/encoders.py::DuplicationAwareGATConv` |
| RQ1 plan §3, Text-only baseline (TF-IDF specifically) | `data/text_features.py` |
| RQ1 plan §3, 5-fold stratified CV across multiple seeds | `data/cross_validation.py` |
| RQ1 plan §4 Step 5, Ablation 1 (static representation) | `data/duplication_graph.py::build_static_graph` fed into `make_aragcl_dp()` |
| RQ1 plan §4 Step 5, Ablation 2 (remove duplication edges only) | `models/aragcl_dp.py::make_ablation2` |
| RQ1 plan §4 Step 6, contrastive loss convergence curves | `eval/convergence_plots.py` |
| **Real dataset loader** (Weibo-style per-cascade JSON, source+comment+centrality) | `data/weibo_json_loader.py` |

## Loading your real dataset

`data/weibo_json_loader.py` parses the exact format you're using — one
`<tweet_id>.json` file per cascade, with `source`/`comment`/`centrality`
keys — verified against a real sample file. **Before trusting any
results**, run the audit utility across your actual data folder (not
just the one sample):

```python
from data.weibo_json_loader import audit_dataset
audit_dataset("path/to/your/json/folder")
```

This checks three things across your whole dataset that were only
confirmed on ONE file so far:
1. **Label balance** — confirms labels are really binary {0,1} and
   checks the split isn't wildly imbalanced.
2. **Cascade depth** — the sample file had every comment replying
   directly to the source (`parent == -1` for all 80 comments, depth 1
   everywhere). The loader handles deeper trees, but confirm whether
   that's actually typical of your dataset or just this one cascade —
   it changes how meaningful the Depth axis of the 4x3 grid will be.
3. **Centrality array alignment** — flags any file where the
   precomputed centrality array length doesn't match
   `len(comments) + 1`, which would break the index-alignment
   assumption (`node 0 = source, node i+1 = comment i`).

Then load and featurize:

```python
from data.weibo_json_loader import load_dataset, attach_tfidf_features

train_raw = load_dataset("path/to/train_folder")
val_raw   = load_dataset("path/to/val_folder")

train_cascades = [posts for posts, label, cen in train_raw]
train_labels   = [label for posts, label, cen in train_raw]
val_cascades   = [posts for posts, label, cen in val_raw]
val_labels     = [label for posts, label, cen in val_raw]

# fits TF-IDF on train only, fills in .feature for every post in both sets
attach_tfidf_features(train_cascades, other_cascade_sets=[val_cascades])

# now build graphs exactly as in run_all.py:
from data.duplication_graph import build_duplication_graph
train_graphs = [build_duplication_graph(posts) for posts in train_cascades]
```

**Two assumptions baked into the loader that you must confirm, not
just trust:**
- `label: 1 = rumor, 0 = non-rumor` — this is the standard convention
  for this dataset family but isn't verifiable from one file alone
  (the sample only has label=1). If it's inverted, every accuracy/F1
  number downstream is silently flipped.
- Chinese text needs `analyzer="char", ngram_range=(1,2)` for TF-IDF
  (the default word-level tokenizer assumes whitespace-delimited
  text, which Chinese doesn't have). This is a reasonable
  dependency-free default, but if you have a real Chinese segmenter
  (e.g. `jieba`) available in your environment, word-level TF-IDF on
  segmented tokens will likely produce more meaningful features than
  raw character n-grams — worth trying both and comparing.

The dataset also ships **pre-computed centrality** (Degree/PageRank/
Eigenvector/Betweenness) per cascade. `augmentation/views.py` now has
a `"dataset"` strategy that reads directly from this instead of
recomputing centrality — use `models/aragcl_dp.py::make_dataset_centrality()`
as an additional RQ2 comparison point (does using the dataset's own
precomputed centrality change anything vs. recomputing degree/PageRank
yourself?).

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py          # synthetic-data smoke test
```

If this runs cleanly end-to-end, every module is wired correctly.
Then replace `make_synthetic_cascade()` / `make_dataset()` in
`run_all.py` with a real loader for Twitter15/16, Weibo or DRWeibo —
everything downstream expects the `RawPost` / `DuplicationGraph`
interfaces defined in `data/duplication_graph.py`, so nothing else
needs to change.

## The 4x3 grid — two things to know before you present it

1. **3 of the 12 cells are structurally impossible, by the plan's own
   definitions**, not a bug: `Extensive` requires depth ≤2, so
   `Extensive x Moderate` and `Extensive x Deep` can never be
   populated; `Cascading` requires depth ≥3, so `Cascading x Shallow`
   can never be populated. `grid_to_table()` marks these `N/A` rather
   than `0` so they don't read as a graph-construction failure. See
   the full note at the top of `data/reshare_classifier.py`.
2. **`N_r == 0` (organic replies, zero resharing) isn't defined by the
   plan's breadth categories** (`Single` is stated as exactly `N_r==1`).
   This is folded into `Single` (see the comment in
   `classify_breadth()`) rather than silently mislabeled as
   `Non-Extensive`. Confirm this matches your intent, or redefine
   `Single` as `N_r <= 1` explicitly in your methodology text.

## Known open items — please read before you run real experiments

1. **Real dataset loader is not included.** You'll need to write a
   function that reads your chosen dataset (Twitter15/16/Weibo/
   DRWeibo) and produces `RawPost` objects per cascade — mapping raw
   fields (post id, parent id, timestamp, text, and a pre-computed
   content embedding) into the dataclass in `data/duplication_graph.py`.

2. **Duplicate detection defaults to exact text match.** `_is_duplicate()`
   in `data/duplication_graph.py` only handles exact matches, per your
   own stated definition (§3.5). If your dataset needs near-duplicate
   detection (e.g. retweet-with-comment), implement the
   `near_dup_threshold` branch using cosine similarity on the same
   content embeddings you're already computing.

3. **Eq 3.2 formula discrepancy.** As transcribed in the proposal, the
   node-drop probability equation is written in a way that's inverted
   relative to the text describing it (see the long comment at the top
   of `augmentation/centrality.py`). The code implements the *stated
   principle* (higher centrality → lower drop probability), consistent
   with Eq 3.6. Double-check this against Cui & Jia (2024)'s original
   equations and correct the proposal text before your final write-up
   — a supervisor or examiner comparing your equations to the cited
   paper will catch this otherwise.

4. **GACL is approximated, not faithfully implemented.** The
   config-variant approach in `models/aragcl_dp.py` treats GraphCL,
   GRACE and RAGCL as different augmentation *strategies* over the same
   pipeline, which is accurate. GACL's defining feature is an
   *adversarial* (not random) perturbation, which is a genuinely
   different training loop (e.g. FGSM/PGD between the two forward
   passes) — not implemented here. If GACL needs to be a rigorous
   comparison point in your results, budget separate time for it.

5. **Duplication-importance combination formula is a design choice.**
   `duplication_aware_centrality()` combines duplication frequency and
   propagation depth via a weighted product — your proposal names both
   signals but doesn't specify how to combine them. State this choice
   explicitly in your methodology chapter, and consider the
   depth-only vs. duplication-only vs. combined ablation flagged
   earlier as a next step — the `depth_weight` / `duplication_weight`
   config fields make that ablation a one-line change.

6. **BiGCN's root-reinforcement step is simplified.** `models/baselines.py::BiGCN`
   adds the root feature at every layer as described in Bian et al.
   (2020), but uses a plain additive broadcast rather than their exact
   concatenation-based mechanism — close enough for a fair baseline
   comparison, but worth a footnote if you want to claim an exact
   reproduction.

7. **Multi-seed reporting.** `train/metrics.py::mean_std_table` is
   ready to use — wrap your training calls in a seed loop (3–5 seeds
   recommended) and pass the resulting `Metrics` lists in, so every
   number in your results chapter carries a variance estimate.
   `data/cross_validation.py::stratified_cv_splits` generates the
   (seed, fold) pairs to loop over — `run_all.py` currently only
   *prints* the split balance check as a sanity check and still trains
   on one fixed split for smoke-test speed; swap in a real loop over
   `stratified_cv_splits()` before reporting real numbers.

8. **GAT backbone is a genuinely different combination rule than the
   GCN one.** `DuplicationAwareGATConv` runs attention separately over
   reply-neighbors and duplication-neighbors, then combines with the
   same additive λ-weighted form as the GCN version (rather than a
   softmax-normalized attention combination), specifically so λ means
   the same thing across both backbones and the RQ2 λ-sensitivity
   ablation transfers cleanly. If you want GAT's attention to also
   weigh the reply-vs-duplication balance (not just λ), that's a
   further architectural choice to make and justify separately.

## Suggested order to get real results

1. Run `run_all.py` on synthetic data — confirms wiring.
2. Write the real dataset loader (open item 1).
3. Re-run `run_all.py` on one real dataset (start with the smallest,
   e.g. Twitter16) with `TrainConfig(epochs=..., pretrain_epochs=...)`
   turned up from the smoke-test's small values.
4. Once results look sane, wrap in a seed loop and produce the
   mean±std comparison table.
5. Run `eval/robustness.py::robustness_sweep` for the full p_n × ε
   grid (not just one aggregate point) and `early_detection_curve` for
   the still-pending truncated-cascade evaluation.
6. Run `eval/representation_quality.py` for the RQ3 metrics and
   `project_2d` for the t-SNE/UMAP figures.
