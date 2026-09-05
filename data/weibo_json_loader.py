"""
Loader for the Weibo-style per-cascade JSON format you're actually
using -- one file per cascade, named `<tweet_id>.json`, structured as:

{
  "source": {
    "content": <str>, "time": "YY-M-D H:MM", "user id": <str>,
    "tweet id": <str>, "label": 0 or 1, "theme": <str>
  },
  "comment": [
    {"comment id": <int>, "parent": <int, -1 = replies to source>,
     "children": [<comment id>, ...], "user id": <str>,
     "user name": <str>, "content": <str>, "time": "YY-M-D H:MM"},
    ...
  ],
  "centrality": {
    "Degree": [<float>, ...], "Pagerank": [...],
    "Eigenvector": [...], "Betweenness": [...]
  }
}

Verified against the uploaded sample (A0bSW2Rem.json):
  - `comment id` == list index (0..N-1), so node index i+1 in the
    graph corresponds to comment id i; node index 0 is the source.
  - `centrality` arrays have length N_comments + 1, in that same
    [source, comment_0, comment_1, ...] order -- confirmed by length
    matching len(comments)+1 in the sample (81 == 80 + 1).
  - `parent == -1` means "replies directly to the source post". In
    the sample, ALL 80 comments have parent==-1 (a genuinely shallow,
    1-level-deep cascade -- consistent with your proposal's "Propagation
    Trees Are Shallow" framing). The parser below still handles
    parent >= 0 (replying to another comment, not the source) in case
    other cascades in your dataset are deeper -- confirm this by
    running `python -c "from data.weibo_json_loader import audit_dataset;
    audit_dataset('path/to/your/json/folder')"` (see bottom of this
    file) before assuming every cascade is 1-level.
  - `label` on `source`: 1 in the sample. ASSUMED 1=rumor, 0=non-rumor
    (standard convention for this dataset family) -- confirm this
    against your dataset's documentation before trusting downstream
    accuracy numbers; if it's inverted, every metric in your results
    chapter is silently flipped.

Node features are NOT filled in here -- `RawPost.feature` is left as
a zero-length placeholder per post, then a corpus-level TF-IDF
vectorizer (fit on TRAIN posts only) is used to fill in every post's
`.feature` in one pass via `attach_tfidf_features()` below, so the
whole dataset shares one consistent feature space.
"""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json
import torch

from .duplication_graph import RawPost
from .text_features import fit_tfidf, transform_tfidf


def _parse_time(t: str) -> datetime:
    # "13-7-15 22:41" -> 2-digit year, no zero-padding on month/day/hour
    return datetime.strptime(t, "%y-%m-%d %H:%M")


def load_cascade_json(path: str) -> Tuple[List[RawPost], int, Dict[str, torch.Tensor]]:
    """
    Returns (posts, label, centrality).
      posts     -- RawPost list, .feature is a placeholder zero-vector
                   (dim 1) until attach_tfidf_features() fills it in.
      label     -- int, 1 = rumor / 0 = non-rumor (see module docstring
                   caveat -- confirm this against your dataset docs).
      centrality-- {"Degree": [N] tensor, "Pagerank": [N] tensor, ...},
                   aligned with node index (0 = source, i+1 = comment i).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    src = data["source"]
    comments = sorted(data["comment"], key=lambda c: c["comment id"])
    cascade_id = hash(src["tweet id"]) & 0xFFFFFFFF  # stable int id from the string tweet id

    src_time = _parse_time(src["time"])

    posts = [RawPost(
        post_id=0, parent_id=None, cascade_id=cascade_id,
        timestamp=0.0,  # root defines t=0; everything else is relative seconds
        text=src["content"], feature=torch.zeros(1),
    )]

    # comment id -> post_id mapping (post_id 0 is reserved for source,
    # so comment id i becomes post_id i+1, matching the centrality
    # array's [source, comment_0, comment_1, ...] ordering).
    comment_post_id = {c["comment id"]: c["comment id"] + 1 for c in comments}

    for c in comments:
        parent_comment_id = c["parent"]
        parent_post_id = 0 if parent_comment_id == -1 else comment_post_id.get(parent_comment_id, 0)
        rel_seconds = (_parse_time(c["time"]) - src_time).total_seconds()
        posts.append(RawPost(
            post_id=comment_post_id[c["comment id"]],
            parent_id=parent_post_id,
            cascade_id=cascade_id,
            timestamp=rel_seconds,
            text=c["content"],
            feature=torch.zeros(1),
        ))

    centrality = {
        metric: torch.tensor(values, dtype=torch.float)
        for metric, values in data.get("centrality", {}).items()
    }

    return posts, int(src["label"]), centrality


def load_dataset(directory: str, pattern: str = "*.json"
                  ) -> List[Tuple[List[RawPost], int, Dict[str, torch.Tensor]]]:
    """Loads every cascade file in a directory. Point this at the
    folder containing all your per-cascade .json files (the one
    A0bSW2Rem.json you shared is one example of many such files)."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return [load_cascade_json(str(p)) for p in paths]


def attach_tfidf_features(train_cascades: List[List[RawPost]],
                           other_cascade_sets: List[List[List[RawPost]]] = (),
                           max_features: int = 512, analyzer: str = "char",
                           ngram_range: Tuple[int, int] = (1, 2)):
    """
    Fits ONE TF-IDF vectorizer on every post's text (source + all
    comments) across `train_cascades` only, then transforms and
    assigns `.feature` in place for train_cascades and every cascade
    list in other_cascade_sets (val/test) using that same fitted
    vectorizer -- fit-on-train-only avoids vocabulary leakage.

    analyzer='char', ngram_range=(1,2) by default: this dataset's text
    is Chinese, which has no whitespace word boundaries, so the
    default word-level TF-IDF tokenizer (built for space-delimited
    languages) would badly under-segment it. Character bigrams are a
    reasonable dependency-free default; if you have a proper Chinese
    segmenter available (e.g. jieba) in your real environment, prefer
    word-level TF-IDF over segmented tokens instead -- it will
    generally produce more semantically meaningful features than raw
    character n-grams.
    """
    def _texts(cascades):
        return [p.text for posts in cascades for p in posts]

    vectorizer = fit_tfidf(_texts(train_cascades), max_features=max_features)
    # fit_tfidf() in text_features.py doesn't currently expose analyzer/
    # ngram_range -- re-fit directly here with the CJK-appropriate
    # settings instead of using the default word-level vectorizer.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True,
                                  analyzer=analyzer, ngram_range=ngram_range)
    vectorizer.fit(_texts(train_cascades))

    def _assign(cascades):
        texts = _texts(cascades)
        feats = transform_tfidf(vectorizer, texts)
        i = 0
        for posts in cascades:
            for p in posts:
                p.feature = feats[i]
                i += 1

    _assign(train_cascades)
    for cascade_set in other_cascade_sets:
        _assign(cascade_set)

    return vectorizer


def audit_dataset(directory: str, pattern: str = "*.json", max_files: int = 200):
    """
    Sanity-check utility -- run this BEFORE trusting any results.
    Reports, across up to `max_files` cascades:
      - label distribution (confirms 0/1 balance + that label really
        is binary, per the "confirm the label convention" caveat above)
      - how many cascades have parent values other than -1 (confirms
        whether the "all shallow, depth==1" pattern in the one sample
        file holds across the dataset, or whether deeper trees exist
        that the loader needs to handle)
      - centrality array length mismatches (flags any file where
        len(centrality) != len(comments) + 1, which would mean the
        index-alignment assumption above doesn't hold for that file)
    """
    import collections
    paths = sorted(Path(directory).glob(pattern))[:max_files]
    label_counts = collections.Counter()
    non_root_parent_files = 0
    centrality_mismatch_files = 0

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        label_counts[data["source"]["label"]] += 1
        if any(c["parent"] != -1 for c in data["comment"]):
            non_root_parent_files += 1
        cen = data.get("centrality", {})
        if cen:
            first_metric_len = len(next(iter(cen.values())))
            if first_metric_len != len(data["comment"]) + 1:
                centrality_mismatch_files += 1

    print(f"Audited {len(paths)} files from {directory}")
    print(f"Label distribution: {dict(label_counts)}")
    print(f"Files with non-root parent (depth > 1 somewhere): {non_root_parent_files}")
    print(f"Files with centrality-length mismatch: {centrality_mismatch_files}")
