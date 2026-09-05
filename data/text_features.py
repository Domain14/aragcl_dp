"""
TF-IDF text features for the Text-only baseline (RQ1 plan, Section 3:
"Text-only Baseline (TF-IDF): Classifies rumor cascades using strictly
content features while completely ignoring the network topology").

Your existing TextOnlyBaseline (models/baselines.py) already accepts
any pre-computed feature vector, so this module only needs to produce
TF-IDF vectors as that input -- no change needed to the model itself.
"""
from typing import List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(train_texts: List[str], max_features: int = 512) -> TfidfVectorizer:
    """Fit on TRAINING root-post texts only, to avoid leaking val/test
    vocabulary statistics into the feature space."""
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    vectorizer.fit(train_texts)
    return vectorizer


def transform_tfidf(vectorizer: TfidfVectorizer, texts: List[str]) -> torch.Tensor:
    matrix = vectorizer.transform(texts)
    return torch.tensor(matrix.toarray(), dtype=torch.float)


def root_texts_from_cascades(cascades) -> List[str]:
    """Extracts each cascade's root-post text -- Text-only baseline
    classifies from the source claim, not the full reply thread."""
    texts = []
    for posts in cascades:
        root = next(p for p in posts if p.parent_id is None)
        texts.append(root.text)
    return texts
