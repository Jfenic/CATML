from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def is_text_column(series: pd.Series, sample_size: int = 500) -> bool:
    """
    Heuristic to determine if a pandas Series contains freeform natural language text
    rather than categorical tokens or identifiers.
    """
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return False

    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return False

    if len(clean) > sample_size:
        clean = clean.sample(sample_size, random_state=42)

    # Calculate average string length and word count
    lengths = clean.str.len()
    avg_len = float(lengths.mean())
    avg_words = float(clean.str.split().str.len().mean())
    unique_ratio = len(clean.unique()) / len(clean)

    # Freeform text typically has longer text, multiple words, and high diversity
    if avg_len >= 30 and avg_words >= 3.0:
        return True
    if avg_words >= 4.0:
        return True
    if avg_len >= 50 and unique_ratio >= 0.3:
        return True

    return False


class LightweightTextExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that extracts n-gram TF-IDF representations
    from single or multiple text columns without heavy deep learning dependencies.
    """

    def __init__(
        self,
        max_features: int = 50,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        column_prefix: str = "text",
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.column_prefix = column_prefix
        self.vectorizer_: TfidfVectorizer | None = None
        self.feature_names_out_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> LightweightTextExtractor:
        texts = self._to_text_series(X)
        effective_min_df = 1 if len(texts) < 10 else self.min_df

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=effective_min_df,
            sublinear_tf=True,
        )
        self.vectorizer_.fit(texts)

        words = self.vectorizer_.get_feature_names_out()
        self.feature_names_out_ = [f"{self.column_prefix}_tfidf_{w.replace(' ', '_')}" for w in words]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.vectorizer_ is None:
            raise RuntimeError("LightweightTextExtractor is not fitted.")
        texts = self._to_text_series(X)
        sparse_mat = self.vectorizer_.transform(texts)
        return sparse_mat.toarray().astype(np.float32)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def _to_text_series(self, X: Any) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            # Combine all columns into single whitespace-separated string per row
            return X.fillna("").astype(str).agg(" ".join, axis=1)
        elif isinstance(X, pd.Series):
            return X.fillna("").astype(str)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return pd.Series(X).fillna("").astype(str)
            else:
                df = pd.DataFrame(X)
                return df.fillna("").astype(str).agg(" ".join, axis=1)
        elif isinstance(X, (list, tuple)):
            return pd.Series(X).fillna("").astype(str)
        else:
            return pd.Series(list(X)).fillna("").astype(str)
