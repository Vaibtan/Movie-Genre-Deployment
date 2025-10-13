from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


# Create binary label matrix from genre IDs
def create_label_matrix(movies_df: pd.DataFrame, genres_df: pd.DataFrame,) -> Tuple[npt.NDArray[np.int_], List[str]]:
    if "genre_ids" not in movies_df.columns: raise ValueError("movies_df must have 'genre_ids' column")
    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y: npt.NDArray[np.int_] = mlb.fit_transform(movies_df["genre_ids"])
    # Map genre IDs to names
    genre_id_to_name: pd.Series = genres_df.set_index("id")["name"]
    genre_names: List[str] = [genre_id_to_name.loc[gid] for gid in mlb.classes_]
    print(f"Created label matrix: {y.shape[0]} samples × {y.shape[1]} genres")
    print(f"Label density: {y.mean():.3f} (avg labels per movie: {y.sum(axis=1).mean():.2f})")
    return y, genre_names

# Fit TF-IDF vectorizer on training data and transform both sets
def vectorize_text(train_texts: pd.Series, val_texts: pd.Series, max_features: int = 20000, \
    ngram_range: Tuple[int, int] = (1, 2),) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], TfidfVectorizer]:
    if len(train_texts) == 0: raise ValueError("Training texts cannot be empty")
    if len(val_texts) == 0: raise ValueError("Validation texts cannot be empty")
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range, \
        stop_words = "english", lowercase = True, strip_accents = "unicode", \
            min_df = 2, max_df = 0.95, sublinear_tf = True)
    print(f"Fitting TF-IDF vectorizer with max_features={max_features}...")
    X_train: npt.NDArray[np.float64] = vectorizer.fit_transform(train_texts).toarray()
    X_val: npt.NDArray[np.float64] = vectorizer.transform(val_texts).toarray()
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")
    print(f"Train matrix: {X_train.shape}, Val matrix: {X_val.shape}")
    return X_train, X_val, vectorizer