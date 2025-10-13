from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
from data_loader import load_and_clean_data
from features import create_label_matrix, vectorize_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import iterative_train_test_split

from metrics import precision_at_k, recall_at_k


def train_binary_relevance(movies_df: pd.DataFrame, genres_df: pd.DataFrame, \
    test_size: float = 0.2, random_state: int = 42, artifact_dir: Path = Path("artifacts"), \
        max_features: int = 20000, C: float = 1.0) -> Dict[str, float]:
    """    
    Args:
        movies_df: Movie DataFrame with genre_ids
        genres_df: Genre mapping DataFrame
        test_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        artifact_dir: Directory to save model artifacts
        max_features: Maximum TF-IDF features
        C: Regularization parameter for LogisticRegression
        
    Returns: Dictionary of evaluation metrics
    """
    mlflow.set_experiment("MovieGenre-BinaryRelevance")
    with mlflow.start_run(run_name = "BR-LogisticRegression"):
        params: Dict[str, object] = {
            "test_size": test_size,
            "random_state": random_state,
            "tfidf_max_features": max_features,
            "model": "LogisticRegression-OvR",
            "C": C,
            "solver": "liblinear",
        }
        mlflow.log_params(params)
        y: npt.NDArray[np.int_]
        genre_names: List[str]
        y, genre_names = create_label_matrix(movies_df, genres_df)
        X_text: pd.Series = movies_df["overview"]
        # Multi-label stratified split
        X_train_text: npt.NDArray[np.object_]
        y_train: npt.NDArray[np.int_]
        X_val_text: npt.NDArray[np.object_]
        y_val: npt.NDArray[np.int_]
        X_train_text, y_train, X_val_text, y_val = \
            iterative_train_test_split(X_text.values.reshape(-1, 1), y, test_size = test_size)
        # Flatten text arrays
        X_train_text_flat: npt.NDArray[np.str_] = X_train_text.flatten()
        X_val_text_flat: npt.NDArray[np.str_] = X_val_text.flatten()
        # Vectorize text
        X_train: npt.NDArray[np.float64]
        X_val: npt.NDArray[np.float64]
        X_train, X_val, vectorizer = vectorize_text(pd.Series(X_train_text_flat), \
            pd.Series(X_val_text_flat), max_features = max_features)
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Number of genres: {len(genre_names)}")
        # Train Binary Relevance model
        base_clf = LogisticRegression(C = C, solver = "liblinear", \
            random_state = random_state, max_iter = 1000)
        model = MultiOutputClassifier(base_clf, n_jobs = -1)
        print("Training Binary Relevance model...")
        model.fit(X_train, y_train)
        print("Training complete!") 
        # Predict probabilities for metrics
        y_val_pred_proba: npt.NDArray[np.float64] = np.array([est.\
            predict_proba(X_val)[:, 1] for est in model.estimators_]).T
        # Binary predictions (threshold = 0.5)
        y_val_pred: npt.NDArray[np.int_] = (y_val_pred_proba >= 0.5).astype(np.int_)
        # Compute metrics
        metrics: Dict[str, float] = {
            "hamming_loss": float(hamming_loss(y_val, y_val_pred)),
            "subset_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "f1_micro": float(f1_score(y_val, y_val_pred, average = "micro")),
            "f1_macro": float(f1_score(y_val, y_val_pred, average = "macro")),
            "precision_at_3": float(precision_at_k(y_val, y_val_pred_proba, k = 3)),
            "recall_at_3": float(recall_at_k(y_val, y_val_pred_proba, k = 3)),
        }        
        mlflow.log_metrics(metrics)
        artifact_dir.mkdir(parents = True, exist_ok = True)
        model_path: Path = artifact_dir / "binary_relevance_model.pkl"
        vectorizer_path: Path = artifact_dir / "tfidf_vectorizer.pkl"
        genre_names_path: Path = artifact_dir / "genre_names.txt"
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        with open(genre_names_path, "w", encoding="utf-8") as f: f.write("\n".join(genre_names))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(vectorizer_path))
        mlflow.log_artifact(str(genre_names_path))
        
        print("\n✅ Binary Relevance training complete!")
        print("Metrics:")
        for metric_name, metric_value in metrics.items(): print(f"  {metric_name}: {metric_value:.4f}")
        return metrics