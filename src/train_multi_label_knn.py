from pathlib import Path
from typing import Dict, List, Optional

import joblib
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from skmultilearn.model_selection import iterative_train_test_split

from custom_mlknn import FixedMLkNN
from features import create_label_matrix, vectorize_text
from metrics import precision_at_k, recall_at_k
from storage_manager import StorageManager


def train_ml_knn(movies_df: pd.DataFrame, genres_df: pd.DataFrame, \
    test_size: float = 0.2, random_state: int = 42, artifact_dir: Path = Path("artifacts"), \
        max_features: int = 20000, k: int = 10, s: float = 1.0, storage_manager: Optional[StorageManager] = None) -> Dict[str, float]:
    """    
    @args:
        test_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        artifact_dir: Directory to save model artifacts
        max_features: Maximum TF-IDF features
        k: Number of nearest neighbors for ML-kNN
        s: Smoothing parameter for ML-kNN (controls label prediction strength)
    @returns: Dictionary of evaluation metrics    
    """
    if not (0 < test_size < 1): raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if k < 1: raise ValueError(f"k must be at least 1, got {k}")
    if s <= 0: raise ValueError(f"s must be positive, got {s}")
    mlflow.set_experiment("MovieGenre-MLkNN")
    with mlflow.start_run(run_name = "MLkNN-Adaptation"):
        params: Dict[str, object] = { "test_size": test_size, "random_state": random_state, \
            "tfidf_max_features": max_features, "model": "MLkNN", "k_neighbors": k, "smoothing_s": s }
        mlflow.log_params(params)
        print(f"\n{'='*60}")
        print(f"Training ML-kNN (Algorithmic Adaptation)")
        print(f"{'='*60}")        
        y: npt.NDArray[np.int_]
        genre_names: List[str]
        y, genre_names = create_label_matrix(movies_df, genres_df)
        X_text: pd.Series = movies_df["overview"]
        X_train_text: npt.NDArray[np.object_]
        y_train: npt.NDArray[np.int_]
        X_val_text: npt.NDArray[np.object_]
        y_val: npt.NDArray[np.int_]
        X_train_text, y_train, X_val_text, y_val = \
            iterative_train_test_split(X_text.values.reshape(-1, 1), y, test_size = test_size)
        # Flatten text arrays
        X_train_text_flat: npt.NDArray[np.str_] = X_train_text.flatten()
        X_val_text_flat: npt.NDArray[np.str_] = X_val_text.flatten()
        print(f"  Training samples: {len(X_train_text_flat)}")
        print(f"  Validation samples: {len(X_val_text_flat)}")
        X_train_sparse: npt.NDArray[np.float64]
        X_val_sparse: npt.NDArray[np.float64]        
        X_train_sparse, X_val_sparse, vectorizer = vectorize_text(pd.Series(X_train_text_flat), \
            pd.Series(X_val_text_flat), max_features = max_features)
        # CRITICAL: Convert sparse to dense for ML-kNN        
        if issparse(X_train_sparse):
            X_train: npt.NDArray[np.float64] = X_train_sparse.toarray()
            X_val: npt.NDArray[np.float64] = X_val_sparse.toarray()
            print(f"  Converted from sparse to dense arrays")
        else:
            X_train = X_train_sparse
            X_val = X_val_sparse
        print(f"  Train matrix shape: {X_train.shape}")
        print(f"  Val matrix shape: {X_val.shape}")
        print(f"  Memory usage: ~{X_train.nbytes / 1024**2:.1f} MB (train)")        
        model = FixedMLkNN(k = k, s = s)
        model.fit(X_train, y_train)        
        y_val_pred_sparse = model.predict(X_val)
        y_val_pred_proba_sparse = model.predict_proba(X_val)
        # Convert sparse predictions to dense
        y_val_pred: npt.NDArray[np.int_] = y_val_pred_sparse.toarray()
        y_val_pred_proba: npt.NDArray[np.float64] = y_val_pred_proba_sparse.toarray()        
        metrics: Dict[str, float] = {
            "hamming_loss": float(hamming_loss(y_val, y_val_pred)),
            "subset_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "f1_micro": float(f1_score(y_val, y_val_pred, average = "micro")),
            "f1_macro": float(f1_score(y_val, y_val_pred, average = "macro")),
            "precision_at_3": float(precision_at_k(y_val, y_val_pred_proba, k = 3)),
            "recall_at_3": float(recall_at_k(y_val, y_val_pred_proba, k = 3)),
        }        
        mlflow.log_metrics(metrics)        
        print(f"\n💾 Saving model artifacts to {artifact_dir}...")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path: Path = artifact_dir / "mlknn_model.pkl"
        vectorizer_path: Path = artifact_dir / "tfidf_vectorizer_mlkn.pkl"
        genre_names_path: Path = artifact_dir / "genre_names_mlkn.txt"        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)        
        with open(genre_names_path, "w", encoding="utf-8") as f: f.write("\n".join(genre_names))
        print(f"  ✓ Model saved: {model_path}")
        print(f"  ✓ Vectorizer saved: {vectorizer_path}")
        print(f"  ✓ Genre names saved: {genre_names_path}")        
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(vectorizer_path))
        mlflow.log_artifact(str(genre_names_path))
        if storage_manager:
            print(f"\n☁️  Uploading artifacts to cloud storage...")
            storage_manager.save_artifact(model_path, "mlknn_model.pkl")
            storage_manager.save_artifact(vectorizer_path, "tfidf_vectorizer_mlkn.pkl")
            storage_manager.save_artifact(genre_names_path, "genre_names_mlkn.txt")
            print(f"✅ Artifacts uploaded successfully")
        print(f"\n{'='*60}")
        print(f"✅ ML-kNN Training Complete!")
        print(f"{'='*60}")
        print(f"\nPerformance Metrics:")
        print(f"  Hamming Loss:     {metrics['hamming_loss']:.4f} ↓")
        print(f"  Subset Accuracy:  {metrics['subset_accuracy']:.4f} ↑")
        print(f"  F1 (Micro):       {metrics['f1_micro']:.4f} ↑")
        print(f"  F1 (Macro):       {metrics['f1_macro']:.4f} ↑")
        print(f"  Precision@3:      {metrics['precision_at_3']:.4f} ↑")
        print(f"  Recall@3:         {metrics['recall_at_3']:.4f} ↑")
        print(f"\n💡 Tip: Lower Hamming Loss and higher other metrics are better")
        print(f"{'='*60}\n")
        return metrics