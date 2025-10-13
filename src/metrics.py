from typing import Set

import numpy as np
import numpy.typing as npt


# Compute Precision@k for multi-label classification.
def precision_at_k(y_true: npt.NDArray[np.int_], y_pred_proba: npt.NDArray[np.float64], k: int = 3) -> float:
    """
    Precision@k measures the proportion of predicted top-k labels
    that are actually relevant.
    Args:
        y_true: True binary label matrix (n_samples, n_labels)
        y_pred_proba: Predicted probability matrix (n_samples, n_labels)
        k: Number of top predictions to consider
    Returns: Mean Precision@k across all samples
    """
    if y_true.shape != y_pred_proba.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred_proba {y_pred_proba.shape}")
    if k <= 0: raise ValueError(f"k must be positive, got {k}")
    n_samples, n_labels = y_true.shape[0], y_true.shape[1]
    k_effective: int = min(k, n_labels)
    # Get top-k predictions for each sample
    top_k_indices: npt.NDArray[np.intp] = np.argsort(-y_pred_proba, axis = 1)[:, :k_effective]
    precision_scores: npt.NDArray[np.float64] = np.zeros(n_samples, dtype = np.float64)
    for i in range(n_samples):
        true_labels: Set[int] = set(np.where(y_true[i] == 1)[0].tolist())
        pred_labels: Set[int] = set(top_k_indices[i].tolist())        
        # Calculate precision
        if len(pred_labels) > 0: precision_scores[i] = len(true_labels & pred_labels) / len(pred_labels)
        else: precision_scores[i] = 0.0
    return float(np.mean(precision_scores))

# Compute Recall@k for multi-label classification
def recall_at_k(y_true: npt.NDArray[np.int_], y_pred_proba: npt.NDArray[np.float64], k: int = 3) -> float:
    """
    Recall@k measures the proportion of relevant labels
    that are in the top-k predictions.
    Args:
        y_true: True binary label matrix (n_samples, n_labels)
        y_pred_proba: Predicted probability matrix (n_samples, n_labels)
        k: Number of top predictions to consider
    Returns: Mean Recall@k across all samples        
    """
    if y_true.shape != y_pred_proba.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred_proba {y_pred_proba.shape}")
    if k <= 0: raise ValueError(f"k must be positive, got {k}")
    n_samples, n_labels = y_true.shape[0], y_true.shape[1]
    k_effective: int = min(k, n_labels)    
    # Get top-k predictions for each sample
    top_k_indices: npt.NDArray[np.intp] = np.argsort(-y_pred_proba, axis=1)[:, :k_effective]
    recall_scores: npt.NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        true_labels: Set[int] = set(np.where(y_true[i] == 1)[0].tolist())        
        pred_labels: Set[int] = set(top_k_indices[i].tolist())        
        if len(true_labels) > 0: recall_scores[i] = len(true_labels & pred_labels) / len(true_labels)
        else: recall_scores[i] = 1.0
    return float(np.mean(recall_scores))

# Compute Normalized Discounted Cumulative Gain (NDCG@k)
def ndcg_at_k(y_true: npt.NDArray[np.int_], y_pred_proba: npt.NDArray[np.float64], k: int = 3) -> float:
    """
    NDCG@k measures ranking quality with position-based discounting.
    Args:
        y_true: True binary label matrix (n_samples, n_labels)
        y_pred_proba: Predicted probability matrix (n_samples, n_labels)
        k: Number of top predictions to consider
    Returns: Mean NDCG@k across all samples
    """
    if y_true.shape != y_pred_proba.shape: raise ValueError("Shape mismatch between y_true and y_pred_proba")
    n_samples, k_effective = y_true.shape[0], min(k, y_true.shape[1])
    ndcg_scores: npt.NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        # Get top-k predictions
        top_k_indices: npt.NDArray[np.intp] = np.argsort(-y_pred_proba[i])[:k_effective]
        # DCG: sum of (relevance / log2(position + 1))
        dcg: float = 0.0
        for pos, idx in enumerate(top_k_indices):
            relevance: int = int(y_true[i, idx])
            dcg += relevance / np.log2(pos + 2)
        # IDCG: DCG of perfect ranking
        n_true: int = int(y_true[i].sum())
        if n_true == 0: ndcg_scores[i] = 1.0  # No true labels → perfect score
        else:
            idcg: float = sum(1.0 / np.log2(pos + 2) for pos in range(min(n_true, k_effective)))
            ndcg_scores[i] = dcg / idcg if idcg > 0 else 0.0
    return float(np.mean(ndcg_scores))