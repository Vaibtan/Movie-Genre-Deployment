import time
from pathlib import Path
from typing import List, Tuple, Union

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.adapt import MLkNN


class ModelLoader:    
    def __init__(self, model_path: Path, vectorizer_path: Path, genre_names_path: Path) -> None:
        """        
        Args:
            model_path: Path to serialized model (.pkl)
            vectorizer_path: Path to TF-IDF vectorizer (.pkl)
            genre_names_path: Path to genre names (.txt)            
        """
        if not model_path.exists(): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists(): raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        if not genre_names_path.exists(): raise FileNotFoundError(f"Genre names file not found: {genre_names_path}")
        # Load artifacts
        self.model: Union[MultiOutputClassifier, MLkNN] = joblib.load(model_path)
        self.vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)        
        with open(genre_names_path, "r", encoding="utf-8") as f:
            self.genre_names: List[str] = [line.strip() for line in f.readlines()]
        if not isinstance(self.model, (MultiOutputClassifier, MLkNN)):
            raise ValueError(
                f"Unsupported model type: {type(self.model).__name__}. "
                "Expected MultiOutputClassifier or MLkNN."
            )        
        # Pre-compute genre name array for fast indexing
        self._genre_array: npt.NDArray[np.str_] = np.array(self.genre_names)
    
    # Predict top-k genres for a movie overview
    def predict(self, overview: str, k: int = 3, min_confidence: float = 0.1) -> Tuple[List[str], float, npt.NDArray[np.float64]]:
        """
        Args:
            overview: Movie overview text
            k: Number of top genres to return
            min_confidence: Minimum confidence threshold for inclusion
        Returns:
            - List of predicted genre names (length <= k)
            - Inference time in seconds
            - Full confidence array for all genres            
        """
        if not overview or not overview.strip(): raise ValueError("Overview cannot be empty")
        start_time: float = time.perf_counter()
        X: npt.NDArray[np.float64] = self.vectorizer.transform([overview]).toarray()        
        # Get probabilities based on model type
        probas: npt.NDArray[np.float64]
        if isinstance(self.model, MultiOutputClassifier): probas = self.predict_binary_relevance(X)
        elif isinstance(self.model, MLkNN): probas = self.model.predict_proba(X).toarray()[0]
        else: raise ValueError(f"Unsupported model type: {type(self.model)}")
        # Get top-k indices sorted by confidence
        top_k_indices: npt.NDArray[np.intp] = np.argsort(-probas)[:k]
        # Filter by minimum confidence and get genre names
        top_genres: List[str] = [self.genre_names[idx] for idx in \
            top_k_indices if probas[idx] >= min_confidence]
        inference_time: float = time.perf_counter() - start_time        
        return top_genres, inference_time, probas
    
    def predict_binary_relevance(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Args: X: Feature matrix (n_samples, n_features)
        Returns: Probability array (n_genres,)
        """
        # Extract probability of positive class from each estimator
        probas_list: List[float] = []
        for estimator in self.model.estimators_:
            proba_matrix: npt.NDArray[np.float64] = estimator.predict_proba(X)
            if proba_matrix.shape[1] == 1: probas_list.append(proba_matrix[0, 0])
            else: probas_list.append(proba_matrix[0, 1])
        return np.array(probas_list, dtype = np.float64)
    
    # Predict genres for multiple overviews (batch prediction)
    def predict_batch(self, overviews: List[str], k: int = 3, \
        min_confidence: float = 0.1) -> List[Tuple[List[str], float, npt.NDArray[np.float64]]]:
        """
        Args:
            overviews: List of movie overview texts
            k: Number of top genres to return per overview
            min_confidence: Minimum confidence threshold     
        Returns:
            List of tuples, each containing:
            - List of predicted genre names
            - Inference time (for entire batch)
            - Confidence array for all genres
        """
        if not overviews: return []
        start_time: float = time.perf_counter()
        X: npt.NDArray[np.float64] = self.vectorizer.transform(overviews).toarray()
        # Get probabilities for all samples
        if isinstance(self.model, MultiOutputClassifier):
            all_probas: npt.NDArray[np.float64] = np.array([estimator.predict_proba(X)[:, 1] \
                for estimator in self.model.estimators_]).T
        elif isinstance(self.model, MLkNN): all_probas = self.model.predict_proba(X).toarray()
        else: raise ValueError(f"Unsupported model type: {type(self.model)}")
        # Process each sample
        results: List[Tuple[List[str], float, npt.NDArray[np.float64]]] = []
        batch_time: float = time.perf_counter() - start_time
        per_sample_time: float = batch_time / len(overviews)
        for probas in all_probas:
            top_k_indices: npt.NDArray[np.intp] = np.argsort(-probas)[:k]
            top_genres: List[str] = [self.genre_names[idx] for idx in top_k_indices \
                if probas[idx] >= min_confidence]
            results.append((top_genres, per_sample_time, probas))
        return results
    
    # Get confidence score for a specific genre
    def get_genre_confidence(self, overview: str, genre_name: str) -> float:
        if genre_name not in self.genre_names:
            raise ValueError(
                f"Genre '{genre_name}' not found in model. "
                f"Available genres: {', '.join(self.genre_names)}"
            )
        
        _, _, probas = self.predict(overview, k=len(self.genre_names))
        genre_idx: int = self.genre_names.index(genre_name)
        return float(probas[genre_idx])
    
    # Return number of genres the model can predict
    @property
    def n_genres(self) -> int: return len(self.genre_names)
    
    @property
    def model_type(self) -> str:
        if isinstance(self.model, MultiOutputClassifier): return "binary_relevance"
        elif isinstance(self.model, MLkNN): return "ml_knn"
        else: return "unknown"