from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, issparse, lil_matrix
from sklearn.neighbors import NearestNeighbors
from skmultilearn.adapt import MLkNN


class FixedMLkNN(MLkNN):
    def __init__(self, k: int = 10, s: float = 1.0, ignore_first_neighbours: int = 0):
        """
        @args:
            k: Number of nearest neighbors to use
            s: Smoothing parameter for Bayesian inference
            ignore_first_neighbours: Number of first neighbors to ignore (usually 0)
        """
        super().__init__(k = k, s = s, ignore_first_neighbours = ignore_first_neighbours)
    
    # Compute prior probabilities P(H_l = 1) for each label
    def _compute_prior(self, y: npt.NDArray[np.int_]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        @args: y: Binary label matrix (n_samples, n_labels)
        @returns: Tuple of (prior_prob_true, prior_prob_false) arrays
        """
        # Sum labels for each class (handle both dense and sparse)
        label_sums = np.asarray(y.sum(axis = 0)).flatten()
        # Compute prior probabilities with Laplace smoothing
        prior_prob_true: npt.NDArray[np.float64] = ((self.s + label_sums) / (self.s * 2 + self._num_instances))
        prior_prob_false: npt.NDArray[np.float64] = 1 - prior_prob_true
        return prior_prob_true, prior_prob_false
    
    # Compute conditional probabilities using k-nearest neighbors
    def _compute_cond(self, X: npt.NDArray[np.float64], \
        y: npt.NDArray[np.int_]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """        
        @args:
            X: Feature matrix (n_samples, n_features)
            y: Binary label matrix (n_samples, n_labels)
        @returns: Tuple of (P(E_l^j | H_l = 1), P(E_l^j | H_l = 0)) arrays
        """
        self.knn_ = NearestNeighbors(n_neighbors = self.k).fit(X)        
        _, neighbors = self.knn_.kneighbors(X)        
        if self.ignore_first_neighbours > 0: neighbors = neighbors[:, self.ignore_first_neighbours:]
        # Count label occurrences in neighborhoods
        c = np.zeros((self._num_labels, self.k + 1), dtype = np.float64)
        cn = np.zeros((self._num_labels, self.k + 1), dtype = np.float64)
        for instance_idx in range(self._num_instances):
            # Get neighbors' labels - ensure proper array indexing
            neighbor_labels = y[neighbors[instance_idx]]
            for label_idx in range(self._num_labels):
                if len(neighbor_labels.shape) == 1: delta = int(neighbor_labels.sum())
                else: delta = int(neighbor_labels[:, label_idx].sum())
                # Update counts based on whether instance has label
                if y[instance_idx, label_idx] == 1: c[label_idx][delta] += 1
                else: cn[label_idx][delta] += 1
        # Compute conditional probabilities with smoothing
        c_sum = c.sum(axis = 1); cn_sum = cn.sum(axis = 1)
        cond_prob_true = np.zeros((self._num_labels, self.k + 1), dtype = np.float64)
        cond_prob_false = np.zeros((self._num_labels, self.k + 1), dtype = np.float64)
        for label_idx in range(self._num_labels):
            for neighbor_count in range(self.k + 1):
                # P(E_l^j | H_l = 1) with Laplace smoothing
                cond_prob_true[label_idx][neighbor_count] = ((self.s + \
                    c[label_idx][neighbor_count]) / (self.s * (self.k + 1) + c_sum[label_idx]))
                # P(E_l^j | H_l = 0) with Laplace smoothing
                cond_prob_false[label_idx][neighbor_count] = ((self.s + \
                    cn[label_idx][neighbor_count]) / (self.s * (self.k + 1) + cn_sum[label_idx]))
        return cond_prob_true, cond_prob_false
    
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> "FixedMLkNN":
        """        
        @args:
            X: Training feature matrix (n_samples, n_features)
            y: Training label matrix (n_samples, n_labels) 
        @returns: self: Fitted classifier
        """
        self._num_instances: int = X.shape[0]; self._num_labels: int = y.shape[1]
        # Convert to dense if sparse
        if issparse(y): self._label_cache = y.toarray()
        else: self._label_cache = y
        if issparse(X): X = X.toarray()
        # Compute prior probabilities
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self._label_cache)
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(X, self._label_cache)
        return self
    
    def predict(self, X: npt.NDArray[np.float64]) -> csr_matrix:
        """        
        @args: X: Test feature matrix (n_samples, n_features)
        @returns: Predicted binary label matrix (sparse)
        """
        if issparse(X): X = X.toarray()        
        _, neighbors = self.knn_.kneighbors(X)        
        if self.ignore_first_neighbours > 0: neighbors = neighbors[:, self.ignore_first_neighbours:]        
        n_test_instances: int = X.shape[0]
        predictions = lil_matrix((n_test_instances, self._num_labels), dtype=int)        
        for instance_idx in range(n_test_instances):
            neighbor_labels = self._label_cache[neighbors[instance_idx]]
            # For each label, compute posterior probability
            for label_idx in range(self._num_labels):
                # Count how many neighbors have this label
                if len(neighbor_labels.shape) == 1: delta = int(neighbor_labels.sum())
                else: delta = int(neighbor_labels[:, label_idx].sum())                
                delta = min(delta, self.k)
                # Compute P(H_l = 1 | E_l^delta) using Bayes
                p_true = (self._prior_prob_true[label_idx] * self._cond_prob_true[label_idx][delta])
                p_false = (self._prior_prob_false[label_idx] * self._cond_prob_false[label_idx][delta])
                # Predict label if P(H_l = 1 | E) > P(H_l = 0 | E)
                if p_true > p_false: predictions[instance_idx, label_idx] = 1
        return predictions.tocsr()
    
    
    # Predict label probabilities for test instances
    def predict_proba(self, X: npt.NDArray[np.float64]) -> csr_matrix:
        """
        @args: X: Test feature matrix (n_samples, n_features)   
        @returns: Probability matrix (sparse)
        """
        if issparse(X): X = X.toarray()        
        _, neighbors = self.knn_.kneighbors(X)        
        if self.ignore_first_neighbours > 0: neighbors = neighbors[:, self.ignore_first_neighbours:]        
        n_test_instances: int = X.shape[0]
        probabilities = lil_matrix((n_test_instances, self._num_labels), dtype = float)        
        for instance_idx in range(n_test_instances):
            neighbor_labels = self._label_cache[neighbors[instance_idx]]
            # For each label, compute posterior probability
            for label_idx in range(self._num_labels):
                # Count how many neighbors have this label
                if len(neighbor_labels.shape) == 1: delta = int(neighbor_labels.sum())
                else: delta = int(neighbor_labels[:, label_idx].sum())
                delta = min(delta, self.k)
                # Compute P(H_l = 1 | E_l^delta) using Bayes
                p_true = (self._prior_prob_true[label_idx] * self._cond_prob_true[label_idx][delta])
                p_false = (self._prior_prob_false[label_idx] * self._cond_prob_false[label_idx][delta])
                prob = p_true / (p_true + p_false)
                probabilities[instance_idx, label_idx] = prob
        return probabilities.tocsr()