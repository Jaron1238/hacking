"""
Clustering Model Module
"""
import numpy as np
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class ClusteringModel:
    """Clustering-Modell für WLAN-Daten."""
    
    def __init__(self, algorithm: str = 'kmeans', **kwargs):
        """
        Initialisiert das Clustering-Modell.
        
        Args:
            algorithm: Clustering-Algorithmus
            **kwargs: Zusätzliche Parameter
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.labels_ = None
        self.cluster_centers_ = None
        
    def _create_model(self, n_clusters: Optional[int] = None) -> Any:
        """Erstellt das Clustering-Modell."""
        if self.algorithm == 'kmeans':
            n_clusters = n_clusters or self.kwargs.get('n_clusters', 5)
            return KMeans(n_clusters=n_clusters, random_state=42, **self.kwargs)
        elif self.algorithm == 'dbscan':
            return DBSCAN(**self.kwargs)
        elif self.algorithm == 'hierarchical':
            n_clusters = n_clusters or self.kwargs.get('n_clusters', 5)
            return AgglomerativeClustering(n_clusters=n_clusters, **self.kwargs)
        elif self.algorithm == 'gmm':
            n_clusters = n_clusters or self.kwargs.get('n_clusters', 5)
            return GaussianMixture(n_components=n_clusters, random_state=42, **self.kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X: np.ndarray) -> 'ClusteringModel':
        """Trainiert das Clustering-Modell."""
        if len(X) < 2:
            raise ValueError("Not enough samples for clustering")
        
        self.model = self._create_model()
        
        if self.algorithm == 'gmm':
            self.model.fit(X)
            self.labels_ = self.model.predict(X)
        else:
            self.labels_ = self.model.fit_predict(X)
        
        self.is_fitted = True
        
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            self.cluster_centers_ = self.model.means_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.algorithm == 'gmm':
            return self.model.predict(X)
        else:
            return self.model.fit_predict(X)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Gibt Cluster-Zentren zurück."""
        return self.cluster_centers_
    
    def evaluate(self, X: np.ndarray) -> Dict[str, Any]:
        """Evaluiert das Modell."""
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted first")
        
        metrics = {}
        
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, self.labels_)
        else:
            metrics['silhouette_score'] = 0.0
        
        metrics['n_clusters'] = len(np.unique(self.labels_))
        
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = self.model.inertia_
        
        return metrics