"""
Clustering Analysis Module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Clustering-Analyzer für WLAN-Daten."""
    
    def __init__(self, algorithm: str = 'kmeans', **kwargs):
        """
        Initialisiert den Clustering-Analyzer.
        
        Args:
            algorithm: Clustering-Algorithmus ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            **kwargs: Zusätzliche Parameter für den Algorithmus
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.labels_ = None
        self.cluster_centers_ = None
        
    def _create_model(self, n_clusters: Optional[int] = None) -> Any:
        """Erstellt das Clustering-Modell basierend auf dem Algorithmus."""
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
    
    def cluster_data(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Führt Clustering auf den Daten durch.
        
        Args:
            data: Feature-Matrix (n_samples, n_features)
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Cluster-Labels
        """
        if len(data) < 2:
            raise ValueError("Not enough samples for clustering")
        
        # Parameter zusammenführen
        params = {**self.kwargs, **kwargs}
        
        # Modell erstellen
        self.model = self._create_model(params.get('n_clusters'))
        
        # Clustering durchführen
        if self.algorithm == 'gmm':
            self.model.fit(data)
            self.labels_ = self.model.predict(data)
        else:
            self.labels_ = self.model.fit_predict(data)
        
        self.is_fitted = True
        
        # Cluster-Zentren extrahieren (falls verfügbar)
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            self.cluster_centers_ = self.model.means_
        
        return self.labels_
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Gibt die Cluster-Zentren zurück."""
        return self.cluster_centers_
    
    def evaluate(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Evaluiert die Clustering-Qualität.
        
        Args:
            data: Feature-Matrix
            
        Returns:
            Evaluations-Metriken
        """
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted first")
        
        metrics = {}
        
        # Silhouette Score
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(data, self.labels_)
        else:
            metrics['silhouette_score'] = 0.0
        
        # Anzahl Cluster
        metrics['n_clusters'] = len(np.unique(self.labels_))
        
        # Inertia (nur für K-Means)
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = self.model.inertia_
        
        return metrics