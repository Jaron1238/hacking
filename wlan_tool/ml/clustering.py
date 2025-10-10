#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Clustering Algorithms für das WLAN-Analyse-Tool.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    MeanShift, AffinityPropagation, OPTICS, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

from ..constants import Constants, ErrorCodes, get_error_message
from ..exceptions import ValidationError, ResourceError
from ..validation import validate_dataframe
from ..metrics import record_timing, record_counter

logger = logging.getLogger(__name__)


class BaseClusteringAlgorithm(ABC):
    """Basis-Klasse für alle Clustering-Algorithmen."""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.labels_ = None
        self.n_clusters_ = 0
        self.cluster_centers_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusteringAlgorithm':
        """Trainiere Clustering-Modell."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Trainiere und mache Vorhersagen."""
        return self.fit(X).predict(X)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Hole Cluster-Zentren."""
        return self.cluster_centers_


class KMeansClustering(BaseClusteringAlgorithm):
    """K-Means Clustering mit erweiterten Features."""
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', 
                 n_init: int = 10, max_iter: int = 300, random_state: int = 42):
        super().__init__("KMeans")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """Trainiere K-Means."""
        logger.info(f"Training K-Means with {self.n_clusters} clusters")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.cluster_centers_ = self.model.cluster_centers_
        
        logger.info(f"K-Means trained: {self.n_clusters_} clusters found")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_inertia(self) -> float:
        """Hole Inertia (Within-Cluster Sum of Squares)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.inertia_


class DBSCANClustering(BaseClusteringAlgorithm):
    """DBSCAN Clustering mit automatischer Parameter-Optimierung."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 metric: str = 'euclidean', algorithm: str = 'auto'):
        super().__init__("DBSCAN")
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm
        )
    
    def fit(self, X: np.ndarray) -> 'DBSCANClustering':
        """Trainiere DBSCAN."""
        logger.info(f"Training DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        
        # Anzahl Cluster (ohne Noise-Punkte)
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Noise-Punkte identifizieren
        self.n_noise_ = list(self.labels_).count(-1)
        
        logger.info(f"DBSCAN trained: {self.n_clusters_} clusters, {self.n_noise_} noise points")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.fit_predict(X_scaled)
    
    def get_core_samples(self) -> np.ndarray:
        """Hole Core-Samples."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.core_sample_indices_


class HierarchicalClustering(BaseClusteringAlgorithm):
    """Hierarchical Clustering mit verschiedenen Linkage-Methoden."""
    
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward', 
                 metric: str = 'euclidean', distance_threshold: Optional[float] = None):
        super().__init__("Hierarchical")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.distance_threshold = distance_threshold
        
        if distance_threshold is not None:
            self.model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage,
                metric=metric
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric
            )
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """Trainiere Hierarchical Clustering."""
        logger.info(f"Training Hierarchical Clustering with {self.linkage} linkage")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        
        logger.info(f"Hierarchical Clustering trained: {self.n_clusters_} clusters")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.fit_predict(X_scaled)


class SpectralClustering(BaseClusteringAlgorithm):
    """Spectral Clustering für nicht-lineare Cluster."""
    
    def __init__(self, n_clusters: int = 8, eigen_solver: str = 'arpack',
                 random_state: int = 42, n_init: int = 10, gamma: float = 1.0):
        super().__init__("Spectral")
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma
        )
    
    def fit(self, X: np.ndarray) -> 'SpectralClustering':
        """Trainiere Spectral Clustering."""
        logger.info(f"Training Spectral Clustering with {self.n_clusters} clusters")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        
        logger.info(f"Spectral Clustering trained: {self.n_clusters_} clusters")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.fit_predict(X_scaled)


class GaussianMixtureClustering(BaseClusteringAlgorithm):
    """Gaussian Mixture Model Clustering."""
    
    def __init__(self, n_components: int = 8, covariance_type: str = 'full',
                 random_state: int = 42, max_iter: int = 100):
        super().__init__("GaussianMixture")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter
        )
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureClustering':
        """Trainiere Gaussian Mixture Model."""
        logger.info(f"Training Gaussian Mixture with {self.n_components} components")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.predict(X_scaled)
        self.n_clusters_ = self.n_components
        self.cluster_centers_ = self.model.means_
        
        logger.info(f"Gaussian Mixture trained: {self.n_clusters_} components")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Cluster-Wahrscheinlichkeiten."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_aic(self) -> float:
        """Akaike Information Criterion."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.aic(X_scaled)
    
    def get_bic(self) -> float:
        """Bayesian Information Criterion."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.bic(X_scaled)


class MeanShiftClustering(BaseClusteringAlgorithm):
    """Mean Shift Clustering für automatische Cluster-Anzahl."""
    
    def __init__(self, bandwidth: Optional[float] = None, seeds: Optional[np.ndarray] = None,
                 bin_seeding: bool = False, min_bin_freq: int = 1, cluster_all: bool = True):
        super().__init__("MeanShift")
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.model = MeanShift(
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all
        )
    
    def fit(self, X: np.ndarray) -> 'MeanShiftClustering':
        """Trainiere Mean Shift."""
        logger.info("Training Mean Shift Clustering")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.cluster_centers_ = self.model.cluster_centers_
        
        logger.info(f"Mean Shift trained: {self.n_clusters_} clusters")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class AffinityPropagationClustering(BaseClusteringAlgorithm):
    """Affinity Propagation Clustering."""
    
    def __init__(self, damping: float = 0.5, max_iter: int = 200, 
                 convergence_iter: int = 15, preference: Optional[float] = None,
                 affinity: str = 'euclidean', random_state: int = 42):
        super().__init__("AffinityPropagation")
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.model = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            preference=preference,
            affinity=affinity,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'AffinityPropagationClustering':
        """Trainiere Affinity Propagation."""
        logger.info("Training Affinity Propagation Clustering")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.cluster_centers_ = self.model.cluster_centers_
        
        logger.info(f"Affinity Propagation trained: {self.n_clusters_} clusters")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class OPTICSClustering(BaseClusteringAlgorithm):
    """OPTICS Clustering für hierarchische Cluster."""
    
    def __init__(self, min_samples: int = 5, max_eps: float = np.inf,
                 metric: str = 'minkowski', p: int = 2, cluster_method: str = 'xi',
                 eps: Optional[float] = None, xi: float = 0.05, 
                 predecessor_correction: bool = True, min_cluster_size: Optional[int] = None):
        super().__init__("OPTICS")
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            p=p,
            cluster_method=cluster_method,
            eps=eps,
            xi=xi,
            predecessor_correction=predecessor_correction,
            min_cluster_size=min_cluster_size
        )
    
    def fit(self, X: np.ndarray) -> 'OPTICSClustering':
        """Trainiere OPTICS."""
        logger.info("Training OPTICS Clustering")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        
        logger.info(f"OPTICS trained: {self.n_clusters_} clusters")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Cluster-Labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.fit_predict(X_scaled)


class ClusteringEvaluator:
    """Evaluator für Clustering-Algorithmen."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluiere Clustering-Ergebnisse."""
        if len(set(labels)) < 2:
            return {'error': 'Not enough clusters for evaluation'}
        
        # Silhouette Score
        try:
            silhouette = silhouette_score(X, labels)
            self.metrics['silhouette_score'] = silhouette
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            self.metrics['silhouette_score'] = -1
        
        # Calinski-Harabasz Index
        try:
            calinski_harabasz = calinski_harabasz_score(X, labels)
            self.metrics['calinski_harabasz_score'] = calinski_harabasz
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
            self.metrics['calinski_harabasz_score'] = 0
        
        # Davies-Bouldin Index
        try:
            davies_bouldin = davies_bouldin_score(X, labels)
            self.metrics['davies_bouldin_score'] = davies_bouldin
        except Exception as e:
            logger.warning(f"Davies-Bouldin score calculation failed: {e}")
            self.metrics['davies_bouldin_score'] = float('inf')
        
        # Anzahl Cluster
        self.metrics['n_clusters'] = len(set(labels))
        
        # Noise-Punkte (für DBSCAN, OPTICS)
        self.metrics['n_noise'] = list(labels).count(-1)
        
        return self.metrics.copy()
    
    def compare_algorithms(self, X: np.ndarray, algorithms: Dict[str, BaseClusteringAlgorithm]) -> pd.DataFrame:
        """Vergleiche verschiedene Clustering-Algorithmen."""
        results = []
        
        for name, algorithm in algorithms.items():
            try:
                # Clustering durchführen
                labels = algorithm.fit_predict(X)
                
                # Evaluieren
                metrics = self.evaluate(X, labels)
                metrics['algorithm'] = name
                results.append(metrics)
                
                logger.info(f"Evaluated {name}: {metrics['n_clusters']} clusters, "
                          f"silhouette={metrics['silhouette_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results.append({
                    'algorithm': name,
                    'error': str(e),
                    'n_clusters': 0,
                    'silhouette_score': -1
                })
        
        return pd.DataFrame(results)


class ClusteringPipeline:
    """Pipeline für automatisches Clustering."""
    
    def __init__(self, algorithms: Optional[List[str]] = None):
        self.algorithms = algorithms or [
            'kmeans', 'dbscan', 'hierarchical', 'spectral', 
            'gaussian_mixture', 'mean_shift', 'affinity_propagation'
        ]
        self.evaluator = ClusteringEvaluator()
        self.results = {}
    
    def create_algorithm(self, algorithm_name: str, **kwargs) -> BaseClusteringAlgorithm:
        """Erstelle Clustering-Algorithmus."""
        if algorithm_name == 'kmeans':
            return KMeansClustering(**kwargs)
        elif algorithm_name == 'dbscan':
            return DBSCANClustering(**kwargs)
        elif algorithm_name == 'hierarchical':
            return HierarchicalClustering(**kwargs)
        elif algorithm_name == 'spectral':
            return SpectralClustering(**kwargs)
        elif algorithm_name == 'gaussian_mixture':
            return GaussianMixtureClustering(**kwargs)
        elif algorithm_name == 'mean_shift':
            return MeanShiftClustering(**kwargs)
        elif algorithm_name == 'affinity_propagation':
            return AffinityPropagationClustering(**kwargs)
        elif algorithm_name == 'optics':
            return OPTICSClustering(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def auto_cluster(self, X: np.ndarray, target_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Automatisches Clustering mit verschiedenen Algorithmen."""
        logger.info(f"Starting auto-clustering with {len(X)} samples")
        
        # Algorithmen-Parameter
        algorithm_configs = {
            'kmeans': {'n_clusters': target_clusters or 8},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'hierarchical': {'n_clusters': target_clusters or 8},
            'spectral': {'n_clusters': target_clusters or 8},
            'gaussian_mixture': {'n_components': target_clusters or 8},
            'mean_shift': {},
            'affinity_propagation': {}
        }
        
        # Nur verfügbare Algorithmen verwenden
        available_algorithms = {k: v for k, v in algorithm_configs.items() if k in self.algorithms}
        
        # Algorithmen trainieren und evaluieren
        trained_algorithms = {}
        for name, config in available_algorithms.items():
            try:
                algorithm = self.create_algorithm(name, **config)
                labels = algorithm.fit_predict(X)
                trained_algorithms[name] = algorithm
                
                # Metriken aufzeichnen
                record_timing(f"clustering_{name}_duration", 0)  # Placeholder
                record_counter(f"clustering_{name}_clusters", len(set(labels)))
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Algorithmen vergleichen
        comparison_df = self.evaluator.compare_algorithms(X, trained_algorithms)
        
        # Bestes Modell auswählen (basierend auf Silhouette Score)
        best_algorithm = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'algorithm']
        best_model = trained_algorithms[best_algorithm]
        
        # Ergebnisse
        self.results = {
            'best_algorithm': best_algorithm,
            'best_model': best_model,
            'comparison': comparison_df.to_dict('records'),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        logger.info(f"Auto-clustering completed: best algorithm is {best_algorithm}")
        return self.results
    
    def get_best_model(self) -> Optional[BaseClusteringAlgorithm]:
        """Hole bestes Clustering-Modell."""
        return self.results.get('best_model')
    
    def get_comparison_results(self) -> pd.DataFrame:
        """Hole Vergleichsergebnisse."""
        return pd.DataFrame(self.results.get('comparison', []))


# Convenience-Funktionen
def create_kmeans_clustering(n_clusters: int = 8) -> KMeansClustering:
    """Erstelle K-Means Clustering."""
    return KMeansClustering(n_clusters=n_clusters)


def create_dbscan_clustering(eps: float = 0.5, min_samples: int = 5) -> DBSCANClustering:
    """Erstelle DBSCAN Clustering."""
    return DBSCANClustering(eps=eps, min_samples=min_samples)


def create_hierarchical_clustering(n_clusters: int = 8) -> HierarchicalClustering:
    """Erstelle Hierarchical Clustering."""
    return HierarchicalClustering(n_clusters=n_clusters)


def create_spectral_clustering(n_clusters: int = 8) -> SpectralClustering:
    """Erstelle Spectral Clustering."""
    return SpectralClustering(n_clusters=n_clusters)


def create_gaussian_mixture_clustering(n_components: int = 8) -> GaussianMixtureClustering:
    """Erstelle Gaussian Mixture Clustering."""
    return GaussianMixtureClustering(n_components=n_components)


def create_auto_clustering_pipeline() -> ClusteringPipeline:
    """Erstelle automatische Clustering-Pipeline."""
    return ClusteringPipeline()