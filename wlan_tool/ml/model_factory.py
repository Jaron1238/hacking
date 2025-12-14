# ML Model Factory Pattern
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import logging

logger = logging.getLogger(__name__)

class MLModelFactory:
    """Factory für ML-Modelle - erstellt Modelle basierend auf Konfiguration"""
    
    _models = {
        'device_classifier': RandomForestClassifier,
        'anomaly_detector': IsolationForest,
        'behavior_predictor': MLPRegressor,
        'one_class_svm': OneClassSVM,
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'spectral': SpectralClustering
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> Any:
        """Erstellt ML-Modell basierend auf Typ und Konfiguration"""
        if model_type not in cls._models:
            raise ValueError(f"Unbekannter Modell-Typ: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(**config.get('params', {}))
    
    @classmethod
    def load_model(cls, model_path: str) -> Any:
        """Lädt gespeichertes Modell (Lazy Loading)"""
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return None
    
    @classmethod
    def save_model(cls, model: Any, model_path: str) -> bool:
        """Speichert trainiertes Modell"""
        try:
            joblib.dump(model, model_path)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {e}")
            return False