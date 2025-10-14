"""
Ensemble Model Module
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble-Modell für WLAN-Daten."""
    
    def __init__(self, algorithm: str = 'voting', **kwargs):
        """
        Initialisiert das Ensemble-Modell.
        
        Args:
            algorithm: Ensemble-Algorithmus
            **kwargs: Zusätzliche Parameter
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.estimators_ = None
        
    def _create_model(self, base_estimator=None, estimators=None) -> Any:
        """Erstellt das Ensemble-Modell."""
        if self.algorithm == 'voting':
            if estimators is None:
                raise ValueError("Voting requires estimators list")
            return VotingClassifier(estimators=estimators, **self.kwargs)
        elif self.algorithm == 'bagging':
            if base_estimator is None:
                raise ValueError("Bagging requires base_estimator")
            return BaggingClassifier(base_estimator=base_estimator, **self.kwargs)
        elif self.algorithm == 'boosting':
            if base_estimator is None:
                raise ValueError("Boosting requires base_estimator")
            return AdaBoostClassifier(base_estimator=base_estimator, **self.kwargs)
        else:
            raise ValueError(f"Unsupported ensemble algorithm: {self.algorithm}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleModel':
        """Trainiert das Ensemble-Modell."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) < 2:
            raise ValueError("Not enough samples for training")
        
        params = {**self.kwargs, **kwargs}
        
        if self.algorithm == 'voting':
            estimators = params.get('estimators', [])
            if not estimators:
                raise ValueError("At least one estimator required")
            self.model = self._create_model(estimators=estimators)
        else:
            base_estimator = params.get('base_estimator')
            if base_estimator is None:
                raise ValueError("base_estimator required for bagging/boosting")
            self.model = self._create_model(base_estimator=base_estimator)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        if hasattr(self.model, 'estimators_'):
            self.estimators_ = self.model.estimators_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluiert das Modell."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        return metrics
    
    def get_individual_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Gibt Performance einzelner Estimators zurück."""
        if not self.is_fitted or self.estimators_ is None:
            raise ValueError("Model must be fitted with estimators")
        
        scores = {}
        
        for i, estimator in enumerate(self.estimators_):
            estimator_name = f"estimator_{i}"
            if hasattr(estimator, 'predict'):
                predictions = estimator.predict(X)
                score = accuracy_score(y, predictions)
                scores[estimator_name] = score
        
        return scores