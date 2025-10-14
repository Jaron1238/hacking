"""
Classification Model Module
"""
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class ClassificationModel:
    """Klassifikations-Modell für WLAN-Daten."""
    
    def __init__(self, algorithm: str = 'random_forest', **kwargs):
        """
        Initialisiert das Klassifikations-Modell.
        
        Args:
            algorithm: Klassifikations-Algorithmus
            **kwargs: Zusätzliche Parameter
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_importances_ = None
        self.support_vectors_ = None
        
    def _create_model(self) -> Any:
        """Erstellt das Klassifikations-Modell."""
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(random_state=42, **self.kwargs)
        elif self.algorithm == 'svm':
            return SVC(random_state=42, **self.kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassificationModel':
        """Trainiert das Klassifikations-Modell."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) < 2:
            raise ValueError("Not enough samples for training")
        
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        if hasattr(self.model, 'support_vectors_'):
            self.support_vectors_ = self.model.support_vectors_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Macht Wahrscheinlichkeits-Vorhersagen."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluiert das Modell."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        """Führt Cross-Validation durch."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        test_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        train_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'test_score': test_scores.tolist(),
            'train_score': train_scores.tolist()
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Gibt Feature-Importance zurück."""
        return self.feature_importances_