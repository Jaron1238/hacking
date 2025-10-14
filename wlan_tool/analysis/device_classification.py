"""
Device Classification Module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class DeviceClassifier:
    """Geräte-Klassifikator für WLAN-Daten."""
    
    def __init__(self, algorithm: str = 'random_forest', **kwargs):
        """
        Initialisiert den Geräte-Klassifikator.
        
        Args:
            algorithm: Klassifikations-Algorithmus ('random_forest', 'svm')
            **kwargs: Zusätzliche Parameter für den Algorithmus
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_importances_ = None
        self.support_vectors_ = None
        
    def _create_model(self) -> Any:
        """Erstellt das Klassifikations-Modell basierend auf dem Algorithmus."""
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(random_state=42, **self.kwargs)
        elif self.algorithm == 'svm':
            return SVC(random_state=42, **self.kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'DeviceClassifier':
        """
        Trainiert das Klassifikations-Modell.
        
        Args:
            X: Feature-Matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            Selbst-Referenz für Method-Chaining
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) < 2:
            raise ValueError("Not enough samples for training")
        
        # Modell erstellen
        self.model = self._create_model()
        
        # Training durchführen
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Feature-Importance extrahieren (falls verfügbar)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        # Support Vectors extrahieren (falls verfügbar)
        if hasattr(self.model, 'support_vectors_'):
            self.support_vectors_ = self.model.support_vectors_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Macht Vorhersagen für neue Daten.
        
        Args:
            X: Feature-Matrix (n_samples, n_features)
            
        Returns:
            Vorhergesagte Labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Macht Wahrscheinlichkeits-Vorhersagen für neue Daten.
        
        Args:
            X: Feature-Matrix (n_samples, n_features)
            
        Returns:
            Vorhergesagte Wahrscheinlichkeiten
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback für Modelle ohne predict_proba
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluiert die Klassifikations-Performance.
        
        Args:
            X: Feature-Matrix
            y: Wahre Labels
            
        Returns:
            Evaluations-Metriken
        """
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
        """
        Führt Cross-Validation durch.
        
        Args:
            X: Feature-Matrix
            y: Labels
            cv: Anzahl CV-Folds
            
        Returns:
            Cross-Validation-Ergebnisse
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Cross-Validation-Scores
        test_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        train_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'test_score': test_scores.tolist(),
            'train_score': train_scores.tolist()
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Gibt die Feature-Importance zurück."""
        return self.feature_importances_