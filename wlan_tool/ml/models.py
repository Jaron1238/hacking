#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Models für das WLAN-Analyse-Tool.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

from ..constants import Constants, ErrorCodes, get_error_message
from ..exceptions import ValidationError, ResourceError
from ..validation import validate_dataframe

logger = logging.getLogger(__name__)


class BaseMLModel(ABC):
    """Basis-Klasse für alle ML-Modelle."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        self.class_names = []
    
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Trainiere das Modell."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mache Vorhersagen."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Speichere Modell."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Lade Modell."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        logger.info(f"Model loaded: {filepath}")


class DeviceClassifier(BaseMLModel):
    """Geräte-Klassifikationsmodell."""
    
    def __init__(self, model_name: str = "device_classifier"):
        super().__init__(model_name)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Trainiere Device-Klassifikator."""
        logger.info(f"Training device classifier with {len(X)} samples")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Labels encodieren
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # Modell trainieren
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Training-Metadaten
        training_info = {
            'model_name': self.model_name,
            'sample_count': len(X),
            'feature_count': X.shape[1],
            'class_count': len(self.class_names),
            'classes': self.class_names
        }
        
        logger.info(f"Device classifier trained: {len(self.class_names)} classes")
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Klassifiziere Geräte."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Klassifikations-Wahrscheinlichkeiten."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Hole Feature-Importance."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))


class AnomalyDetector(BaseMLModel):
    """Anomalie-Erkennungsmodell."""
    
    def __init__(self, model_name: str = "anomaly_detector"):
        super().__init__(model_name)
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.threshold = 0.0
    
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Trainiere Anomalie-Detektor."""
        logger.info(f"Training anomaly detector with {len(X)} samples")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Anomalie-Scores berechnen
        scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, 10)  # 10% als Anomalien
        
        # Training-Metadaten
        training_info = {
            'model_name': self.model_name,
            'sample_count': len(X),
            'feature_count': X.shape[1],
            'threshold': self.threshold,
            'contamination': self.model.contamination
        }
        
        logger.info(f"Anomaly detector trained: threshold={self.threshold:.4f}")
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Erkenne Anomalien."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomalie-Scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Prüfe ob Anomalien vorliegen."""
        scores = self.decision_function(X)
        return scores < self.threshold


class BehaviorPredictor(BaseMLModel):
    """Verhaltens-Vorhersagemodell."""
    
    def __init__(self, model_name: str = "behavior_predictor"):
        super().__init__(model_name)
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Trainiere Behavior-Predictor."""
        logger.info(f"Training behavior predictor with {len(X)} samples")
        
        # Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Modell trainieren
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Training-Metadaten
        training_info = {
            'model_name': self.model_name,
            'sample_count': len(X),
            'feature_count': X.shape[1],
            'target_range': [float(np.min(y)), float(np.max(y))]
        }
        
        logger.info(f"Behavior predictor trained")
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage von Verhalten."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage mit Konfidenz-Intervall."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Einfache Konfidenz-Schätzung basierend auf Vorhersage-Varianz
        # In Produktion würde man Bootstrap oder andere Methoden verwenden
        confidence = np.ones_like(predictions) * 0.8  # Placeholder
        
        return predictions, confidence


class MLModelManager:
    """Manager für alle ML-Modelle."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_models: Dict[str, BaseMLModel] = {}
    
    def register_model(self, model: BaseMLModel) -> None:
        """Registriere Modell."""
        self.active_models[model.model_name] = model
        logger.info(f"Model registered: {model.model_name}")
    
    def get_model(self, model_name: str) -> Optional[BaseMLModel]:
        """Hole Modell."""
        return self.active_models.get(model_name)
    
    def train_all_models(self, training_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]) -> Dict[str, Dict[str, Any]]:
        """Trainiere alle Modelle."""
        results = {}
        
        for model_name, (X, y) in training_data.items():
            if model_name in self.active_models:
                model = self.active_models[model_name]
                try:
                    result = model.train(X, y)
                    results[model_name] = result
                    logger.info(f"Model {model_name} trained successfully")
                except Exception as e:
                    logger.error(f"Error training model {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Vorhersagen mit allen Modellen."""
        predictions = {}
        
        for model_name, model in self.active_models.items():
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    predictions[model_name] = None
        
        return predictions
    
    def save_all_models(self) -> None:
        """Speichere alle Modelle."""
        for model_name, model in self.active_models.items():
            if model.is_trained:
                filepath = self.models_dir / f"{model_name}.joblib"
                model.save_model(str(filepath))
    
    def load_all_models(self) -> None:
        """Lade alle Modelle."""
        for model_file in self.models_dir.glob("*.joblib"):
            model_name = model_file.stem
            try:
                # Modell-Typ basierend auf Namen bestimmen
                if "classifier" in model_name:
                    model = DeviceClassifier(model_name)
                elif "anomaly" in model_name:
                    model = AnomalyDetector(model_name)
                elif "behavior" in model_name:
                    model = BehaviorPredictor(model_name)
                else:
                    continue
                
                model.load_model(str(model_file))
                self.active_models[model_name] = model
                logger.info(f"Model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")


# Convenience-Funktionen
def create_device_classifier() -> DeviceClassifier:
    """Erstelle Device-Klassifikator."""
    return DeviceClassifier()


def create_anomaly_detector() -> AnomalyDetector:
    """Erstelle Anomalie-Detektor."""
    return AnomalyDetector()


def create_behavior_predictor() -> BehaviorPredictor:
    """Erstelle Behavior-Predictor."""
    return BehaviorPredictor()


def create_model_manager(models_dir: str = "models") -> MLModelManager:
    """Erstelle Model Manager."""
    return MLModelManager(models_dir)