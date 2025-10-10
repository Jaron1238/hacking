#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Model Training für das WLAN-Analyse-Tool.
"""

import os
import json
import pickle
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

from ..constants import Constants, ErrorCodes, get_error_message
from ..exceptions import ValidationError, ResourceError
from ..validation import validate_dataframe, validate_timestamp
from ..metrics import record_timing, record_counter

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Zentraler ML Model Trainer für verschiedene Aufgaben."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.models: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
    
    def prepare_training_data(self, data: pd.DataFrame, target_column: str, 
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Bereite Trainingsdaten vor."""
        logger.info(f"Preparing training data: {len(data)} samples")
        
        # Validiere DataFrame
        validate_dataframe(data, required_columns=[target_column])
        
        # Feature-Spalten bestimmen
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Features und Target extrahieren
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Fehlende Werte behandeln
        X = self._handle_missing_values(X)
        
        # Kategorische Variablen encodieren
        X = self._encode_categorical_features(X, feature_columns)
        
        # Features normalisieren
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Scaler speichern
        self.scalers['default'] = scaler
        
        logger.info(f"Prepared data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Behandle fehlende Werte."""
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(strategy='median')
        return imputer.fit_transform(X)
    
    def _encode_categorical_features(self, X: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """Encode kategorische Features."""
        # Vereinfachte Implementierung - in Produktion würde man OneHotEncoder verwenden
        return X
    
    def train_device_classifier(self, X: np.ndarray, y: np.ndarray, 
                              model_name: str = "device_classifier") -> Dict[str, Any]:
        """Trainiere Device-Klassifikationsmodell."""
        logger.info(f"Training device classifier: {model_name}")
        start_time = time.time()
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verschiedene Modelle definieren
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        # Ensemble Model
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['random_forest']),
                ('gb', models['gradient_boosting']),
                ('svm', models['svm'])
            ],
            voting='soft'
        )
        
        # Hyperparameter-Tuning für Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Modelle trainieren
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            trained_models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Ensemble trainieren
        logger.info("Training ensemble...")
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        trained_models['ensemble'] = ensemble
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'classification_report': classification_report(y_test, y_pred_ensemble, output_dict=True)
        }
        
        # Hyperparameter-Tuning
        logger.info("Performing hyperparameter tuning...")
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        y_pred_best = best_rf.predict(X_test)
        best_accuracy = accuracy_score(y_test, y_pred_best)
        
        trained_models['best_rf'] = best_rf
        results['best_rf'] = {
            'accuracy': best_accuracy,
            'best_params': rf_grid.best_params_,
            'classification_report': classification_report(y_test, y_pred_best, output_dict=True)
        }
        
        # Bestes Modell auswählen
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = trained_models[best_model_name]
        
        # Training-Metadaten
        training_info = {
            'model_name': model_name,
            'timestamp': time.time(),
            'training_duration': time.time() - start_time,
            'best_model': best_model_name,
            'best_accuracy': results[best_model_name]['accuracy'],
            'all_results': results,
            'feature_count': X.shape[1],
            'sample_count': len(X),
            'test_accuracy': results[best_model_name]['accuracy']
        }
        
        # Modell speichern
        self._save_model(best_model, model_name, training_info)
        
        # Metriken aufzeichnen
        record_timing("ml_training_duration", training_info['training_duration'])
        record_counter("ml_models_trained", 1)
        
        logger.info(f"Training completed: {best_model_name} with {results[best_model_name]['accuracy']:.4f} accuracy")
        
        return training_info
    
    def train_anomaly_detector(self, X: np.ndarray, model_name: str = "anomaly_detector") -> Dict[str, Any]:
        """Trainiere Anomalie-Erkennungsmodell."""
        logger.info(f"Training anomaly detector: {model_name}")
        start_time = time.time()
        
        # Unsupervised Learning für Anomalie-Erkennung
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.covariance import EllipticEnvelope
        
        models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=0.1,
                kernel='rbf'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                n_jobs=-1
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=0.1,
                random_state=42
            )
        }
        
        # Modelle trainieren
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X)
            
            # Anomalie-Scores berechnen
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
            else:
                scores = model.score_samples(X)
            
            # Anomalien identifizieren
            anomalies = model.predict(X)
            anomaly_count = np.sum(anomalies == -1)
            
            trained_models[name] = model
            results[name] = {
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / len(X),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            
            logger.info(f"{name}: {anomaly_count} anomalies detected ({anomaly_count/len(X)*100:.2f}%)")
        
        # Bestes Modell auswählen (basierend auf Anomalie-Rate)
        best_model_name = min(results.keys(), key=lambda k: abs(results[k]['anomaly_rate'] - 0.1))
        best_model = trained_models[best_model_name]
        
        # Training-Metadaten
        training_info = {
            'model_name': model_name,
            'timestamp': time.time(),
            'training_duration': time.time() - start_time,
            'best_model': best_model_name,
            'best_anomaly_rate': results[best_model_name]['anomaly_rate'],
            'all_results': results,
            'feature_count': X.shape[1],
            'sample_count': len(X)
        }
        
        # Modell speichern
        self._save_model(best_model, model_name, training_info)
        
        # Metriken aufzeichnen
        record_timing("ml_anomaly_training_duration", training_info['training_duration'])
        record_counter("ml_anomaly_models_trained", 1)
        
        logger.info(f"Anomaly detector training completed: {best_model_name}")
        
        return training_info
    
    def train_behavior_predictor(self, X: np.ndarray, y: np.ndarray, 
                                model_name: str = "behavior_predictor") -> Dict[str, Any]:
        """Trainiere Verhaltens-Vorhersagemodell."""
        logger.info(f"Training behavior predictor: {model_name}")
        start_time = time.time()
        
        # Time Series Prediction Models
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Modelle trainieren
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluation
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            r2 = model.score(X_test, y_test)
            
            trained_models[name] = model
            results[name] = {
                'mse': mse,
                'r2_score': r2,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"{name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        # Bestes Modell auswählen (basierend auf R² Score)
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'])
        best_model = trained_models[best_model_name]
        
        # Training-Metadaten
        training_info = {
            'model_name': model_name,
            'timestamp': time.time(),
            'training_duration': time.time() - start_time,
            'best_model': best_model_name,
            'best_r2_score': results[best_model_name]['r2_score'],
            'all_results': results,
            'feature_count': X.shape[1],
            'sample_count': len(X)
        }
        
        # Modell speichern
        self._save_model(best_model, model_name, training_info)
        
        # Metriken aufzeichnen
        record_timing("ml_behavior_training_duration", training_info['training_duration'])
        record_counter("ml_behavior_models_trained", 1)
        
        logger.info(f"Behavior predictor training completed: {best_model_name}")
        
        return training_info
    
    def _save_model(self, model: Any, model_name: str, training_info: Dict[str, Any]) -> None:
        """Speichere trainiertes Modell."""
        model_path = self.model_dir / f"{model_name}.joblib"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        # Modell speichern
        joblib.dump(model, model_path)
        
        # Metadaten speichern
        with open(metadata_path, 'w') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        # Training History aktualisieren
        self.training_history.append(training_info)
        
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Lade trainiertes Modell."""
        model_path = self.model_dir / f"{model_name}.joblib"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Modell laden
        model = joblib.load(model_path)
        
        # Metadaten laden
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model loaded: {model_name}")
        return model, metadata
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Hole Training-Historie."""
        return self.training_history.copy()
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluiere Modell auf Testdaten."""
        start_time = time.time()
        
        # Vorhersagen
        y_pred = model.predict(X_test)
        
        # Metriken berechnen
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Accuracy für Klassifikation
        if len(np.unique(y_test)) < 10:  # Klassifikation
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation = {
                'type': 'classification',
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        else:  # Regression
            mse = np.mean((y_test - y_pred) ** 2)
            r2 = model.score(X_test, y_test) if hasattr(model, 'score') else 0
            
            evaluation = {
                'type': 'regression',
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'mae': np.mean(np.abs(y_test - y_pred))
            }
        
        evaluation['evaluation_duration'] = time.time() - start_time
        evaluation['predictions'] = y_pred.tolist()
        
        if y_proba is not None:
            evaluation['probabilities'] = y_proba.tolist()
        
        return evaluation


class AutoMLPipeline:
    """Automatisierte ML-Pipeline für verschiedene Aufgaben."""
    
    def __init__(self, trainer: MLModelTrainer):
        self.trainer = trainer
        self.pipelines: Dict[str, Pipeline] = {}
    
    def create_device_classification_pipeline(self) -> Pipeline:
        """Erstelle Pipeline für Device-Klassifikation."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        self.pipelines['device_classification'] = pipeline
        return pipeline
    
    def create_anomaly_detection_pipeline(self) -> Pipeline:
        """Erstelle Pipeline für Anomalie-Erkennung."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('anomaly_detector', IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        self.pipelines['anomaly_detection'] = pipeline
        return pipeline
    
    def run_automated_training(self, data: pd.DataFrame, task_type: str, 
                             target_column: Optional[str] = None) -> Dict[str, Any]:
        """Führe automatisiertes Training durch."""
        logger.info(f"Running automated training for {task_type}")
        
        if task_type == "device_classification":
            if target_column is None:
                target_column = "device_type"
            
            X, y = self.trainer.prepare_training_data(data, target_column)
            return self.trainer.train_device_classifier(X, y)
        
        elif task_type == "anomaly_detection":
            X = data.select_dtypes(include=[np.number]).values
            return self.trainer.train_anomaly_detector(X)
        
        elif task_type == "behavior_prediction":
            if target_column is None:
                target_column = "behavior_score"
            
            X, y = self.trainer.prepare_training_data(data, target_column)
            return self.trainer.train_behavior_predictor(X, y)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def optimize_hyperparameters(self, pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimiere Hyperparameter automatisch."""
        logger.info("Optimizing hyperparameters...")
        
        # Parameter-Grid definieren
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Grid Search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }