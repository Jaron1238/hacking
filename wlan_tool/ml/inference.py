#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Inference Engine für das WLAN-Analyse-Tool.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..caching import get_cache
from ..constants import Constants, ErrorCodes, get_error_message
from ..exceptions import ResourceError, ValidationError
from ..metrics import record_counter, record_timing
from ..validation import validate_dataframe
from .clustering import BaseClusteringAlgorithm, ClusteringPipeline
from .models import AnomalyDetector, BaseMLModel, BehaviorPredictor, DeviceClassifier

logger = logging.getLogger(__name__)


class MLInferenceEngine:
    """Zentrale Inference Engine für alle ML-Modelle."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_models: Dict[str, BaseMLModel] = {}
        self.clustering_models: Dict[str, BaseClusteringAlgorithm] = {}
        self.cache = get_cache("ml_inference")
        self.inference_history: List[Dict[str, Any]] = []

    def load_model(self, model_name: str, model_type: str = "classification") -> bool:
        """Lade ML-Modell."""
        try:
            model_path = self.models_dir / f"{model_name}.joblib"

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Modell-Typ bestimmen und laden
            if model_type == "classification":
                model = DeviceClassifier(model_name)
            elif model_type == "anomaly":
                model = AnomalyDetector(model_name)
            elif model_type == "behavior":
                model = BehaviorPredictor(model_name)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False

            model.load_model(str(model_path))
            self.active_models[model_name] = model

            logger.info(f"Model loaded: {model_name} ({model_type})")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def load_clustering_model(self, model_name: str, algorithm_type: str) -> bool:
        """Lade Clustering-Modell."""
        try:
            from .clustering import (
                DBSCANClustering,
                GaussianMixtureClustering,
                HierarchicalClustering,
                KMeansClustering,
                SpectralClustering,
            )

            # Algorithmus-Typ bestimmen
            algorithm_map = {
                "kmeans": KMeansClustering,
                "dbscan": DBSCANClustering,
                "hierarchical": HierarchicalClustering,
                "spectral": SpectralClustering,
                "gaussian_mixture": GaussianMixtureClustering,
            }

            if algorithm_type not in algorithm_map:
                logger.error(f"Unknown clustering algorithm: {algorithm_type}")
                return False

            # Modell erstellen und laden
            model = algorithm_map[algorithm_type]()
            model_path = self.models_dir / f"{model_name}_clustering.joblib"

            if model_path.exists():
                model_data = joblib.load(model_path)
                model.model = model_data["model"]
                model.scaler = model_data["scaler"]
                model.is_fitted = model_data["is_fitted"]
                model.labels_ = model_data["labels_"]
                model.n_clusters_ = model_data["n_clusters_"]
                model.cluster_centers_ = model_data.get("cluster_centers_")

            self.clustering_models[model_name] = model

            logger.info(f"Clustering model loaded: {model_name} ({algorithm_type})")
            return True

        except Exception as e:
            logger.error(f"Error loading clustering model {model_name}: {e}")
            return False

    def predict_device_type(
        self, features: np.ndarray, model_name: str = "device_classifier"
    ) -> Dict[str, Any]:
        """Vorhersage von Gerätetyp."""
        start_time = time.time()

        # Cache prüfen
        cache_key = f"device_prediction_{hash(features.tobytes())}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Using cached device prediction")
            return cached_result

        try:
            if model_name not in self.active_models:
                if not self.load_model(model_name, "classification"):
                    raise ValueError(f"Model {model_name} not available")

            model = self.active_models[model_name]

            if not model.is_trained:
                raise ValueError(f"Model {model_name} not trained")

            # Vorhersage
            prediction = model.predict(features.reshape(1, -1))[0]

            # Wahrscheinlichkeiten (falls verfügbar)
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features.reshape(1, -1))[0]

            result = {
                "predicted_type": prediction,
                "probabilities": (
                    probabilities.tolist() if probabilities is not None else None
                ),
                "confidence": (
                    float(np.max(probabilities)) if probabilities is not None else 1.0
                ),
                "model_name": model_name,
                "inference_time": time.time() - start_time,
            }

            # Cache speichern
            self.cache.set(cache_key, result, ttl=3600)  # 1 Stunde

            # Metriken aufzeichnen
            record_timing("ml_device_prediction_duration", result["inference_time"])
            record_counter("ml_device_predictions", 1)

            # History aktualisieren
            self.inference_history.append(
                {
                    "timestamp": time.time(),
                    "type": "device_prediction",
                    "model": model_name,
                    "prediction": prediction,
                    "confidence": result["confidence"],
                }
            )

            logger.debug(
                f"Device prediction: {prediction} (confidence: {result['confidence']:.3f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error in device prediction: {e}")
            return {
                "error": str(e),
                "predicted_type": "unknown",
                "confidence": 0.0,
                "inference_time": time.time() - start_time,
            }

    def detect_anomaly(
        self, features: np.ndarray, model_name: str = "anomaly_detector"
    ) -> Dict[str, Any]:
        """Anomalie-Erkennung."""
        start_time = time.time()

        # Cache prüfen
        cache_key = f"anomaly_detection_{hash(features.tobytes())}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Using cached anomaly detection")
            return cached_result

        try:
            if model_name not in self.active_models:
                if not self.load_model(model_name, "anomaly"):
                    raise ValueError(f"Model {model_name} not available")

            model = self.active_models[model_name]

            if not model.is_trained:
                raise ValueError(f"Model {model_name} not trained")

            # Anomalie-Score berechnen
            if hasattr(model, "decision_function"):
                score = model.decision_function(features.reshape(1, -1))[0]
            else:
                score = model.predict(features.reshape(1, -1))[0]

            # Anomalie-Status
            is_anomaly = score < 0 if hasattr(model, "threshold") else score == -1

            result = {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(score),
                "confidence": abs(score),
                "model_name": model_name,
                "inference_time": time.time() - start_time,
            }

            # Cache speichern
            self.cache.set(cache_key, result, ttl=1800)  # 30 Minuten

            # Metriken aufzeichnen
            record_timing("ml_anomaly_detection_duration", result["inference_time"])
            record_counter("ml_anomaly_detections", 1)

            # History aktualisieren
            self.inference_history.append(
                {
                    "timestamp": time.time(),
                    "type": "anomaly_detection",
                    "model": model_name,
                    "is_anomaly": is_anomaly,
                    "score": score,
                }
            )

            logger.debug(f"Anomaly detection: {is_anomaly} (score: {score:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "error": str(e),
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "inference_time": time.time() - start_time,
            }

    def predict_behavior(
        self, features: np.ndarray, model_name: str = "behavior_predictor"
    ) -> Dict[str, Any]:
        """Verhaltens-Vorhersage."""
        start_time = time.time()

        # Cache prüfen
        cache_key = f"behavior_prediction_{hash(features.tobytes())}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Using cached behavior prediction")
            return cached_result

        try:
            if model_name not in self.active_models:
                if not self.load_model(model_name, "behavior"):
                    raise ValueError(f"Model {model_name} not available")

            model = self.active_models[model_name]

            if not model.is_trained:
                raise ValueError(f"Model {model_name} not trained")

            # Vorhersage
            prediction = model.predict(features.reshape(1, -1))[0]

            # Konfidenz (falls verfügbar)
            confidence = None
            if hasattr(model, "predict_confidence"):
                prediction, confidence = model.predict_confidence(
                    features.reshape(1, -1)
                )
                prediction = prediction[0]
                confidence = confidence[0]

            result = {
                "predicted_behavior": float(prediction),
                "confidence": float(confidence) if confidence is not None else 1.0,
                "model_name": model_name,
                "inference_time": time.time() - start_time,
            }

            # Cache speichern
            self.cache.set(cache_key, result, ttl=1800)  # 30 Minuten

            # Metriken aufzeichnen
            record_timing("ml_behavior_prediction_duration", result["inference_time"])
            record_counter("ml_behavior_predictions", 1)

            # History aktualisieren
            self.inference_history.append(
                {
                    "timestamp": time.time(),
                    "type": "behavior_prediction",
                    "model": model_name,
                    "prediction": prediction,
                    "confidence": confidence,
                }
            )

            logger.debug(
                f"Behavior prediction: {prediction} (confidence: {confidence:.3f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error in behavior prediction: {e}")
            return {
                "error": str(e),
                "predicted_behavior": 0.0,
                "confidence": 0.0,
                "inference_time": time.time() - start_time,
            }

    def cluster_data(
        self, features: np.ndarray, model_name: str = "clustering_model"
    ) -> Dict[str, Any]:
        """Daten-Clustering."""
        start_time = time.time()

        # Cache prüfen
        cache_key = f"clustering_{hash(features.tobytes())}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Using cached clustering result")
            return cached_result

        try:
            if model_name not in self.clustering_models:
                # Automatisches Clustering
                from .clustering import ClusteringPipeline

                pipeline = ClusteringPipeline()
                clustering_result = pipeline.auto_cluster(features)
                best_model = clustering_result["best_model"]
            else:
                best_model = self.clustering_models[model_name]

            if not best_model.is_fitted:
                raise ValueError(f"Clustering model {model_name} not fitted")

            # Clustering
            labels = best_model.predict(features)

            # Cluster-Statistiken
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            result = {
                "labels": labels.tolist(),
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "model_name": model_name,
                "inference_time": time.time() - start_time,
            }

            # Cache speichern
            self.cache.set(cache_key, result, ttl=3600)  # 1 Stunde

            # Metriken aufzeichnen
            record_timing("ml_clustering_duration", result["inference_time"])
            record_counter("ml_clustering_operations", 1)

            # History aktualisieren
            self.inference_history.append(
                {
                    "timestamp": time.time(),
                    "type": "clustering",
                    "model": model_name,
                    "n_clusters": n_clusters,
                    "n_samples": len(features),
                }
            )

            logger.debug(f"Clustering completed: {n_clusters} clusters")
            return result

        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {
                "error": str(e),
                "labels": [],
                "n_clusters": 0,
                "inference_time": time.time() - start_time,
            }

    def batch_predict(
        self, features_list: List[np.ndarray], prediction_type: str = "device"
    ) -> List[Dict[str, Any]]:
        """Batch-Vorhersagen für mehrere Samples."""
        logger.info(
            f"Batch prediction: {len(features_list)} samples, type: {prediction_type}"
        )

        results = []

        for i, features in enumerate(features_list):
            try:
                if prediction_type == "device":
                    result = self.predict_device_type(features)
                elif prediction_type == "anomaly":
                    result = self.detect_anomaly(features)
                elif prediction_type == "behavior":
                    result = self.predict_behavior(features)
                elif prediction_type == "clustering":
                    result = self.cluster_data(features)
                else:
                    result = {"error": f"Unknown prediction type: {prediction_type}"}

                results.append(result)

            except Exception as e:
                logger.error(f"Error in batch prediction {i}: {e}")
                results.append({"error": str(e)})

        # Metriken aufzeichnen
        record_counter(f"ml_batch_{prediction_type}_predictions", len(features_list))

        return results

    async def async_predict(
        self, features: np.ndarray, prediction_type: str = "device"
    ) -> Dict[str, Any]:
        """Asynchrone Vorhersage."""
        loop = asyncio.get_event_loop()

        if prediction_type == "device":
            return await loop.run_in_executor(None, self.predict_device_type, features)
        elif prediction_type == "anomaly":
            return await loop.run_in_executor(None, self.detect_anomaly, features)
        elif prediction_type == "behavior":
            return await loop.run_in_executor(None, self.predict_behavior, features)
        elif prediction_type == "clustering":
            return await loop.run_in_executor(None, self.cluster_data, features)
        else:
            return {"error": f"Unknown prediction type: {prediction_type}"}

    def get_inference_stats(self) -> Dict[str, Any]:
        """Hole Inference-Statistiken."""
        if not self.inference_history:
            return {"total_predictions": 0}

        # Statistiken berechnen
        total_predictions = len(self.inference_history)

        # Nach Typ gruppieren
        type_counts = {}
        for entry in self.inference_history:
            pred_type = entry["type"]
            type_counts[pred_type] = type_counts.get(pred_type, 0) + 1

        # Durchschnittliche Konfidenz
        confidences = [
            entry.get("confidence", 0)
            for entry in self.inference_history
            if "confidence" in entry
        ]
        avg_confidence = np.mean(confidences) if confidences else 0

        # Modell-Usage
        model_usage = {}
        for entry in self.inference_history:
            model = entry.get("model", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1

        return {
            "total_predictions": total_predictions,
            "type_distribution": type_counts,
            "average_confidence": avg_confidence,
            "model_usage": model_usage,
            "cache_hit_rate": self.cache.get_stats().get("hit_rate", 0),
        }

    def clear_cache(self) -> None:
        """Leere Inference-Cache."""
        self.cache.clear()
        logger.info("Inference cache cleared")

    def export_inference_history(self, filepath: str) -> None:
        """Exportiere Inference-Historie."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.inference_history, f, indent=2, default=str)

        logger.info(f"Inference history exported to {filepath}")


# Convenience-Funktionen
def create_inference_engine(models_dir: str = "models") -> MLInferenceEngine:
    """Erstelle ML Inference Engine."""
    return MLInferenceEngine(models_dir)


def quick_device_classification(
    features: np.ndarray, model_name: str = "device_classifier"
) -> str:
    """Schnelle Geräte-Klassifikation."""
    engine = create_inference_engine()
    result = engine.predict_device_type(features, model_name)
    return result.get("predicted_type", "unknown")


def quick_anomaly_detection(
    features: np.ndarray, model_name: str = "anomaly_detector"
) -> bool:
    """Schnelle Anomalie-Erkennung."""
    engine = create_inference_engine()
    result = engine.detect_anomaly(features, model_name)
    return result.get("is_anomaly", False)


def quick_behavior_prediction(
    features: np.ndarray, model_name: str = "behavior_predictor"
) -> float:
    """Schnelle Verhaltens-Vorhersage."""
    engine = create_inference_engine()
    result = engine.predict_behavior(features, model_name)
    return result.get("predicted_behavior", 0.0)
