"""
Unit Tests für Machine Learning Modelle.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

import pytest
from wlan_tool.ml_models import ClassificationModel, ClusteringModel, EnsembleModel


class TestClusteringModel:
    """Tests für ClusteringModel."""

    def test_init(self):
        """Test Initialisierung."""
        model = ClusteringModel()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_fit_kmeans(self, sample_features):
        """Test K-Means Clustering."""
        model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        model.fit(sample_features)

        assert model.is_fitted
        assert hasattr(model, "cluster_centers_")

    def test_fit_dbscan(self, sample_features):
        """Test DBSCAN Clustering."""
        model = ClusteringModel(algorithm="dbscan", eps=0.5, min_samples=5)
        model.fit(sample_features)

        assert model.is_fitted
        assert hasattr(model, "labels_")

    def test_predict_before_fit(self, sample_features):
        """Test Vorhersage vor Training."""
        model = ClusteringModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(sample_features)

    def test_predict_after_fit(self, sample_features):
        """Test Vorhersage nach Training."""
        model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        model.fit(sample_features)

        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(0 <= p < 3 for p in predictions)

    def test_get_cluster_centers(self, sample_features):
        """Test Cluster-Zentren abrufen."""
        model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        model.fit(sample_features)

        centers = model.get_cluster_centers()

        assert centers.shape[0] == 3
        assert centers.shape[1] == sample_features.shape[1]

    def test_evaluate_clustering(self, sample_features):
        """Test Clustering-Evaluation."""
        model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        model.fit(sample_features)

        metrics = model.evaluate(sample_features)

        assert "silhouette_score" in metrics
        assert "inertia" in metrics
        assert "n_clusters" in metrics
        assert isinstance(metrics["silhouette_score"], float)
        assert isinstance(metrics["inertia"], float)
        assert metrics["n_clusters"] == 3

    def test_invalid_algorithm(self):
        """Test mit ungültigem Algorithmus."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ClusteringModel(algorithm="invalid_algorithm")

    def test_hyperparameter_validation(self):
        """Test Hyperparameter-Validierung."""
        # Ungültige n_clusters
        with pytest.raises(ValueError, match="n_clusters must be positive"):
            ClusteringModel(algorithm="kmeans", n_clusters=0)

        # Ungültige eps
        with pytest.raises(ValueError, match="eps must be positive"):
            ClusteringModel(algorithm="dbscan", eps=-0.1)

        # Ungültige min_samples
        with pytest.raises(ValueError, match="min_samples must be positive"):
            ClusteringModel(algorithm="dbscan", min_samples=0)


class TestClassificationModel:
    """Tests für ClassificationModel."""

    def test_init(self):
        """Test Initialisierung."""
        model = ClassificationModel()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_fit_random_forest(self, sample_features, sample_labels):
        """Test Random Forest Training."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        assert model.is_fitted
        assert hasattr(model, "feature_importances_")

    def test_fit_svm(self, sample_features, sample_labels):
        """Test SVM Training."""
        model = ClassificationModel(algorithm="svm", kernel="rbf")
        model.fit(sample_features, sample_labels)

        assert model.is_fitted
        assert hasattr(model, "support_vectors_")

    def test_predict_before_fit(self, sample_features):
        """Test Vorhersage vor Training."""
        model = ClassificationModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(sample_features)

    def test_predict_after_fit(self, sample_features, sample_labels):
        """Test Vorhersage nach Training."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(p in sample_labels for p in predictions)

    def test_predict_proba(self, sample_features, sample_labels):
        """Test Wahrscheinlichkeits-Vorhersage."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        probabilities = model.predict_proba(sample_features)

        assert probabilities.shape[0] == len(sample_features)
        assert probabilities.shape[1] == len(np.unique(sample_labels))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluate_classification(self, sample_features, sample_labels):
        """Test Klassifikations-Evaluation."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        metrics = model.evaluate(sample_features, sample_labels)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_cross_validation(self, sample_features, sample_labels):
        """Test Cross-Validation."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)

        cv_scores = model.cross_validate(sample_features, sample_labels, cv=3)

        assert "test_score" in cv_scores
        assert "train_score" in cv_scores
        assert len(cv_scores["test_score"]) == 3
        assert len(cv_scores["train_score"]) == 3

    def test_feature_importance(self, sample_features, sample_labels):
        """Test Feature-Importance."""
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        importance = model.get_feature_importance()

        assert len(importance) == sample_features.shape[1]
        assert all(imp >= 0 for imp in importance)
        assert abs(sum(importance) - 1.0) < 1e-6


class TestEnsembleModel:
    """Tests für EnsembleModel."""

    def test_init(self):
        """Test Initialisierung."""
        model = EnsembleModel()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_fit_voting_classifier(self, sample_features, sample_labels):
        """Test Voting Classifier Training."""
        model = EnsembleModel(
            algorithm="voting",
            estimators=[
                ("rf", ClassificationModel(algorithm="random_forest")),
                ("svm", ClassificationModel(algorithm="svm")),
            ],
        )
        model.fit(sample_features, sample_labels)

        assert model.is_fitted
        assert len(model.estimators_) == 2

    def test_fit_bagging(self, sample_features, sample_labels):
        """Test Bagging Training."""
        model = EnsembleModel(
            algorithm="bagging",
            base_estimator=ClassificationModel(algorithm="random_forest"),
            n_estimators=5,
        )
        model.fit(sample_features, sample_labels)

        assert model.is_fitted
        assert len(model.estimators_) == 5

    def test_fit_boosting(self, sample_features, sample_labels):
        """Test Boosting Training."""
        model = EnsembleModel(
            algorithm="boosting",
            base_estimator=ClassificationModel(algorithm="random_forest"),
            n_estimators=5,
        )
        model.fit(sample_features, sample_labels)

        assert model.is_fitted
        assert len(model.estimators_) == 5

    def test_predict_ensemble(self, sample_features, sample_labels):
        """Test Ensemble-Vorhersage."""
        model = EnsembleModel(
            algorithm="voting",
            estimators=[
                ("rf", ClassificationModel(algorithm="random_forest")),
                ("svm", ClassificationModel(algorithm="svm")),
            ],
        )
        model.fit(sample_features, sample_labels)

        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)

    def test_evaluate_ensemble(self, sample_features, sample_labels):
        """Test Ensemble-Evaluation."""
        model = EnsembleModel(
            algorithm="voting",
            estimators=[
                ("rf", ClassificationModel(algorithm="random_forest")),
                ("svm", ClassificationModel(algorithm="svm")),
            ],
        )
        model.fit(sample_features, sample_labels)

        metrics = model.evaluate(sample_features, sample_labels)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_individual_estimator_performance(self, sample_features, sample_labels):
        """Test Performance einzelner Estimators."""
        model = EnsembleModel(
            algorithm="voting",
            estimators=[
                ("rf", ClassificationModel(algorithm="random_forest")),
                ("svm", ClassificationModel(algorithm="svm")),
            ],
        )
        model.fit(sample_features, sample_labels)

        individual_scores = model.get_individual_scores(sample_features, sample_labels)

        assert len(individual_scores) == 2
        assert "rf" in individual_scores
        assert "svm" in individual_scores
        assert all(isinstance(score, float) for score in individual_scores.values())

    def test_invalid_algorithm(self):
        """Test mit ungültigem Ensemble-Algorithmus."""
        with pytest.raises(ValueError, match="Unsupported ensemble algorithm"):
            EnsembleModel(algorithm="invalid_ensemble")

    def test_empty_estimators(self):
        """Test mit leerer Estimator-Liste."""
        with pytest.raises(ValueError, match="At least one estimator required"):
            EnsembleModel(algorithm="voting", estimators=[])


class TestModelPersistence:
    """Tests für Model-Persistierung."""

    def test_save_load_model(self, sample_features, sample_labels, temp_dir):
        """Test Speichern und Laden von Modellen."""
        # Training
        model = ClassificationModel(algorithm="random_forest", n_estimators=10)
        model.fit(sample_features, sample_labels)

        # Speichern
        model_path = temp_dir / "test_model.pkl"
        model.save(model_path)

        assert model_path.exists()

        # Laden
        loaded_model = ClassificationModel.load(model_path)

        assert loaded_model.is_fitted
        assert loaded_model.algorithm == model.algorithm

        # Vorhersage vergleichen
        original_predictions = model.predict(sample_features)
        loaded_predictions = loaded_model.predict(sample_features)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_save_load_ensemble(self, sample_features, sample_labels, temp_dir):
        """Test Speichern und Laden von Ensemble-Modellen."""
        # Training
        model = EnsembleModel(
            algorithm="voting",
            estimators=[
                ("rf", ClassificationModel(algorithm="random_forest")),
                ("svm", ClassificationModel(algorithm="svm")),
            ],
        )
        model.fit(sample_features, sample_labels)

        # Speichern
        model_path = temp_dir / "test_ensemble.pkl"
        model.save(model_path)

        assert model_path.exists()

        # Laden
        loaded_model = EnsembleModel.load(model_path)

        assert loaded_model.is_fitted
        assert loaded_model.algorithm == model.algorithm
        assert len(loaded_model.estimators_) == len(model.estimators_)
