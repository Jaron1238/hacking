"""
Memory-Profiling-Tests für das WLAN-Tool.
"""

import os

import numpy as np
import pandas as pd
import psutil
from memory_profiler import profile

import pytest
from wlan_tool.analysis import ClusteringAnalyzer, DeviceClassifier
from wlan_tool.data_processing import FeatureExtractor, WiFiDataProcessor


class TestMemoryUsage:
    """Tests für Speichernutzung."""

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_data_processing(self, large_dataset):
        """Test Speichernutzung bei Datenverarbeitung."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        processor = WiFiDataProcessor()
        processed_data = processor.process_data(large_dataset)

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        # Speicher-Zunahme sollte unter 200 MB sein
        assert (
            memory_increase < 200
        ), f"Zu viel Speicher verwendet: {memory_increase:.2f} MB"

        # Verarbeitete Daten sollten korrekt sein
        assert len(processed_data) == len(large_dataset)
        assert "processed_timestamp" in processed_data.columns

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_feature_extraction(self, large_dataset):
        """Test Speichernutzung bei Feature-Extraktion."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        extractor = FeatureExtractor()
        features = extractor.extract_features(large_dataset)

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        # Feature-Extraktion sollte effizient sein
        assert (
            memory_increase < 300
        ), f"Zu viel Speicher für Features: {memory_increase:.2f} MB"
        assert features.shape[0] == len(large_dataset)
        assert features.shape[1] > 0

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_clustering(self, large_dataset):
        """Test Speichernutzung beim Clustering."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        analyzer = ClusteringAnalyzer()
        feature_extractor = FeatureExtractor()

        # Features extrahieren
        features = feature_extractor.extract_features(large_dataset)
        memory_after_features = process.memory_info().rss / 1024 / 1024

        # Clustering durchführen
        labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)
        memory_after_clustering = process.memory_info().rss / 1024 / 1024

        feature_memory = memory_after_features - memory_before
        clustering_memory = memory_after_clustering - memory_after_features
        total_memory = memory_after_clustering - memory_before

        # Speichernutzung sollte angemessen sein
        assert total_memory < 500, f"Zu viel Speicher insgesamt: {total_memory:.2f} MB"
        assert (
            clustering_memory < 200
        ), f"Zu viel Speicher für Clustering: {clustering_memory:.2f} MB"

        # Clustering-Ergebnisse sollten korrekt sein
        assert len(labels) == len(features)
        assert all(isinstance(label, (int, np.integer)) for label in labels)

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_classification(self, large_dataset):
        """Test Speichernutzung bei Klassifikation."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        classifier = DeviceClassifier()
        feature_extractor = FeatureExtractor()

        # Features und Labels generieren
        features = feature_extractor.extract_features(large_dataset)
        labels = np.random.randint(0, 5, len(features))

        # Training
        classifier.train(features, labels)
        memory_after_training = process.memory_info().rss / 1024 / 1024

        # Vorhersage
        predictions = classifier.predict(features)
        memory_after_prediction = process.memory_info().rss / 1024 / 1024

        training_memory = memory_after_training - memory_before
        prediction_memory = memory_after_prediction - memory_after_training
        total_memory = memory_after_prediction - memory_before

        # Speichernutzung sollte angemessen sein
        assert (
            total_memory < 400
        ), f"Zu viel Speicher für Klassifikation: {total_memory:.2f} MB"
        assert (
            training_memory < 300
        ), f"Zu viel Speicher für Training: {training_memory:.2f} MB"

        # Klassifikations-Ergebnisse sollten korrekt sein
        assert len(predictions) == len(features)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test auf Memory-Leaks bei wiederholten Operationen."""
        process = psutil.Process(os.getpid())

        # Initiale Speichernutzung
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Wiederholte Operationen
        for i in range(10):
            # Große Daten generieren
            data = np.random.randn(5000, 20)

            # Clustering durchführen
            analyzer = ClusteringAnalyzer()
            labels = analyzer.cluster_data(data, algorithm="kmeans", n_clusters=3)

            # Speicher nach jeder Iteration messen
            current_memory = process.memory_info().rss / 1024 / 1024

            # Speicher sollte nicht kontinuierlich steigen
            memory_increase = current_memory - initial_memory
            assert (
                memory_increase < 100
            ), f"Memory-Leak erkannt nach {i+1} Iterationen: {memory_increase:.2f} MB"

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_large_matrices(self):
        """Test Speichernutzung bei sehr großen Matrizen."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        # Sehr große Matrix
        large_matrix = np.random.randn(50000, 100)

        # Matrix-Operationen
        result = np.dot(large_matrix.T, large_matrix)
        eigenvalues = np.linalg.eigvals(result)

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        # Speicher für große Matrizen sollte angemessen sein
        assert (
            memory_increase < 1000
        ), f"Zu viel Speicher für große Matrizen: {memory_increase:.2f} MB"

        # Ergebnisse sollten korrekt sein
        assert result.shape == (100, 100)
        assert len(eigenvalues) == 100


class TestMemoryProfiling:
    """Detaillierte Memory-Profiling-Tests."""

    @pytest.mark.memory
    @pytest.mark.performance
    def test_detailed_memory_profiling(self, large_dataset):
        """Detailliertes Memory-Profiling der WLAN-Analyse."""
        from memory_profiler import memory_usage

        def analyze_wifi_data():
            """WLAN-Datenanalyse mit Memory-Tracking."""
            processor = WiFiDataProcessor()
            extractor = FeatureExtractor()
            analyzer = ClusteringAnalyzer()

            # 1. Daten verarbeiten
            processed_data = processor.process_data(large_dataset)

            # 2. Features extrahieren
            features = extractor.extract_features(processed_data)

            # 3. Clustering
            labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)

            # 4. Ergebnisse zusammenfassen
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            return {
                "n_clusters": len(unique_labels),
                "cluster_sizes": cluster_sizes,
                "features_shape": features.shape,
            }

        # Memory-Usage während der Ausführung messen
        mem_usage = memory_usage(analyze_wifi_data, interval=0.1)

        # Memory-Usage sollte stabil sein
        max_memory = max(mem_usage)
        min_memory = min(mem_usage)
        memory_variance = max_memory - min_memory

        # Varianz sollte nicht zu groß sein (keine Memory-Leaks)
        assert (
            memory_variance < 200
        ), f"Hohe Memory-Varianz erkannt: {memory_variance:.2f} MB"

        # Maximaler Speicher sollte angemessen sein
        assert max_memory < 1000, f"Zu viel maximaler Speicher: {max_memory:.2f} MB"

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_profiling_with_tracemalloc(self):
        """Memory-Profiling mit Python's tracemalloc."""
        import tracemalloc

        # Tracemalloc starten
        tracemalloc.start()

        # Snapshot vor der Ausführung
        snapshot1 = tracemalloc.take_snapshot()

        # WLAN-Analyse ausführen
        data = np.random.randn(10000, 50)
        analyzer = ClusteringAnalyzer()
        labels = analyzer.cluster_data(data, algorithm="kmeans", n_clusters=5)

        # Snapshot nach der Ausführung
        snapshot2 = tracemalloc.take_snapshot()

        # Top 10 Speicher-Verbraucher
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Speicher-Statistiken
        current, peak = tracemalloc.get_traced_memory()

        # Peak-Speicher sollte angemessen sein
        assert (
            peak / 1024 / 1024 < 500
        ), f"Zu viel Peak-Speicher: {peak / 1024 / 1024:.2f} MB"

        # Aktueller Speicher sollte nicht zu hoch sein
        assert (
            current / 1024 / 1024 < 200
        ), f"Zu viel aktueller Speicher: {current / 1024 / 1024:.2f} MB"

        tracemalloc.stop()

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_profiling_garbage_collection(self):
        """Test Memory-Profiling mit Garbage Collection."""
        import gc

        process = psutil.Process(os.getpid())

        def create_large_objects():
            """Erstellt große Objekte und führt Garbage Collection durch."""
            # Große Objekte erstellen
            large_arrays = [np.random.randn(1000, 100) for _ in range(10)]

            # Memory vor GC
            memory_before_gc = process.memory_info().rss / 1024 / 1024

            # Garbage Collection
            gc.collect()

            # Memory nach GC
            memory_after_gc = process.memory_info().rss / 1024 / 1024

            return memory_before_gc, memory_after_gc

        memory_before, memory_after = create_large_objects()
        memory_freed = memory_before - memory_after

        # Garbage Collection sollte Speicher freigeben
        assert memory_freed > 0, "Garbage Collection hat keinen Speicher freigegeben"

        # Speicher sollte effizient freigegeben werden
        assert (
            memory_freed > 50
        ), f"Zu wenig Speicher freigegeben: {memory_freed:.2f} MB"


class TestMemoryOptimization:
    """Tests für Memory-Optimierungen."""

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_efficient_processing(self, large_dataset):
        """Test memory-effiziente Verarbeitung."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        # Chunk-basierte Verarbeitung
        chunk_size = 1000
        results = []

        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset.iloc[i : i + chunk_size]

            # Verarbeitung des Chunks
            processor = WiFiDataProcessor()
            processed_chunk = processor.process_data(chunk)
            results.append(processed_chunk)

            # Speicher nach jedem Chunk
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - memory_before

            # Speicher sollte nicht kontinuierlich steigen
            assert (
                memory_increase < 100
            ), f"Memory-Leak in Chunk {i//chunk_size}: {memory_increase:.2f} MB"

        # Ergebnisse zusammenführen
        final_result = pd.concat(results, ignore_index=True)

        # Finale Speichernutzung
        memory_after = process.memory_info().rss / 1024 / 1024
        total_memory_increase = memory_after - memory_before

        assert (
            total_memory_increase < 200
        ), f"Zu viel Speicher für chunk-basierte Verarbeitung: {total_memory_increase:.2f} MB"
        assert len(final_result) == len(large_dataset)

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_efficient_clustering(self):
        """Test memory-effizientes Clustering mit MiniBatchKMeans."""
        from sklearn.cluster import MiniBatchKMeans

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        # Große Daten
        X = np.random.randn(50000, 50)

        # MiniBatchKMeans für memory-effizientes Clustering
        kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1000, random_state=42)
        labels = kmeans.fit_predict(X)

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        # MiniBatchKMeans sollte memory-effizient sein
        assert (
            memory_increase < 300
        ), f"Zu viel Speicher für MiniBatchKMeans: {memory_increase:.2f} MB"
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 10

    @pytest.mark.memory
    @pytest.mark.performance
    def test_memory_usage_data_types(self):
        """Test Speichernutzung verschiedener Datentypen."""
        process = psutil.Process(os.getpid())

        # Test verschiedene Datentypen
        data_types = [
            ("float32", np.float32),
            ("float64", np.float64),
            ("int32", np.int32),
            ("int64", np.int64),
        ]

        memory_usage = {}

        for dtype_name, dtype in data_types:
            memory_before = process.memory_info().rss / 1024 / 1024

            # Große Array mit spezifischem Datentyp
            array = np.random.randn(10000, 100).astype(dtype)

            memory_after = process.memory_info().rss / 1024 / 1024
            memory_usage[dtype_name] = memory_after - memory_before

        # float32 sollte weniger Speicher verwenden als float64
        assert memory_usage["float32"] < memory_usage["float64"]

        # int32 sollte weniger Speicher verwenden als int64
        assert memory_usage["int32"] < memory_usage["int64"]

        # Alle Datentypen sollten angemessenen Speicher verwenden
        for dtype_name, usage in memory_usage.items():
            assert usage < 100, f"Zu viel Speicher für {dtype_name}: {usage:.2f} MB"
