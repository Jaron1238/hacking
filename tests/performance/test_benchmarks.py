"""
Performance-Benchmark-Tests für das WLAN-Tool.
"""

import time

import numpy as np
import pandas as pd

import pytest
from wlan_tool.analysis import ClusteringAnalyzer, DeviceClassifier
from wlan_tool.data_processing import FeatureExtractor, WiFiDataProcessor


class TestClusteringBenchmarks:
    """Benchmark-Tests für Clustering-Algorithmen."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_kmeans_performance(self, benchmark, large_dataset):
        """Benchmark für K-Means Clustering."""
        analyzer = ClusteringAnalyzer()

        # Features extrahieren
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(large_dataset)

        def clustering_func():
            return analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)

        result = benchmark(clustering_func)

        # Performance-Assertions
        assert result.stats.mean < 2.0  # Sollte unter 2 Sekunden sein
        assert result.stats.std < 0.5  # Geringe Varianz

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_dbscan_performance(self, benchmark, large_dataset):
        """Benchmark für DBSCAN Clustering."""
        analyzer = ClusteringAnalyzer()

        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(large_dataset)

        def clustering_func():
            return analyzer.cluster_data(
                features, algorithm="dbscan", eps=0.5, min_samples=5
            )

        result = benchmark(clustering_func)

        # DBSCAN kann langsamer sein
        assert result.stats.mean < 5.0
        assert result.stats.std < 1.0

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_hierarchical_performance(self, benchmark, large_dataset):
        """Benchmark für Hierarchical Clustering."""
        analyzer = ClusteringAnalyzer()

        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(large_dataset)

        def clustering_func():
            return analyzer.cluster_data(
                features, algorithm="hierarchical", n_clusters=5
            )

        result = benchmark(clustering_func)

        # Hierarchical Clustering ist O(n²), daher langsamer
        assert result.stats.mean < 10.0
        assert result.stats.std < 2.0

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_clustering_scalability(self, benchmark):
        """Test Skalierbarkeit mit verschiedenen Datengrößen."""
        sizes = [1000, 5000, 10000, 20000]
        times = []

        for size in sizes:
            # Generiere Test-Daten
            X = np.random.randn(size, 20)
            analyzer = ClusteringAnalyzer()

            def clustering_func():
                return analyzer.cluster_data(X, algorithm="kmeans", n_clusters=5)

            result = benchmark(clustering_func)
            times.append(result.stats.mean)

        # Prüfe, dass Skalierung linear ist (O(n))
        for i in range(1, len(times)):
            ratio = times[i] / times[i - 1]
            size_ratio = sizes[i] / sizes[i - 1]

            # Performance-Ratio sollte nicht viel schlechter als Größen-Ratio sein
            assert (
                ratio < size_ratio * 2.0
            ), f"Skalierbarkeit problematisch bei {sizes[i]} Samples"


class TestDataProcessingBenchmarks:
    """Benchmark-Tests für Datenverarbeitung."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_data_processing_performance(self, benchmark, large_dataset):
        """Benchmark für Datenverarbeitung."""
        processor = WiFiDataProcessor()

        def processing_func():
            return processor.process_data(large_dataset)

        result = benchmark(processing_func)

        # Datenverarbeitung sollte schnell sein
        assert result.stats.mean < 1.0
        assert result.stats.std < 0.2

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_feature_extraction_performance(self, benchmark, large_dataset):
        """Benchmark für Feature-Extraktion."""
        extractor = FeatureExtractor()

        def extraction_func():
            return extractor.extract_features(large_dataset)

        result = benchmark(extraction_func)

        # Feature-Extraktion sollte effizient sein
        assert result.stats.mean < 2.0
        assert result.stats.std < 0.5

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_feature_scaling_performance(self, benchmark):
        """Benchmark für Feature-Skalierung."""
        extractor = FeatureExtractor()

        # Große Feature-Matrix
        features = np.random.randn(50000, 100)

        def scaling_func():
            return extractor.scale_features(features)

        result = benchmark(scaling_func)

        # Skalierung sollte sehr schnell sein
        assert result.stats.mean < 0.5
        assert result.stats.std < 0.1


class TestClassificationBenchmarks:
    """Benchmark-Tests für Klassifikation."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_random_forest_performance(self, benchmark, large_dataset):
        """Benchmark für Random Forest Klassifikation."""
        classifier = DeviceClassifier()

        # Features und Labels generieren
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(large_dataset)
        labels = np.random.randint(0, 5, len(features))

        def classification_func():
            classifier.train(features, labels)
            return classifier.predict(features)

        result = benchmark(classification_func)

        # Random Forest sollte effizient sein
        assert result.stats.mean < 3.0
        assert result.stats.std < 0.5

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_svm_performance(self, benchmark, large_dataset):
        """Benchmark für SVM Klassifikation."""
        classifier = DeviceClassifier(algorithm="svm")

        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(large_dataset)
        labels = np.random.randint(0, 5, len(features))

        def classification_func():
            classifier.train(features, labels)
            return classifier.predict(features)

        result = benchmark(classification_func)

        # SVM kann langsamer sein
        assert result.stats.mean < 5.0
        assert result.stats.std < 1.0


class TestMemoryBenchmarks:
    """Memory-Benchmark-Tests."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.memory
    def test_memory_usage_clustering(self, benchmark, large_dataset):
        """Benchmark für Speichernutzung beim Clustering."""
        analyzer = ClusteringAnalyzer()
        feature_extractor = FeatureExtractor()

        def memory_intensive_func():
            # Features extrahieren
            features = feature_extractor.extract_features(large_dataset)

            # Clustering durchführen
            labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)

            # Zusätzliche Berechnungen
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            return {
                "n_clusters": len(unique_labels),
                "cluster_sizes": cluster_sizes,
                "features_shape": features.shape,
            }

        result = benchmark(memory_intensive_func)

        # Memory-Intensive Operationen sollten nicht zu langsam sein
        assert result.stats.mean < 5.0
        assert result.stats.std < 1.0

    @pytest.mark.benchmark
    @pytest.mark.performance
    @pytest.mark.memory
    def test_memory_usage_large_dataset(self, benchmark):
        """Benchmark für Speichernutzung mit sehr großen Datensätzen."""
        # Sehr großer Datensatz
        n_samples = 100000
        n_features = 50

        def large_dataset_func():
            # Große Daten generieren
            X = np.random.randn(n_samples, n_features)

            # Clustering mit reduzierter Komplexität
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1000)
            labels = kmeans.fit_predict(X)

            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_clusters": len(np.unique(labels)),
            }

        result = benchmark(large_dataset_func)

        # Auch bei großen Datensätzen sollte es in angemessener Zeit laufen
        assert result.stats.mean < 10.0
        assert result.stats.std < 2.0


class TestConcurrentBenchmarks:
    """Benchmark-Tests für parallele Verarbeitung."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_parallel_processing_performance(self, benchmark):
        """Benchmark für parallele Verarbeitung."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        def process_chunk(chunk_data):
            """Verarbeitet einen Daten-Chunk."""
            analyzer = ClusteringAnalyzer()
            return analyzer.cluster_data(chunk_data, algorithm="kmeans", n_clusters=3)

        def parallel_processing_func():
            # Daten in Chunks aufteilen
            n_chunks = 4
            chunk_size = 2500
            X = np.random.randn(n_chunks * chunk_size, 20)
            chunks = [X[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]

            # Parallele Verarbeitung
            with ThreadPoolExecutor(max_workers=n_chunks) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                results = [future.result() for future in futures]

            return results

        result = benchmark(parallel_processing_func)

        # Parallele Verarbeitung sollte effizienter sein
        assert result.stats.mean < 3.0
        assert result.stats.std < 0.5


class TestIOBenchmarks:
    """Benchmark-Tests für I/O-Operationen."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_file_io_performance(self, benchmark, temp_dir):
        """Benchmark für Datei-I/O."""
        import pickle

        # Große Daten generieren
        large_data = np.random.randn(10000, 50)
        file_path = temp_dir / "test_data.pkl"

        def io_func():
            # Speichern
            with open(file_path, "wb") as f:
                pickle.dump(large_data, f)

            # Laden
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)

            return loaded_data.shape

        result = benchmark(io_func)

        # I/O sollte schnell sein
        assert result.stats.mean < 1.0
        assert result.stats.std < 0.2

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_csv_io_performance(self, benchmark, temp_dir, large_dataset):
        """Benchmark für CSV-I/O."""
        csv_path = temp_dir / "test_data.csv"

        def csv_io_func():
            # Speichern
            large_dataset.to_csv(csv_path, index=False)

            # Laden
            loaded_data = pd.read_csv(csv_path)

            return loaded_data.shape

        result = benchmark(csv_io_func)

        # CSV-I/O kann langsamer sein
        assert result.stats.mean < 2.0
        assert result.stats.std < 0.5
