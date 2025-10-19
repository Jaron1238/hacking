"""
End-to-End Integration Tests für das WLAN-Tool.
Testet den kompletten Workflow von Datenverarbeitung bis zur Analyse.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from wlan_tool.analysis import ClusteringAnalyzer, DeviceClassifier
from wlan_tool.data_processing import FeatureExtractor, WiFiDataProcessor
from wlan_tool.visualization import WiFiVisualizer


class TestEndToEndWorkflow:
    """End-to-End Tests für den kompletten WLAN-Analyse-Workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_wifi_analysis_workflow(self, large_dataset, temp_dir):
        """Test des kompletten WLAN-Analyse-Workflows."""
        # 1. Datenverarbeitung
        processor = WiFiDataProcessor()
        processed_data = processor.process_data(large_dataset)

        assert len(processed_data) == len(large_dataset)
        assert "processed_timestamp" in processed_data.columns

        # 2. Feature-Extraktion
        extractor = FeatureExtractor()
        features = extractor.extract_features(processed_data)

        assert features.shape[0] == len(processed_data)
        assert features.shape[1] > 0

        # 3. Clustering-Analyse
        analyzer = ClusteringAnalyzer()
        labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)

        assert len(labels) == len(features)
        assert all(isinstance(label, (int, np.integer)) for label in labels)

        # 4. Geräte-Klassifikation
        classifier = DeviceClassifier()
        device_labels = np.random.randint(0, 3, len(features))  # Simulierte Labels
        classifier.train(features, device_labels)
        predictions = classifier.predict(features)

        assert len(predictions) == len(features)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)

        # 5. Visualisierung
        visualizer = WiFiVisualizer()

        # Signal-Stärke-Visualisierung
        signal_plot = visualizer.plot_signal_strength(processed_data)
        assert signal_plot is not None

        # Clustering-Visualisierung
        cluster_plot = visualizer.plot_clusters(features, labels)
        assert cluster_plot is not None

        # 6. Ergebnisse speichern
        results = pd.DataFrame(
            {
                "device_id": processed_data["device_id"],
                "cluster": labels,
                "device_type": predictions,
                "signal_strength": processed_data["signal_strength"],
            }
        )

        output_file = temp_dir / "analysis_results.csv"
        results.to_csv(output_file, index=False)

        assert output_file.exists()
        assert len(results) == len(processed_data)

    @pytest.mark.integration
    def test_plugin_integration_workflow(self, sample_wifi_data, temp_dir):
        """Test der Plugin-Integration im Workflow."""
        from plugins import load_all_plugins

        # Plugins laden
        plugin_dir = Path("plugins")
        plugins = load_all_plugins(plugin_dir)

        assert len(plugins) > 0

        # Features extrahieren
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_wifi_data)

        # Jedes Plugin testen
        for plugin in plugins:
            # Plugin-Dependencies prüfen
            if not plugin.validate_dependencies():
                pytest.skip(f"Plugin {plugin.metadata.name} hat fehlende Dependencies")

            # Plugin ausführen
            try:
                result = plugin.run(features, sample_wifi_data)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Plugin {plugin.metadata.name} fehlgeschlagen: {e}")

    @pytest.mark.integration
    def test_data_pipeline_with_file_io(self, temp_dir):
        """Test der Datenpipeline mit Datei-I/O."""
        # Test-Daten generieren
        n_samples = 1000
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=n_samples, freq="1min"
                ),
                "device_id": np.random.choice(
                    ["device_1", "device_2", "device_3"], n_samples
                ),
                "signal_strength": np.random.normal(-50, 10, n_samples),
                "frequency": np.random.choice([2.4, 5.0], n_samples),
                "channel": np.random.randint(1, 14, n_samples),
                "mac_address": [f"00:11:22:33:44:{i:02x}" for i in range(n_samples)],
                "ssid": np.random.choice(
                    ["WiFi_Home", "WiFi_Office", "Public_WiFi"], n_samples
                ),
                "encryption": np.random.choice(["WPA2", "WPA3", "Open"], n_samples),
                "data_rate": np.random.normal(54, 10, n_samples),
                "packet_count": np.random.poisson(100, n_samples),
                "bytes_transferred": np.random.exponential(1000, n_samples),
            }
        )

        # 1. Daten in CSV speichern
        input_file = temp_dir / "input_data.csv"
        test_data.to_csv(input_file, index=False)

        # 2. Daten laden und verarbeiten
        loaded_data = pd.read_csv(input_file)
        processor = WiFiDataProcessor()
        processed_data = processor.process_data(loaded_data)

        # 3. Features extrahieren
        extractor = FeatureExtractor()
        features = extractor.extract_features(processed_data)

        # 4. Clustering
        analyzer = ClusteringAnalyzer()
        labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=3)

        # 5. Ergebnisse speichern
        results = pd.DataFrame(
            {
                "device_id": processed_data["device_id"],
                "cluster": labels,
                "signal_strength": processed_data["signal_strength"],
            }
        )

        output_file = temp_dir / "output_results.csv"
        results.to_csv(output_file, index=False)

        # 6. Ergebnisse validieren
        assert input_file.exists()
        assert output_file.exists()
        assert len(results) == len(test_data)
        assert len(np.unique(labels)) <= 3

    @pytest.mark.integration
    def test_error_handling_and_recovery(self, temp_dir):
        """Test der Fehlerbehandlung und Wiederherstellung."""
        processor = WiFiDataProcessor()
        extractor = FeatureExtractor()
        analyzer = ClusteringAnalyzer()

        # Test mit ungültigen Daten
        invalid_data = pd.DataFrame(
            {
                "timestamp": ["invalid_date", "2024-01-01", "2024-01-02"],
                "signal_strength": [np.nan, -50, -60],
                "frequency": [1.0, 2.4, 5.0],  # Ungültige Frequenz
                "device_id": ["device_1", "device_2", "device_3"],
                "mac_address": [
                    "00:11:22:33:44:01",
                    "00:11:22:33:44:02",
                    "00:11:22:33:44:03",
                ],
                "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                "encryption": ["WPA2", "WPA3", "Open"],
                "channel": [1, 6, 11],
                "data_rate": [54, 100, 54],
                "packet_count": [100, 200, 150],
                "bytes_transferred": [1000, 2000, 1500],
            }
        )

        # Verarbeitung sollte Fehler behandeln
        try:
            processed_data = processor.process_data(invalid_data)
            # Wenn erfolgreich, sollten ungültige Daten bereinigt sein
            assert len(processed_data) <= len(invalid_data)
        except Exception as e:
            # Fehler sollten informativ sein
            assert "Invalid" in str(e) or "Missing" in str(e) or "Unknown datetime" in str(e)

        # Test mit leeren Daten
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty dataset"):
            processor.process_data(empty_data)

        # Test mit zu wenigen Features für Clustering
        minimal_features = np.random.randn(2, 1)  # Nur 2 Samples, 1 Feature

        with pytest.raises(ValueError, match="Not enough samples"):
            analyzer.cluster_data(minimal_features, algorithm="kmeans", n_clusters=3)

    @pytest.mark.integration
    def test_performance_under_load(self, temp_dir):
        """Test der Performance unter Last."""
        import time

        # Große Datenmenge
        n_samples = 50000
        large_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1s"),
                "device_id": np.random.choice(
                    [f"device_{i}" for i in range(100)], n_samples
                ),
                "signal_strength": np.random.normal(-50, 10, n_samples),
                "frequency": np.random.choice([2.4, 5.0], n_samples),
                "channel": np.random.randint(1, 14, n_samples),
                "mac_address": [f"00:11:22:33:44:{i:02x}" for i in range(n_samples)],
                "ssid": np.random.choice(
                    ["WiFi_Home", "WiFi_Office", "Public_WiFi"], n_samples
                ),
                "encryption": np.random.choice(["WPA2", "WPA3", "Open"], n_samples),
                "data_rate": np.random.normal(54, 10, n_samples),
                "packet_count": np.random.poisson(100, n_samples),
                "bytes_transferred": np.random.exponential(1000, n_samples),
            }
        )

        processor = WiFiDataProcessor()
        extractor = FeatureExtractor()
        analyzer = ClusteringAnalyzer()

        # Performance-Messung
        start_time = time.time()

        # Verarbeitung
        processed_data = processor.process_data(large_data)
        features = extractor.extract_features(processed_data)
        labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=5)

        end_time = time.time()
        execution_time = end_time - start_time

        # Performance sollte angemessen sein (unter 30 Sekunden)
        assert execution_time < 30, f"Verarbeitung zu langsam: {execution_time:.2f}s"

        # Ergebnisse sollten korrekt sein
        assert len(labels) == len(features)
        assert len(np.unique(labels)) <= 5

    @pytest.mark.integration
    def test_concurrent_processing(self, temp_dir):
        """Test der parallelen Verarbeitung."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        def process_chunk(chunk_data):
            """Verarbeitet einen Daten-Chunk."""
            processor = WiFiDataProcessor()
            extractor = FeatureExtractor()
            analyzer = ClusteringAnalyzer()

            processed_data = processor.process_data(chunk_data)
            features = extractor.extract_features(processed_data)
            labels = analyzer.cluster_data(features, algorithm="kmeans", n_clusters=3)

            return {
                "chunk_id": threading.current_thread().ident,
                "n_samples": len(chunk_data),
                "n_clusters": len(np.unique(labels)),
            }

        # Daten in Chunks aufteilen
        n_chunks = 4
        chunk_size = 1000
        n_samples = n_chunks * chunk_size

        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=n_samples, freq="1min"
                ),
                "device_id": np.random.choice(
                    [f"device_{i}" for i in range(50)], n_samples
                ),
                "signal_strength": np.random.normal(-50, 10, n_samples),
                "frequency": np.random.choice([2.4, 5.0], n_samples),
                "channel": np.random.randint(1, 14, n_samples),
                "mac_address": [f"00:11:22:33:44:{i:02x}" for i in range(n_samples)],
                "ssid": np.random.choice(
                    ["WiFi_Home", "WiFi_Office", "Public_WiFi"], n_samples
                ),
                "encryption": np.random.choice(["WPA2", "WPA3", "Open"], n_samples),
                "data_rate": np.random.normal(54, 10, n_samples),
                "packet_count": np.random.poisson(100, n_samples),
                "bytes_transferred": np.random.exponential(1000, n_samples),
            }
        )

        chunks = [
            test_data.iloc[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_chunks)
        ]

        # Parallele Verarbeitung
        with ThreadPoolExecutor(max_workers=n_chunks) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]

        # Ergebnisse validieren
        assert len(results) == n_chunks
        for result in results:
            assert result["n_samples"] == chunk_size
            assert result["n_clusters"] <= 3
            assert result["chunk_id"] is not None
