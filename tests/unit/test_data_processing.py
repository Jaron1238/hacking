"""
Unit Tests für Datenverarbeitungs-Module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

import pytest
from wlan_tool.data_processing import DataValidator, FeatureExtractor, WiFiDataProcessor


class TestWiFiDataProcessor:
    """Tests für WiFiDataProcessor."""

    def test_init(self):
        """Test Initialisierung."""
        processor = WiFiDataProcessor()
        assert processor is not None
        assert hasattr(processor, "process_data")

    def test_process_data_basic(self, sample_wifi_data):
        """Test grundlegende Datenverarbeitung."""
        processor = WiFiDataProcessor()
        result = processor.process_data(sample_wifi_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "processed_timestamp" in result.columns

    def test_process_data_empty(self):
        """Test mit leerem DataFrame."""
        processor = WiFiDataProcessor()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty dataset"):
            processor.process_data(empty_df)

    def test_process_data_invalid_columns(self):
        """Test mit ungültigen Spalten."""
        processor = WiFiDataProcessor()
        invalid_df = pd.DataFrame({"invalid_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            processor.process_data(invalid_df)

    @pytest.mark.parametrize(
        "signal_strength,expected",
        [
            (-30, "excellent"),
            (-50, "good"),
            (-70, "fair"),
            (-90, "poor"),
        ],
    )
    def test_signal_quality_classification(self, signal_strength, expected):
        """Test Signal-Qualitäts-Klassifikation."""
        processor = WiFiDataProcessor()
        quality = processor._classify_signal_quality(signal_strength)
        assert quality == expected

    def test_handle_missing_values(self):
        """Test Behandlung fehlender Werte."""
        processor = WiFiDataProcessor()

        data_with_nulls = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5),
                "signal_strength": [-50, -60, np.nan, -70, -80],
                "frequency": [2.4, 5.0, 2.4, np.nan, 5.0],
                "device_id": [
                    "device_1",
                    "device_2",
                    "device_3",
                    "device_4",
                    "device_5",
                ],
                "mac_address": [
                    "00:11:22:33:44:01",
                    "00:11:22:33:44:02",
                    "00:11:22:33:44:03",
                    "00:11:22:33:44:04",
                    "00:11:22:33:44:05",
                ],
                "ssid": [
                    "WiFi_Home",
                    "WiFi_Office",
                    "Public_WiFi",
                    "WiFi_Home",
                    "WiFi_Office",
                ],
                "encryption": ["WPA2", "WPA3", "Open", "WPA2", "WPA3"],
                "channel": [1, 6, 11, 1, 6],
                "data_rate": [54, 100, 54, 100, 54],
                "packet_count": [100, 200, 150, 300, 250],
                "bytes_transferred": [1000, 2000, 1500, 3000, 2500],
            }
        )

        result = processor.process_data(data_with_nulls)

        # Prüfe, dass keine NaN-Werte in kritischen Spalten sind
        assert not result["signal_strength"].isna().any()
        assert not result["frequency"].isna().any()


class TestFeatureExtractor:
    """Tests für FeatureExtractor."""

    def test_init(self):
        """Test Initialisierung."""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_features")

    def test_extract_basic_features(self, sample_wifi_data):
        """Test Extraktion grundlegender Features."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_wifi_data)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_wifi_data)
        assert features.shape[1] > 0

    def test_extract_time_features(self, sample_wifi_data):
        """Test Extraktion von Zeit-Features."""
        extractor = FeatureExtractor()
        features = extractor.extract_time_features(sample_wifi_data)

        assert "hour" in features.columns
        assert "day_of_week" in features.columns
        assert "is_weekend" in features.columns

    def test_extract_signal_features(self, sample_wifi_data):
        """Test Extraktion von Signal-Features."""
        extractor = FeatureExtractor()
        features = extractor.extract_signal_features(sample_wifi_data)

        assert "signal_mean" in features.columns
        assert "signal_std" in features.columns
        assert "signal_min" in features.columns
        assert "signal_max" in features.columns

    def test_extract_network_features(self, sample_wifi_data):
        """Test Extraktion von Netzwerk-Features."""
        extractor = FeatureExtractor()
        features = extractor.extract_network_features(sample_wifi_data)

        assert "unique_ssids" in features.columns
        assert "encryption_types" in features.columns
        assert "channel_diversity" in features.columns

    def test_feature_scaling(self, sample_features):
        """Test Feature-Skalierung."""
        extractor = FeatureExtractor()
        scaled_features = extractor.scale_features(sample_features)

        # Prüfe, dass Features skaliert sind (Mittelwert ≈ 0, Std ≈ 1)
        assert np.allclose(scaled_features.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(scaled_features.std(axis=0), 1, atol=1e-6)


class TestDataValidator:
    """Tests für DataValidator."""

    def test_init(self):
        """Test Initialisierung."""
        validator = DataValidator()
        assert validator is not None
        assert hasattr(validator, "validate")

    def test_validate_valid_data(self, sample_wifi_data):
        """Test Validierung gültiger Daten."""
        validator = DataValidator()
        is_valid, errors = validator.validate(sample_wifi_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_columns(self):
        """Test Validierung mit fehlenden Spalten."""
        validator = DataValidator()
        invalid_data = pd.DataFrame({"invalid_col": [1, 2, 3]})

        is_valid, errors = validator.validate(invalid_data)

        assert not is_valid
        assert len(errors) > 0
        assert any("Missing required columns" in error for error in errors)

    def test_validate_invalid_signal_strength(self):
        """Test Validierung mit ungültigen Signal-Stärken."""
        validator = DataValidator()

        invalid_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "signal_strength": [200, -200, np.nan],  # Ungültige Werte
                "frequency": [2.4, 5.0, 2.4],
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

        is_valid, errors = validator.validate(invalid_data)

        assert not is_valid
        assert any("Invalid signal strength" in error for error in errors)

    def test_validate_invalid_frequency(self):
        """Test Validierung mit ungültigen Frequenzen."""
        validator = DataValidator()

        invalid_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "signal_strength": [-50, -60, -70],
                "frequency": [1.0, 10.0, 2.4],  # Ungültige Frequenzen
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

        is_valid, errors = validator.validate(invalid_data)

        assert not is_valid
        assert any("Invalid frequency" in error for error in errors)

    def test_validate_empty_dataframe(self):
        """Test Validierung mit leerem DataFrame."""
        validator = DataValidator()
        empty_df = pd.DataFrame()

        is_valid, errors = validator.validate(empty_df)

        assert not is_valid
        assert any("Empty dataset" in error for error in errors)

    def test_validate_duplicate_timestamps(self):
        """Test Validierung mit doppelten Timestamps."""
        validator = DataValidator()

        duplicate_data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 10:00:00"]
                ),
                "signal_strength": [-50, -60],
                "frequency": [2.4, 5.0],
                "device_id": ["device_1", "device_2"],
                "mac_address": ["00:11:22:33:44:01", "00:11:22:33:44:02"],
                "ssid": ["WiFi_Home", "WiFi_Office"],
                "encryption": ["WPA2", "WPA3"],
                "channel": [1, 6],
                "data_rate": [54, 100],
                "packet_count": [100, 200],
                "bytes_transferred": [1000, 2000],
            }
        )

        is_valid, errors = validator.validate(duplicate_data)

        # Duplicate timestamps sollten Warnung sein, aber nicht Fehler
        assert is_valid or any("Duplicate timestamps" in error for error in errors)
