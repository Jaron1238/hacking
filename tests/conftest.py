"""
Pytest-Konfiguration und gemeinsame Fixtures für alle Tests.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd

import pytest


# Test-Daten Fixtures
@pytest.fixture
def sample_wifi_data() -> pd.DataFrame:
    """Erstellt Beispieldaten für WLAN-Tests."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
        "device_id": np.random.choice(["device_1", "device_2", "device_3"], n_samples),
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

    return pd.DataFrame(data)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Erstellt Feature-Matrix für ML-Tests."""
    np.random.seed(42)
    return np.random.randn(1000, 20)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Erstellt Beispiel-Labels für Clustering-Tests."""
    np.random.seed(42)
    return np.random.randint(0, 5, 1000)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Erstellt temporäres Verzeichnis für Tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_pcap_file(temp_dir: Path) -> Path:
    """Erstellt Mock-PCAP-Datei für Tests."""
    pcap_file = temp_dir / "test_capture.pcap"
    pcap_file.write_bytes(b"Mock PCAP data")
    return pcap_file


# Plugin-spezifische Fixtures
@pytest.fixture
def clustering_analyzer():
    """Mock für ClusteringAnalyzer."""
    from wlan_tool.analysis.clustering import ClusteringAnalyzer

    analyzer = ClusteringAnalyzer()
    return analyzer


@pytest.fixture
def device_classifier():
    """Mock für DeviceClassifier."""
    from wlan_tool.analysis.device_classification import DeviceClassifier

    classifier = DeviceClassifier()
    return classifier


@pytest.fixture
def ensemble_model():
    """Mock für EnsembleModel."""
    from wlan_tool.analysis.ensemble import EnsembleModel

    model = EnsembleModel()
    return model


# Performance-Test Fixtures
@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Großer Datensatz für Performance-Tests."""
    np.random.seed(42)
    n_samples = 10000

    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1s"),
        "device_id": np.random.choice([f"device_{i}" for i in range(100)], n_samples),
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

    return pd.DataFrame(data)


@pytest.fixture
def mock_sklearn_model():
    """Mock für sklearn-Modelle."""
    mock_model = Mock()
    mock_model.fit.return_value = mock_model
    mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])
    mock_model.score.return_value = 0.85
    return mock_model


# Property-based Testing
@pytest.fixture
def wifi_data_strategy():
    """Hypothesis-Strategie für WLAN-Daten."""
    from hypothesis import strategies as st
    from hypothesis.extra.pandas import column, data_frames, range_indexes

    return data_frames(
        index=range_indexes(min_size=1, max_size=1000),
        columns=[
            column("signal_strength", dtype=float, elements=st.floats(-100, 0)),
            column("frequency", dtype=float, elements=st.sampled_from([2.4, 5.0])),
            column("channel", dtype=int, elements=st.integers(1, 14)),
            column("ssid", dtype=str, elements=st.text(min_size=1, max_size=32)),
            column(
                "encryption",
                dtype=str,
                elements=st.sampled_from(["WPA2", "WPA3", "Open"]),
            ),
        ],
    )


# Mock-Objekte für externe Abhängigkeiten
@pytest.fixture
def mock_rich_console():
    """Mock für Rich Console."""
    console = Mock()
    console.print = Mock()
    console.log = Mock()
    return console


@pytest.fixture
def mock_logger():
    """Mock für Logger."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


# Test-Marker
def pytest_configure(config):
    """Konfiguriert pytest-Marker."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "plugin: Plugin tests")
    config.addinivalue_line("markers", "clustering: Clustering algorithm tests")
    config.addinivalue_line("markers", "ensemble: Ensemble model tests")
    config.addinivalue_line("markers", "rl: Reinforcement learning tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "memory: Memory profiling tests")
    config.addinivalue_line("markers", "load: Load testing")
    config.addinivalue_line("markers", "security: Security tests")


def pytest_collection_modifyitems(config, items):
    """Modifiziert Test-Items basierend auf Markern."""
    for item in items:
        # Markiere langsame Tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Markiere Performance-Tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # Markiere Memory-Tests
        if "memory" in item.nodeid:
            item.add_marker(pytest.mark.memory)
