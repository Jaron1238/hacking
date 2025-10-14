"""
Tests für das Advanced Clustering Plugin.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

from plugins.clustering_advanced.plugin import Plugin
from wlan_tool.storage.state import WifiAnalysisState, ClientState


class TestAdvancedClusteringPlugin:
    """Test-Klasse für das Advanced Clustering Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
    @pytest.fixture
    def mock_state(self):
        """Mock State mit Test-Clients."""
        state = WifiAnalysisState()
        
        # Erstelle Test-Clients
        for i in range(10):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.probe_requests = [f"SSID_{i}", f"Network_{i}"]
            client.first_seen = 1000.0
            client.last_seen = 1100.0 + i * 10
            client.rssi_history = [-50 - i*2, -55 - i*2, -60 - i*2]
            client.information_elements = {0: [f"SSID_{i}"], 48: ["0100000fac040100000fac020100000fac028c00"]}
            client.vendor = f"Vendor_{i % 3}"
            state.clients[mac] = client
        
        return state
    
    @pytest.fixture
    def mock_events(self):
        """Mock Events für Tests."""
        return [
            {"ts": 1000.0, "type": "probe_req", "client": "aa:bb:cc:dd:ee:00"},
            {"ts": 1001.0, "type": "probe_req", "client": "aa:bb:cc:dd:ee:01"},
        ]
    
    @pytest.fixture
    def mock_console(self):
        """Mock Console für Tests."""
        console = MagicMock()
        console.print = MagicMock()
        return console
    
    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Temporäres Ausgabeverzeichnis."""
        return tmp_path / "output"
    
    def test_plugin_metadata(self, plugin):
        """Test Plugin-Metadaten."""
        metadata = plugin.get_metadata()
        assert metadata.name == "Advanced Clustering"
        assert metadata.version == "1.0.0"
        assert "Clustering" in metadata.description
        assert "sklearn" in metadata.dependencies
    
    def test_extract_features_for_clustering(self, plugin, mock_state, mock_events):
        """Test Feature-Extraktion."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        assert len(features) == 10  # 10 Test-Clients
        assert len(client_macs) == 10
        assert features.shape[1] > 0  # Mindestens ein Feature
        
        # Überprüfe, dass alle Features numerisch sind
        assert np.isfinite(features).all()
    
    def test_spectral_clustering(self, plugin, mock_state, mock_events):
        """Test Spectral Clustering."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        labels, metrics = plugin._run_spectral_clustering(features, n_clusters=3)
        
        assert len(labels) == len(features)
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "Spectral Clustering"
        assert "n_clusters" in metrics
        assert "silhouette_score" in metrics
    
    def test_hierarchical_clustering(self, plugin, mock_state, mock_events):
        """Test Hierarchical Clustering."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        labels, metrics = plugin._run_hierarchical_clustering(features, n_clusters=3)
        
        assert len(labels) == len(features)
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "Hierarchical Clustering"
    
    def test_gaussian_mixture(self, plugin, mock_state, mock_events):
        """Test Gaussian Mixture Model."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        labels, metrics = plugin._run_gaussian_mixture(features, n_clusters=3)
        
        assert len(labels) == len(features)
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "Gaussian Mixture Model"
        assert "aic" in metrics
        assert "bic" in metrics
    
    def test_optics_clustering(self, plugin, mock_state, mock_events):
        """Test OPTICS Clustering."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        labels, metrics = plugin._run_optics_clustering(features)
        
        assert len(labels) == len(features)
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "OPTICS"
    
    def test_hdbscan_clustering(self, plugin, mock_state, mock_events):
        """Test HDBSCAN Clustering."""
        features, client_macs = plugin._extract_features_for_clustering(mock_state, mock_events)
        
        labels, metrics = plugin._run_hdbscan_clustering(features)
        
        assert len(labels) == len(features)
        assert "algorithm" in metrics
        assert metrics["algorithm"] == "HDBSCAN"
        assert "n_noise" in metrics
    
    def test_plugin_run_with_sufficient_data(self, plugin, mock_state, mock_events, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit ausreichenden Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('joblib.dump') as mock_dump:
            plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
        
        # Überprüfe, dass Ergebnisse gespeichert wurden
        mock_dump.assert_called_once()
    
    def test_plugin_run_with_insufficient_data(self, plugin, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit unzureichenden Daten."""
        # Leerer State
        empty_state = WifiAnalysisState()
        empty_events = []
        
        temp_outdir.mkdir(exist_ok=True)
        
        plugin.run(empty_state, empty_events, mock_console, temp_outdir)
        
        # Überprüfe Warnung für unzureichende Daten
        warning_calls = [call for call in mock_console.print.call_args_list 
                        if "Nicht genügend" in str(call)]
        assert len(warning_calls) > 0
    
    def test_clustering_with_edge_cases(self, plugin):
        """Test Clustering mit Edge Cases."""
        # Test mit nur einem Client
        single_client_features = np.array([[1, 2, 3, 4, 5]])
        single_client_macs = ["aa:bb:cc:dd:ee:ff"]
        
        # Alle Algorithmen sollten mit Edge Cases umgehen können
        algorithms = [
            plugin._run_spectral_clustering,
            plugin._run_hierarchical_clustering,
            plugin._run_gaussian_mixture,
            plugin._run_optics_clustering,
            plugin._run_hdbscan_clustering
        ]
        
        for algo_func in algorithms:
            if algo_func in [plugin._run_spectral_clustering, plugin._run_hierarchical_clustering, plugin._run_gaussian_mixture]:
                labels, metrics = algo_func(single_client_features, n_clusters=1)
            else:
                labels, metrics = algo_func(single_client_features)
            
            assert len(labels) == 1
            assert "error" in metrics or "algorithm" in metrics
    
    @pytest.mark.parametrize("n_clients", [0, 1, 2, 5, 10])
    def test_feature_extraction_with_different_client_counts(self, plugin, n_clients):
        """Test Feature-Extraktion mit verschiedenen Client-Anzahlen."""
        state = WifiAnalysisState()
        
        for i in range(n_clients):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.probe_requests = [f"SSID_{i}"]
            client.first_seen = 1000.0
            client.last_seen = 1100.0
            client.rssi_history = [-50, -55, -60]
            state.clients[mac] = client
        
        features, client_macs = plugin._extract_features_for_clustering(state, [])
        
        assert len(features) == n_clients
        assert len(client_macs) == n_clients
        
        if n_clients > 0:
            assert features.shape[1] > 0
            assert np.isfinite(features).all()