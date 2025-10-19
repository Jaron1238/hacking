"""
Tests für das Ensemble Models Plugin.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

from plugins.ensemble_models.plugin import Plugin
from wlan_tool.storage.state import WifiAnalysisState, ClientState


class TestEnsembleModelsPlugin:
    """Test-Klasse für das Ensemble Models Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
    @pytest.fixture
    def mock_state(self):
        """Mock State mit Test-Clients."""
        state = WifiAnalysisState()
        
        # Erstelle Test-Clients für verschiedene Gerätetypen (mindestens 5 pro Typ für Stratification)
        # Verwende Vendor-Namen, die mit der Heuristik übereinstimmen
        device_configs = [
            # Smartphones (>50 probes)
            ("aa:bb:cc:dd:ee:00", "Apple", 60, "smartphone"),
            ("aa:bb:cc:dd:ee:01", "Samsung", 55, "smartphone"),
            ("aa:bb:cc:dd:ee:02", "Huawei", 65, "smartphone"),
            ("aa:bb:cc:dd:ee:03", "Xiaomi", 70, "smartphone"),
            ("aa:bb:cc:dd:ee:04", "OnePlus", 80, "smartphone"),
            ("aa:bb:cc:dd:ee:05", "Apple", 75, "smartphone"),
            ("aa:bb:cc:dd:ee:06", "Samsung", 85, "smartphone"),
            ("aa:bb:cc:dd:ee:07", "Huawei", 90, "smartphone"),
            # Laptops (Microsoft, Dell, Lenovo, HP)
            ("aa:bb:cc:dd:ee:08", "Microsoft", 20, "laptop"),
            ("aa:bb:cc:dd:ee:09", "Dell", 15, "laptop"),
            ("aa:bb:cc:dd:ee:0a", "Lenovo", 25, "laptop"),
            ("aa:bb:cc:dd:ee:0b", "HP", 18, "laptop"),
            ("aa:bb:cc:dd:ee:0c", "Microsoft", 22, "laptop"),
            ("aa:bb:cc:dd:ee:0d", "Dell", 16, "laptop"),
            ("aa:bb:cc:dd:ee:0e", "Lenovo", 24, "laptop"),
            ("aa:bb:cc:dd:ee:0f", "HP", 19, "laptop"),
            # IoT Devices (<5 probes)
            ("aa:bb:cc:dd:ee:10", "Google", 2, "iot_device"),
            ("aa:bb:cc:dd:ee:11", "Amazon", 3, "iot_device"),
            ("aa:bb:cc:dd:ee:12", "Google", 1, "iot_device"),
            ("aa:bb:cc:dd:ee:13", "Amazon", 4, "iot_device"),
            ("aa:bb:cc:dd:ee:14", "Google", 2, "iot_device"),
            ("aa:bb:cc:dd:ee:15", "Amazon", 3, "iot_device"),
            # Routers (TP-Link, Netgear)
            ("aa:bb:cc:dd:ee:16", "TP-Link", 10, "router"),
            ("aa:bb:cc:dd:ee:17", "Netgear", 8, "router"),
            ("aa:bb:cc:dd:ee:18", "TP-Link", 12, "router"),
            ("aa:bb:cc:dd:ee:19", "Netgear", 9, "router"),
        ]
        
        for mac, vendor, probe_count, device_type in device_configs:
            client = ClientState(mac=mac)
            client.vendor = vendor
            client.device_type = device_type
            client.probe_requests = [f"SSID_{i}" for i in range(probe_count)]
            client.first_seen = 1000.0
            client.last_seen = 1100.0
            client.rssi_history = [-50, -55, -60]
            client.information_elements = {0: [f"SSID_{i}" for i in range(probe_count)]}
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
        assert metadata.name == "Ensemble Models"
        assert metadata.version == "1.0.0"
        assert "Ensemble" in metadata.description
        assert "sklearn" in metadata.dependencies
    
    def test_extract_features_for_classification(self, plugin, mock_state, mock_events):
        """Test Feature-Extraktion für Klassifizierung."""
        X, y, client_macs = plugin._extract_features_for_classification(mock_state, mock_events)
        
        assert len(X) == 6  # 6 Test-Clients
        assert len(y) == 6
        assert len(client_macs) == 6
        assert X.shape[1] > 0  # Mindestens ein Feature
        
        # Überprüfe, dass alle Features numerisch sind
        assert np.isfinite(X).all()
        
        # Überprüfe Label-Mapping
        unique_labels = set(y)
        assert len(unique_labels) > 1  # Sollte verschiedene Gerätetypen haben
    
    def test_device_type_classification(self, plugin, mock_state, mock_events):
        """Test Gerätetyp-Klassifizierung."""
        X, y, client_macs = plugin._extract_features_for_classification(mock_state, mock_events)
        
        # Überprüfe, dass verschiedene Gerätetypen erkannt werden
        device_types = ['smartphone', 'laptop', 'tablet', 'iot_device', 'router', 'unknown']
        unique_labels = set(y)
        
        # Sollte mindestens 2 verschiedene Gerätetypen haben
        assert len(unique_labels) >= 2
        assert all(0 <= label < len(device_types) for label in unique_labels)
    
    def test_ensemble_model_builder_creation(self, plugin):
        """Test Ensemble Model Builder."""
        builder = plugin._EnsembleModelBuilder()
        
        # Test Basis-Modelle
        base_models = builder.create_base_models()
        assert len(base_models) > 0
        assert 'random_forest' in base_models
        assert 'logistic_regression' in base_models
        
        # Test Voting Ensembles
        hard_voting, soft_voting = builder.create_voting_ensemble(base_models)
        assert hard_voting is not None
        assert soft_voting is not None
        
        # Test Bagging
        bagging = builder.create_bagging_ensemble(base_models['random_forest'])
        assert bagging is not None
        
        # Test Boosting
        boosting_models = builder.create_boosting_ensemble()
        assert len(boosting_models) > 0
    
    def test_model_performance_evaluation(self, plugin, mock_state, mock_events):
        """Test Modell-Performance-Evaluation."""
        X, y, client_macs = plugin._extract_features_for_classification(mock_state, mock_events)
        
        # Erstelle ein einfaches Mock-Modell
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        
        # Evaluiere Performance
        perf = plugin._evaluate_model_performance(model, X, y, "test_model")
        
        assert perf.name == "test_model"
        assert 0 <= perf.accuracy <= 1
        assert 0 <= perf.precision <= 1
        assert 0 <= perf.recall <= 1
        assert 0 <= perf.f1_score <= 1
        assert perf.training_time >= 0
        assert perf.prediction_time >= 0
    
    def test_plugin_run_with_sufficient_data(self, plugin, mock_state, mock_events, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit ausreichenden Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('joblib.dump') as mock_dump:
            with patch('builtins.open', create=True) as mock_open:
                plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
        
        # Überprüfe, dass Ergebnisse gespeichert wurden
        mock_dump.assert_called()
        mock_open.assert_called()
    
    def test_plugin_run_with_insufficient_data(self, plugin, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit unzureichenden Daten."""
        # Leerer State mit nur wenigen Clients
        empty_state = WifiAnalysisState()
        for i in range(3):  # Nur 3 Clients (weniger als 10)
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.vendor = f"Vendor_{i}"
            client.probe_requests = [f"SSID_{i}"]
            client.first_seen = 1000.0
            client.last_seen = 1100.0
            empty_state.clients[mac] = client
        
        empty_events = []
        temp_outdir.mkdir(exist_ok=True)
        
        plugin.run(empty_state, empty_events, mock_console, temp_outdir)
        
        # Überprüfe Warnung für unzureichende Daten
        warning_calls = [call for call in mock_console.print.call_args_list 
                        if "Nicht genügend Daten" in str(call)]
        assert len(warning_calls) > 0
    
    def test_feature_extraction_edge_cases(self, plugin):
        """Test Feature-Extraktion mit Edge Cases."""
        state = WifiAnalysisState()
        
        # Client ohne vendor
        client1 = ClientState(mac="aa:bb:cc:dd:ee:01")
        client1.probe_requests = []
        client1.first_seen = 1000.0
        client1.last_seen = 1000.0  # Gleiche Zeit
        state.clients["aa:bb:cc:dd:ee:01"] = client1
        
        # Client ohne rssi_history
        client2 = ClientState(mac="aa:bb:cc:dd:ee:02")
        client2.vendor = "Test"
        client2.probe_requests = ["SSID1"]
        client2.first_seen = 1000.0
        client2.last_seen = 1100.0
        state.clients["aa:bb:cc:dd:ee:02"] = client2
        
        X, y, client_macs = plugin._extract_features_for_classification(state, [])
        
        assert len(X) == 2
        assert len(y) == 2
        assert len(client_macs) == 2
        assert np.isfinite(X).all()
    
    def test_performance_visualization_creation(self, plugin, temp_outdir):
        """Test Erstellung der Performance-Visualisierung."""
        temp_outdir.mkdir(exist_ok=True)
        
        # Mock Performance-Metriken
        from plugins.ensemble_models.plugin import ModelPerformance
        performance_metrics = [
            ModelPerformance("Model1", 0.8, 0.75, 0.8, 0.77, 0.78, 0.02, 1.0, 0.1),
            ModelPerformance("Model2", 0.85, 0.82, 0.85, 0.83, 0.84, 0.01, 1.5, 0.15),
        ]
        
        with patch('plotly.graph_objects.go') as mock_go:
            with patch('plotly.subplots.make_subplots') as mock_subplots:
                mock_fig = MagicMock()
                mock_subplots.return_value = mock_fig
                
                plugin._create_performance_visualization(performance_metrics, temp_outdir)
                
                # Überprüfe, dass Plotly-Funktionen aufgerufen wurden
                mock_subplots.assert_called_once()
                mock_fig.write_html.assert_called_once()
    
    def test_ensemble_model_builder_integration(self, plugin, mock_state, mock_events):
        """Test Integration des Ensemble Model Builders."""
        X, y, client_macs = plugin._extract_features_for_classification(mock_state, mock_events)
        
        # Erstelle Builder
        builder = plugin._EnsembleModelBuilder()
        
        # Test alle Builder-Funktionen
        base_models = builder.create_base_models()
        assert len(base_models) > 0
        
        hard_voting, soft_voting = builder.create_voting_ensemble(base_models)
        assert hard_voting is not None
        assert soft_voting is not None
        
        stacking = builder.create_stacking_ensemble(base_models)
        # Stacking kann None sein wenn sklearn.ensemble.StackingClassifier nicht verfügbar
        
        bagging = builder.create_bagging_ensemble(list(base_models.values())[0])
        assert bagging is not None
        
        boosting_models = builder.create_boosting_ensemble()
        assert len(boosting_models) > 0