#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrationstests für das gesamte WLAN-Analyse-Tool.
"""

import pytest
import tempfile
import time
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from wlan_tool.storage.state import WifiAnalysisState
from wlan_tool.analysis import logic as analysis
from wlan_tool import utils
from wlan_tool.storage import database


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-End-Workflow-Tests."""
    
    def test_complete_analysis_pipeline(self, sample_events, temp_db_file):
        """Test kompletter Analyse-Pipeline."""
        # 1. Erstelle State aus Events
        state = WifiAnalysisState()
        state.build_from_events(sample_events)
        
        assert len(state.aps) > 0
        assert len(state.clients) > 0
        
        # 2. Führe Inferenz durch
        results = analysis.score_pairs_with_recency_and_matching(state)
        assert isinstance(results, list)
        
        # 3. Führe Client-Clustering durch
        clustered_df, feature_df = analysis.cluster_clients(state, n_clusters=2)
        if clustered_df is not None:
            assert len(clustered_df) > 0
        
        # 4. Führe AP-Clustering durch
        ap_clustered_df = analysis.cluster_aps(state, n_clusters=2)
        if ap_clustered_df is not None:
            assert len(ap_clustered_df) > 0
        
        # 5. Teste Graph-Export
        if ap_clustered_df is not None and not ap_clustered_df.empty:
            success = analysis.export_ap_graph(
                state,
                ap_clustered_df,
                {ap: state.aps[ap] for ap in state.aps.keys()},
                {},
                "/tmp/test_integration.gexf",
                include_clients=False
            )
            assert success is True
    
    def test_database_integration(self, sample_events, temp_db_file):
        """Test Datenbank-Integration."""
        # Schreibe Events in Datenbank
        with database.db_conn_ctx(temp_db_file) as conn:
            for event in sample_events:
                database.add_event(conn, event)
        
        # Lade Events aus Datenbank
        with database.db_conn_ctx(temp_db_file) as conn:
            loaded_events = list(database.fetch_events(conn))
        
        assert len(loaded_events) == len(sample_events)
        
        # Erstelle State aus geladenen Events
        state = WifiAnalysisState()
        state.build_from_events(loaded_events)
        
        assert len(state.aps) > 0
        assert len(state.clients) > 0
    
    def test_config_integration(self):
        """Test Konfigurations-Integration."""
        # Teste Konfigurations-Laden
        config_data = utils.load_config()
        
        assert isinstance(config_data, dict)
        assert "capture" in config_data
        assert "database" in config_data
        assert "scanning" in config_data
        
        # Teste spezifische Konfigurationswerte
        assert "interface" in config_data["capture"]
        assert "duration" in config_data["capture"]
        assert "path" in config_data["database"]


@pytest.mark.integration
class TestDataFlow:
    """Tests für Datenfluss zwischen Modulen."""
    
    def test_event_to_state_to_analysis(self, sample_events):
        """Test Datenfluss: Event -> State -> Analyse."""
        # 1. Events zu State
        state = WifiAnalysisState()
        state.build_from_events(sample_events)
        
        # 2. State zu Features
        for mac, client in state.clients.items():
            features = analysis.features_for_client(client)
            if features is not None:
                assert isinstance(features, dict)
                assert "vendor" in features
        
        # 3. State zu Clustering
        clustered_df, feature_df = analysis.cluster_clients(state, n_clusters=2)
        if clustered_df is not None:
            assert len(clustered_df) > 0
        
        # 4. State zu Inferenz
        results = analysis.score_pairs_with_recency_and_matching(state)
        assert isinstance(results, list)
    
    def test_state_persistence(self, sample_events, temp_db_file):
        """Test State-Persistierung."""
        # Erstelle State
        state1 = WifiAnalysisState()
        state1.build_from_events(sample_events)
        
        # Schreibe in Datenbank
        with database.db_conn_ctx(temp_db_file) as conn:
            for event in sample_events:
                database.add_event(conn, event)
        
        # Lade aus Datenbank und erstelle neuen State
        with database.db_conn_ctx(temp_db_file) as conn:
            loaded_events = list(database.fetch_events(conn))
        
        state2 = WifiAnalysisState()
        state2.build_from_events(loaded_events)
        
        # Vergleiche States
        assert len(state1.aps) == len(state2.aps)
        assert len(state1.clients) == len(state2.clients)
        assert len(state1.ssid_map) == len(state2.ssid_map)


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDataset:
    """Tests mit großen Datensätzen."""
    
    def test_large_dataset_processing(self):
        """Test Verarbeitung großer Datensätze."""
        # Erstelle großen State
        state = WifiAnalysisState()
        
        # Erstelle 100 APs
        for i in range(100):
            bssid = f"08:96:d7:1a:21:{i:02x}"
            ap = state.aps[bssid] = state.aps.get(bssid, type(state).__dict__['aps'].__class__())()
            ap.bssid = bssid
            ap.ssid = f"AP_{i}"
            ap.channel = (i % 13) + 1
            ap.beacon_count = 10 + i
            ap.rssi_w = type(state).__dict__['aps'].__class__()()
            ap.rssi_w.update(-50 - i)
        
        # Erstelle 200 Clients
        for i in range(200):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = state.clients[mac] = state.clients.get(mac, type(state).__dict__['clients'].__class__())()
            client.mac = mac
            client.probes = {f"SSID_{i % 50}"}
            client.all_packet_ts = [time.time() - 100 + i, time.time() - 50 + i]
            client.rssi_w = type(state).__dict__['clients'].__class__()()
            client.rssi_w.update(-50 - i)
        
        # Teste Clustering
        clustered_df, feature_df = analysis.cluster_clients(state, n_clusters=5)
        if clustered_df is not None:
            assert len(clustered_df) == 200
        
        # Teste AP-Clustering
        ap_clustered_df = analysis.cluster_aps(state, n_clusters=5)
        if ap_clustered_df is not None:
            assert len(ap_clustered_df) == 100
        
        # Teste Inferenz
        results = analysis.score_pairs_with_recency_and_matching(state)
        assert isinstance(results, list)


@pytest.mark.integration
@pytest.mark.network
class TestNetworkIntegration:
    """Tests die Netzwerk-Zugriff benötigen."""
    
    @patch('urllib.request.urlopen')
    def test_oui_download_integration(self, mock_urlopen):
        """Test OUI-Download-Integration."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"test oui data"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        result = utils.download_oui_file()
        
        # Sollte erfolgreich sein oder graceful fehlschlagen
        assert isinstance(result, bool)
    
    def test_vendor_lookup_integration(self):
        """Test Vendor-Lookup-Integration."""
        # Teste verschiedene MAC-Adressen
        test_macs = [
            "a8:51:ab:0c:b9:e9",  # Apple
            "b2:87:23:15:7f:f2",  # Randomisiert
            "08:96:d7:1a:21:1c",  # Unbekannt
        ]
        
        for mac in test_macs:
            vendor = utils.lookup_vendor(mac)
            assert vendor is None or isinstance(vendor, str)


@pytest.mark.integration
class TestErrorRecovery:
    """Tests für Fehlerbehandlung und Recovery."""
    
    def test_malformed_event_handling(self):
        """Test Behandlung fehlerhafter Events."""
        state = WifiAnalysisState()
        
        # Teste verschiedene fehlerhafte Events
        malformed_events = [
            {},  # Leeres Event
            {'ts': time.time()},  # Nur Zeitstempel
            {'ts': time.time(), 'type': 'invalid'},  # Ungültiger Typ
            {'ts': time.time(), 'type': 'beacon'},  # Beacon ohne BSSID
        ]
        
        for event in malformed_events:
            # Sollte nicht crashen
            try:
                state.update_from_event(event)
            except Exception as e:
                pytest.fail(f"Malformed event should be handled gracefully: {e}")
    
    def test_database_error_recovery(self):
        """Test Datenbank-Fehlerbehandlung."""
        # Teste mit ungültiger Datenbank
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            invalid_db = f.name
        
        try:
            # Sollte graceful fehlschlagen
            with database.db_conn_ctx(invalid_db) as conn:
                events = list(database.fetch_events(conn))
                assert events == []
        except Exception as e:
            # Das ist OK, sollte einen Fehler werfen
            assert isinstance(e, sqlite3.OperationalError)
        finally:
            Path(invalid_db).unlink(missing_ok=True)
    
    def test_analysis_error_recovery(self):
        """Test Analyse-Fehlerbehandlung."""
        # Teste mit leerem State
        empty_state = WifiAnalysisState()
        
        # Alle Analyse-Funktionen sollten graceful mit leerem State umgehen
        results = analysis.score_pairs_with_recency_and_matching(empty_state)
        assert results == []
        
        clustered_df, feature_df = analysis.cluster_clients(empty_state)
        assert clustered_df is None or len(clustered_df) == 0
        
        ap_clustered_df = analysis.cluster_aps(empty_state)
        assert ap_clustered_df is None or len(ap_clustered_df) == 0


@pytest.mark.integration
class TestMemoryManagement:
    """Tests für Speicherverwaltung."""
    
    def test_memory_usage_large_state(self):
        """Test Speicherverbrauch mit großem State."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Erstelle großen State
        state = WifiAnalysisState()
        
        for i in range(1000):
            mac = f"aa:bb:cc:dd:ee:{i:04x}"
            client = state.clients[mac] = state.clients.get(mac, type(state).__dict__['clients'].__class__())()
            client.mac = mac
            client.probes = {f"SSID_{i % 100}"}
            client.all_packet_ts = [time.time() - 100 + i, time.time() - 50 + i]
            client.rssi_w = type(state).__dict__['clients'].__class__()()
            client.rssi_w.update(-50 - i)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Speicherzunahme sollte nicht zu groß sein (weniger als 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_state_pruning_memory_release(self):
        """Test Speicherfreigabe durch State-Pruning."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Erstelle State mit alten und neuen Daten
        state = WifiAnalysisState()
        
        # Alte Clients (werden gepruned)
        for i in range(100):
            mac = f"old:aa:bb:cc:dd:ee:{i:02x}"
            client = state.clients[mac] = state.clients.get(mac, type(state).__dict__['clients'].__class__())()
            client.mac = mac
            client.last_seen = time.time() - 10000  # Sehr alt
        
        # Neue Clients (bleiben)
        for i in range(100):
            mac = f"new:aa:bb:cc:dd:ee:{i:02x}"
            client = state.clients[mac] = state.clients.get(mac, type(state).__dict__['clients'].__class__())()
            client.mac = mac
            client.last_seen = time.time()  # Neu
        
        initial_memory = process.memory_info().rss
        
        # Führe Pruning durch
        pruned_count = state.prune_state(time.time(), threshold_s=7200)
        
        final_memory = process.memory_info().rss
        
        assert pruned_count > 0
        assert len(state.clients) == 100  # Nur neue Clients sollten bleiben
        # Speicher sollte reduziert worden sein (oder zumindest nicht stark zugenommen haben)
        memory_change = final_memory - initial_memory
        assert memory_change < 50 * 1024 * 1024  # Weniger als 50MB Zunahme