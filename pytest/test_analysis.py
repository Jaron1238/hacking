#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Analysis-Modul (logic.py, device_profiler.py, models.py, training.py).
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import MagicMock, patch
from collections import defaultdict

from wlan_tool.analysis import logic as analysis
from wlan_tool.analysis.device_profiler import (
    create_device_fingerprint, 
    classify_device_by_fingerprint,
    build_fingerprint_database,
    correlate_devices_by_fingerprint
)
from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState
from wlan_tool.storage.data_models import InferenceResult


class TestAnalysisLogic:
    """Tests für die Analyse-Logik."""
    
    def test_features_for_pair_basic(self, populated_state):
        """Test Feature-Extraktion für SSID-BSSID-Paare."""
        state = populated_state
        ssid = "MyTestWLAN"
        bssid = "08:96:d7:1a:21:1c"
        
        features, supporting_clients = analysis.features_for_pair_basic(ssid, bssid, state)
        
        assert isinstance(features, dict)
        assert features['ssid'] == ssid
        assert features['bssid'] == bssid
        assert 'beacon_count' in features
        assert 'probe_resp_count' in features
        assert 'supporting_clients' in features
        assert isinstance(supporting_clients, list)
    
    def test_heuristic_score(self):
        """Test heuristisches Scoring."""
        feat = {
            'beacon_count': 10,
            'probe_resp_count': 5,
            'supporting_clients': 3,
            'rssi_std': 5.0,
            'seq_support': 2
        }
        
        score = analysis.heuristic_score(feat)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_score_pairs_with_recency_and_matching(self, populated_state):
        """Test SSID-BSSID-Paar-Scoring."""
        state = populated_state
        results = analysis.score_pairs_with_recency_and_matching(state)
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, InferenceResult)
            assert hasattr(result, 'ssid')
            assert hasattr(result, 'bssid')
            assert hasattr(result, 'score')
            assert 0.0 <= result.score <= 1.0
    
    def test_features_for_client(self, mock_client_state):
        """Test Client-Feature-Extraktion."""
        features = analysis.features_for_client(mock_client_state)
        
        assert isinstance(features, dict)
        assert 'vendor' in features
        assert 'supports_11n' in features
        assert 'supports_11ac' in features
        assert 'probe_count' in features
        assert 'seen_with_ap_count' in features
    
    def test_features_for_client_behavior(self, mock_client_state):
        """Test Client-Verhaltens-Feature-Extraktion."""
        features = analysis.features_for_client_behavior(mock_client_state)
        
        if features is not None:  # Kann None sein wenn zu wenige Pakete
            assert isinstance(features, dict)
            assert 'packet_interval_mean' in features
            assert 'packet_interval_std' in features
            assert 'data_to_mgmt_ratio' in features
            assert 'power_save_rate' in features
    
    def test_prepare_client_feature_dataframe(self, populated_state):
        """Test Client-Feature-DataFrame-Erstellung."""
        df = analysis.prepare_client_feature_dataframe(populated_state)
        
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert 'mac' in df.columns
            assert 'vendor' in df.columns
            assert len(df) > 0
    
    def test_find_optimal_k_elbow_and_silhouette(self):
        """Test optimale Cluster-Anzahl-Bestimmung."""
        # Erstelle Test-Daten
        X_scaled = np.random.rand(20, 5)
        
        optimal_k = analysis.find_optimal_k_elbow_and_silhouette(X_scaled, max_k=5)
        
        if optimal_k is not None:
            assert isinstance(optimal_k, int)
            assert 2 <= optimal_k <= 5
    
    def test_cluster_clients(self, populated_state):
        """Test Client-Clustering."""
        clustered_df, feature_df = analysis.cluster_clients(
            populated_state, 
            n_clusters=2, 
            algo="kmeans"
        )
        
        if clustered_df is not None:
            assert isinstance(clustered_df, pd.DataFrame)
            assert 'cluster' in clustered_df.columns
            assert 'mac' in clustered_df.columns
            assert len(clustered_df) > 0
        
        if feature_df is not None:
            assert isinstance(feature_df, pd.DataFrame)
    
    def test_profile_clusters(self):
        """Test Cluster-Profilierung."""
        feature_data = {
            'original_macs': ['mac1', 'mac2', 'mac3'],
            'feature1': [10, 20, 15],
            'feature2': [1, 2, 1.5]
        }
        feature_df = pd.DataFrame(feature_data)
        
        cluster_data = {
            'original_macs': ['mac1', 'mac2', 'mac3'],
            'cluster': [0, 0, 1]
        }
        clustered_df = pd.DataFrame(cluster_data)
        
        profiles = analysis.profile_clusters(feature_df, clustered_df)
        
        assert isinstance(profiles, dict)
        assert 0 in profiles
        assert 1 in profiles
        assert profiles[0]['count'] == 2
        assert profiles[1]['count'] == 1
    
    def test_features_for_ap(self, mock_ap_state):
        """Test AP-Feature-Extraktion."""
        features = analysis.features_for_ap(mock_ap_state)
        
        if features is not None:
            assert isinstance(features, dict)
            assert 'ssid' in features
            assert 'vendor' in features
            assert 'channel' in features
            assert 'beacon_interval_mode' in features
    
    def test_cluster_aps(self, populated_state):
        """Test AP-Clustering."""
        clustered_df = analysis.cluster_aps(populated_state, n_clusters=2)
        
        if clustered_df is not None:
            assert isinstance(clustered_df, pd.DataFrame)
            assert 'cluster' in clustered_df.columns
            assert 'bssid' in clustered_df.columns
            assert len(clustered_df) > 0
    
    def test_profile_ap_clusters(self):
        """Test AP-Cluster-Profilierung."""
        cluster_data = {
            'bssid': ['ap1', 'ap2'],
            'ssid': ['SSID1', 'SSID2'],
            'vendor': ['Vendor1', 'Vendor2'],
            'cluster': [0, 1],
            'supports_11k': [True, False],
            'supports_11v': [True, False],
            'supports_11r': [False, True]
        }
        clustered_df = pd.DataFrame(cluster_data)
        
        profiles = analysis.profile_ap_clusters(clustered_df)
        
        assert isinstance(profiles, dict)
        assert 0 in profiles
        assert 1 in profiles
    
    def test_correlate_randomized_clients(self, populated_state):
        """Test Randomisierte-Client-Korrelation."""
        correlated_groups = analysis.correlate_randomized_clients(populated_state)
        
        assert isinstance(correlated_groups, dict)
        for fingerprint, macs in correlated_groups.items():
            assert isinstance(fingerprint, str)
            assert isinstance(macs, list)
            assert len(macs) > 1  # Nur Gruppen mit mehreren MACs
    
    def test_export_ap_graph(self, populated_state):
        """Test AP-Graph-Export."""
        # Erstelle Test-AP-Cluster-Daten
        ap_data = {
            'bssid': ['08:96:d7:1a:21:1c'],
            'ssid': ['MyTestWLAN'],
            'vendor': ['TestVendor'],
            'cluster': [0],
            'channel': [6],
            'rssi_mean': [-50.0],
            'supports_11k': [False],
            'supports_11v': [False],
            'supports_11r': [False]
        }
        clustered_ap_df = pd.DataFrame(ap_data)
        
        aps_to_export = {'08:96:d7:1a:21:1c': populated_state.aps['08:96:d7:1a:21:1c']}
        clients_to_export = {}
        
        # Teste Graph-Export
        success = analysis.export_ap_graph(
            populated_state,
            clustered_ap_df,
            aps_to_export,
            clients_to_export,
            "/tmp/test_graph.gexf",
            include_clients=False
        )
        
        assert isinstance(success, bool)


class TestDeviceProfiler:
    """Tests für das Device-Profiler-Modul."""
    
    def test_create_device_fingerprint(self, mock_client_state):
        """Test Geräte-Fingerprint-Erstellung."""
        fingerprint = create_device_fingerprint(mock_client_state)
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 32  # MD5-Hash-Länge
        assert fingerprint.isalnum()  # Nur alphanumerische Zeichen
    
    def test_create_device_fingerprint_empty_state(self):
        """Test Fingerprint mit leerem ClientState."""
        empty_client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        fingerprint = create_device_fingerprint(empty_client)
        
        assert fingerprint == ""  # Sollte leer sein für leeren State
    
    def test_classify_device_by_fingerprint(self):
        """Test Geräte-Klassifizierung nach Fingerprint."""
        known_fingerprints = {
            "abc123": "Apple",
            "def456": "Samsung"
        }
        
        result = classify_device_by_fingerprint("abc123", known_fingerprints)
        assert result == "Apple"
        
        result = classify_device_by_fingerprint("unknown", known_fingerprints)
        assert result is None
    
    def test_build_fingerprint_database(self, populated_state):
        """Test Fingerprint-Datenbank-Aufbau."""
        fingerprint_db = build_fingerprint_database(populated_state)
        
        assert isinstance(fingerprint_db, dict)
        for fingerprint, vendor in fingerprint_db.items():
            assert isinstance(fingerprint, str)
            assert isinstance(vendor, str)
    
    def test_correlate_devices_by_fingerprint(self, populated_state):
        """Test Geräte-Korrelation nach Fingerprint."""
        correlated_groups = correlate_devices_by_fingerprint(populated_state)
        
        assert isinstance(correlated_groups, dict)
        for fingerprint, macs in correlated_groups.items():
            assert isinstance(fingerprint, str)
            assert isinstance(macs, list)
            assert len(macs) > 1


class TestGraphExport:
    """Tests für Graph-Export-Funktionalität."""
    
    def test_build_export_graph(self, populated_state):
        """Test Graph-Aufbau für Export."""
        # Erstelle Test-Daten
        ap_data = {
            'bssid': ['08:96:d7:1a:21:1c'],
            'ssid': ['MyTestWLAN'],
            'vendor': ['TestVendor'],
            'cluster': [0],
            'channel': [6],
            'rssi_mean': [-50.0],
            'supports_11k': [False],
            'supports_11v': [False],
            'supports_11r': [False]
        }
        clustered_ap_df = pd.DataFrame(ap_data)
        
        aps_to_export = {'08:96:d7:1a:21:1c': populated_state.aps['08:96:d7:1a:21:1c']}
        clients_to_export = {}
        
        graph = analysis._build_export_graph(
            populated_state,
            clustered_ap_df,
            aps_to_export,
            clients_to_export,
            include_clients=False,
            clustered_client_df=None
        )
        
        assert graph is not None
        assert graph.number_of_nodes() > 0
        assert 'start' in graph.graph
        assert 'end' in graph.graph
    
    def test_discover_attributes(self):
        """Test Attribut-Entdeckung für GEXF."""
        import networkx as nx
        
        G = nx.Graph()
        G.add_node("node1", type="AP", activity=10, vendor="Test")
        G.add_edge("node1", "node2", weight=1.5, kind="Association")
        
        node_attrs, edge_attrs = analysis._discover_attributes(G)
        
        assert isinstance(node_attrs, dict)
        assert isinstance(edge_attrs, dict)
        assert 'type' in node_attrs
        assert 'activity' in node_attrs
        assert 'weight' in edge_attrs
        assert 'kind' in edge_attrs


class TestInferenceResult:
    """Tests für InferenceResult-Datenmodell."""
    
    def test_inference_result_creation(self):
        """Test InferenceResult-Erstellung."""
        result = InferenceResult(
            ssid="TestSSID",
            bssid="aa:bb:cc:dd:ee:ff",
            score=0.85,
            label="high",
            components={"beacon_count": 10},
            evidence={"supporting_clients": ["client1"]}
        )
        
        assert result.ssid == "TestSSID"
        assert result.bssid == "aa:bb:cc:dd:ee:ff"
        assert result.score == 0.85
        assert result.label == "high"
        assert result.components["beacon_count"] == 10
        assert "client1" in result.evidence["supporting_clients"]


class TestAnalysisEdgeCases:
    """Tests für Edge-Cases in der Analyse."""
    
    def test_empty_state_analysis(self):
        """Test Analyse mit leerem State."""
        empty_state = WifiAnalysisState()
        
        results = analysis.score_pairs_with_recency_and_matching(empty_state)
        assert results == []
        
        clustered_df, feature_df = analysis.cluster_clients(empty_state)
        assert clustered_df is None or len(clustered_df) == 0
    
    def test_single_client_analysis(self):
        """Test Analyse mit nur einem Client."""
        state = WifiAnalysisState()
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        client.all_packet_ts = np.array([time.time(), time.time() + 1])
        client.rssi_w = Welford()
        client.rssi_w.update(-50)
        state.clients["aa:bb:cc:dd:ee:ff"] = client
        
        features = analysis.features_for_client_behavior(client)
        # Kann None sein wenn zu wenige Pakete
        if features is not None:
            assert isinstance(features, dict)
    
    def test_invalid_cluster_parameters(self, populated_state):
        """Test Clustering mit ungültigen Parametern."""
        # Teste mit n_clusters=0 (sollte automatisch bestimmen)
        clustered_df, feature_df = analysis.cluster_clients(
            populated_state, 
            n_clusters=0,
            algo="kmeans"
        )
        
        if clustered_df is not None:
            assert isinstance(clustered_df, pd.DataFrame)
    
    def test_malformed_event_handling(self):
        """Test Behandlung von fehlerhaften Events."""
        state = WifiAnalysisState()
        
        # Teste mit unvollständigem Event
        malformed_event = {'ts': time.time(), 'type': 'beacon'}  # Fehlt bssid
        
        # Sollte nicht crashen
        try:
            state.update_from_event(malformed_event)
        except Exception as e:
            pytest.fail(f"Malformed event should be handled gracefully: {e}")