# test_app.py

import pytest
from unittest.mock import MagicMock, patch
import time
import sqlite3
import pandas as pd
import queue
from pathlib import Path
import numpy as np

# Importiere die echten Scapy-Layer, die wir zum Bauen des Pakets benötigen
from scapy.all import RadioTap, Dot11, Dot11Beacon, Dot11Elt

# Importiere die zu testenden Module aus deinem Projekt
from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState
from wlan_tool import analysis, utils
from wlan_tool.capture import sniffer as capture
from wlan_tool.storage import database
from wlan_tool.storage.data_models import Welford

# --- Fixtures: Wiederverwendbare Testdaten ---

@pytest.fixture
def sample_events():
    """Eine Liste von simulierten WifiEvent-Wörterbüchern für Tests."""
    ts = time.time()
    return [
        {'ts': ts, 'type': 'beacon', 'bssid': '08:96:d7:1a:21:1c', 'ssid': 'MyTestWLAN', 'rssi': -50, 'noise': -95, 'channel': 6, 'ies': {0: ['MyTestWLAN'], 1: ['82848b96'], 48: ['0100000fac040100000fac020100000fac028c00']}},
        {'ts': ts + 1, 'type': 'probe_req', 'client': 'a8:51:ab:0c:b9:e9', 'rssi': -65, 'noise': -98, 'ies': {0: [''], 221: ['0017f20a010103040507080c']}},
        {'ts': ts + 2.0, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 'bssid': '08:96:d7:1a:21:1c', 'rssi': -55, 'mcs_index': 7, 'noise': -90},
        {'ts': ts + 2.1, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 'bssid': '08:96:d7:1a:21:1c', 'rssi': -56, 'mcs_index': 7, 'noise': -90},
        {'ts': ts + 2.2, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 'bssid': '08:96:d7:1a:21:1c', 'rssi': -54, 'mcs_index': 8, 'noise': -91},
        {'ts': ts + 2.3, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 'bssid': '08:96:d7:1a:21:1c', 'rssi': -55, 'mcs_index': 8, 'noise': -90},
        {'ts': ts + 2.4, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 'bssid': '08:96:d7:1a:21:1c', 'rssi': -55, 'mcs_index': 7, 'noise': -90},
        {'ts': ts + 3, 'type': 'probe_req', 'client': 'b2:87:23:15:7f:f2', 'rssi': -70, 'ies': {0: ['MyOtherWLAN']}},
        {'ts': ts - 8000, 'type': 'probe_req', 'client': 'DE:AD:BE:EF:00:00', 'rssi': -80, 'ies': {0: ['OldWLAN']}}
    ]

@pytest.fixture
def populated_state(sample_events):
    state = WifiAnalysisState()
    state.build_from_events(sample_events)
    return state

@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:")
    db_path = Path(__file__).parent.parent / "wlan_tool" / "assets" / "sql_data" / "versions"
    for migration_file in sorted(db_path.glob("*.sql")):
        sql_script = migration_file.read_text()
        conn.executescript(sql_script)
    yield conn
    conn.close()


class TestUtilsModule:
    def test_parse_ies_simple(self):
        ies = {45: ['somehexdata']}
        parsed = utils.parse_ies(ies)
        assert "802.11n" in parsed["standards"]

    def test_parse_ies_detailed_rsn(self):
        ies = {48: ['0100000fac040100000fac020100000fac028c00']}
        parsed = utils.parse_ies(ies, detailed=True)
        assert "WPA2/3" in parsed["security"]
        assert "PSK" in parsed["rsn_details"]["akm_suites"]
        assert parsed["rsn_details"]["mfp_capable"] is True

    def test_intelligent_vendor_lookup(self):
        assert "Apple" in utils.lookup_vendor("a8:51:ab:0c:b9:e9")
        assert "Randomisiert" in utils.lookup_vendor("b2:87:23:15:7f:f2")
        mock_client = ClientState(mac="b2:87:23:15:7f:f2")
        mock_client.parsed_ies = {"vendor_specific": {"Apple": True}}
        assert "Apple?" in utils.intelligent_vendor_lookup(mock_client.mac, mock_client)


class TestStateModule:
    def test_state_update_from_beacon(self, sample_events):
        state = WifiAnalysisState()
        state.update_from_event(sample_events[0])
        assert '08:96:d7:1a:21:1c' in state.aps
        ap = state.aps['08:96:d7:1a:21:1c']
        assert ap.ssid == "MyTestWLAN"
        assert ap.beacon_count == 1
        assert ap.noise_w.mean == -95

    def test_state_update_from_probe_req(self, sample_events):
        state = WifiAnalysisState()
        state.update_from_event(sample_events[1])
        assert 'a8:51:ab:0c:b9:e9' in state.clients
        client = state.clients['a8:51:ab:0c:b9:e9']
        assert "" not in client.probes
        assert len(client.probes) == 0

    def test_prune_state(self, populated_state):
        state = populated_state
        assert 'DE:AD:BE:EF:00:00' in state.clients
        pruned_count = state.prune_state(time.time(), threshold_s=7200)
        assert pruned_count > 0
        assert 'DE:AD:BE:EF:00:00' not in state.clients
        assert 'a8:51:ab:0c:b9:e9' in state.clients


class TestAnalysisModule:
    def test_features_for_client_behavior(self, populated_state):
        client = populated_state.clients['a8:51:ab:0c:b9:e9']
        features = analysis.features_for_client_behavior(client)
        assert features is not None
        expected_snr = np.mean([-65, -55, -56, -54, -55, -55]) - np.mean([-98, -90, -90, -91, -90, -90])
        assert "avg_mcs_rate" in features
        assert features["avg_mcs_rate"] == pytest.approx(7.4)
        assert "avg_snr" in features
        assert features["avg_snr"] == pytest.approx(expected_snr)

    def test_cluster_clients_runs(self, populated_state):
        clustered_df, feature_df = analysis.cluster_clients(populated_state, algo="kmeans", n_clusters=2)
        assert isinstance(clustered_df, pd.DataFrame)
        assert "cluster" in clustered_df.columns
        assert "mac" in clustered_df.columns
        assert isinstance(feature_df, pd.DataFrame)

    def test_profile_clusters(self):
        feature_data = {'original_macs': ['mac1', 'mac2'], 'feature1': [10, 20], 'feature2': [1, 2]}
        feature_df = pd.DataFrame(feature_data)
        cluster_data = {'original_macs': ['mac1', 'mac2'], 'cluster': [0, 0]}
        clustered_df = pd.DataFrame(cluster_data)
        profiles = analysis.profile_clusters(feature_df, clustered_df)
        assert 0 in profiles
        assert profiles[0]['count'] == 2
        assert profiles[0]['feature1'] == 15.0


class TestCaptureModule:
    def test_packet_to_event_beacon(self):
        # --- Erstelle ein echtes, korrekt formatiertes Scapy-Paket ---
        rt_layer = RadioTap(
            # KORREKTUR: Das 'present'-Feld explizit setzen
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-42
        )
        dot11_layer = Dot11(
            type=0, subtype=8,
            addr2='aa:bb:cc:dd:ee:ff',
            addr3='aa:bb:cc:dd:ee:ff'
        )
        beacon_layer = Dot11Beacon(
            beacon_interval=102,
            cap=0x11
        )
        ssid_ie = Dot11Elt(
            ID=0,
            info=b'MyTestSSID'
        )
        
        pkt = rt_layer / dot11_layer / beacon_layer / ssid_ie
        pkt.time = time.time()

        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'beacon'
        assert event['bssid'] == 'aa:bb:cc:dd:ee:ff'
        assert event['rssi'] == -42
        assert event['channel'] == 1
        assert event['ssid'] == 'MyTestSSID'
        assert event['beacon_interval'] == 102
        assert event['cap'] == 0x11


class TestDatabaseModule:
    def test_db_write_and_fetch(self, in_memory_db, sample_events):
        conn = in_memory_db
        q = queue.Queue()
        
        test_event = sample_events[0]
        q.put(test_event)
        
        writer = database.BatchedEventWriter(db_path=":memory:", q=q, batch_size=1, flush_interval=1)
        writer._flush(conn, [writer._event_to_tuple(test_event)], [], [], [])
        
        fetched_events = list(database.fetch_events(conn))
        
        assert len(fetched_events) == 1
        fetched_event = fetched_events[0]
        
        assert fetched_event['bssid'] == test_event['bssid']
        assert fetched_event['ssid'] == test_event['ssid']