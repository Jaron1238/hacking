#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest-Konfiguration und gemeinsame Fixtures für alle Tests.
"""

import pytest
import tempfile
import sqlite3
import time
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

# Importiere die zu testenden Module
from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState, Welford
from wlan_tool.storage.data_models import WifiEvent, EventType
from wlan_tool.analysis import logic as analysis
from wlan_tool.analysis.device_profiler import create_device_fingerprint
from wlan_tool import utils, config
from wlan_tool.storage import database
from wlan_tool.capture import sniffer as capture
from wlan_tool.presentation import cli, reporting
from wlan_tool.controllers import CaptureController, AnalysisController

# Scapy-Imports für Paket-Tests
from scapy.all import RadioTap, Dot11, Dot11Beacon, Dot11Elt, Dot11ProbeReq


# ==============================================================================
# GEMEINSAME FIXTURES
# ==============================================================================

@pytest.fixture
def sample_timestamp():
    """Ein fester Zeitstempel für konsistente Tests."""
    return 1640995200.0  # 2022-01-01 00:00:00 UTC


@pytest.fixture
def sample_events(sample_timestamp):
    """Eine umfassende Liste von simulierten WifiEvent-Wörterbüchern."""
    ts = sample_timestamp
    return [
        # Beacon-Frame
        {
            'ts': ts, 'type': 'beacon', 'bssid': '08:96:d7:1a:21:1c', 
            'ssid': 'MyTestWLAN', 'rssi': -50, 'noise': -95, 'channel': 6,
            'ies': {0: ['MyTestWLAN'], 1: ['82848b96'], 48: ['0100000fac040100000fac020100000fac028c00']},
            'beacon_interval': 102, 'cap': 0x11, 'ie_order_hash': 12345
        },
        # Probe Request
        {
            'ts': ts + 1, 'type': 'probe_req', 'client': 'a8:51:ab:0c:b9:e9', 
            'rssi': -65, 'noise': -98, 'ies': {0: [''], 221: ['0017f20a010103040507080c']},
            'ie_order_hash': 54321
        },
        # Data Frames
        {'ts': ts + 2.0, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 
         'bssid': '08:96:d7:1a:21:1c', 'rssi': -55, 'mcs_index': 7, 'noise': -90},
        {'ts': ts + 2.1, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 
         'bssid': '08:96:d7:1a:21:1c', 'rssi': -56, 'mcs_index': 7, 'noise': -90},
        {'ts': ts + 2.2, 'type': 'data', 'client': 'a8:51:ab:0c:b9:e9', 
         'bssid': '08:96:d7:1a:21:1c', 'rssi': -54, 'mcs_index': 8, 'noise': -91},
        # Weitere Clients
        {'ts': ts + 3, 'type': 'probe_req', 'client': 'b2:87:23:15:7f:f2', 
         'rssi': -70, 'ies': {0: ['MyOtherWLAN']}, 'ie_order_hash': 67890},
        # Alter Client (wird beim Pruning entfernt)
        {'ts': ts - 8000, 'type': 'probe_req', 'client': 'DE:AD:BE:EF:00:00', 
         'rssi': -80, 'ies': {0: ['OldWLAN']}, 'ie_order_hash': 11111}
    ]


@pytest.fixture
def populated_state(sample_events):
    """Ein WifiAnalysisState mit Testdaten."""
    state = WifiAnalysisState()
    state.build_from_events(sample_events)
    return state


@pytest.fixture
def in_memory_db():
    """Eine In-Memory SQLite-Datenbank mit Schema."""
    conn = sqlite3.connect(":memory:")
    
    # Lade Schema aus den Migrationen
    migrations_path = Path(__file__).parent.parent / "wlan_tool" / "assets" / "sql_data" / "versions"
    for migration_file in sorted(migrations_path.glob("*.sql")):
        sql_script = migration_file.read_text()
        conn.executescript(sql_script)
    
    yield conn
    conn.close()


@pytest.fixture
def temp_db_file():
    """Eine temporäre Datenbankdatei."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Initialisiere Schema
    database.migrate_db(db_path)
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_client_state():
    """Ein Mock ClientState für Tests."""
    client = ClientState(mac="a8:51:ab:0c:b9:e9")
    client.probes = {"MyTestWLAN", "MyOtherWLAN"}
    client.seen_with = {"08:96:d7:1a:21:1c"}
    client.all_packet_ts = np.array([1640995200.0, 1640995201.0, 1640995202.0])
    client.rssi_w = Welford()
    client.rssi_w.update(-50)
    client.rssi_w.update(-55)
    client.rssi_w.update(-60)
    client.parsed_ies = {
        "standards": ["802.11n", "802.11ac"],
        "ht_caps": {"streams": 2},
        "vendor_specific": {"Apple": True}
    }
    client.ie_order_hashes = {12345, 54321}
    return client


@pytest.fixture
def mock_ap_state():
    """Ein Mock APState für Tests."""
    ap = APState(bssid="08:96:d7:1a:21:1c", ssid="MyTestWLAN")
    ap.beacon_count = 10
    ap.probe_resp_count = 5
    ap.channel = 6
    ap.rssi_w = Welford()
    ap.rssi_w.update(-50)
    ap.rssi_w.update(-55)
    ap.ies = {0: ['MyTestWLAN'], 48: ['0100000fac040100000fac020100000fac028c00']}
    ap.parsed_ies = {
        "standards": ["802.11n", "802.11ac"],
        "rsn_details": {"akm_suites": ["PSK"], "mfp_capable": True}
    }
    return ap


@pytest.fixture
def sample_scapy_packet():
    """Ein echtes Scapy-Paket für Capture-Tests."""
    rt_layer = RadioTap(
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
    return pkt


@pytest.fixture
def mock_console():
    """Ein Mock Console-Objekt für CLI-Tests."""
    console = MagicMock()
    console.print = MagicMock()
    console.input = MagicMock(return_value="test")
    return console


# ==============================================================================
# PARAMETRIZED FIXTURES
# ==============================================================================

@pytest.fixture(params=[
    "a8:51:ab:0c:b9:e9",  # Apple
    "b2:87:23:15:7f:f2",  # Randomisiert
    "08:96:d7:1a:21:1c",  # Unbekannt
])
def test_mac_address(request):
    """Verschiedene MAC-Adressen für Tests."""
    return request.param


@pytest.fixture(params=[
    {"type": "beacon", "expected": True},
    {"type": "probe_req", "expected": True},
    {"type": "data", "expected": True},
    {"type": "invalid", "expected": False},
])
def event_type_test(request):
    """Verschiedene Event-Typen für Tests."""
    return request.param


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_test_state_with_clients(num_clients=5):
    """Erstellt einen Test-State mit mehreren Clients."""
    state = WifiAnalysisState()
    
    for i in range(num_clients):
        mac = f"aa:bb:cc:dd:ee:{i:02x}"
        client = ClientState(mac=mac)
        client.probes = {f"SSID_{i}"}
        client.all_packet_ts = np.array([time.time() - 100 + i*10, time.time() - 50 + i*10])
        client.rssi_w = Welford()
        client.rssi_w.update(-50 - i*5)
        state.clients[mac] = client
    
    return state


def create_test_state_with_aps(num_aps=3):
    """Erstellt einen Test-State mit mehreren APs."""
    state = WifiAnalysisState()
    
    for i in range(num_aps):
        bssid = f"08:96:d7:1a:21:{i:02x}"
        ap = APState(bssid=bssid, ssid=f"TestAP_{i}")
        ap.channel = 6 + i*5
        ap.beacon_count = 10 + i
        ap.rssi_w = Welford()
        ap.rssi_w.update(-50 - i*10)
        state.aps[bssid] = ap
    
    return state