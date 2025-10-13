#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Storage-Modul (state.py, data_models.py, database.py).
"""

import pytest
import sqlite3
import time
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState, Welford
from wlan_tool.storage.data_models import WifiEvent, EventType
from wlan_tool.storage import database


class TestWelford:
    """Tests für die Welford-Klasse."""
    
    def test_welford_initialization(self):
        """Test Welford-Initialisierung."""
        w = Welford()
        assert w.n == 0
        assert w.mean == 0.0
        assert w.M2 == 0.0
    
    def test_welford_single_update(self):
        """Test Welford mit einem Wert."""
        w = Welford()
        w.update(10.0)
        assert w.n == 1
        assert w.mean == 10.0
        assert w.std() == 0.0
    
    def test_welford_multiple_updates(self):
        """Test Welford mit mehreren Werten."""
        w = Welford()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            w.update(v)
        
        assert w.n == len(values)
        assert w.mean == pytest.approx(3.0)
        assert w.std() == pytest.approx(1.581, abs=0.01)
    
    def test_welford_std_with_single_value(self):
        """Test Standardabweichung mit einem Wert."""
        w = Welford()
        w.update(5.0)
        assert w.std() == 0.0
    
    def test_welford_combine(self):
        """Test Welford-Kombination."""
        w1 = Welford()
        w1.update(1.0)
        w1.update(2.0)
        
        w2 = Welford()
        w2.update(3.0)
        w2.update(4.0)
        
        w3 = w1.combine(w2)
        assert w3.n == 4
        assert w3.mean == pytest.approx(2.5)
        assert w3.std() == pytest.approx(1.291, abs=0.01)


class TestClientState:
    """Tests für die ClientState-Klasse."""
    
    def test_client_state_initialization(self):
        """Test ClientState-Initialisierung."""
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        assert client.mac == "aa:bb:cc:dd:ee:ff"
        assert client.count == 0
        assert client.probes == set()
        assert client.seen_with == set()
        assert client.randomized == False
        assert client.rssi_w.n == 0
    
    def test_client_state_update_from_event(self):
        """Test ClientState-Update aus Event."""
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        event = {
            'ts': time.time(),
            'type': 'probe_req',
            'client': 'aa:bb:cc:dd:ee:ff',
            'rssi': -50,
            'ies': {0: ['TestSSID']}
        }
        
        client.update_from_event(event)
        assert client.count == 1
        assert 'TestSSID' in client.probes
        assert client.rssi_w.n == 1
        assert client.rssi_w.mean == -50
    
    def test_client_state_powersave_detection(self):
        """Test Power-Save-Erkennung."""
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        event = {
            'ts': time.time(),
            'type': 'data',
            'client': 'aa:bb:cc:dd:ee:ff',
            'is_powersave': True
        }
        
        client.update_from_event(event)
        assert client.last_powersave_ts > 0
        assert client.power_save_transitions == 1


class TestAPState:
    """Tests für die APState-Klasse."""
    
    def test_ap_state_initialization(self):
        """Test APState-Initialisierung."""
        ap = APState(bssid="aa:bb:cc:dd:ee:ff", ssid="TestAP")
        assert ap.bssid == "aa:bb:cc:dd:ee:ff"
        assert ap.ssid == "TestAP"
        assert ap.beacon_count == 0
        assert ap.probe_resp_count == 0
        assert ap.channel is None
    
    def test_ap_state_update_from_beacon(self):
        """Test APState-Update aus Beacon."""
        ap = APState(bssid="aa:bb:cc:dd:ee:ff", ssid="TestAP")
        event = {
            'ts': time.time(),
            'type': 'beacon',
            'bssid': 'aa:bb:cc:dd:ee:ff',
            'ssid': 'TestAP',
            'rssi': -50,
            'channel': 6,
            'beacon_interval': 102
        }
        
        ap.update_from_event(event)
        assert ap.beacon_count == 1
        assert ap.channel == 6
        assert ap.beacon_intervals[102] == 1
        assert ap.rssi_w.n == 1
        assert ap.rssi_w.mean == -50


class TestWifiAnalysisState:
    """Tests für die WifiAnalysisState-Klasse."""
    
    def test_state_initialization(self):
        """Test State-Initialisierung."""
        state = WifiAnalysisState()
        assert len(state.aps) == 0
        assert len(state.clients) == 0
        assert len(state.ssid_map) == 0
    
    def test_state_update_from_beacon(self, sample_events):
        """Test State-Update aus Beacon-Event."""
        state = WifiAnalysisState()
        event = sample_events[0]  # Beacon-Event
        
        state.update_from_event(event)
        
        assert '08:96:d7:1a:21:1c' in state.aps
        ap = state.aps['08:96:d7:1a:21:1c']
        assert ap.ssid == "MyTestWLAN"
        assert ap.beacon_count == 1
        assert 'MyTestWLAN' in state.ssid_map
        assert '08:96:d7:1a:21:1c' in state.ssid_map['MyTestWLAN']['bssids']
    
    def test_state_update_from_probe_req(self, sample_events):
        """Test State-Update aus Probe-Request-Event."""
        state = WifiAnalysisState()
        event = sample_events[1]  # Probe-Request-Event
        
        state.update_from_event(event)
        
        assert 'a8:51:ab:0c:b9:e9' in state.clients
        client = state.clients['a8:51:ab:0c:b9:e9']
        assert client.count == 1
        assert client.rssi_w.n == 1
    
    def test_state_build_from_events(self, sample_events):
        """Test State-Aufbau aus Event-Liste."""
        state = WifiAnalysisState()
        state.build_from_events(sample_events)
        
        assert len(state.aps) == 1
        assert len(state.clients) == 3
        assert 'MyTestWLAN' in state.ssid_map
    
    def test_state_pruning(self, populated_state):
        """Test State-Pruning."""
        state = populated_state
        assert 'DE:AD:BE:EF:00:00' in state.clients
        
        pruned_count = state.prune_state(time.time(), threshold_s=7200)
        
        assert pruned_count > 0
        assert 'DE:AD:BE:EF:00:00' not in state.clients
        assert 'a8:51:ab:0c:b9:e9' in state.clients  # Sollte bleiben
    
    def test_state_ssid_mapping(self, sample_events):
        """Test SSID-Mapping-Funktionalität."""
        state = WifiAnalysisState()
        state.build_from_events(sample_events)
        
        # Teste SSID-Map
        assert 'MyTestWLAN' in state.ssid_map
        ssid_info = state.ssid_map['MyTestWLAN']
        assert '08:96:d7:1a:21:1c' in ssid_info['bssids']
        assert 'a8:51:ab:0c:b9:e9' in ssid_info['sources']['probe_req']


class TestDatabaseModule:
    """Tests für das Database-Modul."""
    
    def test_db_connection_context(self, temp_db_file):
        """Test Datenbankverbindungskontext."""
        with database.db_conn_ctx(temp_db_file) as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Teste, ob Schema existiert
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert 'events' in tables
    
    def test_event_writer(self, temp_db_file, sample_events):
        """Test Event-Writer."""
        import queue
        
        q = queue.Queue()
        for event in sample_events[:3]:
            q.put(event)
        
        writer = database.BatchedEventWriter(
            db_path=temp_db_file,
            q=q,
            batch_size=2,
            flush_interval=1.0
        )
        
        # Simuliere Writer-Ausführung
        writer.start()
        time.sleep(0.1)  # Kurz warten
        writer.stop()
        
        # Prüfe, ob Events geschrieben wurden
        with database.db_conn_ctx(temp_db_file) as conn:
            events = list(database.fetch_events(conn))
            assert len(events) >= 2  # Mindestens 2 Events sollten geschrieben sein
    
    def test_fetch_events(self, in_memory_db, sample_events):
        """Test Event-Abruf."""
        conn = in_memory_db
        
        # Schreibe Test-Events
        for event in sample_events[:2]:
            database.add_event(conn, event)
        
        # Hole Events zurück
        events = list(database.fetch_events(conn))
        assert len(events) == 2
        assert events[0]['bssid'] == sample_events[0]['bssid']
    
    def test_add_label(self, in_memory_db):
        """Test Label-Hinzufügung."""
        conn = in_memory_db
        
        database.add_label(conn, "TestSSID", "aa:bb:cc:dd:ee:ff", 1)
        
        cursor = conn.execute("SELECT * FROM labels WHERE ssid = ? AND bssid = ?", 
                            ("TestSSID", "aa:bb:cc:dd:ee:ff"))
        result = cursor.fetchone()
        assert result is not None
        assert result[2] == 1  # label = 1
    
    def test_migrate_db(self, temp_db_file):
        """Test Datenbankmigration."""
        # Migration sollte bereits beim Erstellen der temp_db_file ausgeführt worden sein
        with database.db_conn_ctx(temp_db_file) as conn:
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            assert version > 0  # Sollte eine Version > 0 haben


class TestDataModels:
    """Tests für die Data-Models."""
    
    def test_wifi_event_creation(self):
        """Test WifiEvent-Erstellung."""
        event_data = {
            'ts': time.time(),
            'type': 'beacon',
            'bssid': 'aa:bb:cc:dd:ee:ff',
            'ssid': 'TestAP',
            'rssi': -50
        }
        
        event = WifiEvent(**event_data)
        assert event['ts'] == event_data['ts']
        assert event['type'] == 'beacon'
        assert event['bssid'] == 'aa:bb:cc:dd:ee:ff'
    
    def test_event_type_enum(self):
        """Test EventType-Enum."""
        assert EventType.BEACON == "beacon"
        assert EventType.PROBE_REQ == "probe_req"
        assert EventType.DATA == "data"