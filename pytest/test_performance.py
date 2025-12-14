#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance-Tests für das WLAN-Analyse-Tool.
"""

import pytest
import time
import numpy as np
from pathlib import Path
import tempfile
import psutil
import os

from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState, Welford
from wlan_tool.analysis import logic as analysis
from wlan_tool import utils
from wlan_tool.storage import database


@pytest.mark.slow
class TestPacketProcessingPerformance:
    """Tests für Paket-Verarbeitungs-Performance."""
    
    def test_packet_parsing_speed(self):
        """Test Paket-Parsing-Geschwindigkeit."""
        from wlan_tool.capture import sniffer as capture
        from scapy.all import RadioTap, Dot11, Dot11Beacon, Dot11Elt
        
        # Erstelle Test-Paket
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-50
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
        ssid_ie = Dot11Elt(ID=0, info=b'TestSSID')
        
        pkt = rt_layer / dot11_layer / beacon_layer / ssid_ie
        pkt.time = time.time()
        
        # Teste Verarbeitungsgeschwindigkeit
        num_packets = 1000
        start_time = time.time()
        
        for _ in range(num_packets):
            capture.packet_to_event(pkt)
        
        end_time = time.time()
        processing_time = end_time - start_time
        packets_per_second = num_packets / processing_time
        
        # Sollte mindestens 1000 Pakete pro Sekunde verarbeiten können
        assert packets_per_second > 1000, f"Only {packets_per_second:.0f} packets/second, expected > 1000"
    
    def test_ie_parsing_performance(self):
        """Test IE-Parsing-Performance."""
        # Erstelle große IE-Struktur
        large_ies = {}
        for i in range(50):  # 50 verschiedene IEs
            large_ies[i] = [f"hexdata{i:04x}" * 10]  # Jede IE mit 40 Zeichen
        
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            utils.parse_ies(large_ies, detailed=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        iterations_per_second = num_iterations / processing_time
        
        # Sollte mindestens 50 Iterationen pro Sekunde schaffen
        assert iterations_per_second > 50, f"Only {iterations_per_second:.0f} iterations/second, expected > 50"
    
    def test_vendor_lookup_performance(self):
        """Test Vendor-Lookup-Performance."""
        test_macs = [
            "a8:51:ab:0c:b9:e9",  # Apple
            "b2:87:23:15:7f:f2",  # Randomisiert
            "08:96:d7:1a:21:1c",  # Unbekannt
        ] * 100  # 300 MACs
        
        start_time = time.time()
        
        for mac in test_macs:
            utils.lookup_vendor(mac)
        
        end_time = time.time()
        processing_time = end_time - start_time
        lookups_per_second = len(test_macs) / processing_time
        
        # Sollte mindestens 1000 Lookups pro Sekunde schaffen
        assert lookups_per_second > 1000, f"Only {lookups_per_second:.0f} lookups/second, expected > 1000"


@pytest.mark.performance
class TestAnalysisPerformance:
    """Tests für Analyse-Performance."""
    
    def test_client_feature_extraction_performance(self):
        """Test Client-Feature-Extraktion-Performance."""
        # Erstelle Test-Client
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        client.probes = {"SSID1", "SSID2", "SSID3"}
        client.seen_with = {"bssid1", "bssid2"}
        client.all_packet_ts = np.array([time.time() - 100 + i for i in range(100)])
        client.rssi_w = Welford()
        for i in range(100):
            client.rssi_w.update(-50 - i)
        client.parsed_ies = {
            "standards": ["802.11n", "802.11ac"],
            "ht_caps": {"streams": 2},
            "vendor_specific": {"Apple": True}
        }
        
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            analysis.features_for_client(client)
            analysis.features_for_client_behavior(client)
        
        end_time = time.time()
        processing_time = end_time - start_time
        iterations_per_second = num_iterations / processing_time
        
        # Sollte mindestens 50 Iterationen pro Sekunde schaffen
        assert iterations_per_second > 50, f"Only {iterations_per_second:.0f} iterations/second, expected > 50"
    
    def test_clustering_performance(self):
        """Test Clustering-Performance."""
        # Erstelle State mit vielen Clients
        state = WifiAnalysisState()
        
        from wlan_tool.storage.data_models import ClientState, Welford
        
        for i in range(200):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.probes = {f"SSID_{i % 20}"}
            client.all_packet_ts = np.array([time.time() - 100 + j for j in range(10)])
            client.rssi_w = Welford()
            for j in range(10):
                client.rssi_w.update(-50 - i)
            client.parsed_ies = {
                "standards": ["802.11n"] if i % 2 == 0 else ["802.11ac"],
                "ht_caps": {"streams": 1 + (i % 4)},
                "vendor_specific": {"Apple": i % 3 == 0}
            }
            state.clients[mac] = client
        
        start_time = time.time()
        
        clustered_df, feature_df = analysis.cluster_clients(state, n_clusters=5)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Sollte unter 5 Sekunden für 200 Clients sein
        assert processing_time < 5.0, f"Clustering took {processing_time:.2f}s, expected < 5.0s"
        assert clustered_df is not None
        assert len(clustered_df) == 200
    
    def test_inference_performance(self):
        """Test Inferenz-Performance."""
        # Erstelle State mit APs und Clients
        state = WifiAnalysisState()
        
        from wlan_tool.storage.data_models import APState, ClientState, Welford
        
        # Erstelle 50 APs
        for i in range(50):
            bssid = f"08:96:d7:1a:21:{i:02x}"
            ap = APState(bssid=bssid, ssid=f"AP_{i}")
            ap.channel = (i % 13) + 1
            ap.beacon_count = 10 + i
            ap.rssi_w = Welford()
            ap.rssi_w.update(-50 - i)
            state.aps[bssid] = ap
            state.ssid_map[ap.ssid] = {"bssids": {bssid}, "sources": {}}
        
        # Erstelle 100 Clients
        for i in range(100):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.probes = {f"AP_{i % 50}"}
            client.seen_with = {f"08:96:d7:1a:21:{i % 50:02x}"}
            client.all_packet_ts = np.array([time.time() - 100 + j for j in range(10)])
            client.rssi_w = Welford()
            for j in range(10):
                client.rssi_w.update(-50 - i)
            state.clients[mac] = client
        
        start_time = time.time()
        
        results = analysis.score_pairs_with_recency_and_matching(state)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Sollte unter 2 Sekunden für 50 APs und 100 Clients sein
        assert processing_time < 2.0, f"Inference took {processing_time:.2f}s, expected < 2.0s"
        assert isinstance(results, list)


@pytest.mark.performance
class TestMemoryPerformance:
    """Tests für Speicher-Performance."""
    
    def test_memory_usage_large_state(self):
        """Test Speicherverbrauch mit großem State."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Erstelle großen State
        state = WifiAnalysisState()
        
        from wlan_tool.storage.data_models import APState, ClientState, Welford
        
        # 1000 Clients
        for i in range(1000):
            mac = f"aa:bb:cc:dd:ee:{i:04x}"
            client = ClientState(mac=mac)
            client.probes = {f"SSID_{i % 100}"}
            client.all_packet_ts = np.array([time.time() - 100 + j for j in range(20)])
            client.rssi_w = Welford()
            for j in range(20):
                client.rssi_w.update(-50 - i)
            client.parsed_ies = {
                "standards": ["802.11n", "802.11ac"],
                "ht_caps": {"streams": 1 + (i % 4)},
                "vendor_specific": {"Apple": i % 3 == 0, "Samsung": i % 5 == 0}
            }
            state.clients[mac] = client
        
        # 100 APs
        for i in range(100):
            bssid = f"08:96:d7:1a:21:{i:02x}"
            ap = APState(bssid=bssid, ssid=f"AP_{i}")
            ap.channel = (i % 13) + 1
            ap.beacon_count = 10 + i
            ap.rssi_w = Welford()
            ap.rssi_w.update(-50 - i)
            state.aps[bssid] = ap
            state.ssid_map[ap.ssid] = {"bssids": {bssid}, "sources": {}}
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Speicherzunahme sollte unter 200MB sein
        assert memory_increase < 200 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB, expected < 200MB"
    
    def test_memory_cleanup_after_pruning(self):
        """Test Speicherbereinigung nach Pruning."""
        process = psutil.Process(os.getpid())
        
        # Erstelle State mit alten und neuen Daten
        state = WifiAnalysisState()
        
        from wlan_tool.storage.data_models import ClientState, Welford
        
        # 500 alte Clients (werden gepruned)
        for i in range(500):
            mac = f"old:aa:bb:cc:dd:ee:{i:03x}"
            client = ClientState(mac=mac)
            client.last_seen = time.time() - 10000  # Sehr alt
            client.all_packet_ts = np.array([time.time() - 10000 + j for j in range(50)])
            client.rssi_w = Welford()
            for j in range(50):
                client.rssi_w.update(-50 - i)
            state.clients[mac] = client
        
        # 500 neue Clients (bleiben)
        for i in range(500):
            mac = f"new:aa:bb:cc:dd:ee:{i:03x}"
            client = ClientState(mac=mac)
            client.last_seen = time.time()  # Neu
            client.all_packet_ts = np.array([time.time() - 100 + j for j in range(10)])
            client.rssi_w = Welford()
            for j in range(10):
                client.rssi_w.update(-50 - i)
            state.clients[mac] = client
        
        initial_memory = process.memory_info().rss
        
        # Führe Pruning durch
        pruned_count = state.prune_state(time.time(), threshold_s=7200)
        
        final_memory = process.memory_info().rss
        memory_change = final_memory - initial_memory
        
        assert pruned_count > 0
        assert len(state.clients) == 500  # Nur neue Clients sollten bleiben
        # Speicher sollte reduziert worden sein
        assert memory_change < 50 * 1024 * 1024, f"Memory change: {memory_change / 1024 / 1024:.1f}MB, expected < 50MB"


@pytest.mark.performance
class TestDatabasePerformance:
    """Tests für Datenbank-Performance."""
    
    def test_database_write_performance(self, temp_db_file):
        """Test Datenbank-Schreib-Performance."""
        # Erstelle Test-Events
        events = []
        for i in range(1000):
            event = {
                'ts': time.time() + i,
                'type': 'beacon' if i % 2 == 0 else 'probe_req',
                'bssid': f'08:96:d7:1a:21:{i % 100:02x}',
                'ssid': f'SSID_{i % 50}',
                'rssi': -50 - (i % 50),
                'channel': (i % 13) + 1
            }
            events.append(event)
        
        start_time = time.time()
        
        with database.db_conn_ctx(temp_db_file) as conn:
            for event in events:
                database.add_event(conn, event)
        
        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = len(events) / processing_time
        
        # Sollte mindestens 100 Events pro Sekunde schreiben können
        assert events_per_second > 100, f"Only {events_per_second:.0f} events/second, expected > 100"
    
    def test_database_read_performance(self, temp_db_file):
        """Test Datenbank-Lese-Performance."""
        # Schreibe Test-Events
        events = []
        for i in range(1000):
            event = {
                'ts': time.time() + i,
                'type': 'beacon' if i % 2 == 0 else 'probe_req',
                'bssid': f'08:96:d7:1a:21:{i % 100:02x}',
                'ssid': f'SSID_{i % 50}',
                'rssi': -50 - (i % 50),
                'channel': (i % 13) + 1
            }
            events.append(event)
        
        with database.db_conn_ctx(temp_db_file) as conn:
            for event in events:
                database.add_event(conn, event)
        
        # Teste Lese-Performance
        start_time = time.time()
        
        with database.db_conn_ctx(temp_db_file) as conn:
            loaded_events = list(database.fetch_events(conn))
        
        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = len(loaded_events) / processing_time
        
        # Sollte mindestens 500 Events pro Sekunde lesen können
        assert events_per_second > 500, f"Only {events_per_second:.0f} events/second, expected > 500"
        assert len(loaded_events) == 1000


@pytest.mark.performance
class TestConcurrentPerformance:
    """Tests für gleichzeitige Verarbeitung."""
    
    def test_concurrent_state_updates(self):
        """Test gleichzeitige State-Updates."""
        import threading
        import queue
        
        state = WifiAnalysisState()
        event_queue = queue.Queue()
        
        # Erstelle Test-Events
        for i in range(1000):
            event = {
                'ts': time.time() + i,
                'type': 'beacon' if i % 2 == 0 else 'probe_req',
                'bssid': f'08:96:d7:1a:21:{i % 100:02x}',
                'ssid': f'SSID_{i % 50}',
                'rssi': -50 - (i % 50),
                'channel': (i % 13) + 1
            }
            event_queue.put(event)
        
        def worker():
            while True:
                try:
                    event = event_queue.get_nowait()
                    state.update_from_event(event)
                    event_queue.task_done()
                except queue.Empty:
                    break
        
        # Starte mehrere Worker-Threads
        num_threads = 4
        threads = []
        
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Warte auf alle Threads
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Sollte unter 1 Sekunde für 1000 Events mit 4 Threads sein
        assert processing_time < 1.0, f"Concurrent processing took {processing_time:.2f}s, expected < 1.0s"
        assert len(state.aps) > 0 or len(state.clients) > 0


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityPerformance:
    """Tests für Skalierbarkeits-Performance."""
    
    def test_scalability_with_dataset_size(self):
        """Test Skalierbarkeit mit Datensatz-Größe."""
        dataset_sizes = [100, 500, 1000]
        processing_times = []
        
        for size in dataset_sizes:
            # Erstelle State mit gegebener Größe
            state = WifiAnalysisState()
            
            from wlan_tool.storage.data_models import ClientState, Welford
            
            for i in range(size):
                mac = f"aa:bb:cc:dd:ee:{i:04x}"
                client = ClientState(mac=mac)
                client.probes = {f"SSID_{i % 20}"}
                client.all_packet_ts = np.array([time.time() - 100 + j for j in range(10)])
                client.rssi_w = Welford()
                for j in range(10):
                    client.rssi_w.update(-50 - i)
                state.clients[mac] = client
            
            # Teste Clustering-Performance
            start_time = time.time()
            clustered_df, feature_df = analysis.cluster_clients(state, n_clusters=5)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
        
        # Verarbeitungszeit sollte linear oder sublinear mit der Datensatz-Größe skalieren
        # (nicht quadratisch)
        time_ratio_500_100 = processing_times[1] / processing_times[0]
        time_ratio_1000_500 = processing_times[2] / processing_times[1]
        
        # Verhältnis sollte nicht zu groß sein (maximal 3x für 5x mehr Daten)
        assert time_ratio_500_100 < 3.0, f"Time ratio 500/100: {time_ratio_500_100:.2f}, expected < 3.0"
        assert time_ratio_1000_500 < 3.0, f"Time ratio 1000/500: {time_ratio_1000_500:.2f}, expected < 3.0"
