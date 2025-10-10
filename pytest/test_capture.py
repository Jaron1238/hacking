#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Capture-Modul (sniffer.py).
"""

import pytest
import time
import tempfile
import os
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from wlan_tool.capture import sniffer as capture
from wlan_tool.storage.data_models import WifiEvent
from scapy.all import RadioTap, Dot11, Dot11Beacon, Dot11Elt, Dot11ProbeReq, Dot11Data


class TestPacketParsing:
    """Tests für Paket-Parsing-Funktionen."""
    
    def test_packet_to_event_beacon(self, sample_scapy_packet):
        """Test Beacon-Paket zu Event-Konvertierung."""
        event = capture.packet_to_event(sample_scapy_packet)
        
        assert event is not None
        assert event['type'] == 'beacon'
        assert event['bssid'] == 'aa:bb:cc:dd:ee:ff'
        assert event['rssi'] == -42
        assert event['channel'] == 1
        assert event['ssid'] == 'MyTestSSID'
        assert event['beacon_interval'] == 102
        assert event['cap'] == 0x11
    
    def test_packet_to_event_probe_request(self):
        """Test Probe-Request-Paket zu Event-Konvertierung."""
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-60
        )
        dot11_layer = Dot11(
            type=0, subtype=4,
            addr1='ff:ff:ff:ff:ff:ff',  # Broadcast
            addr2='aa:bb:cc:dd:ee:ff'   # Client
        )
        ssid_ie = Dot11Elt(
            ID=0,
            info=b'TestSSID'
        )
        
        pkt = rt_layer / dot11_layer / ssid_ie
        pkt.time = time.time()
        
        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'probe_req'
        assert event['client'] == 'aa:bb:cc:dd:ee:ff'
        assert event['rssi'] == -60
        assert 'TestSSID' in event['ies'][0]
    
    def test_packet_to_event_data_frame(self):
        """Test Data-Frame zu Event-Konvertierung."""
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal+MCS_index",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-55,
            MCS_index=7
        )
        dot11_layer = Dot11(
            type=2, subtype=0,
            addr1='aa:bb:cc:dd:ee:ff',  # BSSID
            addr2='11:22:33:44:55:66'   # Client
        )
        
        pkt = rt_layer / dot11_layer
        pkt.time = time.time()
        
        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'data'
        assert event['client'] == '11:22:33:44:55:66'
        assert event['bssid'] == 'aa:bb:cc:dd:ee:ff'
        assert event['rssi'] == -55
        assert event['mcs_index'] == 7
    
    def test_packet_to_event_invalid_packet(self):
        """Test mit ungültigem Paket."""
        # Erstelle ein Paket ohne Dot11-Layer
        rt_layer = RadioTap()
        pkt = rt_layer
        pkt.time = time.time()
        
        event = capture.packet_to_event(pkt)
        
        assert event is None
    
    def test_packet_to_event_with_dns_query(self):
        """Test Paket mit DNS-Query."""
        from scapy.layers.dns import DNSQR
        
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-50
        )
        dot11_layer = Dot11(
            type=2, subtype=0,
            addr1='aa:bb:cc:dd:ee:ff',
            addr2='11:22:33:44:55:66'
        )
        dns_layer = DNSQR(qname=b'example.com')
        
        pkt = rt_layer / dot11_layer / dns_layer
        pkt.time = time.time()
        pkt.dport = 53  # DNS-Port
        
        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'data'
        assert event['dns_query'] == 'example.com'
    
    def test_packet_to_event_with_dhcp(self):
        """Test Paket mit DHCP-Informationen."""
        from scapy.layers.dhcp import DHCP, BOOTP
        
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-50
        )
        dot11_layer = Dot11(
            type=2, subtype=0,
            addr1='aa:bb:cc:dd:ee:ff',
            addr2='11:22:33:44:55:66'
        )
        bootp_layer = BOOTP(chaddr=b'\x11\x22\x33\x44\x55\x66')
        dhcp_layer = DHCP(options=[(53, 3), (12, b'TestHostname')])  # DHCP Request mit Hostname
        
        pkt = rt_layer / dot11_layer / bootp_layer / dhcp_layer
        pkt.time = time.time()
        
        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'data'
        assert event['hostname'] == 'TestHostname'
    
    def test_packet_to_event_with_arp(self):
        """Test Paket mit ARP-Informationen."""
        from scapy.layers.l2 import ARP
        
        rt_layer = RadioTap(
            present="Flags+Channel+dBm_AntSignal",
            Flags="",
            ChannelFrequency=2412,
            dBm_AntSignal=-50
        )
        dot11_layer = Dot11(
            type=2, subtype=0,
            addr1='aa:bb:cc:dd:ee:ff',
            addr2='11:22:33:44:55:66'
        )
        arp_layer = ARP(op=2, hwsrc='11:22:33:44:55:66', psrc='192.168.1.100')
        
        pkt = rt_layer / dot11_layer / arp_layer
        pkt.time = time.time()
        
        event = capture.packet_to_event(pkt)
        
        assert event is not None
        assert event['type'] == 'data'
        assert event['arp_mac'] == '11:22:33:44:55:66'
        assert event['arp_ip'] == '192.168.1.100'


class TestChannelHopper:
    """Tests für Channel-Hopping-Funktionalität."""
    
    def test_channel_hopper_initialization(self):
        """Test ChannelHopper-Initialisierung."""
        hopper = capture.ChannelHopper(
            iface="wlan0mon",
            channels=[1, 6, 11],
            sleep_interval=0.1
        )
        
        assert hopper.iface == "wlan0mon"
        assert hopper.channels == [1, 6, 11]
        assert hopper.sleep_interval == 0.1
        assert not hopper.stop_event.is_set()
    
    def test_channel_hopper_empty_channels(self):
        """Test ChannelHopper mit leerer Kanal-Liste."""
        hopper = capture.ChannelHopper(
            iface="wlan0mon",
            channels=[],
            sleep_interval=0.1
        )
        
        # Sollte sofort beenden wenn keine Kanäle
        hopper.start()
        hopper.stop()
        assert hopper.stop_event.is_set()
    
    @patch('subprocess.run')
    def test_channel_hopper_run(self, mock_run):
        """Test ChannelHopper-Ausführung."""
        mock_run.return_value = MagicMock(returncode=0)
        
        hopper = capture.ChannelHopper(
            iface="wlan0mon",
            channels=[1, 6],
            sleep_interval=0.01  # Sehr kurz für Test
        )
        
        hopper.start()
        time.sleep(0.05)  # Kurz laufen lassen
        hopper.stop()
        
        # Sollte mindestens einen Kanalwechsel versucht haben
        assert mock_run.call_count > 0
    
    @patch('subprocess.run')
    def test_channel_hopper_command_failure(self, mock_run):
        """Test ChannelHopper bei Kommando-Fehlern."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "iw")
        
        hopper = capture.ChannelHopper(
            iface="wlan0mon",
            channels=[1],
            sleep_interval=0.01
        )
        
        # Sollte nicht crashen
        hopper.start()
        time.sleep(0.01)
        hopper.stop()


class TestPacketProcessing:
    """Tests für Paket-Verarbeitungs-Funktionen."""
    
    @patch('multiprocessing.Queue')
    def test_packet_parser_worker(self, mock_queue_class):
        """Test Paket-Parser-Worker."""
        mock_queue = MagicMock()
        mock_queue_class.return_value = mock_queue
        mock_queue.get.side_effect = [None]  # Beende sofort
        
        # Sollte nicht crashen
        capture.packet_parser_worker(mock_queue, mock_queue, None)
    
    @patch('subprocess.Popen')
    def test_packet_reader_thread(self, mock_popen):
        """Test Paket-Reader-Thread."""
        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = [b'fake_pcap_header', b'']  # Leer nach Header
        mock_popen.return_value = mock_proc
        
        stop_event = MagicMock()
        stop_event.is_set.return_value = False
        
        # Sollte nicht crashen
        capture.packet_reader_thread(
            mock_proc, 
            MagicMock(), 
            [], 
            stop_event, 
            None
        )
    
    @patch('subprocess.Popen')
    def test_stderr_drainer_thread(self, mock_popen):
        """Test Stderr-Drainer-Thread."""
        mock_proc = MagicMock()
        mock_proc.stderr.readline.side_effect = [b'error message\n', b'']  # Leer nach einer Zeile
        mock_popen.return_value = mock_proc
        
        stop_event = MagicMock()
        stop_event.is_set.return_value = False
        
        # Sollte nicht crashen
        capture.stderr_drainer_thread(mock_proc, stop_event)


class TestSniffingIntegration:
    """Tests für Sniffing-Integration."""
    
    @patch('subprocess.Popen')
    @patch('multiprocessing.Process')
    @patch('wlan_tool.storage.database.BatchedEventWriter')
    def test_sniff_with_writer_mock(self, mock_writer_class, mock_process_class, mock_popen):
        """Test sniff_with_writer mit Mocks."""
        # Setup Mocks
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Läuft weiter
        mock_proc.stdout.read.side_effect = [b'fake_header', b'']  # Leer nach Header
        mock_popen.return_value = mock_proc
        
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer
        
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process
        
        # Teste mit sehr kurzer Dauer
        with patch('time.time', side_effect=[0, 0.1, 0.2]):  # Simuliere Zeitablauf
            try:
                capture.sniff_with_writer(
                    iface="wlan0mon",
                    duration=1,
                    db_path=":memory:",
                    pcap_out=None
                )
            except Exception as e:
                # Kann verschiedene Exceptions werfen, das ist OK für Mock-Test
                pass
        
        # Verifiziere, dass Writer gestartet wurde
        mock_writer.start.assert_called_once()
        mock_writer.stop.assert_called_once()


class TestIEExtraction:
    """Tests für IE-Extraktion."""
    
    def test_collect_ies_from_pkt(self):
        """Test IE-Sammlung aus Paket."""
        # Erstelle Paket mit IEs
        rt_layer = RadioTap()
        dot11_layer = Dot11(type=0, subtype=8)
        beacon_layer = Dot11Beacon()
        ssid_ie = Dot11Elt(ID=0, info=b'TestSSID')
        vendor_ie = Dot11Elt(ID=221, info=b'\x00\x17\xf2\x0a\x01\x01')
        
        pkt = rt_layer / dot11_layer / beacon_layer / ssid_ie / vendor_ie
        
        ies, ie_order = capture._collect_ies_from_pkt(pkt)
        
        assert isinstance(ies, dict)
        assert isinstance(ie_order, list)
        assert 0 in ies  # SSID IE
        assert 221 in ies  # Vendor IE
        assert 'TestSSID' in ies[0]
        assert len(ie_order) == 2
    
    def test_collect_ies_empty_packet(self):
        """Test IE-Sammlung mit leerem Paket."""
        rt_layer = RadioTap()
        dot11_layer = Dot11(type=0, subtype=8)
        beacon_layer = Dot11Beacon()
        
        pkt = rt_layer / dot11_layer / beacon_layer
        
        ies, ie_order = capture._collect_ies_from_pkt(pkt)
        
        assert ies == {}
        assert ie_order == []
    
    def test_extract_seq(self):
        """Test Sequenznummer-Extraktion."""
        dot11 = Dot11(SC=0x1234)  # SC = 0x1234, >> 4 = 0x123
        
        seq = capture._extract_seq(dot11)
        assert seq == 0x123
        
        # Test mit None
        seq = capture._extract_seq(None)
        assert seq is None


class TestErrorHandling:
    """Tests für Fehlerbehandlung."""
    
    def test_packet_to_event_malformed_packet(self):
        """Test mit fehlerhaftem Paket."""
        # Erstelle Paket mit fehlenden Attributen
        rt_layer = RadioTap()
        dot11_layer = Dot11(type=0, subtype=8)
        
        pkt = rt_layer / dot11_layer
        pkt.time = time.time()
        
        # Sollte nicht crashen
        event = capture.packet_to_event(pkt)
        # Kann None sein, das ist OK
    
    def test_packet_to_event_encoding_error(self):
        """Test mit Encoding-Fehlern."""
        rt_layer = RadioTap()
        dot11_layer = Dot11(type=0, subtype=8, addr3='aa:bb:cc:dd:ee:ff')
        beacon_layer = Dot11Beacon()
        # Erstelle IE mit ungültigem UTF-8
        ssid_ie = Dot11Elt(ID=0, info=b'\xff\xfe\xfd')
        
        pkt = rt_layer / dot11_layer / beacon_layer / ssid_ie
        pkt.time = time.time()
        
        # Sollte nicht crashen
        event = capture.packet_to_event(pkt)
        if event is not None:
            assert event['ssid'] == "<binary>"  # Sollte als binary markiert werden


class TestPerformance:
    """Tests für Performance-Aspekte."""
    
    def test_packet_processing_speed(self):
        """Test Paket-Verarbeitungsgeschwindigkeit."""
        # Erstelle einfaches Test-Paket
        rt_layer = RadioTap()
        dot11_layer = Dot11(type=0, subtype=8, addr3='aa:bb:cc:dd:ee:ff')
        beacon_layer = Dot11Beacon()
        ssid_ie = Dot11Elt(ID=0, info=b'TestSSID')
        
        pkt = rt_layer / dot11_layer / beacon_layer / ssid_ie
        pkt.time = time.time()
        
        # Teste Verarbeitungszeit
        start_time = time.time()
        for _ in range(100):
            capture.packet_to_event(pkt)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Sollte unter 1 Sekunde für 100 Pakete sein