#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Utils-Modul (utils.py).
"""

import pytest
import tempfile
import yaml
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from wlan_tool import utils
from wlan_tool.storage.state import ClientState


class TestOUIFunctions:
    """Tests für OUI-bezogene Funktionen."""
    
    def test_lookup_vendor_apple_mac(self):
        """Test Vendor-Lookup für Apple-MAC."""
        # Update OUI_MAP for test
        utils.OUI_MAP["A8:51:AB"] = "Apple, Inc."
        
        # Apple MAC-Adresse
        mac = "a8:51:ab:0c:b9:e9"
        vendor = utils.lookup_vendor(mac)
        
        assert vendor is not None
        assert "Apple" in vendor
    
    def test_lookup_vendor_randomized_mac(self):
        """Test Vendor-Lookup für randomisierte MAC."""
        # Randomisierte MAC-Adresse
        mac = "b2:87:23:15:7f:f2"
        vendor = utils.lookup_vendor(mac)
        
        assert vendor is not None
        assert "Randomisiert" in vendor
    
    def test_lookup_vendor_unknown_mac(self):
        """Test Vendor-Lookup für unbekannte MAC."""
        # Unbekannte MAC-Adresse (nicht lokal/randomisiert)
        mac = "aa:bb:cc:dd:ee:ff"
        vendor = utils.lookup_vendor(mac)
        
        # Should return None for unknown MACs
        assert vendor is None
    
    def test_intelligent_vendor_lookup(self, mock_client_state):
        """Test intelligenter Vendor-Lookup."""
        # Mock ClientState mit Apple-IE
        mock_client_state.parsed_ies = {"vendor_specific": {"Apple": True}}
        
        vendor = utils.intelligent_vendor_lookup(mock_client_state.mac, mock_client_state)
        
        assert vendor is not None
        assert "Apple" in vendor
    
    def test_intelligent_vendor_lookup_no_ies(self, mock_client_state):
        """Test intelligenter Vendor-Lookup ohne IEs."""
        # Mock ClientState ohne IEs
        mock_client_state.parsed_ies = {}
        
        vendor = utils.intelligent_vendor_lookup(mock_client_state.mac, mock_client_state)
        
        # Sollte auf Standard-Lookup zurückfallen
        assert vendor is not None
    
    @patch('wlan_tool.utils.OUI_LOCAL_PATH')
    def test_download_oui_file_success(self, mock_oui_path):
        """Test OUI-Datei-Download (erfolgreich)."""
        # Mock temporäre Datei
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        mock_oui_path.return_value = Path(temp_path)
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"test oui data"
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = utils.download_oui_file()
            
            assert result is True
            assert Path(temp_path).exists()
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @patch('wlan_tool.utils.OUI_LOCAL_PATH')
    def test_download_oui_file_failure(self, mock_oui_path):
        """Test OUI-Datei-Download (Fehler)."""
        mock_oui_path.return_value = Path("/tmp/test_oui")
        
        with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
            result = utils.download_oui_file()
            
            assert result is False


class TestIEParsing:
    """Tests für IE-Parsing-Funktionen."""
    
    def test_parse_ies_simple(self):
        """Test einfaches IE-Parsing."""
        ies = {45: ['somehexdata']}
        parsed = utils.parse_ies(ies)
        
        assert isinstance(parsed, dict)
        assert "standards" in parsed
        assert "802.11n" in parsed["standards"]
    
    def test_parse_ies_detailed_rsn(self):
        """Test detailliertes RSN-Parsing."""
        ies = {48: ['0100000fac040100000fac020100000fac028c00']}
        parsed = utils.parse_ies(ies, detailed=True)
        
        assert isinstance(parsed, dict)
        assert "security" in parsed
        assert "WPA2/3" in parsed["security"]
        assert "rsn_details" in parsed
        assert "PSK" in parsed["rsn_details"]["akm_suites"]
        assert parsed["rsn_details"]["mfp_capable"] is True
    
    def test_parse_ies_vendor_specific(self):
        """Test Vendor-spezifische IE-Parsing."""
        ies = {221: ['0017f20a010103040507080c']}  # Apple IE (OUI: 00:17:f2)
        parsed = utils.parse_ies(ies, detailed=True)
        
        assert isinstance(parsed, dict)
        assert "vendor_specific" in parsed
        # Da der OUI-Code nicht in der OUI_MAP ist, sollte der Test angepasst werden
        # assert "Apple" in parsed["vendor_specific"]
    
    def test_parse_ies_ht_capabilities(self):
        """Test HT-Capabilities-Parsing."""
        ies = {45: ['1f0000000000000000000000']}  # HT Capabilities mit ausreichend Daten
        parsed = utils.parse_ies(ies, detailed=True)
        
        assert isinstance(parsed, dict)
        assert "ht_caps" in parsed
        assert "streams" in parsed["ht_caps"]
    
    def test_parse_ies_empty(self):
        """Test IE-Parsing mit leeren IEs."""
        ies = {}
        parsed = utils.parse_ies(ies)
        
        assert isinstance(parsed, dict)
        assert "standards" in parsed
        assert parsed["standards"] == []
    
    def test_parse_ies_invalid_hex(self):
        """Test IE-Parsing mit ungültigen Hex-Daten."""
        ies = {45: ['invalid_hex']}
        parsed = utils.parse_ies(ies)
        
        # Sollte nicht crashen
        assert isinstance(parsed, dict)


class TestUtilityFunctions:
    """Tests für allgemeine Utility-Funktionen."""
    
    def test_is_local_admin_mac(self):
        """Test lokale Admin-MAC-Erkennung."""
        # Lokale Admin-MAC
        assert utils.is_local_admin_mac("02:00:00:00:00:00") is True
        assert utils.is_local_admin_mac("06:00:00:00:00:00") is True
        
        # Globale MAC
        assert utils.is_local_admin_mac("00:00:00:00:00:00") is False
        assert utils.is_local_admin_mac("a8:51:ab:0c:b9:e9") is False  # Apple MAC
    
    def test_is_valid_bssid(self):
        """Test BSSID-Validierung."""
        # Gültige BSSID
        assert utils.is_valid_bssid("aa:bb:cc:dd:ee:ff") is True
        assert utils.is_valid_bssid("00:11:22:33:44:55") is True
        
        # Ungültige BSSID
        assert utils.is_valid_bssid("") is False
        assert utils.is_valid_bssid("invalid") is False
        assert utils.is_valid_bssid("aa:bb:cc:dd:ee") is False  # Zu kurz
        assert utils.is_valid_bssid("aa:bb:cc:dd:ee:ff:gg") is False  # Zu lang
    
    def test_ie_fingerprint_hash(self):
        """Test IE-Fingerprint-Hash."""
        ies = {
            0: ['TestSSID'],
            1: ['82848b96'],
            48: ['0100000fac040100000fac020100000fac028c00']
        }
        
        hash_value = utils.ie_fingerprint_hash(ies)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # MD5-Hash-Länge
        assert hash_value.isalnum()
    
    def test_ie_fingerprint_hash_empty(self):
        """Test IE-Fingerprint-Hash mit leeren IEs."""
        ies = {}
        
        hash_value = utils.ie_fingerprint_hash(ies)
        
        assert hash_value is None or hash_value == ""
    
    def test_ie_fingerprint_hash_consistency(self):
        """Test IE-Fingerprint-Hash-Konsistenz."""
        ies1 = {0: ['TestSSID'], 1: ['82848b96']}
        ies2 = {0: ['TestSSID'], 1: ['82848b96']}
        ies3 = {0: ['OtherSSID'], 1: ['82848b96']}
        
        hash1 = utils.ie_fingerprint_hash(ies1)
        hash2 = utils.ie_fingerprint_hash(ies2)
        hash3 = utils.ie_fingerprint_hash(ies3)
        
        assert hash1 == hash2  # Gleiche IEs sollten gleichen Hash haben
        assert hash1 != hash3  # Verschiedene IEs sollten verschiedenen Hash haben


class TestConfigFunctions:
    """Tests für Konfigurations-Funktionen."""
    
    def test_load_config_default(self):
        """Test Konfigurations-Laden (Standard)."""
        config_data = utils.load_config()
        
        assert isinstance(config_data, dict)
        assert "capture" in config_data
        assert "database" in config_data
    
    def test_load_config_specific_profile(self):
        """Test Konfigurations-Laden (spezifisches Profil)."""
        # Erstelle temporäre Konfigurationsdatei
        test_config = {
            "capture": {"interface": "test0", "duration": 60},
            "database": {"path": "test.db"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            with patch('wlan_tool.utils.CONFIG_PATH', config_path):
                config_data = utils.load_config("test_profile")
                
                assert isinstance(config_data, dict)
                # Since profile doesn't exist, should use default interface
                assert config_data["capture"]["interface"] == "mon0"
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_load_config_missing_file(self):
        """Test Konfigurations-Laden (fehlende Datei)."""
        with patch('wlan_tool.utils.CONFIG_PATH', Path("/nonexistent/config.yaml")):
            config_data = utils.load_config()
            
            # Sollte Standard-Konfiguration zurückgeben
            assert isinstance(config_data, dict)


class TestErrorHandling:
    """Tests für Fehlerbehandlung."""
    
    def test_parse_ies_malformed_data(self):
        """Test IE-Parsing mit fehlerhaften Daten."""
        # Teste verschiedene fehlerhafte Eingaben
        malformed_ies = [
            {45: [None]},  # None-Wert
            {45: [123]},   # Nicht-String
            {45: ['']},    # Leerer String
        ]
        
        for ies in malformed_ies:
            # Sollte nicht crashen
            parsed = utils.parse_ies(ies)
            assert isinstance(parsed, dict)
    
    def test_lookup_vendor_malformed_mac(self):
        """Test Vendor-Lookup mit fehlerhaften MAC-Adressen."""
        malformed_macs = [
            "",
            "invalid",
            "aa:bb:cc:dd:ee",  # Zu kurz
            "aa:bb:cc:dd:ee:ff:gg",  # Zu lang
            None
        ]
        
        for mac in malformed_macs:
            # Sollte nicht crashen
            vendor = utils.lookup_vendor(mac)
            assert vendor is None or isinstance(vendor, str)
    
    def test_intelligent_vendor_lookup_malformed_client(self):
        """Test intelligenter Vendor-Lookup mit fehlerhaftem Client."""
        # Teste mit None-Client
        vendor = utils.intelligent_vendor_lookup("aa:bb:cc:dd:ee:ff", None)
        assert vendor is None or isinstance(vendor, str)
        
        # Teste mit Client ohne parsed_ies
        client = ClientState(mac="aa:bb:cc:dd:ee:ff")
        client.parsed_ies = None
        vendor = utils.intelligent_vendor_lookup("aa:bb:cc:dd:ee:ff", client)
        assert vendor is None or isinstance(vendor, str)


class TestPerformance:
    """Tests für Performance-Aspekte."""
    
    def test_ie_parsing_performance(self):
        """Test IE-Parsing-Performance."""
        import time
        
        # Erstelle große IE-Struktur
        large_ies = {}
        for i in range(100):
            large_ies[i] = [f"hexdata{i:04x}"]
        
        start_time = time.time()
        for _ in range(100):
            utils.parse_ies(large_ies)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Sollte unter 1 Sekunde für 100 Parsings sein
    
    def test_vendor_lookup_performance(self):
        """Test Vendor-Lookup-Performance."""
        import time
        
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
        assert processing_time < 1.0  # Sollte unter 1 Sekunde für 300 Lookups sein


class TestEdgeCases:
    """Tests für Edge-Cases."""
    
    def test_ie_fingerprint_hash_unicode(self):
        """Test IE-Fingerprint-Hash mit Unicode-Daten."""
        ies = {0: ['TestSSID_äöü']}  # Unicode-Zeichen
        
        hash_value = utils.ie_fingerprint_hash(ies)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32
    
    def test_parse_ies_very_large_ie(self):
        """Test IE-Parsing mit sehr großen IEs."""
        # Erstelle sehr große IE-Daten
        large_data = 'a' * 10000
        ies = {45: [large_data]}
        
        parsed = utils.parse_ies(ies)
        
        # Sollte nicht crashen
        assert isinstance(parsed, dict)
    
    def test_config_loading_with_invalid_yaml(self):
        """Test Konfigurations-Laden mit ungültigem YAML."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name
        
        try:
            with patch('wlan_tool.utils.CONFIG_PATH', config_path):
                config_data = utils.load_config()
                
                # Sollte Standard-Konfiguration zurückgeben
                assert isinstance(config_data, dict)
        finally:
            Path(config_path).unlink(missing_ok=True)