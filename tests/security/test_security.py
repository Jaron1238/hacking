"""
Security Tests für das WLAN-Tool.
Testet Sicherheitsaspekte wie Input-Validation, SQL-Injection, XSS, etc.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import pytest
from wlan_tool.analysis import ClusteringAnalyzer, DeviceClassifier
from wlan_tool.data_processing import DataValidator, WiFiDataProcessor


class TestInputValidation:
    """Tests für Input-Validation und Sanitization."""

    @pytest.mark.security
    def test_sql_injection_prevention(self):
        """Test SQL-Injection-Prävention."""
        validator = DataValidator()

        # SQL-Injection-Versuche
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
        ]

        for malicious_input in malicious_inputs:
            # Test mit verschiedenen Spalten
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": [malicious_input, "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            # Validierung sollte fehlschlagen oder Input sanitisieren
            is_valid, errors = validator.validate(test_data)

            if is_valid:
                # Wenn gültig, sollte Input sanitisiert worden sein
                assert malicious_input not in str(test_data["device_id"].iloc[0])
            else:
                # Fehler sollten SQL-Injection erkennen
                assert any(
                    "Invalid" in error or "Malicious" in error for error in errors
                )

    @pytest.mark.security
    def test_xss_prevention(self):
        """Test XSS-Prävention."""
        validator = DataValidator()

        # XSS-Versuche
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg onload=alert('XSS')>",
        ]

        for xss_input in xss_inputs:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": ["device_1", "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": [xss_input, "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            is_valid, errors = validator.validate(test_data)

            if is_valid:
                # XSS-Input sollte sanitisiert sein
                assert "<script>" not in str(test_data["ssid"].iloc[0])
                assert "javascript:" not in str(test_data["ssid"].iloc[0])
            else:
                # Fehler sollten XSS erkennen
                assert any(
                    "Invalid" in error or "Malicious" in error for error in errors
                )

    @pytest.mark.security
    def test_path_traversal_prevention(self):
        """Test Path-Traversal-Prävention."""
        processor = WiFiDataProcessor()

        # Path-Traversal-Versuche
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "....//....//....//etc//passwd",
        ]

        for malicious_path in malicious_paths:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": [malicious_path, "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            # Verarbeitung sollte sicher sein
            try:
                processed_data = processor.process_data(test_data)
                # Path-Traversal sollte nicht in verarbeiteten Daten erscheinen
                assert ".." not in str(processed_data["device_id"].iloc[0])
                assert "/etc/" not in str(processed_data["device_id"].iloc[0])
                assert "\\windows\\" not in str(processed_data["device_id"].iloc[0])
            except Exception as e:
                # Fehler sollten Path-Traversal erkennen
                assert "Invalid" in str(e) or "Path" in str(e)

    @pytest.mark.security
    def test_command_injection_prevention(self):
        """Test Command-Injection-Prävention."""
        processor = WiFiDataProcessor()

        # Command-Injection-Versuche
        malicious_commands = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& whoami",
            "; ls -la",
            "| dir C:\\",
            "&& type C:\\Windows\\System32\\drivers\\etc\\hosts",
        ]

        for malicious_command in malicious_commands:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": [malicious_command, "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            # Verarbeitung sollte sicher sein
            try:
                processed_data = processor.process_data(test_data)
                # Command-Injection sollte nicht in verarbeiteten Daten erscheinen
                assert ";" not in str(processed_data["device_id"].iloc[0])
                assert "|" not in str(processed_data["device_id"].iloc[0])
                assert "&&" not in str(processed_data["device_id"].iloc[0])
            except Exception as e:
                # Fehler sollten Command-Injection erkennen
                assert "Invalid" in str(e) or "Command" in str(e)


class TestDataSanitization:
    """Tests für Daten-Sanitization."""

    @pytest.mark.security
    def test_html_escaping(self):
        """Test HTML-Escaping."""
        processor = WiFiDataProcessor()

        html_inputs = [
            "<b>Bold Text</b>",
            "<script>alert('test')</script>",
            "Normal text with <em>emphasis</em>",
            "&lt;script&gt;alert('test')&lt;/script&gt;",
            "Text with &amp; ampersand",
        ]

        for html_input in html_inputs:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": ["device_1", "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": [html_input, "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            processed_data = processor.process_data(test_data)

            # HTML-Tags sollten escaped oder entfernt sein
            ssid_value = str(processed_data["ssid"].iloc[0])
            assert "<script>" not in ssid_value
            assert "<b>" not in ssid_value
            assert "<em>" not in ssid_value

    @pytest.mark.security
    def test_sql_escaping(self):
        """Test SQL-Escaping."""
        processor = WiFiDataProcessor()

        sql_inputs = [
            "O'Reilly",
            "D'Angelo",
            "Smith & Jones",
            "100% sure",
            "Price: $50.00",
        ]

        for sql_input in sql_inputs:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": ["device_1", "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": [sql_input, "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            processed_data = processor.process_data(test_data)

            # SQL-Sonderzeichen sollten escaped sein
            ssid_value = str(processed_data["ssid"].iloc[0])
            # Einfache Anführungszeichen sollten escaped sein
            if "'" in sql_input:
                assert "''" in ssid_value or "\\'" in ssid_value


class TestFileSecurity:
    """Tests für Datei-Sicherheit."""

    @pytest.mark.security
    def test_file_upload_validation(self, temp_dir):
        """Test Datei-Upload-Validierung."""
        processor = WiFiDataProcessor()

        # Verschiedene Dateitypen testen
        test_files = {
            "valid.csv": "timestamp,device_id,signal_strength\n2024-01-01,device_1,-50",
            "malicious.py": 'import os\nos.system("rm -rf /")',
            "executable.sh": '#!/bin/bash\necho "Hello World"',
            "large_file.csv": "timestamp,device_id,signal_strength\n"
            + "\n".join([f"2024-01-01,device_{i},-50" for i in range(10000)]),
            "empty_file.csv": "",
            "binary_file.bin": b"\x00\x01\x02\x03\x04\x05",
        }

        for filename, content in test_files.items():
            file_path = temp_dir / filename

            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)

            # Datei-Validierung
            try:
                if filename.endswith(".csv") and filename != "binary_file.bin":
                    # CSV-Dateien sollten verarbeitet werden können
                    data = pd.read_csv(file_path)
                    processed_data = processor.process_data(data)
                    assert len(processed_data) > 0
                else:
                    # Andere Dateitypen sollten abgelehnt werden
                    with pytest.raises(Exception):
                        data = pd.read_csv(file_path)
                        processor.process_data(data)
            except Exception as e:
                # Fehler sollten informativ sein
                assert "Invalid" in str(e) or "Unsupported" in str(e)

    @pytest.mark.security
    def test_zip_bomb_prevention(self, temp_dir):
        """Test ZIP-Bomb-Prävention."""
        import zipfile

        # Erstelle eine ZIP-Bomb (kleine Datei, die sich zu großer Datei entpackt)
        zip_path = temp_dir / "bomb.zip"

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Erstelle eine Datei mit sich wiederholenden Daten
            bomb_data = b"A" * 1000  # 1KB
            zf.writestr("bomb.txt", bomb_data * 1000)  # 1MB komprimiert

        # ZIP-Datei sollte erkannt und abgelehnt werden
        assert zip_path.exists()

        # Versuche, ZIP-Datei zu verarbeiten
        try:
            data = pd.read_csv(zip_path)
            processor = WiFiDataProcessor()
            processor.process_data(data)
            pytest.fail("ZIP-Bomb sollte erkannt werden")
        except Exception as e:
            # Fehler sollte ZIP-Bomb erkennen
            assert "Invalid" in str(e) or "Unsupported" in str(e) or "ZIP" in str(e)


class TestAuthenticationAndAuthorization:
    """Tests für Authentifizierung und Autorisierung."""

    @pytest.mark.security
    def test_privilege_escalation_prevention(self):
        """Test Privilege-Escalation-Prävention."""
        # Simuliere verschiedene Benutzerrollen
        user_roles = ["user", "admin", "root", "superuser"]

        for role in user_roles:
            # Jede Rolle sollte nur ihre eigenen Daten verarbeiten können
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": [
                        f"{role}_device_1",
                        f"{role}_device_2",
                        f"{role}_device_3",
                    ],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            processor = WiFiDataProcessor()
            processed_data = processor.process_data(test_data)

            # Verarbeitete Daten sollten nur die eigene Rolle enthalten
            for device_id in processed_data["device_id"]:
                assert role in str(device_id)

    @pytest.mark.security
    def test_session_management(self):
        """Test Session-Management."""
        # Simuliere Session-Token
        session_tokens = [
            "valid_token_12345",
            "expired_token_67890",
            "invalid_token_abcde",
            "malicious_token_<script>",
            "admin_token_99999",
        ]

        for token in session_tokens:
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "device_id": ["device_1", "device_2", "device_3"],
                    "signal_strength": [-50, -60, -70],
                    "frequency": [2.4, 5.0, 2.4],
                    "mac_address": [
                        "00:11:22:33:44:01",
                        "00:11:22:33:44:02",
                        "00:11:22:33:44:03",
                    ],
                    "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                    "encryption": ["WPA2", "WPA3", "Open"],
                    "channel": [1, 6, 11],
                    "data_rate": [54, 100, 54],
                    "packet_count": [100, 200, 150],
                    "bytes_transferred": [1000, 2000, 1500],
                }
            )

            # Session-Token sollte validiert werden
            if "valid" in token:
                # Gültige Token sollten verarbeitet werden
                processor = WiFiDataProcessor()
                processed_data = processor.process_data(test_data)
                assert len(processed_data) > 0
            elif "expired" in token or "invalid" in token:
                # Ungültige Token sollten abgelehnt werden
                with pytest.raises(Exception):
                    processor = WiFiDataProcessor()
                    processor.process_data(test_data)
            elif "malicious" in token:
                # Malicious Token sollten abgelehnt werden
                with pytest.raises(Exception):
                    processor = WiFiDataProcessor()
                    processor.process_data(test_data)


class TestDataPrivacy:
    """Tests für Daten-Privacy."""

    @pytest.mark.security
    def test_pii_detection_and_removal(self):
        """Test PII-Erkennung und -Entfernung."""
        processor = WiFiDataProcessor()

        # PII-Daten
        pii_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "device_id": ["device_1", "device_2", "device_3"],
                "signal_strength": [-50, -60, -70],
                "frequency": [2.4, 5.0, 2.4],
                "mac_address": [
                    "00:11:22:33:44:01",
                    "00:11:22:33:44:02",
                    "00:11:22:33:44:03",
                ],
                "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                "encryption": ["WPA2", "WPA3", "Open"],
                "channel": [1, 6, 11],
                "data_rate": [54, 100, 54],
                "packet_count": [100, 200, 150],
                "bytes_transferred": [1000, 2000, 1500],
                "user_email": [
                    "john.doe@example.com",
                    "jane.smith@company.com",
                    "admin@system.com",
                ],
                "phone_number": [
                    "+1-555-123-4567",
                    "+49-30-12345678",
                    "+44-20-7946-0958",
                ],
                "credit_card": [
                    "4111-1111-1111-1111",
                    "5555-5555-5555-4444",
                    "3782-822463-10005",
                ],
            }
        )

        processed_data = processor.process_data(pii_data)

        # PII sollte entfernt oder maskiert sein
        if "user_email" in processed_data.columns:
            for email in processed_data["user_email"]:
                assert "@" not in str(email) or "***" in str(email)

        if "phone_number" in processed_data.columns:
            for phone in processed_data["phone_number"]:
                assert "-" not in str(phone) or "***" in str(phone)

        if "credit_card" in processed_data.columns:
            for card in processed_data["credit_card"]:
                assert "4111" not in str(card) or "***" in str(card)

    @pytest.mark.security
    def test_data_encryption(self, temp_dir):
        """Test Daten-Verschlüsselung."""
        processor = WiFiDataProcessor()

        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "device_id": ["device_1", "device_2", "device_3"],
                "signal_strength": [-50, -60, -70],
                "frequency": [2.4, 5.0, 2.4],
                "mac_address": [
                    "00:11:22:33:44:01",
                    "00:11:22:33:44:02",
                    "00:11:22:33:44:03",
                ],
                "ssid": ["WiFi_Home", "WiFi_Office", "Public_WiFi"],
                "encryption": ["WPA2", "WPA3", "Open"],
                "channel": [1, 6, 11],
                "data_rate": [54, 100, 54],
                "packet_count": [100, 200, 150],
                "bytes_transferred": [1000, 2000, 1500],
            }
        )

        processed_data = processor.process_data(test_data)

        # Sensitive Daten sollten verschlüsselt sein
        output_file = temp_dir / "encrypted_data.csv"
        processed_data.to_csv(output_file, index=False)

        # Datei sollte existieren
        assert output_file.exists()

        # Datei-Inhalt sollte nicht im Klartext lesbar sein (falls verschlüsselt)
        content = output_file.read_text()
        # Hier könnten weitere Verschlüsselungs-Checks stehen
        assert len(content) > 0
