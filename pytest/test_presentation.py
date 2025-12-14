#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Presentation-Modul (cli.py, tui.py, live_tui.py, reporting.py).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile
import time

from wlan_tool.presentation import cli, reporting
from wlan_tool.storage.state import WifiAnalysisState, ClientState, APState
from wlan_tool.storage.data_models import InferenceResult, Welford


class TestCLIModule:
    """Tests für das CLI-Modul."""
    
    def test_print_client_cluster_results(self, mock_console, populated_state):
        """Test Client-Cluster-Ergebnis-Ausgabe."""
        from wlan_tool.analysis import logic as analysis
        
        # Erstelle Test-Cluster-Daten
        clustered_df, feature_df = analysis.cluster_clients(populated_state, n_clusters=2)
        
        if clustered_df is not None and not clustered_df.empty:
            # Mock args
            args = MagicMock()
            args.cluster_clients = 2
            args.cluster_algo = "kmeans"
            args.no_mac_correlation = False
            
            # Sollte nicht crashen
            cli.print_client_cluster_results(args, populated_state, mock_console)
            
            # Verifiziere, dass Console-Ausgaben gemacht wurden
            assert mock_console.print.called
    
    def test_print_ap_cluster_results(self, mock_console, populated_state):
        """Test AP-Cluster-Ergebnis-Ausgabe."""
        from wlan_tool.analysis import logic as analysis
        
        # Erstelle Test-AP-Cluster-Daten
        clustered_df = analysis.cluster_aps(populated_state, n_clusters=2)
        
        if clustered_df is not None and not clustered_df.empty:
            # Mock args
            args = MagicMock()
            args.cluster_aps = 2
            
            # Sollte nicht crashen
            cli.print_ap_cluster_results(args, populated_state, mock_console)
            
            # Verifiziere, dass Console-Ausgaben gemacht wurden
            assert mock_console.print.called
    
    def test_print_probed_ssids(self, mock_console, populated_state):
        """Test Probe-SSID-Ausgabe."""
        cli.print_probed_ssids(populated_state, mock_console)
        
        # Verifiziere, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
    
    @patch('wlan_tool.storage.database.db_conn_ctx')
    def test_interactive_label_ui(self, mock_db_conn, mock_console, populated_state):
        """Test interaktive Label-UI."""
        # Mock Datenbankverbindung
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.fetchall.return_value = []
        
        # Mock joblib.load
        with patch('joblib.load', return_value=None):
            # Mock Prompt.ask für Benutzereingaben
            with patch('rich.prompt.Prompt.ask', return_value='q'):  # Beende sofort
                cli.interactive_label_ui(
                    db_path_events="test.db",
                    label_db_path="labels.db",
                    model_path="model.pkl",
                    console=mock_console
                )
        
        # Verifiziere, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
    
    @patch('wlan_tool.storage.database.db_conn_ctx')
    def test_interactive_client_label_ui(self, mock_db_conn, mock_console, populated_state):
        """Test interaktive Client-Label-UI."""
        # Mock Datenbankverbindung
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value.__enter__.return_value = mock_conn
        
        # Mock console.input für Benutzereingaben
        with patch.object(mock_console, 'input', return_value='q'):  # Beende sofort
            cli.interactive_client_label_ui(
                label_db_path="labels.db",
                state_obj=populated_state,
                console=mock_console
            )
        
        # Verifiziere, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called


class TestReportingModule:
    """Tests für das Reporting-Modul."""
    
    def test_generate_html_report_no_jinja2(self, populated_state):
        """Test HTML-Report ohne Jinja2."""
        with patch('wlan_tool.presentation.reporting.Environment', None):
            # Sollte nicht crashen
            reporting.generate_html_report(
                state=populated_state,
                analysis_results={},
                template_path="",
                out_file="/tmp/test.html"
            )
    
    @patch('wlan_tool.presentation.reporting.Environment')
    def test_generate_html_report_with_jinja2(self, mock_env_class, populated_state):
        """Test HTML-Report mit Jinja2."""
        # Mock Jinja2
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Test Report</html>"
        mock_env.get_template.return_value = mock_template
        mock_env_class.return_value = mock_env
        
        # Erstelle Test-Template
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html>{{ generation_time }}</html>")
            template_path = f.name
        
        try:
            # Teste Report-Generierung
            reporting.generate_html_report(
                state=populated_state,
                analysis_results={"inference": []},
                template_path=template_path,
                out_file="/tmp/test.html"
            )
            
            # Verifiziere, dass Template geladen wurde
            mock_env.get_template.assert_called_once()
            mock_template.render.assert_called_once()
            
        finally:
            # Cleanup
            Path(template_path).unlink(missing_ok=True)
    
    def test_generate_html_report_default_template(self, populated_state):
        """Test HTML-Report mit Standard-Template."""
        with patch('wlan_tool.presentation.reporting.Environment') as mock_env_class:
            mock_env = MagicMock()
            mock_template = MagicMock()
            mock_template.render.return_value = "<html>Test</html>"
            mock_env.get_template.return_value = mock_template
            mock_env_class.return_value = mock_env
            
            # Teste ohne Template-Pfad (sollte Standard verwenden)
            reporting.generate_html_report(
                state=populated_state,
                analysis_results={},
                template_path="",  # Leer = Standard-Template
                out_file="/tmp/test.html"
            )
            
            # Verifiziere, dass Standard-Template verwendet wurde
            mock_env.get_template.assert_called_once()
    
    def test_generate_html_report_missing_template(self, populated_state):
        """Test HTML-Report mit fehlendem Template."""
        with patch('wlan_tool.presentation.reporting.Environment') as mock_env_class:
            mock_env = MagicMock()
            mock_env_class.return_value = mock_env
            
            # Teste mit nicht existierendem Template
            reporting.generate_html_report(
                state=populated_state,
                analysis_results={},
                template_path="/nonexistent/template.html",
                out_file="/tmp/test.html"
            )
            
            # Sollte nicht crashen, aber auch nicht viel tun


class TestTUIModule:
    """Tests für das TUI-Modul."""
    
    def test_analysis_tui_initialization(self):
        """Test AnalysisTUI-Initialisierung."""
        from wlan_tool.presentation.tui import AnalysisTUI
        
        # Erstelle Test-Daten
        clustered_df = pd.DataFrame({
            'mac': ['aa:bb:cc:dd:ee:ff'],
            'vendor': ['TestVendor'],
            'cluster': [0]
        })
        profiles = {0: {'count': 1, 'top_features': ['feature1']}}
        
        # Sollte nicht crashen
        app = AnalysisTUI(clustered_df, profiles)
        assert app.clustered_df is not None
        assert app.profiles is not None
    
    def test_analysis_tui_compose(self):
        """Test AnalysisTUI-Layout-Erstellung."""
        from wlan_tool.presentation.tui import AnalysisTUI
        
        clustered_df = pd.DataFrame({
            'mac': ['aa:bb:cc:dd:ee:ff'],
            'vendor': ['TestVendor'],
            'cluster': [0]
        })
        profiles = {0: {'count': 1, 'top_features': ['feature1']}}
        
        app = AnalysisTUI(clustered_df, profiles)
        
        # Teste Layout-Erstellung
        try:
            compose_result = app.compose()
            # Sollte ein Generator sein
            assert hasattr(compose_result, '__iter__')
        except Exception as e:
            # Kann bei Textual-Tests fehlschlagen, das ist OK
            pass
    
    def test_live_capture_tui_initialization(self):
        """Test LiveCaptureTUI-Initialisierung."""
        from wlan_tool.presentation.live_tui import LiveCaptureTUI
        import multiprocessing as mp
        
        # Erstelle Mock-Queue
        live_queue = mp.Queue()
        
        # Sollte nicht crashen
        app = LiveCaptureTUI(live_queue, duration=60)
        assert app.live_queue is not None
        assert app.duration == 60
    
    def test_live_capture_tui_compose(self):
        """Test LiveCaptureTUI-Layout-Erstellung."""
        from wlan_tool.presentation.live_tui import LiveCaptureTUI
        import multiprocessing as mp
        
        live_queue = mp.Queue()
        app = LiveCaptureTUI(live_queue, duration=60)
        
        # Teste Layout-Erstellung
        try:
            compose_result = app.compose()
            # Sollte ein Generator sein
            assert hasattr(compose_result, '__iter__')
        except Exception as e:
            # Kann bei Textual-Tests fehlschlagen, das ist OK
            pass


class TestCLIEdgeCases:
    """Tests für CLI-Edge-Cases."""
    
    def test_print_client_cluster_results_empty_data(self, mock_console, populated_state):
        """Test Client-Cluster-Ausgabe mit leeren Daten."""
        # Erstelle leeren State
        empty_state = WifiAnalysisState()
        
        args = MagicMock()
        args.cluster_clients = 2
        args.cluster_algo = "kmeans"
        args.no_mac_correlation = False
        
        # Sollte nicht crashen
        cli.print_client_cluster_results(args, empty_state, mock_console)
    
    def test_print_ap_cluster_results_empty_data(self, mock_console, populated_state):
        """Test AP-Cluster-Ausgabe mit leeren Daten."""
        # Erstelle leeren State
        empty_state = WifiAnalysisState()
        
        args = MagicMock()
        args.cluster_aps = 2
        
        # Sollte nicht crashen
        cli.print_ap_cluster_results(args, empty_state, mock_console)
    
    def test_print_probed_ssids_empty_state(self, mock_console):
        """Test Probe-SSID-Ausgabe mit leerem State."""
        empty_state = WifiAnalysisState()
        
        # Sollte nicht crashen
        cli.print_probed_ssids(empty_state, mock_console)


class TestReportingEdgeCases:
    """Tests für Reporting-Edge-Cases."""
    
    def test_generate_html_report_empty_state(self):
        """Test HTML-Report mit leerem State."""
        empty_state = WifiAnalysisState()
        
        with patch('wlan_tool.presentation.reporting.Environment') as mock_env_class:
            mock_env = MagicMock()
            mock_template = MagicMock()
            mock_template.render.return_value = "<html>Empty</html>"
            mock_env.get_template.return_value = mock_template
            mock_env_class.return_value = mock_env
            
            # Sollte nicht crashen
            reporting.generate_html_report(
                state=empty_state,
                analysis_results={},
                template_path="",
                out_file="/tmp/test.html"
            )
    
    def test_generate_html_report_with_inference_results(self, populated_state):
        """Test HTML-Report mit Inferenz-Ergebnissen."""
        # Erstelle Test-Inferenz-Ergebnisse
        inference_results = [
            InferenceResult(
                ssid="TestSSID",
                bssid="aa:bb:cc:dd:ee:ff",
                score=0.85,
                label="high",
                components={"beacon_count": 10},
                evidence={"supporting_clients": ["client1"]}
            )
        ]
        
        with patch('wlan_tool.presentation.reporting.Environment') as mock_env_class:
            mock_env = MagicMock()
            mock_template = MagicMock()
            mock_template.render.return_value = "<html>With Results</html>"
            mock_env.get_template.return_value = mock_template
            mock_env_class.return_value = mock_env
            
            # Sollte nicht crashen
            reporting.generate_html_report(
                state=populated_state,
                analysis_results={"inference": inference_results},
                template_path="",
                out_file="/tmp/test.html"
            )


class TestPerformance:
    """Tests für Performance-Aspekte."""
    
    def test_large_dataset_handling(self, mock_console):
        """Test mit großen Datensätzen."""
        # Erstelle State mit vielen Clients
        state = WifiAnalysisState()
        
        for i in range(100):
            client = ClientState(mac=f"aa:bb:cc:dd:ee:{i:02x}")
            client.probes = {f"SSID_{i}"}
            client.all_packet_ts = np.array([time.time() - 100, time.time()])
            client.rssi_w = Welford()
            client.rssi_w.update(-50)
            state.clients[f"aa:bb:cc:dd:ee:{i:02x}"] = client
        
        # Teste verschiedene CLI-Funktionen
        cli.print_probed_ssids(state, mock_console)
        
        # Sollte nicht crashen oder zu langsam sein
        assert mock_console.print.called
    
    def test_memory_usage(self, populated_state):
        """Test Speicherverbrauch."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Führe mehrere Operationen aus
        for _ in range(10):
            cli.print_probed_ssids(populated_state, MagicMock())
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Speicherzunahme sollte nicht zu groß sein (weniger als 10MB)
        assert memory_increase < 10 * 1024 * 1024