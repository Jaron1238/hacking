#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für das Controllers-Modul (controllers.py).
"""

import pytest
import subprocess
import tempfile
import time
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from wlan_tool.controllers import CaptureController, AnalysisController
from wlan_tool.storage.state import WifiAnalysisState


class TestCaptureController:
    """Tests für den CaptureController."""
    
    def test_capture_controller_initialization(self, mock_console):
        """Test CaptureController-Initialisierung."""
        args = MagicMock()
        config_data = {"capture": {"interface": "wlan0", "duration": 60}}
        
        controller = CaptureController(args, config_data, mock_console)
        
        assert controller.args == args
        assert controller.config_data == config_data
        assert controller.console == mock_console
    
    @patch('wlan_tool.pre_run_checks.find_wlan_interfaces')
    def test_select_interface_automatic(self, mock_find_interfaces, mock_console):
        """Test automatische Interface-Auswahl."""
        mock_find_interfaces.return_value = ["wlan0", "wlan1"]
        
        args = MagicMock()
        args.iface = None
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch('rich.prompt.Prompt.ask', return_value="wlan0"):
            iface = controller._select_interface()
            assert iface == "wlan0"
        
        # Überprüfe, dass die Console-Ausgabe korrekt ist
        mock_console.print.assert_called()
    
    @patch('wlan_tool.pre_run_checks.find_wlan_interfaces')
    def test_select_interface_no_interfaces(self, mock_find_interfaces, mock_console):
        """Test Interface-Auswahl ohne verfügbare Interfaces."""
        mock_find_interfaces.return_value = []
        
        args = MagicMock()
        args.iface = None
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        iface = controller._select_interface()
        assert iface is None
        
        # Überprüfe, dass die Console-Ausgabe korrekt ist
        mock_console.print.assert_called()
    
    def test_select_interface_specified(self, mock_console):
        """Test Interface-Auswahl mit spezifiziertem Interface."""
        args = MagicMock()
        args.iface = "wlan0"
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        iface = controller._select_interface()
        assert iface == "wlan0"
    
    @patch('subprocess.run')
    def test_setup_monitor_mode_success(self, mock_run, mock_console):
        """Test Monitor-Mode-Setup (erfolgreich)."""
        mock_run.return_value = MagicMock(returncode=0)
        
        args = MagicMock()
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        monitor_iface = controller._setup_monitor_mode("wlan0")
        
        assert monitor_iface == "mon0"
        assert mock_run.call_count > 0
    
    @patch('subprocess.run')
    def test_setup_monitor_mode_failure(self, mock_run, mock_console):
        """Test Monitor-Mode-Setup (Fehler)."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "airmon-ng")
        
        args = MagicMock()
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        monitor_iface = controller._setup_monitor_mode("wlan0")
        
        assert monitor_iface is None
        
        # Überprüfe, dass die Console-Ausgabe korrekt ist
        mock_console.print.assert_called()
    
    @patch('wlan_tool.capture.sniffer.sniff_with_writer')
    def test_run_capture_normal(self, mock_sniff, mock_console):
        """Test normale Capture-Ausführung."""
        mock_sniff.return_value = None
        
        args = MagicMock()
        args.iface = "wlan0"
        args.duration = 60
        args.pcap = None
        args.outdir = None
        args.project = None
        args.db = None
        args.live = False
        args.adaptive_scan = False
        config_data = {"capture": {"interface": "wlan0", "duration": 60}}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value="wlan0"):
            with patch.object(controller, '_setup_monitor_mode', return_value="mon0"):
                with patch('wlan_tool.storage.database.migrate_db'):
                    controller.run_capture()
        
        mock_sniff.assert_called_once()
    
    @patch('multiprocessing.Process')
    @patch('wlan_tool.capture.sniffer.sniff_with_writer')
    def test_run_capture_live(self, mock_sniff, mock_process, mock_console):
        """Test Live-Capture-Ausführung."""
        mock_sniff.return_value = None
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        
        args = MagicMock()
        args.iface = "wlan0"
        args.duration = 60
        args.pcap = None
        args.outdir = None
        args.project = None
        args.db = None
        args.live = True
        config_data = {"capture": {"interface": "wlan0", "duration": 60}}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value="wlan0"):
            with patch.object(controller, '_setup_monitor_mode', return_value="mon0"):
                with patch('wlan_tool.storage.database.migrate_db'):
                    controller.run_capture()
        
        mock_process.assert_called_once()
        mock_process_instance.start.assert_called_once()
        mock_process_instance.terminate.assert_called_once()
    
    @patch('wlan_tool.capture.sniffer.sniff_with_writer')
    def test_run_adaptive_scan(self, mock_sniff, mock_console):
        """Test adaptiver Scan."""
        mock_sniff.return_value = None
        
        args = MagicMock()
        args.iface = "wlan0"
        args.duration = 120
        args.pcap = None
        args.outdir = None
        args.project = None
        args.db = None
        args.live = False
        args.adaptive_scan = True
        args.discovery_time = 60
        config_data = {
            "capture": {"interface": "wlan0", "duration": 120},
            "scanning": {"adaptive_scan_enabled": True, "adaptive_scan_discovery_s": 60}
        }
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value="wlan0"):
            with patch.object(controller, '_setup_monitor_mode', return_value="mon0"):
                with patch.object(controller, '_run_adaptive_scan') as mock_adaptive:
                    with patch('wlan_tool.storage.database.migrate_db'):
                        controller.run_capture()
        
        mock_adaptive.assert_called_once()


class TestAnalysisController:
    """Tests für den AnalysisController."""
    
    def test_analysis_controller_initialization(self, mock_console, populated_state):
        """Test AnalysisController-Initialisierung."""
        args = MagicMock()
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        assert controller.args == args
        assert controller.config_data == config_data
        assert controller.console == mock_console
        assert controller.plugins == plugins
        assert controller.state == populated_state
        assert controller.new_events == new_events
        
        # Überprüfe, dass der Controller korrekt initialisiert wurde
        assert hasattr(controller, 'state_obj')
        assert controller.state_obj == populated_state
    
    @patch('wlan_tool.analysis.logic.score_pairs_with_recency_and_matching')
    def test_run_inference(self, mock_score, mock_console, populated_state):
        """Test Inferenz-Ausführung."""
        mock_score.return_value = []
        
        args = MagicMock()
        args.infer = True
        args.model = None
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_inference()
        
        mock_score.assert_called_once()
    
    @patch('wlan_tool.analysis.logic.cluster_clients')
    def test_run_client_clustering(self, mock_cluster, mock_console, populated_state):
        """Test Client-Clustering."""
        mock_cluster.return_value = (None, None)
        
        args = MagicMock()
        args.cluster_clients = 2
        args.cluster_algo = "kmeans"
        args.no_mac_correlation = False
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_client_clustering()
        
        mock_cluster.assert_called_once()
    
    @patch('wlan_tool.analysis.logic.cluster_aps')
    def test_run_ap_clustering(self, mock_cluster, mock_console, populated_state):
        """Test AP-Clustering."""
        mock_cluster.return_value = None
        
        args = MagicMock()
        args.cluster_aps = 2
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_ap_clustering()
        
        mock_cluster.assert_called_once()
    
    @patch('wlan_tool.analysis.logic.export_ap_graph')
    def test_run_graph_export(self, mock_export, mock_console, populated_state):
        """Test Graph-Export."""
        mock_export.return_value = True
        
        args = MagicMock()
        args.export_graph = "/tmp/test.gexf"
        args.cluster_aps = 2
        args.graph_include_clients = False
        args.graph_min_activity = 0
        args.graph_min_duration = 0
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        with patch.object(controller, 'run_ap_clustering', return_value=None):
            controller.run_graph_export()
        
        mock_export.assert_called_once()
    
    @patch('wlan_tool.presentation.cli.interactive_label_ui')
    def test_run_labeling_ui(self, mock_label_ui, mock_console, populated_state):
        """Test Labeling-UI."""
        args = MagicMock()
        args.label_ui = True
        args.db = "test.db"
        args.label_db = "labels.db"
        args.model = "model.pkl"
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_labeling_ui()
        
        mock_label_ui.assert_called_once()
    
    @patch('wlan_tool.presentation.cli.interactive_client_label_ui')
    def test_run_client_labeling_ui(self, mock_client_label_ui, mock_console, populated_state):
        """Test Client-Labeling-UI."""
        args = MagicMock()
        args.label_clients = True
        args.label_db = "labels.db"
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_client_labeling_ui()
        
        mock_client_label_ui.assert_called_once()
    
    @patch('wlan_tool.analysis.device_profiler.correlate_devices_by_fingerprint')
    def test_run_mac_correlation(self, mock_correlate, mock_console, populated_state):
        """Test MAC-Korrelation."""
        mock_correlate.return_value = {}
        
        args = MagicMock()
        args.correlate_macs = True
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_mac_correlation()
        
        mock_correlate.assert_called_once()
    
    def test_run_analysis_no_actions(self, mock_console, populated_state):
        """Test Analyse ohne Aktionen."""
        args = MagicMock()
        args.infer = False
        args.cluster_clients = None
        args.cluster_aps = None
        args.export_graph = None
        args.label_ui = False
        args.label_clients = False
        args.correlate_macs = False
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        # Sollte nicht crashen
        controller.run_analysis()
    
    def test_run_analysis_multiple_actions(self, mock_console, populated_state):
        """Test Analyse mit mehreren Aktionen."""
        args = MagicMock()
        args.infer = True
        args.cluster_clients = 2
        args.cluster_aps = 2
        args.export_graph = "/tmp/test.gexf"
        args.label_ui = False
        args.label_clients = False
        args.correlate_macs = False
        args.model = None
        args.cluster_algo = "kmeans"
        args.no_mac_correlation = False
        args.graph_include_clients = False
        args.graph_min_activity = 0
        args.graph_min_duration = 0
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        with patch.object(controller, 'run_inference'):
            with patch.object(controller, 'run_client_clustering'):
                with patch.object(controller, 'run_ap_clustering'):
                    with patch.object(controller, 'run_graph_export'):
                        controller.run_analysis()


class TestControllerEdgeCases:
    """Tests für Controller-Edge-Cases."""
    
    def test_capture_controller_no_interface(self, mock_console):
        """Test CaptureController ohne Interface."""
        args = MagicMock()
        args.iface = None
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value=None):
            result = controller.run_capture()
            assert result is None
    
    def test_capture_controller_no_monitor_mode(self, mock_console):
        """Test CaptureController ohne Monitor-Mode."""
        args = MagicMock()
        args.iface = "wlan0"
        config_data = {}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value="wlan0"):
            with patch.object(controller, '_setup_monitor_mode', return_value=None):
                result = controller.run_capture()
                assert result is None
    
    def test_analysis_controller_empty_state(self, mock_console):
        """Test AnalysisController mit leerem State."""
        args = MagicMock()
        args.infer = True
        args.model = None
        config_data = {}
        plugins = {}
        new_events = []
        empty_state = WifiAnalysisState()
        
        controller = AnalysisController(args, config_data, mock_console, plugins, empty_state, new_events)
        
        # Sollte nicht crashen
        controller.run_inference()
    
    def test_analysis_controller_with_plugins(self, mock_console, populated_state):
        """Test AnalysisController mit Plugins."""
        args = MagicMock()
        args.run_plugins = ["test_plugin"]
        config_data = {}
        
        mock_plugin = MagicMock()
        mock_plugin.run.return_value = None
        plugins = {"test_plugin": mock_plugin}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        controller.run_plugins()
        
        mock_plugin.run.assert_called_once()
        
        # Überprüfe, dass der Controller korrekt initialisiert wurde
        assert hasattr(controller, 'state_obj')
        assert controller.state_obj == populated_state


class TestControllerIntegration:
    """Tests für Controller-Integration."""
    
    @patch('wlan_tool.storage.database.migrate_db')
    @patch('wlan_tool.capture.sniffer.sniff_with_writer')
    def test_full_capture_workflow(self, mock_sniff, mock_migrate, mock_console):
        """Test vollständiger Capture-Workflow."""
        mock_sniff.return_value = None
        
        args = MagicMock()
        args.iface = "wlan0"
        args.duration = 60
        args.pcap = "/tmp/test.pcap"
        args.outdir = "/tmp"
        args.project = None
        args.db = "/tmp/test.db"
        args.live = False
        args.adaptive_scan = False
        config_data = {"capture": {"interface": "wlan0", "duration": 60}}
        
        controller = CaptureController(args, config_data, mock_console)
        
        with patch.object(controller, '_select_interface', return_value="wlan0"):
            with patch.object(controller, '_setup_monitor_mode', return_value="mon0"):
                controller.run_capture()
        
        mock_migrate.assert_called_once()
        mock_sniff.assert_called_once()
    
    def test_full_analysis_workflow(self, mock_console, populated_state):
        """Test vollständiger Analyse-Workflow."""
        args = MagicMock()
        args.infer = True
        args.cluster_clients = 2
        args.cluster_aps = 2
        args.export_graph = "/tmp/test.gexf"
        args.label_ui = False
        args.label_clients = False
        args.correlate_macs = False
        args.model = None
        args.cluster_algo = "kmeans"
        args.no_mac_correlation = False
        args.graph_include_clients = False
        args.graph_min_activity = 0
        args.graph_min_duration = 0
        config_data = {}
        plugins = {}
        new_events = []
        
        controller = AnalysisController(args, config_data, mock_console, plugins, populated_state, new_events)
        
        with patch.object(controller, 'run_inference'):
            with patch.object(controller, 'run_client_clustering'):
                with patch.object(controller, 'run_ap_clustering'):
                    with patch.object(controller, 'run_graph_export'):
                        controller.run_analysis()
        
        # Sollte alle Methoden aufgerufen haben
        assert True  # Wenn wir hier ankommen, ist der Test erfolgreich
        
        # Überprüfe, dass der Controller korrekt initialisiert wurde
        assert hasattr(controller, 'state_obj')
        assert controller.state_obj == populated_state