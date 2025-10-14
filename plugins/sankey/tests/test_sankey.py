"""
Tests für das Sankey Plugin.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

from plugins.sankey.plugin import Plugin
from wlan_tool.storage.state import WifiAnalysisState, ClientState


class TestSankeyPlugin:
    """Test-Klasse für das Sankey Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
    @pytest.fixture
    def mock_state(self):
        """Mock State mit Test-Clients."""
        state = WifiAnalysisState()
        
        # Erstelle Test-Clients mit Roaming-Events
        for i in range(3):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            state.clients[mac] = client
        
        return state
    
    @pytest.fixture
    def mock_events_with_roaming(self):
        """Mock Events mit Roaming-Übergängen."""
        return [
            {"ts": 1000.0, "type": "data", "client": "aa:bb:cc:dd:ee:00", "bssid": "ap1"},
            {"ts": 1001.0, "type": "data", "client": "aa:bb:cc:dd:ee:00", "bssid": "ap2"},
            {"ts": 1002.0, "type": "data", "client": "aa:bb:cc:dd:ee:01", "bssid": "ap2"},
            {"ts": 1003.0, "type": "data", "client": "aa:bb:cc:dd:ee:01", "bssid": "ap3"},
        ]
    
    @pytest.fixture
    def mock_console(self):
        """Mock Console für Tests."""
        console = MagicMock()
        console.print = MagicMock()
        return console
    
    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Temporäres Ausgabeverzeichnis."""
        return tmp_path / "output"
    
    def test_plugin_metadata(self, plugin):
        """Test Plugin-Metadaten."""
        metadata = plugin.get_metadata()
        assert metadata.name == "Sankey Diagram"
        assert metadata.version == "1.0.0"
        assert "Roaming" in metadata.description
        assert "plotly" in metadata.dependencies
    
    def test_plugin_run_with_roaming_events(self, plugin, mock_state, mock_events_with_roaming, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit Roaming-Events."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('plotly.graph_objects.go') as mock_go:
            with patch('plotly.graph_objects.Figure') as mock_figure:
                mock_fig = MagicMock()
                mock_figure.return_value = mock_fig
                
                plugin.run(mock_state, mock_events_with_roaming, mock_console, temp_outdir)
                
                # Überprüfe, dass Plotly-Funktionen aufgerufen wurden
                mock_go.Sankey.assert_called_once()
                mock_fig.write_html.assert_called_once()
    
    def test_plugin_run_without_roaming_events(self, plugin, mock_state, mock_console, temp_outdir):
        """Test Plugin-Ausführung ohne Roaming-Events."""
        temp_outdir.mkdir(exist_ok=True)
        
        # Leere Events
        empty_events = []
        
        plugin.run(mock_state, empty_events, mock_console, temp_outdir)
        
        # Überprüfe Warnung für keine Roaming-Übergänge
        warning_calls = [call for call in mock_console.print.call_args_list 
                        if "Keine Roaming-Übergänge" in str(call)]
        assert len(warning_calls) > 0
    
    def test_plugin_run_without_plotly(self, plugin, mock_state, mock_events_with_roaming, mock_console, temp_outdir):
        """Test Plugin-Ausführung ohne Plotly."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('plugins.sankey.plugin.go', None):
            plugin.run(mock_state, mock_events_with_roaming, mock_console, temp_outdir)
            
            # Überprüfe Warnung für fehlendes Plotly
            warning_calls = [call for call in mock_console.print.call_args_list 
                            if "Plotly nicht verfügbar" in str(call)]
            assert len(warning_calls) > 0
    
    def test_roaming_transition_detection(self, plugin, mock_state):
        """Test Erkennung von Roaming-Übergängen."""
        events = [
            {"ts": 1000.0, "client": "aa:bb:cc:dd:ee:00", "bssid": "ap1"},
            {"ts": 1001.0, "client": "aa:bb:cc:dd:ee:00", "bssid": "ap2"},  # Übergang ap1 -> ap2
            {"ts": 1002.0, "client": "aa:bb:cc:dd:ee:00", "bssid": "ap2"},  # Gleicher AP
            {"ts": 1003.0, "client": "aa:bb:cc:dd:ee:00", "bssid": "ap3"},  # Übergang ap2 -> ap3
        ]
        
        # Simuliere die interne Logik
        transitions = defaultdict(int)
        
        for client in mock_state.clients.values():
            client_events = [ev for ev in events if ev.get("client") == client.mac and ev.get("bssid")]
            client_events.sort(key=lambda x: x['ts'])
            
            if len(client_events) < 2:
                continue

            last_bssid = None
            for ev in client_events:
                current_bssid = ev.get('bssid')
                if current_bssid and last_bssid and current_bssid != last_bssid:
                    transitions[(last_bssid, current_bssid)] += 1
                last_bssid = current_bssid
        
        # Überprüfe erkannte Übergänge
        assert ("ap1", "ap2") in transitions
        assert ("ap2", "ap3") in transitions
        assert transitions[("ap1", "ap2")] == 1
        assert transitions[("ap2", "ap3")] == 1
    
    def test_plugin_run_with_sufficient_data(self, plugin, mock_state, mock_events_with_roaming, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit ausreichenden Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('plotly.graph_objects.go') as mock_go:
            with patch('plotly.graph_objects.Figure') as mock_figure:
                mock_fig = MagicMock()
                mock_figure.return_value = mock_fig
                
                plugin.run(mock_state, mock_events_with_roaming, mock_console, temp_outdir)
                
                # Überprüfe, dass Console-Ausgaben gemacht wurden
                assert mock_console.print.called
                
                # Überprüfe, dass HTML-Datei gespeichert wurde
                mock_fig.write_html.assert_called_once()
                output_file = mock_fig.write_html.call_args[0][0]
                assert "roaming_sankey.html" in output_file