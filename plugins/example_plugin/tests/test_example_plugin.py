"""
Tests für das Example_Plugin Plugin.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from plugins.example_plugin.plugin import Plugin


class TestExample_PluginPlugin:
    """Test-Klasse für das Example_Plugin Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
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
        assert metadata.name == "Example_Plugin"
        assert metadata.version == "1.0.0"
    
    def test_plugin_run(self, plugin, mock_console, temp_outdir):
        """Test Plugin-Ausführung."""
        temp_outdir.mkdir(exist_ok=True)
        
        # Mock state und events
        mock_state = {}
        mock_events = []
        
        plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
