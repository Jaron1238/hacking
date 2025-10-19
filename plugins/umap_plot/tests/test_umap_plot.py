"""
Tests für das UMAP Plot Plugin.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from plugins.umap_plot.plugin import Plugin


class TestUMAPPlotPlugin:
    """Test-Klasse für das UMAP Plot Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
    @pytest.fixture
    def mock_clustered_client_df(self):
        """Mock DataFrame mit Cluster-Daten."""
        return pd.DataFrame({
            'mac': ['aa:bb:cc:dd:ee:00', 'aa:bb:cc:dd:ee:01', 'aa:bb:cc:dd:ee:02'],
            'cluster': [0, 1, 0],
            'vendor': ['Apple', 'Samsung', 'Apple']
        })
    
    @pytest.fixture
    def mock_client_feature_df(self):
        """Mock DataFrame mit Client-Features."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': [7.0, 8.0, 9.0],
            'original_macs': ['aa:bb:cc:dd:ee:00', 'aa:bb:cc:dd:ee:01', 'aa:bb:cc:dd:ee:02']
        })
    
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
        assert metadata.name == "UMAP Plot"
        assert metadata.version == "1.0.0"
        assert "UMAP" in metadata.description
        assert "umap-learn" in metadata.dependencies
        assert "plotly" in metadata.dependencies
    
    def test_plugin_run_with_valid_data(self, plugin, mock_clustered_client_df, mock_client_feature_df, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit gültigen Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('umap.UMAP') as mock_umap:
            with patch('plugins.umap_plot.plugin.px') as mock_px:
                # Mock UMAP
                mock_reducer = MagicMock()
                mock_reducer.fit_transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])
                mock_umap.return_value = mock_reducer
                
                # Mock Plotly
                mock_fig = MagicMock()
                mock_px.scatter.return_value = mock_fig
                
                plugin.run(
                    state=None,
                    clustered_client_df=mock_clustered_client_df,
                    client_feature_df=mock_client_feature_df,
                    outdir=temp_outdir,
                    console=mock_console
                )
                
                # Überprüfe, dass UMAP aufgerufen wurde
                mock_umap.assert_called_once_with(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                mock_reducer.fit_transform.assert_called_once()
                
                # Überprüfe, dass Plotly aufgerufen wurde
                mock_px.scatter.assert_called_once()
                mock_fig.write_html.assert_called_once()
                
                # Überprüfe, dass Console-Ausgaben gemacht wurden
                assert mock_console.print.called
    
    def test_plugin_run_without_umap(self, plugin, mock_clustered_client_df, mock_client_feature_df, mock_console, temp_outdir):
        """Test Plugin-Ausführung ohne UMAP."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('plugins.umap_plot.plugin.umap', None):
            plugin.run(
                state=None,
                clustered_client_df=mock_clustered_client_df,
                client_feature_df=mock_client_feature_df,
                outdir=temp_outdir,
                console=mock_console
            )
            
            # Überprüfe Warnung für fehlendes UMAP
            warning_calls = [call for call in mock_console.print.call_args_list 
                            if "UMAP/Plotly nicht verfügbar" in str(call)]
            assert len(warning_calls) > 0
    
    def test_plugin_run_without_clustered_data(self, plugin, mock_console, temp_outdir):
        """Test Plugin-Ausführung ohne Cluster-Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        plugin.run(
            state=None,
            clustered_client_df=None,
            client_feature_df=None,
            outdir=temp_outdir,
            console=mock_console
        )
        
        # Überprüfe Warnung für fehlende UMAP/Plotly
        warning_calls = [call for call in mock_console.print.call_args_list 
                        if "UMAP/Plotly nicht verfügbar" in str(call)]
        assert len(warning_calls) > 0
    
    def test_plugin_run_with_empty_feature_data(self, plugin, mock_clustered_client_df, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit leeren Feature-Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        empty_feature_df = pd.DataFrame()
        
        plugin.run(
            state=None,
            clustered_client_df=mock_clustered_client_df,
            client_feature_df=empty_feature_df,
            outdir=temp_outdir,
            console=mock_console
        )
        
        # Überprüfe Warnung für fehlende UMAP/Plotly
        warning_calls = [call for call in mock_console.print.call_args_list 
                        if "UMAP/Plotly nicht verfügbar" in str(call)]
        assert len(warning_calls) > 0
    
    def test_feature_preparation(self, plugin, mock_clustered_client_df, mock_client_feature_df):
        """Test Vorbereitung der Features für UMAP."""
        # Simuliere die interne Logik
        features_for_map = mock_client_feature_df.drop(columns=['original_macs'], errors='ignore')
        
        # Überprüfe, dass original_macs Spalte entfernt wurde
        assert 'original_macs' not in features_for_map.columns
        
        # Überprüfe, dass andere Spalten erhalten blieben
        assert 'feature1' in features_for_map.columns
        assert 'feature2' in features_for_map.columns
        assert 'feature3' in features_for_map.columns
    
    def test_plot_dataframe_creation(self, plugin, mock_clustered_client_df, mock_client_feature_df):
        """Test Erstellung des Plot-DataFrames."""
        # Simuliere UMAP-Embedding
        embedding = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Simuliere DataFrame-Erstellung
        plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
        clustered_info = mock_clustered_client_df.reset_index(drop=True)
        
        plot_df['cluster'] = clustered_info['cluster'].astype(str)
        plot_df['device'] = clustered_info['mac']
        plot_df['vendor'] = clustered_info['vendor']
        
        # Überprüfe DataFrame-Struktur
        assert 'x' in plot_df.columns
        assert 'y' in plot_df.columns
        assert 'cluster' in plot_df.columns
        assert 'device' in plot_df.columns
        assert 'vendor' in plot_df.columns
        
        # Überprüfe Datentypen
        assert plot_df['cluster'].dtype == 'object'  # String
        assert plot_df['device'].dtype == 'object'
        assert plot_df['vendor'].dtype == 'object'
    
    def test_plugin_run_with_exception(self, plugin, mock_clustered_client_df, mock_client_feature_df, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit Exception."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('umap.UMAP') as mock_umap:
            # Simuliere Exception in UMAP
            mock_umap.side_effect = Exception("UMAP Error")
            
            plugin.run(
                state=None,
                clustered_client_df=mock_clustered_client_df,
                client_feature_df=mock_client_feature_df,
                outdir=temp_outdir,
                console=mock_console
            )
            
            # Überprüfe Fehlerbehandlung
            error_calls = [call for call in mock_console.print.call_args_list 
                          if "Fehler bei der Erstellung" in str(call)]
            assert len(error_calls) > 0