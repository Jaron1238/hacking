#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweiterte Visualisierungsmodule für WLAN-Analyse.
Implementiert 3D Network Visualization, Time-series Plots und Custom Report Generation.
"""

from __future__ import annotations

import logging
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import io

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from jinja2 import Template

logger = logging.getLogger(__name__)


class AdvancedVisualizer:
    """Erweiterte Visualisierungs-Klasse für WLAN-Daten."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialisiert den Advanced Visualizer."""
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Farbpaletten für verschiedene Visualisierungen
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
        
        # Template für Reports
        self.report_template = self._load_report_template()
    
    def create_3d_network_visualization(self, 
                                      devices: List[Dict[str, Any]],
                                      connections: List[Tuple[str, str]] = None) -> go.Figure:
        """
        Erstellt 3D-Netzwerk-Visualisierung.
        
        Args:
            devices: Liste von Gerätedaten
            connections: Liste von Verbindungen (MAC1, MAC2)
            
        Returns:
            Plotly 3D Figure
        """
        if not devices:
            return go.Figure()
        
        # Positionen berechnen (vereinfachte 3D-Positionierung)
        positions = self._calculate_3d_positions(devices)
        
        # Knoten-Daten vorbereiten
        node_x, node_y, node_z = [], [], []
        node_text, node_info = [], []
        node_colors, node_sizes = [], []
        
        for device in devices:
            mac = device.get('mac_address', '')
            if mac in positions:
                pos = positions[mac]
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_z.append(pos[2])
                
                # Node-Info
                device_type = device.get('device_type', 'Unknown')
                signal_strength = device.get('signal_strength', 0)
                channel = device.get('channel', 0)
                
                node_text.append(f"{mac[:8]}...")
                node_info.append(f"""
                MAC: {mac}<br>
                Type: {device_type}<br>
                Signal: {signal_strength} dBm<br>
                Channel: {channel}
                """)
                
                # Farbe basierend auf Gerätetyp
                node_colors.append(self._get_device_color(device_type))
                
                # Größe basierend auf Signalstärke
                node_sizes.append(max(10, min(50, abs(signal_strength))))
        
        # 3D Scatter Plot für Knoten
        fig = go.Figure(data=go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="top center",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_info
        ))
        
        # Verbindungen hinzufügen
        if connections:
            edge_x, edge_y, edge_z = [], [], []
            for mac1, mac2 in connections:
                if mac1 in positions and mac2 in positions:
                    pos1, pos2 = positions[mac1], positions[mac2]
                    edge_x.extend([pos1[0], pos2[0], None])
                    edge_y.extend([pos1[1], pos2[1], None])
                    edge_z.extend([pos1[2], pos2[2], None])
            
            if edge_x:
                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Layout anpassen
        fig.update_layout(
            title="3D Network Topology",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _calculate_3d_positions(self, devices: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float, float]]:
        """Berechnet 3D-Positionen für Geräte."""
        positions = {}
        
        # Vereinfachte Positionierung basierend auf Kanal und Signalstärke
        for i, device in enumerate(devices):
            mac = device.get('mac_address', '')
            channel = device.get('channel', 6)
            signal_strength = device.get('signal_strength', -70)
            
            # X: Kanal-basiert
            x = (channel - 1) * 2.0
            
            # Y: Signalstärke-basiert
            y = (signal_strength + 100) * 0.5
            
            # Z: Index-basiert (für Variation)
            z = (i % 3) * 2.0
            
            positions[mac] = (x, y, z)
        
        return positions
    
    def _get_device_color(self, device_type: str) -> str:
        """Gibt Farbe für Gerätetyp zurück."""
        color_map = {
            'AP': self.colors['primary'],
            'Client': self.colors['secondary'],
            'Router': self.colors['success'],
            'Bridge': self.colors['warning'],
            'Unknown': self.colors['info']
        }
        return color_map.get(device_type, self.colors['dark'])
    
    def create_time_series_plots(self, 
                               data: pd.DataFrame,
                               metrics: List[str] = None) -> Dict[str, Figure]:
        """
        Erstellt detaillierte Zeitverlaufs-Diagramme.
        
        Args:
            data: DataFrame mit Zeitdaten
            metrics: Liste der zu visualisierenden Metriken
            
        Returns:
            Dictionary mit Figure-Objekten
        """
        if metrics is None:
            metrics = ['rssi', 'throughput', 'packet_count', 'error_rate']
        
        figures = {}
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Zeitachse vorbereiten
            if 'timestamp' in data.columns:
                time_data = pd.to_datetime(data['timestamp'])
                ax.plot(time_data, data[metric], linewidth=2, alpha=0.8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax.plot(data[metric], linewidth=2, alpha=0.8)
            
            # Styling
            ax.set_title(f'{metric.replace("_", " ").title()} over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Statistiken hinzufügen
            mean_val = data[metric].mean()
            std_val = data[metric].std()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.2f}')
            ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.2f}')
            
            ax.legend()
            plt.tight_layout()
            
            figures[metric] = fig
        
        return figures
    
    def create_signal_quality_heatmap(self, 
                                    data: pd.DataFrame,
                                    x_col: str = 'channel',
                                    y_col: str = 'device_type',
                                    value_col: str = 'rssi') -> Figure:
        """
        Erstellt Signal Quality Heatmap.
        
        Args:
            data: DataFrame mit Signal-Daten
            x_col: X-Achse Spalte
            y_col: Y-Achse Spalte
            value_col: Werte-Spalte
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot Table erstellen
        pivot_data = data.pivot_table(
            values=value_col,
            index=y_col,
            columns=x_col,
            aggfunc='mean'
        )
        
        # Heatmap erstellen
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': f'{value_col.replace("_", " ").title()}'}
        )
        
        ax.set_title(f'Signal Quality Heatmap: {value_col.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_device_activity_heatmap(self, 
                                     activity_data: Dict[str, List[float]],
                                     time_labels: List[str] = None) -> Figure:
        """
        Erstellt Device Activity Heatmap.
        
        Args:
            activity_data: Dictionary mit MAC -> Aktivitätsdaten
            time_labels: Zeit-Labels für X-Achse
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Daten in Matrix umwandeln
        devices = list(activity_data.keys())
        max_length = max(len(data) for data in activity_data.values())
        
        # Matrix mit NaN auffüllen
        matrix = np.full((len(devices), max_length), np.nan)
        for i, device in enumerate(devices):
            data = activity_data[device]
            matrix[i, :len(data)] = data
        
        # Heatmap erstellen
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Labels setzen
        ax.set_yticks(range(len(devices)))
        ax.set_yticklabels([mac[:8] + '...' for mac in devices])
        
        if time_labels:
            ax.set_xticks(range(0, len(time_labels), max(1, len(time_labels) // 10)))
            ax.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), max(1, len(time_labels) // 10))])
        
        # Colorbar hinzufügen
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity Level', rotation=270, labelpad=20)
        
        ax.set_title('Device Activity Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Devices', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_network_topology_diagram(self, 
                                      devices: List[Dict[str, Any]],
                                      connections: List[Tuple[str, str]] = None) -> Figure:
        """
        Erstellt 2D-Netzwerk-Topologie-Diagramm.
        
        Args:
            devices: Liste von Gerätedaten
            connections: Liste von Verbindungen
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Positionen berechnen
        positions = self._calculate_2d_positions(devices)
        
        # Knoten zeichnen
        for device in devices:
            mac = device.get('mac_address', '')
            if mac in positions:
                x, y = positions[mac]
                device_type = device.get('device_type', 'Unknown')
                signal_strength = device.get('signal_strength', -70)
                
                # Knoten-Größe basierend auf Signalstärke
                size = max(100, min(1000, abs(signal_strength) * 10))
                
                # Knoten zeichnen
                ax.scatter(x, y, s=size, c=self._get_device_color(device_type), 
                          alpha=0.7, edgecolors='black', linewidth=2)
                
                # Label hinzufügen
                ax.annotate(mac[:8] + '...', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # Verbindungen zeichnen
        if connections:
            for mac1, mac2 in connections:
                if mac1 in positions and mac2 in positions:
                    x1, y1 = positions[mac1]
                    x2, y2 = positions[mac2]
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=1)
        
        ax.set_title('Network Topology', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Legende hinzufügen
        legend_elements = []
        for device_type in set(d.get('device_type', 'Unknown') for d in devices):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=self._get_device_color(device_type),
                                            markersize=10, label=device_type))
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig
    
    def _calculate_2d_positions(self, devices: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Berechnet 2D-Positionen für Geräte."""
        positions = {}
        
        for i, device in enumerate(devices):
            mac = device.get('mac_address', '')
            channel = device.get('channel', 6)
            signal_strength = device.get('signal_strength', -70)
            
            # X: Kanal-basiert
            x = (channel - 1) * 3.0
            
            # Y: Signalstärke-basiert
            y = (signal_strength + 100) * 0.3
            
            positions[mac] = (x, y)
        
        return positions
    
    def create_custom_report(self, 
                           analysis_data: Dict[str, Any],
                           output_path: str = "wlan_analysis_report.html") -> str:
        """
        Erstellt benutzerdefinierten HTML-Report.
        
        Args:
            analysis_data: Analyse-Daten
            output_path: Ausgabepfad für Report
            
        Returns:
            Pfad zum erstellten Report
        """
        # Report-Daten vorbereiten
        report_data = self._prepare_report_data(analysis_data)
        
        # Template rendern
        html_content = self.report_template.render(**report_data)
        
        # Report speichern
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report erstellt: {output_file}")
        return str(output_file)
    
    def _prepare_report_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereitet Report-Daten vor."""
        return {
            'title': 'WLAN Analysis Report',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'devices': analysis_data.get('devices', []),
            'metrics': analysis_data.get('metrics', {}),
            'insights': analysis_data.get('insights', {}),
            'charts': self._generate_chart_placeholders(analysis_data)
        }
    
    def _generate_chart_placeholders(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generiert Chart-Platzhalter für Report."""
        charts = []
        
        # Signal Quality Chart
        if 'signal_data' in analysis_data:
            charts.append(self._create_signal_chart(analysis_data['signal_data']))
        
        # Traffic Chart
        if 'traffic_data' in analysis_data:
            charts.append(self._create_traffic_chart(analysis_data['traffic_data']))
        
        return charts
    
    def _create_signal_chart(self, signal_data: Dict[str, Any]) -> str:
        """Erstellt Signal-Chart als Base64-String."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Vereinfachtes Signal-Chart
        ax.plot([1, 2, 3, 4, 5], [-50, -60, -55, -65, -70], 'b-', linewidth=2)
        ax.set_title('Signal Strength over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('RSSI (dBm)')
        ax.grid(True, alpha=0.3)
        
        # Chart zu Base64 konvertieren
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{chart_data}"
    
    def _create_traffic_chart(self, traffic_data: Dict[str, Any]) -> str:
        """Erstellt Traffic-Chart als Base64-String."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Vereinfachtes Traffic-Chart
        ax.bar(['Upload', 'Download'], [10, 15], color=['blue', 'green'], alpha=0.7)
        ax.set_title('Traffic Distribution')
        ax.set_ylabel('Throughput (Mbps)')
        
        # Chart zu Base64 konvertieren
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{chart_data}"
    
    def _load_report_template(self) -> Template:
        """Lädt HTML-Report-Template."""
        template_str = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 20px 0; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
        .device-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .device-card { padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .insights { background: #e7f3ff; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p class="timestamp">Generiert am: {{ timestamp }}</p>
        </div>
        
        <div class="section">
            <h2>Zusammenfassung</h2>
            <div class="insights">
                <p><strong>Anzahl Geräte:</strong> {{ devices|length }}</p>
                <p><strong>Analyse-Zeitraum:</strong> {{ insights.get('timeframe', 'N/A') }}</p>
                <p><strong>Netzwerk-Gesundheit:</strong> {{ insights.get('health_score', 'N/A') }}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Metriken</h2>
            <div class="metric">
                <strong>Durchschnittliche Signalstärke:</strong><br>
                {{ metrics.get('avg_rssi', 'N/A') }} dBm
            </div>
            <div class="metric">
                <strong>Gesamtdurchsatz:</strong><br>
                {{ metrics.get('total_throughput', 'N/A') }} Mbps
            </div>
            <div class="metric">
                <strong>Paket-Fehlerrate:</strong><br>
                {{ metrics.get('error_rate', 'N/A') }}%
            </div>
        </div>
        
        <div class="section">
            <h2>Geräte-Übersicht</h2>
            <div class="device-list">
                {% for device in devices %}
                <div class="device-card">
                    <h4>{{ device.get('mac_address', 'Unknown') }}</h4>
                    <p><strong>Typ:</strong> {{ device.get('device_type', 'Unknown') }}</p>
                    <p><strong>Signal:</strong> {{ device.get('signal_strength', 'N/A') }} dBm</p>
                    <p><strong>Kanal:</strong> {{ device.get('channel', 'N/A') }}</p>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2>Visualisierungen</h2>
            {% for chart in charts %}
            <div class="chart">
                <img src="{{ chart }}" alt="Chart">
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>Erkenntnisse</h2>
            <div class="insights">
                {% for insight in insights.get('recommendations', []) %}
                <p>• {{ insight }}</p>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_str)
    
    def save_plot(self, fig: Figure, filename: str, format: str = 'png') -> str:
        """Speichert Plot als Datei."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(fig, 'write_html'):
            # Plotly Figure
            fig.write_html(str(output_path.with_suffix('.html')))
        else:
            # Matplotlib Figure
            fig.savefig(str(output_path), format=format, dpi=300, bbox_inches='tight')
        
        logger.info(f"Plot gespeichert: {output_path}")
        return str(output_path)