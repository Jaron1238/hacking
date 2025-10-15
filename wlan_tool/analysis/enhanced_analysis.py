#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweiterte Analyse-Integration für alle neuen Features.
Kombiniert DPI, Metriken und Visualisierung.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

from .deep_packet_inspection import DeepPacketInspector
from .advanced_metrics import AdvancedMetricsCalculator
from ..visualization.advanced_visualizer import AdvancedVisualizer

logger = logging.getLogger(__name__)


class EnhancedAnalysisEngine:
    """Erweiterte Analyse-Engine für umfassende WLAN-Analyse."""
    
    def __init__(self, output_dir: str = "analysis_output"):
        """Initialisiert die erweiterte Analyse-Engine."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyse-Module
        self.dpi = DeepPacketInspector()
        self.metrics_calculator = AdvancedMetricsCalculator()
        self.visualizer = AdvancedVisualizer()
        
        # Daten-Sammlung
        self.analysis_data = {
            'devices': {},
            'network_insights': {},
            'dpi_insights': {},
            'performance_metrics': {},
            'visualizations': {}
        }
    
    def analyze_wifi_data(self, 
                         wifi_events: List[Dict[str, Any]],
                         pcap_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Führt umfassende WLAN-Analyse durch.
        
        Args:
            wifi_events: Liste von WiFi-Events
            pcap_file: Optional PCAP-Datei für DPI
            
        Returns:
            Umfassende Analyse-Ergebnisse
        """
        logger.info(f"Starte erweiterte Analyse mit {len(wifi_events)} Events...")
        
        # 1. Basis-Daten verarbeiten
        self._process_wifi_events(wifi_events)
        
        # 2. DPI-Analyse (falls PCAP verfügbar)
        if pcap_file and Path(pcap_file).exists():
            self._perform_dpi_analysis(pcap_file)
        
        # 3. Erweiterte Metriken berechnen
        self._calculate_advanced_metrics()
        
        # 4. Visualisierungen erstellen
        self._create_visualizations()
        
        # 5. Netzwerk-Insights generieren
        self._generate_network_insights()
        
        logger.info("Erweiterte Analyse abgeschlossen")
        return self.analysis_data
    
    def _process_wifi_events(self, wifi_events: List[Dict[str, Any]]):
        """Verarbeitet WiFi-Events für erweiterte Analyse."""
        logger.info("Verarbeite WiFi-Events...")
        
        # Events nach Geräten gruppieren
        devices = {}
        for event in wifi_events:
            mac = event.get('client') or event.get('bssid')
            if not mac:
                continue
            
            if mac not in devices:
                devices[mac] = {
                    'mac_address': mac,
                    'events': [],
                    'signal_data': [],
                    'traffic_data': [],
                    'activity_data': []
                }
            
            devices[mac]['events'].append(event)
            
            # Signal-Daten extrahieren
            if 'rssi' in event:
                devices[mac]['signal_data'].append({
                    'timestamp': event['ts'],
                    'rssi': event['rssi'],
                    'noise': event.get('noise', -95),
                    'channel': event.get('channel', 0)
                })
            
            # Traffic-Daten extrahieren
            if event['type'] == 'data':
                packet_size = event.get('packet_size', 0)
                direction = 'upload' if event.get('from_ds', False) else 'download'
                
                devices[mac]['traffic_data'].append({
                    'timestamp': event['ts'],
                    'size': packet_size,
                    'direction': direction,
                    'channel': event.get('channel', 0)
                })
            
            # Aktivitäts-Daten extrahieren
            activity_level = 1.0 if event['type'] in ['data', 'beacon'] else 0.5
            devices[mac]['activity_data'].append({
                'timestamp': event['ts'],
                'activity_level': activity_level,
                'channel': event.get('channel', 0)
            })
        
        self.analysis_data['devices'] = devices
        logger.info(f"Verarbeitet {len(devices)} Geräte")
    
    def _perform_dpi_analysis(self, pcap_file: str):
        """Führt DPI-Analyse durch."""
        logger.info(f"Führe DPI-Analyse durch: {pcap_file}")
        
        try:
            from scapy.all import PcapReader
            
            dpi_events = []
            with PcapReader(pcap_file) as pcap_reader:
                for packet in pcap_reader:
                    dpi_analysis = self.dpi.analyze_packet(packet)
                    if dpi_analysis:
                        dpi_events.append(dpi_analysis)
            
            # DPI-Events nach Geräten gruppieren
            device_dpi = {}
            for dpi_event in dpi_events:
                mac = dpi_event.source_mac
                if mac not in device_dpi:
                    device_dpi[mac] = []
                device_dpi[mac].append(dpi_event)
            
            # DPI-Daten zu Geräten hinzufügen
            for mac, dpi_events in device_dpi.items():
                if mac in self.analysis_data['devices']:
                    self.analysis_data['devices'][mac]['dpi_events'] = [
                        event.__dict__ for event in dpi_events
                    ]
            
            # DPI-Insights generieren
            self.analysis_data['dpi_insights'] = self.dpi.get_network_insights()
            
            logger.info(f"DPI-Analyse abgeschlossen: {len(dpi_events)} Events")
            
        except Exception as e:
            logger.error(f"Fehler bei DPI-Analyse: {e}")
    
    def _calculate_advanced_metrics(self):
        """Berechnet erweiterte Metriken für alle Geräte."""
        logger.info("Berechne erweiterte Metriken...")
        
        for mac, device_data in self.analysis_data['devices'].items():
            try:
                # Signal Quality Metriken
                if device_data['signal_data']:
                    signal_metrics = self.metrics_calculator.calculate_signal_quality(
                        mac_address=mac,
                        rssi_values=[d['rssi'] for d in device_data['signal_data']],
                        noise_values=[d['noise'] for d in device_data['signal_data']]
                    )
                    device_data['signal_quality'] = signal_metrics.__dict__
                
                # Traffic Pattern Metriken
                if device_data['traffic_data']:
                    traffic_metrics = self.metrics_calculator.calculate_traffic_patterns(
                        mac_address=mac,
                        packet_data=device_data['traffic_data']
                    )
                    device_data['traffic_patterns'] = traffic_metrics.__dict__
                
                # Activity Heatmap
                if device_data['activity_data']:
                    activity_heatmap = self.metrics_calculator.create_activity_heatmap(
                        mac_address=mac,
                        activity_data=device_data['activity_data']
                    )
                    device_data['activity_heatmap'] = activity_heatmap.__dict__
                
                # DPI Fingerprint
                device_data['dpi_fingerprint'] = self.dpi.get_device_fingerprint(mac)
                
            except Exception as e:
                logger.debug(f"Fehler bei Metriken-Berechnung für {mac}: {e}")
        
        # Netzwerk-Insights
        self.analysis_data['network_insights'] = self.metrics_calculator.get_network_insights()
        
        # Performance-Benchmark
        try:
            benchmark = self.metrics_calculator.run_performance_benchmark("enhanced_analysis")
            self.analysis_data['performance_metrics'] = benchmark.__dict__
        except Exception as e:
            logger.debug(f"Fehler bei Performance-Benchmark: {e}")
    
    def _create_visualizations(self):
        """Erstellt erweiterte Visualisierungen."""
        logger.info("Erstelle Visualisierungen...")
        
        try:
            # 1. 3D Network Visualization
            devices_for_3d = []
            connections = []
            
            for mac, device_data in self.analysis_data['devices'].items():
                device_info = {
                    'mac_address': mac,
                    'device_type': self._determine_device_type(device_data),
                    'signal_strength': device_data.get('signal_quality', {}).get('rssi_mean', -70),
                    'channel': device_data.get('signal_data', [{}])[0].get('channel', 0) if device_data.get('signal_data') else 0
                }
                devices_for_3d.append(device_info)
            
            if devices_for_3d:
                fig_3d = self.visualizer.create_3d_network_visualization(devices_for_3d, connections)
                self.analysis_data['visualizations']['3d_network'] = fig_3d
            
            # 2. Time-series Plots
            time_series_data = self._prepare_time_series_data()
            if not time_series_data.empty:
                time_series_figs = self.visualizer.create_time_series_plots(time_series_data)
                self.analysis_data['visualizations']['time_series'] = time_series_figs
            
            # 3. Signal Quality Heatmap
            signal_heatmap_data = self._prepare_signal_heatmap_data()
            if not signal_heatmap_data.empty:
                fig_heatmap = self.visualizer.create_signal_quality_heatmap(signal_heatmap_data)
                self.analysis_data['visualizations']['signal_heatmap'] = fig_heatmap
            
            # 4. Device Activity Heatmap
            activity_heatmap_data = self._prepare_activity_heatmap_data()
            if activity_heatmap_data:
                fig_activity = self.visualizer.create_device_activity_heatmap(activity_heatmap_data)
                self.analysis_data['visualizations']['activity_heatmap'] = fig_activity
            
            # 5. Network Topology Diagram
            if devices_for_3d:
                fig_topology = self.visualizer.create_network_topology_diagram(devices_for_3d, connections)
                self.analysis_data['visualizations']['network_topology'] = fig_topology
            
            logger.info("Visualisierungen erstellt")
            
        except Exception as e:
            logger.error(f"Fehler bei Visualisierung: {e}")
    
    def _determine_device_type(self, device_data: Dict[str, Any]) -> str:
        """Bestimmt Gerätetyp basierend auf verfügbaren Daten."""
        # Vereinfachte Geräteerkennung
        mac = device_data.get('mac_address', '').upper()
        
        # DPI-basierte Erkennung
        dpi_fingerprint = device_data.get('dpi_fingerprint', {})
        protocols = dpi_fingerprint.get('protocols', set())
        
        if 'HTTP' in protocols:
            user_agents = dpi_fingerprint.get('user_agents', set())
            for ua in user_agents:
                ua_lower = ua.lower()
                if 'android' in ua_lower:
                    return 'Android Device'
                elif 'iphone' in ua_lower or 'ipad' in ua_lower:
                    return 'iOS Device'
                elif 'windows' in ua_lower:
                    return 'Windows Device'
                elif 'linux' in ua_lower:
                    return 'Linux Device'
        
        # Signal-basierte Erkennung
        signal_quality = device_data.get('signal_quality', {})
        if signal_quality.get('rssi_mean', -100) > -50:
            return 'AP/Router'
        elif signal_quality.get('rssi_mean', -100) > -70:
            return 'Client (Near)'
        else:
            return 'Client (Far)'
    
    def _prepare_time_series_data(self) -> pd.DataFrame:
        """Bereitet Zeitreihen-Daten für Visualisierung vor."""
        data = []
        
        for mac, device_data in self.analysis_data['devices'].items():
            for signal in device_data.get('signal_data', []):
                data.append({
                    'timestamp': signal['timestamp'],
                    'mac_address': mac,
                    'rssi': signal['rssi'],
                    'noise': signal['noise'],
                    'channel': signal['channel']
                })
            
            for traffic in device_data.get('traffic_data', []):
                data.append({
                    'timestamp': traffic['timestamp'],
                    'mac_address': mac,
                    'throughput': traffic['size'] / 1024,  # KB
                    'direction': traffic['direction'],
                    'channel': traffic['channel']
                })
        
        return pd.DataFrame(data)
    
    def _prepare_signal_heatmap_data(self) -> pd.DataFrame:
        """Bereitet Signal-Heatmap-Daten vor."""
        data = []
        
        for mac, device_data in self.analysis_data['devices'].items():
            device_type = self._determine_device_type(device_data)
            signal_quality = device_data.get('signal_quality', {})
            
            if signal_quality:
                data.append({
                    'mac_address': mac,
                    'device_type': device_type,
                    'channel': device_data.get('signal_data', [{}])[0].get('channel', 0) if device_data.get('signal_data') else 0,
                    'rssi': signal_quality.get('rssi_mean', -70),
                    'snr': signal_quality.get('snr_mean', 0)
                })
        
        return pd.DataFrame(data)
    
    def _prepare_activity_heatmap_data(self) -> Dict[str, List[float]]:
        """Bereitet Activity-Heatmap-Daten vor."""
        activity_data = {}
        
        for mac, device_data in self.analysis_data['devices'].items():
            activity_list = device_data.get('activity_data', [])
            if activity_list:
                # Aktivitätsdaten in Zeitfenster gruppieren (15-Minuten-Intervalle)
                activity_values = []
                for activity in activity_list:
                    activity_values.append(activity['activity_level'])
                
                activity_data[mac] = activity_values
        
        return activity_data
    
    def _generate_network_insights(self):
        """Generiert umfassende Netzwerk-Insights."""
        logger.info("Generiere Netzwerk-Insights...")
        
        insights = {
            'total_devices': len(self.analysis_data['devices']),
            'device_types': {},
            'signal_quality_summary': {},
            'traffic_summary': {},
            'security_insights': {},
            'recommendations': []
        }
        
        # Gerätetyp-Verteilung
        device_types = {}
        for device_data in self.analysis_data['devices'].values():
            device_type = self._determine_device_type(device_data)
            device_types[device_type] = device_types.get(device_type, 0) + 1
        insights['device_types'] = device_types
        
        # Signal Quality Summary
        all_rssi = []
        all_snr = []
        for device_data in self.analysis_data['devices'].values():
            signal_quality = device_data.get('signal_quality', {})
            if signal_quality:
                all_rssi.append(signal_quality.get('rssi_mean', -70))
                all_snr.append(signal_quality.get('snr_mean', 0))
        
        if all_rssi:
            insights['signal_quality_summary'] = {
                'avg_rssi': np.mean(all_rssi),
                'min_rssi': np.min(all_rssi),
                'max_rssi': np.max(all_rssi),
                'avg_snr': np.mean(all_snr) if all_snr else 0
            }
        
        # Traffic Summary
        total_throughput = 0
        for device_data in self.analysis_data['devices'].values():
            traffic_patterns = device_data.get('traffic_patterns', {})
            if traffic_patterns:
                total_throughput += traffic_patterns.get('upload_rate_mbps', 0) + traffic_patterns.get('download_rate_mbps', 0)
        
        insights['traffic_summary'] = {
            'total_throughput_mbps': total_throughput,
            'active_devices': len([d for d in self.analysis_data['devices'].values() if d.get('traffic_data')])
        }
        
        # Security Insights
        suspicious_activity = 0
        for device_data in self.analysis_data['devices'].values():
            dpi_fingerprint = device_data.get('dpi_fingerprint', {})
            if dpi_fingerprint.get('suspicious_activity'):
                suspicious_activity += 1
        
        insights['security_insights'] = {
            'suspicious_devices': suspicious_activity,
            'encrypted_traffic': len([d for d in self.analysis_data['devices'].values() if 'HTTPS' in d.get('dpi_fingerprint', {}).get('protocols', set())])
        }
        
        # Empfehlungen generieren
        recommendations = []
        
        if insights['signal_quality_summary'].get('avg_rssi', -100) < -70:
            recommendations.append("Signalstärke ist schwach - Router-Position überprüfen")
        
        if suspicious_activity > 0:
            recommendations.append(f"{suspicious_activity} Geräte zeigen verdächtige Aktivität - weitere Untersuchung empfohlen")
        
        if insights['traffic_summary']['total_throughput_mbps'] > 100:
            recommendations.append("Hoher Netzwerk-Durchsatz - Bandbreiten-Management prüfen")
        
        insights['recommendations'] = recommendations
        
        self.analysis_data['network_insights'] = insights
    
    def save_analysis_results(self):
        """Speichert Analyse-Ergebnisse."""
        logger.info("Speichere Analyse-Ergebnisse...")
        
        # JSON-Report
        report_path = self.output_dir / "enhanced_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.analysis_data, f, indent=2, default=str)
        
        # HTML-Report
        html_report_path = self.output_dir / "enhanced_analysis_report.html"
        self.visualizer.create_custom_report(
            self.analysis_data,
            str(html_report_path)
        )
        
        # Visualisierungen speichern
        for name, fig in self.analysis_data['visualizations'].items():
            if hasattr(fig, 'write_html'):
                # Plotly Figure
                output_path = self.output_dir / f"{name}.html"
                fig.write_html(str(output_path))
            else:
                # Matplotlib Figure
                output_path = self.output_dir / f"{name}.png"
                self.visualizer.save_plot(fig, str(output_path))
        
        logger.info(f"Analyse-Ergebnisse gespeichert in: {self.output_dir}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Gibt Zusammenfassung der Analyse zurück."""
        return {
            'total_devices': len(self.analysis_data['devices']),
            'analysis_timestamp': datetime.now().isoformat(),
            'network_insights': self.analysis_data.get('network_insights', {}),
            'performance_metrics': self.analysis_data.get('performance_metrics', {}),
            'visualizations_created': list(self.analysis_data['visualizations'].keys())
        }