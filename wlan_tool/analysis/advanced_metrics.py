#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweiterte Metriken und Statistiken für WLAN-Analyse.
Implementiert Signal Quality Metrics, Traffic Pattern Analysis und Performance Benchmarking.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalQualityMetrics:
    """Signal Quality Metriken für ein Gerät."""
    mac_address: str
    rssi_mean: float
    rssi_std: float
    rssi_min: int
    rssi_max: int
    snr_mean: float
    snr_std: float
    snr_min: float
    snr_max: float
    packet_error_rate: float
    signal_stability: float
    channel_utilization: float
    noise_floor: float
    signal_to_noise_ratio: float
    timestamp: datetime


@dataclass
class TrafficPatternMetrics:
    """Traffic Pattern Metriken."""
    mac_address: str
    upload_rate_mbps: float
    download_rate_mbps: float
    total_bytes: int
    packet_count: int
    burst_count: int
    avg_packet_size: float
    inter_arrival_time_ms: float
    traffic_variance: float
    peak_throughput_mbps: float
    timestamp: datetime


@dataclass
class DeviceActivityHeatmap:
    """Device Activity Heatmap Daten."""
    mac_address: str
    time_slots: List[datetime]
    activity_levels: List[float]
    channel_activity: Dict[int, List[float]]
    peak_hours: List[int]
    activity_pattern: str  # "constant", "bursty", "periodic", "random"


@dataclass
class NetworkTopologyNode:
    """Netzwerk-Topologie Knoten."""
    mac_address: str
    device_type: str
    ip_address: Optional[str]
    ssid: Optional[str]
    channel: int
    signal_strength: float
    connections: List[str]  # Verbundene MAC-Adressen
    role: str  # "ap", "client", "bridge", "unknown"


@dataclass
class PerformanceBenchmark:
    """Performance Benchmark Ergebnisse."""
    test_name: str
    throughput_mbps: float
    latency_ms: float
    packet_loss_rate: float
    jitter_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    timestamp: datetime


class AdvancedMetricsCalculator:
    """Berechnet erweiterte Metriken für WLAN-Analyse."""
    
    def __init__(self):
        """Initialisiert den Metriken-Rechner."""
        self.signal_history = defaultdict(list)
        self.traffic_history = defaultdict(list)
        self.activity_history = defaultdict(list)
        self.performance_data = []
        
    def calculate_signal_quality(self, mac_address: str, 
                               rssi_values: List[int],
                               noise_values: List[int] = None,
                               packet_errors: int = 0,
                               total_packets: int = 0) -> SignalQualityMetrics:
        """
        Berechnet Signal Quality Metriken.
        
        Args:
            mac_address: MAC-Adresse des Geräts
            rssi_values: Liste von RSSI-Werten
            noise_values: Liste von Noise-Werten
            packet_errors: Anzahl fehlerhafter Pakete
            total_packets: Gesamtanzahl Pakete
            
        Returns:
            SignalQualityMetrics
        """
        if not rssi_values:
            raise ValueError("RSSI-Werte sind erforderlich")
        
        # RSSI-Statistiken
        rssi_mean = statistics.mean(rssi_values)
        rssi_std = statistics.stdev(rssi_values) if len(rssi_values) > 1 else 0.0
        rssi_min = min(rssi_values)
        rssi_max = max(rssi_values)
        
        # SNR-Berechnung
        if noise_values:
            noise_floor = statistics.mean(noise_values)
            snr_values = [rssi - noise for rssi, noise in zip(rssi_values, noise_values)]
            snr_mean = statistics.mean(snr_values)
            snr_std = statistics.stdev(snr_values) if len(snr_values) > 1 else 0.0
            snr_min = min(snr_values)
            snr_max = max(snr_values)
        else:
            noise_floor = -95.0  # Default Noise Floor
            snr_values = [rssi - noise_floor for rssi in rssi_values]
            snr_mean = statistics.mean(snr_values)
            snr_std = statistics.stdev(snr_values) if len(snr_values) > 1 else 0.0
            snr_min = min(snr_values)
            snr_max = max(snr_values)
        
        # Packet Error Rate
        per = packet_errors / total_packets if total_packets > 0 else 0.0
        
        # Signal Stability (invers zur Varianz)
        signal_stability = 1.0 / (1.0 + rssi_std) if rssi_std > 0 else 1.0
        
        # Channel Utilization (vereinfacht)
        channel_utilization = min(1.0, total_packets / 1000.0)  # Normalisiert
        
        # Signal-to-Noise Ratio
        signal_to_noise_ratio = rssi_mean - noise_floor
        
        return SignalQualityMetrics(
            mac_address=mac_address,
            rssi_mean=rssi_mean,
            rssi_std=rssi_std,
            rssi_min=rssi_min,
            rssi_max=rssi_max,
            snr_mean=snr_mean,
            snr_std=snr_std,
            snr_min=snr_min,
            snr_max=snr_max,
            packet_error_rate=per,
            signal_stability=signal_stability,
            channel_utilization=channel_utilization,
            noise_floor=noise_floor,
            signal_to_noise_ratio=signal_to_noise_ratio,
            timestamp=datetime.now()
        )
    
    def calculate_traffic_patterns(self, mac_address: str,
                                 packet_data: List[Dict[str, Any]]) -> TrafficPatternMetrics:
        """
        Berechnet Traffic Pattern Metriken.
        
        Args:
            mac_address: MAC-Adresse des Geräts
            packet_data: Liste von Paket-Daten mit 'size', 'timestamp', 'direction'
            
        Returns:
            TrafficPatternMetrics
        """
        if not packet_data:
            raise ValueError("Paket-Daten sind erforderlich")
        
        # Daten sortieren nach Timestamp
        sorted_data = sorted(packet_data, key=lambda x: x['timestamp'])
        
        # Upload/Download trennen
        upload_packets = [p for p in sorted_data if p.get('direction') == 'upload']
        download_packets = [p for p in sorted_data if p.get('direction') == 'download']
        
        # Durchsatz berechnen
        total_bytes = sum(p['size'] for p in sorted_data)
        packet_count = len(sorted_data)
        
        # Zeitraum berechnen
        if len(sorted_data) > 1:
            time_span = (sorted_data[-1]['timestamp'] - sorted_data[0]['timestamp']).total_seconds()
            if time_span > 0:
                total_throughput_mbps = (total_bytes * 8) / (time_span * 1_000_000)
            else:
                total_throughput_mbps = 0.0
        else:
            total_throughput_mbps = 0.0
        
        # Upload/Download Raten
        upload_bytes = sum(p['size'] for p in upload_packets)
        download_bytes = sum(p['size'] for p in download_packets)
        
        upload_rate_mbps = (upload_bytes * 8) / (time_span * 1_000_000) if time_span > 0 else 0.0
        download_rate_mbps = (download_bytes * 8) / (time_span * 1_000_000) if time_span > 0 else 0.0
        
        # Burst-Erkennung
        burst_count = self._detect_bursts(sorted_data)
        
        # Durchschnittliche Paketgröße
        avg_packet_size = total_bytes / packet_count if packet_count > 0 else 0.0
        
        # Inter-Arrival Time
        if len(sorted_data) > 1:
            intervals = []
            for i in range(1, len(sorted_data)):
                interval = (sorted_data[i]['timestamp'] - sorted_data[i-1]['timestamp']).total_seconds() * 1000
                intervals.append(interval)
            inter_arrival_time_ms = statistics.mean(intervals) if intervals else 0.0
        else:
            inter_arrival_time_ms = 0.0
        
        # Traffic Variance
        packet_sizes = [p['size'] for p in sorted_data]
        traffic_variance = statistics.variance(packet_sizes) if len(packet_sizes) > 1 else 0.0
        
        # Peak Throughput (1-Sekunden-Fenster)
        peak_throughput_mbps = self._calculate_peak_throughput(sorted_data)
        
        return TrafficPatternMetrics(
            mac_address=mac_address,
            upload_rate_mbps=upload_rate_mbps,
            download_rate_mbps=download_rate_mbps,
            total_bytes=total_bytes,
            packet_count=packet_count,
            burst_count=burst_count,
            avg_packet_size=avg_packet_size,
            inter_arrival_time_ms=inter_arrival_time_ms,
            traffic_variance=traffic_variance,
            peak_throughput_mbps=peak_throughput_mbps,
            timestamp=datetime.now()
        )
    
    def _detect_bursts(self, packet_data: List[Dict[str, Any]], 
                      burst_threshold: float = 0.1) -> int:
        """Erkennt Burst-Patterns in Paket-Daten."""
        if len(packet_data) < 2:
            return 0
        
        burst_count = 0
        current_burst = False
        
        for i in range(1, len(packet_data)):
            time_diff = (packet_data[i]['timestamp'] - packet_data[i-1]['timestamp']).total_seconds()
            
            if time_diff < burst_threshold:
                if not current_burst:
                    burst_count += 1
                    current_burst = True
            else:
                current_burst = False
        
        return burst_count
    
    def _calculate_peak_throughput(self, packet_data: List[Dict[str, Any]], 
                                 window_size: int = 1) -> float:
        """Berechnet Peak-Throughput in einem Zeitfenster."""
        if len(packet_data) < 2:
            return 0.0
        
        max_throughput = 0.0
        
        for i in range(len(packet_data) - window_size):
            window_data = packet_data[i:i + window_size + 1]
            
            if len(window_data) > 1:
                time_span = (window_data[-1]['timestamp'] - window_data[0]['timestamp']).total_seconds()
                if time_span > 0:
                    window_bytes = sum(p['size'] for p in window_data)
                    throughput = (window_bytes * 8) / (time_span * 1_000_000)
                    max_throughput = max(max_throughput, throughput)
        
        return max_throughput
    
    def create_activity_heatmap(self, mac_address: str,
                              activity_data: List[Dict[str, Any]],
                              time_resolution_minutes: int = 15) -> DeviceActivityHeatmap:
        """
        Erstellt Activity Heatmap für ein Gerät.
        
        Args:
            mac_address: MAC-Adresse des Geräts
            activity_data: Liste von Aktivitätsdaten mit 'timestamp', 'channel', 'activity_level'
            time_resolution_minutes: Zeitauflösung in Minuten
            
        Returns:
            DeviceActivityHeatmap
        """
        if not activity_data:
            return DeviceActivityHeatmap(
                mac_address=mac_address,
                time_slots=[],
                activity_levels=[],
                channel_activity={},
                peak_hours=[],
                activity_pattern="constant"
            )
        
        # Zeiträume erstellen
        start_time = min(d['timestamp'] for d in activity_data)
        end_time = max(d['timestamp'] for d in activity_data)
        
        time_slots = []
        current_time = start_time
        while current_time <= end_time:
            time_slots.append(current_time)
            current_time += timedelta(minutes=time_resolution_minutes)
        
        # Aktivitätslevel pro Zeitslot berechnen
        activity_levels = []
        for slot in time_slots:
            slot_end = slot + timedelta(minutes=time_resolution_minutes)
            slot_data = [d for d in activity_data 
                        if slot <= d['timestamp'] < slot_end]
            
            if slot_data:
                avg_activity = statistics.mean(d['activity_level'] for d in slot_data)
            else:
                avg_activity = 0.0
            
            activity_levels.append(avg_activity)
        
        # Kanal-Aktivität
        channel_activity = defaultdict(list)
        for data in activity_data:
            channel = data.get('channel', 0)
            channel_activity[channel].append(data['activity_level'])
        
        # Peak-Hours identifizieren
        peak_hours = []
        if activity_levels:
            threshold = statistics.mean(activity_levels) + statistics.stdev(activity_levels)
            for i, level in enumerate(activity_levels):
                if level > threshold:
                    peak_hours.append(time_slots[i].hour)
        
        # Aktivitäts-Pattern erkennen
        activity_pattern = self._classify_activity_pattern(activity_levels)
        
        return DeviceActivityHeatmap(
            mac_address=mac_address,
            time_slots=time_slots,
            activity_levels=activity_levels,
            channel_activity=dict(channel_activity),
            peak_hours=list(set(peak_hours)),
            activity_pattern=activity_pattern
        )
    
    def _classify_activity_pattern(self, activity_levels: List[float]) -> str:
        """Klassifiziert das Aktivitäts-Pattern."""
        if not activity_levels:
            return "constant"
        
        # Varianz berechnen
        variance = statistics.variance(activity_levels) if len(activity_levels) > 1 else 0.0
        mean_activity = statistics.mean(activity_levels)
        
        # Pattern-Klassifizierung
        if variance < 0.1:
            return "constant"
        elif variance > 1.0:
            return "bursty"
        else:
            # Periodizität prüfen (vereinfacht)
            if len(activity_levels) > 24:  # Mindestens 24 Zeitslots
                return "periodic"
            else:
                return "random"
    
    def build_network_topology(self, devices: List[Dict[str, Any]]) -> List[NetworkTopologyNode]:
        """
        Baut Netzwerk-Topologie auf.
        
        Args:
            devices: Liste von Gerätedaten
            
        Returns:
            Liste von NetworkTopologyNode
        """
        topology_nodes = []
        
        for device in devices:
            # Gerätetyp bestimmen
            device_type = self._determine_device_type(device)
            
            # Rolle bestimmen
            role = self._determine_device_role(device)
            
            # Verbindungen finden
            connections = self._find_connections(device, devices)
            
            node = NetworkTopologyNode(
                mac_address=device.get('mac_address', ''),
                device_type=device_type,
                ip_address=device.get('ip_address'),
                ssid=device.get('ssid'),
                channel=device.get('channel', 0),
                signal_strength=device.get('signal_strength', 0.0),
                connections=connections,
                role=role
            )
            
            topology_nodes.append(node)
        
        return topology_nodes
    
    def _determine_device_type(self, device: Dict[str, Any]) -> str:
        """Bestimmt den Gerätetyp basierend auf verfügbaren Daten."""
        # Vereinfachte Geräteerkennung
        mac = device.get('mac_address', '').upper()
        
        # OUI-basierte Erkennung (vereinfacht)
        if mac.startswith('00:50:56') or mac.startswith('08:00:27'):
            return "Virtual"
        elif mac.startswith('00:1B:44') or mac.startswith('00:0C:29'):
            return "VMware"
        elif mac.startswith('52:54:00'):
            return "QEMU"
        else:
            return "Physical"
    
    def _determine_device_role(self, device: Dict[str, Any]) -> str:
        """Bestimmt die Rolle des Geräts im Netzwerk."""
        # Vereinfachte Rollenbestimmung
        if device.get('is_ap', False):
            return "ap"
        elif device.get('is_client', False):
            return "client"
        elif device.get('is_bridge', False):
            return "bridge"
        else:
            return "unknown"
    
    def _find_connections(self, device: Dict[str, Any], 
                         all_devices: List[Dict[str, Any]]) -> List[str]:
        """Findet Verbindungen zu anderen Geräten."""
        connections = []
        
        # Vereinfachte Verbindungserkennung
        # In einer echten Implementierung würde man Paket-Flows analysieren
        for other_device in all_devices:
            if other_device['mac_address'] != device['mac_address']:
                # Gleicher Kanal = mögliche Verbindung
                if device.get('channel') == other_device.get('channel'):
                    connections.append(other_device['mac_address'])
        
        return connections
    
    def run_performance_benchmark(self, test_name: str,
                                test_duration_seconds: int = 60) -> PerformanceBenchmark:
        """
        Führt Performance-Benchmark durch.
        
        Args:
            test_name: Name des Tests
            test_duration_seconds: Testdauer in Sekunden
            
        Returns:
            PerformanceBenchmark
        """
        import psutil
        import time
        
        # System-Metriken vor Test
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Simuliere Test-Arbeit
        # In einer echten Implementierung würde hier echte Netzwerk-Tests laufen
        test_data = []
        for i in range(test_duration_seconds * 10):  # 10 Tests pro Sekunde
            test_data.append({
                'timestamp': datetime.now(),
                'data': np.random.bytes(1024)  # 1KB Test-Daten
            })
            time.sleep(0.1)
        
        end_time = time.time()
        
        # System-Metriken nach Test
        cpu_after = psutil.cpu_percent()
        memory_after = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Berechnungen
        actual_duration = end_time - start_time
        total_data = len(test_data) * 1024  # Bytes
        throughput_mbps = (total_data * 8) / (actual_duration * 1_000_000)
        
        # Latenz (vereinfacht)
        latency_ms = actual_duration * 1000 / len(test_data)
        
        # Jitter (Varianz der Latenz)
        latencies = [0.1] * len(test_data)  # Vereinfacht
        jitter_ms = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0
        
        # Packet Loss (vereinfacht)
        packet_loss_rate = 0.0  # In echten Tests würde man verlorene Pakete zählen
        
        benchmark = PerformanceBenchmark(
            test_name=test_name,
            throughput_mbps=throughput_mbps,
            latency_ms=latency_ms,
            packet_loss_rate=packet_loss_rate,
            jitter_ms=jitter_ms,
            cpu_usage_percent=(cpu_before + cpu_after) / 2,
            memory_usage_mb=memory_after - memory_before,
            timestamp=datetime.now()
        )
        
        self.performance_data.append(benchmark)
        return benchmark
    
    def get_network_insights(self) -> Dict[str, Any]:
        """Gibt umfassende Netzwerk-Insights zurück."""
        insights = {
            "total_devices": len(self.signal_history),
            "signal_quality_summary": {},
            "traffic_patterns": {},
            "activity_patterns": {},
            "performance_metrics": {},
            "network_health_score": 0.0
        }
        
        # Signal Quality Summary
        if self.signal_history:
            all_rssi = []
            for mac, rssi_list in self.signal_history.items():
                all_rssi.extend(rssi_list)
            
            insights["signal_quality_summary"] = {
                "avg_rssi": statistics.mean(all_rssi),
                "min_rssi": min(all_rssi),
                "max_rssi": max(all_rssi),
                "rssi_std": statistics.stdev(all_rssi) if len(all_rssi) > 1 else 0.0
            }
        
        # Traffic Patterns
        if self.traffic_history:
            total_throughput = 0.0
            for mac, traffic_list in self.traffic_history.items():
                for traffic in traffic_list:
                    total_throughput += traffic.get('throughput', 0.0)
            
            insights["traffic_patterns"] = {
                "total_throughput_mbps": total_throughput,
                "active_devices": len(self.traffic_history)
            }
        
        # Network Health Score (vereinfacht)
        health_factors = []
        if insights["signal_quality_summary"]:
            avg_rssi = insights["signal_quality_summary"]["avg_rssi"]
            if avg_rssi > -50:
                health_factors.append(1.0)
            elif avg_rssi > -70:
                health_factors.append(0.7)
            else:
                health_factors.append(0.3)
        
        if health_factors:
            insights["network_health_score"] = statistics.mean(health_factors)
        
        return insights