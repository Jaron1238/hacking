#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweiterter WiFi-Sniffer mit Deep Packet Inspection und erweiterten Metriken.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from tqdm import tqdm
from scapy.all import PcapReader, wrpcap

from .. import config
from ..storage import database
from ..storage.data_models import WifiEvent
from ..analysis.deep_packet_inspection import DeepPacketInspector
from ..analysis.advanced_metrics import AdvancedMetricsCalculator
from .sniffer import packet_to_event, ChannelHopper

logger = logging.getLogger(__name__)


class EnhancedWiFiSniffer:
    """Erweiterter WiFi-Sniffer mit DPI und Metriken."""
    
    def __init__(self, iface: str, channels: List[int], duration: int, outdir: str):
        self.iface = iface
        self.channels = channels
        self.duration = duration
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.outdir / "db" / "events.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.pcap_path = self.outdir / "capture.pcap"
        self.hopper = None
        self.stop_event = threading.Event()
        self.packet_count = 0
        self.start_time = None
        
        # Erweiterte Analyse-Module
        self.dpi = DeepPacketInspector()
        self.metrics_calculator = AdvancedMetricsCalculator()
        
        # Daten-Sammlung
        self.signal_data = defaultdict(list)
        self.traffic_data = defaultdict(list)
        self.activity_data = defaultdict(list)
        self.dpi_events = []
        
    def start_capture(self):
        """Startet die erweiterte Paket-Erfassung."""
        logger.info("Starte erweiterte WiFi-Erfassung...")
        self.start_time = time.time()
        
        # Channel Hopper starten
        self.hopper = ChannelHopper(
            self.iface, 
            self.channels, 
            config.CHANNEL_HOP_SLEEP_S
        )
        self.hopper.start()
        
        # Sniffing mit erweiterten Features
        self._run_enhanced_sniffing()
        
    def _run_enhanced_sniffing(self):
        """Führt erweiterte Sniffing-Operation durch."""
        try:
            from .sniffer import sniff_with_writer
            
            # Live-Queue für erweiterte Analyse
            live_queue = mp.Queue(maxsize=1000)
            
            # Sniffing starten
            sniff_with_writer(
                iface=self.iface,
                duration=self.duration,
                db_path=str(self.db_path),
                pcap_out=str(self.pcap_path),
                live_queue=live_queue,
                channels_override=self.channels
            )
            
            # Erweiterte Analyse der Live-Daten
            self._process_live_data(live_queue)
            
        except Exception as e:
            logger.error(f"Fehler beim erweiterten Sniffing: {e}")
        finally:
            if self.hopper:
                self.hopper.stop()
    
    def _process_live_data(self, live_queue: mp.Queue):
        """Verarbeitet Live-Daten für erweiterte Analyse."""
        logger.info("Starte Live-Datenverarbeitung...")
        
        while not self.stop_event.is_set():
            try:
                # Event aus Queue holen
                event = live_queue.get(timeout=1.0)
                if event is None:
                    break
                
                # Basis-Event verarbeiten
                self._process_wifi_event(event)
                
                # DPI-Analyse (falls PCAP verfügbar)
                self._process_dpi_analysis(event)
                
                # Metriken sammeln
                self._collect_metrics(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"Fehler bei Live-Datenverarbeitung: {e}")
    
    def _process_wifi_event(self, event: WifiEvent):
        """Verarbeitet WiFi-Event für erweiterte Analyse."""
        mac = event.get('client') or event.get('bssid')
        if not mac:
            return
        
        # Signal-Daten sammeln
        if 'rssi' in event:
            self.signal_data[mac].append({
                'timestamp': event['ts'],
                'rssi': event['rssi'],
                'noise': event.get('noise', -95),
                'channel': event.get('channel', 0)
            })
        
        # Traffic-Daten sammeln
        if event['type'] == 'data':
            packet_size = event.get('packet_size', 0)
            direction = 'upload' if event.get('from_ds', False) else 'download'
            
            self.traffic_data[mac].append({
                'timestamp': event['ts'],
                'size': packet_size,
                'direction': direction,
                'channel': event.get('channel', 0)
            })
        
        # Aktivitäts-Daten sammeln
        activity_level = 1.0 if event['type'] in ['data', 'beacon'] else 0.5
        self.activity_data[mac].append({
            'timestamp': event['ts'],
            'activity_level': activity_level,
            'channel': event.get('channel', 0)
        })
    
    def _process_dpi_analysis(self, event: WifiEvent):
        """Führt DPI-Analyse durch."""
        # Hier würde man das ursprüngliche Paket für DPI verwenden
        # Da wir nur Events haben, simulieren wir DPI-Daten
        if event['type'] == 'data' and 'dns_query' in event:
            # DNS-Analyse
            dns_analysis = self.dpi._analyze_dns_simulation(event)
            if dns_analysis:
                self.dpi_events.append({
                    'timestamp': event['ts'],
                    'mac': event.get('client'),
                    'protocol': 'DNS',
                    'analysis': dns_analysis.__dict__
                })
    
    def _collect_metrics(self, event: WifiEvent):
        """Sammelt Metriken für erweiterte Analyse."""
        mac = event.get('client') or event.get('bssid')
        if not mac:
            return
        
        # Signal Quality Metriken berechnen (alle 10 Events)
        if len(self.signal_data[mac]) % 10 == 0 and len(self.signal_data[mac]) > 0:
            try:
                signal_metrics = self.metrics_calculator.calculate_signal_quality(
                    mac_address=mac,
                    rssi_values=[d['rssi'] for d in self.signal_data[mac]],
                    noise_values=[d['noise'] for d in self.signal_data[mac]]
                )
                logger.debug(f"Signal Quality für {mac}: {signal_metrics.snr_mean:.1f} dB SNR")
            except Exception as e:
                logger.debug(f"Fehler bei Signal-Quality-Berechnung: {e}")
        
        # Traffic Pattern Metriken berechnen (alle 50 Events)
        if len(self.traffic_data[mac]) % 50 == 0 and len(self.traffic_data[mac]) > 0:
            try:
                traffic_metrics = self.metrics_calculator.calculate_traffic_patterns(
                    mac_address=mac,
                    packet_data=self.traffic_data[mac]
                )
                logger.debug(f"Traffic Pattern für {mac}: {traffic_metrics.upload_rate_mbps:.1f} Mbps up, {traffic_metrics.download_rate_mbps:.1f} Mbps down")
            except Exception as e:
                logger.debug(f"Fehler bei Traffic-Pattern-Berechnung: {e}")
    
    def generate_enhanced_report(self) -> Dict[str, Any]:
        """Generiert erweiterten Analyse-Report."""
        logger.info("Generiere erweiterten Analyse-Report...")
        
        report = {
            'timestamp': time.time(),
            'duration': self.duration,
            'devices': {},
            'network_insights': {},
            'dpi_insights': {},
            'performance_metrics': {}
        }
        
        # Geräte-Analyse
        for mac in set(list(self.signal_data.keys()) + list(self.traffic_data.keys())):
            device_report = {
                'mac_address': mac,
                'signal_quality': None,
                'traffic_patterns': None,
                'activity_heatmap': None,
                'dpi_fingerprint': None
            }
            
            # Signal Quality
            if mac in self.signal_data and len(self.signal_data[mac]) > 0:
                try:
                    signal_metrics = self.metrics_calculator.calculate_signal_quality(
                        mac_address=mac,
                        rssi_values=[d['rssi'] for d in self.signal_data[mac]],
                        noise_values=[d['noise'] for d in self.signal_data[mac]]
                    )
                    device_report['signal_quality'] = signal_metrics.__dict__
                except Exception as e:
                    logger.debug(f"Fehler bei Signal-Quality für {mac}: {e}")
            
            # Traffic Patterns
            if mac in self.traffic_data and len(self.traffic_data[mac]) > 0:
                try:
                    traffic_metrics = self.metrics_calculator.calculate_traffic_patterns(
                        mac_address=mac,
                        packet_data=self.traffic_data[mac]
                    )
                    device_report['traffic_patterns'] = traffic_metrics.__dict__
                except Exception as e:
                    logger.debug(f"Fehler bei Traffic-Pattern für {mac}: {e}")
            
            # Activity Heatmap
            if mac in self.activity_data and len(self.activity_data[mac]) > 0:
                try:
                    activity_heatmap = self.metrics_calculator.create_activity_heatmap(
                        mac_address=mac,
                        activity_data=self.activity_data[mac]
                    )
                    device_report['activity_heatmap'] = activity_heatmap.__dict__
                except Exception as e:
                    logger.debug(f"Fehler bei Activity-Heatmap für {mac}: {e}")
            
            # DPI Fingerprint
            device_report['dpi_fingerprint'] = self.dpi.get_device_fingerprint(mac)
            
            report['devices'][mac] = device_report
        
        # Netzwerk-Insights
        report['network_insights'] = self.metrics_calculator.get_network_insights()
        report['dpi_insights'] = self.dpi.get_network_insights()
        
        # Performance-Metriken
        try:
            benchmark = self.metrics_calculator.run_performance_benchmark("wifi_analysis")
            report['performance_metrics'] = benchmark.__dict__
        except Exception as e:
            logger.debug(f"Fehler bei Performance-Benchmark: {e}")
        
        return report
    
    def save_enhanced_data(self, report: Dict[str, Any]):
        """Speichert erweiterte Daten."""
        # Report als JSON speichern
        report_path = self.outdir / "enhanced_analysis_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Erweiterte Analyse-Daten gespeichert: {report_path}")
        
        # Signal-Daten als CSV speichern
        signal_df = self._create_signal_dataframe()
        if not signal_df.empty:
            signal_path = self.outdir / "signal_data.csv"
            signal_df.to_csv(signal_path, index=False)
            logger.info(f"Signal-Daten gespeichert: {signal_path}")
        
        # Traffic-Daten als CSV speichern
        traffic_df = self._create_traffic_dataframe()
        if not traffic_df.empty:
            traffic_path = self.outdir / "traffic_data.csv"
            traffic_df.to_csv(traffic_path, index=False)
            logger.info(f"Traffic-Daten gespeichert: {traffic_path}")
    
    def _create_signal_dataframe(self):
        """Erstellt DataFrame aus Signal-Daten."""
        import pandas as pd
        
        data = []
        for mac, signal_list in self.signal_data.items():
            for signal in signal_list:
                data.append({
                    'mac_address': mac,
                    'timestamp': signal['timestamp'],
                    'rssi': signal['rssi'],
                    'noise': signal['noise'],
                    'channel': signal['channel']
                })
        
        return pd.DataFrame(data)
    
    def _create_traffic_dataframe(self):
        """Erstellt DataFrame aus Traffic-Daten."""
        import pandas as pd
        
        data = []
        for mac, traffic_list in self.traffic_data.items():
            for traffic in traffic_list:
                data.append({
                    'mac_address': mac,
                    'timestamp': traffic['timestamp'],
                    'size': traffic['size'],
                    'direction': traffic['direction'],
                    'channel': traffic['channel']
                })
        
        return pd.DataFrame(data)
    
    def stop(self):
        """Stoppt den Sniffer."""
        self.stop_event.set()
        if self.hopper:
            self.hopper.stop()
        logger.info("Enhanced WiFi Sniffer gestoppt.")


def enhanced_packet_parser_worker(
    pcap_filename_queue: mp.Queue, 
    db_queue: mp.Queue, 
    live_queue: Optional[mp.Queue],
    dpi_queue: Optional[mp.Queue] = None
):
    """Erweiterte Packet Parser Worker mit DPI."""
    dpi = DeepPacketInspector()
    
    while True:
        try:
            filename = pcap_filename_queue.get()
            if filename is None:
                break
                
            with PcapReader(filename) as pcap_reader:
                for pkt in pcap_reader:
                    # Standard WiFi Event
                    if ev := packet_to_event(pkt):
                        db_queue.put(ev)
                        if live_queue:
                            try:
                                live_queue.put_nowait(ev)
                            except queue.Full:
                                pass
                    
                    # DPI-Analyse
                    if dpi_queue:
                        dpi_analysis = dpi.analyze_packet(pkt)
                        if dpi_analysis:
                            try:
                                dpi_queue.put_nowait(dpi_analysis)
                            except queue.Full:
                                pass
                                
            os.remove(filename)
        except Exception as e:
            logger.debug(f"Fehler im erweiterten Parser-Worker: {e}")