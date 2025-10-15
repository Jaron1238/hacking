#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beispiel-Skript für erweiterte WLAN-Analyse mit DPI, Metriken und Visualisierung.
"""

import sys
import os
from pathlib import Path

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from wlan_tool.analysis.enhanced_analysis import EnhancedAnalysisEngine
from wlan_tool.capture.enhanced_sniffer import EnhancedWiFiSniffer
from rich.console import Console

console = Console()

def main():
    """Hauptfunktion für erweiterte Analyse."""
    console.print("[bold blue]WLAN Enhanced Analysis Example[/bold blue]")
    console.print("=" * 50)
    
    # 1. Erweiterte Capture (falls gewünscht)
    if len(sys.argv) > 1 and sys.argv[1] == "--capture":
        console.print("[yellow]Starte erweiterte Capture...[/yellow]")
        
        sniffer = EnhancedWiFiSniffer(
            iface="wlan0mon",
            channels=[1, 6, 11, 36, 40, 44, 48],
            duration=300,  # 5 Minuten
            outdir="enhanced_capture_demo"
        )
        
        try:
            sniffer.start_capture()
            report = sniffer.generate_enhanced_report()
            sniffer.save_enhanced_data(report)
            console.print("[green]Capture abgeschlossen![/green]")
        except KeyboardInterrupt:
            console.print("[yellow]Capture unterbrochen.[/yellow]")
        finally:
            sniffer.stop()
    
    # 2. Erweiterte Analyse mit Beispieldaten
    console.print("[cyan]Führe erweiterte Analyse mit Beispieldaten durch...[/cyan]")
    
    # Beispieldaten erstellen
    sample_events = create_sample_wifi_events()
    
    # Enhanced Analysis Engine initialisieren
    engine = EnhancedAnalysisEngine(output_dir="enhanced_analysis_demo")
    
    # Analyse durchführen
    analysis_results = engine.analyze_wifi_data(sample_events)
    
    # Ergebnisse speichern
    engine.save_analysis_results()
    
    # Zusammenfassung anzeigen
    summary = engine.get_analysis_summary()
    
    console.print("\n[bold green]Analyse-Ergebnisse:[/bold green]")
    console.print(f"  📊 Geräte: {summary['total_devices']}")
    console.print(f"  📈 Visualisierungen: {', '.join(summary['visualizations_created'])}")
    console.print(f"  📁 Ausgabe: {engine.output_dir}")
    
    # Netzwerk-Insights anzeigen
    insights = summary.get('network_insights', {})
    if insights:
        console.print("\n[bold blue]Netzwerk-Insights:[/bold blue]")
        console.print(f"  🔍 Gerätetypen: {insights.get('device_types', {})}")
        console.print(f"  📶 Signal-Qualität: {insights.get('signal_quality_summary', {})}")
        console.print(f"  🌐 Traffic: {insights.get('traffic_summary', {})}")
        console.print(f"  🔒 Sicherheit: {insights.get('security_insights', {})}")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            console.print("\n[bold yellow]Empfehlungen:[/bold yellow]")
            for rec in recommendations:
                console.print(f"  • {rec}")
    
    console.print("\n[bold green]Erweiterte Analyse abgeschlossen![/bold green]")
    console.print(f"📁 Ergebnisse finden Sie in: {engine.output_dir}")


def create_sample_wifi_events():
    """Erstellt Beispieldaten für die Analyse."""
    import time
    import random
    
    events = []
    base_time = time.time()
    
    # Verschiedene Geräte simulieren
    devices = [
        {"mac": "00:11:22:33:44:55", "type": "AP", "rssi_range": (-40, -60)},
        {"mac": "aa:bb:cc:dd:ee:ff", "type": "Client", "rssi_range": (-50, -80)},
        {"mac": "11:22:33:44:55:66", "type": "Client", "rssi_range": (-60, -90)},
        {"mac": "ff:ee:dd:cc:bb:aa", "type": "Router", "rssi_range": (-30, -50)},
    ]
    
    for i in range(1000):  # 1000 Events
        device = random.choice(devices)
        event_time = base_time + i * 0.1  # 0.1 Sekunden Abstand
        
        # Event-Typ zufällig wählen
        event_type = random.choice(["beacon", "probe_req", "data", "probe_resp"])
        
        event = {
            "ts": event_time,
            "type": event_type,
            "client": device["mac"] if event_type in ["probe_req", "data"] else None,
            "bssid": device["mac"] if event_type in ["beacon", "probe_resp"] else None,
            "rssi": random.randint(*device["rssi_range"]),
            "noise": random.randint(-95, -85),
            "channel": random.choice([1, 6, 11, 36, 40, 44, 48]),
            "packet_size": random.randint(64, 1500),
            "from_ds": random.choice([True, False]),
        }
        
        # Zusätzliche Daten für DPI-Simulation
        if event_type == "data" and random.random() < 0.1:
            event["dns_query"] = random.choice([
                "google.com", "facebook.com", "youtube.com", 
                "amazon.com", "github.com", "stackoverflow.com"
            ])
        
        events.append(event)
    
    return events


if __name__ == "__main__":
    main()