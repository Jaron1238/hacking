#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haupt-CLI-Skript für das WLAN-Analyse-Tool mit Plugin-Architektur.
Delegiert die Hauptlogik an Controller-Klassen.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import importlib.util
# Importiere nur noch die notwendigen Teile für den Start
from wlan_tool import utils, config, pre_run_checks
from wlan_tool.controllers import CaptureController, AnalysisController
from wlan_tool.analysis.enhanced_analysis import EnhancedAnalysisEngine
from wlan_tool.capture.enhanced_sniffer import EnhancedWiFiSniffer
from wlan_tool.storage import database, state
from wlan_tool.storage.state import WifiAnalysisState
from rich.console import Console
from collections import Counter
from typing import Optional, Dict

import joblib

logger = logging.getLogger(__name__)

def setup_logging():
    """Konfiguriert das Logging für die Anwendung."""
    LOG_LEVEL = logging.INFO
    app_logger = logging.getLogger('wifi_analysis')
    
    # Verhindere doppelte Handler
    if app_logger.hasHandlers():
        return app_logger
        
    app_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('wifi_analysis.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    return app_logger



def parse_args():
    """Parst alle Kommandozeilenargumente."""
    p = argparse.ArgumentParser(description="WLAN SSID-BSSID Correlation and Analysis Tool")
    p.add_argument("--project", help="Name für das Analyse-Projekt. Alle Dateien werden in einem Verzeichnis dieses Namens gespeichert/geladen.")
    p.add_argument("--profile", help="Name des zu verwendenden Konfigurationsprofils (z.B. fast_scan)")
    p.add_argument("--capture_mode", action="store_true", help= "Wenn aktiviert kann gecaptured werden")
    g_out = p.add_argument_group("Output Options")
    g_out.add_argument("--outdir", help="Basis-Verzeichnis für alle Ausgabedateien (wird von --project überschrieben).")
    g_out.add_argument("--status-led", action="store_true", help="Lässt die rote PWR-LED des Raspberry Pi blinken, während das Skript läuft.")
    
    g_cap = p.add_argument_group("Capture Options")
    g_cap.add_argument("--iface", help="Das physische WLAN-Interface (z.B. wlan0). Überschreibt den Wert aus der config.yaml.")
    g_cap.add_argument("--duration", type=int, help="Dauer der Erfassung in Sekunden")
    g_cap.add_argument("--pcap", help="Erfasste Pakete in einer PCAP-Datei speichern")
    g_cap.add_argument("--adaptive-scan", action="store_true", help="Aktiviert den zweistufigen Scan (Discovery, dann gezielt)")
    g_cap.add_argument("--discovery-time", type=int, help="Dauer der Discovery-Phase für den adaptiven Scan in Sekunden")

    g_db = p.add_argument_group("Database & File Options")
    g_db.add_argument("--db", help="Pfad zur Event-Datenbank (wird von --project überschrieben).")
    g_db.add_argument("--label-db", help="Pfad zur Label-Datenbank (wird von --project überschrieben).")
    g_db.add_argument("--update-oui", action="store_true", help="Erzwingt ein Update der OUI-Hersteller-Datei vor der Ausführung.")
    g_db.add_argument("--json", help="JSON-Ausgabe der Bewertung in diese Datei speichern")
    g_db.add_argument("--html-report", help="Einen zusammenfassenden HTML-Bericht in diese Datei exportieren.")

    g_an = p.add_argument_group("Analysis Options")
    g_an.add_argument("--infer", action="store_true", help="SSID-BSSID-Inferenz auf der Datenbank ausführen")
    g_an.add_argument("--live", action="store_true", help="Analyse im Live-Modus ausführen, Aktualisierung alle 10 Sekunden")
    g_an.add_argument("--detailed-ies", action="store_true", help="Tiefes Parsen von 802.11n/ac/ax/k/v/r und Sicherheitseinstellungen aktivieren")
    g_an.add_argument("--filter-ssid", help="Ergebnisse nur für diese SSID anzeigen (Regex unterstützt)")
    g_an.add_argument("--min-rssi", type=int, help="Nur APs mit einem durchschnittlichen RSSI über diesem Wert berücksichtigen")
    g_an.add_argument("--plot", action="store_true", help="Terminal-Plots für RSSI und Paket-Zeitleiste anzeigen")
    g_an.add_argument("--plot-client-rssi", metavar="MAC", help="Zeigt einen Zeitverlaufs-Plot der Signalstärke (RSSI) für einen bestimmten Client an.")
    g_an.add_argument("--cluster-aps", type=int, nargs='?', const=0, default=None, help="Access Points in 'Flotten' gruppieren. Ohne Zahl wird die optimale Anzahl gesucht (Standard: 0=auto).")
    g_an.add_argument("--export-graph", help="Exportiert einen AP-Graphen für die Analyse in Gephi in die angegebene GEXF-Datei. Benötigt --cluster-aps.")
    g_an.add_argument("--export-csv", action="store_true", help="Exportiert den Graphen als Nodes/Edges CSV-Dateien für einen robusten Gephi-Import.")
    g_an.add_argument("--graph-include-clients", action="store_true", help="Nimmt auch die Clients als Knoten in den Graphen-Export auf (erzeugt einen bipartiten Graphen).")
    g_an.add_argument("--graph-min-activity", type=int, default=0, help="Exportiert nur Geräte (Knoten) mit mindestens N Paketen (Standard: 0=alle).")
    g_an.add_argument("--graph-min-duration", type=int, default=0, help="Exportiert nur Geräte (Knoten), die mindestens S Sekunden aktiv waren (Standard: 0=alle).")
    g_an.add_argument("--model", help="Pfad zu einer Joblib-Modelldatei für die Bewertung")
    g_an.add_argument("--classify-clients", metavar="MODEL_PATH", help="Klassifiziert alle Clients anhand ihrer technischen Merkmale mit einem trainierten Modell. Benötigt einen Pfad zur Modelldatei.")
    g_an.add_argument("--cluster-clients", type=int, nargs='?', const=0, default=None, help="Clients in Cluster gruppieren. Ohne Zahl wird die optimale Anzahl gesucht (Standard: 0=auto).")
    g_an.add_argument("--cluster-algo", choices=["kmeans", "dbscan"], default="kmeans", help="Der für das Client-Clustering zu verwendende Algorithmus (Standard: kmeans).")
    g_an.add_argument("--run-plugins", nargs='*', help="Führt spezifische Analyse-Plugins aus (z.B. umap_plot sankey). Ohne Angabe werden alle ausgeführt.")
    g_an.add_argument("--tui", action="store_true", help="Startet die interaktive Terminal-Benutzeroberfläche zur Analyse der Ergebnisse.")
    g_an.add_argument("--no-mac-correlation", action="store_true", help="Deaktiviert die automatische Zusammenfassung von randomisierten MACs vor dem Client-Clustering.")
    g_an.add_argument("--correlate-macs", action="store_true", help="Versucht, randomisierte MAC-Adressen basierend auf ihren Probe-Listen zu gruppieren (nur Anzeige).")
    g_an.add_argument("--train-encoder", help="Trainiert einen Autoencoder für besseres Client-Clustering. Pfad zur Ausgabemodelldatei angeben (z.B. client_encoder.pth).")
    g_an.add_argument("--use-encoder", help="Verwendet einen trainierten Encoder für das Client-Clustering. Pfad zur Modelldatei angeben.")
    g_an.add_argument("--show-probed-ssids", action="store_true", help="Zeigt eine Rangliste der am häufigsten gesuchten SSIDs an.")
    
    # Erweiterte Analyse-Optionen
    g_an.add_argument("--enhanced-analysis", action="store_true", help="Führt erweiterte Analyse mit DPI, Metriken und Visualisierung durch.")
    g_an.add_argument("--deep-packet-inspection", action="store_true", help="Aktiviert Deep Packet Inspection für HTTP, DNS, DHCP Analyse.")
    g_an.add_argument("--advanced-metrics", action="store_true", help="Berechnet erweiterte Metriken (SNR, PER, Traffic Patterns).")
    g_an.add_argument("--3d-visualization", action="store_true", help="Erstellt 3D-Netzwerk-Visualisierung.")
    g_an.add_argument("--time-series-plots", action="store_true", help="Erstellt detaillierte Zeitverlaufs-Diagramme.")
    g_an.add_argument("--custom-report", action="store_true", help="Generiert benutzerdefinierten HTML-Report.")
    
    g_bh = p.add_argument_group("Behavioral Analysis Options")
    g_bh.add_argument("--train-behavior-model", help="Trainiert ein Modell zur Erkennung von Gerätetypen (z.B. iot, smartphone) anhand des Verhaltens. Benötigt einen Pfad zur Ausgabemodelldatei.")
    g_bh.add_argument("--classify-client-behavior", help="Klassifiziert alle Clients im Datensatz mit einem trainierten Verhaltensmodell. Benötigt einen Pfad zur Modelldatei.")
    g_bh.add_argument("--show-dns", action="store_true", help="Analysiert und zeigt die häufigsten DNS-Anfragen pro Client an.")
    g_bh.add_argument("--show-network-map", action="store_true", help="Zeigt eine Zuordnung von MAC- zu IP-Adressen und Hostnamen an.")
    
    g_tr = p.add_argument_group("Labeling & Training Options")
    g_tr.add_argument("--label-ui", action="store_true", help="Startet die interaktive UI für SSID-BSSID-Paare.")
    g_tr.add_argument("--label-clients", action="store_true", help="Startet die interaktive UI zum Labeln von Gerätetypen.")
    g_tr.add_argument("--profile-from-router", action="store_true", help="Versucht, Geräte automatisch über UPnP oder einen aktiven Scan zu identifizieren und zu labeln.")
    g_tr.add_argument("--profile-iface", help="Das Netzwerk-Interface, das für den aktiven ARP- und Port-Scan verwendet werden soll (z.B. eth0 oder wlan0).")
    g_tr.add_argument("--profile-net", help="Der Netzwerkbereich im CIDR-Format, der für den ARP-Scan verwendet werden soll (z.B. 192.168.1.0/24).")
    g_tr.add_argument("--fritzbox-password", help="Das Passwort für die TR-064-Schnittstelle der FRITZ!Box.")
    g_tr.add_argument("--export-training-csv", help="Exportiert bestätigte Labels in eine CSV-Datei")
    g_tr.add_argument("--auto-retrain", action="store_true", help="Trainiert das SSID-BSSID-Modell automatisch neu")
    g_tr.add_argument("--min-confirmed", type=int, default=config.AUTO_RETRAIN_MIN_CONFIRMED_LABELS)
    return p.parse_args()

def load_plugins(console: Console):
    """Sucht und lädt alle Analyse-Plugins aus dem 'plugins'-Verzeichnis."""
    from plugins import load_all_plugins
    
    plugin_dir = Path(__file__).parent / "plugins"
    plugins = load_all_plugins(plugin_dir)
    
    # Fallback: Lade alte Plugin-Struktur für Kompatibilität
    if not plugins:
        console.print("[yellow]Keine neuen Plugins gefunden, versuche alte Struktur...[/yellow]")
        plugins = load_legacy_plugins(console)
    
    return plugins

def load_legacy_plugins(console: Console):
    """Lädt Plugins aus der alten Struktur (analysis_*.py)."""
    plugins = {}
    plugin_dir = Path(__file__).parent / "plugins"
    if not plugin_dir.is_dir():
        return {}
        
    for f in plugin_dir.glob("analysis_*.py"):
        plugin_name = f.stem.replace("analysis_", "")
        try:
            module_spec = importlib.util.spec_from_file_location(f"plugins.{f.stem}", f)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            if hasattr(module, 'run'):
                plugins[plugin_name] = module
                console.print(f"[dim]Legacy Plugin '{plugin_name}' erfolgreich geladen.[/dim]")
        except Exception as e:
            console.print(f"[bold red]Fehler beim Laden des Legacy-Plugins '{plugin_name}': {e}[/bold red]")
            
    return plugins

def main():
    logger = setup_logging()
    args = parse_args()
    console = Console()

    # --- Pre-Flight Checks & Setup ---
    pre_run_checks.check_dependencies()
    if args.update_oui:
        utils.download_oui_file()

    status_led = None
    if args.status_led:
        try:
            from wlan_tool.led_controller import StatusLED
            status_led = StatusLED()
            status_led.start()
        except ImportError:
            logger.warning("led_controller.py nicht gefunden. Status-LED deaktiviert.")
        except Exception as e:
            logger.error(f"Fehler beim Starten der Status-LED: {e}")
            
    try:
        config_data = utils.load_config(args.profile)
        plugins = load_plugins(console)

        # --- Projekt- und Pfad-Setup ---
        if args.project:
            project_path = Path(args.project)
            project_path.mkdir(parents=True, exist_ok=True)
            (project_path / "db").mkdir(exist_ok=True)
            args.outdir = str(project_path)
            console.print(f"[bold green]Projekt '{args.project}' wird verwendet. Alle Dateien in: {project_path.resolve()}[/bold green]")
        cap = args.capture_mode
        logger.info(cap)
        # --- Modus-Auswahl basierend auf den Argumenten ---
        is_capture_mode = True if cap else False
        
        analysis_actions = [
            args.infer, args.auto_retrain, args.classify_clients,
            args.cluster_clients is not None, args.cluster_aps is not None, args.label_clients,
            args.profile_from_router, args.correlate_macs, args.train_behavior_model, 
            args.classify_client_behavior, args.train_encoder, args.html_report, 
            args.show_dns, args.show_network_map, args.run_plugins is not None, 
            args.export_graph, args.export_csv, args.label_ui, args.show_probed_ssids,
            args.tui, args.enhanced_analysis, args.deep_packet_inspection,
            args.advanced_metrics, args.custom_report
        ]
        is_analysis_mode = any(analysis_actions)
        if is_capture_mode:
            # Prüfe ob erweiterte Analyse gewünscht ist
            if args.enhanced_analysis or args.deep_packet_inspection or args.advanced_metrics:
                console.print("[bold green]Starte erweiterte Capture mit DPI und Metriken...[/bold green]")
                enhanced_sniffer = EnhancedWiFiSniffer(
                    iface=args.iface or config_data.get("capture", {}).get("interface", "wlan0mon"),
                    channels=config_data.get("channels", {}).get("to_hop", [1, 6, 11]),
                    duration=args.duration or config_data.get("capture", {}).get("duration", 300),
                    outdir=args.outdir or "enhanced_capture"
                )
                enhanced_sniffer.start_capture()
                
                # Erweiterte Analyse nach Capture
                if args.enhanced_analysis:
                    console.print("[cyan]Führe erweiterte Analyse durch...[/cyan]")
                    report = enhanced_sniffer.generate_enhanced_report()
                    enhanced_sniffer.save_enhanced_data(report)
                    console.print(f"[green]Erweiterte Analyse abgeschlossen. Ergebnisse in: {enhanced_sniffer.outdir}[/green]")
            else:
                capture_controller = CaptureController(args, config_data, console)
                capture_controller.run_capture()

        if is_analysis_mode:
            # KORREKTUR: Lade den Zustand VOR der Initialisierung des Controllers.
            outdir = Path(args.outdir) if args.outdir else Path.cwd()
            db_path = args.db or str(outdir / "db" / "events.db")
            state_file = outdir / "wifi.state"
            state = None
            last_timestamp = 0
            if state_file.exists():
                try:
                    console.print(f"[cyan]Lade gespeicherten Zustand von {state_file}...[/cyan]")
                    state = joblib.load(state_file)
                    if state.clients:
                        last_timestamp = max(c.last_seen for c in state.clients.values())
                except Exception as e:
                    console.print(f"[red]Fehler beim Laden des Zustands: {e}. Baue Zustand neu auf.[/red]")
                    state = None

            if state is None:
                state = WifiAnalysisState()
                
            console.print(f"[cyan]Lade Events aus der Datenbank (von {db_path})...[/cyan]")
            with database.db_conn_ctx(db_path) as conn:
                new_events = list(database.fetch_events(conn, start_ts=last_timestamp if last_timestamp > 0 else None))

            if not new_events and not state.clients:
                logger.warning("Keine Events in der Datenbank und kein gespeicherter Zustand. Analyse wird übersprungen, aber Profiling ist möglich.")
            
            if new_events:
                state.build_from_events(new_events, detailed_ies=args.detailed_ies)
            
            console.print(f"[green]Zustand aufgebaut/aktualisiert: {len(state.aps)} APs, {len(state.clients)} Clients, {len(state.ssid_map)} SSIDs.[/green]")
            
            # Prüfe ob erweiterte Analyse gewünscht ist
            if args.enhanced_analysis or args.deep_packet_inspection or args.advanced_metrics:
                console.print("[bold green]Starte erweiterte Analyse-Engine...[/bold green]")
                enhanced_engine = EnhancedAnalysisEngine(output_dir=str(outdir / "enhanced_analysis"))
                
                # WiFi-Events für erweiterte Analyse vorbereiten
                wifi_events = []
                for event in new_events:
                    wifi_events.append({
                        'ts': event.ts,
                        'type': event.type,
                        'client': getattr(event, 'client', None),
                        'bssid': getattr(event, 'bssid', None),
                        'rssi': getattr(event, 'rssi', None),
                        'noise': getattr(event, 'noise', None),
                        'channel': getattr(event, 'channel', None),
                        'packet_size': getattr(event, 'packet_size', 0),
                        'from_ds': getattr(event, 'from_ds', False),
                        'dns_query': getattr(event, 'dns_query', None)
                    })
                
                # PCAP-Datei für DPI (falls vorhanden)
                pcap_file = outdir / "capture.pcap"
                pcap_path = str(pcap_file) if pcap_file.exists() else None
                
                # Erweiterte Analyse durchführen
                analysis_results = enhanced_engine.analyze_wifi_data(wifi_events, pcap_path)
                enhanced_engine.save_analysis_results()
                
                # Zusammenfassung anzeigen
                summary = enhanced_engine.get_analysis_summary()
                console.print(f"[green]Erweiterte Analyse abgeschlossen:[/green]")
                console.print(f"  - Geräte: {summary['total_devices']}")
                console.print(f"  - Visualisierungen: {', '.join(summary['visualizations_created'])}")
                console.print(f"  - Ergebnisse: {enhanced_engine.output_dir}")
            else:
                # Standard-Analyse
                analysis_controller = AnalysisController(args, config_data, console, plugins, state, new_events)
                analysis_controller.run_analysis()
        
        if not is_capture_mode and not is_analysis_mode:
            console.print("[yellow]Keine Aktion angegeben. Führen Sie '--help' für eine Liste der Optionen aus.[/yellow]")

    finally:
        logger.info("Alles erledigt.")
        if status_led:
            status_led.stop()
if __name__ == "__main__":
    main()