# data/controllers.py

import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
from fritzconnection import FritzConnection
from fritzconnection.lib.fritzhosts import FritzHosts
from rich.console import Console
from rich.prompt import Prompt

from . import config, led_controller, utils
from .analysis import logic as analysis
from .analysis import training as ml_training
from .capture import sniffer as capture
from .presentation import cli, live_tui
from .storage import database, state
from .storage.state import WifiAnalysisState

logger = logging.getLogger(__name__)


class CaptureController:
    """Steuert den gesamten Datenerfassungs-Prozess."""

    def __init__(self, args, config_data, console):
        self.args = args
        self.config_data = config_data
        self.console = console

    def run_capture(self):
        """Startet den Scan-Prozess."""
        physical_iface = self._select_interface()
        if not physical_iface:
            return

        monitor_iface = self._setup_monitor_mode(physical_iface)
        if not monitor_iface:
            return

        duration = self.args.duration or self.config_data.get("capture", {}).get(
            "duration", 120
        )
        outdir = Path(self.args.outdir) if self.args.outdir else Path.cwd()
        pcap_out = self.args.pcap or (
            outdir / "capture.pcap" if self.args.project else None
        )
        db_path = self.args.db or str(outdir / "events.db")

        # Initialisiere Datenbank falls nötig
        self.console.print(f"[cyan]Prüfe Datenbank-Schema: {db_path}[/cyan]")
        database.migrate_db(db_path)

        if self.args.live:
            self._run_live_capture(monitor_iface, duration, db_path, pcap_out)
            return

        use_adaptive_scan = self.args.adaptive_scan or self.config_data.get(
            "scanning", {}
        ).get("adaptive_scan_enabled", False)
        discovery_time = self.args.discovery_time or self.config_data.get(
            "scanning", {}
        ).get("adaptive_scan_discovery_s", 60)

        if use_adaptive_scan and duration > discovery_time:
            self._run_adaptive_scan(
                monitor_iface, duration, discovery_time, db_path, pcap_out
            )
        else:
            if use_adaptive_scan:
                logger.warning(
                    "Gesamtdauer ist kürzer als Discovery-Zeit. Führe normalen Scan durch."
                )
            capture.sniff_with_writer(
                monitor_iface, duration, db_path, pcap_out=pcap_out
            )

    def _run_live_capture(
        self, monitor_iface: str, duration: int, db_path: str, pcap_out: Optional[str]
    ):
        """Führt die Erfassung mit Live-TUI durch."""
        import multiprocessing as mp

        # Erstelle eine Queue für Live-Daten
        live_queue = mp.Queue(maxsize=1000)

        # Starte die Live-TUI in einem separaten Prozess
        tui_process = mp.Process(
            target=live_tui.start_live_tui, args=(live_queue, duration)
        )
        tui_process.start()

        try:
            # Starte die Erfassung mit Live-Queue
            self.console.print(
                f"[cyan]Starte Live-Erfassung auf {monitor_iface} für {duration}s...[/cyan]"
            )
            capture.sniff_with_writer(
                monitor_iface,
                duration,
                db_path,
                pcap_out=pcap_out,
                live_queue=live_queue,
            )
        finally:
            # Beende die TUI
            tui_process.terminate()
            tui_process.join(timeout=5)
            if tui_process.is_alive():
                tui_process.kill()

    def _select_interface(self) -> Optional[str]:
        """Wählt das zu verwendende physische WLAN-Interface aus."""
        iface = self.args.iface
        if not iface:
            self.console.print(
                "[bold cyan]Kein Interface angegeben. Suche nach geeigneten WLAN-Interfaces...[/bold cyan]"
            )
            suitable_ifaces = pre_run_checks.find_wlan_interfaces()
            if not suitable_ifaces:
                self.console.print(
                    "[bold red]Fehler: Keine WLAN-Interfaces gefunden.[/bold red]"
                )
                return None
            if len(suitable_ifaces) == 1:
                iface = suitable_ifaces[0]
                self.console.print(
                    f"[green]Ein Interface gefunden: {iface}. Wird automatisch verwendet.[/green]"
                )
            else:
                iface = Prompt.ask(
                    "Bitte wählen Sie ein Interface", choices=suitable_ifaces
                )
        return iface

    def _setup_monitor_mode(self, physical_iface: str) -> Optional[str]:
        """
        Erstellt ein dediziertes Monitor-Interface (mon0) aus einem physischen Interface (z.B. wlan0)
        und beendet störende Prozesse. Dies ist die empfohlene Methode für Nexmon.
        """
        monitor_iface = "mon0"
        logger.info(
            f"Erstelle dediziertes Monitor-Interface '{monitor_iface}' aus '{physical_iface}'..."
        )

        try:
            # --- Schritt 1: Störende Prozesse beenden, um den Chip freizugeben ---
            kill_commands = [
                "sudo systemctl stop wpa_supplicant",
                "sudo airmon-ng check kill",
            ]
            self.console.print(
                "[yellow]Beende potenziell störende Netzwerkdienste...[/yellow]"
            )
            for cmd in kill_commands:
                subprocess.run(cmd, shell=True, capture_output=True, text=True)

            # --- Schritt 2: Physisches Interface deaktivieren ---
            # Dies ist der entscheidende Schritt, um den "Operation not supported"-Fehler zu vermeiden.
            self.console.print(
                f"[cyan]Deaktiviere physisches Interface '{physical_iface}'...[/cyan]"
            )
            subprocess.run(
                f"sudo ip link set {physical_iface} down",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

            # --- Schritt 3: Phy-Namen ermitteln ---
            # Dies sollte jetzt auf dem 'down'-Interface funktionieren.
            phy_find_cmd = (
                f"iw dev {physical_iface} info | gawk '/wiphy/ {{printf \"phy\" $2}}'"
            )
            result = subprocess.run(
                phy_find_cmd, shell=True, check=True, capture_output=True, text=True
            )
            phy_name = result.stdout.strip()
            if not phy_name:
                logger.error(
                    f"Konnte den 'phy'-Namen für das Interface '{physical_iface}' nicht ermitteln."
                )
                return None
            logger.info(f"Physisches Gerät gefunden: {phy_name}")

            # --- Schritt 4: Monitor-Interface erstellen und aktivieren ---
            self.console.print(
                f"[cyan]Konfiguriere Interface '{monitor_iface}'...[/cyan]"
            )
            setup_commands = [
                f"sudo iw dev {monitor_iface} del",  # Fehler wird ignoriert
                f"sudo iw phy {phy_name} interface add {monitor_iface} type monitor",
                f"sudo ip link set {monitor_iface} up",
            ]
            for cmd in setup_commands:
                subprocess.run(cmd, shell=True, capture_output=True, text=True)

            logger.info(
                f"Monitor-Interface '{monitor_iface}' erfolgreich erstellt und aktiviert."
            )
            return monitor_iface

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Ein Befehl zur Erstellung des Monitor-Interfaces ist fehlgeschlagen."
            )
            logger.error(f"Befehl: '{e.cmd}'")
            logger.error(f"Fehlermeldung: {e.stderr.strip()}")
            # Versuche, den ursprünglichen Zustand wiederherzustellen
            logger.info("Versuche, Netzwerkdienste neu zu starten...")
            subprocess.run(
                "sudo systemctl start wpa_supplicant",
                shell=True,
                capture_output=True,
                text=True,
            )
            return None
        except Exception as e:
            logger.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            return None

    def _run_adaptive_scan(
        self, monitor_iface, duration, discovery_time, db_path, pcap_out
    ):
        """Führt den zweistufigen adaptiven Scan durch."""
        self.console.print(
            f"[bold cyan]Starte adaptive Erfassung: Phase 1 (Discovery) für {discovery_time}s...[/bold cyan]"
        )
        discovery_start_ts = time.time()
        capture.sniff_with_writer(
            monitor_iface, discovery_time, db_path, pcap_out=pcap_out
        )

        self.console.print(
            "[cyan]Analysiere Discovery-Daten, um Zielkanäle zu bestimmen...[/cyan]"
        )
        with database.db_conn_ctx(db_path) as conn:
            events_for_adaptive = list(
                database.fetch_events(conn, start_ts=discovery_start_ts)
            )

        temp_state = WifiAnalysisState()
        temp_state.build_from_events(events_for_adaptive)

        top_channels = []
        priority_channels = config.CHANNELS_TO_HOP

        if not temp_state.aps:
            logger.warning(
                "Keine APs in der Discovery-Phase gefunden. Setze mit normalem Scan fort."
            )
        else:
            channel_scores = {}
            for ap in temp_state.aps.values():
                if ap.channel:
                    score = ap.count + (100 + ap.rssi_w.mean) / 10
                    channel_scores[ap.channel] = (
                        channel_scores.get(ap.channel, 0) + score
                    )

            if channel_scores:
                sorted_channels = sorted(
                    channel_scores, key=channel_scores.get, reverse=True
                )
                top_channels = sorted_channels[
                    : self.config_data.get("scanning", {}).get(
                        "adaptive_scan_top_n_channels", 5
                    )
                ]

                priority_channels = []
                for ch in top_channels:
                    priority_channels.extend(
                        [ch]
                        * self.config_data.get("scanning", {}).get(
                            "adaptive_scan_priority_weight", 3
                        )
                    )

                other_channels = [
                    ch for ch in sorted_channels if ch not in top_channels
                ]
                priority_channels.extend(other_channels)

        self.console.print(
            f"[green]Priorisierte Kanäle für die gezielte Phase: {top_channels or 'Keine gefunden'}[/green]"
        )

        remaining_duration = duration - discovery_time
        self.console.print(
            f"[bold cyan]Starte Phase 2 (Targeted Scan) für {remaining_duration}s...[/bold cyan]"
        )
        capture.sniff_with_writer(
            monitor_iface,
            remaining_duration,
            db_path,
            pcap_out=pcap_out,
            channels_override=priority_channels,
        )


class AnalysisController:
    """Steuert den gesamten Analyse-Prozess."""

    def __init__(self, args, config_data, console, plugins, state, new_events):
        self.args = args
        self.config_data = config_data
        self.console = console
        self.plugins = plugins
        # KORREKTUR: Übernehme die bereits geladenen Objekte
        self.state_obj = state

        self.new_events = new_events

        self.outdir = Path(self.args.outdir) if self.args.outdir else Path.cwd()
        self.db_path = self.args.db or str(self.outdir / "db" / "events.db")
        self.label_db_path = self.args.label_db or str(self.outdir / "db" / "labels.db")
        self.state_path = self.outdir / "wifi.state"

    @property
    def state(self):
        """Getter für state_obj für Kompatibilität mit Tests."""
        return self.state_obj

    def run_inference(self):
        """Führt Inferenz aus."""
        logging.info("Führe Inferenz aus...")
        # Placeholder implementation
        pass

    def run_client_clustering(self):
        """Führt Client-Clustering durch."""
        logging.info("Führe Client-Clustering durch...")
        # Placeholder implementation
        pass

    def run_ap_clustering(self):
        """Führt AP-Clustering durch."""
        logging.info("Führe AP-Clustering durch...")
        # Placeholder implementation
        pass

    def run_labeling_ui(self):
        """Startet Labeling-UI."""
        logging.info("Starte Labeling-UI...")
        # Placeholder implementation
        pass

    def run_client_labeling_ui(self):
        """Startet Client-Labeling-UI."""
        logging.info("Starte Client-Labeling-UI...")
        # Placeholder implementation
        pass

    def run_mac_correlation(self):
        """Führt MAC-Korrelation durch."""
        logging.info("Führe MAC-Korrelation durch...")
        # Placeholder implementation
        pass

    def run_plugins(self):
        """Führt Plugins aus."""
        logging.info("Führe Plugins aus...")
        # Placeholder implementation
        pass

    def run_analysis(self):
        """Startet den Analyse-Prozess."""
        logging.info("Starte Analyse-Prozess...")

        # Initialisiere Datenbanken falls nötig
        self.console.print(f"[cyan]Prüfe Datenbank-Schema: {self.db_path}[/cyan]")
        database.migrate_db(self.db_path)
        self.console.print(
            f"[cyan]Prüfe Label-Datenbank-Schema: {self.label_db_path}[/cyan]"
        )
        database.migrate_db(self.label_db_path)

        if self.args.profile_from_router:
            self._run_profiling()  # Benötigt state, den es jetzt als self.state hat

        if self.args.show_probed_ssids:
            cli.print_probed_ssids(self.state_obj, self.console)

        clustered_client_df, client_feature_df = self._run_client_clustering_if_needed(
            self.state_obj
        )
        clustered_ap_df = self._run_ap_clustering_if_needed(self.state_obj)

        plugins_need_client_clusters = (
            self.args.run_plugins is not None
            and "umap_plot" in (self.args.run_plugins or self.plugins.keys())
        )
        if self.args.cluster_clients is not None or plugins_need_client_clusters:
            clustered_client_df, client_feature_df = cli.print_client_cluster_results(
                self.args, self.state_obj, self.console
            )
        else:
            clustered_client_df, client_feature_df = None, None

        clustered_ap_df = None
        if (
            self.args.cluster_aps is not None
            or self.args.export_graph
            or self.args.export_csv
        ):
            clustered_ap_df = cli.print_ap_cluster_results(
                self.args, self.state_obj, self.console
            )

        if self.args.run_plugins is not None:
            self._run_plugins(
                self.state_obj, self.new_events, clustered_client_df, client_feature_df
            )

        inference_results = None
        if self.args.infer or self.args.html_report:
            model = (
                joblib.load(self.args.model)
                if self.args.model and Path(self.args.model).exists()
                else None
            )
            inference_results = analysis.score_pairs_with_recency_and_matching(
                self.state_obj, model
            )

        if self.args.infer and inference_results:
            self._handle_inference_output(inference_results)

        if self.args.export_graph or self.args.export_csv:
            self._handle_graph_export(
                self.state_obj, clustered_ap_df, clustered_client_df
            )

        if self.args.train_behavior_model:
            self.console.print(f"[cyan]Starte Training für Verhaltensmodell...[/cyan]")
            ml_training.train_behavioral_model(
                self.db_path,
                self.label_db_path,
                self.outdir / self.args.train_behavior_model,
            )

        if self.args.html_report:
            self._generate_html_report(inference_results)

        self._save_state(self.state_obj, self.state_path)

    ################################################################################
    def _load_state_and_events(self, state_path, db_path):
        """Lädt den Zustand und die Events von den korrekten Pfaden."""
        state = None
        last_timestamp = 0
        if state_path.exists():
            try:
                self.console.print(
                    f"[cyan]Lade gespeicherten Zustand von {state_path}...[/cyan]"
                )
                state = joblib.load(state_path)
                if state.clients:
                    last_timestamp = max(c.last_seen for c in state.clients.values())
            except Exception as e:
                self.console.print(
                    f"[red]Fehler beim Laden des Zustands: {e}. Baue Zustand neu auf.[/red]"
                )
                state = None

        if state is None:
            state = WifiAnalysisState()

        self.console.print(
            f"[cyan]Lade Events aus der Datenbank (von {db_path})...[/cyan]"
        )
        with database.db_conn_ctx(db_path) as conn:
            new_events = list(
                database.fetch_events(
                    conn, start_ts=last_timestamp if last_timestamp > 0 else None
                )
            )

        if not new_events and not state.clients:
            logger.warning(
                "Keine Events in der Datenbank und kein gespeicherter Zustand. Analyse wird übersprungen."
            )
            return None, None

        if new_events:
            state.build_from_events(new_events, detailed_ies=self.args.detailed_ies)

        self.console.print(
            f"[green]Zustand aufgebaut/aktualisiert: {len(state.aps)} APs, {len(state.clients)} Clients, {len(state.ssid_map)} SSIDs.[/green]"
        )
        return state, new_events

    def _save_state(self, state_obj, state_path):
        """Speichert den Zustand am korrekten Pfad."""
        try:
            self.console.print(
                f"[cyan]Speichere Analyse-Zustand nach {state_path}...[/cyan]"
            )
            joblib.dump(state_obj, state_path)
        except Exception as e:
            self.console.print(f"[red]Fehler beim Speichern des Zustands: {e}[/red]")

    def _run_profiling(self):
        self.console.print(
            "\n[bold cyan]Starte automatisches Geräte-Profiling...[/bold cyan]"
        )
        device_map = {}

        # --- NEU: Versuch 1: TR-064 (FRITZ!Box) ---
        self.console.print(
            "[cyan]Versuch 1: Identifizierung über FRITZ!Box TR-064...[/cyan]"
        )
        # Passwort aus Argumenten oder interaktiv abfragen
        fritz_password = self.args.fritzbox_password
        if FritzHosts and not fritz_password:
            if (
                self.console.input(
                    "Haben Sie ein Passwort für Ihre FRITZ!Box gesetzt? (y/n): "
                ).lower()
                == "y"
            ):
                fritz_password = self.console.input(
                    "Bitte FRITZ!Box-Passwort eingeben: ", password=True
                )

        device_map = (
            device_profiler.get_devices_from_fritzbox_tr064(password=fritz_password)
            or {}
        )
        if device_map:
            self.console.print(
                f"[bold green]Erfolgreich {len(device_map)} Geräte via TR-064 gefunden![/bold green]"
            )
        else:
            self.console.print(
                "[yellow]TR-064-Versuch fehlgeschlagen oder keine Geräte gefunden.[/yellow]"
            )

        # --- Versuch 2: Standard-UPnP (Fallback) ---
        if not device_map:
            self.console.print(
                "\n[cyan]Versuch 2: Fallback auf aktiven ARP- und Nmap-Scan...[/cyan]"
            )
            profile_iface = self.args.profile_iface or self.config_data.get(
                "profiling", {}
            ).get("interface")
            profile_net = self.args.profile_net or self.config_data.get(
                "profiling", {}
            ).get("network_cidr")
            if not profile_iface or not profile_net:
                self.console.print(
                    "[bold red]Fehler: Für den aktiven Scan müssen --profile-iface und --profile-net angegeben werden (oder in config.yaml gesetzt sein).[/bold red]"
                )
            else:
                all_macs_in_scan = list(self.state_obj.clients.keys())
                clients_with_ips = device_profiler.get_ips_for_macs_arp(profile_iface)
                if clients_with_ips:
                    nmap_results = device_profiler.profile_devices_via_portscan(
                        clients_with_ips
                    )
                    device_map.update(nmap_results)
                else:
                    self.console.print(
                        "[yellow]Aktiver ARP-Scan konnte keine IPs ermitteln.[/yellow]"
                    )

        if not device_map:
            self.console.print(
                "[red]Automatisches Profiling konnte keine Geräte identifizieren.[/red]"
            )
        else:
            rules = device_profiler.load_device_rules()
            auto_classified = {
                mac: info
                for mac, info in device_map.items()
                if (
                    info.update(
                        {
                            "device_type": device_profiler.classify_hostname(
                                info.get("hostname"), rules
                            )
                            or info.get("device_type")
                        }
                    ),
                    info.get("device_type"),
                )
            }
            unknown = {
                mac: info
                for mac, info in device_map.items()
                if mac not in auto_classified
            }
            self.console.print(
                f"[green]{len(auto_classified)} Geräte automatisch via Regeln/Scan klassifiziert.[/green]"
            )
            manual_classified = device_profiler.interactive_classify_unknowns(
                unknown, self.console
            )
            final_map = {**auto_classified, **manual_classified}
            if final_map:
                database.save_labels_from_map(
                    final_map, self.label_db_path, self.console
                )

    def _run_client_clustering_if_needed(self, state_obj):
        if self.args.cluster_clients is not None:
            return cli._handle_client_clustering(self.args, state_obj, self.console)

        plugins_need_clusters = self.args.run_plugins is not None and "umap_plot" in (
            self.args.run_plugins or self.plugins.keys()
        )
        if plugins_need_clusters:
            return analysis.cluster_clients(
                state_obj,
                n_clusters=0,
                algo=self.args.cluster_algo,
                correlate_macs=(not self.args.no_mac_correlation),
            )
        return None, None

    def _run_ap_clustering_if_needed(self, state_obj):
        if self.args.cluster_aps is not None:
            return cli._handle_ap_clustering(self.args, state_obj, self.console)
        return None

    def _run_plugins(
        self, state_obj, new_events, clustered_client_df, client_feature_df
    ):
        plugin_context = {
            "state": state_obj,
            "events": new_events,
            "clustered_client_df": clustered_client_df,
            "client_feature_df": client_feature_df,
            "outdir": self.outdir,
            "console": self.console,
        }
        plugins_to_run = (
            self.args.run_plugins if self.args.run_plugins else self.plugins.keys()
        )
        for name in plugins_to_run:
            if name in self.plugins:
                try:
                    import inspect

                    sig = inspect.signature(self.plugins[name].run)
                    valid_args = {
                        k: v for k, v in plugin_context.items() if k in sig.parameters
                    }
                    self.plugins[name].run(**valid_args)
                except Exception as e:
                    self.console.print(
                        f"[bold red]Fehler bei der Ausführung des Plugins '{name}': {e}[/bold red]"
                    )
                    logger.error(f"Plugin execution error for '{name}'", exc_info=True)
            else:
                self.console.print(
                    f"[yellow]Warnung: Plugin '{name}' nicht gefunden.[/yellow]"
                )

    def _handle_inference_output(self, inference_results):
        """Behandelt die Ausgabe der Inferenz-Ergebnisse."""
        if not inference_results:
            self.console.print("[yellow]Keine Inferenz-Ergebnisse verfügbar.[/yellow]")
            return

        self.console.print(
            f"\n[bold cyan]SSID-BSSID Inferenz-Ergebnisse ({len(inference_results)} Paare):[/bold cyan]"
        )
        from rich.table import Table

        table = Table()
        table.add_column("SSID", style="cyan")
        table.add_column("BSSID", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Label", style="green")

        for result in sorted(inference_results, key=lambda x: x.score, reverse=True)[
            :20
        ]:
            table.add_row(
                result.ssid, result.bssid, f"{result.score:.3f}", result.label
            )
        self.console.print(table)

        if self.args.json:
            import json

            with open(self.args.json, "w") as f:
                json.dump(
                    [
                        {
                            "ssid": r.ssid,
                            "bssid": r.bssid,
                            "score": r.score,
                            "label": r.label,
                        }
                        for r in inference_results
                    ],
                    f,
                    indent=2,
                )
            self.console.print(
                f"[green]Ergebnisse in JSON gespeichert: {self.args.json}[/green]"
            )

    def _handle_graph_export(self, state_obj, clustered_ap_df, clustered_client_df):
        """Exportiert Netzwerk-Graphen für Gephi."""
        if not self.args.export_graph and not self.args.export_csv:
            return

        self.console.print("[cyan]Exportiere Netzwerk-Graph...[/cyan]")
        try:
            from . import analysis

            if self.args.export_graph:
                analysis.export_graph_for_gephi(
                    state_obj,
                    self.args.export_graph,
                    include_clients=self.args.graph_include_clients,
                    min_activity=self.args.graph_min_activity,
                    min_duration=self.args.graph_min_duration,
                    clustered_df=clustered_ap_df,
                )
                self.console.print(
                    f"[green]Graph exportiert nach: {self.args.export_graph}[/green]"
                )

            if self.args.export_csv:
                csv_base = self.outdir / "graph_export"
                analysis.export_graph_csv(
                    state_obj,
                    csv_base,
                    include_clients=self.args.graph_include_clients,
                    clustered_df=clustered_ap_df,
                )
                self.console.print(
                    f"[green]CSV-Dateien exportiert nach: {csv_base}_nodes.csv und {csv_base}_edges.csv[/green]"
                )
        except Exception as e:
            self.console.print(f"[red]Fehler beim Graph-Export: {e}[/red]")
            logger.error("Graph export error", exc_info=True)

    def _generate_html_report(self, inference_results):
        """Generiert einen HTML-Report."""
        try:
            from . import reporting

            self.console.print(f"[cyan]Generiere HTML-Report...[/cyan]")
            reporting.generate_html_report(
                self.state_obj,
                self.args.html_report,
                inference_results=inference_results,
            )
            self.console.print(
                f"[green]HTML-Report erstellt: {self.args.html_report}[/green]"
            )
        except Exception as e:
            self.console.print(
                f"[red]Fehler beim Erstellen des HTML-Reports: {e}[/red]"
            )
            logger.error("HTML report generation error", exc_info=True)
