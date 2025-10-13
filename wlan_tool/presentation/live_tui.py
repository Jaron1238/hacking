#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live TUI für die Erfassung von WiFi-Daten.
Zeigt bereits während des Capturings gefundene Geräte an.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static, Label, ProgressBar
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.timer import Timer
import time
from collections import defaultdict, Counter
from typing import Dict, Set, Optional
import multiprocessing as mp
from queue import Empty
from pathlib import Path

from ..storage.data_models import WifiEvent
from ..storage.state import WifiAnalysisState
from .. import utils


class LiveCaptureTUI(App):
    """Live Terminal-Benutzeroberfläche für die WiFi-Erfassung."""

    TITLE = "WLAN Live Capture Monitor"
    CSS_PATH = str(Path(__file__).parent.parent / "assets" / "templates" / "tui.css")

    # Reaktive Variablen
    devices_found = reactive(0)
    aps_found = reactive(0)
    clients_found = reactive(0)
    packets_processed = reactive(0)
    current_channel = reactive(0)
    capture_duration = reactive(0.0)
    
    def __init__(self, live_queue: mp.Queue, duration: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_queue = live_queue
        self.duration = duration
        self.start_time = time.time()
        self.state = WifiAnalysisState()
        self.update_timer: Optional[Timer] = None
        
        # Statistiken
        self.channel_stats = Counter()
        self.vendor_stats = Counter()
        self.ssid_stats = Counter()
        self.rssi_stats = defaultdict(list)

    def compose(self) -> ComposeResult:
        """Erstellt das Layout der Live-TUI."""
        yield Header()
        
        with Vertical(id="stats-pane"):
            yield Label("📊 Capture-Statistiken", id="stats-title")
            yield Static(id="stats-content")
            
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Label("📡 Access Points", id="aps-title")
                yield DataTable(id="aps-table")
            with Vertical(id="right-pane"):
                yield Label("📱 Clients", id="clients-title")
                yield DataTable(id="clients-table")
        
        with Vertical(id="bottom-pane"):
            yield Label("📈 Kanal-Aktivität", id="channel-title")
            yield DataTable(id="channel-table")
            
        yield Footer()

    def on_mount(self) -> None:
        """Wird aufgerufen, wenn die App startet."""
        # AP-Tabelle initialisieren
        aps_table = self.query_one("#aps-table", DataTable)
        aps_table.add_columns("BSSID", "SSID", "Vendor", "Kanal", "RSSI", "Beacons")
        
        # Client-Tabelle initialisieren
        clients_table = self.query_one("#clients-table", DataTable)
        clients_table.add_columns("MAC", "Vendor", "Probes", "APs", "RSSI")
        
        # Kanal-Tabelle initialisieren
        channel_table = self.query_one("#channel-table", DataTable)
        channel_table.add_columns("Kanal", "APs", "Clients", "Pakete", "Aktivität")
        
        # Update-Timer starten
        self.update_timer = self.set_timer(1.0, self.update_display, repeat=True)

    def update_display(self) -> None:
        """Aktualisiert die Anzeige mit neuen Daten."""
        # Verarbeite neue Events aus der Queue
        self.process_new_events()
        
        # Aktualisiere Statistiken
        self.update_stats()
        
        # Aktualisiere Tabellen
        self.update_aps_table()
        self.update_clients_table()
        self.update_channel_table()
        
        # Aktualisiere Fortschrittsbalken
        self.update_progress()

    def process_new_events(self) -> None:
        """Verarbeitet neue Events aus der Live-Queue."""
        processed = 0
        while True:
            try:
                event_data = self.live_queue.get_nowait()
                if isinstance(event_data, WifiEvent):
                    self.state.update_from_event(event_data)
                    processed += 1
                    
                    # Sammle Statistiken
                    if event_data.get('type') == 'beacon':
                        bssid = event_data.get('bssid', '')
                        ssid = event_data.get('ssid', '<hidden>')
                        channel = event_data.get('channel', 0)
                        rssi = event_data.get('rssi', -100)
                        
                        if bssid:
                            self.channel_stats[channel] += 1
                            self.ssid_stats[ssid] += 1
                            if rssi:
                                self.rssi_stats[bssid].append(rssi)
                                
                    elif event_data.get('type') in ['probe_req', 'data']:
                        client = event_data.get('client', '')
                        if client:
                            self.channel_stats[event_data.get('channel', 0)] += 1
                            
            except Empty:
                break
            except Exception as e:
                self.log.error(f"Fehler beim Verarbeiten von Events: {e}")
                break
        
        self.packets_processed += processed

    def update_stats(self) -> None:
        """Aktualisiert die Statistiken."""
        self.devices_found = len(self.state.aps) + len(self.state.clients)
        self.aps_found = len(self.state.aps)
        self.clients_found = len(self.state.clients)
        self.capture_duration = time.time() - self.start_time

    def update_aps_table(self) -> None:
        """Aktualisiert die AP-Tabelle."""
        aps_table = self.query_one("#aps-table", DataTable)
        aps_table.clear()
        
        # Sortiere APs nach Aktivität
        sorted_aps = sorted(self.state.aps.items(), 
                          key=lambda x: x[1].beacon_count, reverse=True)
        
        for bssid, ap_state in sorted_aps[:20]:  # Top 20 APs
            vendor = utils.lookup_vendor(bssid) or "Unknown"
            avg_rssi = int(ap_state.rssi_w.mean) if ap_state.rssi_w.n > 0 else -100
            aps_table.add_row(
                bssid,
                ap_state.ssid or "<hidden>",
                vendor,
                str(ap_state.channel or 0),
                f"{avg_rssi} dBm",
                str(ap_state.beacon_count)
            )

    def update_clients_table(self) -> None:
        """Aktualisiert die Client-Tabelle."""
        clients_table = self.query_one("#clients-table", DataTable)
        clients_table.clear()
        
        # Sortiere Clients nach Aktivität
        sorted_clients = sorted(self.state.clients.items(),
                              key=lambda x: x[1].count, reverse=True)
        
        for mac, client_state in sorted_clients[:20]:  # Top 20 Clients
            vendor = utils.intelligent_vendor_lookup(mac, client_state) or "Unknown"
            avg_rssi = int(client_state.rssi_w.mean) if client_state.rssi_w.n > 0 else -100
            clients_table.add_row(
                mac,
                vendor,
                str(len(client_state.probes)),
                str(len(client_state.seen_with)),
                f"{avg_rssi} dBm"
            )

    def update_channel_table(self) -> None:
        """Aktualisiert die Kanal-Aktivitätstabelle."""
        channel_table = self.query_one("#channel-table", DataTable)
        channel_table.clear()
        
        # Sortiere Kanäle nach Aktivität
        sorted_channels = sorted(self.channel_stats.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for channel, count in sorted_channels[:10]:  # Top 10 Kanäle
            # Zähle APs und Clients pro Kanal
            aps_on_channel = sum(1 for ap in self.state.aps.values() 
                               if ap.channel == channel)
            clients_on_channel = sum(1 for client in self.state.clients.values()
                                   if any(ap.channel == channel for ap in self.state.aps.values()
                                         if ap.bssid in client.seen_with))
            
            activity_level = "🔴" if count > 100 else "🟡" if count > 50 else "🟢"
            channel_table.add_row(
                str(channel),
                str(aps_on_channel),
                str(clients_on_channel),
                str(count),
                activity_level
            )

    def update_progress(self) -> None:
        """Aktualisiert den Fortschrittsbalken."""
        progress = min(100, (self.capture_duration / self.duration) * 100)
        remaining = max(0, self.duration - self.capture_duration)
        
        stats_content = self.query_one("#stats-content", Static)
        stats_text = f"""
⏱️  Verstrichene Zeit: {self.capture_duration:.0f}s / {self.duration}s
📊 Geräte gefunden: {self.devices_found} (APs: {self.aps_found}, Clients: {self.clients_found})
📦 Pakete verarbeitet: {self.packets_processed}
⏳ Verbleibende Zeit: {remaining:.0f}s
        """.strip()
        stats_content.update(stats_text)

    def on_key(self, event) -> None:
        """Behandelt Tastatureingaben."""
        if event.key == "q":
            self.exit("Capture beendet durch Benutzer")
        elif event.key == "r":
            # Reset Statistiken
            self.channel_stats.clear()
            self.vendor_stats.clear()
            self.ssid_stats.clear()
            self.rssi_stats.clear()
            self.notify("Statistiken zurückgesetzt", timeout=2)


def start_live_tui(live_queue: mp.Queue, duration: int):
    """Startet die Live-TUI für die Erfassung."""
    app = LiveCaptureTUI(live_queue, duration)
    app.run()