# data/tui.py

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static, Label
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
import pandas as pd
from pathlib import Path

class AnalysisTUI(App):
    """Eine interaktive Terminal-Benutzeroberfläche zur Analyse der Cluster-Ergebnisse."""

    TITLE = "WLAN Client Analyse"
    CSS_PATH = str(Path(__file__).parent.parent / "assets" / "templates" / "tui.css")

    # Daten werden beim Start der App übergeben
    clustered_df = pd.DataFrame()
    profiles = {}

    # Reaktive Variable, die bei Änderung die Detailansicht aktualisiert
    selected_client = reactive(None)

    def __init__(self, clustered_df, profiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clustered_df = clustered_df
        self.profiles = profiles
        self.cluster_ids = sorted([cid for cid in clustered_df['cluster'].unique() if cid != -1])

    def compose(self) -> ComposeResult:
        """Erstellt das Layout der TUI."""
        yield Header()
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Label("Cluster-Übersicht")
                yield DataTable(id="cluster-table")
            with Vertical(id="right-pane"):
                yield Label("Geräte im Cluster")
                yield DataTable(id="client-table")
        with Vertical(id="detail-pane"):
            yield Label("Client-Details")
            yield Static(id="client-details", expand=True)
        yield Footer()

    def on_mount(self) -> None:
        """Wird aufgerufen, wenn die App startet. Füllt die Tabellen."""
        # Cluster-Tabelle füllen
        cluster_table = self.query_one("#cluster-table", DataTable)
        cluster_table.add_columns("Cluster ID", "Anzahl Geräte", "Top Merkmal")
        for cid in self.cluster_ids:
            profile = self.profiles.get(cid, {})
            top_feature = profile.get('top_features', ['N/A'])[0]
            cluster_table.add_row(str(cid), str(profile.get('count', 0)), top_feature)
        
        # Ausreißer hinzufügen, falls vorhanden
        if -1 in self.profiles:
            cluster_table.add_row("-1", str(self.profiles[-1].get('count', 0)), "Ausreißer")

        # Client-Tabelle vorbereiten
        client_table = self.query_one("#client-table", DataTable)
        client_table.add_columns("Geräte-ID / MAC", "Hersteller")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Wird aufgerufen, wenn eine Zeile in einer Tabelle ausgewählt wird."""
        if event.control.id == "cluster-table":
            # Wenn ein Cluster ausgewählt wird, fülle die Client-Tabelle
            cluster_id = int(event.cursor_row) if event.cursor_row < len(self.cluster_ids) else -1
            client_table = self.query_one("#client-table", DataTable)
            client_table.clear()
            
            clients_in_cluster = self.clustered_df[self.clustered_df['cluster'] == cluster_id]
            for _, row in clients_in_cluster.iterrows():
                client_table.add_row(row['mac'], row['vendor'])
        
        elif event.control.id == "client-table":
            # Wenn ein Client ausgewählt wird, aktualisiere die Detailansicht
            self.selected_client = event.data_table.get_row_at(event.cursor_row)[0]

    def watch_selected_client(self, client_mac: str) -> None:
        """Aktualisiert die Detailansicht, wenn sich 'selected_client' ändert."""
        details_widget = self.query_one("#client-details", Static)
        if not client_mac:
            details_widget.update("Kein Client ausgewählt.")
            return
        
        # Finde die Profil-Daten für diesen Client
        client_profile = self.profiles.get('details', {}).get(client_mac, {})
        
        if not client_profile:
            details_widget.update(f"Keine Details für {client_mac} gefunden.")
            return

        display_text = f"[bold]{client_mac}[/bold]\n\n"
        for key, value in client_profile.items():
            if isinstance(value, float):
                display_text += f"- {key}: {value:.2f}\n"
            else:
                display_text += f"- {key}: {value}\n"
        
        details_widget.update(display_text)