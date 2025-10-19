"""
Sankey-Diagramm Plugin für Roaming-Visualisierung.
"""

import logging
from typing import Dict, Any
from pathlib import Path
from collections import defaultdict

from plugins import BasePlugin, PluginMetadata

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

logger = logging.getLogger(__name__)

class Plugin(BasePlugin):
    """Sankey-Diagramm für Client-Roaming-Visualisierung."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Sankey Diagram",
            version="1.0.0",
            description="Erstellt Sankey-Diagramme zur Visualisierung von Client-Roaming zwischen Access Points",
            author="WLAN-Tool Team",
            dependencies=["plotly"]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Erstellt ein Sankey-Diagramm zur Visualisierung von Client-Roaming zwischen Access Points.
        """
        if go is None:
            logger.warning("Plotly nicht installiert. Überspringe Sankey-Diagramm.")
            console.print("[yellow]Plotly nicht verfügbar. Installiere mit: pip install plotly[/yellow]")
            return

        console.print("\n[bold cyan]Analysiere Roaming-Verhalten für Sankey-Diagramm...[/bold cyan]")

        try:
            transitions = defaultdict(int)
            
            # Finde für jeden Client die zeitliche Abfolge der gesehenen APs
            for client in state.clients.values():
                
                client_events = [ev for ev in events if ev.get("client") == client.mac and ev.get("bssid")]
                client_events.sort(key=lambda x: x['ts'])
                
                if len(client_events) < 2:
                    continue

                last_bssid = None
                for ev in client_events:
                    current_bssid = ev.get('bssid')
                    # Stelle sicher, dass der BSSID gültig ist und sich geändert hat
                    if current_bssid and last_bssid and current_bssid != last_bssid:
                        transitions[(last_bssid, current_bssid)] += 1
                    last_bssid = current_bssid

            if not transitions:
                logger.warning("Keine Roaming-Übergänge für das Sankey-Diagramm gefunden.")
                console.print("[yellow]Keine Roaming-Übergänge gefunden.[/yellow]")
                return

            # Daten für Plotly vorbereiten
            labels = list(set([item for t in transitions.keys() for item in t]))
            label_map = {label: i for i, label in enumerate(labels)}

            source_indices = [label_map[src] for src, dst in transitions.keys()]
            target_indices = [label_map[dst] for src, dst in transitions.keys()]
            values = list(transitions.values())
            
            # Farben für die Links basierend auf der Anzahl der Übergänge
            colors = [f"rgba(100, 149, 237, {min(1.0, v / (max(values) or 1)) * 0.6 + 0.2})" for v in values]

            # Sankey-Diagramm erstellen
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 25,
                  thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = labels,
                  color = "royalblue"
                ),
                link = dict(
                  source = source_indices,
                  target = target_indices,
                  value = values,
                  color = colors
              ))])

            fig.update_layout(title_text="Client-Roaming-Flüsse zwischen Access Points", font_size=12)
            
            output_file = Path(outdir) / "roaming_sankey.html"
            fig.write_html(str(output_file))
            
            console.print(f"[green]Roaming-Sankey-Diagramm erfolgreich gespeichert in: {output_file}[/green]")

        except Exception as e:
            console.print(f"[red]Fehler bei der Erstellung des Sankey-Diagramms: {e}[/red]")
            logger.error(f"Fehler bei der Erstellung des Sankey-Diagramms: {e}", exc_info=True)