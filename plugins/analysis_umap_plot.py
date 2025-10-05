# data/plugins/analysis_umap_plot.py

import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

try:
    import umap
    import plotly.express as px
except ImportError:
    umap = None
    px = None

logger = logging.getLogger(__name__)

def run(state, clustered_client_df: pd.DataFrame, client_feature_df: pd.DataFrame, outdir: Path, console):
    """
    Erstellt eine interaktive 2D-Karte aller Clients mit UMAP und Plotly.
    """
    if umap is None or px is None:
        logger.warning("UMAP/Plotly nicht installiert. Überspringe Client-Map-Visualisierung.")
        return

    if clustered_client_df is None or client_feature_df is None or client_feature_df.empty:
        logger.warning("Keine Cluster-Daten für die Client-Map-Visualisierung verfügbar.")
        return

    console.print("\n[bold cyan]Erstelle interaktive 2D-Client-Landkarte mit UMAP...[/bold cyan]")

    try:
        # Skalierte Features aus dem Clustering wiederverwenden
        features_for_map = client_feature_df.drop(columns=['original_macs'], errors='ignore')
        
        # UMAP-Reduzierer initialisieren
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        
        # Transformation durchführen
        embedding = reducer.fit_transform(features_for_map)
        
        # Ergebnis-DataFrame erstellen
        plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
        
        # Sicherstellen, dass die Indizes für den Merge ausgerichtet sind
        clustered_info = clustered_client_df.reset_index(drop=True)
        
        plot_df['cluster'] = clustered_info['cluster'].astype(str) # Als String für diskrete Farben
        plot_df['device'] = clustered_info['mac']
        plot_df['vendor'] = clustered_info['vendor']

        # Interaktiven Plot mit Plotly erstellen
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            hover_name='device',
            hover_data=['vendor'],
            title="Interaktive 2D-Karte der Client-Geräte",
            labels={'color': 'Cluster ID'}
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')), opacity=0.8)
        
        # Plot als HTML-Datei speichern
        output_file = outdir / "client_map.html"
        fig.write_html(str(output_file))
        
        console.print(f"[green]Client-Landkarte erfolgreich gespeichert in: {output_file}[/green]")

    except Exception as e:
        logger.error(f"Fehler bei der Erstellung der Client-Map: {e}", exc_info=True)