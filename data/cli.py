# data/cli.py

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from collections import Counter
import time
import sqlite3
from pathlib import Path
import joblib


from . import analysis, utils, database, config, state

def print_client_cluster_results(args, state, console):
    """Führt das Client-Clustering aus und gibt die Ergebnisse formatiert aus."""
    clustered_df, feature_df = analysis.cluster_clients(
        state, n_clusters=args.cluster_clients, 
        algo_name=args.cluster_algo, use_correlation=(not args.no_mac_correlation)
    )
    """Gibt die Ergebnisse des Client-Clusterings formatiert aus."""
    if clustered_df is not None and not clustered_df.empty:
        console.print("[yellow]Keine Client-Daten für das Clustering gefunden.[/yellow]")
        return
    
    title_correlation_str = "& Korrelation" if use_correlation else "& ohne Korrelation"
    title = f"Client-Cluster ({algo_name.upper()} {title_correlation_str})"
    table = Table(title=title)
    if use_correlation:
        table.add_column("Geräte-ID / MAC"); table.add_column("Vendor"); table.add_column("Cluster ID", style="bold yellow"); table.add_column("Zugehörige MACs (bei Gruppen)")
    else:
        table.add_column("MAC"); table.add_column("Vendor"); table.add_column("Cluster ID", style="bold yellow")
    for _, row in clustered_df.head(30).iterrows():
        mac_str = str(row['mac']) if pd.notna(row['mac']) else ""
        if use_correlation:
            original_macs_str = str(row['original_macs']) if "Gruppe" in mac_str else ""
            table.add_row(mac_str, row['vendor'], str(row['cluster']), original_macs_str)
        else:
            table.add_row(mac_str, row['vendor'], str(row['cluster']))
    console.print(table)
    
    summary_title = "pro Gerät/Gruppe" if use_correlation else "pro MAC-Adresse"
    console.print(f"\n[bold]Cluster-Zusammenfassung ({summary_title}):[/bold]\n" + clustered_df['cluster'].value_counts().sort_index().to_string())
    
    console.print("\n[bold cyan]--- Cluster-Profile (Abweichung vom Durchschnitt) ---[/bold cyan]")
    profiles = analysis.profile_clusters(feature_df, clustered_df)
    for cid, profile in profiles.items():
        if cid == -1 or cid == 'details': continue
        
        console.print(f"\n[bold green]Cluster {cid}:[/bold green] {profile['count']} Geräte")
        console.print("  [underline]Charakteristische Merkmale:[/underline]")
        
        for feature in profile.get('top_features', []):
            deviation = profile.get('relative_deviations', {}).get(f"{feature}_rel_diff_pct", 0)
            console.print(f"    - {feature}: {profile.get(feature, 0):.2f} ([bold {'green' if deviation > 0 else 'red'}]{deviation:+.1f}%[/bold])")

def print_ap_cluster_results(args, state, console):
    """Führt das AP-Clustering aus und gibt die Ergebnisse formatiert aus."""
    clustered_df = analysis.cluster_aps(state, n_clusters=args.cluster_aps)
    """Gibt die Ergebnisse des AP-Clusterings formatiert aus."""
    if clustered_df is None or clustered_df.empty:
        console.print("[yellow]Keine AP-Daten für das Clustering gefunden.[/yellow]")
        return

    table = Table(title="AP Clusters (Flotten)")
    table.add_column("BSSID"); table.add_column("SSID", style="cyan"); table.add_column("Vendor"); table.add_column("Auth"); table.add_column("Roaming"); table.add_column("Cluster ID", style="bold yellow")
    for _, row in clustered_df.iterrows():
        auth_str = "ENT" if row.get("is_enterprise_auth") else "PSK"
        roaming_parts = [r[9:] for r in ["supports_11k", "supports_11v", "supports_11r"] if row.get(r)]
        roaming_str = ",".join(roaming_parts) or "-"
        table.add_row(row['bssid'], row['ssid'], row['vendor'], auth_str, roaming_str, str(row['cluster']))
    console.print(table)
    
    console.print("\n[bold]Cluster-Zusammenfassung:[/bold]\n" + clustered_df['cluster'].value_counts().sort_index().to_string())
    
    console.print("\n[bold cyan]--- AP Cluster-Profile ---[/bold cyan]")
    profiles = analysis.profile_ap_clusters(clustered_df)
    for cid, profile in profiles.items():
        console.print(f"\n[bold green]Cluster {cid}:[/bold green] {profile['count']} APs")
        vendor_str = ", ".join([f"{vendor} ({count})" for vendor, count in profile['vendors'].items()])
        console.print(f"  - Hersteller: {vendor_str}")
        channel_str = ", ".join([f"Kanal {ch} ({count})" for ch, count in profile['channels'].items()])
        console.print(f"  - Kanäle: {channel_str}")
        console.print(f"  - Enterprise-Auth: {profile['enterprise_auth_pct']:.1f}%")
        console.print(f"  - Roaming-Support (11k/v/r): {profile['roaming_support_pct']:.1f}%")
    return clustered_df
def print_probed_ssids(state, console):
    """Gibt eine Rangliste der gesuchten SSIDs aus."""
    console.print("\n[bold cyan]--- Top gesuchte SSIDs (Probe Requests) ---[/bold cyan]")
    probed_ssid_counts = Counter(ssid for client in state.clients.values() for ssid in client.probes if ssid != "<broadcast>")
    if probed_ssid_counts:
        table = Table(title="Häufigkeit gesuchter SSIDs")
        table.add_column("SSID", style="cyan")
        table.add_column("Anzahl Clients", style="magenta", justify="right")
        for ssid, count in probed_ssid_counts.most_common(20):
            table.add_row(ssid, str(count))
        console.print(table)
    else:
        console.print("[yellow]Keine gesuchten SSIDs gefunden.[/yellow]")

def interactive_label_ui(db_path_events: str, label_db_path: str, model_path: str, console):
    """Startet die interaktive UI für SSID-BSSID-Paare."""
    with database.db_conn_ctx(db_path_events) as conn:
        events = list(database.fetch_events(conn))
    
    state_obj = state.WifiAnalysisState()
    state_obj.build_from_events(events)
    
    model = joblib.load(model_path) if model_path and Path(model_path).exists() else None
    results = analysis.score_pairs_with_recency_and_matching(state_obj, model=model)
    
    candidates = [c for c in results if config.UI_DEFAULT_MIN_SCORE <= c.score <= config.UI_DEFAULT_MAX_SCORE][:config.UI_MAX_CANDIDATES]
    if not candidates:
        console.print("[yellow]Keine Kandidaten im Score-Fenster gefunden.[/yellow]")
        return
        
    with database.db_conn_ctx(label_db_path) as lab_conn:
        console.print(f"[cyan]Starte interaktive Labeling-UI: {len(candidates)} Kandidaten gefunden.[/cyan]")
        for res in candidates:
            console.print(f"\n--- SSID: [bold]{res.ssid}[/bold] --- BSSID: [bold]{res.bssid}[/bold] ({utils.lookup_vendor(res.bssid) or 'N/A'}) --- Score: {res.score:.3f} ---")
            ans = Prompt.ask("Label (y=bestätigen / n=ablehnen / s=überspringen / q=beenden)", choices=["y", "n", "s", "q"], default="s").lower()
            if ans == 'q':
                break
            elif ans == 'y':
                database.add_label(lab_conn, res.ssid, res.bssid, 1)
                console.print("[green]Bestätigt.[/green]")
            elif ans == 'n':
                database.add_label(lab_conn, res.ssid, res.bssid, 0)
                console.print("[red]Abgelehnt.[/red]")
            else:
                console.print("[yellow]Übersprungen.[/yellow]")

def interactive_client_label_ui(label_db_path: str, state_obj, console):
    """Startet die interaktive UI zum Labeln von Gerätetypen."""
    custom_labels = set()
    with database.db_conn_ctx(label_db_path) as conn:
        cursor = conn.cursor()
        sorted_clients = sorted(state_obj.clients.items(), key=lambda item: item[1].last_seen, reverse=True)
        for mac, client in sorted_clients:
            vendor = utils.intelligent_vendor_lookup(mac, client) or "N/A"
            last_seen_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(client.last_seen))
            probes_str = ", ".join(sorted(list(client.probes))) if client.probes else "Keine"
            console.print("\n" + "="*80)
            console.print(f"MAC-Adresse: [bold magenta]{mac}[/bold magenta]")
            console.print(f"  - Hersteller:       [cyan]{vendor}[/cyan]")
            console.print(f"  - Zuletzt gesehen:  [yellow]{last_seen_str}[/yellow]")
            console.print(f"  - Gesuchte SSIDs:   [green]{probes_str}[/green]")
            console.print("="*80)
            
            all_device_types = sorted(list(set(config.DEVICE_TYPES) | custom_labels))
            console.print("[cyan]Gerätetyp auswählen (Zahl) oder neuen Typ eingeben:[/cyan]")
            for i, dtype in enumerate(all_device_types):
                console.print(f"  {i}) {dtype}")
            console.print("  s) Überspringen, q) Beenden")
            
            choice = console.input("> ").strip()
            selected_type = None
            
            if choice.lower() == 'q': break
            if choice.lower() == 's': continue
            
            try:
                choice_index = int(choice)
                if 0 <= choice_index < len(all_device_types):
                    selected_type = all_device_types[choice_index]
                else:
                    console.print("[red]Ungültige Zahl.[/red]")
            except ValueError:
                new_label = choice.strip()
                if new_label:
                    confirm = Prompt.ask(f"Neuen Label '[bold]{new_label}[/bold]' hinzufügen und verwenden?", choices=["y", "n"], default="y").lower()
                    if confirm == 'y':
                        selected_type = new_label
                        custom_labels.add(new_label)
                    else:
                        console.print("[yellow]Aktion abgebrochen.[/yellow]")

            if selected_type:
                try:
                    cursor.execute("INSERT OR REPLACE INTO client_labels (mac, device_type, source, ts) VALUES (?, ?, ?, ?)",
                                   (mac, selected_type, "ui", time.time()))
                    conn.commit()
                    console.print(f"[green]Gespeichert als '{selected_type}'.[/green]")
                except sqlite3.Error as e:
                    console.print(f"[bold red]Fehler beim Speichern in der Datenbank: {e}[/bold red]")