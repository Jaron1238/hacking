"""
Erweiterte Clustering-Algorithmen Plugin.
"""

import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib

from plugins import BasePlugin, PluginMetadata

try:
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import OPTICS
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError as e:
    logging.warning(f"Einige ML-Bibliotheken nicht verfügbar: {e}")
    HAS_PLOTLY = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    logging.warning("HDBSCAN nicht verfügbar")
    HAS_HDBSCAN = False
    hdbscan = None

logger = logging.getLogger(__name__)

class Plugin(BasePlugin):
    """Erweiterte Clustering-Algorithmen für WiFi-Client-Analyse."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Advanced Clustering",
            version="1.0.0",
            description="Erweiterte Clustering-Algorithmen für WiFi-Client-Analyse",
            author="WLAN-Tool Team",
            dependencies=[
                "sklearn", "numpy", "pandas", "plotly", 
                "hdbscan", "joblib"
            ]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Führt erweiterte Clustering-Algorithmen auf den Client-Daten aus.
        """
        console.print("\n[bold cyan]Führe erweiterte Clustering-Analyse durch...[/bold cyan]")
        
        try:
            # Features extrahieren
            features, client_macs = self._extract_features_for_clustering(state, events)
            
            if len(features) < 2:
                console.print("[yellow]Nicht genügend Client-Daten für Clustering verfügbar.[/yellow]")
                return
            
            console.print(f"[green]Features extrahiert: {features.shape[0]} Clients, {features.shape[1]} Features[/green]")
            
            # Verschiedene Clustering-Algorithmen ausführen
            algorithms = [
                ("Spectral Clustering", self._run_spectral_clustering),
                ("Hierarchical Clustering", self._run_hierarchical_clustering),
                ("Gaussian Mixture Model", self._run_gaussian_mixture),
                ("OPTICS", self._run_optics_clustering),
                ("HDBSCAN", self._run_hdbscan_clustering)
            ]
            
            results = {}
            
            for algo_name, algo_func in algorithms:
                console.print(f"[cyan]Führe {algo_name} aus...[/cyan]")
                
                try:
                    if algo_name in ["OPTICS", "HDBSCAN"]:
                        labels, metrics = algo_func(features)
                    else:
                        # Automatische Cluster-Anzahl schätzen
                        n_clusters = min(8, max(2, len(features) // 3))
                        labels, metrics = algo_func(features, n_clusters)
                    
                    results[algo_name] = {
                        'labels': labels,
                        'metrics': metrics,
                        'client_macs': client_macs
                    }
                    
                    # Metriken anzeigen
                    if 'error' not in metrics:
                        console.print(f"[green]{algo_name}: {metrics['n_clusters']} Cluster, "
                                    f"Silhouette: {metrics.get('silhouette_score', 0):.3f}[/green]")
                    else:
                        console.print(f"[red]{algo_name}: {metrics['error']}[/red]")
                    
                    # Visualisierung erstellen
                    if 'error' not in metrics and len(set(labels)) > 1:
                        self._create_clustering_visualization(features, labels, client_macs, algo_name, outdir)
                    
                except Exception as e:
                    console.print(f"[red]Fehler bei {algo_name}: {e}[/red]")
                    logger.error(f"Fehler bei {algo_name}: {e}", exc_info=True)
            
            # Bestes Modell auswählen
            best_algorithm = None
            best_score = -1
            
            for algo_name, result in results.items():
                if 'error' not in result['metrics']:
                    score = result['metrics'].get('silhouette_score', 0)
                    if score > best_score:
                        best_score = score
                        best_algorithm = algo_name
            
            if best_algorithm:
                console.print(f"[bold green]Bestes Clustering-Modell: {best_algorithm} "
                             f"(Silhouette Score: {best_score:.3f})[/bold green]")
                
                # Detaillierte Cluster-Analyse
                best_result = results[best_algorithm]
                labels = best_result['labels']
                
                # Cluster-Statistiken
                cluster_stats = defaultdict(list)
                for i, (mac, label) in enumerate(zip(client_macs, labels)):
                    cluster_stats[label].append(mac)
                
                console.print(f"\n[bold]Cluster-Zusammenfassung:[/bold]")
                for cluster_id, macs in cluster_stats.items():
                    console.print(f"Cluster {cluster_id}: {len(macs)} Clients")
                    if len(macs) <= 5:  # Nur bei wenigen Clients alle anzeigen
                        console.print(f"  MACs: {', '.join(macs[:5])}")
            
            # Ergebnisse speichern
            results_file = outdir / "clustering_results.joblib"
            joblib.dump(results, results_file)
            console.print(f"[green]Clustering-Ergebnisse gespeichert: {results_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Fehler bei der Clustering-Analyse: {e}[/red]")
            logger.error(f"Fehler bei der Clustering-Analyse: {e}", exc_info=True)
    
    def _extract_features_for_clustering(self, state: Dict, events: list) -> Tuple[np.ndarray, List[str]]:
        """Extrahiert Features für das Clustering aus den Client-Daten."""
        features = []
        client_macs = []
        
        for client_mac, client in state.clients.items():
            # Basis-Features
            client_features = []
            
            # 1. Timing-Features
            if hasattr(client, 'first_seen') and hasattr(client, 'last_seen'):
                duration = client.last_seen - client.first_seen
                client_features.extend([
                    duration,
                    len(client.probe_requests) / max(duration, 1),  # Requests pro Sekunde
                ])
            else:
                client_features.extend([0, 0])
            
            # 2. Probe-Request-Features
            client_features.extend([
                len(client.probe_requests),
                len(set(client.probe_requests)),  # Eindeutige SSIDs
                len([req for req in client.probe_requests if req]),  # Nicht-leere Requests
            ])
            
            # 3. RSSI-Features
            if hasattr(client, 'rssi_history') and client.rssi_history:
                rssi_values = [r for r in client.rssi_history if r is not None]
                if rssi_values:
                    client_features.extend([
                        np.mean(rssi_values),
                        np.std(rssi_values),
                        np.min(rssi_values),
                        np.max(rssi_values),
                    ])
                else:
                    client_features.extend([0, 0, 0, 0])
            else:
                client_features.extend([0, 0, 0, 0])
            
            # 4. Information Elements Features
            if hasattr(client, 'information_elements') and client.information_elements:
                ie_count = len(client.information_elements)
                client_features.append(ie_count)
            else:
                client_features.append(0)
            
            # 5. Vendor-Features (OUI-basiert)
            if hasattr(client, 'vendor') and client.vendor:
                # Einfache Vendor-Kodierung
                vendor_hash = hash(client.vendor) % 1000
                client_features.append(vendor_hash)
            else:
                client_features.append(0)
            
            features.append(client_features)
            client_macs.append(client_mac)
        
        return np.array(features), client_macs
    
    def _run_spectral_clustering(self, features: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Führt Spectral Clustering durch."""
        if n_clusters is None:
            n_clusters = min(8, len(features) // 2)
        
        if len(features) < n_clusters:
            return np.zeros(len(features)), {"error": "Nicht genügend Daten für Clustering"}
        
        try:
            # Normalisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Spectral Clustering
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                affinity='rbf',
                gamma=1.0
            )
            labels = spectral.fit_predict(features_scaled)
            
            # Metriken berechnen
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski = calinski_harabasz_score(features_scaled, labels)
                except (NameError, ImportError):
                    calinski = 0
            else:
                silhouette = 0
                calinski = 0
            
            return labels, {
                "algorithm": "Spectral Clustering",
                "n_clusters": len(set(labels)),
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski
            }
        except Exception as e:
            logger.error(f"Fehler bei Spectral Clustering: {e}")
            return np.zeros(len(features)), {"error": str(e)}
    
    def _run_hierarchical_clustering(self, features: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Führt Hierarchical Clustering durch."""
        if n_clusters is None:
            n_clusters = min(8, len(features) // 2)
        
        if len(features) < n_clusters:
            return np.zeros(len(features)), {"error": "Nicht genügend Daten für Clustering"}
        
        try:
            # Normalisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Hierarchical Clustering
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = hierarchical.fit_predict(features_scaled)
            
            # Metriken berechnen
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski = calinski_harabasz_score(features_scaled, labels)
                except (NameError, ImportError):
                    calinski = 0
            else:
                silhouette = 0
                calinski = 0
            
            return labels, {
                "algorithm": "Hierarchical Clustering",
                "n_clusters": len(set(labels)),
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski
            }
        except Exception as e:
            logger.error(f"Fehler bei Hierarchical Clustering: {e}")
            return np.zeros(len(features)), {"error": str(e)}
    
    def _run_gaussian_mixture(self, features: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Führt Gaussian Mixture Model Clustering durch."""
        if n_clusters is None:
            n_clusters = min(8, len(features) // 2)
        
        if len(features) < n_clusters:
            return np.zeros(len(features)), {"error": "Nicht genügend Daten für Clustering"}
        
        try:
            # Normalisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                covariance_type='full'
            )
            labels = gmm.fit_predict(features_scaled)
            
            # Metriken berechnen
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski = calinski_harabasz_score(features_scaled, labels)
                except (NameError, ImportError):
                    calinski = 0
                aic = gmm.aic(features_scaled)
                bic = gmm.bic(features_scaled)
            else:
                silhouette = 0
                calinski = 0
                aic = 0
                bic = 0
            
            return labels, {
                "algorithm": "Gaussian Mixture Model",
                "n_clusters": len(set(labels)),
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "aic": aic,
                "bic": bic
            }
        except Exception as e:
            logger.error(f"Fehler bei Gaussian Mixture Model: {e}")
            return np.zeros(len(features)), {"error": str(e)}
    
    def _run_optics_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Führt OPTICS Clustering durch."""
        if len(features) < 3:
            return np.zeros(len(features)), {"error": "Nicht genügend Daten für OPTICS"}
        
        try:
            # Normalisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # OPTICS Clustering
            optics = OPTICS(
                min_samples=2,
                max_eps=0.5,
                metric='euclidean'
            )
            labels = optics.fit_predict(features_scaled)
            
            # Metriken berechnen
            if len(set(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski = calinski_harabasz_score(features_scaled, labels)
                except (NameError, ImportError):
                    calinski = 0
            else:
                silhouette = 0
                calinski = 0
            
            return labels, {
                "algorithm": "OPTICS",
                "n_clusters": len(set(labels)),
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "reachability": optics.reachability_.tolist() if hasattr(optics, 'reachability_') else []
            }
        except Exception as e:
            logger.error(f"Fehler bei OPTICS Clustering: {e}")
            return np.zeros(len(features)), {"error": str(e)}
    
    def _run_hdbscan_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Führt HDBSCAN Clustering durch."""
        if not HAS_HDBSCAN:
            return np.zeros(len(features)), {"error": "HDBSCAN nicht verfügbar"}
        
        if len(features) < 3:
            return np.zeros(len(features)), {"error": "Nicht genügend Daten für HDBSCAN"}
        
        try:
            # Normalisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # HDBSCAN Clustering
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric='euclidean'
            )
            labels = hdbscan_clusterer.fit_predict(features_scaled)
            
            # Metriken berechnen
            if len(set(labels)) > 1 and -1 not in labels:
                silhouette = silhouette_score(features_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski = calinski_harabasz_score(features_scaled, labels)
                except (NameError, ImportError):
                    calinski = 0
            else:
                silhouette = 0
                calinski = 0
            
            return labels, {
                "algorithm": "HDBSCAN",
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise": sum(1 for l in labels if l == -1),
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "cluster_persistence": hdbscan_clusterer.cluster_persistence_.tolist() if hasattr(hdbscan_clusterer, 'cluster_persistence_') else []
            }
        except Exception as e:
            logger.error(f"Fehler bei HDBSCAN Clustering: {e}")
            return np.zeros(len(features)), {"error": str(e)}
    
    def _create_clustering_visualization(self, features: np.ndarray, labels: np.ndarray, 
                                      client_macs: List[str], algorithm_name: str, 
                                      outdir: Path) -> None:
        """Erstellt Visualisierungen für die Clustering-Ergebnisse."""
        if not HAS_PLOTLY:
            logger.warning("Plotly nicht verfügbar. Überspringe Visualisierung.")
            return
            
        try:
            # PCA für 2D-Visualisierung
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # DataFrame für Plotly
            df = pd.DataFrame({
                'x': features_2d[:, 0],
                'y': features_2d[:, 1],
                'cluster': labels,
                'mac': client_macs
            })
            
            # Scatter Plot
            fig = px.scatter(
                df, x='x', y='y', color='cluster',
                title=f'{algorithm_name} - Clustering Ergebnisse',
                labels={'x': 'PC1', 'y': 'PC2'},
                hover_data=['mac']
            )
            
            # Speichern
            output_file = outdir / f"clustering_{algorithm_name.lower().replace(' ', '_')}.html"
            fig.write_html(str(output_file))
            
            logger.info(f"Clustering-Visualisierung gespeichert: {output_file}")
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung: {e}")