#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse-Logik: Feature-Extraktion, Scoring, Matching und Clustering.
"""
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from datetime import date

# Relative Imports
from .. import config
from ..storage.state import WifiAnalysisState
from ..storage.data_models import ClientState, APState, InferenceResult
from ..utils import intelligent_vendor_lookup, ie_fingerprint_hash, lookup_vendor
from ..exceptions import (
    AnalysisError, ValidationError, ResourceError,
    handle_errors, retry_on_error, ErrorContext, ErrorRecovery
)


import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import networkx as nx
from networkx import Graph, MultiGraph
import torch
from datetime import datetime
try:
    from .models import ClientAutoencoder
except ImportError:
    ClientAutoencoder = None

logger = logging.getLogger(__name__)

# Liste bekannter öffentlicher SSIDs für das Profiling
KNOWN_PUBLIC_SSIDS = {"xfinitywifi", "eduroam", "Telekom_FON", "Vodafone Homespot", "cablewifi", "Spectrum Mobile"}


def features_for_pair_basic(ssid: str, bssid: str, state: WifiAnalysisState) -> Tuple[Dict, List[str]]:
    ap = state.aps.get(bssid)
    probing = state.clients_probing_ssid.get(ssid, set())
    seen_with = state.clients_seen_with_bssid.get(bssid, set())
    supporting_clients = list(probing.intersection(seen_with))
    bi_mc = (ap.beacon_intervals.most_common(1)[0]) if ap and ap.beacon_intervals else (0,0)
    seq_support = sum(state.seq_local.get((c, bssid), {}).get("monotonic_count", 0) for c in supporting_clients)
    ap_parsed = ap.parsed_ies if ap else {}
    vht_caps = ap_parsed.get("vht_caps", {})
    beacon_count = ap.beacon_count if ap else 0
    probe_resp_count = ap.probe_resp_count if ap else 0
    num_supporting_clients = len(supporting_clients)
    probe_to_beacon_ratio = probe_resp_count / (beacon_count + 1e-6)
    client_to_beacon_ratio = num_supporting_clients / (beacon_count + 1e-6)
    ie_fp_hash = ie_fingerprint_hash(ap.ies) if ap else None
    ie_fingerprint_int = int(ie_fp_hash, 16) if ie_fp_hash else 0
    return {
        "ssid": ssid, "bssid": bssid,
        "beacon_count": beacon_count,
        "probe_resp_count": probe_resp_count,
        "supporting_clients": num_supporting_clients,
        "rssi_mean": ap.rssi_w.mean if ap else 0.0,
        "rssi_std": ap.rssi_w.std() if ap else 0.0,
        "channel": ap.channel or 0,
        "beacon_interval_mode": bi_mc[0],
        "beacon_interval_consistency": bi_mc[1],
        "cap_count": len(ap.cap_bits) if ap else 0,
        "ie_count": sum(len(v) for v in (ap.ies.values() if ap else [])),
        "seq_support": seq_support,
        "vendor_known": 1 if lookup_vendor(bssid) else 0,
        "ap_supports_11ax": 1 if "802.11ax" in ap_parsed.get("standards", []) else 0,
        "ap_mu_mimo_capable": 1 if vht_caps.get("mu_beamformer_capable") else 0,
        "probe_to_beacon_ratio": probe_to_beacon_ratio,
        "client_to_beacon_ratio": client_to_beacon_ratio,
        "ie_fingerprint_int": ie_fingerprint_int,
    }, supporting_clients

def heuristic_score(feat: dict, norm: Optional[Dict] = None) -> float:
    if norm is None:
        norm = config.HEURISTIC_NORMALIZATION
    w = config.HEURISTIC_WEIGHTS
    scores = {
        "beacon": min(1.0, feat.get("beacon_count", 0) / (norm["beacon_count"] + 1e-6)),
        "probe": min(1.0, feat.get("probe_resp_count", 0) / (norm["probe_resp_count"] + 1e-6)),
        "client": min(1.0, feat.get("supporting_clients", 0) / (norm["supporting_clients"] + 1e-6)),
        "rssi": max(0.0, 1.0 - (feat.get("rssi_std", 0.0) / (norm["rssi_std_max"] + 1e-6))),
        "channel": 1.0,
        "seq": min(1.0, feat.get("seq_support", 0) / (norm["seq_support"] + 1e-6)),
    }
    return sum(w[k] * scores[k] for k in w)

def score_pairs_with_recency_and_matching(state: WifiAnalysisState, model=None) -> List[InferenceResult]:
    pair_components = {(s, b): features_for_pair_basic(s, b, state)
                       for s, info in state.ssid_map.items() if s and s != "<hidden>"
                       for b in info["bssids"]}
    if not pair_components: return []
    all_features = [feat for feat, _ in pair_components.values()]
    df_features = pd.DataFrame(all_features)
    dynamic_norm = {
        "beacon_count": max(1.0, df_features["beacon_count"].quantile(0.90)),
        "probe_resp_count": max(1.0, df_features["probe_resp_count"].quantile(0.90)),
        "supporting_clients": max(1.0, df_features["supporting_clients"].quantile(0.90)),
        "seq_support": max(1.0, df_features["seq_support"].quantile(0.90)),
        "rssi_std_max": 20.0
    }
    logger.debug(f"Dynamische Normalisierungsfaktoren: {dynamic_norm}")
    pair_scores = {k: heuristic_score(feat, norm=dynamic_norm) for k, (feat, _) in pair_components.items()}
    if model is not None and not df_features.empty:
        logger.info("Verwende trainiertes ML-Modell zur Verfeinerung der Scores.")
        df_for_pred = df_features.drop(columns=["ssid", "bssid"], errors="ignore")
        try:
            if hasattr(model, 'feature_names_in_'):
                 model_features = model.feature_names_in_
            elif hasattr(model.steps[-1][1], 'feature_names_in_'):
                 model_features = model.steps[-1][1].feature_names_in_
            else:
                 model_features = df_for_pred.columns
            df_for_pred = df_for_pred.reindex(columns=model_features, fill_value=0)
            ml_probs = model.predict_proba(df_for_pred)[:, 1]
            for i, (key, score) in enumerate(pair_scores.items()):
                ml_score = ml_probs[i]
                combined_score = (ml_score * config.ML_SCORE_WEIGHT) + (score * config.HEURISTIC_SCORE_WEIGHT)
                pair_scores[key] = combined_score
        except Exception as e:
            logger.warning(f"ML-Modell-Anwendung fehlgeschlagen: {e}. Verwende nur heuristische Scores.")
    ie_groups = defaultdict(set)
    for b, ap in state.aps.items():
        if h := ie_fingerprint_hash(ap.ies): ie_groups[h].add(b)
    multi_ssid_bssids = {b for bset in ie_groups.values() if len({s for b in bset for s, i in state.ssid_map.items() if b in i["bssids"]}) >= config.MULTI_SSID_THRESHOLD for b in bset}
    G = nx.Graph()
    ssids = {s for s, b in pair_scores.keys()}
    bssids = {b for s, b in pair_scores.keys()}
    G.add_nodes_from(ssids, bipartite=0)
    G.add_nodes_from(bssids, bipartite=1)
    for (s, b), score in pair_scores.items():
        if score > (config.SCORE_LABEL_MEDIUM / 2):
             G.add_edge(s, b, weight=score)
    matching = nx.max_weight_matching(G, maxcardinality=False)
    assignment = {}
    for u, v in matching:
        if u in ssids: assignment[u] = (v, G[u][v]['weight'])
        else: assignment[v] = (u, G[u][v]['weight'])
    results: List[InferenceResult] = []
    for s, (b, score) in assignment.items():
        feat, sup = pair_components[(s, b)]
        label = "high" if score >= config.SCORE_LABEL_HIGH else ("medium" if score >= config.SCORE_LABEL_MEDIUM else "low")
        result = InferenceResult(
            ssid=s, bssid=b, score=round(score, 3), label=label, components=feat,
            evidence={"supporting_clients": sup, "multi_ssid_allowed": (b in multi_ssid_bssids)}
        )
        results.append(result)
    return sorted(results, key=lambda r: r.score, reverse=True)

@handle_errors(AnalysisError, "CLIENT_FEATURES_ERROR", default_return=None)
def features_for_client(client_state: ClientState) -> Optional[Dict[str, any]]:
    """Extrahiere Client-Features mit Fehlerbehandlung."""
    with ErrorContext("client_features_extraction", "CLIENT_FEATURES_ERROR"):
        if not client_state:
            raise ValidationError(
                "Client state is None",
                error_code="NULL_CLIENT_STATE",
                details={"function": "features_for_client"}
            )
        
        if not isinstance(client_state, ClientState):
            raise ValidationError(
                f"Invalid client state type: {type(client_state)}",
                error_code="INVALID_CLIENT_STATE_TYPE",
                details={"expected": "ClientState", "actual": str(type(client_state))}
            )
        
        try:
            parsed = client_state.parsed_ies or {}
            ht_caps = parsed.get("ht_caps", {})
            vht_caps = parsed.get("vht_caps", {})
            
            # Validiere kritische Felder
            if not hasattr(client_state, 'mac') or not client_state.mac:
                raise ValidationError(
                    "Client state missing MAC address",
                    error_code="MISSING_MAC_ADDRESS",
                    details={"client_state": str(client_state)}
                )
            
            return {
                "vendor": intelligent_vendor_lookup(client_state.mac, client_state) or "Unknown",
                "supports_11n": 1 if "802.11n" in parsed.get("standards", []) else 0,
                "supports_11ac": 1 if "802.11ac" in parsed.get("standards", []) else 0,
                "supports_11ax": 1 if "802.11ax" in parsed.get("standards", []) else 0,
                "probe_count": len(client_state.probes) if hasattr(client_state, 'probes') else 0,
                "seen_with_ap_count": len(client_state.seen_with) if hasattr(client_state, 'seen_with') else 0,
                "was_in_powersave": 1 if getattr(client_state, 'last_powersave_ts', 0) > 0 else 0,
                "is_randomized_mac": 1 if getattr(client_state, 'randomized', False) else 0,
                "ht_40mhz_support": 1 if ht_caps.get("40mhz_support") else 0,
                "vht_160mhz_support": 1 if vht_caps.get("160mhz_support") else 0,
                "mimo_streams": ht_caps.get("streams", 0),
                "has_apple_ie": 1 if parsed.get("vendor_specific", {}).get("Apple") else 0,
                "has_ms_ie": 1 if parsed.get("vendor_specific", {}).get("Microsoft") else 0,
            }
        except Exception as e:
            raise AnalysisError(
                f"Failed to extract client features: {e}",
                error_code="CLIENT_FEATURES_EXTRACTION_FAILED",
                details={"mac": getattr(client_state, 'mac', 'unknown'), "original_error": str(e)}
            ) from e

def features_for_client_behavior(client_state: ClientState) -> Optional[Dict[str, any]]:
    """Extrahiert technische, Verhaltens- und Session-Merkmale."""
    if np is None: logger.error("Numpy wird für die Verhaltensanalyse benötigt."); return None
    if len(client_state.all_packet_ts) < 5: return None
    
    # --- Standard-Verhaltensmerkmale ---
    intervals = np.diff(client_state.all_packet_ts)
    packet_interval_mean = np.mean(intervals) if len(intervals) > 0 else 0
    packet_interval_std = np.std(intervals) if len(intervals) > 0 else 0
    total_frames = client_state.data_frame_count + client_state.mgmt_frame_count
    data_to_mgmt_ratio = client_state.data_frame_count / (total_frames + 1e-6)
    rssi_std = client_state.rssi_w.std()
    duration = client_state.last_seen - client_state.first_seen
    power_save_rate = client_state.power_save_transitions / (duration + 1e-6)
    
    sessions = []
    if intervals.any():
        session_breaks = np.where(intervals > 10.0)[0] # Session-Ende nach 10s Inaktivität
        session_starts = np.insert(session_breaks + 1, 0, 0)
        session_ends = np.append(session_breaks, len(client_state.all_packet_ts) - 1)
        for start_idx, end_idx in zip(session_starts, session_ends):
            if end_idx > start_idx:
                session_ts = client_state.all_packet_ts[start_idx:end_idx+1]
                duration = session_ts[-1] - session_ts[0]
                sessions.append({'duration': duration, 'packet_count': len(session_ts)})

    avg_session_duration = np.mean([s['duration'] for s in sessions]) if sessions else 0
    avg_packets_per_session = np.mean([s['packet_count'] for s in sessions]) if sessions else 0
    session_frequency = len(sessions) / (duration / 60 + 1e-6) # Sessions pro Minute
    avg_mcs = 0.0
    max_mcs = 0.0
    if client_state.mcs_rates:
        total_mcs_packets = sum(client_state.mcs_rates.values())
        if total_mcs_packets > 0:
            avg_mcs = sum(k * v for k, v in client_state.mcs_rates.items()) / total_mcs_packets
            max_mcs = max(client_state.mcs_rates.keys())

    probed_ssids = {s for s in client_state.probes if s != "<broadcast>"}
    probed_ssid_diversity = len(probed_ssids)
    probed_public_ssid_count = len(probed_ssids.intersection(KNOWN_PUBLIC_SSIDS))
    
    avg_snr = client_state.rssi_w.mean - client_state.noise_w.mean if client_state.noise_w.n > 0 else 0
    fcs_error_rate = client_state.fcs_error_count / client_state.count if client_state.count > 0 else 0
    os_fingerprint_diversity = len(client_state.ie_order_hashes)

    return {
        "mac": client_state.mac, "packet_interval_mean": packet_interval_mean,
        "packet_interval_std": packet_interval_std, "data_to_mgmt_ratio": data_to_mgmt_ratio,
        "rssi_std": rssi_std, "power_save_rate": power_save_rate,
        "unique_probes_count": len(client_state.probes), "unique_aps_seen_count": len(client_state.seen_with),
        "avg_mcs_rate": avg_mcs, "max_mcs_rate": max_mcs,
        "probed_ssid_diversity": probed_ssid_diversity,
        "probed_public_ssids": probed_public_ssid_count,
        "avg_snr": avg_snr,
        "fcs_error_rate": fcs_error_rate,
        "os_fingerprint_diversity": os_fingerprint_diversity,
        "avg_session_duration": avg_session_duration,
        "avg_packets_per_session": avg_packets_per_session,
        "session_frequency": session_frequency
    }

def prepare_client_feature_dataframe(state: WifiAnalysisState, correlate_macs: bool = True) -> Optional[pd.DataFrame]:
    """Sammelt kombinierte technische und Verhaltensmerkmale für alle Clients."""
    clients_for_df = []
    
    def get_combined_features(client_state: ClientState):
        features = features_for_client(client_state)
        if features:
            if behavior_features := features_for_client_behavior(client_state):
                behavior_features.pop('mac', None)
                features.update(behavior_features)
        return features

    if correlate_macs:
        logger.info("Korreliere randomisierte MACs vor der Feature-Extraktion...")
        correlated_groups = correlate_randomized_clients(state)
        processed_macs = set()
        for i, (fingerprint, macs) in enumerate(correlated_groups.items()):
            representative_mac = macs[0]
            if client_state := state.clients.get(representative_mac):
                if features := get_combined_features(client_state):
                    features['mac'] = f"Gruppe {i+1} ({len(macs)} MACs)"
                    features['original_macs'] = ", ".join(macs)
                    clients_for_df.append(features)
            processed_macs.update(macs)
        for mac, client_state in state.clients.items():
            if mac not in processed_macs:
                if features := get_combined_features(client_state):
                    features['mac'] = mac
                    features['original_macs'] = mac
                    clients_for_df.append(features)
    else:
        logger.info("MAC-Korrelation deaktiviert. Extrahiere Features für jede MAC-Adresse einzeln.")
        for mac, client_state in state.clients.items():
            if features := get_combined_features(client_state):
                features['mac'] = mac
                features['original_macs'] = mac
                clients_for_df.append(features)
    
    if not clients_for_df:
        logger.warning("Keine Client-Daten für DataFrame-Erstellung verfügbar.")
        return None
        
    return pd.DataFrame(clients_for_df)


def find_optimal_k_elbow_and_silhouette(X_scaled, max_k: Optional[int] = None) -> Optional[int]:
    if KMeans is None or silhouette_score is None: return None
    num_samples = X_scaled.shape[0]
    if max_k is None: max_k = int(np.sqrt(num_samples))
    max_k = min(max_k, num_samples - 1)
    if max_k < 2:
        logger.warning("Nicht genügend Datenpunkte zum Clustern vorhanden.")
        return None
    logger.info("Suche optimales k für Clustering (2 bis %d) mit kombinierter Methode...", max_k)
    k_values = range(2, max_k + 1)
    inertias, silhouette_scores = [], []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        try:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        except ValueError:
            silhouette_scores.append(-1)
    best_k_silhouette = k_values[np.argmax(silhouette_scores)] if silhouette_scores and max(silhouette_scores) > -1 else None
    try:
        deltas = np.diff(inertias, 2) 
        best_k_elbow = k_values[np.argmax(deltas) + 1]
    except (ValueError, IndexError):
        best_k_elbow = None
    logger.debug(f"Bestes k (Silhouette): {best_k_silhouette}, Bestes k (Ellenbogen): {best_k_elbow}")
    final_k = best_k_silhouette or best_k_elbow
    if final_k:
        logger.info(f"Analyse schlägt k={final_k} vor.")
    else:
        logger.warning("Konnte optimale Cluster-Anzahl nicht bestimmen.")
    return final_k

@handle_errors(AnalysisError, "CLIENT_CLUSTERING_ERROR", default_return=(None, None))
def cluster_clients(
    state: WifiAnalysisState, 
    n_clusters: int = 5, 
    use_encoder_path: Optional[str] = None, 
    correlate_macs: bool = True,
    algo: str = "kmeans"
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Clustere Clients mit Fehlerbehandlung."""
    with ErrorContext("client_clustering", "CLIENT_CLUSTERING_ERROR"):
        # Validiere Parameter
        if not isinstance(state, WifiAnalysisState):
            raise ValidationError(
                f"Invalid state type: {type(state)}",
                error_code="INVALID_STATE_TYPE",
                details={"expected": "WifiAnalysisState", "actual": str(type(state))}
            )
        
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValidationError(
                f"Invalid n_clusters: {n_clusters}",
                error_code="INVALID_N_CLUSTERS",
                details={"n_clusters": n_clusters}
            )
        
        if algo not in ["kmeans", "dbscan"]:
            raise ValidationError(
                f"Invalid clustering algorithm: {algo}",
                error_code="INVALID_CLUSTERING_ALGO",
                details={"algo": algo, "supported": ["kmeans", "dbscan"]}
            )
        
        # Prüfe Abhängigkeiten
        if KMeans is None or DBSCAN is None:
            raise ResourceError(
                "scikit-learn is required for clustering",
                error_code="MISSING_SKLEARN",
                details={"function": "cluster_clients"}
            )
        
        try:
            df = prepare_client_feature_dataframe(state, correlate_macs=correlate_macs)
            
            if df is None or df.empty:
                logger.warning("No client data available for clustering")
                return None, None

            # Validiere DataFrame-Struktur
            required_columns = ['mac', 'original_macs', 'vendor']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(
                    f"Missing required columns: {missing_columns}",
                    error_code="MISSING_DF_COLUMNS",
                    details={"missing_columns": missing_columns, "available_columns": list(df.columns)}
                )

            df.fillna(0, inplace=True)
            mac_addresses = df['mac']
            original_macs = df['original_macs']
            vendor_strings = df['vendor']
            
            # Erstelle numerische Features
            df_numeric = pd.get_dummies(df.drop(columns=['mac', 'original_macs', 'vendor']))
            
            if df_numeric.empty:
                raise AnalysisError(
                    "No numeric features available for clustering",
                    error_code="NO_NUMERIC_FEATURES",
                    details={"df_shape": df.shape, "numeric_shape": df_numeric.shape}
                )
            
            X_scaled = StandardScaler().fit_transform(df_numeric)
            data_to_cluster = X_scaled

    if use_encoder_path and ClientAutoencoder:
        logger.info("Autoencoder wird verwendet. Manuelle Feature-Gewichtung wird übersprungen.")
        if torch and Path(use_encoder_path).exists():
            try:
                embedding_dim = 16
                encoder = ClientAutoencoder(input_dim=X_scaled.shape[1], embedding_dim=embedding_dim).encoder
                encoder.load_state_dict(torch.load(use_encoder_path))
                encoder.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    embeddings = encoder(X_tensor).numpy()
                data_to_cluster = embeddings
                logger.info(f"Client-Features in {embedding_dim}-dimensionale Embeddings umgewandelt.")
            except Exception as e:
                logger.error(f"Fehler beim Laden/Anwenden des Encoders: {e}. Fallback auf Standard.")
        else:
            logger.warning(f"Encoder-Modell nicht gefunden oder torch nicht verfügbar. Fallback auf Standard.")
    else:
        logger.info("Wende manuelle Feature-Gewichtungen für Standard-Clustering an...")
        X_scaled_df = pd.DataFrame(X_scaled, columns=df_numeric.columns)
        weights = config.CLIENT_CLUSTERING_FEATURE_WEIGHTS
        for feature, weight in weights.items():
            cols_to_weight = [col for col in X_scaled_df.columns if col.startswith(feature)]
            if cols_to_weight:
                logger.debug(f"Gewichte Feature '{feature}' mit Faktor {weight}.")
                X_scaled_df[cols_to_weight] *= weight
        data_to_cluster = X_scaled_df.to_numpy()
    
    if algo == "dbscan":
        logger.info("Verwende DBSCAN für das Clustering...")
        dbscan = DBSCAN(eps=1.5, min_samples=3)
        labels = dbscan.fit_predict(data_to_cluster)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"DBSCAN fand {n_clusters_found} Cluster und {np.sum(labels == -1)} Ausreißer.")
    else:
        logger.info("Verwende KMeans für das Clustering...")
        if n_clusters == 0:
            optimal_k = find_optimal_k_elbow_and_silhouette(data_to_cluster)
            n_clusters = optimal_k or 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data_to_cluster)
    
    result_df = pd.DataFrame({'mac': mac_addresses, 'original_macs': original_macs, 'vendor': vendor_strings, 'cluster': labels})
    
    df_numeric['original_macs'] = original_macs
    return result_df, df_numeric


def profile_clusters(feature_df: pd.DataFrame, clustered_df: pd.DataFrame) -> Dict:
    """Analysiert die Cluster und gibt ein Profil für jeden zurück."""
    
    if 'mac' in feature_df.columns:
        feature_df = feature_df.drop(columns=['mac'])
        
    full_df = pd.merge(feature_df, clustered_df[['original_macs', 'cluster']], on='original_macs')
    profiles = {}
    global_means = full_df[full_df['cluster'] != -1].mean(numeric_only=True)

    
    for cluster_id in sorted(full_df['cluster'].unique()):
        if cluster_id == -1:
            profiles[-1] = {"count": int(np.sum(full_df['cluster'] == -1)), "description": "Ausreißer / Rauschen"}
            continue
            
        cluster_data = full_df[full_df['cluster'] == cluster_id]
        
        profile = cluster_data.mean(numeric_only=True).to_dict()
        profile['count'] = len(cluster_data)
        relative_profile = {}
        for key, value in profile.items():
            if key in global_means and global_means[key] != 0:
                deviation = (value / global_means[key] - 1) * 100
                relative_profile[f"{key}_rel_diff_pct"] = deviation
        profiles['details'] = full_df.set_index('mac').to_dict(orient='index')

        profile['relative_deviations'] = relative_profile
        global_means = full_df[full_df['cluster'] != -1].mean(numeric_only=True)
        std_devs = full_df[full_df['cluster'] != -1].std(numeric_only=True).replace(0, 1)
        
        z_scores = (pd.Series(profile) - global_means) / std_devs
        
        profile['top_features'] = z_scores.nlargest(3).index.tolist()
        profiles[cluster_id] = profile
    profiles['details'] = full_df.set_index('original_macs').to_dict(orient='index')
          
    return profiles

def features_for_ap(ap_state: APState) -> Optional[Dict[str, any]]:
    if not ap_state: return None
    bi_mc = (ap_state.beacon_intervals.most_common(1)[0]) if ap_state.beacon_intervals else (0, 0)
    parsed = ap_state.parsed_ies
    rsn = parsed.get("rsn_details", {})
    roaming = parsed.get("roaming_features", [])
    return {
        "ssid": ap_state.ssid or "<unknown>", "vendor": lookup_vendor(ap_state.bssid) or "Unknown",
        "channel": ap_state.channel or 0, "beacon_interval_mode": bi_mc[0],
        "beacon_interval_consistency": bi_mc[1] / (ap_state.beacon_count or 1), "rssi_mean": ap_state.rssi_w.mean,
        "rssi_std": ap_state.rssi_w.std(), "cap_count": len(ap_state.cap_bits),
        "ie_fingerprint": int(ie_fingerprint_hash(ap_state.ies), 16),
        "supports_tkip": 1 if "TKIP" in rsn.get("pairwise_ciphers", []) else 0,
        "is_enterprise_auth": 1 if "802.1X (EAP)" in rsn.get("akm_suites", []) else 0,
        "mfp_capable": 1 if rsn.get("mfp_capable") else 0,
        "mfp_required": 1 if rsn.get("mfp_required") else 0,
        "supports_11k": 1 if "802.11k" in roaming else 0, "supports_11v": 1 if "802.11v" in roaming else 0,
        "supports_11r": 1 if "802.11r" in roaming else 0,
    }

def cluster_aps(state: WifiAnalysisState, n_clusters: int = 5) -> Optional["pd.DataFrame"]:
    if KMeans is None: logger.error("scikit-learn wird für AP-Clustering benötigt."); return None
    rows = [dict(bssid=bssid, **(feat or {})) for bssid, ap in state.aps.items() if (feat := features_for_ap(ap))]
    if not rows: logger.warning("Keine APs mit ausreichenden Merkmalen zum Clustern."); return None
    df = pd.DataFrame(rows)
    df_dummies = pd.get_dummies(df, columns=['vendor'], prefix='vendor')
    X = df_dummies.drop(columns=['bssid', 'ssid'])
    X_scaled = StandardScaler().fit_transform(X)
    if n_clusters == 0:
        optimal_k = find_optimal_k_elbow_and_silhouette(X_scaled)
        n_clusters = optimal_k or 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X_scaled)
    return df

def profile_ap_clusters(clustered_ap_df: pd.DataFrame) -> Dict:
    """Analysiert AP-Cluster und gibt ein Profil für jeden zurück."""
    if clustered_ap_df is None or clustered_ap_df.empty:
        return {}
    
    profiles = {}
    
    bool_cols = [col for col in clustered_ap_df.columns if clustered_ap_df[col].dtype == 'bool']
    ap_df_numeric = clustered_ap_df.copy()
    ap_df_numeric[bool_cols] = ap_df_numeric[bool_cols].astype(int)

    for cluster_id in sorted(ap_df_numeric['cluster'].unique()):
        cluster_data = ap_df_numeric[ap_df_numeric['cluster'] == cluster_id]
        
        roaming_cols = ['supports_11k', 'supports_11v', 'supports_11r']
        has_roaming = any(col in cluster_data.columns for col in roaming_cols)
        roaming_support_pct = 0
        if has_roaming:
            roaming_flags = cluster_data[roaming_cols].any(axis=1)
            roaming_support_pct = roaming_flags.mean() * 100

        profile = {
            "count": len(cluster_data),
            "vendors": cluster_data['vendor'].value_counts().to_dict(),
            "channels": cluster_data['channel'].value_counts().to_dict(),
            "avg_rssi": cluster_data['rssi_mean'].mean(),
            "enterprise_auth_pct": cluster_data['is_enterprise_auth'].mean() * 100 if 'is_enterprise_auth' in cluster_data else 0,
            "roaming_support_pct": roaming_support_pct
        }
        profiles[cluster_id] = profile
        
    return profiles

def correlate_randomized_clients(state: WifiAnalysisState) -> Dict[str, List[str]]:
    probe_fingerprints = defaultdict(list)
    randomized_clients = [c for c in state.clients.values() if c.randomized and c.probes]
    for client in randomized_clients:
        fingerprint = "|".join(sorted(list(client.probes)))
        probe_fingerprints[fingerprint].append(client.mac)
    correlated_groups = {fp: macs for fp, macs in probe_fingerprints.items() if len(macs) > 1}
    logger.info("Fand %d Gruppen von potenziell zusammengehörigen randomisierten MACs.", len(correlated_groups))
    return correlated_groups

# ==============================================================================
# START: Verbesserter, standardkonformer Graphen-Export (GEXF 1.3)
# ==============================================================================

def _build_export_graph(
    state: WifiAnalysisState,
    clustered_ap_df: pd.DataFrame,
    aps_to_export: Dict[str, APState],
    clients_to_export: Dict[str, ClientState],
    include_clients: bool,
    clustered_client_df: Optional[pd.DataFrame]
) -> nx.Graph:
    """
    Baut den NetworkX-Graphen mit allen erweiterten Attributen auf.
    Erstellt ein explizites Intervall-Attribut für maximale Kompatibilität mit Gephi.
    """
    
    G = nx.Graph()
    G.graph['mode'] = 'dynamic'
    G.graph['timeformat'] = 'datetime'
    
    for _, row in clustered_ap_df.iterrows():
        bssid_str = str(row['bssid'])
        if bssid_str not in aps_to_export: continue
            
        ap_state = aps_to_export[bssid_str]
        vendor_str = str(row.get('vendor', 'Unknown'))
        
        # Convert timestamps to ISO format for Gephi timeline animation
        start_iso = datetime.fromtimestamp(ap_state.first_seen).isoformat()
        end_iso = datetime.fromtimestamp(ap_state.last_seen).isoformat()
        interval_string = f"<[{ap_state.first_seen}, {ap_state.last_seen}]>"
        roaming_support = bool(row.get('supports_11k') or row.get('supports_11v') or row.get('supports_11r'))

        G.add_node(
            bssid_str,
            type='AP',
            label=f"{bssid_str} ({vendor_str})",
            vendor=vendor_str,
            cluster=int(row['cluster']),
            activity=int(ap_state.count),
            roaming_support=roaming_support,
            channel=int(row['channel']),
            rssi_mean=float(row['rssi_mean']),
            time_interval=interval_string,
            start=start_iso,
            end=end_iso
        )

    if include_clients:
        mac_to_cluster = {}
        if clustered_client_df is not None and not clustered_client_df.empty:
            mac_to_cluster = {
                mac: int(row['cluster']) 
                for _, row in clustered_client_df.iterrows() 
                for mac in str(row.get('original_macs', '')).split(', ') if mac
            }

        for mac, client_state in clients_to_export.items():
            # Convert timestamps to ISO format for Gephi timeline animation
            start_iso = datetime.fromtimestamp(client_state.first_seen).isoformat()
            end_iso = datetime.fromtimestamp(client_state.last_seen).isoformat()
            interval_string = f"<[{client_state.first_seen}, {client_state.last_seen}]>"
            behavior_features = features_for_client_behavior(client_state) or {}

            G.add_node(
                mac,
                type='Client',
                label=f"{mac} ({intelligent_vendor_lookup(mac, client_state) or 'N/A'})",
                vendor=intelligent_vendor_lookup(mac, client_state) or "Unknown",
                cluster=mac_to_cluster.get(mac, -1),
                activity=int(client_state.count),
                is_randomized=bool(client_state.randomized),
                time_interval=interval_string,
                start=start_iso,
                end=end_iso,
                avg_datarate=behavior_features.get("avg_mcs_rate", 0.0),
                power_save_rate=behavior_features.get("power_save_rate", 0.0),
                probed_ssid_count=behavior_features.get("probed_ssid_diversity", 0),
                avg_snr=behavior_features.get("avg_snr", 0.0),
                fcs_error_rate=behavior_features.get("fcs_error_rate", 0.0),
                os_fingerprints=behavior_features.get("os_fingerprint_diversity", 0)
            )
            
            for bssid in client_state.seen_with:
                if G.has_node(mac) and G.has_node(bssid):
                    rssi_dbm = float(client_state.rssi_w.mean if client_state.rssi_w.n > 0 else -100)
                    positive_weight = max(0.1, 100 + rssi_dbm)
                    # Add ISO timestamps for edge timeline animation
                    edge_start_iso = datetime.fromtimestamp(client_state.first_seen).isoformat()
                    edge_end_iso = datetime.fromtimestamp(client_state.last_seen).isoformat()
                    G.add_edge(mac, bssid, kind='Association', weight=positive_weight, 
                              start=edge_start_iso, end=edge_end_iso)
    return G


def _discover_attributes(G: nx.Graph) -> tuple[dict, dict]:
    """
    Analysiert den Graphen und erkennt automatisch alle Knoten- und Kantenattribute
    sowie deren Datentypen für die GEXF-Deklaration.
    """
    node_attrs, edge_attrs = {}, {}
    type_map = {int: "integer", float: "float", bool: "boolean", str: "string"}

    for _, attrs in G.nodes(data=True):
        for k, v in attrs.items():
            if k not in node_attrs and v is not None and k not in ['label', 'start', 'end']:
                node_attrs[k] = type_map.get(type(v), "string")

    for _, _, attrs in G.edges(data=True):
        for k, v in attrs.items():
            if k not in edge_attrs and v is not None and k not in ['weight', 'start', 'end', 'kind']:
                 edge_attrs[k] = type_map.get(type(v), "string")
    return node_attrs, edge_attrs


def write_gexf13_enhanced(G: nx.Graph, out_file: str):
    """
    Exportiert einen NetworkX-Graphen als standardkonforme GEXF 1.3-Datei.
    """
    gexf = ET.Element("gexf", {
        "xmlns": "http://www.gexf.net/1.3", "xmlns:viz": "http://www.gexf.net/1.3/viz",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.gexf.net/1.3 http://www.gexf.net/1.3/gexf.xsd",
        "version": "1.3"
    })
    meta = ET.SubElement(gexf, "meta", {"lastmodifieddate": str(date.today())})
    ET.SubElement(meta, "creator").text = "Wifi Analysis Export"
    graph = ET.SubElement(gexf, "graph", {
        "mode": G.graph.get("mode", "dynamic"),
        "timeformat": G.graph.get("timeformat", "datetime"),
        "defaultedgetype": "undirected"
    })

    node_attrs, edge_attrs = _discover_attributes(G)
    
    if node_attrs:
        attributes_nodes = ET.SubElement(graph, "attributes", {"class": "node", "mode": "static"})
        for idx, (title, attr_type) in enumerate(node_attrs.items()):
            ET.SubElement(attributes_nodes, "attribute", {"id": str(idx), "title": title, "type": attr_type})
    
    if edge_attrs:
        attributes_edges = ET.SubElement(graph, "attributes", {"class": "edge", "mode": "static"})
        for idx, (title, attr_type) in enumerate(edge_attrs.items()):
            ET.SubElement(attributes_edges, "attribute", {"id": str(idx), "title": title, "type": attr_type})

    nodes = ET.SubElement(graph, "nodes")
    for n_id, attrs in G.nodes(data=True):
        node_el = ET.SubElement(nodes, "node", {"id": str(n_id), "label": str(attrs.get("label", n_id))})
        if "start" in attrs: node_el.set("start", str(attrs['start']))
        if "end" in attrs: node_el.set("end", str(attrs['end']))
        attvalues_el = ET.SubElement(node_el, "attvalues")
        for idx, (title, _) in enumerate(node_attrs.items()):
            if title in attrs:
                value = attrs[title]
                if isinstance(value, bool): value = str(value).lower()
                ET.SubElement(attvalues_el, "attvalue", {"for": str(idx), "value": str(value)})

    edges = ET.SubElement(graph, "edges")
    for i, (u, v, attrs) in enumerate(G.edges(data=True)):
        edge_el = ET.SubElement(edges, "edge", {
            "id": str(i), "source": str(u), "target": str(v),
            "weight": str(attrs.get("weight", 1.0))
        })
        if "kind" in attrs: edge_el.set("label", str(attrs['kind']))
        if "start" in attrs: edge_el.set("start", str(attrs['start']))
        if "end" in attrs: edge_el.set("end", str(attrs['end']))

    tree = ET.ElementTree(gexf)
    ET.indent(tree, space="  ")
    tree.write(out_file, encoding="utf-8", xml_declaration=True)
    logger.info(f"GEXF 1.3 (enhanced) erfolgreich nach '{out_file}' exportiert.")


def export_ap_graph(
    state: WifiAnalysisState,
    clustered_ap_df: pd.DataFrame,
    aps_to_export: Dict[str, APState],
    clients_to_export: Dict[str, ClientState],
    out_file: str,
    include_clients: bool = False,
    clustered_client_df: Optional[pd.DataFrame] = None,
    out_format: str = "gexf"
) -> bool:
    """
    Hauptfunktion, die den Graphen baut und ihn dann im gewünschten Format exportiert.
    """
    if clustered_ap_df is None or clustered_ap_df.empty:
        logger.warning("Keine AP-Cluster-Daten für den Graph-Export verfügbar.")
        return False
    try:
        graph_to_export = _build_export_graph(
            state, clustered_ap_df, aps_to_export, clients_to_export, 
            include_clients, clustered_client_df
        )
        if out_format.lower() == "graphml":
            nx.write_graphml(graph_to_export, out_file)
            logger.info(f"GraphML erfolgreich nach '{out_file}' exportiert. Knoten: {graph_to_export.number_of_nodes()}, Kanten: {graph_to_export.number_of_edges()}.")
        else:
            write_gexf13_enhanced(graph_to_export, out_file)
        return True
    except Exception as e:
        logger.error("Fehler beim Exportieren des Graphen: %s", e, exc_info=True)
        return False

def export_to_csv_files(G: nx.Graph, outdir: Path):
    """
    Exportiert den Graphen als zwei separate CSV-Dateien (nodes.csv, edges.csv)
    für einen robusten Import in Gephi.
    """
    logger.info("Exportiere Graphen als CSV-Dateien für Gephi...")
    
    nodes_data = []
    for node_id, attrs in G.nodes(data=True):
        node_record = {'Id': node_id, 'Label': attrs.get('label', node_id)}
        
        start = attrs.get('start')
        end = attrs.get('end')
        if start is not None and end is not None:
            node_record['Interval'] = f"<[{start}, {end}]>"
            
        for key, value in attrs.items():
            if key not in ['label', 'start', 'end']:
                node_record[key] = value
        nodes_data.append(node_record)
        
    nodes_df = pd.DataFrame(nodes_data)
    nodes_file = outdir / "nodes.csv"
    nodes_df.to_csv(nodes_file, index=False)
    logger.info(f"Knoten-Datei erfolgreich gespeichert: {nodes_file}")

    edges_data = []
    for i, (source, target, attrs) in enumerate(G.edges(data=True)):
        edge_record = {
            'Source': source,
            'Target': target,
            'Type': 'Undirected',
            'Id': i,
            'Weight': attrs.get('weight', 1.0)
        }
        edges_data.append(edge_record)
        
    edges_df = pd.DataFrame(edges_data)
    edges_file = outdir / "edges.csv"
    edges_df.to_csv(edges_file, index=False)
    logger.info(f"Kanten-Datei erfolgreich gespeichert: {edges_file}")

    return True