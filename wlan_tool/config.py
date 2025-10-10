#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrale Konfiguration und Konstanten für das WLAN-Analyse-Tool.
Liest Konfiguration aus config.yaml.
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Lade Konfiguration aus YAML-Datei
_config_path = Path(__file__).parent.parent / "config.yaml"
with open(_config_path, 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)

# --- Sniffing & Channel Hopping ---
CHANNELS_TO_HOP: List[int] = _config['channels']['to_hop']
CHANNEL_HOP_SLEEP_S: float = _config['channels']['hop_sleep_s']

# --- Datenbank ---
DB_BATCH_SIZE: int = _config['database']['batch_size']
DB_FLUSH_INTERVAL_S: float = _config['database']['flush_interval_s']

# --- Heuristisches Scoring ---
HEURISTIC_WEIGHTS = _config['heuristic']['weights']
HEURISTIC_NORMALIZATION = _config['heuristic']['normalization']

# --- Analyse & Matching ---
SCORE_LABEL_HIGH: float = _config['analysis']['score_label_high']
SCORE_LABEL_MEDIUM: float = _config['analysis']['score_label_medium']
MULTI_SSID_THRESHOLD: int = _config['analysis']['multi_ssid_threshold']

# Gewichtung für die Kombination von Heuristik und ML-Modell
ML_SCORE_WEIGHT: float = _config['analysis']['ml_score_weight']
HEURISTIC_SCORE_WEIGHT: float = _config['analysis']['heuristic_score_weight']

# --- State Management ---
STATE_PRUNING_THRESHOLD_S: int = _config['state']['pruning_threshold_s']
STATE_PRUNING_INTERVAL_S: int = _config['state']['pruning_interval_s']

# --- Client Clustering ---
CLIENT_CLUSTERING_FEATURE_WEIGHTS = _config['clustering']['client_feature_weights']

# --- UI & Training ---
UI_DEFAULT_MIN_SCORE: float = _config['ui']['default_min_score']
UI_DEFAULT_MAX_SCORE: float = _config['ui']['default_max_score']
UI_MAX_CANDIDATES: int = _config['ui']['max_candidates']
AUTO_RETRAIN_MIN_CONFIRMED_LABELS: int = _config['ui']['auto_retrain_min_confirmed_labels']

# --- Client Klassifizierung ---
DEVICE_TYPES: List[str] = _config['device_types']

# --- Debugging ---
LOG_PACKET_FLOW: bool = _config['debug']['log_packet_flow']

# --- Capture Settings ---
DEFAULT_INTERFACE: str = _config['capture']['interface']
DEFAULT_DURATION: int = _config['capture']['duration']
DEFAULT_PCAP_FILE: str = _config['capture']['pcap_file']

# --- Database Settings ---
DEFAULT_DB_PATH: str = _config['database']['path']
DEFAULT_LABEL_DB_PATH: str = _config['database']['label_db']

# --- Adaptive Scan ---
ADAPTIVE_SCAN_ENABLED: bool = _config['scanning']['adaptive_scan_enabled']
ADAPTIVE_SCAN_DISCOVERY_S: int = _config['scanning']['adaptive_scan_discovery_s']
ADAPTIVE_SCAN_TOP_N_CHANNELS: int = _config['scanning']['adaptive_scan_top_n_channels']
ADAPTIVE_SCAN_PRIORITY_WEIGHT: int = _config['scanning']['adaptive_scan_priority_weight']

# --- Kanalbandbreite ---
BANDWIDTH_5GHZ: str = _config['scanning']['bandwidth_5ghz']
BANDWIDTH_2_4GHZ: str = _config['scanning']['bandwidth_2_4ghz']

def get_config() -> Dict[str, Any]:
    """Gibt die komplette Konfiguration als Dictionary zurück."""
    return _config

def reload_config():
    """Lädt die Konfiguration aus der YAML-Datei neu."""
    global _config
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)