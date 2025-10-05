#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrale Konfiguration und Konstanten für das WLAN-Analyse-Tool.
"""
from typing import List

# --- Sniffing & Channel Hopping ---
CHANNELS_TO_HOP: List[int] = [
    1, 6, 11, 36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 112, 
    116, 132, 140, 149, 153, 157, 161, 165
]
CHANNEL_HOP_SLEEP_S: float = 0.3

# --- Datenbank ---
DB_BATCH_SIZE: int = 512
DB_FLUSH_INTERVAL_S: float = 5.0

# --- Heuristisches Scoring ---
HEURISTIC_WEIGHTS = {
    "beacon": 0.30, "probe": 0.20, "client": 0.25,
    "rssi": 0.10, "channel": 0.10, "seq": 0.05,
}
HEURISTIC_NORMALIZATION = {
    "beacon_count": 5.0, "probe_resp_count": 5.0, "supporting_clients": 4.0,
    "rssi_std_max": 20.0, "seq_support": 3.0,
}

# --- Analyse & Matching ---
SCORE_LABEL_HIGH: float = 0.75
SCORE_LABEL_MEDIUM: float = 0.40
MULTI_SSID_THRESHOLD: int = 2

# Gewichtung für die Kombination von Heuristik und ML-Modell
ML_SCORE_WEIGHT: float = 0.60
HEURISTIC_SCORE_WEIGHT: float = 0.40

# --- State Management ---
STATE_PRUNING_THRESHOLD_S: int = 7200 # 2 Stunde (vorher 600)
STATE_PRUNING_INTERVAL_S: int = 1200  # Alle 10 Minuten (vorher 300)

# --- Client Clustering ---
CLIENT_CLUSTERING_FEATURE_WEIGHTS = {
    "supports_11ax": 3.0, "supports_11ac": 2.0, "has_apple_ie": 2.5,
    "has_ms_ie": 2.0, "mimo_streams": 1.5, "is_randomized_mac": 1.2,
    "was_in_powersave": 1.2, "probe_count": 1.0, "seen_with_ap_count": 1.0,
}

# --- UI & Training ---
UI_DEFAULT_MIN_SCORE: float = 0.4
UI_DEFAULT_MAX_SCORE: float = 0.75
UI_MAX_CANDIDATES: int = 200
AUTO_RETRAIN_MIN_CONFIRMED_LABELS: int = 50

# --- Client Klassifizierung ---
DEVICE_TYPES: List[str] = [
]

LOG_PACKET_FLOW: bool = False

# ==============================================================================
# NEU: Konfiguration für adaptives Scannen und Kanalbandbreite
# ==============================================================================

# --- Adaptiver Scan ---
# Wenn True, wird ein zweistufiger Scan durchgeführt (Discovery -> Targeted)
ADAPTIVE_SCAN_ENABLED: bool = True
# Dauer der initialen Discovery-Phase in Sekunden
ADAPTIVE_SCAN_DISCOVERY_S: int = 60
# Anzahl der besten Kanäle, die für die gezielte Phase priorisiert werden sollen
ADAPTIVE_SCAN_TOP_N_CHANNELS: int = 5
# Wie stark die Top-Kanäle gewichtet werden (z.B. 3 = sie werden 3x so oft gescannt)
ADAPTIVE_SCAN_PRIORITY_WEIGHT: int = 3

# --- Kanalbandbreite ---
# Legt die zu verwendende Bandbreite für die jeweiligen Bänder fest.
# Gültige Werte für 5 GHz: "20MHz", "40MHz", "80MHz", "160MHz"
# Gültige Werte für 2.4 GHz: "HT20", "HT40"
# Hinweis: Nicht alle WLAN-Karten unterstützen alle Modi. Das Skript hat einen Fallback.
BANDWIDTH_5GHZ: str = "80MHz"
BANDWIDTH_2_4GHZ: str = "HT40"