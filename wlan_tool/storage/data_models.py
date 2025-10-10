#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrale Definitionen für alle Datenstrukturen und -modelle des Projekts.
Diese Datei ist die "Single Source of Truth" für die Form der Daten.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict, Set, Counter, Tuple
import math
from pathlib import Path

# --- 1. Enumeration für Event-Typen ---
class EventType(str, Enum):
    BEACON = "beacon"
    PROBE_RESP = "probe_resp"
    PROBE_REQ = "probe_req"
    DATA = "data"
    DNS_QUERY = "dns_query"
    ARP_MAP = "arp_map"
    DHCP_REQ = "dhcp_req"


# --- 2. Datenstruktur für ein einzelnes erfasstes Ereignis ---
class WifiEvent(TypedDict, total=False):
    ts: float
    type: EventType
    bssid: Optional[str]
    ssid: Optional[str]
    client: Optional[str]
    channel: Optional[int]
    rssi: Optional[int]
    ies: Dict[int, List[str]]
    seq: Optional[int]
    beacon_interval: Optional[int]
    cap: Optional[int]
    beacon_timestamp: Optional[int]
    is_powersave: bool
    dns_query: str
    hostname: str
    arp_mac: str
    arp_ip: str


# --- 3. Hilfsklasse für statistische Berechnungen ---
@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: Optional[float]) -> None:
        if x is None: return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def variance(self) -> float:
        return (self.M2 / (self.n - 1)) if self.n > 1 else 0.0

    def std(self) -> float:
        return math.sqrt(self.variance()) if self.n > 1 else 0.0

@dataclass
class APState:
    """Speichert den aggregierten Zustand eines Access Points."""
    bssid: str
    ssid: Optional[str] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 0
    beacon_count: int = 0
    probe_resp_count: int = 0
    channel: Optional[int] = None
    ies: Dict[int, List[str]] = field(default_factory=dict)
    parsed_ies: Dict[str, any] = field(default_factory=dict)
    rssi_w: Welford = field(default_factory=Welford)
    beacon_intervals: Counter = field(default_factory=Counter)
    cap_bits: Set[int] = field(default_factory=set)
    last_beacon_timestamp: Optional[int] = None
    uptime_seconds: Optional[float] = None
    # KORREKTUR: Fehlende RF-Attribute hinzugefügt
    noise_w: Welford = field(default_factory=Welford)
    fcs_error_count: int = 0

@dataclass
class ClientState:
    """Speichert den aggregierten Zustand eines Client-Geräts."""
    mac: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 0
    probes: Set[str] = field(default_factory=set)
    seen_with: Set[str] = field(default_factory=set)
    randomized: bool = False
    is_in_powersave: bool = False
    last_powersave_ts: float = 0.0
    parsed_ies: Dict[str, any] = field(default_factory=dict)
    all_packet_ts: List[float] = field(default_factory=list)
    rssi_w: Welford = field(default_factory=Welford)
    data_frame_count: int = 0
    mgmt_frame_count: int = 0
    power_save_transitions: int = 0
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    mcs_rates: Counter = field(default_factory=Counter)
    noise_w: Welford = field(default_factory=Welford)
    fcs_error_count: int = 0
    ie_order_hashes: Set[int] = field(default_factory=set)


# --- 5. Datenstruktur für Analyse-Ergebnisse ---
@dataclass
class InferenceResult:
    """Kapselt das Ergebnis der SSID-BSSID-Korrelation."""
    ssid: str
    bssid: str
    score: float
    label: str
    components: Dict[str, any] = field(default_factory=dict)
    evidence: Dict[str, any] = field(default_factory=dict)