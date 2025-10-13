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
    
    def combine(self, other: 'Welford') -> 'Welford':
        """Kombiniert zwei Welford-Statistiken."""
        if self.n == 0:
            return other
        if other.n == 0:
            return self
            
        combined = Welford()
        combined.n = self.n + other.n
        combined.mean = (self.n * self.mean + other.n * other.mean) / combined.n
        
        # Kombiniere M2-Werte
        delta1 = self.mean - combined.mean
        delta2 = other.mean - combined.mean
        combined.M2 = self.M2 + other.M2 + self.n * delta1 * delta1 + other.n * delta2 * delta2
        
        return combined

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
    
    def update_from_beacon(self, ev: dict, detailed_ies: bool = False):
        """Aktualisiert den AP-State basierend auf einem Beacon-Event."""
        from ..utils import parse_ies
        
        ts = ev.get("ts", 0.0)
        self.last_seen = ts
        self.count += 1
        self.beacon_count += 1
        
        # RSSI und Noise aktualisieren
        if rssi := ev.get("rssi"):
            self.rssi_w.update(rssi)
        if noise := ev.get("noise"):
            self.noise_w.update(noise)
            
        # Channel aktualisieren
        if channel := ev.get("channel"):
            self.channel = channel
            
        # SSID aktualisieren
        if ssid := ev.get("ssid"):
            if ssid != "<hidden>":
                self.ssid = ssid
                
        # IEs verarbeiten
        if ies := ev.get("ies"):
            for k, arr in ies.items():
                current_ies = self.ies.setdefault(int(k), [])
                current_ies.extend(v for v in arr if v not in current_ies)
            self.parsed_ies = parse_ies(self.ies, detailed=detailed_ies)
            
        # Beacon-spezifische Daten
        if bi := ev.get("beacon_interval"):
            self.beacon_intervals[bi] += 1
        if cap := ev.get("cap"):
            self.cap_bits.add(cap)
        if bt := ev.get("beacon_timestamp"):
            self.last_beacon_timestamp = bt
            self.uptime_seconds = bt / 1_000_000.0
            
        # FCS-Fehler
        if ev.get("fcs_error"):
            self.fcs_error_count += 1

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
    
    def update_from_event(self, ev: dict, detailed_ies: bool = False):
        """Aktualisiert den Client-State basierend auf einem Event."""
        from ..utils import parse_ies, is_local_admin_mac
        
        ts = ev.get("ts", 0.0)
        ev_type = ev.get("type")
        
        # Grundlegende Updates
        if self.first_seen == 0.0:
            self.first_seen = ts
        self.last_seen = ts
        self.count += 1
        
        # RSSI und Noise aktualisieren
        if rssi := ev.get("rssi"):
            self.rssi_w.update(rssi)
        if noise := ev.get("noise"):
            self.noise_w.update(noise)
            
        # FCS-Fehler
        if ev.get("fcs_error"):
            self.fcs_error_count += 1
            
        # IE-Order-Hash
        if ie_hash := ev.get("ie_order_hash"):
            self.ie_order_hashes.add(ie_hash)
            
        # Event-spezifische Updates
        if ev_type == "probe_req":
            self.mgmt_frame_count += 1
            if ies := ev.get("ies", {}):
                self.parsed_ies = parse_ies(ies, detailed=detailed_ies)
                # Extrahiere SSIDs aus den IEs
                if ssid_ies := ies.get(0, []):
                    for ssid in ssid_ies:
                        if ssid and ssid.strip():
                            self.probes.add(ssid.strip())
        elif ev_type == "data":
            self.data_frame_count += 1
            if mcs := ev.get("mcs_index"):
                self.mcs_rates[mcs] += 1
                
            # Power-Save-Detection
            is_ps = ev.get("is_powersave", False)
            if is_ps != self.is_in_powersave:
                self.is_in_powersave = is_ps
                self.power_save_transitions += 1
                if is_ps:
                    self.last_powersave_ts = ts
                    
            # BSSID-Tracking
            if bssid := ev.get("bssid"):
                self.seen_with.add(bssid)
                
        # IP und Hostname
        if ip := ev.get("ip_address"):
            self.ip_address = ip
        if hostname := ev.get("hostname"):
            self.hostname = hostname


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