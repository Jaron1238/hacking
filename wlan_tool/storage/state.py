# data/state.py

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterable
from dataclasses import dataclass, field
import math

from .. import utils
from .data_models import APState, ClientState, Welford, EventType
from ..utils import is_local_admin_mac, parse_ies, is_valid_bssid

logger = logging.getLogger(__name__)

# KORREKTUR: Definiere benannte Funktionen auf der obersten Ebene des Moduls
# als Ersatz für die Lambdas.

def _create_default_sources_entry():
    """Erstellt ein defaultdict(set) für die 'sources' in ssid_map."""
    return defaultdict(set)

def _create_default_ssid_map_entry():
    """Erstellt einen Standardeintrag für das ssid_map defaultdict."""
    return {"bssids": set(), "sources": _create_default_sources_entry()}

def _create_default_seq_local_entry():
    """Erstellt einen Standardeintrag für das seq_local defaultdict."""
    return {"last_seq": None, "monotonic_count": 0}


class WifiAnalysisState:
    """Kapselt den gesamten Analyse-Zustand."""
    def __init__(self):
        self.aps: Dict[str, APState] = {}
        self.clients: Dict[str, ClientState] = {}
        # KORREKTUR: Verwende die benannten Funktionen anstelle der Lambdas
        self.ssid_map: Dict[str, Dict] = defaultdict(_create_default_ssid_map_entry)
        self.seq_local: Dict[Tuple[str, str], Dict] = defaultdict(_create_default_seq_local_entry)
        self.clients_probing_ssid: Dict[str, Set[str]] = defaultdict(set)
        self.clients_seen_with_bssid: Dict[str, Set[str]] = defaultdict(set)

    def build_from_events(self, ev_iterator: Iterable[Dict], detailed_ies: bool = False):
        """Akzeptiert einen Iterator, um Speicher zu sparen."""
        logger.info("Baue/Aktualisiere Zustand aus Event-Stream auf...")
        count = 0
        for ev in ev_iterator:
            self.update_from_event(ev, detailed_ies=detailed_ies)
            count += 1
        logger.info("Zustand aus %d Events aufgebaut/aktualisiert: APs=%d, Clients=%d, SSIDs=%d", count, len(self.aps), len(self.clients), len(self.ssid_map))


    def update_from_event(self, ev: dict, detailed_ies: bool = False):
        ts, ev_type = ev["ts"], ev.get("type")
        
        client_mac = ev.get("client")
        if client_mac:
            if client_mac not in self.clients:
                self.clients[client_mac] = ClientState(mac=client_mac, first_seen=ts, randomized=is_local_admin_mac(client_mac))
            client = self.clients[client_mac]
            client.last_seen, client.count = ts, client.count + 1
            client.all_packet_ts.append(ts)
            client.rssi_w.update(ev.get("rssi"))
            client.noise_w.update(ev.get("noise"))
            if ev.get("fcs_error"):
                client.fcs_error_count += 1
            if ev.get("ie_order_hash"):
                client.ie_order_hashes.add(ev['ie_order_hash'])

        bssid = ev.get("bssid")
        if bssid and not is_valid_bssid(bssid): return
        if bssid and bssid not in self.aps:
            self.aps[bssid] = APState(bssid=bssid, first_seen=ts)
        
        if ev_type in ("beacon", "probe_resp"):
            ap = self.aps[bssid]
            ap.last_seen, ap.count = ts, ap.count + 1
            ap.rssi_w.update(ev.get("rssi"))
            ap.noise_w.update(ev.get("noise"))
            if ev.get("fcs_error"):
                ap.fcs_error_count += 1
            if ch := ev.get("channel"): ap.channel = ch
            if ssid := ev.get("ssid"):
                if ssid != "<hidden>": ap.ssid = ssid
                self.ssid_map[ssid]["bssids"].add(bssid)
            if ies := ev.get("ies"):
                for k, arr in ies.items():
                    current_ies = ap.ies.setdefault(int(k), [])
                    current_ies.extend(v for v in arr if v not in current_ies)
                ap.parsed_ies = parse_ies(ap.ies, detailed=detailed_ies)
            if ev_type == "beacon":
                ap.beacon_count += 1
                if bi := ev.get("beacon_interval"): ap.beacon_intervals[bi] += 1
                if cap := ev.get("cap"): ap.cap_bits.add(cap)
                if bt := ev.get("beacon_timestamp"):
                    ap.last_beacon_timestamp = bt
                    ap.uptime_seconds = bt / 1_000_000.0
                if ssid: self.ssid_map[ssid]["sources"]["beacon"].add(bssid)
            else:
                ap.probe_resp_count += 1
                if ssid: self.ssid_map[ssid]["sources"]["probe_resp"].add(bssid)
        
        elif ev_type == "probe_req":
            client = self.clients[client_mac]
            client.mgmt_frame_count += 1
            if ies := ev.get("ies", {}):
                client.parsed_ies = parse_ies(ies, detailed=detailed_ies)
                if probes := ies.get("probes"):
                    new_probes = set(probes) - client.probes
                    client.probes.update(new_probes)
                    for s in new_probes: 
                        if s:
                            self.ssid_map[s]["sources"]["probe_req"].add(client_mac)
                            self.clients_probing_ssid[s].add(client_mac)
        
        elif ev_type == "data":
            client = self.clients[client_mac]
            client.data_frame_count += 1
            if (mcs := ev.get("mcs_index")) is not None:
                client.mcs_rates[mcs] += 1
            is_ps = ev.get("is_powersave", False)
            if is_ps != client.is_in_powersave:
                client.power_save_transitions += 1
            client.is_in_powersave = is_ps
            if is_ps:
                client.last_powersave_ts = ts
            if bssid:
                if bssid not in client.seen_with:
                    client.seen_with.add(bssid)
                    self.clients_seen_with_bssid[bssid].add(client_mac)
                ap = self.aps[bssid]
                ap.last_seen = ts
                ap.count += 1
                ap.rssi_w.update(ev.get("rssi"))
                for s in client.probes:
                    if s:
                        self.ssid_map[s]["bssids"].add(bssid)
                        self.ssid_map[s]["sources"]["client_probe_assoc"].add(bssid)
                key, seq = (client_mac, bssid), ev.get("seq")
                if seq is not None:
                    rec = self.seq_local[key]
                    if (last_seq := rec.get("last_seq")) is not None:
                        diff = (seq - last_seq) if seq >= last_seq else (seq + 4096 - last_seq)
                        if 0 < diff <= 1000:
                            rec["monotonic_count"] = rec.get("monotonic_count", 0) + 1
                    rec["last_seq"] = seq
        
        elif ev_type == 'dhcp_req':
            if client_mac in self.clients: self.clients[client_mac].hostname = ev.get('hostname')
        elif ev_type == 'arp_map':
            mac = ev.get('arp_mac')
            if mac in self.clients: self.clients[mac].ip_address = ev.get('arp_ip')

    def prune_state(self, current_ts: float, threshold_s: int) -> int:
        stale_bssids = {bssid for bssid, ap in self.aps.items() if (current_ts - ap.last_seen) > threshold_s}
        if stale_bssids:
            for bssid in stale_bssids:
                del self.aps[bssid]
                if bssid in self.clients_seen_with_bssid: del self.clients_seen_with_bssid[bssid]
        stale_clients = {mac for mac, client in self.clients.items() if (current_ts - client.last_seen) > threshold_s}
        if stale_clients:
            for mac in stale_clients:
                del self.clients[mac]
        stale_seq_keys = {key for key in self.seq_local if key[0] in stale_clients or key[1] in stale_bssids}
        if stale_seq_keys:
            for key in stale_seq_keys:
                del self.seq_local[key]
        stale_ssids = set()
        all_active_bssids = set(self.aps.keys())
        all_active_clients = set(self.clients.keys())
        for ssid, sources_info in list(self.ssid_map.items()):
            sources_info["bssids"].intersection_update(all_active_bssids)
            for source_type, mac_set in list(sources_info["sources"].items()):
                mac_set.intersection_update(all_active_clients | all_active_bssids)
                if not mac_set: del sources_info["sources"][source_type]
            if not sources_info["bssids"] and not sources_info["sources"]:
                stale_ssids.add(ssid)
        if stale_ssids:
            for ssid in stale_ssids:
                del self.ssid_map[ssid]
                if ssid in self.clients_probing_ssid: del self.clients_probing_ssid[ssid]
        total_pruned = len(stale_bssids) + len(stale_clients) + len(stale_seq_keys) + len(stale_ssids)
        if total_pruned > 0:
            logger.info(
                "Zustand bereinigt: %d APs, %d Clients, %d Seq-Einträge und %d SSIDs entfernt.",
                len(stale_bssids), len(stale_clients), len(stale_seq_keys), len(stale_ssids)
            )
        return total_pruned