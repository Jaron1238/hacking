#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device Profiler für automatische Geräteerkennung.
"""

import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def get_devices_from_fritzbox_tr064(password: Optional[str] = None) -> Dict:
    """Holt Geräte-Informationen von einer FRITZ!Box via TR-064."""
    try:
        from fritzconnection import FritzConnection
        from fritzconnection.lib.fritzhosts import FritzHosts
        
        if not password:
            logger.warning("Kein FRITZ!Box-Passwort angegeben")
            return {}
        
        fc = FritzConnection(password=password)
        fh = FritzHosts(fc)
        
        devices = {}
        hosts = fh.get_hosts_info()
        
        for host in hosts:
            mac = host.get('mac')
            if mac:
                devices[mac] = {
                    'hostname': host.get('name', ''),
                    'ip_address': host.get('ip', ''),
                    'device_type': 'unknown',
                    'source': 'fritzbox_tr064'
                }
        
        return devices
    except Exception as e:
        logger.error(f"Fehler beim TR-064 Zugriff: {e}")
        return {}


def get_ips_for_macs_arp(interface: str) -> Dict:
    """Ermittelt IP-Adressen für MAC-Adressen via ARP."""
    try:
        import subprocess
        result = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            check=False
        )
        
        devices = {}
        for line in result.stdout.split('\n'):
            if '(' in line and ')' in line:
                parts = line.split()
                if len(parts) >= 4:
                    ip = parts[1].strip('()')
                    mac = parts[3]
                    devices[mac] = {
                        'ip_address': ip,
                        'device_type': 'unknown',
                        'source': 'arp'
                    }
        
        return devices
    except Exception as e:
        logger.error(f"Fehler beim ARP-Scan: {e}")
        return {}


def profile_devices_via_portscan(devices: Dict) -> Dict:
    """Profiliert Geräte via Port-Scan."""
    # Placeholder implementation
    return devices


def load_device_rules() -> Dict:
    """Lädt Geräte-Klassifizierungsregeln."""
    return {
        'hostname_patterns': {
            'iphone': 'smartphone',
            'android': 'smartphone',
            'laptop': 'laptop',
            'desktop': 'desktop'
        }
    }


def classify_hostname(hostname: Optional[str], rules: Dict) -> Optional[str]:
    """Klassifiziert ein Gerät basierend auf dem Hostname."""
    if not hostname:
        return None
    
    hostname_lower = hostname.lower()
    patterns = rules.get('hostname_patterns', {})
    
    for pattern, device_type in patterns.items():
        if pattern in hostname_lower:
            return device_type
    
    return None


def interactive_classify_unknowns(unknown_devices: Dict, console) -> Dict:
    """Interaktive Klassifizierung unbekannter Geräte."""
    classified = {}
    
    for mac, info in unknown_devices.items():
        console.print(f"Unbekanntes Gerät: {mac}")
        console.print(f"  Hostname: {info.get('hostname', 'N/A')}")
        console.print(f"  IP: {info.get('ip_address', 'N/A')}")
        
        device_type = console.input("Gerätetyp eingeben (oder 'skip'): ").strip()
        if device_type and device_type != 'skip':
            info['device_type'] = device_type
            classified[mac] = info
    
    return classified


def correlate_devices_by_fingerprint(state) -> List[List[str]]:
    """Korreliert Geräte basierend auf ähnlichen Fingerprints."""
    fingerprints = {}
    
    # Erstelle Fingerprints für alle Clients
    for mac, client in state.clients.items():
        fingerprints[mac] = create_device_fingerprint_dict(client)
    
    # Einfache Korrelation basierend auf IE-Order-Hashes
    groups = []
    processed = set()
    
    for mac1, fp1 in fingerprints.items():
        if mac1 in processed:
            continue
            
        group = [mac1]
        processed.add(mac1)
        
        for mac2, fp2 in fingerprints.items():
            if mac2 in processed:
                continue
                
            # Prüfe Ähnlichkeit
            if (fp1.get('ie_order_hashes') and fp2.get('ie_order_hashes') and
                set(fp1['ie_order_hashes']) & set(fp2['ie_order_hashes'])):
                group.append(mac2)
                processed.add(mac2)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups


def build_fingerprint_database(state) -> Dict[str, str]:
    """Baut eine Fingerprint-Datenbank aus dem State auf."""
    fingerprint_db = {}
    
    for mac, client in state.clients.items():
        fingerprint = create_device_fingerprint(client)
        
        # Verwende Vendor-Lookup als Label
        from ..utils import lookup_vendor
        vendor = lookup_vendor(mac) or "Unknown"
        
        # Verwende den Fingerprint-Dict als String-Key
        fp_key = str(hash(str(fingerprint)))
        fingerprint_db[fp_key] = vendor
    
    return fingerprint_db


def create_device_fingerprint(client_state) -> str:
    """Erstellt einen Device-Fingerprint-Hash für einen Client."""
    import hashlib
    
    # Prüfe ob ClientState leer ist
    if (not client_state.probes and not client_state.seen_with and 
        client_state.data_frame_count == 0 and client_state.mgmt_frame_count == 0):
        return hashlib.md5(f"empty:{client_state.mac}".encode()).hexdigest()
    
    fingerprint_data = {
        'randomized': client_state.randomized,
        'probe_count': len(client_state.probes),
        'seen_with_count': len(client_state.seen_with),
        'data_frames': client_state.data_frame_count,
        'mgmt_frames': client_state.mgmt_frame_count,
        'power_save_transitions': client_state.power_save_transitions,
        'ie_order_hashes': sorted(list(client_state.ie_order_hashes)),
        'parsed_ies_keys': sorted(list(client_state.parsed_ies.keys()))
    }
    
    # Erstelle Hash aus den Fingerprint-Daten
    fingerprint_str = str(sorted(fingerprint_data.items()))
    return hashlib.md5(fingerprint_str.encode()).hexdigest()


def create_device_fingerprint_dict(client_state) -> Dict:
    """Erstellt einen Device-Fingerprint als Dictionary für einen Client."""
    fingerprint = {
        'mac': client_state.mac,
        'randomized': client_state.randomized,
        'probe_count': len(client_state.probes),
        'seen_with_count': len(client_state.seen_with),
        'data_frames': client_state.data_frame_count,
        'mgmt_frames': client_state.mgmt_frame_count,
        'power_save_transitions': client_state.power_save_transitions,
        'ie_order_hashes': list(client_state.ie_order_hashes),
        'parsed_ies': client_state.parsed_ies
    }
    
    return fingerprint


def classify_device_by_fingerprint(fingerprint_key: str, known_fingerprints: Dict[str, str]) -> Optional[str]:
    """Klassifiziert ein Gerät basierend auf seinem Fingerprint-Key."""
    return known_fingerprints.get(fingerprint_key)


def classify_device_by_features(fingerprint: Dict) -> str:
    """Klassifiziert ein Gerät basierend auf seinen Features."""
    # Einfache Heuristiken für Geräteklassifizierung
    if fingerprint.get('randomized', False):
        if fingerprint.get('probe_count', 0) > 5:
            return 'mobile_device'
        else:
            return 'iot_device'
    
    # Basierend auf Aktivität
    data_frames = fingerprint.get('data_frames', 0)
    mgmt_frames = fingerprint.get('mgmt_frames', 0)
    
    if data_frames > 100:
        return 'laptop'
    elif mgmt_frames > 50:
        return 'smartphone'
    else:
        return 'unknown'