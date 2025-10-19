# data/device_profiler.py

import hashlib
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import re
    import subprocess

    import upnpclient
    import yaml
except ImportError:
    upnpclient = None

try:
    import nmap
except ImportError:
    nmap = None
from .. import config

try:
    from fritzconnection import FritzConnection
    from fritzconnection.lib.fritzhosts import FritzHosts
except ImportError:
    FritzHosts = None
    FritzConnection = None

from scapy.all import ARP, Ether, arping, conf, srp

from .. import utils

logger = logging.getLogger(__name__)


def create_device_fingerprint(client_state) -> str:
    """
    Erstellt einen erweiterten Geräte-Fingerprint basierend auf:
    - IE-Reihenfolge (ie_order_hash)
    - Paket-Timing (all_packet_ts)
    - Technische Merkmale
    """
    if not client_state:
        return ""

    # Sammle verschiedene Fingerprint-Komponenten
    fingerprint_parts = []
    
    # Prüfe, ob der ClientState überhaupt Daten hat
    has_data = (
        client_state.ie_order_hashes or 
        len(client_state.all_packet_ts) > 0 or
        client_state.ssid_requests or
        client_state.bssid_requests
    )
    
    if not has_data:
        return ""

    # 1. IE-Reihenfolge Hash (bereits in client_state.ie_order_hashes)
    if client_state.ie_order_hashes:
        ie_hashes = sorted(list(client_state.ie_order_hashes))
        fingerprint_parts.append(f"ie_order:{hash(tuple(ie_hashes))}")

    # 2. Paket-Timing-Muster
    if len(client_state.all_packet_ts) > 1:
        intervals = np.diff(client_state.all_packet_ts)
        if len(intervals) > 0:
            # Charakteristische Timing-Merkmale
            timing_features = [
                f"mean_interval:{np.mean(intervals):.3f}",
                f"std_interval:{np.std(intervals):.3f}",
                f"min_interval:{np.min(intervals):.3f}",
                f"max_interval:{np.max(intervals):.3f}",
                f"burst_pattern:{len(intervals[intervals < 0.1])}",  # Kurze Intervalle (Bursts)
            ]
            fingerprint_parts.extend(timing_features)

    # 3. Technische Merkmale
    parsed_ies = client_state.parsed_ies
    if parsed_ies:
        tech_features = []

        # WiFi-Standards
        standards = parsed_ies.get("standards", [])
        if standards:
            tech_features.append(f"standards:{','.join(sorted(standards))}")

        # MIMO-Streams
        ht_caps = parsed_ies.get("ht_caps", {})
        if ht_caps.get("streams"):
            tech_features.append(f"mimo_streams:{ht_caps['streams']}")

        # Vendor-spezifische IEs
        vendor_ies = parsed_ies.get("vendor_specific", {})
        if vendor_ies:
            vendors = sorted(vendor_ies.keys())
            tech_features.append(f"vendors:{','.join(vendors)}")

        fingerprint_parts.extend(tech_features)

    # 4. Verhaltensmerkmale
    behavior_features = []

    # Probe-Pattern
    if client_state.probes:
        probe_count = len(client_state.probes)
        behavior_features.append(f"probe_count:{probe_count}")

        # Spezifische SSIDs (erste 3)
        specific_probes = [p for p in client_state.probes if p != "<broadcast>"][:3]
        if specific_probes:
            behavior_features.append(f"probes:{','.join(specific_probes)}")

    # Power Save Verhalten
    if client_state.last_powersave_ts > 0:
        behavior_features.append("powersave:true")

    # Randomisierte MAC
    if client_state.randomized:
        behavior_features.append("randomized_mac:true")

    fingerprint_parts.extend(behavior_features)

    # Kombiniere alle Teile zu einem Hash
    fingerprint_string = "|".join(sorted(fingerprint_parts))
    return hashlib.md5(fingerprint_string.encode()).hexdigest()


def classify_device_by_fingerprint(
    fingerprint: str, known_fingerprints: Dict[str, str]
) -> Optional[str]:
    """
    Klassifiziert ein Gerät basierend auf seinem Fingerprint.
    """
    return known_fingerprints.get(fingerprint)


def build_fingerprint_database(state) -> Dict[str, str]:
    """
    Baut eine Datenbank von Geräte-Fingerprints auf.
    """
    fingerprint_db = {}

    for mac, client_state in state.clients.items():
        fingerprint = create_device_fingerprint(client_state)
        if fingerprint:
            # Verwende den intelligenten Vendor-Lookup als Basis-Klassifizierung
            vendor = utils.intelligent_vendor_lookup(mac, client_state) or "Unknown"
            fingerprint_db[fingerprint] = vendor

    return fingerprint_db


def correlate_devices_by_fingerprint(state) -> Dict[str, List[str]]:
    """
    Korreliert Geräte basierend auf ähnlichen Fingerprints.
    """
    fingerprint_groups = defaultdict(list)

    for mac, client_state in state.clients.items():
        fingerprint = create_device_fingerprint(client_state)
        if fingerprint:
            fingerprint_groups[fingerprint].append(mac)

    # Filtere Gruppen mit mehr als einem Gerät
    correlated_groups = {
        fp: macs for fp, macs in fingerprint_groups.items() if len(macs) > 1
    }

    logger.info(
        f"Fand {len(correlated_groups)} Gruppen von Geräten mit identischen Fingerprints."
    )
    return correlated_groups


def load_device_rules() -> List[Dict]:
    """Lädt die Regex-Regeln aus der device_rules.yaml."""
    rules_path = "/home/pi/hacking/device_rules.yaml"
    try:
        with open(rules_path, "r") as f:
            rules_data = yaml.safe_load(f)
        return rules_data.get("device_rules", [])
    except (yaml.YAMLError, Exception) as e:
        logger.error(f"Fehler beim Laden von 'device_rules.yaml': {e}")
        return []


def classify_hostname(hostname: str, rules: List[Dict]) -> Optional[str]:
    """Wendet die Regex-Regeln auf einen Hostnamen an, um den Gerätetyp zu finden."""
    if not hostname or hostname == "N/A":
        return None

    for rule in rules:
        # re.IGNORECASE macht die Suche case-insensitive
        if re.search(rule["regex"], hostname, re.IGNORECASE):
            return rule["type"]
    return None  # Kein Treffer


def interactive_classify_unknowns(
    unknown_devices: Dict[str, Dict], console
) -> Dict[str, Dict]:
    """
    Startet eine interaktive Sitzung, um vom Benutzer Labels für unbekannte
    Hostnamen zu erhalten.
    """
    if not unknown_devices:
        return {}

    console.print(
        "\n[bold cyan]--- Interaktive Klassifizierung für unbekannte Hostnamen ---[/bold cyan]"
    )

    classified_devices = {}
    # Verwende eine Kopie, um sie während der Iteration zu verändern
    for mac, info in list(unknown_devices.items()):
        hostname = info["hostname"]
        console.print(
            f"\nUnbekannter Hostname gefunden: [bold yellow]{hostname}[/bold yellow] (MAC: {mac})"
        )

        # Biete dem Benutzer eine Liste von Standard-Typen an
        all_device_types = sorted(list(set(config.DEVICE_TYPES)))
        console.print(
            "[cyan]Gerätetyp auswählen (Zahl) oder neuen Typ eingeben:[/cyan]"
        )
        for i, dtype in enumerate(all_device_types):
            console.print(f"  {i}) {dtype}")
        console.print("  s) Überspringen")

        choice = console.input("> ").strip()
        selected_type = None

        if choice.lower() == "s":
            continue

        try:
            choice_index = int(choice)
            if 0 <= choice_index < len(all_device_types):
                selected_type = all_device_types[choice_index]
        except ValueError:
            new_label = choice.strip()
            if new_label:
                confirm = (
                    console.input(
                        f"Neuen Label '[bold]{new_label}[/bold]' verwenden? (y/n): "
                    )
                    .strip()
                    .lower()
                )
                if confirm == "y":
                    selected_type = new_label

        if selected_type:
            info["device_type"] = selected_type
            classified_devices[mac] = info
            console.print(
                f"[green]'{hostname}' als '{selected_type}' klassifiziert.[/green]"
            )

    return classified_devices


UPNP_HOST_SERVICES = [
    "urn:schemas-upnp-org:service:Hosts:1",
    "urn:dslforum-org:service:Hosts:1",
]

PORT_TO_DEVICE_TYPE = {
    21: "FTP-Server",  # FTP
    22: "Linux/SSH-Server",  # SSH
    23: "Telnet-Gerät",  # Telnet (oft IoT)
    25: "Mailserver (SMTP)",  # SMTP
    53: "DNS-Server",  # DNS
    80: "Web-Interface (HTTP)",  # HTTP
    110: "Mailserver (POP3)",  # POP3
    111: "NFS/RPC-Dienst",  # Portmapper/RPC
    123: "NTP-Zeitserver",  # Network Time Protocol
    135: "Windows-RPC",  # RPC Endpoint Mapper
    139: "Windows/Samba-Share",  # NetBIOS
    143: "Mailserver (IMAP)",  # IMAP
    161: "SNMP-Gerät",  # Netzwerkgeräte/IoT
    389: "LDAP-Verzeichnisdienst",  # LDAP
    443: "Web-Interface (HTTPS)",  # HTTPS
    445: "Windows/Samba-Share",  # SMB
    465: "SMTP (SSL)",  # Secure SMTP
    514: "Syslog-Server",  # Logging
    587: "SMTP (Submission)",  # Mail-Relay
    631: "CUPS-Drucker",  # Drucker
    993: "IMAP (SSL)",  # IMAPS
    995: "POP3 (SSL)",  # POP3S
    1433: "MSSQL-Datenbank",  # Microsoft SQL Server
    1521: "Oracle-Datenbank",  # Oracle
    1723: "VPN (PPTP)",  # VPN-Tunnel
    1883: "MQTT-Broker",  # IoT-Geräte (MQTT)
    1900: "UPnP-Gerät",  # SSDP/UPnP
    2049: "NFS-Dateisystem",  # NFS
    3306: "MySQL/MariaDB",  # Datenbank
    3389: "Windows (RDP)",  # Remote Desktop
    4433: "HTTPS-Alternative",  # Alternativer HTTPS-Port
    5000: "Synology/Plex",  # NAS/Plex
    5060: "VoIP (SIP)",  # Telefonanlagen
    5222: "XMPP/Jabber",  # Chat
    5353: "Apple-Gerät (mDNS)",  # Bonjour/mDNS
    5432: "PostgreSQL",  # Datenbank
    5900: "VNC-Server",  # Remote-Desktop
    5984: "CouchDB",  # Datenbank
    6379: "Redis-Server",  # In-Memory-DB
    8080: "Web-Interface (HTTP)",  # HTTP-Alternative
    8443: "Web-Interface (HTTPS)",  # HTTPS-Alternative
    9000: "SonarQube/Portainer",  # Admin-Tools
    9200: "Elasticsearch",  # Suche/DB
    10000: "Webmin-Interface",  # Server-Management
    27017: "MongoDB",  # NoSQL-Datenbank
    32400: "Plex Media Server",  # Streaming
    62078: "iPhone/iPad (lockdownd)",  # Apple iTunes-Dienst
}


# in data/device_profiler.py


def get_ips_for_macs_arp(network_interface: str) -> Dict[str, str]:
    """
    Sucht aktiv nach ALLEN Geräten im Netzwerk des angegebenen Interfaces
    mit dem 'arp-scan'-Tool und gibt ein MAC-zu-IP-Mapping zurück.
    """
    logger.info(
        f"Starte aktiven ARP-Scan auf Interface {network_interface}, um alle lokalen IPs zu finden..."
    )

    # Der Befehl arp-scan --localnet scannt das gesamte Subnetz des angegebenen Interfaces.
    # --quiet unterdrückt unnötige Ausgaben, --ignore-dups vermeidet doppelte Einträge.
    command = [
        "sudo",
        "arp-scan",
        "--localnet",
        "--quiet",
        "--interface",
        network_interface,
    ]

    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )

        mac_to_ip = {}
        # Parse die Ausgabe von arp-scan, die typischerweise so aussieht:
        # 192.168.178.1   00:1a:2b:3c:4d:5e   (Hersteller)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                ip_address = parts[0]
                mac_address = parts[1].upper()
                mac_to_ip[mac_address] = ip_address

        logger.info(
            f"ARP-Scan abgeschlossen. {len(mac_to_ip)} Geräte im lokalen Netzwerk gefunden."
        )
        return mac_to_ip

    except FileNotFoundError:
        logger.error(
            "Befehl 'arp-scan' nicht gefunden. Bitte installieren Sie es (sudo apt install arp-scan)."
        )
        return {}
    except subprocess.CalledProcessError as e:
        logger.error(f"arp-scan ist fehlgeschlagen. Fehler: {e.stderr.strip()}")
        return {}
    except subprocess.TimeoutExpired:
        logger.error("arp-scan hat zu lange gedauert und wurde abgebrochen.")
        return {}


def discover_routers() -> List:
    """Sucht im lokalen Netzwerk nach UPnP-fähigen Geräten (potenziellen Routern)."""
    if upnpclient is None:
        logger.warning(
            "Die 'upnpclient'-Bibliothek ist nicht installiert. Router-Scan nicht möglich."
        )
        return []

    try:
        # KORREKTUR: Erhöhe den Timeout und fange spezifischere Fehler ab.
        # Ein Timeout von 5 Sekunden gibt den Geräten mehr Zeit zum Antworten.
        devices = upnpclient.discover(timeout=10)

        if not devices:
            logger.info("Keine UPnP-Geräte im Netzwerk gefunden.")
            return []

        # Filtere nach Geräten, die wahrscheinlich Router sind
        routers = [
            dev
            for dev in devices
            if "InternetGatewayDevice" in dev.device_type
            and any(s in dev for s in UPNP_HOST_SERVICES)
        ]

        if not routers:
            logger.warning(
                "UPnP-Geräte gefunden, aber keines scheint ein kompatibler Router mit Host-Diensten zu sein."
            )

        return routers

    except Exception as e:
        # KORREKTUR: Gib eine aussagekräftigere Fehlermeldung aus.
        logger.error(
            f"Fehler bei der UPnP-Gerätesuche. Mögliche Ursachen: Netzwerk blockiert Multicast, Firewall aktiv oder kein UPnP-Router im Netz."
        )
        logger.debug(
            f"Ursprünglicher Fehler: {e!r}"
        )  # !r zeigt die Repräsentation des Fehlers
        return []

    try:
        devices = upnpclient.discover()
        routers = [
            dev
            for dev in devices
            if "InternetGatewayDevice" in dev.device_type
            and any(s in dev for s in UPNP_HOST_SERVICES)
        ]
        if not routers:
            logger.warning("Keine UPnP-fähigen Router mit Host-Diensten gefunden.")
        return routers
    except Exception as e:
        logger.error(f"Fehler bei der UPnP-Gerätesuche: {e}")
        return []


def get_connected_devices_from_router(router) -> Optional[Dict[str, Dict]]:
    """Fragt einen gegebenen Router nach der Liste der verbundenen Geräte ab."""
    if router is None:
        return None

    host_service = None
    for service_urn in UPNP_HOST_SERVICES:
        if service_urn in router:
            host_service = router[service_urn]
            break

    if not host_service:
        logger.error(
            f"Konnte keinen kompatiblen Host-Dienst auf dem Gerät {router.friendly_name} finden."
        )
        return None

    devices = {}
    try:
        if "GetHostNumberOfEntries" in host_service.actions:
            num_hosts = host_service.GetHostNumberOfEntries()["NewHostNumberOfEntries"]
            logger.info(
                f"Router '{router.friendly_name}' meldet {num_hosts} verbundene Geräte."
            )

            for i in range(num_hosts):
                try:
                    if "GetGenericHostEntry" in host_service.actions:
                        host = host_service.GetGenericHostEntry(NewIndex=i)
                    else:
                        continue

                    mac = host.get("NewMACAddress", "").upper()
                    if mac:
                        devices[mac] = {
                            "hostname": host.get("NewHostName", "N/A"),
                            "ip_address": host.get("NewIPAddress", "N/A"),
                            "source": f"upnp_{router.friendly_name}",
                        }
                except Exception as e_host:
                    logger.debug(f"Konnte Host-Eintrag {i} nicht abrufen: {e_host}")
                    continue
        else:
            logger.warning(
                "Router unterstützt nicht die 'GetHostNumberOfEntries'-Aktion. Kann Geräte nicht auflisten."
            )
            return None

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Geräteliste vom Router: {e}")
        return None

    return devices


def profile_devices_via_portscan(clients_with_ips: Dict[str, str]) -> Dict[str, Dict]:
    """
    Führt einen schnellen Nmap-Scan auf den Clients aus, um offene Ports zu finden
    und daraus den Gerätetyp abzuleiten.
    """
    if nmap is None:
        logger.warning(
            "Die 'python-nmap'-Bibliothek ist nicht installiert. Port-Scan nicht möglich."
        )
        return {}

    if not clients_with_ips:
        logger.info(
            "Keine Clients mit bekannten IP-Adressen für den Port-Scan gefunden."
        )
        return {}

    try:
        nm = nmap.PortScanner()
    except nmap.PortScannerError:
        logger.error(
            "Nmap wurde nicht im Systempfad gefunden. Bitte installieren Sie es (z.B. 'sudo apt install nmap')."
        )
        return {}

    ip_list = " ".join(clients_with_ips.values())
    logger.info(f"Führe einen schnellen Port-Scan auf den folgenden IPs aus: {ip_list}")
    try:
        nm.scan(hosts=ip_list, arguments="-T4 -F -Pn")
    except Exception as e:
        logger.error(f"Nmap-Scan ist fehlgeschlagen: {e}")
        return {}

    identified_devices = {}
    ip_to_mac = {ip: mac for mac, ip in clients_with_ips.items()}

    for host_ip in nm.all_hosts():
        if nm[host_ip].state() == "up":
            mac = ip_to_mac.get(host_ip)
            if not mac:
                continue

            device_type = "Unbekannt"
            open_ports = []

            if "tcp" in nm[host_ip]:
                open_ports = list(nm[host_ip]["tcp"].keys())
                for port in PORT_TO_DEVICE_TYPE:
                    if port in open_ports:
                        device_type = PORT_TO_DEVICE_TYPE[port]
                        break

            identified_devices[mac] = {
                "hostname": nm[host_ip].hostname() or "N/A",
                "ip_address": host_ip,
                "device_type": device_type,
                "open_ports": open_ports,
                "source": "nmap_portscan",
            }

    return identified_devices


def get_devices_from_fritzbox_tr064(
    address="192.168.178.1", password=None
) -> Optional[Dict[str, Dict]]:
    """
    Fragt eine FRITZ!Box über die TR-064-Schnittstelle nach verbundenen Geräten.
    Benötigt möglicherweise ein Passwort.
    """
    if FritzHosts is None:
        logger.warning(
            "Die 'fritzconnection'-Bibliothek ist nicht installiert. TR-064-Scan nicht möglich."
        )
        return None

    logger.info(
        f"Versuche, eine TR-064-Verbindung zur FRITZ!Box unter {address} herzustellen..."
    )

    try:
        # Wenn kein Passwort angegeben ist, wird versucht, ohne Authentifizierung zu verbinden
        fc = FritzHosts(address=address, password=password)
    except Exception as e:
        logger.warning(
            f"Verbindung zur FRITZ!Box TR-064-Schnittstelle fehlgeschlagen: {e}"
        )
        logger.warning(
            "Möglicherweise ist TR-064 deaktiviert oder es wird ein Passwort benötigt."
        )
        return None

    devices = {}
    try:
        hosts = fc.get_hosts_info()
        logger.info(f"TR-064: {len(hosts)} Geräte von der FRITZ!Box erhalten.")

        for host in hosts:
            mac = host.get("mac", "").upper()
            if mac:
                devices[mac] = {
                    "hostname": host.get("name", "N/A"),
                    "ip_address": host.get("ip", "N/A"),
                    "source": "tr064_fritzbox",
                }
        return devices
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Host-Liste via TR-064: {e}")
        return None
