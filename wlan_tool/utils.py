# data/utils.py

import csv
import hashlib
import logging
import typing
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

if typing.TYPE_CHECKING:
    from .storage.state import ClientState

logger = logging.getLogger(__name__)

# MEHRERE ZUVERLÄSSIGE QUELLEN FÜR OUI-DATEN
OUI_SOURCES = [
    {
        "name": "IEEE Official",
        "url": "https://standards-oui.ieee.org/oui/oui.txt",
        "format": "ieee",
    },
    {
        "name": "Nmap GitHub",
        "url": "https://raw.githubusercontent.com/nmap/nmap/master/nmap-mac-prefixes",
        "format": "nmap",
    },
    {
        "name": "Wireshark GitLab",
        "url": "https://gitlab.com/wireshark/wireshark/-/raw/master/manuf",
        "format": "wireshark",
    },
    {
        "name": "Wireshark SVN",
        "url": "https://anonsvn.wireshark.org/wireshark/trunk/manuf",
        "format": "wireshark",
    },
]
OUI_LOCAL_PATH = Path(__file__).parent / "assets" / "manuf"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def download_oui_file(force_update: bool = False, progress_callback=None):
    """
    Lädt die OUI-Herstellerdatei von der bestmöglichen Quelle herunter.
    
    Args:
        force_update: Erzwingt Download auch wenn Datei bereits existiert
        progress_callback: Callback-Funktion für Download-Progress
    """
    # Prüfe ob Update nötig ist
    if not force_update and OUI_LOCAL_PATH.exists():
        file_age = time.time() - OUI_LOCAL_PATH.stat().st_mtime
        if file_age < 7 * 24 * 3600:  # 7 Tage
            logger.info(f"OUI-Datei ist aktuell (Alter: {file_age/3600:.1f}h)")
            return True
    
    # Erstelle Verzeichnis falls nötig
    OUI_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    for i, source in enumerate(OUI_SOURCES):
        logger.info(f"[{i+1}/{len(OUI_SOURCES)}] Download: {source['name']}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "text/plain,text/html,*/*",
            }
            
            req = urllib.request.Request(source["url"], headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                content_length = response.headers.get('Content-Length')
                total_size = int(content_length) if content_length else None
                
                downloaded = 0
                data_chunks = []
                
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    
                    data_chunks.append(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)
                
                data = b''.join(data_chunks)
                
                if len(data) < 1000:
                    raise ValueError(f"Download zu klein: {len(data)} bytes")
                
                # Backup alte Datei
                if OUI_LOCAL_PATH.exists():
                    backup_path = OUI_LOCAL_PATH.with_suffix('.bak')
                    OUI_LOCAL_PATH.rename(backup_path)
                
                # Neue Datei schreiben
                with open(OUI_LOCAL_PATH, "wb") as out_file:
                    out_file.write(data)
                
                # Metadaten speichern
                (OUI_LOCAL_PATH.parent / ".manuf_format").write_text(source["format"])
                
                logger.info(f"✓ OUI-Datei heruntergeladen: {len(data):,} bytes")
                return True
                
        except urllib.error.HTTPError as e:
            logger.warning(f"HTTP-Fehler bei {source['name']}: {e.code}")
        except Exception as e:
            logger.warning(f"Fehler bei {source['name']}: {type(e).__name__}")
        
        if i < len(OUI_SOURCES) - 1:
            time.sleep(1)
    
    logger.error("✗ Alle OUI-Download-Quellen fehlgeschlagen")
    return False


def _parse_ieee_format(file_path: Path) -> Dict[str, str]:
    """Parser für das IEEE OUI.txt Format."""
    mapping = {}
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "(hex)" not in line:
                continue
            try:
                if "(hex)" in line and "\t" in line:
                    hex_part = line.split("(hex)")[0].strip()
                    vendor_part = line.split("\t")[-1].strip()
                    if hex_part and vendor_part:
                        mac_prefix = hex_part.replace("-", ":").upper()
                        mapping[mac_prefix] = vendor_part
            except (ValueError, IndexError):
                continue
    return mapping

def _parse_nmap_format(file_path: Path) -> Dict[str, str]:
    """Parser für das 'nmap-mac-prefixes' Format (z.B. '000000 Apple')."""
    mapping = {}
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mac_hex, vendor = parts
                if len(mac_hex) == 6:
                    pref_norm = f"{mac_hex[0:2].upper()}:{mac_hex[2:4].upper()}:{mac_hex[4:6].upper()}"
                    mapping[pref_norm] = vendor
    return mapping


def _parse_wireshark_format(file_path: Path) -> Dict[str, str]:
    """Parser für das Wireshark 'manuf' Format (z.B. '00:00:01\tXerox')."""
    mapping = {}
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                mac_prefix, rest = parts
                vendor = rest.split("#", 1)[0].strip()
                if vendor:
                    prefix_parts = mac_prefix.split(":")
                    if len(prefix_parts) >= 3:
                        pref_norm = f"{prefix_parts[0].upper()}:{prefix_parts[1].upper()}:{prefix_parts[2].upper()}"
                        mapping[pref_norm] = vendor
    return mapping


def load_config(profile: Optional[str] = None) -> Dict:
    """Lädt die Konfiguration aus config.yaml und wendet ein Profil an."""
    # Default config
    default_config = {
        "capture": {"interface": "mon0", "duration": 120},
        "database": {"path": "events.db"}
    }
    
    # Versuche, die config.yaml im aktuellen Verzeichnis oder im Projekt-Stammverzeichnis zu finden
    config_path = Path.cwd() / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        logger.warning("Keine 'config.yaml'-Datei gefunden. Verwende Standardwerte.")
        return default_config

    try:
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or default_config
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in base_config:
                base_config[key] = value
            elif isinstance(value, dict) and isinstance(base_config[key], dict):
                for subkey, subvalue in value.items():
                    if subkey not in base_config[key]:
                        base_config[key][subkey] = subvalue
        
        if profile and profile in base_config.get("profiles", {}):
            profile_settings = base_config["profiles"][profile]
            for key, value in profile_settings.items():
                if isinstance(value, dict) and isinstance(base_config.get(key), dict):
                    base_config.get(key, {}).update(value)
                else:
                    base_config[key] = value
            logger.info("Konfigurationsprofil '%s' geladen.", profile)
        return base_config
    except (yaml.YAMLError, Exception) as e:
        logger.error(f"Fehler beim Laden oder Parsen von {config_path}: {e}")
        return default_config


def build_oui_map() -> Dict[str, str]:
    """
    Erstellt das OUI-Mapping aus der lokalen Datei und wählt den korrekten Parser.
    """
    if not OUI_LOCAL_PATH.exists():
        logger.warning(f"OUI-Datei nicht gefunden. Versuche Download...")
        if not download_oui_file():
            logger.error("OUI-Download fehlgeschlagen. Vendor-Lookup nicht verfügbar.")
            return {}

    format_file = OUI_LOCAL_PATH.parent / ".manuf_format"
    file_format = "wireshark"
    if format_file.exists():
        file_format = format_file.read_text().strip()

    logger.info(f"Lade OUI-Datenbank im Format: '{file_format}'")
    
    try:
        start_time = time.time()
        
        if file_format == "ieee":
            mapping = _parse_ieee_format(OUI_LOCAL_PATH)
        elif file_format == "nmap":
            mapping = _parse_nmap_format(OUI_LOCAL_PATH)
        else:
            mapping = _parse_wireshark_format(OUI_LOCAL_PATH)
        
        load_time = time.time() - start_time
        
        if mapping:
            logger.info(f"✓ OUI-Datenbank geladen: {len(mapping):,} Einträge in {load_time:.2f}s")
        else:
            logger.warning("⚠️ Keine OUI-Einträge geladen. Vendor-Lookup nicht verfügbar.")
            
    except Exception as exc:
        logger.error(f"✗ Fehler beim Parsen der OUI-Datei: {exc}")
        return {}

    return mapping


# Lazy-Loading für bessere Startup-Performance
OUI_MAP = {}


def is_local_admin_mac(mac: Optional[str]) -> bool:
    try:
        if not mac or len(mac) < 2:
            return False
        # Prüfe ob MAC-Format korrekt ist
        if ':' not in mac:
            return False
        first_octet = mac.split(":")[0]
        if len(first_octet) != 2:
            return False
        val = int(first_octet, 16)
        return bool(val & 0x02)
    except (ValueError, IndexError):
        return True  # Bei Fehlern als lokal/randomisiert behandeln


def is_valid_bssid(mac: Optional[str]) -> bool:
    if not mac or mac == "ff:ff:ff:ff:ff:ff":
        return False
    try:
        # Prüfe MAC-Format
        parts = mac.split(":")
        if len(parts) != 6:
            return False
        for part in parts:
            if len(part) != 2:
                return False
            int(part, 16)  # Validiere Hex-Format
        
        first_octet = int(parts[0], 16)
        return (first_octet & 1) == 0
    except (ValueError, IndexError):
        return False


def lookup_vendor(mac: Optional[str]) -> Optional[str]:
    """Einfacher Vendor-Lookup ohne Kontext."""
    if not mac:
        return None
    if is_local_admin_mac(mac):
        return "(Lokal / Randomisiert)"
    
    pref = mac.replace("-", ":").upper()[:8]
    
    # Lazy-Loading der OUI-Map
    global OUI_MAP
    if not OUI_MAP:
        logger.info("OUI-Map wird geladen...")
        OUI_MAP = build_oui_map()
    
    return OUI_MAP.get(pref)


def intelligent_vendor_lookup(
    mac: Optional[str], client_state: Optional["ClientState"] = None
) -> Optional[str]:
    """
    Sucht den Hersteller einer MAC-Adresse und versucht, bei unbekannten oder
    randomisierten MACs den Hersteller/Typ anhand von Kontext-Hinweisen zu erraten.
    """
    if not mac:
        return None

    if not is_local_admin_mac(mac):
        vendor = OUI_MAP.get(mac.upper()[:8])
        if vendor:
            return vendor

    context_clues = []

    if client_state:
        if client_state.hostname:
            return client_state.hostname

        if client_state.parsed_ies:
            vendor_ies = client_state.parsed_ies.get("vendor_specific", {})
            if "Apple" in vendor_ies:
                context_clues.append("Apple?")
            elif "Microsoft" in vendor_ies:
                context_clues.append("Microsoft?")
            elif "Wi-Fi Direct" in vendor_ies:
                context_clues.append("Android/Smart?")

    base_string = "(Lokal / Randomisiert)" if is_local_admin_mac(mac) else "N/A"
    if context_clues:
        return f"{base_string} ({', '.join(context_clues)})"
    else:
        return base_string


_IE_FP_CACHE: Dict[int, str] = {}


def ie_fingerprint_hash(ies: Dict[int, List[str]]) -> Optional[str]:
    if not ies:
        return None
    key = hash(tuple((k, tuple(v)) for k, v in sorted(ies.items())))
    if key in _IE_FP_CACHE:
        return _IE_FP_CACHE[key]
    items = [f"{k}:" + ",".join(sorted(ies[k])) for k in sorted(ies.keys())]
    s = "|".join(items)
    h = hashlib.md5(s.encode()).hexdigest()
    _IE_FP_CACHE[key] = h
    return h


def _parse_ht_capabilities(hex_data: str) -> Dict:
    caps = {}
    try:
        data = bytes.fromhex(hex_data)
        cap_info = int.from_bytes(data[:2], "little")
        caps["40mhz_support"] = bool(cap_info & 0x0002)
        if len(data) >= 12:
            rx_mcs = data[4:8]
            if rx_mcs[3] != 0:
                caps["streams"] = 4
            elif rx_mcs[2] != 0:
                caps["streams"] = 3
            elif rx_mcs[1] != 0:
                caps["streams"] = 2
            elif rx_mcs[0] != 0:
                caps["streams"] = 1
            else:
                # Fallback: Setze streams auf 1 wenn keine MCS-Daten vorhanden sind
                caps["streams"] = 1
    except (ValueError, IndexError):
        pass
    return caps


def _parse_vht_capabilities(hex_data: str) -> Dict:
    caps = {}
    try:
        data = bytes.fromhex(hex_data)
        cap_info = int.from_bytes(data[:4], "little")
        ch_width = (cap_info >> 2) & 0b11
        if ch_width == 1:
            caps["160mhz_support"] = True
        caps["mu_beamformer_capable"] = bool(cap_info & (1 << 19))
    except (ValueError, IndexError):
        pass
    return caps


def _parse_rsn_details(hex_data: str) -> Dict:
    details = {
        "pairwise_ciphers": set(),
        "akm_suites": set(),
        "mfp_capable": False,
        "mfp_required": False,
    }
    try:
        data = bytes.fromhex(hex_data)
        # Version (2 bytes) + Group Cipher (4 bytes)
        offset = 6

        # Pairwise Ciphers
        count = int.from_bytes(data[offset : offset + 2], "little")
        offset += 2  # KORREKTUR: Offset nach dem Lesen der Anzahl erhöhen
        for _ in range(count):
            suite = data[offset : offset + 4].hex()
            if suite == "000fac02":
                details["pairwise_ciphers"].add("TKIP")
            elif suite == "000fac04":
                details["pairwise_ciphers"].add("CCMP-128 (AES)")
            offset += 4

        # AKM Suites
        count = int.from_bytes(data[offset : offset + 2], "little")
        offset += 2  # KORREKTUR: Offset nach dem Lesen der Anzahl erhöhen
        for _ in range(count):
            suite = data[offset : offset + 4].hex()
            if suite == "000fac01":
                details["akm_suites"].add("802.1X (EAP)")
            elif suite == "000fac02":
                details["akm_suites"].add("PSK")
            offset += 4

        # RSN Capabilities
        if len(data) > offset:
            caps = int.from_bytes(data[offset : offset + 2], "little")
            details["mfp_capable"] = bool(caps & (1 << 7))
            details["mfp_required"] = bool(caps & (1 << 6))

    except (ValueError, IndexError):
        pass
    details["pairwise_ciphers"] = sorted(list(details["pairwise_ciphers"]))
    details["akm_suites"] = sorted(list(details["akm_suites"]))
    return details


def parse_ies(ies_dict: Dict[int, List[str]], detailed: bool = False) -> Dict[str, any]:
    parsed = {
        "security": set(),
        "standards": set(),
        "vendor_specific": {},
        "roaming_features": set(),
    }
    if 48 in ies_dict:
        parsed["security"].add("WPA2/3")
        if detailed:
            parsed["rsn_details"] = _parse_rsn_details(ies_dict[48][0])
    if 45 in ies_dict or 61 in ies_dict:
        parsed["standards"].add("802.11n")
        if detailed and 45 in ies_dict:
            parsed["ht_caps"] = _parse_ht_capabilities(ies_dict[45][0])
    if 191 in ies_dict:
        parsed["standards"].add("802.11ac")
        if detailed and 191 in ies_dict:
            parsed["vht_caps"] = _parse_vht_capabilities(ies_dict[191][0])
    if 255 in ies_dict:
        for hex_data in ies_dict[255]:
            if hex_data.startswith("23"):
                parsed["standards"].add("802.11ax")
                if detailed:
                    parsed["he_caps"] = {"present": True}
    if 55 in ies_dict:
        parsed["roaming_features"].add("802.11r")
    if 221 in ies_dict:
        for hex_data in ies_dict[221]:
            try:
                if hex_data.startswith("506f9a"):
                    if hex_data.startswith("506f9a09"):
                        parsed["vendor_specific"]["Wi-Fi Direct"] = True
                    if hex_data.startswith("506f9a10"):
                        parsed["roaming_features"].add("802.11v")
                elif hex_data.startswith("000b86") or hex_data.startswith("004096"):
                    parsed["roaming_features"].add("802.11k")
                oui = hex_data[:6]
                vendor = OUI_MAP.get(f"{oui[0:2]}:{oui[2:4]}:{oui[4:6]}".upper())
                if vendor:
                    parsed["vendor_specific"][vendor] = hex_data[6:]
            except IndexError:
                continue
    parsed["security"] = sorted(list(parsed["security"]))
    parsed["standards"] = sorted(list(parsed["standards"]))
    parsed["roaming_features"] = sorted(list(parsed["roaming_features"]))
    return parsed
