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
# Das Skript wird sie in dieser Reihenfolge ausprobieren.
OUI_SOURCES = [
    {
        "name": "Nmap",
        "url": "https://raw.githubusercontent.com/nmap/nmap/master/nmap-mac-prefixes",
        "format": "nmap",
    },
    {
        "name": "Wireshark (SVN)",
        "url": "https://anonsvn.wireshark.org/wireshark/trunk/manuf",
        "format": "wireshark",
    },
]
OUI_LOCAL_PATH = Path(__file__).parent / "assets" / "manuf"


def download_oui_file():
    """
    Lädt die OUI-Herstellerdatei von der bestmöglichen Quelle herunter.
    Versucht nacheinander mehrere URLs, bis eine erfolgreich ist.
    """
    for source in OUI_SOURCES:
        logger.info(f"Versuche OUI-Download von Quelle: {source['name']}...")
        try:
            req = urllib.request.Request(
                source["url"], headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as response, open(
                OUI_LOCAL_PATH, "wb"
            ) as out_file:
                data = response.read()
                out_file.write(data)
            logger.info(
                f"OUI-Datei von {source['name']} erfolgreich nach '{OUI_LOCAL_PATH}' heruntergeladen."
            )
            # Speichere das Format der heruntergeladenen Datei für den Parser
            (OUI_LOCAL_PATH.parent / ".manuf_format").write_text(source["format"])
            return True
        except Exception as e:
            logger.warning(f"Download von {source['name']} fehlgeschlagen: {e}")
            continue  # Versuche die nächste Quelle

    logger.error("Alle Download-Quellen für die OUI-Datei sind fehlgeschlagen.")
    return False


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
                # Konvertiere '001122' zu '00:11:22'
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


def load_config(profile: Optional[str]) -> Dict:
    """Lädt die Konfiguration aus config.yaml und wendet ein Profil an."""
    # Versuche, die config.yaml im aktuellen Verzeichnis oder im Projekt-Stammverzeichnis zu finden
    config_path = Path.cwd() / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        logger.warning("Keine 'config.yaml'-Datei gefunden. Verwende Standardwerte.")
        return {}

    try:
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or {}
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
        return {}


def build_oui_map() -> Dict[str, str]:
    """
    Erstellt das OUI-Mapping aus der lokalen Datei und wählt den korrekten Parser.
    """
    if not OUI_LOCAL_PATH.exists():
        logger.warning(
            f"Lokale OUI-Datei ('{OUI_LOCAL_PATH}') nicht gefunden. Versuche Download..."
        )
        if not download_oui_file():
            return {}

    format_file = OUI_LOCAL_PATH.parent / ".manuf_format"
    file_format = "wireshark"  # Standard-Fallback
    if format_file.exists():
        file_format = format_file.read_text().strip()

    logger.info(f"Lese OUI-Datei im Format: '{file_format}'")
    try:
        if file_format == "nmap":
            mapping = _parse_nmap_format(OUI_LOCAL_PATH)
        else:  # Standard ist wireshark
            mapping = _parse_wireshark_format(OUI_LOCAL_PATH)
    except Exception as exc:
        logger.error("Fehler beim Parsen der OUI-Datei %s: %s", OUI_LOCAL_PATH, exc)
        return {}

    if not mapping:
        logger.warning(
            "Keine Einträge aus der OUI-Datei geladen. Hersteller-Lookup wird nicht funktionieren."
        )

    return mapping


OUI_MAP = build_oui_map()


def is_local_admin_mac(mac: Optional[str]) -> bool:
    try:
        if not mac:
            return False
        first_octet = mac.split(":")[0]
        val = int(first_octet, 16)
        return bool(val & 0x02)
    except (ValueError, IndexError):
        return False


def is_valid_bssid(mac: Optional[str]) -> bool:
    if not mac or mac == "ff:ff:ff:ff:ff:ff":
        return False
    try:
        first_octet = int(mac.split(":")[0], 16)
        return (first_octet & 1) == 0
    except (ValueError, IndexError):
        return False


def lookup_vendor(mac: Optional[str]) -> Optional[str]:
    """Einfacher Vendor-Lookup ohne Kontext. Bevorzuge intelligent_vendor_lookup."""
    if not mac:
        return None
    if is_local_admin_mac(mac):
        return "(Lokal / Randomisiert)"
    pref = mac.replace("-", ":").upper()[:8]
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


def ie_fingerprint_hash(ies: Dict[int, List[str]]) -> str:
    key = hash(tuple((k, tuple(v)) for k, v in sorted(ies.items())))
    if key in _IE_FP_CACHE:
        return _IE_FP_CACHE[key]
    items = [f"{k}:" + ",".join(sorted(ies[k])) for k in sorted(ies.keys())]
    s = "|".join(items)
    h = hashlib.sha1(s.encode()).hexdigest()
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
        offset = 4
        count = int.from_bytes(data[2:4], "little")
        for _ in range(count):
            suite = data[offset : offset + 4].hex()
            if suite == "000fac02":
                details["pairwise_ciphers"].add("TKIP")
            elif suite == "000fac04":
                details["pairwise_ciphers"].add("CCMP-128 (AES)")
            offset += 4
        count = int.from_bytes(data[offset : offset + 2], "little")
        offset += 2
        for _ in range(count):
            suite = data[offset : offset + 4].hex()
            if suite == "000fac01":
                details["akm_suites"].add("802.1X (EAP)")
            elif suite == "000fac02":
                details["akm_suites"].add("PSK")
            offset += 4
        if len(data) > offset:
            caps = int.from_bytes(data[offset : offset + 2], "little")
            details["mfp_capable"] = bool(caps & (1 << 7))
            details["mfp_required"] = bool(caps & (1 << 6))
    except (ValueError, IndexError):
        pass
    details["pairwise_ciphers"] = sorted(list(details["pairwise_ciphers"]))
    details["akm_suites"] = sorted(list(details["akm_suites"]))
    return details


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
