# Enhanced OUI download functionality

import csv
import hashlib
import logging
import typing
import urllib.request
import urllib.error
import time
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
        "timeout": 30,
    },
    {
        "name": "Nmap GitHub",
        "url": "https://raw.githubusercontent.com/nmap/nmap/master/nmap-mac-prefixes",
        "format": "nmap",
        "timeout": 20,
    },
    {
        "name": "Wireshark GitLab",
        "url": "https://gitlab.com/wireshark/wireshark/-/raw/master/manuf",
        "format": "wireshark",
        "timeout": 20,
    },
    {
        "name": "Wireshark SVN",
        "url": "https://anonsvn.wireshark.org/wireshark/trunk/manuf",
        "format": "wireshark",
        "timeout": 15,
    },
]

OUI_LOCAL_PATH = Path(__file__).parent / "assets" / "manuf"

def download_oui_file_enhanced(force_update: bool = False, progress_callback=None):
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
        logger.info(f"[{i+1}/{len(OUI_SOURCES)}] Versuche Download: {source['name']}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "text/plain,text/html,*/*",
                "Connection": "keep-alive",
            }
            
            req = urllib.request.Request(source["url"], headers=headers)
            
            with urllib.request.urlopen(req, timeout=source.get("timeout", 30)) as response:
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
                
                logger.info(f"✓ OUI-Datei heruntergeladen: {len(data):,} bytes von {source['name']}")
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

def build_oui_map_enhanced() -> Dict[str, str]:
    """
    Erstellt das OUI-Mapping mit verbesserter Funktionalität.
    """
    if not OUI_LOCAL_PATH.exists():
        logger.warning(f"OUI-Datei nicht gefunden. Versuche Download...")
        if not download_oui_file_enhanced():
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
            from .utils import _parse_nmap_format
            mapping = _parse_nmap_format(OUI_LOCAL_PATH)
        else:
            from .utils import _parse_wireshark_format
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