# data/pre_run_checks.py

import shutil
import sys
import logging
import subprocess
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def check_dependencies():
    """
    Überprüft, ob alle erforderlichen Kommandozeilen-Tools im System-PATH verfügbar sind.
    Beendet das Programm mit einer Fehlermeldung, wenn ein Tool fehlt.
    """
    required_tools = ["tcpdump", "iw", "ip"]
    missing_tools = []

    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)

    if missing_tools:
        logger.error("FEHLER: Die folgenden Abhängigkeiten wurden nicht gefunden:")
        for tool in missing_tools:
            logger.error(f"  - {tool}")
        logger.error("Bitte installieren Sie die fehlenden Tools und versuchen Sie es erneut.")
        logger.error("Auf Debian/Ubuntu-Systemen können die meisten mit 'sudo apt install tcpdump iw iproute2' installiert werden.")
        sys.exit(1)
    
    logger.info("Alle Systemabhängigkeiten sind vorhanden.")


def find_wlan_interfaces() -> List[str]:
    """Sucht nach WLAN-Interfaces im System."""
    try:
        result = subprocess.run(['iw', 'dev'], capture_output=True, text=True, check=True)
        interfaces = [line.split()[1] for line in result.stdout.splitlines() if line.strip().startswith('Interface')]
        return interfaces
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def get_interface_capabilities(interface: str) -> Optional[Dict]:
    """Prüft die Fähigkeiten eines spezifischen WLAN-Interfaces."""
    try:
        # Finde den zugehörigen phy-Namen
        result_dev = subprocess.run(['iw', 'dev', interface, 'info'], capture_output=True, text=True, check=True)
        phy_name = [line.split()[1] for line in result_dev.stdout.splitlines() if 'wiphy' in line][0]
        
        # Prüfe die Fähigkeiten des phy
        result_phy = subprocess.run(['iw', 'phy', phy_name, 'info'], capture_output=True, text=True, check=True)
        output = result_phy.stdout
        
        capabilities = {
            "supports_monitor": "monitor" in output,
            "supports_5ghz": "5180 MHz" in output,
            "supports_vht": "VHT Capabilities" in output, # 802.11ac
            "supports_he": "HE Capabilities" in output    # 802.11ax
        }
        return capabilities
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None