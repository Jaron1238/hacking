#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Run-Checks für das WLAN-Analyse-Tool.
"""

import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)


def find_wlan_interfaces() -> List[str]:
    """Findet verfügbare WLAN-Interfaces."""
    try:
        # Versuche iwconfig zu verwenden
        result = subprocess.run(
            ["iwconfig"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        interfaces = []
        for line in result.stdout.split('\n'):
            if 'IEEE 802.11' in line:
                interface = line.split()[0]
                interfaces.append(interface)
        
        return interfaces
    except Exception as e:
        logger.warning(f"Fehler beim Suchen nach WLAN-Interfaces: {e}")
        return []


def check_monitor_mode_support(interface: str) -> bool:
    """Prüft ob ein Interface Monitor-Mode unterstützt."""
    try:
        result = subprocess.run(
            ["iw", interface, "info"],
            capture_output=True,
            text=True,
            check=False
        )
        return "monitor" in result.stdout.lower()
    except Exception:
        return False


def check_root_privileges() -> bool:
    """Prüft ob Root-Rechte vorhanden sind."""
    import os
    return os.geteuid() == 0