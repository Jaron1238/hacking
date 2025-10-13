#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Globale pytest-Konfiguration für das WLAN-Analyse-Tool.
Setzt automatisch den Python-Pfad, damit Module gefunden werden.
"""

import sys
from pathlib import Path

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
workspace_root = Path(__file__).parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))