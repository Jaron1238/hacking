# -*- coding: utf-8 -*-
"""
Funktionen zur Erstellung von Analyse-Berichten, z.B. im HTML-Format.
"""
import logging
from pathlib import Path
import time
from typing import Dict, Any, List

from ..storage.state import WifiAnalysisState
from ..storage.data_models import InferenceResult

logger = logging.getLogger(__name__)

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = FileSystemLoader = None

def generate_html_report(state: WifiAnalysisState, analysis_results: Dict[str, Any], template_path: str, out_file: str):
    if Environment is None:
        logger.error("Jinja2 ist nicht installiert. HTML-Bericht kann nicht erstellt werden.")
        return
    
    # Use default template if no path provided
    if not template_path:
        template_path = str(Path(__file__).parent / "template.html")
    
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    if not template_dir.exists() or not Path(template_path).is_file():
        logger.error(f"Template-Datei '{template_path}' nicht gefunden.")
        return
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_name)
    inference_list: List[InferenceResult] = analysis_results.get("inference") or []
    report_data = {
        "generation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "ap_count": len(state.aps), "client_count": len(state.clients), "ssid_count": len(state.ssid_map),
        "inference_results": inference_list,
        "client_clusters": analysis_results.get("client_clusters"),
        "ap_clusters": analysis_results.get("ap_clusters")
    }
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(template.render(report_data))
        logger.info(f"HTML-Bericht erfolgreich nach '{out_file}' geschrieben.")
    except Exception as e:
        logger.error(f"Fehler beim Schreiben des HTML-Berichts: {e}")