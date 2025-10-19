"""
Plugin-System für WLAN-Analyse-Tool.

Jedes Plugin sollte in einem eigenen Unterverzeichnis organisiert werden:
plugins/
├── clustering_advanced/
│   ├── __init__.py
│   ├── plugin.py
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_clustering_advanced.py
│   └── requirements.txt
├── ensemble_models/
│   ├── __init__.py
│   ├── plugin.py
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_ensemble_models.py
│   └── requirements.txt
└── ...
"""

from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util
import logging

logger = logging.getLogger(__name__)

class PluginMetadata:
    """Metadaten für ein Plugin."""
    def __init__(self, name: str, version: str, description: str, 
                 author: str, dependencies: list = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []

class BasePlugin:
    """Basis-Klasse für alle Plugins."""
    
    def __init__(self):
        self.metadata = self.get_metadata()
    
    def get_metadata(self) -> PluginMetadata:
        """Muss von Unterklassen implementiert werden."""
        raise NotImplementedError
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """Hauptfunktion des Plugins. Muss von Unterklassen implementiert werden."""
        raise NotImplementedError
    
    def validate_dependencies(self) -> bool:
        """Überprüft, ob alle Dependencies verfügbar sind."""
        missing_deps = []
        for dep in self.metadata.dependencies:
            try:
                # Spezielle Behandlung für verschiedene Pakete
                if dep == "sklearn":
                    __import__("sklearn")
                elif dep == "plotly":
                    __import__("plotly")
                elif dep == "hdbscan":
                    __import__("hdbscan")
                elif dep == "torch":
                    __import__("torch")
                elif dep == "umap":
                    __import__("umap")
                else:
                    __import__(dep)
            except ImportError as e:
                missing_deps.append(f"{dep} ({e})")
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler beim Importieren von {dep}: {e}")
                missing_deps.append(f"{dep} (unexpected error)")
        
        if missing_deps:
            logger.warning(f"Dependencies {missing_deps} für Plugin '{self.metadata.name}' nicht verfügbar")
            return False
        return True

def load_plugin_from_directory(plugin_dir: Path) -> Optional[BasePlugin]:
    """Lädt ein Plugin aus einem Verzeichnis."""
    plugin_file = plugin_dir / "plugin.py"
    if not plugin_file.exists():
        logger.warning(f"Keine plugin.py in {plugin_dir} gefunden")
        return None
    
    try:
        # Lade das Plugin-Modul
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_dir.name}.plugin", 
            plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Finde die Plugin-Klasse (sollte Plugin heißen)
        if hasattr(module, 'Plugin'):
            plugin_class = getattr(module, 'Plugin')
            plugin_instance = plugin_class()
            
            # Validiere Dependencies
            if plugin_instance.validate_dependencies():
                return plugin_instance
            else:
                logger.warning(f"Plugin {plugin_dir.name} hat fehlende Dependencies - wird trotzdem geladen")
                # Lade Plugin auch mit fehlenden Dependencies, aber markiere es
                plugin_instance._has_missing_deps = True
                return plugin_instance
        else:
            logger.warning(f"Keine Plugin-Klasse in {plugin_file} gefunden")
            return None
            
    except Exception as e:
        logger.error(f"Fehler beim Laden des Plugins {plugin_dir.name}: {e}")
        return None

def load_all_plugins(plugin_base_dir: Path) -> Dict[str, BasePlugin]:
    """Lädt alle Plugins aus dem Plugin-Verzeichnis."""
    plugins = {}
    
    if not plugin_base_dir.exists():
        logger.warning(f"Plugin-Verzeichnis {plugin_base_dir} existiert nicht")
        return plugins
    
    for plugin_dir in plugin_base_dir.iterdir():
        if plugin_dir.is_dir() and not plugin_dir.name.startswith('_'):
            plugin = load_plugin_from_directory(plugin_dir)
            if plugin:
                plugins[plugin_dir.name] = plugin
                logger.info(f"Plugin '{plugin.metadata.name}' v{plugin.metadata.version} geladen")
    
    return plugins