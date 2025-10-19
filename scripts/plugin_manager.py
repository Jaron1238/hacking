#!/usr/bin/env python3
"""
Plugin-Manager für das WLAN-Analyse-Tool.
"""

import argparse
import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plugins import BasePlugin, load_all_plugins

logger = logging.getLogger("plugin_manager")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class PluginManager:
    """Manager für Plugin-Operationen."""

    plugin_dir: Path
    plugins: Dict[str, BasePlugin]

    def __init__(self, plugin_dir: Optional[Path] = None) -> None:
        self.plugin_dir = plugin_dir or project_root / "plugins"
        self.plugins = {}
        self.load_plugins()

    def load_plugins(self) -> None:
        """Lädt alle verfügbaren Plugins."""
        self.plugins = load_all_plugins(self.plugin_dir)

    def list_plugins(self) -> None:
        """Listet alle verfügbaren Plugins auf."""
        logger.info("Verfügbare Plugins:")
        logger.info("=" * 50)

        if not self.plugins:
            logger.warning("Keine Plugins gefunden.")
            return

        for name, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            is_available = plugin.validate_dependencies()
            has_missing_deps = getattr(plugin, '_has_missing_deps', False)
            
            logger.info(f"Name: {metadata.name}")
            logger.info(f"  Version: {metadata.version}")
            logger.info(f"  Beschreibung: {metadata.description}")
            logger.info(f"  Autor: {metadata.author}")
            logger.info(f"  Dependencies: {', '.join(metadata.dependencies)}")
            
            if is_available:
                logger.info("  Status: ✅ Verfügbar")
            elif has_missing_deps:
                logger.info("  Status: ⚠️  Geladen mit fehlenden Dependencies")
            else:
                logger.info("  Status: ❌ Nicht verfügbar (Dependencies fehlen)")
            
            logger.info("")

    def install_dependencies(self, plugin_name: Optional[str] = None) -> bool:
        """Installiert Dependencies für ein Plugin oder alle Plugins."""
        if plugin_name:
            plugin = self.plugins.get(plugin_name)
            if not plugin:
                logger.error(f"Plugin '{plugin_name}' nicht gefunden.")
                return False
            return self._install_plugin_dependencies(plugin)
        else:
            # Installiere Dependencies für alle Plugins
            success = True
            for name, plugin in self.plugins.items():
                logger.info(f"Installiere Dependencies für {name}...")
                if not self._install_plugin_dependencies(plugin):
                    success = False
            return success

    def _install_plugin_dependencies(self, plugin: BasePlugin) -> bool:
        """Installiert Dependencies für ein spezifisches Plugin."""
        metadata = plugin.get_metadata()
        if not metadata.dependencies:
            logger.info(f"  Keine Dependencies für {metadata.name}")
            return True
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + metadata.dependencies
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"  Dependencies für {metadata.name} erfolgreich installiert")
                return True
            else:
                logger.error(f"  Fehler beim Installieren der Dependencies für {metadata.name}:")
                logger.error(f"  {result.stderr}")
                return False
        except subprocess.SubprocessError as e:
            logger.error(f"  Fehler beim Ausführen von pip für {metadata.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"  Unerwarteter Fehler beim Installieren der Dependencies für {metadata.name}: {e}")
            return False

    def test_plugin(self, plugin_name: str) -> bool:
        """Testet ein spezifisches Plugin."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            logger.error(f"Plugin '{plugin_name}' nicht gefunden.")
            return False
        
        # Prüfe Dependencies vor dem Test
        if not plugin.validate_dependencies():
            logger.warning(f"Plugin '{plugin_name}' hat fehlende Dependencies - Tests können fehlschlagen")
        
        plugin_dir = self.plugin_dir / plugin_name
        tests_dir = plugin_dir / "tests"
        if not tests_dir.exists():
            logger.warning(f"Keine Tests für Plugin '{plugin_name}' gefunden.")
            return False
        
        try:
            # Führe Tests mit besserer Fehlerbehandlung aus
            cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            logger.info(f"Test-Ergebnisse für {plugin_name}:")
            logger.info(result.stdout)
            
            if result.stderr:
                logger.error("Fehler:")
                logger.error(result.stderr)
            
            # Analysiere Test-Ergebnisse
            if result.returncode == 0:
                logger.info(f"✅ Alle Tests für {plugin_name} erfolgreich")
                return True
            else:
                logger.error(f"❌ Tests für {plugin_name} fehlgeschlagen (Exit Code: {result.returncode})")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Tests für {plugin_name} überschritten Zeitlimit (5 Minuten)")
            return False
        except subprocess.SubprocessError as e:
            logger.error(f"Fehler beim Ausführen der Tests für {plugin_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Testen von {plugin_name}: {e}")
            return False

    def test_all_plugins(self) -> Dict[str, bool]:
        """Testet alle Plugins."""
        results: Dict[str, bool] = {}
        for name in self.plugins.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Teste Plugin: {name}")
            logger.info("=" * 50)
            results[name] = self.test_plugin(name)
        return results

    def _create_init_py(self, plugin_dir: Path, plugin_name: str) -> None:
        init_content = f'"""
{plugin_name.title()} Plugin für WLAN-Analyse-Tool.
"""

from .plugin import Plugin

__all__ = [\'Plugin\']
'
        (plugin_dir / "__init__.py").write_text(init_content)

    def _create_plugin_py(self, plugin_dir: Path, plugin_name: str, description: str = None, author: str = "Plugin Author", dependencies: list = None) -> None:
        plugin_content = f'"""
{plugin_name.title()} Plugin.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from plugins import BasePlugin, PluginMetadata

logger = logging.getLogger(__name__)

class Plugin(BasePlugin):
    """{plugin_name.title()} Plugin."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{plugin_name.title()}",
            version="1.0.0",
            description="{description or f'Beschreibung des {plugin_name} Plugins'}",
            author="{author}",
            dependencies={dependencies or []}
        )

    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Hauptfunktion des Plugins.
        """
        console.print(f"\n[bold cyan]{plugin_name.title()} Plugin wird ausgeführt...[\/bold cyan]")
        try:
            # Plugin-Logik hier implementieren
            console.print(f"[green]{plugin_name.title()} Plugin erfolgreich ausgeführt![\/green]")
        except Exception as e:
            console.print(f"[red]Fehler im {plugin_name} Plugin: {e}[\/red]")
            logger.error(f"Fehler im {plugin_name} Plugin: {e}", exc_info=True)
'
        (plugin_dir / "plugin.py").write_text(plugin_content)

    def _create_tests(self, tests_dir: Path, plugin_name: str) -> None:
        (tests_dir / "__init__.py").write_text('"""Tests für das Plugin."""')
        test_content = f'"""
Tests für das {plugin_name.title()} Plugin.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from plugins.{plugin_name}.plugin import Plugin

class Test{plugin_name.title()}Plugin:
    """Test-Klasse für das {plugin_name.title()} Plugin."""

    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()

    @pytest.fixture
    def mock_console(self):
        """Mock Console für Tests."""
        console = MagicMock()
        console.print = MagicMock()
        return console

    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Temporäres Ausgabeverzeichnis."""
        return tmp_path / "output"

    def test_plugin_metadata(self, plugin):
        """Test Plugin-Metadaten."""
        metadata = plugin.get_metadata()
        assert metadata.name == "{plugin_name.title()}"
        assert metadata.version == "1.0.0"

    def test_plugin_run(self, plugin, mock_console, temp_outdir):
        """Test Plugin-Ausführung."""
        temp_outdir.mkdir(exist_ok=True)
        mock_state = {}
        mock_events = []
        plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        assert mock_console.print.called
'
        (tests_dir / f"test_{plugin_name}.py").write_text(test_content)

    def _create_requirements(self, plugin_dir: Path, dependencies: list = None) -> None:
        if dependencies:
            req_content = "# Dependencies für das Plugin\n" + "\n".join(dependencies) + "\n"
        else:
            req_content = "# Dependencies für das Plugin\n# Füge hier deine Dependencies hinzu\n# Beispiel:\n# numpy\n# pandas\n# matplotlib\n"
        (plugin_dir / "requirements.txt").write_text(req_content)
    
    def _create_readme(self, plugin_dir: Path, plugin_name: str, description: str = None) -> None:
        readme_content = f"""# {plugin_name.title()} Plugin

{description or f'Ein Plugin für das WLAN-Analyse-Tool'}

## Installation

1. Installiere die Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Das Plugin wird automatisch vom Plugin Manager erkannt.

## Verwendung

Das Plugin wird automatisch ausgeführt, wenn es in der Plugin-Liste aktiviert ist.

## Entwicklung

### Tests ausführen

```bash
python -m pytest tests/ -v
```

### Plugin testen

```bash
python scripts/plugin_manager.py test {plugin_name}
```

## Struktur

- `plugin.py` - Haupt-Plugin-Implementierung
- `tests/` - Unit Tests
- `requirements.txt` - Python Dependencies
- `__init__.py` - Plugin-Initialisierung
"""
        (plugin_dir / "README.md").write_text(readme_content)

    def create_plugin_template(self, plugin_name: str, description: str = None, author: str = "Plugin Author", dependencies: list = None) -> bool:
        """Erstellt ein Template für ein neues Plugin."""
        plugin_dir = self.plugin_dir / plugin_name
        tests_dir = plugin_dir / "tests"
        
        if plugin_dir.exists():
            logger.error(f"Plugin-Verzeichnis '{plugin_name}' existiert bereits.")
            return False
        
        if not plugin_name.replace("_", "").replace("-", "").isalnum():
            logger.error(f"Plugin-Name '{plugin_name}' enthält ungültige Zeichen. Verwende nur Buchstaben, Zahlen, _ und -")
            return False
        
        try:
            plugin_dir.mkdir(parents=True, exist_ok=True)
            tests_dir.mkdir(parents=True, exist_ok=True)
            
            # Erstelle Plugin-Dateien
            self._create_init_py(plugin_dir, plugin_name)
            self._create_plugin_py(plugin_dir, plugin_name, description, author, dependencies)
            self._create_tests(tests_dir, plugin_name)
            self._create_requirements(plugin_dir, dependencies)
            
            # Erstelle README
            self._create_readme(plugin_dir, plugin_name, description)
            
            logger.info(f"✅ Plugin-Template für '{plugin_name}' erstellt in: {plugin_dir}")
            logger.info(f"📁 Plugin-Verzeichnis: {plugin_dir}")
            logger.info(f"🧪 Test-Verzeichnis: {tests_dir}")
            logger.info(f"📋 Nächste Schritte:")
            logger.info(f"   1. Bearbeite {plugin_dir}/plugin.py")
            logger.info(f"   2. Füge Dependencies zu {plugin_dir}/requirements.txt hinzu")
            logger.info(f"   3. Führe Tests aus: python -m pytest {tests_dir}")
            
            return True
        except OSError as e:
            logger.error(f"Fehler beim Erstellen des Plugin-Templates: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Erstellen des Plugin-Templates: {e}")
            return False

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Gibt detaillierte Informationen über ein Plugin zurück."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return None
        metadata = plugin.get_metadata()
        is_available = plugin.validate_dependencies()
        has_missing_deps = getattr(plugin, '_has_missing_deps', False)
        
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "dependencies": metadata.dependencies,
            "dependencies_available": is_available,
            "has_missing_deps": has_missing_deps,
            "status": "available" if is_available else "missing_deps" if has_missing_deps else "unavailable",
            "plugin_dir": str(self.plugin_dir / plugin_name),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Führt eine umfassende Gesundheitsprüfung aller Plugins durch."""
        results = {
            "total_plugins": len(self.plugins),
            "available_plugins": 0,
            "plugins_with_missing_deps": 0,
            "unavailable_plugins": 0,
            "plugin_details": {}
        }
        
        for name, plugin in self.plugins.items():
            info = self.get_plugin_info(name)
            results["plugin_details"][name] = info
            
            if info["status"] == "available":
                results["available_plugins"] += 1
            elif info["status"] == "missing_deps":
                results["plugins_with_missing_deps"] += 1
            else:
                results["unavailable_plugins"] += 1
        
        return results

# ...Main-Funktion bleibt unverändert...