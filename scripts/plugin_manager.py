#!/usr/bin/env python3
"""
Plugin-Manager für das WLAN-Analyse-Tool.

Verwaltet Plugins: installieren, deinstallieren, testen, etc.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Optional

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plugins import load_all_plugins, BasePlugin


class PluginManager:
    """Manager für Plugin-Operationen."""
    
    def __init__(self, plugin_dir: Path = None):
        self.plugin_dir = plugin_dir or project_root / "plugins"
        self.plugins = {}
        self.load_plugins()
    
    def load_plugins(self):
        """Lädt alle verfügbaren Plugins."""
        self.plugins = load_all_plugins(self.plugin_dir)
    
    def list_plugins(self) -> None:
        """Listet alle verfügbaren Plugins auf."""
        print("Verfügbare Plugins:")
        print("=" * 50)
        
        if not self.plugins:
            print("Keine Plugins gefunden.")
            return
        
        for name, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            print(f"Name: {metadata.name}")
            print(f"  Version: {metadata.version}")
            print(f"  Beschreibung: {metadata.description}")
            print(f"  Autor: {metadata.author}")
            print(f"  Dependencies: {', '.join(metadata.dependencies)}")
            print(f"  Verfügbar: {'Ja' if plugin.validate_dependencies() else 'Nein (Dependencies fehlen)'}")
            print()
    
    def install_dependencies(self, plugin_name: str = None) -> bool:
        """Installiert Dependencies für ein Plugin oder alle Plugins."""
        if plugin_name:
            if plugin_name not in self.plugins:
                print(f"Plugin '{plugin_name}' nicht gefunden.")
                return False
            
            plugin = self.plugins[plugin_name]
            return self._install_plugin_dependencies(plugin)
        else:
            # Installiere Dependencies für alle Plugins
            success = True
            for name, plugin in self.plugins.items():
                print(f"Installiere Dependencies für {name}...")
                if not self._install_plugin_dependencies(plugin):
                    success = False
            return success
    
    def _install_plugin_dependencies(self, plugin: BasePlugin) -> bool:
        """Installiert Dependencies für ein spezifisches Plugin."""
        metadata = plugin.get_metadata()
        
        if not metadata.dependencies:
            print(f"  Keine Dependencies für {metadata.name}")
            return True
        
        try:
            # Versuche pip install
            cmd = [sys.executable, "-m", "pip", "install"] + metadata.dependencies
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  Dependencies für {metadata.name} erfolgreich installiert")
                return True
            else:
                print(f"  Fehler beim Installieren der Dependencies für {metadata.name}:")
                print(f"  {result.stderr}")
                return False
        except Exception as e:
            print(f"  Fehler beim Installieren der Dependencies für {metadata.name}: {e}")
            return False
    
    def test_plugin(self, plugin_name: str) -> bool:
        """Testet ein spezifisches Plugin."""
        if plugin_name not in self.plugins:
            print(f"Plugin '{plugin_name}' nicht gefunden.")
            return False
        
        plugin_dir = self.plugin_dir / plugin_name
        tests_dir = plugin_dir / "tests"
        
        if not tests_dir.exists():
            print(f"Keine Tests für Plugin '{plugin_name}' gefunden.")
            return False
        
        try:
            # Führe pytest für das Plugin aus
            cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print(f"Test-Ergebnisse für {plugin_name}:")
            print(result.stdout)
            if result.stderr:
                print("Fehler:")
                print(result.stderr)
            
            return result.returncode == 0
        except Exception as e:
            print(f"Fehler beim Ausführen der Tests für {plugin_name}: {e}")
            return False
    
    def test_all_plugins(self) -> Dict[str, bool]:
        """Testet alle Plugins."""
        results = {}
        
        for name in self.plugins.keys():
            print(f"\n{'='*50}")
            print(f"Teste Plugin: {name}")
            print('='*50)
            results[name] = self.test_plugin(name)
        
        return results
    
    def create_plugin_template(self, plugin_name: str) -> bool:
        """Erstellt ein Template für ein neues Plugin."""
        plugin_dir = self.plugin_dir / plugin_name
        tests_dir = plugin_dir / "tests"
        
        if plugin_dir.exists():
            print(f"Plugin-Verzeichnis '{plugin_name}' existiert bereits.")
            return False
        
        try:
            # Erstelle Verzeichnisse
            plugin_dir.mkdir(parents=True, exist_ok=True)
            tests_dir.mkdir(parents=True, exist_ok=True)
            
            # Erstelle __init__.py
            init_content = f'''"""
{plugin_name.title()} Plugin für WLAN-Analyse-Tool.
"""

from .plugin import Plugin

__all__ = ['Plugin']
'''
            (plugin_dir / "__init__.py").write_text(init_content)
            
            # Erstelle plugin.py Template
            plugin_content = f'''"""
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
            description="Beschreibung des {plugin_name} Plugins",
            author="Dein Name",
            dependencies=[]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Hauptfunktion des Plugins.
        """
        console.print(f"\\n[bold cyan]{plugin_name.title()} Plugin wird ausgeführt...[/bold cyan]")
        
        try:
            # Plugin-Logik hier implementieren
            console.print(f"[green]{plugin_name.title()} Plugin erfolgreich ausgeführt![/green]")
            
        except Exception as e:
            console.print(f"[red]Fehler im {plugin_name} Plugin: {{e}}[/red]")
            logger.error(f"Fehler im {plugin_name} Plugin: {{e}}", exc_info=True)
'''
            (plugin_dir / "plugin.py").write_text(plugin_content)
            
            # Erstelle tests/__init__.py
            (tests_dir / "__init__.py").write_text('"""Tests für das Plugin."""')
            
            # Erstelle test_plugin.py Template
            test_content = f'''"""
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
        
        # Mock state und events
        mock_state = {{}}
        mock_events = []
        
        plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
'''
            (tests_dir / f"test_{plugin_name}.py").write_text(test_content)
            
            # Erstelle requirements.txt
            (plugin_dir / "requirements.txt").write_text("# Dependencies für das Plugin\n# Füge hier deine Dependencies hinzu\n")
            
            print(f"Plugin-Template für '{plugin_name}' erstellt in: {plugin_dir}")
            return True
            
        except Exception as e:
            print(f"Fehler beim Erstellen des Plugin-Templates: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Gibt detaillierte Informationen über ein Plugin zurück."""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        metadata = plugin.get_metadata()
        
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "dependencies": metadata.dependencies,
            "dependencies_available": plugin.validate_dependencies(),
            "plugin_dir": str(self.plugin_dir / plugin_name)
        }


def main():
    """Hauptfunktion des Plugin-Managers."""
    parser = argparse.ArgumentParser(description="Plugin-Manager für WLAN-Analyse-Tool")
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")
    
    # List command
    subparsers.add_parser("list", help="Liste alle Plugins auf")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Installiere Plugin-Dependencies")
    install_parser.add_argument("plugin", nargs="?", help="Plugin-Name (optional, installiert alle wenn nicht angegeben)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Teste Plugins")
    test_parser.add_argument("plugin", nargs="?", help="Plugin-Name (optional, testet alle wenn nicht angegeben)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Erstelle ein neues Plugin-Template")
    create_parser.add_argument("name", help="Name des neuen Plugins")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Zeige Plugin-Informationen")
    info_parser.add_argument("plugin", help="Plugin-Name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = PluginManager()
    
    if args.command == "list":
        manager.list_plugins()
    
    elif args.command == "install":
        if args.plugin:
            success = manager.install_dependencies(args.plugin)
            sys.exit(0 if success else 1)
        else:
            success = manager.install_dependencies()
            sys.exit(0 if success else 1)
    
    elif args.command == "test":
        if args.plugin:
            success = manager.test_plugin(args.plugin)
            sys.exit(0 if success else 1)
        else:
            results = manager.test_all_plugins()
            all_passed = all(results.values())
            print(f"\nZusammenfassung: {sum(results.values())}/{len(results)} Plugins erfolgreich getestet")
            sys.exit(0 if all_passed else 1)
    
    elif args.command == "create":
        success = manager.create_plugin_template(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == "info":
        info = manager.get_plugin_info(args.plugin)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Plugin '{args.plugin}' nicht gefunden.")
            sys.exit(1)


if __name__ == "__main__":
    main()