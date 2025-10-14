"""
Plugin-Discovery für pytest.

Automatische Erkennung und Ausführung von Plugin-Tests.
"""

import pytest
from pathlib import Path
import sys
import importlib.util


def pytest_configure(config):
    """Konfiguriert pytest für Plugin-Discovery."""
    # Füge Plugin-Verzeichnis zum Python-Pfad hinzu
    plugin_dir = Path(__file__).parent.parent / "plugins"
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    
    # Registriere Plugin-Tests
    register_plugin_tests(config, plugin_dir)


def register_plugin_tests(config, plugin_dir):
    """Registriert Tests für alle gefundenen Plugins."""
    if not plugin_dir.exists():
        return
    
    for plugin_subdir in plugin_dir.iterdir():
        if plugin_subdir.is_dir() and not plugin_subdir.name.startswith('_'):
            tests_dir = plugin_subdir / "tests"
            if tests_dir.exists():
                # Füge Test-Verzeichnis zur pytest-Konfiguration hinzu
                config.addinivalue_line(
                    "testpaths", 
                    str(tests_dir)
                )


def pytest_collect_file(file_path, parent):
    """Sammelt Test-Dateien aus Plugin-Verzeichnissen."""
    if file_path.suffix == ".py" and file_path.name.startswith("test_"):
        # Prüfe, ob es sich um eine Plugin-Test-Datei handelt
        if "plugins" in str(file_path):
            return pytest.Module.from_parent(parent, path=file_path)


def pytest_generate_tests(metafunc):
    """Generiert Tests für Plugin-spezifische Parameter."""
    # Hier können Plugin-spezifische Test-Parameter generiert werden
    pass