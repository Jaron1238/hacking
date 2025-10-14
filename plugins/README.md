# 🔌 Plugin-System

Das WLAN-Analyse-Tool verwendet ein modulares Plugin-System, das es ermöglicht, neue Analyse-Funktionen einfach hinzuzufügen und zu verwalten.

## 📁 Plugin-Struktur

Jedes Plugin sollte in einem eigenen Unterverzeichnis organisiert werden:

```
plugins/
├── clustering_advanced/          # Erweiterte Clustering-Algorithmen
│   ├── __init__.py
│   ├── plugin.py                 # Haupt-Plugin-Implementierung
│   ├── requirements.txt          # Plugin-spezifische Dependencies
│   └── tests/                    # Plugin-Tests
│       ├── __init__.py
│       └── test_clustering_advanced.py
├── ensemble_models/              # Ensemble Machine Learning
├── reinforcement_learning/       # RL-basierte Optimierung
├── sankey/                       # Roaming-Visualisierung
└── umap_plot/                    # UMAP-basierte Visualisierung
```

## 🚀 Plugin erstellen

### 1. Automatisch mit Template

```bash
# Erstelle ein neues Plugin-Template
python scripts/plugin_manager.py create mein_plugin
```

### 2. Manuell erstellen

1. **Verzeichnis erstellen:**
   ```bash
   mkdir plugins/mein_plugin
   mkdir plugins/mein_plugin/tests
   ```

2. **Plugin-Implementierung (`plugin.py`):**
   ```python
   from plugins import BasePlugin, PluginMetadata
   
   class Plugin(BasePlugin):
       def get_metadata(self) -> PluginMetadata:
           return PluginMetadata(
               name="Mein Plugin",
               version="1.0.0",
               description="Beschreibung meines Plugins",
               author="Dein Name",
               dependencies=["numpy", "pandas"]
           )
       
       def run(self, state, events, console, outdir, **kwargs):
           console.print("Mein Plugin läuft!")
           # Plugin-Logik hier
   ```

3. **Tests erstellen (`tests/test_mein_plugin.py`):**
   ```python
   import pytest
   from plugins.mein_plugin.plugin import Plugin
   
   def test_plugin():
       plugin = Plugin()
       assert plugin.get_metadata().name == "Mein Plugin"
   ```

4. **Dependencies definieren (`requirements.txt`):**
   ```
   numpy>=1.20.0
   pandas>=1.3.0
   ```

## 🧪 Plugin testen

### Alle Plugin-Tests ausführen
```bash
cd pytest
make test-plugins
```

### Spezifisches Plugin testen
```bash
cd pytest
make test-clustering-advanced
```

### Mit pytest direkt
```bash
# Alle Plugin-Tests
pytest -v -m plugin

# Spezifisches Plugin
pytest -v plugins/clustering_advanced/tests/

# Mit Coverage
pytest -v --cov=plugins
```

## 📦 Plugin-Management

### Plugin-Manager verwenden

```bash
# Alle Plugins auflisten
python scripts/plugin_manager.py list

# Dependencies installieren
python scripts/plugin_manager.py install
python scripts/plugin_manager.py install clustering_advanced

# Plugins testen
python scripts/plugin_manager.py test
python scripts/plugin_manager.py test clustering_advanced

# Plugin-Informationen anzeigen
python scripts/plugin_manager.py info clustering_advanced
```

### Dependencies verwalten

```bash
# Dependencies für alle Plugins installieren
cd pytest
make install-deps

# Dependencies für spezifisches Plugin
pip install -r plugins/clustering_advanced/requirements.txt
```

## 🔧 Plugin-API

### BasePlugin-Klasse

Alle Plugins müssen von `BasePlugin` erben:

```python
class Plugin(BasePlugin):
    def get_metadata(self) -> PluginMetadata:
        """Muss implementiert werden."""
        pass
    
    def run(self, state, events, console, outdir, **kwargs):
        """Hauptfunktion des Plugins."""
        pass
    
    def validate_dependencies(self) -> bool:
        """Überprüft Dependencies (automatisch)."""
        pass
```

### PluginMetadata

```python
PluginMetadata(
    name="Plugin Name",           # Anzeigename
    version="1.0.0",             # Versionsnummer
    description="Beschreibung",   # Kurze Beschreibung
    author="Autor",              # Autor des Plugins
    dependencies=["numpy"]       # Liste der Dependencies
)
```

### Plugin-Parameter

Die `run()`-Funktion erhält folgende Parameter:

- `state`: WifiAnalysisState-Objekt mit APs und Clients
- `events`: Liste der WiFi-Events
- `console`: Rich Console für Ausgaben
- `outdir`: Path-Objekt für Ausgabedateien
- `**kwargs`: Zusätzliche Parameter (z.B. clustered_client_df)

## 🎯 Best Practices

### 1. Plugin-Design
- **Ein Plugin, eine Aufgabe**: Jedes Plugin sollte einen spezifischen Zweck haben
- **Robuste Fehlerbehandlung**: Try-catch für alle kritischen Operationen
- **Logging**: Verwende `logger` für Debugging und Fehler
- **Dokumentation**: Dokumentiere alle öffentlichen Methoden

### 2. Testing
- **Unit-Tests**: Teste alle wichtigen Funktionen
- **Mock-Objekte**: Verwende Mocks für externe Dependencies
- **Edge Cases**: Teste mit leeren Daten, ungültigen Eingaben, etc.
- **Performance**: Markiere langsame Tests mit `@pytest.mark.slow`

### 3. Dependencies
- **Minimale Dependencies**: Nur notwendige Pakete
- **Versionen spezifizieren**: Verwende `>=` für Mindestversionen
- **Fallback-Implementierungen**: Für optionale Dependencies

### 4. Ausgabe
- **Strukturierte Ausgabe**: Verwende Rich Console für schöne Ausgaben
- **Dateien speichern**: Alle Ergebnisse in `outdir` speichern
- **Logging**: Wichtige Ereignisse loggen

## 🔍 Debugging

### Plugin-Loading debuggen
```python
from plugins import load_all_plugins
plugins = load_all_plugins(Path("plugins"))
print(f"Geladene Plugins: {list(plugins.keys())}")
```

### Dependencies prüfen
```python
plugin = plugins["clustering_advanced"]
print(f"Dependencies verfügbar: {plugin.validate_dependencies()}")
```

### Tests debuggen
```bash
# Mit detaillierter Ausgabe
pytest -v -s plugins/clustering_advanced/tests/

# Nur fehlgeschlagene Tests
pytest -v --tb=long plugins/clustering_advanced/tests/
```

## 📚 Verfügbare Plugins

### clustering_advanced
- **Zweck**: Erweiterte Clustering-Algorithmen
- **Algorithmen**: Spectral, Hierarchical, GMM, OPTICS, HDBSCAN
- **Dependencies**: scikit-learn, plotly, hdbscan
- **Status**: ✅ Vollständig getestet

### ensemble_models
- **Zweck**: Ensemble Machine Learning Modelle
- **Features**: Voting, Stacking, Bagging, Boosting
- **Dependencies**: scikit-learn, plotly
- **Status**: ✅ Vollständig getestet

### reinforcement_learning
- **Zweck**: RL-basierte WiFi-Scanning-Optimierung
- **Algorithmen**: Q-Learning, Deep Q-Learning
- **Dependencies**: gym, torch (optional)
- **Status**: ✅ Vollständig getestet

### sankey
- **Zweck**: Roaming-Visualisierung
- **Features**: Client-AP-Übergänge
- **Dependencies**: plotly
- **Status**: ✅ Vollständig getestet

### umap_plot
- **Zweck**: UMAP-basierte Client-Visualisierung
- **Features**: 2D-Embedding von Client-Features
- **Dependencies**: umap, plotly
- **Status**: ✅ Vollständig getestet

## 🔧 Plugin-Management

### Health Check
```bash
# Plugin-Gesundheit prüfen
python3 scripts/plugin_health_check.py

# Mit JSON-Output
python3 scripts/plugin_health_check.py --output health_report.json

# Über Makefile
make health-check
```

### Erweiterte Plugin-Verwaltung
```bash
# Health Check
python3 scripts/plugin_manager.py health

# Dependencies aktualisieren
python3 scripts/plugin_manager.py update

# Alle verfügbaren Befehle
python3 scripts/plugin_manager.py --help
```

## 🤝 Beitragen

1. **Fork** das Repository
2. **Erstelle** ein neues Plugin oder verbessere ein bestehendes
3. **Schreibe** Tests für dein Plugin
4. **Dokumentiere** dein Plugin
5. **Erstelle** einen Pull Request

## 📝 Changelog

### v1.0.0
- Neues Plugin-System implementiert
- Ordner-basierte Plugin-Struktur
- Automatische pytest-Integration
- Plugin-Manager-Tool
- Basis-Plugin-Klasse
- Dependencies-Management