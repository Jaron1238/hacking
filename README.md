# WLAN Analysis Tool 📡

Ein umfassendes WLAN-Analyse-Tool mit **Machine Learning**, **Advanced Clustering** und **Enterprise-Features** zur Erfassung, Analyse und Klassifizierung von WiFi-Netzwerken und Clients.

## 🚀 **Neue Features (v2.0)**

### 🤖 **Machine Learning & AI**
- **Automatisches ML-Training** mit verschiedenen Algorithmen
- **Advanced Clustering** (K-Means, DBSCAN, Spectral, Hierarchical, etc.)
- **Device Classification** mit Ensemble-Methoden
- **Anomaly Detection** mit Isolation Forest und One-Class SVM
- **Behavior Prediction** mit MLP Regressor

### 🔧 **Enterprise-Features**
- **Async/Await Support** für bessere Performance
- **Multi-Level Caching** (Memory + File)
- **Real-time Metrics Collection** mit Prometheus-Export
- **Health Check System** für alle Komponenten
- **Comprehensive Error Handling** mit Recovery-Mechanismen

### 📊 **Code-Qualität & Wartbarkeit**
- **Vollständige Type Hints** für bessere IDE-Unterstützung
- **Zentrale Konstanten** ohne Magic Numbers
- **Robuste Input-Validation** mit detaillierten Fehlermeldungen
- **Strukturierte Error-Codes** für bessere Fehlerbehandlung
- **Umfassende Test-Suite** mit 80%+ Coverage

## ✨ **Kern-Features**

- 📊 **WiFi-Erfassung**: Monitor-Mode Packet Capture mit adaptivem Channel-Hopping
- 🤖 **Machine Learning**: Client-Klassifizierung, Anomalie-Erkennung, Verhaltensanalyse
- 📈 **Analyse**: SSID-BSSID-Korrelation, Vendor-Lookup, Netzwerk-Mapping
- 🎨 **Visualisierung**: Terminal UI (TUI), Live-TUI, HTML-Reports, Graph-Export
- 🔌 **Plugin-System**: Erweiterbare Analyse-Module
- 🏷️ **Interactive Labeling**: UI zum Trainieren von ML-Modellen
- 📊 **Real-time Monitoring**: Live-TUI während Capture
- 🔍 **Device Fingerprinting**: IE-Order-Hash und Packet-Timing

## 🛠️ **Voraussetzungen**

- **Python 3.8+** (empfohlen: 3.10+)
- **Linux** mit WLAN-Interface (Monitor-Mode fähig)
- **Root-Rechte** für Packet Capture
- **4GB RAM** (empfohlen für ML-Features)
- **2GB freier Speicherplatz** (für Modelle und Daten)
- **Wurde auf einem Raspberry Pi 4 mit Nexmon getestet**


## 📦 **Installation**

### **Schnelle Installation**
```bash
# Repository klonen
git clone https://github.com/Jaron1238/hacking.git
cd hacking

# Automatische Installation
source setup.sh

# Oder manuell
pip install -r requirements.txt
```

## 🚀 **Schnellstart**

### **1. WiFi-Pakete erfassen**
```bash
# Basis-Capture
sudo python main.py --capture_mode --iface wlan0 --duration 300 --project my_scan

# Mit Live-TUI
sudo python main.py --capture_mode --iface wlan0 --live-tui --project my_scan
```

### **2. Daten analysieren**
```bash
# Basis-Analyse
python main.py --project my_scan --infer --cluster-clients --tui

# Mit ML-Klassifizierung
python main.py --project my_scan --ml-classify --tui
```

### **3. Reports erstellen**
```bash
# HTML-Report erstellen
python main.py --project my_scan --html-report report.html

# Graph-Export für Gephi
python main.py --project my_scan --export-graph network.gexf
```

## 🏗️ **Projektstruktur**

```
hacking/
├── main.py                    # Haupteinstiegspunkt
├── config.yaml               # Zentrale Konfiguration
├── requirements.txt          # Dependencies
├── wlan_tool/               # Haupt-Package
│   ├── __init__.py
│   ├── constants.py         # Zentrale Konstanten & Error-Codes
│   ├── validation.py        # Input-Validation-System
│   ├── exceptions.py        # Exception-Hierarchie
│   ├── logging_config.py    # Logging-System
│   ├── recovery.py          # Error-Recovery-System
│   ├── async_utils.py       # Async-Utilities
│   ├── caching.py           # Multi-Level-Caching
│   ├── metrics.py           # Metrics-Collection
│   ├── health.py            # Health-Check-System
│   ├── capture/             # Packet-Capture-Module
│   │   ├── sniffer.py       # Packet-Sniffer
│   │   └── channel_hopper.py # Channel-Hopping
│   ├── analysis/            # Analyse-Module
│   │   ├── logic.py         # Analyse-Logik
│   │   └── device_profiler.py # Device-Profiling
│   ├── presentation/        # UI-Module
│   │   ├── cli.py           # Command-Line-Interface
│   │   ├── tui.py           # Terminal-UI
│   │   ├── live_tui.py      # Live-TUI
│   │   └── reporting.py     # Report-Generierung
│   ├── storage/             # Daten-Speicherung
│   │   ├── database.py      # SQLite-Datenbank
│   │   ├── state.py         # State-Management
│   │   └── data_models.py   # Datenmodelle
│   ├── ml/                  # Machine Learning
│   │   ├── training.py      # ML-Training
│   │   ├── models.py        # ML-Modelle
│   │   ├── clustering.py    # Clustering-Algorithmen
│   │   ├── inference.py     # ML-Inference
│   │   └── evaluation.py    # Modell-Evaluation
│   └── utils/               # Utilities
│       ├── oui.py           # OUI-Lookup
│       └── ie_parser.py     # IE-Parser
├── pytest/                  # Umfassende Test-Suite
│   ├── conftest.py          # Test-Konfiguration
│   ├── test_*.py            # Unit-Tests
│   ├── test_integration.py  # Integration-Tests
│   ├── test_performance.py  # Performance-Tests
│   └── test_error_handling.py # Error-Handling-Tests
└── assets/                  # Statische Assets
    ├── sql_data/           # SQL-Migrationen
    ├── templates/          # HTML-Templates
    └── css/                # CSS-Styles
```

## 🎯 **Wichtige Kommandos**

### **Capture-Modi**
```bash
# Basis-Capture
--capture_mode --iface wlan0 --duration 300

# Mit Live-TUI
--capture_mode --live-tui --iface wlan0mon

# PCAP-Export
--pcap capture.pcap --duration 300
```

### **Analyse-Optionen**
```bash
# Basis-Analyse
--infer --cluster-clients --tui

# ML-Klassifizierung
--ml-classify --model device_classifier

# Graph-Export
--export-graph network.gexf --format gephi
```

### **Machine Learning**
```bash
# Automatisches ML-Training
--auto-ml --train-all

# Spezifisches Modell trainieren
--train-model device_classifier

# Modell-Evaluation
--evaluate-model --model device_classifier
```

## ⚙️ **Konfiguration**

### **config.yaml - Zentrale Konfiguration**
```yaml
# Interface-Konfiguration
interfaces:
  primary: "wlan0mon"
  channels: [1, 6, 11, 36, 40, 44, 48]

# ML-Konfiguration
machine_learning:
  auto_training: true
  models:
    device_classifier:
      algorithm: "random_forest"
      confidence_threshold: 0.8

# Clustering-Konfiguration
clustering:
  algorithms: ["kmeans", "dbscan", "spectral"]
  auto_select: true
  max_clusters: 20

# Performance-Konfiguration
performance:
  async_mode: true
  cache_size: 1000
  batch_size: 100
```

## 🧪 **Testing & Qualitätssicherung**

### **Test-Suite ausführen**
```bash
# Alle Tests
pytest pytest/

# Spezifische Test-Kategorien
pytest pytest/ -m unit          # Unit-Tests
pytest pytest/ -m integration   # Integration-Tests
pytest pytest/ -m performance   # Performance-Tests

# Mit Coverage-Report
pytest pytest/ --cov=wlan_tool --cov-report=html
```

### **Code-Qualität**
```bash
# Linting
flake8 wlan_tool/
black wlan_tool/
isort wlan_tool/

# Type-Checking
mypy wlan_tool/
```

## 🔌 **Plugin-System**

### **Neues Plugin erstellen**
```python
# wlan_tool/plugins/analysis_my_plugin.py
from wlan_tool.exceptions import WLANToolError
from wlan_tool.validation import validate_dataframe

def run(state, console, args, **kwargs):
    """Mein Custom-Plugin."""
    try:
        # Plugin-Logik hier
        console.print("[bold green]Mein Plugin läuft![/bold green]")
        
        # Daten validieren
        validate_dataframe(state.clients_df)
        
        return result
        
    except Exception as e:
        raise WLANToolError(f"Plugin-Fehler: {e}") from e
```

## 📊 **Monitoring & Metriken**

### **Real-time Monitoring**
```bash
# Metrics anzeigen
python main.py --show-metrics

# Health-Status prüfen
python main.py --health-check
```

## 🚀 **Performance-Optimierung**

### **Async-Modus aktivieren**
```bash
# Asynchrone Verarbeitung
python main.py --async-mode

# Caching aktivieren
python main.py --enable-caching
```

## 🚨 **Troubleshooting**

### **Häufige Probleme**

#### **Import-Fehler**
```bash
# Dependencies installieren
pip install -r requirements.txt

# Virtual Environment aktivieren
source .venv/bin/activate
```

#### **Keine Pakete erfasst**
```bash
# Interface in Monitor-Mode setzen
sudo airmon-ng start wlan0

# Rechte prüfen
sudo -v
```

#### **ML-Modelle laden nicht**
```bash
# Modelle neu trainieren
python main.py --train-all
```

### **Debug-Modus**
```bash
# Ausführliche Logs
python main.py --debug --verbose --log-level DEBUG
```

## 📚 **Dokumentation**

### **API-Dokumentation**
- **Code-Dokumentation**: Inline Docstrings
- **Beispiele**: In den Modulen

## 🤝 **Mitwirken**

### **Entwicklung**
1. **Fork** das Repository
2. **Feature-Branch** erstellen: `git checkout -b feature/amazing-feature`
3. **Änderungen committen**: `git commit -m 'Add amazing feature'`
4. **Branch pushen**: `git push origin feature/amazing-feature`
5. **Pull Request** erstellen

### **Code-Standards**
- **Python**: PEP 8, Black, isort
- **Tests**: pytest mit 80%+ Coverage
- **Dokumentation**: Google-Style Docstrings
- **Type Hints**: Vollständige Type-Annotationen

## 📄 **Lizenz & Hinweise**

### **Lizenz**
Dieses Projekt steht unter der **MIT-Lizenz**.

### **Wichtige Hinweise**
⚠️ **Rechtliche Hinweise**:
- Dieses Tool ist nur für **autorisierte Netzwerk-Analysen** gedacht
- Verwenden Sie es nur in Netzwerken, für die Sie die **Berechtigung** haben
- **Compliance** mit lokalen Gesetzen beachten

### **Disclaimer**
Die Entwickler übernehmen keine Verantwortung für den Missbrauch dieses Tools. Verwenden Sie es verantwortungsvoll und ethisch.

## 🏆 **Acknowledgments**

- **Scapy** für Packet-Manipulation
- **scikit-learn** für Machine Learning
- **Textual** für Terminal-UI
- **Rich** für schöne Konsolen-Ausgaben
- **SQLAlchemy** für Datenbank-ORM

---

**Made with ❤️ for the WiFi Security Community**
