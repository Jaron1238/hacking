# WLAN Analysis Tool 📡

Ein umfassendes, professionelles WLAN-Analyse-Tool mit **Machine Learning**, **Advanced Clustering**, **Real-time Monitoring** und **Enterprise-Features** zur Erfassung, Analyse und Klassifizierung von WiFi-Netzwerken und Clients.

## 🚀 **Neue Features (v2.0)**

### 🤖 **Machine Learning & AI**
- **Automatisches ML-Training** mit 8+ Algorithmen
- **Advanced Clustering** (K-Means, DBSCAN, Spectral, Hierarchical, etc.)
- **Behavioral Analysis Engine** für Anomalie-Erkennung
- **Device Classification** mit Ensemble-Methoden
- **Predictive Analytics** für Netzwerk-Verhalten

### 🔧 **Enterprise-Features**
- **Async/Await Support** für bessere Performance
- **Multi-Level Caching** (Memory + File)
- **Real-time Metrics Collection** mit Prometheus-Export
- **Health Check System** für alle Komponenten
- **Comprehensive Error Handling** mit Recovery-Mechanismen

### 📊 **Advanced Analytics**
- **Packet Replay & Injection** für Penetration Testing
- **3D Network Visualization** mit Three.js
- **Real-time Alerts System** (Email, SMS, Slack)
- **Forensic Analysis** mit Timeline Reconstruction
- **Compliance & Auditing** (GDPR, HIPAA)

## ✨ **Kern-Features**

- 📊 **WiFi-Erfassung**: Monitor-Mode Packet Capture mit adaptivem Channel-Hopping
- 🤖 **Machine Learning**: Client-Klassifizierung, Anomalie-Erkennung, Verhaltensanalyse
- 📈 **Advanced Analytics**: SSID-BSSID-Korrelation, Vendor-Lookup, Netzwerk-Mapping
- 🎨 **Visualisierung**: Terminal UI (TUI), Web Dashboard, 3D-Visualisierung, HTML-Reports
- 🔌 **Plugin-System**: Erweiterbare Analyse-Module
- 🏷️ **Interactive Labeling**: UI zum Trainieren von ML-Modellen
- 🔐 **Security Testing**: Penetration Testing, Vulnerability Scanning
- 📱 **Mobile Support**: React Native App, Web Dashboard

## 🛠️ **Voraussetzungen**

- **Python 3.8+** (empfohlen: 3.10+)
- **Linux** mit WLAN-Interface (Monitor-Mode fähig)
- **Root-Rechte** für Packet Capture
- **8GB RAM** (empfohlen für ML-Features)
- **10GB freier Speicherplatz** (für Modelle und Daten)

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
pip install -r requirements-ml.txt  # Für ML-Features
```

### **Docker Installation**
```bash
# Docker-Container erstellen
docker build -t wlan-tool .

# Container starten
docker run -it --privileged --net=host wlan-tool
```

## 🚀 **Schnellstart**

### **1. WiFi-Pakete erfassen**
```bash
# Basis-Capture
sudo python main.py --capture_mode --iface wlan0 --duration 300 --project my_scan

# Mit Live-TUI
sudo python main.py --capture_mode --iface wlan0 --live-tui --project my_scan

# Mit automatischem ML-Training
sudo python main.py --capture_mode --iface wlan0 --auto-ml --project my_scan
```

### **2. Daten analysieren**
```bash
# Basis-Analyse
python main.py --project my_scan --infer --cluster-clients --tui

# Mit ML-Klassifizierung
python main.py --project my_scan --ml-classify --behavior-analysis --tui

# Mit Advanced Clustering
python main.py --project my_scan --advanced-clustering --spectral --tui
```

### **3. Reports und Visualisierung**
```bash
# HTML-Report erstellen
python main.py --project my_scan --html-report report.html

# 3D-Visualisierung
python main.py --project my_scan --3d-visualization --export-3d network.html

# Graph-Export für Gephi
python main.py --project my_scan --export-graph network.gexf
```

## 🏗️ **Projektstruktur**

```
hacking/
├── main.py                    # Haupteinstiegspunkt
├── config.yaml               # Zentrale Konfiguration
├── requirements.txt          # Basis-Dependencies
├── requirements-ml.txt       # ML-Dependencies
├── requirements-test.txt     # Test-Dependencies
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
├── assets/                  # Statische Assets
│   ├── sql_data/           # SQL-Migrationen
│   ├── templates/          # HTML-Templates
│   └── css/                # CSS-Styles
└── docs/                   # Dokumentation
    ├── api/                # API-Dokumentation
    └── guides/             # Benutzerhandbücher
```

## 🎯 **Wichtige Kommandos**

### **Capture-Modi**
```bash
# Basis-Capture
--capture_mode --iface wlan0 --duration 300

# Adaptive Capture mit ML
--adaptive-scan --ml-training --duration 600

# Live-TUI während Capture
--capture_mode --live-tui --iface wlan0mon

# Multi-Interface Capture
--multi-interface --ifaces wlan0,wlan1 --duration 300

# PCAP-Export
--pcap capture.pcap --duration 300
```

### **Analyse-Optionen**
```bash
# Basis-Analyse
--infer --cluster-clients --tui

# ML-Klassifizierung
--ml-classify --model device_classifier --confidence 0.8

# Advanced Clustering
--advanced-clustering --algorithm spectral --clusters 8

# Behavioral Analysis
--behavior-analysis --anomaly-detection --real-time

# Graph-Export
--export-graph network.gexf --format gephi
```

### **Machine Learning**
```bash
# Automatisches ML-Training
--auto-ml --train-all --cross-validation

# Spezifisches Modell trainieren
--train-model device_classifier --data training_data.csv

# Modell-Evaluation
--evaluate-model --model device_classifier --test-data test.csv

# Hyperparameter-Tuning
--tune-hyperparameters --model device_classifier --grid-search
```

### **Security & Testing**
```bash
# Penetration Testing
--pentest --test-wps --test-deauth --test-evil-twin

# Vulnerability Scanning
--vuln-scan --cve-database --compliance-check

# Forensic Analysis
--forensic --timeline --evidence-collection

# Packet Replay
--replay-packets capture.pcap --delay 0.1
```

### **Visualisierung & Reports**
```bash
# HTML-Report
--html-report report.html --template custom

# 3D-Visualisierung
--3d-visualization --export-3d network.html

# Real-time Dashboard
--web-dashboard --port 8080 --real-time

# Mobile App
--mobile-app --react-native --build
```

## ⚙️ **Konfiguration**

### **config.yaml - Zentrale Konfiguration**
```yaml
# Interface-Konfiguration
interfaces:
  primary: "wlan0mon"
  secondary: "wlan1mon"
  channels: [1, 6, 11, 36, 40, 44, 48]

# ML-Konfiguration
machine_learning:
  auto_training: true
  models:
    device_classifier:
      algorithm: "random_forest"
      confidence_threshold: 0.8
    anomaly_detector:
      algorithm: "isolation_forest"
      contamination: 0.1

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
  max_workers: 4

# Monitoring-Konfiguration
monitoring:
  metrics_enabled: true
  health_checks: true
  alerting:
    email: "admin@example.com"
    slack_webhook: "https://hooks.slack.com/..."
```

### **Profile verwenden**
```bash
# Entwicklungs-Profil
python main.py --profile dev

# Produktions-Profil
python main.py --profile prod

# Test-Profil
python main.py --profile test
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
pytest pytest/ -m "not slow"    # Ohne langsame Tests

# Mit Coverage-Report
pytest pytest/ --cov=wlan_tool --cov-report=html

# Performance-Benchmarks
pytest pytest/test_performance.py --benchmark-only
```

### **Code-Qualität**
```bash
# Linting
flake8 wlan_tool/
black wlan_tool/
isort wlan_tool/

# Type-Checking
mypy wlan_tool/

# Security-Scan
bandit -r wlan_tool/
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
        
        # Analyse durchführen
        result = analyze_network(state)
        
        return result
        
    except Exception as e:
        raise WLANToolError(f"Plugin-Fehler: {e}") from e

def analyze_network(state):
    """Netzwerk-Analyse-Logik."""
    # Deine Analyse hier
    pass
```

### **Plugin registrieren**
```yaml
# config.yaml
plugins:
  analysis_my_plugin:
    enabled: true
    priority: 10
    config:
      custom_param: "value"
```

## 📊 **Monitoring & Metriken**

### **Real-time Monitoring**
```bash
# Metrics anzeigen
python main.py --show-metrics --format prometheus

# Health-Status prüfen
python main.py --health-check --detailed

# Performance-Monitoring
python main.py --performance-monitor --interval 60
```

### **Prometheus-Integration**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'wlan-tool'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## 🚀 **Performance-Optimierung**

### **Async-Modus aktivieren**
```bash
# Asynchrone Verarbeitung
python main.py --async-mode --max-workers 8

# Caching aktivieren
python main.py --enable-caching --cache-size 10000

# Memory-Optimierung
python main.py --memory-optimize --batch-size 500
```

### **Skalierung**
```bash
# Multi-Process-Modus
python main.py --multi-process --processes 4

# Distributed Processing
python main.py --distributed --workers 192.168.1.100:8080,192.168.1.101:8080
```

## 🔐 **Sicherheit & Compliance**

### **Sicherheits-Features**
- **Verschlüsselungsanalyse** (WPA/WPA2/WPA3)
- **Schwachstellen-Scanner** mit CVE-Datenbank
- **Forensische Analyse** mit Timeline-Reconstruction
- **Compliance-Checking** (GDPR, HIPAA)

### **Penetration Testing**
```bash
# WPS-Schwachstellen testen
python main.py --pentest --test-wps --target-ssid "TestNetwork"

# Deauthentication-Angriffe simulieren
python main.py --pentest --test-deauth --target-mac "aa:bb:cc:dd:ee:ff"

# Evil Twin erstellen
python main.py --pentest --evil-twin --original-ssid "OriginalNetwork"
```

## 📱 **Mobile & Web-Interfaces**

### **Web-Dashboard**
```bash
# Web-Interface starten
python main.py --web-dashboard --port 8080 --host 0.0.0.0

# Mit Real-time Updates
python main.py --web-dashboard --real-time --websocket
```

### **Mobile App**
```bash
# React Native App bauen
python main.py --mobile-app --build --platform android

# iOS App bauen
python main.py --mobile-app --build --platform ios
```

## 🐳 **Docker & Containerisierung**

### **Docker verwenden**
```bash
# Container bauen
docker build -t wlan-tool .

# Container starten
docker run -it --privileged --net=host wlan-tool

# Mit Volume-Mount
docker run -it --privileged --net=host -v $(pwd)/data:/app/data wlan-tool
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  wlan-tool:
    build: .
    privileged: true
    network_mode: host
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - PYTHONPATH=/app
```

## 🚨 **Troubleshooting**

### **Häufige Probleme**

#### **Import-Fehler**
```bash
# Dependencies installieren
pip install -r requirements.txt
pip install -r requirements-ml.txt

# Virtual Environment aktivieren
source .venv/bin/activate
```

#### **Keine Pakete erfasst**
```bash
# Interface in Monitor-Mode setzen
sudo airmon-ng start wlan0

# Interface-Status prüfen
iwconfig wlan0mon

# Rechte prüfen
sudo -v
```

#### **ML-Modelle laden nicht**
```bash
# Modelle neu trainieren
python main.py --train-all --force-retrain

# Modelle herunterladen
python main.py --download-models
```

#### **Performance-Probleme**
```bash
# Async-Modus aktivieren
python main.py --async-mode

# Caching aktivieren
python main.py --enable-caching

# Memory-Limit setzen
python main.py --memory-limit 4GB
```

### **Debug-Modus**
```bash
# Ausführliche Logs
python main.py --debug --verbose --log-level DEBUG

# Profiling aktivieren
python main.py --profile --profile-output profile.prof

# Memory-Profiling
python main.py --memory-profile --memray
```

## 📚 **Dokumentation**

### **API-Dokumentation**
- **Swagger/OpenAPI**: `http://localhost:8080/docs`
- **Code-Dokumentation**: `docs/api/`
- **Beispiele**: `docs/examples/`

### **Benutzerhandbücher**
- **Schnellstart**: `docs/guides/quickstart.md`
- **ML-Guide**: `docs/guides/machine_learning.md`
- **Plugin-Entwicklung**: `docs/guides/plugin_development.md`
- **Deployment**: `docs/guides/deployment.md`

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

### **Issue-Reporting**
- **Bug-Reports**: Verwende das Bug-Template
- **Feature-Requests**: Verwende das Feature-Template
- **Security-Issues**: Kontaktiere uns privat

## 📄 **Lizenz & Hinweise**

### **Lizenz**
Dieses Projekt steht unter der **MIT-Lizenz**. Siehe `LICENSE` für Details.

### **Wichtige Hinweise**
⚠️ **Rechtliche Hinweise**:
- Dieses Tool ist nur für **autorisierte Netzwerk-Analysen** gedacht
- Verwenden Sie es nur in Netzwerken, für die Sie die **Berechtigung** haben
- **Penetration Testing** nur mit schriftlicher Erlaubnis
- **Compliance** mit lokalen Gesetzen beachten

### **Disclaimer**
Die Entwickler übernehmen keine Verantwortung für den Missbrauch dieses Tools. Verwenden Sie es verantwortungsvoll und ethisch.

## 🏆 **Acknowledgments**

- **Scapy** für Packet-Manipulation
- **scikit-learn** für Machine Learning
- **Textual** für Terminal-UI
- **Rich** für schöne Konsolen-Ausgaben
- **SQLAlchemy** für Datenbank-ORM
- **FastAPI** für Web-API

## 📞 **Support & Community**

- **GitHub Issues**: [Issues](https://github.com/Jaron1238/hacking/issues)
- **Discussions**: [Discussions](https://github.com/Jaron1238/hacking/discussions)
- **Wiki**: [Wiki](https://github.com/Jaron1238/hacking/wiki)
- **Discord**: [Discord-Server](https://discord.gg/wlan-tool)

---

**Made with ❤️ for the WiFi Security Community**