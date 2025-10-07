# WLAN Analysis Tool 📡

Ein umfassendes WLAN-Analyse-Tool mit Machine Learning-Funktionen zur Erfassung, Analyse und Klassifizierung von WiFi-Netzwerken und Clients.

## Features

- 📊 **WiFi-Erfassung**: Monitor-Mode Packet Capture mit adaptivem Channel-Hopping
- 🤖 **Machine Learning**: Client-Klassifizierung und Clustering
- 📈 **Analyse**: SSID-BSSID-Korrelation, Vendor-Lookup, Netzwerk-Mapping
- 🎨 **Visualisierung**: Terminal-basierte UI (TUI), HTML-Reports, Graph-Export
- 🔌 **Plugin-System**: Erweiterbare Analyse-Module
- 🏷️ **Labeling**: Interaktive UI zum Trainieren von ML-Modellen

## Voraussetzungen

- Python 3.8+
- Linux mit WLAN-Interface (Monitor-Mode fähig)
- Root-Rechte für Packet Capture

## Installation

```bash
source setup.sh
```
## Schnellstart

### 1. WiFi-Pakete erfassen
```bash
sudo python main.py --capture_mode --iface wlan0 --duration 300 --project my_scan
```

### 2. Daten analysieren
```bash
python main.py --project my_scan --infer --cluster-clients --tui
```

### 3. HTML-Report erstellen
```bash
python main.py --project my_scan --html-report report.html
```

## Projektstruktur

```
hacking/
├── main.py              # Haupteinstiegspunkt
├── config.yaml          # Konfiguration
├── requirements.txt     # Python-Dependencies
├── data/                # Core-Module
│   ├── capture.py       # Packet Capture
│   ├── analysis.py      # Analyse-Logik
│   ├── controllers.py   # MVC-Controller
│   ├── database.py      # SQLite-Datenbankzugriff
│   ├── ml_training.py   # ML-Training
│   ├── tui.py          # Terminal UI
│   └── plugins/         # Analyse-Plugins
├── test/                # Unit-Tests
└── .venv/              # Virtual Environment (nicht im Git)
```

## Wichtige Kommandos

### Capture-Modi
- `--adaptive-scan`: Intelligentes Channel-Hopping
- `--pcap FILE`: Pakete in PCAP speichern
- `--duration SECS`: Erfassungsdauer

### Analyse-Optionen
- `--infer`: SSID-BSSID-Korrelation
- `--cluster-clients N`: Client-Clustering
- `--classify-clients MODEL`: ML-Klassifizierung
- `--show-dns`: DNS-Query-Analyse
- `--export-graph FILE`: Gephi-Export

### Training
- `--label-clients`: Interaktives Client-Labeling
- `--train-behavior-model FILE`: Verhaltensmodell trainieren
- `--auto-retrain`: Automatisches Neutraining

## Konfiguration

Die `config.yaml` enthält alle Standard-Einstellungen:
- Interface-Name
- Scan-Dauer und -Parameter
- Datenbankpfade
- Adaptive-Scan-Optionen

Profile können mit `--profile NAME` geladen werden.

## Entwicklung

### Tests ausführen
```bash
pytest pytest/
```

### Neues Plugin erstellen
```python
# data/plugins/analysis_my_plugin.py
def run(state, console, args, **kwargs):
    console.print("[bold]Mein Plugin läuft![/bold]")
    # Deine Analyse-Logik hier
```

## Lizenz & Hinweise

⚠️ **Wichtig**: Dieses Tool ist nur für autorisierte Netzwerk-Analysen gedacht. 
Verwenden Sie es nur in Netzwerken, für die Sie die Berechtigung haben.

## Troubleshooting

- **Import-Fehler**: `pip install -r requirements.txt` ausführen
- **Keine Pakete erfasst**: Interface in Monitor-Mode setzen
- **Rechte-Fehler**: Mit `sudo` ausführen für Packet Capture

## Mitwirken

Pull Requests sind willkommen! Bitte beachten Sie:
1. Code-Stil konsistent halten
2. Tests für neue Features hinzufügen
3. Dokumentation aktualisieren
