# WLAN Analysis Tool - Kommandozeilen-Argumente

Diese Dokumentation beschreibt alle verfügbaren Kommandozeilen-Argumente des WLAN Analysis Tools.

## 📋 **Übersicht**

Das Tool unterstützt zwei Hauptmodi:
- **Capture Mode** (`--capture_mode`): Erfassung von WiFi-Paketen
- **Analysis Mode**: Analyse und Visualisierung der erfassten Daten

---

## 🎯 **Capture Mode Argumente**

### **Basis-Capture**
```bash
--capture_mode
```
Aktiviert den Capture-Modus für die Erfassung von WiFi-Paketen.

### **Interface & Hardware**
```bash
--iface INTERFACE
```
WiFi-Interface für Packet Capture (z.B. `wlan0mon`, `wlan1mon`).

```bash
--pcap PCAP_FILE
```
Spezifische PCAP-Datei für Analyse (anstatt Live-Capture).

### **Zeit & Dauer**
```bash
--duration SECONDS
```
Capture-Dauer in Sekunden (Standard: 120).

```bash
--project PROJECT_NAME
```
Projektname für organisierte Datenspeicherung.

### **Ausgabe & Speicher**
```bash
--outdir OUTPUT_DIRECTORY
```
Ausgabeverzeichnis für Capture-Daten (Standard: aktuelles Verzeichnis).

```bash
--db DATABASE_PATH
```
Spezifischer Pfad zur SQLite-Datenbank.

### **Erweiterte Capture-Features**
```bash
--live-tui
```
Aktiviert Live-Terminal-UI während des Captures.

```bash
--enhanced-analysis
```
Aktiviert erweiterte Analyse mit DPI, Metriken und Visualisierung.

```bash
--deep-packet-inspection
```
Aktiviert Deep Packet Inspection für HTTP, DNS, DHCP Analyse.

```bash
--advanced-metrics
```
Berechnet erweiterte Metriken (SNR, PER, Traffic Patterns).

---

## 🔍 **Analysis Mode Argumente**

### **Basis-Analyse**
```bash
--infer
```
Führt grundlegende Inferenz und Datenverarbeitung durch.

```bash
--project PROJECT_NAME
```
Projektname für Analyse (muss existieren).

### **Machine Learning**
```bash
--ml-classify
```
Führt ML-basierte Geräteklassifizierung durch.

```bash
--auto-ml
```
Aktiviert automatisches ML-Training mit allen verfügbaren Algorithmen.

```bash
--train-all
```
Trainiert alle ML-Modelle mit den verfügbaren Daten.

```bash
--train-model MODEL_NAME
```
Trainiert ein spezifisches ML-Modell.

```bash
--evaluate-model --model MODEL_NAME
```
Evaluiert ein trainiertes ML-Modell.

### **Clustering & Klassifizierung**
```bash
--cluster-clients N_CLUSTERS
```
Führt Client-Clustering mit N Clustern durch.

```bash
--cluster-aps N_CLUSTERS
```
Führt Access Point-Clustering durch.

```bash
--classify-clients
```
Klassifiziert Clients basierend auf trainierten Modellen.

```bash
--classify-client-behavior
```
Klassifiziert Client-Verhalten mit ML-Modellen.

### **Labeling & Training**
```bash
--label-clients
```
Startet interaktives Labeling für Client-Training.

```bash
--label-ui
```
Öffnet Labeling-UI für manuelle Datenannotation.

```bash
--auto-retrain
```
Automatisches Re-Training der Modelle nach neuen Labels.

### **Verhaltensanalyse**
```bash
--train-behavior-model
```
Trainiert Verhaltensvorhersage-Modelle.

```bash
--profile-from-router
```
Erstellt Geräteprofile basierend auf Router-Daten.

### **MAC-Adress-Korrelation**
```bash
--correlate-macs
```
Korrelliert randomisierte MAC-Adressen basierend auf Probe-Listen.

```bash
--no-mac-correlation
```
Deaktiviert automatische MAC-Adress-Korrelation.

### **Autoencoder & Clustering**
```bash
--train-encoder MODEL_PATH
```
Trainiert einen Autoencoder für besseres Client-Clustering.

```bash
--use-encoder MODEL_PATH
```
Verwendet einen trainierten Encoder für Clustering.

---

## 📊 **Visualisierung & Reports**

### **Terminal-UI**
```bash
--tui
```
Startet interaktive Terminal-Benutzeroberfläche.

```bash
--live-tui
```
Live-Terminal-UI während Capture.

### **HTML-Reports**
```bash
--html-report REPORT_FILE
```
Erstellt HTML-Report (z.B. `report.html`).

```bash
--custom-report
```
Generiert benutzerdefinierten HTML-Report mit Visualisierungen.

### **Graph-Export**
```bash
--export-graph GRAPH_FILE
```
Exportiert Netzwerk-Graph für Gephi (z.B. `network.gexf`).

```bash
--export-csv CSV_FILE
```
Exportiert Daten als CSV-Datei.

### **Erweiterte Visualisierung**
```bash
--3d-visualization
```
Erstellt 3D-Netzwerk-Visualisierung.

```bash
--time-series-plots
```
Erstellt detaillierte Zeitverlaufs-Diagramme.

### **Netzwerk-Mapping**
```bash
--show-network-map
```
Zeigt interaktive Netzwerk-Karte.

```bash
--show-dns
```
Zeigt DNS-Abfragen und -Antworten.

```bash
--show-probed-ssids
```
Zeigt Rangliste der am häufigsten gesuchten SSIDs.

---

## 🔌 **Plugin-System**

### **Plugin-Management**
```bash
--run-plugins [PLUGIN_NAMES]
```
Führt spezifische Analyse-Plugins aus.

**Verfügbare Plugins:**
- `umap_plot` - UMAP-Dimensionalitätsreduktion
- `sankey` - Sankey-Diagramm für Datenflüsse
- `heatmap` - Signal-Quality-Heatmap
- `timeline` - Zeitverlaufs-Visualisierung

**Beispiele:**
```bash
# Alle Plugins ausführen
--run-plugins

# Spezifische Plugins
--run-plugins umap_plot sankey

# Einzelnes Plugin
--run-plugins heatmap
```

---

## ⚙️ **Konfiguration & Einstellungen**

### **Konfigurationsdateien**
```bash
--profile PROFILE_NAME
```
Verwendet spezifisches Konfigurationsprofil.

```bash
--config CONFIG_FILE
```
Lädt Konfiguration aus spezifischer Datei.

### **Debug & Logging**
```bash
--debug
```
Aktiviert Debug-Modus mit detailliertem Logging.

```bash
--verbose
```
Aktiviert ausführliche Ausgabe.

```bash
--log-level LEVEL
```
Setzt Log-Level (DEBUG, INFO, WARNING, ERROR).

### **Datenbank-Einstellungen**
```bash
--detailed-ies
```
Aktiviert detaillierte Information Element-Analyse.

```bash
--update-oui
```
Aktualisiert OUI-Datenbank vor Capture.

---

## 🚀 **Erweiterte Analyse-Features**

### **Umfassende Analyse**
```bash
--enhanced-analysis
```
Aktiviert alle erweiterten Analyse-Features:
- Deep Packet Inspection
- Erweiterte Metriken
- 3D-Visualisierung
- Zeitverlaufs-Diagramme
- Custom Reports

### **Performance & Monitoring**
```bash
--performance-monitor
```
Aktiviert Performance-Monitoring während der Analyse.

```bash
--memory-optimization
```
Aktiviert Speicher-Optimierungen für große Datensätze.

---

## 📝 **Beispiele für häufige Verwendungen**

### **Schnelle Analyse**
```bash
# Basis-Capture und Analyse
sudo python main.py --capture_mode --iface wlan0 --duration 300 --project quick_scan
python main.py --project quick_scan --infer --cluster-clients 5 --tui
```

### **Erweiterte Analyse mit ML**
```bash
# Capture mit erweiterten Features
sudo python main.py --capture_mode --iface wlan0 --enhanced-analysis --project ml_scan

# ML-Analyse
python main.py --project ml_scan --auto-ml --classify-clients --3d-visualization
```

### **Plugin-basierte Analyse**
```bash
# Alle verfügbaren Plugins
python main.py --project my_scan --run-plugins --html-report analysis.html

# Spezifische Visualisierungen
python main.py --project my_scan --run-plugins umap_plot sankey --export-graph network.gexf
```

### **Professionelle Reports**
```bash
# Umfassender Report mit allen Features
python main.py --project my_scan --enhanced-analysis --custom-report --3d-visualization --time-series-plots
```

### **Debugging und Entwicklung**
```bash
# Debug-Modus mit detailliertem Logging
sudo python main.py --capture_mode --iface wlan0 --debug --verbose --project debug_scan
```

---

## 🔧 **Technische Details**

### **Erforderliche Berechtigungen**
- **Root-Rechte** für Packet Capture (`sudo` erforderlich)
- **Monitor-Mode** für WiFi-Interface
- **Schreibberechtigung** für Ausgabeverzeichnis

### **Performance-Empfehlungen**
- **RAM**: Mindestens 4GB für ML-Features
- **CPU**: Multi-Core für parallele Verarbeitung
- **Speicher**: 2GB+ für Modelle und Daten

### **Unterstützte Formate**
- **Input**: PCAP-Dateien, Live-Capture
- **Output**: HTML, CSV, GEXF, JSON, PNG, SVG

---

## ❓ **Hilfe & Support**

### **Hilfe anzeigen**
```bash
python main.py --help
```

### **Spezifische Hilfe**
```bash
python main.py --capture_mode --help
python main.py --analysis --help
```

### **Version anzeigen**
```bash
python main.py --version
```

---

## 📚 **Weitere Ressourcen**

- **README.md** - Hauptdokumentation
- **examples/** - Beispiel-Skripte
- **docs/** - Detaillierte Dokumentation
- **tests/** - Test-Suite und Beispiele

---

*Letzte Aktualisierung: Januar 2025*