# WLAN-Analyse-Tool Test Suite

Diese Test-Suite bietet umfassende Tests für das WLAN-Analyse-Tool mit verschiedenen Test-Kategorien und Performance-Messungen.

## 📁 Test-Struktur

```
pytest/
├── conftest.py              # Gemeinsame Fixtures und Konfiguration
├── test_storage.py          # Tests für Storage-Module (state, database, data_models)
├── test_analysis.py         # Tests für Analysis-Module (logic, device_profiler)
├── test_capture.py          # Tests für Capture-Module (sniffer)
├── test_presentation.py     # Tests für Presentation-Module (cli, tui, reporting)
├── test_utils.py            # Tests für Utils-Module
├── test_controllers.py      # Tests für Controller-Module
├── test_integration.py      # Integrationstests
├── test_performance.py      # Performance-Tests
├── pytest.ini              # Pytest-Konfiguration
├── requirements-test.txt    # Test-Abhängigkeiten
├── Makefile                 # Test-Ausführungs-Ziele
└── README.md               # Diese Datei
```

## 🚀 Schnellstart

### Installation der Test-Abhängigkeiten

```bash
pip install -r requirements-test.txt
```

### Alle Tests ausführen

```bash
make test
# oder
pytest -v
```

### Tests mit Coverage-Report

```bash
make test-coverage
# oder
pytest --cov=wlan_tool --cov-report=html:htmlcov
```

## 📊 Test-Kategorien

### 1. Unit-Tests
Testen einzelne Funktionen und Klassen isoliert:

```bash
make test-unit
# oder
pytest -m "not integration and not performance"
```

**Abgedeckte Module:**
- `test_storage.py` - State-Management, Datenbank, Datenmodelle
- `test_analysis.py` - Analyse-Logik, Clustering, Inferenz
- `test_capture.py` - Paket-Parsing, Channel-Hopping
- `test_presentation.py` - CLI, TUI, Reporting
- `test_utils.py` - Utility-Funktionen, OUI-Lookup
- `test_controllers.py` - Controller-Logik

### 2. Integrationstests
Testen das Zusammenspiel verschiedener Module:

```bash
make test-integration
# oder
pytest -m "integration"
```

**Abgedeckte Bereiche:**
- End-to-End-Workflows
- Datenfluss zwischen Modulen
- Datenbank-Integration
- Konfigurations-Integration
- Fehlerbehandlung und Recovery

### 3. Performance-Tests
Messen und validieren Performance-Charakteristika:

```bash
make test-performance
# oder
pytest -m "performance"
```

**Gemessene Metriken:**
- Paket-Verarbeitungsgeschwindigkeit
- Analyse-Performance
- Speicherverbrauch
- Datenbank-Performance
- Skalierbarkeit

## 🏷️ Test-Marker

Die Tests sind mit verschiedenen Markern kategorisiert:

- `@pytest.mark.slow` - Langsame Tests (> 1 Sekunde)
- `@pytest.mark.integration` - Integrationstests
- `@pytest.mark.unit` - Unit-Tests
- `@pytest.mark.performance` - Performance-Tests
- `@pytest.mark.network` - Tests mit Netzwerk-Zugriff
- `@pytest.mark.hardware` - Tests mit Hardware-Anforderungen

### Tests nach Marker ausführen

```bash
# Nur schnelle Tests
make test-fast
pytest -m "not slow"

# Nur langsame Tests
make test-slow
pytest -m "slow"

# Tests mit Netzwerk-Zugriff
make test-network
pytest -m "network"
```

## 📈 Coverage-Report

### HTML-Report generieren

```bash
make test-coverage
```

Der HTML-Report wird in `htmlcov/index.html` generiert.

### Coverage-Schwellenwerte

- **Minimum Coverage**: 80%
- **Aktuelle Konfiguration**: Siehe `pytest.ini`

### Coverage-Report anzeigen

```bash
# Terminal-Report
pytest --cov=wlan_tool --cov-report=term-missing

# XML-Report (für CI/CD)
pytest --cov=wlan_tool --cov-report=xml
```

## 🔧 Test-Konfiguration

### pytest.ini

Die Test-Konfiguration ist in `pytest.ini` definiert:

- **Python-Pfad**: Automatisch auf `..` gesetzt
- **Test-Pfade**: Aktuelles Verzeichnis
- **Coverage**: 80% Mindest-Coverage
- **Warnings**: Gefiltert für bessere Lesbarkeit
- **Output**: Farbig und detailliert

### Anpassung der Konfiguration

```ini
# pytest.ini anpassen
[tool:pytest]
addopts = -v --tb=short --cov=wlan_tool --cov-fail-under=80
```

## 🐛 Debugging

### Tests im Debug-Modus ausführen

```bash
make test-debug
# oder
pytest -vv --tb=long --pdb
```

### Einzelne Tests ausführen

```bash
# Spezifische Test-Datei
pytest test_storage.py -v

# Spezifische Test-Klasse
pytest test_storage.py::TestWifiAnalysisState -v

# Spezifische Test-Funktion
pytest test_storage.py::TestWifiAnalysisState::test_state_initialization -v
```

### Verbose-Ausgabe

```bash
make test-verbose
# oder
pytest -vv --tb=long
```

## 📊 Performance-Benchmarks

### Benchmark-Ergebnisse anzeigen

```bash
pytest test_performance.py --benchmark-only
```

### Performance-Tests mit Profiling

```bash
# Memory-Profiling
pytest test_performance.py::TestMemoryPerformance -v

# CPU-Profiling
pytest test_performance.py::TestAnalysisPerformance -v --profile
```

## 🧹 Cleanup

### Test-Artefakte bereinigen

```bash
make clean
```

Entfernt:
- Coverage-Reports (`htmlcov/`, `.coverage`)
- Cache-Dateien (`.pytest_cache/`)
- Temporäre Datenbanken (`*.db`)
- Test-Ausgabedateien (`/tmp/test_*`)

## 🔄 CI/CD-Integration

### GitHub Actions Beispiel

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r pytest/requirements-test.txt
    - name: Run tests
      run: |
        cd pytest
        make test-coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Lokale CI-Simulation

```bash
# Alle Tests mit Coverage
make test-coverage

# Nur kritische Tests
make test-unit

# Performance-Tests
make test-performance
```

## 📝 Test-Dokumentation

### Fixtures

Alle gemeinsamen Fixtures sind in `conftest.py` definiert:

- `sample_events` - Test-WiFi-Events
- `populated_state` - WifiAnalysisState mit Testdaten
- `in_memory_db` - In-Memory SQLite-Datenbank
- `mock_console` - Mock Console-Objekt
- `sample_scapy_packet` - Test-Scapy-Paket

### Test-Daten

Test-Daten werden dynamisch generiert und sind in den Fixtures definiert. Für reproduzierbare Tests werden feste Zeitstempel verwendet.

### Mocking

Umfangreiches Mocking für:
- Externe Abhängigkeiten (scapy, sklearn, etc.)
- System-Aufrufe (subprocess, file I/O)
- Netzwerk-Zugriffe (OUI-Download)
- Hardware-Zugriffe (WLAN-Interfaces)

## 🚨 Bekannte Einschränkungen

1. **Hardware-Tests**: Benötigen echte WLAN-Hardware
2. **Netzwerk-Tests**: Benötigen Internet-Verbindung
3. **Performance-Tests**: Abhängig von System-Ressourcen
4. **Integrationstests**: Können bei fehlenden Abhängigkeiten fehlschlagen

## 🤝 Beitragen

### Neue Tests hinzufügen

1. Wähle die passende Test-Datei oder erstelle eine neue
2. Verwende bestehende Fixtures aus `conftest.py`
3. Füge passende Marker hinzu (`@pytest.mark.unit`, etc.)
4. Dokumentiere den Test mit Docstrings

### Test-Standards

- **Naming**: `test_<function_name>_<scenario>`
- **Structure**: Arrange, Act, Assert
- **Coverage**: Mindestens 80% Code-Coverage
- **Performance**: Unit-Tests < 1s, Integration-Tests < 10s

### Test-Review-Checkliste

- [ ] Test ist deterministisch (keine Race Conditions)
- [ ] Test ist isoliert (keine Abhängigkeiten zu anderen Tests)
- [ ] Test ist schnell (< 1s für Unit-Tests)
- [ ] Test hat aussagekräftige Assertions
- [ ] Test ist gut dokumentiert
- [ ] Test verwendet passende Fixtures