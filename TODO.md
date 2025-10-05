# TODO - Projektverbesserungen 📝

## ✅ Erledigt (automatisch)
- [x] `.gitignore` erstellt
- [x] Doppelte Imports in `main.py` entfernt
- [x] Logger-Handling verbessert
- [x] README.md mit Dokumentation erstellt
- [x] Cleanup-Script erstellt
- [x] Requirements mit Versionsangaben vorbereitet

## 🔴 Kritisch (Sofort)

### 1. Manuelle Bereinigung
```bash
chmod +x cleanup.sh
./cleanup.sh
```
Oder manuell:
- [ ] `.swp` Dateien löschen
- [ ] `.DS_Store` Dateien löschen
- [ ] `config__.yaml` löschen (Duplikat)
- [ ] `testfile.txt` löschen
- [ ] `test_app.py` nach `test/` verschieben

### 2. Dependencies stabilisieren
```bash
# Im aktivierten venv:
pip freeze > requirements.txt
```
- [ ] Requirements mit exakten Versionen erstellen
- [ ] Testen, ob alles noch funktioniert

### 3. "hi" Verzeichnis klären
- [ ] Was ist `/data/hi/`?
- [ ] Umbenennen oder löschen

## 🟡 Wichtig (Diese Woche)

### Code-Qualität
- [ ] Type Hints zu kritischen Funktionen hinzufügen
- [ ] Docstrings vervollständigen
- [ ] Lange Funktionen in `main.py` aufteilen (z.B. `parse_args()`)

### Tests
- [ ] `pytest test/` ausführen und Fehler fixen
- [ ] Test-Coverage prüfen: `pytest --cov=data test/`
- [ ] Fehlende Tests für neue Features schreiben

### Dokumentation
- [ ] `CHANGELOG.md` erstellen
- [ ] API-Dokumentation mit Sphinx generieren
- [ ] Beispiel-Konfigurationen hinzufügen

## 🟢 Nice-to-have (Langfristig)

### Architektur
- [ ] Controller-Klassen weiter entkoppeln
- [ ] Dependency Injection einführen
- [ ] Event-System für Plugin-Kommunikation

### CLI-Verbesserung
```python
# Statt argparse:
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--iface', help='WiFi interface')
def capture(iface):
    """Start packet capture"""
    pass
```

### Performance
- [ ] Async I/O für Capture (mit `asyncio`)
- [ ] Parallele Analyse mit `multiprocessing`
- [ ] Caching mit `functools.lru_cache`

### DevOps
- [ ] `Dockerfile` erstellen
- [ ] `docker-compose.yml` für Testing-Umgebung
- [ ] GitHub Actions CI/CD
- [ ] Pre-commit hooks (`pre-commit install`)

### Monitoring
- [ ] Prometheus-Metriken exportieren
- [ ] Health-Check-Endpoint
- [ ] Performance-Profiling (`cProfile`)

## 📊 Metriken

### Aktueller Stand
- Zeilen Code: ~5000+ (geschätzt)
- Test-Coverage: ? (messen!)
- Kritische TODOs: 3
- Dependencies: 18

### Ziele
- [ ] Test-Coverage > 80%
- [ ] Code-Style: 100% Black/Ruff-konform
- [ ] Alle Warnings behoben
- [ ] Dokumentation vollständig

## 🐛 Bekannte Probleme

1. **Logger-Konflikt**: main.py überschreibt Root-Logger
   - ✅ Behoben durch namespaced Logger

2. **Config-Duplikate**: config.yaml existiert 2x
   - ⏳ Manuell zu beheben

3. **Fehlende Versionen**: requirements.txt ohne Pins
   - ⏳ `pip freeze` ausführen

## 📝 Notizen

### Projekt-Struktur-Vorschlag
```
hacking/
├── src/                    # Neuer Source-Ordner
│   ├── wifi_analysis/     # Package-Name
│   │   ├── __init__.py
│   │   ├── cli/
│   │   ├── capture/
│   │   ├── analysis/
│   │   ├── ml/
│   │   └── plugins/
│   └── setup.py           # Für pip install -e .
├── tests/                 # Statt "test"
├── docs/                  # Sphinx-Dokumentation
├── configs/               # Beispiel-Configs
├── scripts/               # Hilfsskripte
└── data/                  # Runtime-Daten (gitignored)
```

### Empfohlene Tools
```bash
# Code-Qualität
pip install black ruff mypy

# Testing
pip install pytest pytest-cov pytest-mock

# Pre-commit
pip install pre-commit
pre-commit install

# Dokumentation
pip install sphinx sphinx-rtd-theme
```

---

**Zuletzt aktualisiert:** $(date)
**Nächster Review:** In 2 Wochen
