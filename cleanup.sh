#!/bin/bash
# cleanup.sh - Bereinigt temporäre Dateien im Projekt
# ACHTUNG: Prüfe vor dem Ausführen, ob du ein Backup hast!

set -e  # Bei Fehler abbrechen

echo "🧹 Starte Projekt-Bereinigung..."

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Zähler
DELETED=0

# Funktion zum sicheren Löschen
safe_delete() {
    if [ -f "$1" ]; then
        echo -e "${YELLOW}Lösche: $1${NC}"
        rm "$1"
        ((DELETED++))
    fi
}

safe_delete_pattern() {
    echo -e "${YELLOW}Suche nach: $1${NC}"
    find . -name "$1" -type f -print -delete
    COUNT=$(find . -name "$1" -type f 2>/dev/null | wc -l)
    DELETED=$((DELETED + COUNT))
}

# 1. Vim Swap-Dateien
echo -e "\n${GREEN}1. Entferne Vim-Swap-Dateien (.swp)${NC}"
safe_delete_pattern "*.swp"

# 2. macOS Metadaten
echo -e "\n${GREEN}2. Entferne macOS .DS_Store Dateien${NC}"
safe_delete_pattern ".DS_Store"

# 3. Python Cache
echo -e "\n${GREEN}3. Entferne Python-Cache${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# 4. Test-Dateien
echo -e "\n${GREEN}4. Entferne Test-Artefakte${NC}"
safe_delete "./testfile.txt"
safe_delete "./test_capture.pcap"
safe_delete "./test.log"

# 5. Duplikat-Config
echo -e "\n${GREEN}5. Entferne Duplikat-Konfigurationen${NC}"
if [ -f "./config__.yaml" ]; then
    echo -e "${YELLOW}⚠️  config__.yaml gefunden (Duplikat?)${NC}"
    echo "Behalte config.yaml im Root? (j/n)"
    read -r response
    if [[ "$response" =~ ^[Jj]$ ]]; then
        safe_delete "./config__.yaml"
    fi
fi

# 6. Logs (optional - Nutzer entscheidet)
echo -e "\n${GREEN}6. Log-Dateien bereinigen?${NC}"
echo "Möchtest du alle .log Dateien löschen? (j/n)"
read -r response
if [[ "$response" =~ ^[Jj]$ ]]; then
    safe_delete_pattern "*.log"
fi

# 7. Verschiebe test_app.py
if [ -f "./test_app.py" ]; then
    echo -e "\n${GREEN}7. Verschiebe test_app.py ins test/ Verzeichnis${NC}"
    mkdir -p test
    mv ./test_app.py ./test/ 2>/dev/null && echo -e "${GREEN}✓ Verschoben${NC}"
fi

# Zusammenfassung
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ Bereinigung abgeschlossen!${NC}"
echo -e "${GREEN}📊 $DELETED Dateien gelöscht${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Empfehlungen
echo -e "\n${YELLOW}💡 Nächste Schritte:${NC}"
echo "1. git status # Prüfe Änderungen"
echo "2. pip freeze > requirements.txt # Versionen festschreiben"
echo "3. pytest test/ # Tests ausführen"
echo "4. git add . && git commit -m 'Cleanup durchgeführt'"
