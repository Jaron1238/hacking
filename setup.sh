#!/bin/bash
# Setup-Skript für WLAN Analysis Tool (macOS ohne Admin-Rechte)
# Installiert Python mit pyenv und richtet eine virtuelle Umgebung ein.

set -e # Beendet das Skript sofort, wenn ein Befehl fehlschlägt

# --- Konfiguration ---
PYTHON_VERSION_WANTED="3.11.9"

# --- Farben für die Ausgabe ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=================================================="
echo "Setup für WLAN Analysis Tool (macOS)"
echo "Ziel-Python-Version: $PYTHON_VERSION_WANTED"
echo "=================================================="
echo ""

# --- Schritt 1: Prüfe System-Abhängigkeiten ---
echo -e "${YELLOW}Schritt 1: Prüfe verfügbare System-Tools...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git ist nicht installiert. Bitte installiere Xcode Command Line Tools:${NC}"
    echo "xcode-select --install"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}curl ist nicht verfügbar.${NC}"
    exit 1
fi

echo "System-Tools sind verfügbar."
echo ""

# --- Schritt 2: pyenv installieren und einrichten ---
echo -e "${YELLOW}Schritt 2: pyenv wird installiert...${NC}"
if [ -d "$HOME/.pyenv" ]; then
    echo "pyenv ist bereits installiert."
else
    # pyenv direkt installieren (ohne Homebrew)
    echo "Installiere pyenv direkt..."
    curl https://pyenv.run | bash
fi

# pyenv zur Shell hinzufügen
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Konfiguration für zsh (Standard-Shell auf macOS)
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="~/.zshrc"
    grep -qF 'PYENV_ROOT' ~/.zshrc || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    grep -qF 'pyenv init' ~/.zshrc || {
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    }
else
    # Fallback für bash
    grep -qF 'PYENV_ROOT' ~/.bashrc || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    grep -qF 'pyenv init' ~/.bashrc || {
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    }
fi
echo ""

# --- Schritt 3: Gewünschte Python-Version installieren ---
if [ "$PYTHON_VERSION_WANTED" == "LATEST" ]; then
    echo -e "${YELLOW}Suche nach der neuesten stabilen Python-Version..."
    # Findet die letzte stabile Version (keine -rc oder -dev)
    PYTHON_VERSION_TO_INSTALL=$(pyenv install --list | grep -E "^\s*[0-9]+\.[0-9]+\.[0-9]+$" | tail -1 | tr -d ' ')
    echo "Neueste gefundene Version: $PYTHON_VERSION_TO_INSTALL"
else
    PYTHON_VERSION_TO_INSTALL=$PYTHON_VERSION_WANTED
fi

echo -e "${YELLOW}Schritt 3: Installiere Python $PYTHON_VERSION_TO_INSTALL mit pyenv..."
if pyenv versions --bare | grep -q "^$PYTHON_VERSION_TO_INSTALL$"; then
    echo "Python $PYTHON_VERSION_TO_INSTALL ist bereits installiert."
else
    echo "Die Installation wird eine Weile dauern. Bitte habe Geduld."
    # Kompiliert und installiert die gewünschte Python-Version
    pyenv install "$PYTHON_VERSION_TO_INSTALL"
fi
echo ""

# --- Schritt 4: Projekt für die neue Python-Version einrichten ---
VENV_DIR="$PWD/.venv"
echo -e "${YELLOW}Schritt 4: Richte das Projekt für Python $PYTHON_VERSION_TO_INSTALL ein..."

# Legt die Python-Version für das aktuelle Verzeichnis fest
pyenv local "$PYTHON_VERSION_TO_INSTALL"
echo "Python-Version für dieses Verzeichnis auf $PYTHON_VERSION_TO_INSTALL gesetzt."

# Erstellt die virtuelle Umgebung mit der neuen Python-Version
echo "Erstelle Virtual Environment unter '$VENV_DIR'..."
if [ -d "$VENV_DIR" ]; then
    # Sicherstellen, dass die venv auch mit der richtigen Python-Version erstellt wurde
    VENV_PYTHON_VERSION=$(source $VENV_DIR/bin/activate && python --version && deactivate | awk '{print $2}')
    if [[ "$VENV_PYTHON_VERSION" != *"$PYTHON_VERSION_TO_INSTALL"* ]]; then
        echo -e "${YELLOW}Die existierende venv nutzt eine falsche Python-Version. Sie wird neu erstellt.${NC}"
        rm -rf "$VENV_DIR"
        python -m venv "$VENV_DIR"
    else
        echo "Virtual Environment existiert bereits und nutzt die korrekte Python-Version."
    fi
else
    python -m venv "$VENV_DIR"
fi
echo ""

# --- Schritt 5: Abhängigkeiten in der venv installieren ---
echo -e "${YELLOW}Schritt 5: Installiere Python-Pakete in der Virtual Environment...${NC}"
source "$VENV_DIR/bin/activate"

echo "Aktualisiere pip..."
pip install --upgrade pip
echo "Installiere Pakete aus requirements.txt..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Alle Python-Pakete wurden erfolgreich in '.venv' installiert.${NC}"
echo ""

# --- Abschluss ---
echo -e "${GREEN}=================================================="
echo "Setup erfolgreich abgeschlossen!"
echo "Python-Version: $(python --version)"
echo "==================================================${NC}"
echo ""
echo -e "${YELLOW}WICHTIG für macOS ohne Admin-Rechte:${NC}"
echo "- Das Tool kann NICHT im Monitor-Mode arbeiten"
echo "- Packet Capture ist ohne Root-Rechte nicht möglich"
echo "- Nur Analyse-Features sind verfügbar"
echo ""
echo "Nächste Schritte:"
echo -e "1. Aktiviere die Umgebung: ${YELLOW}source .venv/bin/activate${NC}"
echo -e "2. Für Analyse vorhandener Daten: ${YELLOW}python main.py --project my_project --infer --tui${NC}"
echo -e "3. Deaktiviere die Umgebung: ${YELLOW}deactivate${NC}"
echo ""
