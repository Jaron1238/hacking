#!/bin/bash
# Sicheres Setup-Skript für WLAN Analysis Tool
# Installiert eine spezifische Python-Version sicher mit pyenv
# und richtet eine virtuelle Umgebung ein.

set -e # Beendet das Skript sofort, wenn ein Befehl fehlschlägt

# --- Konfiguration ---
# Hier kannst du die gewünschte Python-Version eintragen.
# '3.14.0' für eine spezifische Version.
# Oder setze es auf 'LATEST' für die automatisch neueste stabile Version.
PYTHON_VERSION_WANTED="3.11.9"

# --- Farben für die Ausgabe ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=================================================="
echo "Sicheres Setup für WLAN Analysis Tool"
echo "Ziel-Python-Version: $PYTHON_VERSION_WANTED"
echo "=================================================="
echo ""

# --- Schritt 1: System-Update & Abhängigkeiten für pyenv ---
echo -e "${YELLOW}Schritt 1: System wird aktualisiert und Build-Abhängigkeiten werden installiert...${NC}"
sudo apt update
sudo apt upgrade -y
# Notwendige Pakete zum Kompilieren von Python
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
python3-venv python3-pip

# Alte Kernel und ungenutzte Pakete sicher entfernen
echo -e "${YELLOW}Entferne alte Kernel und räume das System auf...${NC}"
sudo apt autoremove -y
echo ""

# --- Schritt 2: pyenv installieren und einrichten ---
echo -e "${YELLOW}Schritt 2: pyenv wird installiert...${NC}"
if [ -d "$HOME/.pyenv" ]; then
    echo "pyenv ist bereits installiert."
else
    # Offizieller pyenv-Installer
    curl https://pyenv.run | bash
fi

# pyenv zur Shell hinzufügen, damit es sofort verfügbar ist
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Sicherstellen, dass die Konfiguration auch für zukünftige Logins gilt
grep -qF 'PYENV_ROOT' ~/.bashrc || echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
grep -qF 'pyenv init' ~/.bashrc || {
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
}
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
echo "Setup erfolgreich und sicher abgeschlossen!"
echo "Python-Version: $(python --version)"
echo "==================================================${NC}"
echo ""
echo "Das System wurde NICHT durch unsichere Änderungen gefährdet."
echo ""
echo "Nächste Schritte, um das Tool zu verwenden:"
echo -e "1. Aktiviere die Umgebung: ${YELLOW}source .venv/bin/activate${NC}"
echo -e "2. Führe dein Programm aus (Beispiel): ${YELLOW}python main.py --deine-optionen${NC}"
echo -e "3. Wenn du fertig bist, deaktiviere die Umgebung: ${YELLOW}deactivate${NC}"
echo ""
