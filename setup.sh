#!/bin/bash
# WLAN Analysis Tool - Setup mit Python-Autoupdate & Backup
# Funktioniert auf Raspberry Pi OS / Debian-basierten Systemen
# Schützt den aktiven Kernel und verhindert initramfs-Fehler

set -e

echo "=================================================="
echo "WLAN Analysis Tool - Setup (mit Python-Update)"
echo "=================================================="
echo ""

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# === Aktuellen Kernel merken ===
CURRENT_KERNEL=$(uname -r)
echo -e "${YELLOW}Aktueller Kernel: $CURRENT_KERNEL${NC}"

# === Alte Kernel / problematische Kernel auf Hold setzen ===
echo -e "${YELLOW}Schütze den aktuellen Kernel und problematische neue Kernel...${NC}"
sudo apt-mark hold linux-image-6.12.47+rpt-rpi-v8 \
                    linux-headers-6.12.47+rpt-rpi-v8 \
                    linux-image-6.12.47+rpt-rpi-2712 \
                    linux-headers-6.12.47+rpt-rpi-2712

# === System-Update ===
echo -e "${YELLOW}System wird aktualisiert...${NC}"
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y build-essential wget curl git python3 python3-venv python3-pip \
libssl-dev zlib1g-dev libncurses5-dev libffi-dev libsqlite3-dev libreadline-dev \
libbz2-dev liblzma-dev

# === Python-Backup ===
if command -v python3 &> /dev/null; then
    CURRENT_VERSION=$(python3 --version | awk '{print $2}')
    BACKUP_DIR="/usr/local/python_backup_${CURRENT_VERSION}"
    echo ""
    echo -e "${YELLOW}Erstelle Backup der aktuellen Python-Version ($CURRENT_VERSION)...${NC}"
    sudo mkdir -p "$BACKUP_DIR"
    sudo cp -r /usr/bin/python3* "$BACKUP_DIR" 2>/dev/null || true
    sudo cp -r /usr/local/bin/python3* "$BACKUP_DIR" 2>/dev/null || true
    sudo cp -r /usr/lib/python3* "$BACKUP_DIR" 2>/dev/null || true
    echo -e "${GREEN}✓ Backup erstellt unter: $BACKUP_DIR${NC}"
else
    echo -e "${YELLOW}Python3 war nicht installiert – kein Backup nötig${NC}"
fi

# === Neueste Python-Version abrufen ===
echo ""
echo "Suche nach der neuesten Python-Version..."
LATEST=$(wget -qO- https://www.python.org/ftp/python/ | grep -oP '(?<=href=")[0-9]+\.[0-9]+\.[0-9]+(?=/")' | sort -V | tail -1)
echo -e "${GREEN}Neueste Version: Python $LATEST${NC}"

# === Prüfen ob Update nötig ===
if python3 --version 2>/dev/null | grep -q "$LATEST"; then
    echo -e "${GREEN}Python ist bereits auf dem neuesten Stand ($LATEST).${NC}"
else
    echo -e "${YELLOW}Installiere Python $LATEST aus Source...${NC}"
    cd /tmp
    wget https://www.python.org/ftp/python/$LATEST/Python-$LATEST.tgz || { echo "Download fehlgeschlagen"; exit 1; }
    tar -xzf Python-$LATEST.tgz
    cd Python-$LATEST
    ./configure --enable-optimizations
    make -j$(nproc)
    sudo make altinstall

    NEW_BIN=$(ls /usr/local/bin/python3.* | sort -V | tail -1)
    sudo ln -sf "$NEW_BIN" /usr/local/bin/python3
    hash -r  # Bash merkt sich neue Version sofort

    echo -e "${GREEN}✓ Python $LATEST erfolgreich installiert.${NC}"
fi

# === Virtual Environment ===
echo ""
echo "Erstelle Virtual Environment..."
VENV_DIR="$PWD/.venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual Environment existiert bereits${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual Environment erstellt${NC}"
fi

# === Aktivieren & Dependencies ===
echo "Aktiviere Virtual Environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "Aktualisiere pip..."
pip install --upgrade pip --quiet

echo ""
echo "Installiere Dependencies..."
pip install --upgrade -r requirements.txt

# === Initramfs für aktiven Kernel neu erstellen ===
echo ""
echo -e "${YELLOW}Erstelle initramfs für den aktiven Kernel $CURRENT_KERNEL...${NC}"
sudo update-initramfs -c -k "$CURRENT_KERNEL"

# === Abschluss ===
echo ""
echo -e "${GREEN}=================================================="
echo "Setup erfolgreich abgeschlossen!"
echo "==================================================${NC}"
echo ""
echo "Backup gespeichert unter: $BACKUP_DIR"
echo ""
echo "Nächste Schritte:"
echo "1. source .venv/bin/activate"
echo "2. sudo .venv/bin/python main.py --capture_mode --project test_scan"
echo "3. python main.py --project test_scan --infer --tui"
echo ""
