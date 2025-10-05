#!/bin/bash

# =================================================================
# Setup-Skript für den intelligenten, automatischen Start
# des WLAN-Scanners
# =================================================================

# --- Konfiguration ---

# WLAN-Namen (SSIDs), bei denen der Scan NICHT ausgeführt werden soll.
# Trennen Sie die Namen mit einem Leerzeichen. Groß-/Kleinschreibung ist egal.
HOME_WLANS=("Wiesenstraße.24" "Wiesenstraße 2.4" "fridge")

# Der Befehl, der ausgeführt werden soll, wenn das WLAN NICHT übereinstimmt.
# Passen Sie hier die Dauer und andere Parameter nach Bedarf an.
SCAN_COMMAND="wifi --project auto_scan --duration 3600 --status-led" # 3600s = 1 Stunde

# --- Systempfade (sollten normalerweise nicht geändert werden müssen) ---
PROJECT_DIR="$(pwd)"
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "Fehler: Bitte führen Sie dieses Skript aus dem Hauptverzeichnis ('~/hacking') aus."
    exit 1
fi
VENV_PATH="$PROJECT_DIR/.venv/bin/python"
SERVICE_NAME="wifi_autoscanner.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
WRAPPER_NAME="run_if_not_in_home_wlan.sh"
WRAPPER_PATH="$PROJECT_DIR/$WRAPPER_NAME"

echo "--- Setup für den automatischen WLAN-Scanner wird gestartet ---"

# --- Schritt 1: Erstelle das Wrapper-Skript mit der Logik ---
echo "Erstelle Logik-Wrapper-Skript unter $WRAPPER_PATH..."

# Konvertiere das Bash-Array in eine Zeichenkette für den Vergleich im Skript
HOME_WLANS_STRING=" ${HOME_WLANS[*]} "

cat > "$WRAPPER_PATH" <<EOF
#!/bin/bash
# Dieses Skript wird von systemd aufgerufen.

# Warte 20 Sekunden, um sicherzustellen, dass das Netzwerk vollständig initialisiert ist
sleep 20

# Liste der Heimnetzwerke
declare -a HOME_WLANS=(${HOME_WLANS[@]})

# Finde das aktuell verbundene WLAN
CURRENT_WLAN=\$(iwgetid -r 2>/dev/null)

echo "Aktuell verbundenes WLAN: '\$CURRENT_WLAN'"

# Standardmäßig gehen wir davon aus, dass wir scannen sollen
SHOULD_SCAN=true

# Überprüfe, ob das aktuelle WLAN in der Liste der Heimnetzwerke ist
if [ -n "\$CURRENT_WLAN" ]; then
    for home_wlan in "\${HOME_WLANS[@]}"; do
        if [[ "\$CURRENT_WLAN" == "\$home_wlan" ]]; then
            echo "Verbunden mit Heimnetzwerk '\$home_wlan'. Scan wird nicht ausgeführt."
            SHOULD_SCAN=false
            break
        fi
    done
fi

# Führe den Scan aus, wenn die Bedingung erfüllt ist
if [ "\$SHOULD_SCAN" = true ]; then
    if [ -z "\$CURRENT_WLAN" ]; then
        echo "Mit keinem WLAN verbunden. Starte den Scan..."
    else
        echo "Nicht mit einem Heimnetzwerk verbunden. Starte den Scan..."
    fi
    
    # Wechsle in das Projektverzeichnis und führe den Befehl mit der venv aus
    cd "$PROJECT_DIR"
    "$VENV_PATH" "$PROJECT_DIR/main.py" $SCAN_COMMAND
    
    echo "Scan beendet."
fi
EOF

# Mache das Wrapper-Skript ausführbar
chmod +x "$WRAPPER_PATH"
echo "Wrapper-Skript erfolgreich erstellt und ausführbar gemacht."

# --- Schritt 2: Erstelle die systemd-Service-Datei ---
echo "Erstelle die systemd-Service-Datei unter $SERVICE_PATH..."
echo "Root-Rechte (sudo) werden für diesen Schritt benötigt."

sudo bash -c "cat > '$SERVICE_PATH'" <<EOF
[Unit]
Description=Automatischer WLAN-Scan, wenn nicht im Heimnetzwerk
# Stellt sicher, dass der Service erst startet, wenn das Netzwerk online ist
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$(whoami)
# Führe unser Wrapper-Skript aus
ExecStart=$WRAPPER_PATH

[Install]
WantedBy=multi-user.target
EOF

echo "systemd-Service-Datei erfolgreich erstellt."

# --- Schritt 3: systemd-Daemon neu laden und Service aktivieren ---
echo "Lade den systemd-Daemon neu und aktiviere den Service..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "--- Setup abgeschlossen! ---"
echo "Der Service '$SERVICE_NAME' wurde erstellt und für den nächsten Systemstart aktiviert."
echo "Sie können den Status mit 'systemctl status $SERVICE_NAME' überprüfen."
echo "Um den Service zu testen (ohne Neustart), führen Sie aus: 'sudo systemctl start $SERVICE_NAME'"