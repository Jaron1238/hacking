#!/bin/bash
# Führe dieses Script auf dem Raspberry Pi aus, um ml_training.py zu patchen

FILE="/home/pi/hacking/data/ml_training.py"

echo "Patche $FILE..."

# Backup erstellen
cp "$FILE" "${FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Patch 1: Füge Mindestanzahl-Check hinzu
sed -i 's/if not X_regulars.empty:/if not X_regulars.empty and len(X_regulars) >= 4:  # Mindestens 4 Samples für 25% Split/' "$FILE"

# Patch 2: Ändere else-Block
sed -i 's/# Seltener Fall: Alle Klassen haben nur ein Mitglied/# Zu wenig Daten für einen sinnvollen Split - nutze alle Daten fürs Training/' "$FILE"
sed -i 's/logger.warning("Alle Klassen haben nur ein Mitglied. Es wird kein Test-Set erstellt.")/logger.warning("Zu wenig Daten für Test\/Train-Split (benötigt mindestens 4 Samples). Alle Daten werden fürs Training verwendet.")/' "$FILE"
sed -i 's/X_train = X_singles/X_train = pd.concat([X_regulars, X_singles]) if not X_regulars.empty else X_singles/' "$FILE"
sed -i 's/y_train = y_singles/y_train = pd.concat([y_regulars, y_singles]) if not y_regulars.empty else y_singles/' "$FILE"
sed -i 's/y_test = pd.Series()/y_test = pd.Series(dtype='"'"'object'"'"')/' "$FILE"

echo "✓ Patch erfolgreich angewendet!"
echo ""
echo "Ein Backup wurde erstellt: ${FILE}.backup_*"
echo ""
echo "Jetzt kannst du erneut ausführen:"
echo "  wifi --project test --train-behavior-model device_classifier.joblib"
