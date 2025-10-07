#!/bin/bash
set -euo pipefail

REPO_DIR="/home/pi/hacking"
LOGFILE="/home/pi/auto_git_sync.log"
cd "$REPO_DIR" || { echo "Repo nicht gefunden: $REPO_DIR"; exit 1; }

# Aktuellen Branch ermitteln
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Fetch + rebase um sauber zu bleiben
echo "[$(date --iso-8601=seconds)] fetching..." >> "$LOGFILE"
git fetch origin >> "$LOGFILE" 2>&1 || { echo "Fetch failed" >> "$LOGFILE"; exit 1; }

REMOTE_HEAD=$(git rev-parse "origin/$BRANCH")
LOCAL_HEAD=$(git rev-parse "$BRANCH")

if [ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]; then
    echo "[$(date --iso-8601=seconds)] Remote-Änderungen gefunden, pull --rebase..." >> "$LOGFILE"
    git pull --rebase origin "$BRANCH" >> "$LOGFILE" 2>&1 || {
        echo "[$(date --iso-8601=seconds)] Pull fehlgeschlagen. Abbruch." >> "$LOGFILE"
        exit 1
    }
else
    echo "[$(date --iso-8601=seconds)] Kein Remote-Update." >> "$LOGFILE"
fi

# Änderungen prüfen, aber temporäre/ungewollte patterns herausfiltern
# Passe die grep-RegEx an, wenn du andere Dateitypen ausschließen willst
CHANGES=$(git status --porcelain | grep -vE '\.log$|\.tmp$|__pycache__|\.pyc$|\.env$|\.db-wal$|\.db-shm$|^db-wal$|^db-shm$|^(\?|\s)*$' || true)

if [ -n "$CHANGES" ]; then
    echo "[$(date --iso-8601=seconds)] Lokale relevante Änderungen gefunden:" >> "$LOGFILE"
    echo "$CHANGES" >> "$LOGFILE"
    git add -A
    # Vermeide Commiten, wenn nur ignored Dateien gestaged wurden
    if [ -n "$(git diff --cached --name-only | grep -vE '\.log$|\.tmp$|__pycache__|\.pyc$|\.env$|\.db-wal$|\.db-shm$|^db-wal$|^db-shm$' || true)" ]; then
        git commit -m "Auto-sync: $(date --iso-8601=seconds)" >> "$LOGFILE" 2>&1 || true
        echo "[$(date --iso-8601=seconds)] Pushing to origin/$BRANCH..." >> "$LOGFILE"
        git push origin "$BRANCH" >> "$LOGFILE" 2>&1 || {
            echo "[$(date --iso-8601=seconds)] Push fehlgeschlagen." >> "$LOGFILE"
            exit 1
        }
        echo "[$(date --iso-8601=seconds)] Push erfolgreich." >> "$LOGFILE"
    else
        echo "[$(date --iso-8601=seconds)] Nur ignorierte Dateien im Stage -> Commit übersprungen." >> "$LOGFILE"
        git reset --mixed >> "$LOGFILE" 2>&1 || true
    fi
else
    echo "[$(date --iso-8601=seconds)] Keine lokalen Änderungen." >> "$LOGFILE"
fi
