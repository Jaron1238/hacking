-- Version 1: Initiales Schema
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, type TEXT NOT NULL,
    bssid TEXT, ssid TEXT, client TEXT, channel INTEGER, ies TEXT, seq INTEGER,
    rssi REAL, beacon_interval INTEGER, cap INTEGER, beacon_timestamp INTEGER,
    is_powersave BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_bssid ON events(bssid);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ssid TEXT, bssid TEXT, label INTEGER,
    source TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_labels_ssid ON labels(ssid);

CREATE TABLE IF NOT EXISTS client_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, mac TEXT UNIQUE NOT NULL,
    device_type TEXT NOT NULL, source TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_client_labels_mac ON client_labels(mac);

PRAGMA user_version = 1;