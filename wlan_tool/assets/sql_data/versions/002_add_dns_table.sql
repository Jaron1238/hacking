-- Version 2: Fügt die DNS-Tabelle hinzu
CREATE TABLE IF NOT EXISTS dns_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    client TEXT NOT NULL,
    domain TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dns_client ON dns_queries(client);

PRAGMA user_version = 2;