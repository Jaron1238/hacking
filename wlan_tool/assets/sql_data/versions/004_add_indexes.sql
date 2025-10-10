-- version 4
-- Fügt Indizes hinzu, um die Abfrageleistung für die Analyse erheblich zu verbessern.

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_bssid ON events(bssid);
CREATE INDEX IF NOT EXISTS idx_events_client ON events(client);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

CREATE INDEX IF NOT EXISTS idx_dns_client ON dns_queries(client);

CREATE INDEX IF NOT EXISTS idx_labels_bssid ON labels(bssid);

PRAGMA user_version = 4;