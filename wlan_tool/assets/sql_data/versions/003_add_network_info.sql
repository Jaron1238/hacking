-- Version 3: Erweitert die client_labels-Tabelle um Netzwerk-Informationen
ALTER TABLE client_labels ADD COLUMN ip_address TEXT;
ALTER TABLE client_labels ADD COLUMN hostname TEXT;

PRAGMA user_version = 3;