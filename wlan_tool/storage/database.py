#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funktionen und Klassen für die SQLite-Datenbankinteraktion.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# Relative Imports
from .. import config
from ..constants import (
    Constants,
    ErrorCodes,
    ErrorMessages,
    get_constant,
    get_error_message,
)
from ..exceptions import (
    DatabaseError,
    ErrorContext,
    ErrorRecovery,
    FileSystemError,
    ValidationError,
    handle_errors,
    retry_on_error,
)

logger = logging.getLogger(__name__)

try:
    import orjson as _json

    def dumps(o):
        return _json.dumps(o).decode()

    def loads(s):
        return _json.loads(s)

except ImportError:
    dumps = lambda o: json.dumps(o, separators=(",", ":"))
    loads = lambda s: json.loads(s)


def _get_db_version(conn: sqlite3.Connection) -> int:
    try:
        cursor = conn.execute("PRAGMA user_version;")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0


@contextmanager
def db_conn_ctx(path: str):
    """Context Manager für Datenbankverbindungen mit Fehlerbehandlung."""
    conn = None
    try:
        with ErrorContext("database_connection", "DB_CONNECTION_ERROR"):
            # Validiere Datenbankpfad
            if not _validate_file_path(path):
                raise ValidationError(
                    f"Invalid database path: {path}",
                    error_code="INVALID_DB_PATH",
                    details={"db_path": path},
                )

            # Erstelle Verzeichnis falls nötig
            db_dir = Path(path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise FileSystemError(
                        f"Cannot create database directory: {db_dir}",
                        error_code="DB_DIR_CREATION_FAILED",
                        details={"directory": str(db_dir), "original_error": str(e)},
                    ) from e

            # Verbinde zur Datenbank
            conn = sqlite3.connect(path, timeout=Constants.DB_CONNECTION_TIMEOUT)
            conn.row_factory = sqlite3.Row

            # Setze pragmas für bessere Performance und Fehlerbehandlung
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA temp_store = MEMORY")

            yield conn

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(
            get_error_message(ErrorCodes.DB_CONNECTION_FAILED, details=str(e)),
            error_code=ErrorCodes.DB_CONNECTION_FAILED.value,
            details={"db_path": path, "sqlite_error": str(e)},
        ) from e
    except Exception as e:
        if conn:
            conn.rollback()
        raise DatabaseError(
            f"Unexpected database error: {e}",
            error_code="DB_UNEXPECTED_ERROR",
            details={"db_path": path, "original_error": str(e)},
        ) from e
    finally:
        if conn:
            try:
                conn.close()
            except sqlite3.Error as e:
                logger.warning(f"Error closing database connection: {e}")


def _validate_file_path(path: str) -> bool:
    """Validiere Dateipfad."""
    try:
        Path(path).resolve()
        return True
    except (OSError, ValueError):
        return False


@handle_errors(DatabaseError, "DB_MIGRATION_ERROR", reraise=True)
def migrate_db(db_path: str):
    """Migriere Datenbank mit Fehlerbehandlung."""
    with ErrorContext("database_migration", "DB_MIGRATION_ERROR"):
        migrations_path = (
            Path(__file__).parent.parent / "assets" / "sql_data" / "versions"
        )

        if not migrations_path.exists():
            raise FileSystemError(
                f"Migrations directory not found: {migrations_path}",
                error_code="MIGRATIONS_DIR_NOT_FOUND",
                details={"migrations_path": str(migrations_path)},
            )

        migration_files = sorted(migrations_path.glob("*.sql"))
        if not migration_files:
            logger.warning("No migration files found")
            return

        with db_conn_ctx(db_path) as conn:
            current_version = _get_db_version(conn)
            logger.info(
                f"Current DB version: {current_version}. Looking for migrations..."
            )

            for migration_file in migration_files:
                try:
                    file_version = int(migration_file.name.split("_")[0])
                    if file_version > current_version:
                        logger.info(
                            f"Applying migration: {migration_file.name} (Version {file_version})"
                        )

                        # Sichere Migration mit Rollback
                        try:
                            sql_script = migration_file.read_text(encoding="utf-8")
                            conn.executescript(sql_script)
                            conn.commit()

                            new_version = _get_db_version(conn)
                            logger.info(
                                f"DB successfully migrated to version {new_version}"
                            )
                            current_version = new_version

                        except sqlite3.Error as e:
                            conn.rollback()
                            raise DatabaseError(
                                f"Migration failed: {migration_file.name}",
                                error_code="MIGRATION_EXECUTION_FAILED",
                                details={
                                    "migration_file": migration_file.name,
                                    "version": file_version,
                                    "sqlite_error": str(e),
                                },
                            ) from e

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Migration file has invalid name and will be ignored: {migration_file.name}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Unexpected error during migration {migration_file.name}: {e}"
                    )
                    continue

        logger.info("Database is up to date.")


@handle_errors(DatabaseError, "DB_FETCH_EVENTS_ERROR", default_return=[])
def fetch_events(
    conn: sqlite3.Connection,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> List[Dict]:
    """Hole Events aus der Datenbank mit Fehlerbehandlung."""
    with ErrorContext("fetch_events", "DB_FETCH_EVENTS_ERROR"):
        # Validiere Parameter
        if start_ts is not None and not isinstance(start_ts, (int, float)):
            raise ValidationError(
                f"Invalid start_ts type: {type(start_ts)}",
                error_code="INVALID_START_TS",
                details={"start_ts": start_ts, "type": str(type(start_ts))},
            )

        if end_ts is not None and not isinstance(end_ts, (int, float)):
            raise ValidationError(
                f"Invalid end_ts type: {type(end_ts)}",
                error_code="INVALID_END_TS",
                details={"end_ts": end_ts, "type": str(type(end_ts))},
            )

        if start_ts is not None and end_ts is not None and start_ts >= end_ts:
            raise ValidationError(
                f"start_ts ({start_ts}) must be less than end_ts ({end_ts})",
                error_code="INVALID_TIME_RANGE",
                details={"start_ts": start_ts, "end_ts": end_ts},
            )

        q = "SELECT ts, type, bssid, ssid, client, channel, ies, seq, rssi, beacon_interval, cap, beacon_timestamp, is_powersave FROM events"
        conds, params = [], []

        if start_ts is not None:
            conds.append("ts > ?")
            params.append(start_ts)
        if end_ts is not None:
            conds.append("ts < ?")
            params.append(end_ts)

        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY ts ASC"

        try:
            cursor = conn.cursor()
            cursor.execute(q, params)
            columns = [desc[0] for desc in cursor.description]

            events = []
            for row in cursor:
                event_dict = dict(zip(columns, row))
                try:
                    event_dict["ies"] = (
                        loads(event_dict["ies"]) if event_dict.get("ies") else {}
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse IEs for event: {e}")
                    event_dict["ies"] = {}
                events.append(event_dict)
            return events

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to fetch events: {e}",
                error_code="FETCH_EVENTS_FAILED",
                details={"query": q, "params": params, "sqlite_error": str(e)},
            ) from e


class BatchedEventWriter(threading.Thread):
    """Thread-sicherer Event-Writer mit Fehlerbehandlung."""

    def __init__(
        self,
        db_path: str,
        q: "queue.Queue[dict]",
        batch_size: int,
        flush_interval: float,
    ):
        super().__init__(daemon=True, name="BatchedEventWriter")

        # Validiere Parameter
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValidationError(
                f"Invalid batch_size: {batch_size}",
                error_code="INVALID_BATCH_SIZE",
                details={"batch_size": batch_size},
            )

        if not isinstance(flush_interval, (int, float)) or flush_interval <= 0:
            raise ValidationError(
                f"Invalid flush_interval: {flush_interval}",
                error_code="INVALID_FLUSH_INTERVAL",
                details={"flush_interval": flush_interval},
            )

        self.db_path = db_path
        self.q = q
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.stop_event = threading.Event()
        self.error_count = 0
        self.max_errors = 10

        # SQL-Statements
        self._insert_sql_events = "INSERT INTO events(ts,type,bssid,ssid,client,channel,ies,seq,rssi,beacon_interval,cap,beacon_timestamp,is_powersave) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
        self._insert_sql_dns = (
            "INSERT INTO dns_queries(ts, client, domain) VALUES (?,?,?)"
        )
        self._update_sql_hostname = "UPDATE client_labels SET hostname = ? WHERE mac = ? AND (hostname IS NULL OR hostname = '')"
        self._update_sql_ip = "UPDATE client_labels SET ip_address = ? WHERE mac = ?"
        self._insert_sql_network_info = "INSERT OR IGNORE INTO client_labels (mac, device_type, source) VALUES (?, 'unknown', 'network_discovery')"

    def run(self) -> None:
        """Hauptschleife des Event-Writers mit Fehlerbehandlung."""
        conn = None
        try:
            with ErrorContext("batched_event_writer", "EVENT_WRITER_ERROR"):
                conn = sqlite3.connect(self.db_path, timeout=30)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")

                event_buffer, dns_buffer, hostname_buffer, ip_buffer = [], [], [], []
                last_flush = time.time()

                while (
                    not self.stop_event.is_set() and self.error_count < self.max_errors
                ):
                    try:
                        # Hole Event aus Queue
                        item = self.q.get(timeout=self.flush_interval / 2)
                        item_type = item.get("type")

                        # Validiere Event
                        if not isinstance(item, dict):
                            logger.warning(f"Invalid event type: {type(item)}")
                            continue

                        # Verarbeite Event basierend auf Typ
                        if item_type == "dns_query":
                            dns_buffer.append(
                                (item["ts"], item["client"], item.get("dns_query"))
                            )
                        elif item_type == "dhcp_req":
                            hostname_buffer.append(item)
                        elif item_type == "arp_map":
                            ip_buffer.append(item)
                        else:
                            event_buffer.append(self._event_to_tuple(item))

                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
                        self.error_count += 1
                        continue

                    # Prüfe ob Flush nötig ist
                    time_since_flush = time.time() - last_flush
                    buffers_are_full = (
                        len(event_buffer) >= self.batch_size
                        or len(dns_buffer) >= self.batch_size
                    )
                    time_is_up = time_since_flush >= self.flush_interval

                    if (buffers_are_full or time_is_up) and (
                        event_buffer or dns_buffer or hostname_buffer or ip_buffer
                    ):
                        try:
                            self._flush(
                                conn,
                                event_buffer,
                                dns_buffer,
                                hostname_buffer,
                                ip_buffer,
                            )
                            event_buffer.clear()
                            dns_buffer.clear()
                            hostname_buffer.clear()
                            ip_buffer.clear()
                            last_flush = time.time()
                            self.error_count = (
                                0  # Reset error count on successful flush
                            )
                        except Exception as e:
                            logger.error(f"Error during flush: {e}")
                            self.error_count += 1
                            continue

                # Finaler Flush
                if event_buffer or dns_buffer or hostname_buffer or ip_buffer:
                    try:
                        self._flush(
                            conn, event_buffer, dns_buffer, hostname_buffer, ip_buffer
                        )
                    except Exception as e:
                        logger.error(f"Error during final flush: {e}")

                if self.error_count >= self.max_errors:
                    logger.error(
                        f"Event writer stopped due to too many errors ({self.error_count})"
                    )

        except Exception as e:
            logger.error(f"Fatal error in event writer: {e}")
            raise DatabaseError(
                f"Event writer failed: {e}",
                error_code="EVENT_WRITER_FATAL_ERROR",
                details={"db_path": self.db_path, "original_error": str(e)},
            ) from e
        finally:
            if conn:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    logger.warning(f"Error closing database connection: {e}")

    def _event_to_tuple(self, e: dict) -> tuple:
        ies_data = e.get("ies") or {}
        # Convert all keys to strings for JSON serialization
        ies_for_json = {str(k): v for k, v in ies_data.items()}
        return (
            e["ts"],
            e["type"],
            e.get("bssid"),
            e.get("ssid"),
            e.get("client"),
            e.get("channel"),
            dumps(ies_for_json),
            e.get("seq"),
            e.get("rssi"),
            e.get("beacon_interval"),
            e.get("cap"),
            e.get("beacon_timestamp"),
            e.get("is_powersave"),
        )

    def _flush(
        self,
        conn: sqlite3.Connection,
        event_buffer,
        dns_buffer,
        hostname_buffer,
        ip_buffer,
    ):
        """Flushe Puffer zur Datenbank mit Fehlerbehandlung."""
        try:
            with ErrorContext("database_flush", "DB_FLUSH_ERROR"):
                with conn:
                    # Events schreiben
                    if event_buffer:
                        try:
                            conn.executemany(self._insert_sql_events, event_buffer)
                            logger.debug(
                                f"Flushed {len(event_buffer)} events to database"
                            )
                        except sqlite3.Error as e:
                            logger.error(f"Failed to insert events: {e}")
                            raise DatabaseError(
                                f"Event insertion failed: {e}",
                                error_code="EVENT_INSERT_FAILED",
                                details={
                                    "event_count": len(event_buffer),
                                    "sqlite_error": str(e),
                                },
                            ) from e

                    # DNS-Queries schreiben
                    if dns_buffer:
                        try:
                            conn.executemany(self._insert_sql_dns, dns_buffer)
                            logger.debug(
                                f"Flushed {len(dns_buffer)} DNS queries to database"
                            )
                        except sqlite3.Error as e:
                            logger.error(f"Failed to insert DNS queries: {e}")
                            raise DatabaseError(
                                f"DNS insertion failed: {e}",
                                error_code="DNS_INSERT_FAILED",
                                details={
                                    "dns_count": len(dns_buffer),
                                    "sqlite_error": str(e),
                                },
                            ) from e

                    # Hostname-Updates
                    if hostname_buffer:
                        try:
                            conn.executemany(
                                self._insert_sql_network_info,
                                [(item["client"],) for item in hostname_buffer],
                            )
                            conn.executemany(
                                self._update_sql_hostname,
                                [
                                    (item["hostname"], item["client"])
                                    for item in hostname_buffer
                                ],
                            )
                            logger.debug(
                                f"Flushed {len(hostname_buffer)} hostname updates to database"
                            )
                        except sqlite3.Error as e:
                            logger.error(f"Failed to update hostnames: {e}")
                            raise DatabaseError(
                                f"Hostname update failed: {e}",
                                error_code="HOSTNAME_UPDATE_FAILED",
                                details={
                                    "hostname_count": len(hostname_buffer),
                                    "sqlite_error": str(e),
                                },
                            ) from e

                    # IP-Updates
                    if ip_buffer:
                        try:
                            conn.executemany(
                                self._insert_sql_network_info,
                                [(item["arp_mac"],) for item in ip_buffer],
                            )
                            conn.executemany(
                                self._update_sql_ip,
                                [
                                    (item["arp_ip"], item["arp_mac"])
                                    for item in ip_buffer
                                ],
                            )
                            logger.debug(
                                f"Flushed {len(ip_buffer)} IP updates to database"
                            )
                        except sqlite3.Error as e:
                            logger.error(f"Failed to update IPs: {e}")
                            raise DatabaseError(
                                f"IP update failed: {e}",
                                error_code="IP_UPDATE_FAILED",
                                details={
                                    "ip_count": len(ip_buffer),
                                    "sqlite_error": str(e),
                                },
                            ) from e

        except DatabaseError:
            raise  # Re-raise DatabaseError
        except sqlite3.Error as e:
            logger.error(f"Database flush failed: {e}")
            raise DatabaseError(
                f"Database flush failed: {e}",
                error_code="DB_FLUSH_FAILED",
                details={"sqlite_error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during flush: {e}")
            raise DatabaseError(
                f"Unexpected flush error: {e}",
                error_code="FLUSH_UNEXPECTED_ERROR",
                details={"original_error": str(e)},
            ) from e

    def stop(self, wait: bool = True, timeout: float = 5.0):
        self.stop_event.set()
        if wait and self.is_alive():
            self.join(timeout=timeout)
            if self.is_alive():
                logger.warning("BatchedEventWriter wurde nicht sauber beendet.")


@handle_errors(DatabaseError, "DB_ADD_EVENT_ERROR", reraise=True)
def add_event(conn: sqlite3.Connection, event: dict):
    """Füge Event zur Datenbank hinzu mit Fehlerbehandlung."""
    with ErrorContext("add_event", "DB_ADD_EVENT_ERROR"):
        # Validiere Event
        if not isinstance(event, dict):
            raise ValidationError(
                f"Invalid event type: {type(event)}",
                error_code="INVALID_EVENT_TYPE",
                details={"event_type": str(type(event))},
            )

        required_fields = ["ts", "type"]
        for field in required_fields:
            if field not in event:
                raise ValidationError(
                    f"Missing required field: {field}",
                    error_code="MISSING_EVENT_FIELD",
                    details={"field": field, "event": event},
                )

        try:
            # Konvertiere Event zu Tupel
            ies_data = event.get("ies", {})
            # Convert all keys to strings for JSON serialization
            ies_for_json = {str(k): v for k, v in ies_data.items()}
            
            event_tuple = (
                event["ts"],
                event["type"],
                event.get("bssid"),
                event.get("ssid"),
                event.get("client"),
                event.get("channel"),
                dumps(ies_for_json),
                event.get("seq"),
                event.get("rssi"),
                event.get("beacon_interval"),
                event.get("cap"),
                event.get("beacon_timestamp"),
                event.get("is_powersave"),
            )

            # Füge Event zur Datenbank hinzu
            conn.execute(
                "INSERT INTO events(ts,type,bssid,ssid,client,channel,ies,seq,rssi,beacon_interval,cap,beacon_timestamp,is_powersave) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                event_tuple,
            )
            conn.commit()
            logger.debug(f"Added event: {event['type']} at {event['ts']}")

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to add event: {e}",
                error_code="ADD_EVENT_FAILED",
                details={"event": event, "sqlite_error": str(e)},
            ) from e


@handle_errors(DatabaseError, "DB_ADD_LABEL_ERROR", reraise=True)
def add_label(
    conn: sqlite3.Connection,
    ssid: str,
    bssid: str,
    label: Optional[int],
    source: str = "ui",
):
    """Füge Label zur Datenbank hinzu mit Fehlerbehandlung."""
    with ErrorContext("add_label", "DB_ADD_LABEL_ERROR"):
        # Validiere Parameter
        if not isinstance(ssid, str) or not ssid.strip():
            raise ValidationError(
                f"Invalid SSID: {ssid}",
                error_code="INVALID_SSID",
                details={"ssid": ssid},
            )

        if not isinstance(bssid, str) or not bssid.strip():
            raise ValidationError(
                f"Invalid BSSID: {bssid}",
                error_code="INVALID_BSSID",
                details={"bssid": bssid},
            )

        if label is not None and not isinstance(label, int):
            raise ValidationError(
                f"Invalid label type: {type(label)}",
                error_code="INVALID_LABEL_TYPE",
                details={"label": label, "type": str(type(label))},
            )

        if not isinstance(source, str) or not source.strip():
            raise ValidationError(
                f"Invalid source: {source}",
                error_code="INVALID_SOURCE",
                details={"source": source},
            )

        try:
            with conn:
                conn.execute(
                    "INSERT INTO labels(ssid,bssid,label,source,ts) VALUES (?,?,?,?,?)",
                    (ssid.strip(), bssid.strip(), label, source.strip(), time.time()),
                )
                logger.debug(
                    f"Added label: {ssid} -> {bssid} = {label} (source: {source})"
                )
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to add label: {e}",
                error_code="ADD_LABEL_FAILED",
                details={
                    "ssid": ssid,
                    "bssid": bssid,
                    "label": label,
                    "source": source,
                    "sqlite_error": str(e),
                },
            ) from e


@handle_errors(FileSystemError, "CSV_EXPORT_ERROR", default_return=0)
def export_confirmed_to_csv(label_db_path: str, out_csv: str) -> int:
    """Exportiere bestätigte Labels zu CSV mit Fehlerbehandlung."""
    with ErrorContext("csv_export", "CSV_EXPORT_ERROR"):
        # Validiere Eingabepfade
        if not _validate_file_path(label_db_path):
            raise ValidationError(
                f"Invalid label database path: {label_db_path}",
                error_code="INVALID_LABEL_DB_PATH",
                details={"label_db_path": label_db_path},
            )

        if not _validate_file_path(out_csv):
            raise ValidationError(
                f"Invalid output CSV path: {out_csv}",
                error_code="INVALID_CSV_PATH",
                details={"out_csv": out_csv},
            )

        # Prüfe ob Label-DB existiert
        if not os.path.exists(label_db_path):
            raise FileSystemError(
                f"Label database not found: {label_db_path}",
                error_code="LABEL_DB_NOT_FOUND",
                details={"label_db_path": label_db_path},
            )

        # Erstelle Ausgabeverzeichnis falls nötig
        out_dir = Path(out_csv).parent
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise FileSystemError(
                    f"Cannot create output directory: {out_dir}",
                    error_code="CSV_DIR_CREATION_FAILED",
                    details={"directory": str(out_dir), "original_error": str(e)},
                ) from e

        # Lade und verarbeite Daten
        try:
            with db_conn_ctx(label_db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT ssid,bssid,label,ts FROM labels WHERE label IS NOT NULL"
                )
                rows = cur.fetchall()

            if not rows:
                logger.warning("No confirmed labels found for export")
                return 0

            # Erstelle Mapping (neueste Labels pro SSID-BSSID-Paar)
            mapping = {}
            for ssid, bssid, label, ts in rows:
                key = (ssid, bssid)
                if (prev := mapping.get(key)) is None or ts > prev["ts"]:
                    mapping[key] = {
                        "ssid": ssid,
                        "bssid": bssid,
                        "label": label,
                        "ts": ts,
                    }

            # Schreibe CSV
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ssid", "bssid", "label"])
                for v in mapping.values():
                    writer.writerow([v["ssid"], v["bssid"], v["label"]])

            logger.info(f"Exported {len(mapping)} confirmed labels to {out_csv}")
            return len(mapping)

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to read labels from database: {e}",
                error_code="LABEL_READ_FAILED",
                details={"label_db_path": label_db_path, "sqlite_error": str(e)},
            ) from e
        except OSError as e:
            raise FileSystemError(
                f"Failed to write CSV file: {e}",
                error_code="CSV_WRITE_FAILED",
                details={"out_csv": out_csv, "original_error": str(e)},
            ) from e


@handle_errors(DatabaseError, "SAVE_LABELS_ERROR", reraise=True)
def save_labels_from_map(device_map, label_db_path, console):
    """Speichert die identifizierten Geräte-Labels aus einem Dictionary in die Datenbank mit Fehlerbehandlung."""
    with ErrorContext("save_labels_from_map", "SAVE_LABELS_ERROR"):
        if not device_map:
            console.print("[yellow]No device map provided, nothing to save.[/yellow]")
            return

        # Validiere Parameter
        if not isinstance(device_map, dict):
            raise ValidationError(
                f"Invalid device_map type: {type(device_map)}",
                error_code="INVALID_DEVICE_MAP_TYPE",
                details={"device_map_type": str(type(device_map))},
            )

        if not _validate_file_path(label_db_path):
            raise ValidationError(
                f"Invalid label database path: {label_db_path}",
                error_code="INVALID_LABEL_DB_PATH",
                details={"label_db_path": label_db_path},
            )

        console.print(
            f"\n[green]Saving {len(device_map)} device labels to database...[/green]"
        )

        try:
            with db_conn_ctx(label_db_path) as conn:
                cursor = conn.cursor()
                labels_added = 0
                errors = []

                for mac, info in device_map.items():
                    try:
                        # Validiere MAC-Adresse
                        if not isinstance(mac, str) or not mac.strip():
                            logger.warning(f"Invalid MAC address: {mac}")
                            continue

                        device_type = info.get("device_type")
                        if not device_type:
                            logger.warning(f"No device type for MAC {mac}")
                            continue

                        # Validiere device_type
                        if not isinstance(device_type, str) or not device_type.strip():
                            logger.warning(
                                f"Invalid device type for MAC {mac}: {device_type}"
                            )
                            continue

                        cursor.execute(
                            "INSERT OR REPLACE INTO client_labels (mac, device_type, source, ts, hostname, ip_address) VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                mac.strip(),
                                device_type.strip(),
                                info.get("source", "profiler"),
                                time.time(),
                                info.get("hostname"),
                                info.get("ip_address"),
                            ),
                        )
                        labels_added += 1

                    except sqlite3.Error as e:
                        error_msg = f"Error saving label for {mac}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue
                    except Exception as e:
                        error_msg = f"Unexpected error saving label for {mac}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue

                conn.commit()

                if errors:
                    console.print(
                        f"[yellow]Saved {labels_added} labels with {len(errors)} errors.[/yellow]"
                    )
                    for error in errors[:5]:  # Zeige nur erste 5 Fehler
                        console.print(f"[red]  {error}[/red]")
                    if len(errors) > 5:
                        console.print(
                            f"[red]  ... and {len(errors) - 5} more errors[/red]"
                        )
                else:
                    console.print(
                        f"[bold green]{labels_added} labels successfully saved.[/bold green]"
                    )

        except sqlite3.Error as e:
            error_msg = f"Database access error: {e}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            raise DatabaseError(
                error_msg,
                error_code="LABEL_SAVE_DB_ERROR",
                details={
                    "label_db_path": label_db_path,
                    "device_count": len(device_map),
                    "sqlite_error": str(e),
                },
            ) from e
