
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funktionen und Klassen für die SQLite-Datenbankinteraktion.
"""
from __future__ import annotations
import json
import logging
import queue
import sqlite3
import threading
import time
import os
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Iterator

# Relative Imports
from .. import config

logger = logging.getLogger(__name__)

try:
    import orjson as _json
    def dumps(o): return _json.dumps(o).decode()
    def loads(s): return _json.loads(s)
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
    conn = sqlite3.connect(path, timeout=30)
    try: yield conn
    finally: conn.close()
    
def migrate_db(db_path: str):
    migrations_path = Path(__file__).parent.parent.parent / "data" / "sql_data" / "versions"
    if not migrations_path.exists():
        logger.error(f"Migrations-Verzeichnis nicht gefunden: {migrations_path}")
        return
    migration_files = sorted(migrations_path.glob("*.sql"))
    with db_conn_ctx(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        current_version = _get_db_version(conn)
        logger.info(f"Aktuelle DB-Version: {current_version}. Suche nach Migrationen...")
        for migration_file in migration_files:
            try:
                file_version = int(migration_file.name.split('_')[0])
                if file_version > current_version:
                    logger.info(f"Wende Migration an: {migration_file.name} (Version {file_version})")
                    sql_script = migration_file.read_text()
                    conn.executescript(sql_script)
                    conn.commit()
                    new_version = _get_db_version(conn)
                    logger.info(f"DB erfolgreich auf Version {new_version} migriert.")
                    current_version = new_version
            except (ValueError, IndexError):
                logger.warning(f"Migrationsdatei hat ungültigen Namen und wird ignoriert: {migration_file.name}")
    logger.info("Datenbank ist auf dem neuesten Stand.")

def fetch_events(conn: sqlite3.Connection, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> Iterator[Dict]:
    q = "SELECT ts, type, bssid, ssid, client, channel, ies, seq, rssi, beacon_interval, cap, beacon_timestamp, is_powersave FROM events"
    conds, params = [], []
    if start_ts is not None: conds.append("ts > ?"); params.append(start_ts)
    if end_ts is not None: conds.append("ts < ?"); params.append(end_ts)
    if conds: q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY ts ASC"
    cursor = conn.cursor()
    cursor.execute(q, params)
    columns = [desc[0] for desc in cursor.description]
    for row in cursor:
        event_dict = dict(zip(columns, row))
        try:
            event_dict["ies"] = loads(event_dict["ies"]) if event_dict.get("ies") else {}
        except (json.JSONDecodeError, ValueError):
            event_dict["ies"] = {}
        yield event_dict


class BatchedEventWriter(threading.Thread):
    def __init__(self, db_path: str, q: "queue.Queue[dict]", batch_size: int, flush_interval: float):
        super().__init__(daemon=True, name="BatchedEventWriter")
        self.db_path = db_path
        self.q = q
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.stop_event = threading.Event()
        self._insert_sql_events = "INSERT INTO events(ts,type,bssid,ssid,client,channel,ies,seq,rssi,beacon_interval,cap,beacon_timestamp,is_powersave) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
        self._insert_sql_dns = "INSERT INTO dns_queries(ts, client, domain) VALUES (?,?,?)"
        self._update_sql_hostname = "UPDATE client_labels SET hostname = ? WHERE mac = ? AND (hostname IS NULL OR hostname = '')"
        self._update_sql_ip = "UPDATE client_labels SET ip_address = ? WHERE mac = ?"
        self._insert_sql_network_info = "INSERT OR IGNORE INTO client_labels (mac, device_type, source) VALUES (?, 'unknown', 'network_discovery')"

    def run(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        event_buffer, dns_buffer, hostname_buffer, ip_buffer = [], [], [], []
        last_flush = time.time()
        
        while not self.stop_event.is_set():
            try:
                item = self.q.get(timeout=self.flush_interval / 2)
                item_type = item.get('type')
                
                if item_type == 'dns_query': dns_buffer.append((item["ts"], item["client"], item.get("dns_query")))
                elif item_type == 'dhcp_req': hostname_buffer.append(item)
                elif item_type == 'arp_map': ip_buffer.append(item)
                else: event_buffer.append(self._event_to_tuple(item))
            
            except queue.Empty:
                pass

            time_since_flush = time.time() - last_flush
            buffers_are_full = len(event_buffer) >= self.batch_size or len(dns_buffer) >= self.batch_size
            time_is_up = time_since_flush >= self.flush_interval
            
            if (buffers_are_full or time_is_up) and (event_buffer or dns_buffer or hostname_buffer or ip_buffer):
                self._flush(conn, event_buffer, dns_buffer, hostname_buffer, ip_buffer)
                event_buffer.clear(); dns_buffer.clear(); hostname_buffer.clear(); ip_buffer.clear()
                last_flush = time.time()
        
        if event_buffer or dns_buffer or hostname_buffer or ip_buffer:
            self._flush(conn, event_buffer, dns_buffer, hostname_buffer, ip_buffer)
        conn.close()

    def _event_to_tuple(self, e: dict) -> tuple:
        ies_data = e.get("ies") or {}; ies_for_json = {str(k): v for k, v in ies_data.items()}
        return (e["ts"], e["type"], e.get("bssid"), e.get("ssid"), e.get("client"), e.get("channel"), dumps(ies_for_json), e.get("seq"), e.get("rssi"), e.get("beacon_interval"), e.get("cap"), e.get("beacon_timestamp"), e.get("is_powersave"))

    def _flush(self, conn: sqlite3.Connection, event_buffer, dns_buffer, hostname_buffer, ip_buffer):
        try:
            with conn:
                if event_buffer: conn.executemany(self._insert_sql_events, event_buffer)
                if dns_buffer: conn.executemany(self._insert_sql_dns, dns_buffer)
                if hostname_buffer:
                    conn.executemany(self._insert_sql_network_info, [(item['client'],) for item in hostname_buffer])
                    conn.executemany(self._update_sql_hostname, [(item['hostname'], item['client']) for item in hostname_buffer])
                if ip_buffer:
                    conn.executemany(self._insert_sql_network_info, [(item['arp_mac'],) for item in ip_buffer])
                    conn.executemany(self._update_sql_ip, [(item['arp_ip'], item['arp_mac']) for item in ip_buffer])
        except sqlite3.Error as e: logger.error("DB-Flush fehlgeschlagen: %s", e)

    def stop(self, wait: bool = True, timeout: float = 5.0):
        self.stop_event.set()
        if wait and self.is_alive():
            self.join(timeout=timeout)
            if self.is_alive(): logger.warning("BatchedEventWriter wurde nicht sauber beendet.")

def add_label(conn: sqlite3.Connection, ssid: str, bssid: str, label: Optional[int], source: str = "ui"):
    with conn: conn.execute("INSERT INTO labels(ssid,bssid,label,source,ts) VALUES (?,?,?,?,?)", (ssid, bssid, label, source, time.time()))

def export_confirmed_to_csv(label_db_path: str, out_csv: str) -> int:
    if not os.path.exists(label_db_path):
        logger.error("Label-DB nicht gefunden: %s", label_db_path)
        return 0
    with db_conn_ctx(label_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT ssid,bssid,label,ts FROM labels WHERE label IS NOT NULL")
        rows = cur.fetchall()
    
    mapping = {}
    for ssid, bssid, label, ts in rows:
        key = (ssid, bssid)
        if (prev := mapping.get(key)) is None or ts > prev["ts"]:
            mapping[key] = {"ssid": ssid, "bssid": bssid, "label": label, "ts": ts}
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ssid", "bssid", "label"])
        for v in mapping.values(): writer.writerow([v["ssid"], v["bssid"], v["label"]])
    
    logger.info("Exportierte %d bestätigte Labels nach %s", len(mapping), out_csv)
    return len(mapping)


def save_labels_from_map(device_map, label_db_path, console):
    """Speichert die identifizierten Geräte-Labels aus einem Dictionary in die Datenbank."""
    if not device_map:
        return
        
    console.print(f"\n[green]Speichere {len(device_map)} Geräte-Labels in die Datenbank...[/green]")
    try:
        with db_conn_ctx(label_db_path) as conn:
            cursor = conn.cursor()
            labels_added = 0
            for mac, info in device_map.items():
                device_type = info.get('device_type')
                if device_type:
                    try:
                        cursor.execute(
                            "INSERT OR REPLACE INTO client_labels (mac, device_type, source, ts, hostname, ip_address) VALUES (?, ?, ?, ?, ?, ?)",
                            (mac, device_type, info.get('source', 'profiler'), time.time(), info.get('hostname'), info.get('ip_address'))
                        )
                        labels_added += 1
                    except sqlite3.Error as e:
                        logger.error(f"Fehler beim Speichern des Labels für {mac}: {e}")
            conn.commit()
            console.print(f"[bold green]{labels_added} Labels erfolgreich geschrieben.[/bold green]")
    except sqlite3.Error as e:
        console.print(f"[bold red]Fehler beim Zugriff auf die Label-Datenbank: {e}[/bold red]")