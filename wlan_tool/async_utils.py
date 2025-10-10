#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async-Utilities für das WLAN-Analyse-Tool.
"""

import asyncio
import aiofiles
import aiosqlite
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable, TypeVar
from pathlib import Path
import json
import time
from contextlib import asynccontextmanager

from .constants import Constants, ErrorCodes, get_error_message
from .exceptions import DatabaseError, FileSystemError, ValidationError
from .validation import validate_path, validate_timestamp, validate_mac


T = TypeVar('T')


class AsyncDatabaseManager:
    """Async Database Manager für bessere Performance."""
    
    def __init__(self, db_path: str):
        self.db_path = validate_path(db_path, must_exist=False)
        self._connection_pool: List[aiosqlite.Connection] = []
        self._max_connections = 10
        self._current_connections = 0
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
    
    async def initialize(self) -> None:
        """Initialisiere Connection Pool."""
        # Erstelle Verzeichnis falls nötig
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Erstelle initiale Verbindung
        conn = await aiosqlite.connect(self.db_path)
        await self._setup_connection(conn)
        self._connection_pool.append(conn)
        self._current_connections = 1
    
    async def _setup_connection(self, conn: aiosqlite.Connection) -> None:
        """Richte Datenbankverbindung ein."""
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.execute("PRAGMA cache_size = 10000")
        await conn.execute("PRAGMA temp_store = MEMORY")
        await conn.commit()
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Hole verfügbare Verbindung aus Pool."""
        if self._connection_pool:
            return self._connection_pool.pop()
        
        if self._current_connections < self._max_connections:
            conn = await aiosqlite.connect(self.db_path)
            await self._setup_connection(conn)
            self._current_connections += 1
            return conn
        
        # Warte auf verfügbare Verbindung
        while not self._connection_pool:
            await asyncio.sleep(0.01)
        
        return self._connection_pool.pop()
    
    async def return_connection(self, conn: aiosqlite.Connection) -> None:
        """Gebe Verbindung an Pool zurück."""
        if len(self._connection_pool) < self._max_connections:
            self._connection_pool.append(conn)
    
    async def close_all(self) -> None:
        """Schließe alle Verbindungen."""
        for conn in self._connection_pool:
            await conn.close()
        self._connection_pool.clear()
        self._current_connections = 0
    
    @asynccontextmanager
    async def get_connection_context(self):
        """Context Manager für Datenbankverbindung."""
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.return_connection(conn)


class AsyncEventProcessor:
    """Async Event Processor für bessere Performance."""
    
    def __init__(self, db_manager: AsyncDatabaseManager, batch_size: int = Constants.DEFAULT_BATCH_SIZE):
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=Constants.MAX_QUEUE_SIZE)
        self.processing_tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
    
    async def start_processing(self, num_workers: int = 3) -> None:
        """Starte Event-Processing-Worker."""
        for i in range(num_workers):
            task = asyncio.create_task(self._process_events_worker(f"worker-{i}"))
            self.processing_tasks.append(task)
    
    async def stop_processing(self) -> None:
        """Stoppe Event-Processing."""
        self._stop_event.set()
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
    
    async def add_event(self, event: Dict[str, Any]) -> None:
        """Füge Event zur Verarbeitungs-Queue hinzu."""
        await self.event_queue.put(event)
    
    async def _process_events_worker(self, worker_name: str) -> None:
        """Worker für Event-Verarbeitung."""
        batch = []
        
        while not self._stop_event.is_set():
            try:
                # Warte auf Event mit Timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                batch.append(event)
                
                # Verarbeite Batch wenn voll oder nach Timeout
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch, worker_name)
                    batch = []
                
            except asyncio.TimeoutError:
                # Verarbeite verbleibende Events im Batch
                if batch:
                    await self._process_batch(batch, worker_name)
                    batch = []
            except Exception as e:
                print(f"Error in {worker_name}: {e}")
        
        # Verarbeite verbleibende Events
        if batch:
            await self._process_batch(batch, worker_name)
    
    async def _process_batch(self, events: List[Dict[str, Any]], worker_name: str) -> None:
        """Verarbeite Batch von Events."""
        if not events:
            return
        
        async with self.db_manager.get_connection_context() as conn:
            try:
                await self._insert_events_batch(conn, events)
                await conn.commit()
                print(f"{worker_name}: Processed {len(events)} events")
            except Exception as e:
                print(f"Error processing batch in {worker_name}: {e}")
                await conn.rollback()
    
    async def _insert_events_batch(self, conn: aiosqlite.Connection, events: List[Dict[str, Any]]) -> None:
        """Füge Batch von Events in Datenbank ein."""
        sql = """
        INSERT INTO events(ts, type, bssid, ssid, client, channel, ies, seq, rssi, 
                          beacon_interval, cap, beacon_timestamp, is_powersave)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        event_tuples = []
        for event in events:
            # Validiere und normalisiere Event
            validated_event = await self._validate_event(event)
            event_tuple = self._event_to_tuple(validated_event)
            event_tuples.append(event_tuple)
        
        await conn.executemany(sql, event_tuples)
    
    async def _validate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Validiere Event-Daten."""
        validated = event.copy()
        
        # Validiere Zeitstempel
        if 'ts' in validated:
            validated['ts'] = validate_timestamp(validated['ts'])
        
        # Validiere MAC-Adressen
        for field in ['bssid', 'client']:
            if field in validated and validated[field]:
                validated[field] = validate_mac(validated[field])
        
        # Validiere RSSI
        if 'rssi' in validated and validated['rssi'] is not None:
            from .validation import validate_rssi
            validated['rssi'] = validate_rssi(validated['rssi'])
        
        return validated
    
    def _event_to_tuple(self, event: Dict[str, Any]) -> tuple:
        """Konvertiere Event zu Tuple für Datenbank."""
        ies_data = event.get("ies") or {}
        ies_json = json.dumps({str(k): v for k, v in ies_data.items()})
        
        return (
            event.get("ts"),
            event.get("type"),
            event.get("bssid"),
            event.get("ssid"),
            event.get("client"),
            event.get("channel"),
            ies_json,
            event.get("seq"),
            event.get("rssi"),
            event.get("beacon_interval"),
            event.get("cap"),
            event.get("beacon_timestamp"),
            event.get("is_powersave")
        )


class AsyncFileManager:
    """Async File Manager für bessere I/O-Performance."""
    
    @staticmethod
    async def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Lese Datei asynchron."""
        path = validate_path(file_path, must_exist=True)
        
        try:
            async with aiofiles.open(path, 'r', encoding=encoding) as f:
                return await f.read()
        except Exception as e:
            raise FileSystemError(
                get_error_message(ErrorCodes.FILE_READ_ERROR, file_path=str(path)),
                error_code=ErrorCodes.FILE_READ_ERROR.value,
                details={"file_path": str(path), "error": str(e)}
            ) from e
    
    @staticmethod
    async def write_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """Schreibe Datei asynchron."""
        path = validate_path(file_path, must_exist=False)
        
        # Erstelle Verzeichnis falls nötig
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(path, 'w', encoding=encoding) as f:
                await f.write(content)
        except Exception as e:
            raise FileSystemError(
                get_error_message(ErrorCodes.FILE_WRITE_ERROR, file_path=str(path)),
                error_code=ErrorCodes.FILE_WRITE_ERROR.value,
                details={"file_path": str(path), "error": str(e)}
            ) from e
    
    @staticmethod
    async def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Lese JSON-Datei asynchron."""
        content = await AsyncFileManager.read_file(file_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in file {file_path}: {e}",
                error_code=ErrorCodes.CONFIG_PARSE_ERROR.value,
                details={"file_path": str(file_path), "error": str(e)}
            ) from e
    
    @staticmethod
    async def write_json(file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
        """Schreibe JSON-Datei asynchron."""
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        await AsyncFileManager.write_file(file_path, content)


class AsyncMetricsCollector:
    """Async Metrics Collector für Performance-Monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def increment_counter(self, name: str, value: int = 1) -> None:
        """Inkrementiere Counter-Metrik."""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = 0
            self.metrics[name] += value
    
    async def set_gauge(self, name: str, value: Union[int, float]) -> None:
        """Setze Gauge-Metrik."""
        async with self._lock:
            self.metrics[name] = value
    
    async def record_timing(self, name: str, duration: float) -> None:
        """Recorde Timing-Metrik."""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Hole alle Metriken."""
        async with self._lock:
            return self.metrics.copy()
    
    async def reset_metrics(self) -> None:
        """Setze alle Metriken zurück."""
        async with self._lock:
            self.metrics.clear()


# Convenience-Funktionen
async def run_async_function(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Führe synchrone Funktion asynchron aus."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def run_async_functions_parallel(*functions: Callable[..., T]) -> List[T]:
    """Führe mehrere Funktionen parallel aus."""
    tasks = [asyncio.create_task(run_async_function(func)) for func in functions]
    return await asyncio.gather(*tasks)


async def with_timeout(coro: Any, timeout: float = Constants.DEFAULT_TIMEOUT) -> Any:
    """Führe Coroutine mit Timeout aus."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


# Async Context Manager für bessere Resource-Verwaltung
@asynccontextmanager
async def async_database_context(db_path: str):
    """Async Context Manager für Datenbankverbindung."""
    db_manager = AsyncDatabaseManager(db_path)
    try:
        await db_manager.initialize()
        yield db_manager
    finally:
        await db_manager.close_all()


@asynccontextmanager
async def async_file_context(file_path: Union[str, Path], mode: str = 'r'):
    """Async Context Manager für Dateioperationen."""
    path = validate_path(file_path, must_exist=(mode == 'r'))
    
    if mode == 'r':
        file_handle = await aiofiles.open(path, 'r')
    elif mode == 'w':
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = await aiofiles.open(path, 'w')
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    try:
        yield file_handle
    finally:
        await file_handle.close()