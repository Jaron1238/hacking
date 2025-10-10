#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caching Layer für das WLAN-Analyse-Tool.
"""

import time
import hashlib
import pickle
import json
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar, Generic
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from functools import wraps
import threading
from collections import OrderedDict

from .constants import Constants, ErrorCodes, get_error_message
from .exceptions import ValidationError, ResourceError


T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class CacheEntry(Generic[T]):
    """Cache-Eintrag mit Metadaten."""
    
    def __init__(self, value: T, created_at: float, ttl: float):
        self.value: T = value
        self.created_at: float = created_at
        self.ttl: float = ttl
        self.access_count: int = 0
        self.last_accessed: float = created_at
    
    @property
    def is_expired(self) -> bool:
        """Prüfe ob Eintrag abgelaufen ist."""
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Alter des Eintrags in Sekunden."""
        return time.time() - self.created_at
    
    def access(self) -> None:
        """Markiere Zugriff."""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache(Generic[T]):
    """In-Memory Cache mit LRU-Eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = Constants.CACHE_TTL_S):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Hole Wert aus Cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access()
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Setze Wert im Cache."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(value, time.time(), ttl)
            
            if key in self._cache:
                # Update existing entry
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Evict least recently used
                self._cache.popitem(last=False)
            
            self._cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Lösche Eintrag aus Cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Leere gesamten Cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def cleanup_expired(self) -> int:
        """Entferne abgelaufene Einträge."""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Hole Cache-Statistiken."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "oldest_entry": min(entry.age for entry in self._cache.values()) if self._cache else 0,
                "newest_entry": max(entry.age for entry in self._cache.values()) if self._cache else 0
            }


class FileCache(Generic[T]):
    """File-based Cache für persistente Speicherung."""
    
    def __init__(self, cache_dir: Union[str, Path], max_file_size: int = Constants.MAX_FILE_SIZE_MB * 1024 * 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self._lock = threading.RLock()
    
    def _get_file_path(self, key: str) -> Path:
        """Hole Dateipfad für Cache-Key."""
        # Erstelle Hash für sicheren Dateinamen
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[T]:
        """Hole Wert aus File-Cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Prüfe TTL
                if data.get('expires_at', 0) < time.time():
                    file_path.unlink()
                    return None
                
                return data['value']
            
            except (pickle.PickleError, KeyError, OSError):
                # Cache-Datei ist beschädigt, lösche sie
                try:
                    file_path.unlink()
                except OSError:
                    pass
                return None
    
    def set(self, key: str, value: T, ttl: float = Constants.CACHE_TTL_S) -> None:
        """Setze Wert im File-Cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            try:
                data = {
                    'value': value,
                    'created_at': time.time(),
                    'expires_at': time.time() + ttl
                }
                
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Prüfe Dateigröße
                if file_path.stat().st_size > self.max_file_size:
                    file_path.unlink()
                    raise ResourceError(
                        f"Cache entry too large: {file_path.stat().st_size} bytes (max: {self.max_file_size})",
                        error_code=ErrorCodes.RESOURCE_DISK_FULL.value,
                        details={"file_path": str(file_path), "size": file_path.stat().st_size, "max_size": self.max_file_size}
                    )
            
            except OSError as e:
                raise ResourceError(
                    f"Failed to write cache file: {e}",
                    error_code=ErrorCodes.FILE_WRITE_ERROR.value,
                    details={"file_path": str(file_path), "error": str(e)}
                ) from e
    
    def delete(self, key: str) -> bool:
        """Lösche Eintrag aus File-Cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except OSError:
                    return False
            return False
    
    def cleanup_expired(self) -> int:
        """Entferne abgelaufene Dateien."""
        with self._lock:
            removed_count = 0
            current_time = time.time()
            
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if data.get('expires_at', 0) < current_time:
                        file_path.unlink()
                        removed_count += 1
                
                except (pickle.PickleError, KeyError, OSError):
                    # Beschädigte Datei löschen
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except OSError:
                        pass
            
            return removed_count
    
    def clear(self) -> None:
        """Leere gesamten File-Cache."""
        with self._lock:
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    file_path.unlink()
                except OSError:
                    pass


class HybridCache(Generic[T]):
    """Hybrid Cache mit Memory + File Backend."""
    
    def __init__(self, memory_size: int = 1000, cache_dir: Optional[Union[str, Path]] = None):
        self.memory_cache = MemoryCache[T](max_size=memory_size)
        self.file_cache = FileCache[T](cache_dir) if cache_dir else None
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[T]:
        """Hole Wert aus Hybrid-Cache."""
        with self._lock:
            # Versuche Memory-Cache zuerst
            value = self.memory_cache.get(key)
            if value is not None:
                return value
            
            # Falls nicht im Memory, versuche File-Cache
            if self.file_cache:
                value = self.file_cache.get(key)
                if value is not None:
                    # Lade zurück in Memory-Cache
                    self.memory_cache.set(key, value)
                    return value
            
            return None
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Setze Wert im Hybrid-Cache."""
        with self._lock:
            # Setze in Memory-Cache
            self.memory_cache.set(key, value, ttl)
            
            # Setze auch in File-Cache falls verfügbar
            if self.file_cache:
                try:
                    self.file_cache.set(key, value, ttl or Constants.CACHE_TTL_S)
                except ResourceError:
                    # File-Cache-Fehler ignorieren, Memory-Cache funktioniert noch
                    pass
    
    def delete(self, key: str) -> bool:
        """Lösche Eintrag aus Hybrid-Cache."""
        with self._lock:
            memory_deleted = self.memory_cache.delete(key)
            file_deleted = self.file_cache.delete(key) if self.file_cache else True
            return memory_deleted or file_deleted
    
    def clear(self) -> None:
        """Leere gesamten Hybrid-Cache."""
        with self._lock:
            self.memory_cache.clear()
            if self.file_cache:
                self.file_cache.clear()
    
    def cleanup_expired(self) -> int:
        """Entferne abgelaufene Einträge."""
        with self._lock:
            memory_cleaned = self.memory_cache.cleanup_expired()
            file_cleaned = self.file_cache.cleanup_expired() if self.file_cache else 0
            return memory_cleaned + file_cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Hole Cache-Statistiken."""
        with self._lock:
            stats = self.memory_cache.get_stats()
            if self.file_cache:
                stats["file_cache_enabled"] = True
            else:
                stats["file_cache_enabled"] = False
            return stats


# Global Cache Instanzen
_global_caches: Dict[str, HybridCache] = {}


def get_cache(name: str = "default", **kwargs) -> HybridCache:
    """Hole globale Cache-Instanz."""
    if name not in _global_caches:
        _global_caches[name] = HybridCache(**kwargs)
    return _global_caches[name]


def clear_all_caches() -> None:
    """Leere alle globalen Caches."""
    for cache in _global_caches.values():
        cache.clear()


# Decorator für automatisches Caching
def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """Decorator für automatisches Caching von Funktionen."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache(cache_name)
            
            # Erstelle Cache-Key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Standard-Key basierend auf Funktionsname und Argumenten
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Versuche aus Cache zu laden
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Führe Funktion aus und speichere Ergebnis
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper  # type: ignore
    return decorator


# Spezielle Cache-Implementierungen
class AnalysisResultCache:
    """Spezieller Cache für Analyse-Ergebnisse."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache = HybridCache[Dict[str, Any]](
            memory_size=100,
            cache_dir=cache_dir
        )
    
    def get_client_features(self, client_mac: str) -> Optional[Dict[str, Any]]:
        """Hole Client-Features aus Cache."""
        return self.cache.get(f"client_features:{client_mac}")
    
    def set_client_features(self, client_mac: str, features: Dict[str, Any], ttl: float = 3600) -> None:
        """Setze Client-Features im Cache."""
        self.cache.set(f"client_features:{client_mac}", features, ttl)
    
    def get_clustering_result(self, state_hash: str) -> Optional[Dict[str, Any]]:
        """Hole Clustering-Ergebnis aus Cache."""
        return self.cache.get(f"clustering:{state_hash}")
    
    def set_clustering_result(self, state_hash: str, result: Dict[str, Any], ttl: float = 1800) -> None:
        """Setze Clustering-Ergebnis im Cache."""
        self.cache.set(f"clustering:{state_hash}", result, ttl)
    
    def get_vendor_lookup(self, mac: str) -> Optional[str]:
        """Hole Vendor-Lookup aus Cache."""
        return self.cache.get(f"vendor:{mac}")
    
    def set_vendor_lookup(self, mac: str, vendor: str, ttl: float = 86400) -> None:
        """Setze Vendor-Lookup im Cache."""
        self.cache.set(f"vendor:{mac}", vendor, ttl)


# Async Cache Support
class AsyncCacheManager:
    """Async Cache Manager für bessere Performance."""
    
    def __init__(self, cache: HybridCache[T]):
        self.cache = cache
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_task(self, interval: float = 300) -> None:
        """Starte automatische Cleanup-Task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                self.cache.cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stoppe Cleanup-Task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def get(self, key: str) -> Optional[T]:
        """Async get."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cache.get, key)
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Async set."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cache.set, key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Async delete."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cache.delete, key)
    
    async def cleanup_expired(self) -> int:
        """Async cleanup."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cache.cleanup_expired)