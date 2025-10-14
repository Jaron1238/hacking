#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Recovery-Mechanismen für das WLAN-Analyse-Tool.
"""

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from .exceptions import (
    AnalysisError,
    DatabaseError,
    FileSystemError,
    HardwareError,
    NetworkError,
    ResourceError,
)


class RecoveryManager:
    """Zentraler Manager für Error-Recovery-Strategien."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies = {}
        self.fallback_handlers = {}
        self.circuit_breakers = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Registriere Standard-Recovery-Strategien."""
        # File System Recovery
        self.register_strategy(
            FileSystemError,
            self._recover_file_system_error,
            max_attempts=3,
            backoff_factor=2.0,
        )

        # Database Recovery
        self.register_strategy(
            DatabaseError,
            self._recover_database_error,
            max_attempts=5,
            backoff_factor=1.5,
        )

        # Resource Recovery
        self.register_strategy(
            ResourceError,
            self._recover_resource_error,
            max_attempts=2,
            backoff_factor=3.0,
        )

        # Network Recovery
        self.register_strategy(
            NetworkError,
            self._recover_network_error,
            max_attempts=3,
            backoff_factor=2.0,
        )

        # Hardware Recovery
        self.register_strategy(
            HardwareError,
            self._recover_hardware_error,
            max_attempts=1,
            backoff_factor=1.0,
        )

    def register_strategy(
        self,
        exception_type: type,
        recovery_func: Callable,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
    ):
        """Registriere Recovery-Strategie für Exception-Typ."""
        self.recovery_strategies[exception_type] = {
            "function": recovery_func,
            "max_attempts": max_attempts,
            "backoff_factor": backoff_factor,
            "attempts": 0,
            "last_attempt": 0,
        }

    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Registriere Fallback-Handler für Operation."""
        self.fallback_handlers[operation_name] = fallback_func

    def recover(self, exception: Exception, operation: Optional[str] = None) -> Any:
        """Versuche Recovery für Exception."""
        exception_type = type(exception)

        # Prüfe ob Recovery-Strategie existiert
        if exception_type not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for {exception_type.__name__}")
            return self._try_fallback(operation)

        strategy = self.recovery_strategies[exception_type]

        # Prüfe Circuit Breaker
        if self._is_circuit_open(exception_type):
            self.logger.warning(f"Circuit breaker open for {exception_type.__name__}")
            return self._try_fallback(operation)

        # Prüfe Max Attempts
        if strategy["attempts"] >= strategy["max_attempts"]:
            self.logger.error(
                f"Max recovery attempts exceeded for {exception_type.__name__}"
            )
            self._open_circuit(exception_type)
            return self._try_fallback(operation)

        # Führe Recovery aus
        try:
            strategy["attempts"] += 1
            strategy["last_attempt"] = time.time()

            self.logger.info(
                f"Attempting recovery for {exception_type.__name__} (attempt {strategy['attempts']})"
            )
            result = strategy["function"](exception, operation)

            # Reset bei Erfolg
            strategy["attempts"] = 0
            self._close_circuit(exception_type)

            return result

        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return self._try_fallback(operation)

    def _try_fallback(self, operation: Optional[str]) -> Any:
        """Versuche Fallback-Handler."""
        if operation and operation in self.fallback_handlers:
            try:
                self.logger.info(f"Using fallback for operation: {operation}")
                return self.fallback_handlers[operation]()
            except Exception as e:
                self.logger.error(f"Fallback failed: {e}")

        return None

    def _is_circuit_open(self, exception_type: type) -> bool:
        """Prüfe ob Circuit Breaker offen ist."""
        if exception_type not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[exception_type]
        if breaker["state"] == "open":
            # Prüfe ob Timeout abgelaufen ist
            if time.time() - breaker["last_failure"] > breaker["timeout"]:
                breaker["state"] = "half-open"
                return False
            return True

        return False

    def _open_circuit(self, exception_type: type, timeout: float = 300.0):
        """Öffne Circuit Breaker."""
        self.circuit_breakers[exception_type] = {
            "state": "open",
            "last_failure": time.time(),
            "timeout": timeout,
        }

    def _close_circuit(self, exception_type: type):
        """Schließe Circuit Breaker."""
        if exception_type in self.circuit_breakers:
            self.circuit_breakers[exception_type]["state"] = "closed"

    # Recovery-Strategien
    def _recover_file_system_error(
        self, exception: FileSystemError, operation: Optional[str]
    ) -> Any:
        """Recovery für FileSystem-Fehler."""
        if "permission" in str(exception).lower():
            # Versuche Berechtigungen zu reparieren
            return self._fix_file_permissions(exception)
        elif "no space" in str(exception).lower():
            # Versuche Speicherplatz freizugeben
            return self._free_disk_space()
        elif "not found" in str(exception).lower():
            # Versuche Verzeichnisse zu erstellen
            return self._create_missing_directories(exception)
        else:
            # Generische File-System-Recovery
            return self._generic_file_recovery(exception)

    def _recover_database_error(
        self, exception: DatabaseError, operation: Optional[str]
    ) -> Any:
        """Recovery für Database-Fehler."""
        if "locked" in str(exception).lower():
            # Warte auf Datenbank-Freigabe
            time.sleep(1.0)
            return True
        elif "corrupt" in str(exception).lower():
            # Versuche Datenbank zu reparieren
            return self._repair_database(exception)
        elif "connection" in str(exception).lower():
            # Versuche Verbindung wiederherzustellen
            return self._reconnect_database(exception)
        else:
            # Generische Database-Recovery
            return self._generic_database_recovery(exception)

    def _recover_resource_error(
        self, exception: ResourceError, operation: Optional[str]
    ) -> Any:
        """Recovery für Resource-Fehler."""
        if "memory" in str(exception).lower():
            # Versuche Speicher freizugeben
            return self._free_memory()
        elif "cpu" in str(exception).lower():
            # Reduziere CPU-Intensität
            return self._reduce_cpu_load()
        else:
            # Generische Resource-Recovery
            return self._generic_resource_recovery(exception)

    def _recover_network_error(
        self, exception: NetworkError, operation: Optional[str]
    ) -> Any:
        """Recovery für Network-Fehler."""
        # Warte und versuche erneut
        time.sleep(2.0)
        return True

    def _recover_hardware_error(
        self, exception: HardwareError, operation: Optional[str]
    ) -> Any:
        """Recovery für Hardware-Fehler."""
        # Hardware-Fehler sind meist nicht recoverable
        self.logger.error(f"Hardware error not recoverable: {exception}")
        return False

    # Spezifische Recovery-Funktionen
    def _fix_file_permissions(self, exception: FileSystemError) -> bool:
        """Repariere Dateiberechtigungen."""
        try:
            # Versuche chmod auf relevante Dateien
            if hasattr(exception, "details") and "file_path" in exception.details:
                file_path = Path(exception.details["file_path"])
                if file_path.exists():
                    file_path.chmod(0o644)
                    return True
        except Exception as e:
            self.logger.error(f"Failed to fix permissions: {e}")
        return False

    def _free_disk_space(self) -> bool:
        """Gebe Speicherplatz frei."""
        try:
            # Lösche temporäre Dateien
            temp_dir = Path(tempfile.gettempdir())
            for temp_file in temp_dir.glob("wlan_tool_*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception:
                    pass

            # Lösche alte Log-Dateien
            log_dir = Path("logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log.*"):
                    try:
                        if (
                            log_file.stat().st_mtime < time.time() - 86400
                        ):  # Älter als 1 Tag
                            log_file.unlink()
                    except Exception:
                        pass

            return True
        except Exception as e:
            self.logger.error(f"Failed to free disk space: {e}")
            return False

    def _create_missing_directories(self, exception: FileSystemError) -> bool:
        """Erstelle fehlende Verzeichnisse."""
        try:
            if hasattr(exception, "details") and "directory" in exception.details:
                directory = Path(exception.details["directory"])
                directory.mkdir(parents=True, exist_ok=True)
                return True
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
        return False

    def _generic_file_recovery(self, exception: FileSystemError) -> bool:
        """Generische File-System-Recovery."""
        # Warte kurz und versuche erneut
        time.sleep(0.5)
        return True

    def _repair_database(self, exception: DatabaseError) -> bool:
        """Repariere Datenbank."""
        try:
            if hasattr(exception, "details") and "db_path" in exception.details:
                db_path = exception.details["db_path"]
                # Erstelle Backup
                backup_path = f"{db_path}.backup"
                shutil.copy2(db_path, backup_path)
                # Versuche VACUUM
                import sqlite3

                with sqlite3.connect(db_path) as conn:
                    conn.execute("VACUUM")
                return True
        except Exception as e:
            self.logger.error(f"Failed to repair database: {e}")
        return False

    def _reconnect_database(self, exception: DatabaseError) -> bool:
        """Stelle Datenbankverbindung wieder her."""
        # Warte und versuche erneut
        time.sleep(1.0)
        return True

    def _generic_database_recovery(self, exception: DatabaseError) -> bool:
        """Generische Database-Recovery."""
        time.sleep(0.5)
        return True

    def _free_memory(self) -> bool:
        """Gebe Speicher frei."""
        try:
            import gc

            gc.collect()

            # Prüfe verfügbaren Speicher
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to free memory: {e}")
            return False

    def _reduce_cpu_load(self) -> bool:
        """Reduziere CPU-Last."""
        try:
            # Warte um CPU zu entlasten
            time.sleep(2.0)
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce CPU load: {e}")
            return False

    def _generic_resource_recovery(self, exception: ResourceError) -> bool:
        """Generische Resource-Recovery."""
        time.sleep(1.0)
        return True


# Globaler Recovery Manager
recovery_manager = RecoveryManager()


def with_recovery(
    operation_name: Optional[str] = None, fallback_func: Optional[Callable] = None
):
    """Decorator für automatische Error-Recovery."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Registriere Fallback falls vorhanden
                if fallback_func:
                    recovery_manager.register_fallback(
                        operation_name or func.__name__, fallback_func
                    )

                # Versuche Recovery
                result = recovery_manager.recover(e, operation_name or func.__name__)
                if result is not None:
                    return result

                # Falls Recovery fehlschlägt, re-raise Exception
                raise

        return wrapper

    return decorator


@contextmanager
def recovery_context(operation_name: str, fallback_func: Optional[Callable] = None):
    """Context Manager für Error-Recovery."""
    try:
        yield
    except Exception as e:
        # Registriere Fallback falls vorhanden
        if fallback_func:
            recovery_manager.register_fallback(operation_name, fallback_func)

        # Versuche Recovery
        result = recovery_manager.recover(e, operation_name)
        if result is not None:
            return result

        # Falls Recovery fehlschlägt, re-raise Exception
        raise


def get_recovery_status() -> Dict[str, Any]:
    """Hole Recovery-Status."""
    return {
        "strategies": {
            exc_type.__name__: {
                "attempts": strategy["attempts"],
                "max_attempts": strategy["max_attempts"],
                "last_attempt": strategy["last_attempt"],
            }
            for exc_type, strategy in recovery_manager.recovery_strategies.items()
        },
        "circuit_breakers": {
            exc_type.__name__: breaker
            for exc_type, breaker in recovery_manager.circuit_breakers.items()
        },
        "fallback_handlers": list(recovery_manager.fallback_handlers.keys()),
    }


def reset_recovery_state():
    """Setze Recovery-Status zurück."""
    for strategy in recovery_manager.recovery_strategies.values():
        strategy["attempts"] = 0
        strategy["last_attempt"] = 0

    recovery_manager.circuit_breakers.clear()
    recovery_manager.fallback_handlers.clear()
