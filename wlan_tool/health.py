#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Check System für das WLAN-Analyse-Tool.
"""

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from .constants import Constants, ErrorCodes, get_error_message
from .exceptions import HardwareError, ResourceError, ValidationError
from .metrics import get_metrics


class HealthStatus(Enum):
    """Health Check Status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Ergebnis eines Health Checks."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Prüfe ob Check gesund ist."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Prüfe ob Check degradiert ist."""
        return self.status == HealthStatus.DEGRADED

    @property
    def is_unhealthy(self) -> bool:
        """Prüfe ob Check ungesund ist."""
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class HealthReport:
    """Gesundheitsbericht des Systems."""

    overall_status: HealthStatus
    checks: List[HealthCheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0

    @property
    def healthy_checks(self) -> List[HealthCheckResult]:
        """Hole gesunde Checks."""
        return [check for check in self.checks if check.is_healthy]

    @property
    def degraded_checks(self) -> List[HealthCheckResult]:
        """Hole degradierte Checks."""
        return [check for check in self.checks if check.is_degraded]

    @property
    def unhealthy_checks(self) -> List[HealthCheckResult]:
        """Hole ungesunde Checks."""
        return [check for check in self.checks if check.is_unhealthy]

    @property
    def summary(self) -> Dict[str, int]:
        """Zusammenfassung der Check-Status."""
        return {
            "total": len(self.checks),
            "healthy": len(self.healthy_checks),
            "degraded": len(self.degraded_checks),
            "unhealthy": len(self.unhealthy_checks),
        }


class HealthChecker:
    """Basis-Klasse für Health Checks."""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Führe Health Check aus."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(self._check_impl(), timeout=self.timeout)
            duration = time.time() - start_time
            result.duration = duration
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                details={"timeout": self.timeout},
                duration=time.time() - start_time,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
                duration=time.time() - start_time,
            )

    async def _check_impl(self) -> HealthCheckResult:
        """Implementierung des Health Checks (zu überschreiben)."""
        raise NotImplementedError


class SystemResourceChecker(HealthChecker):
    """Health Check für System-Ressourcen."""

    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 90.0):
        super().__init__("system_resources")
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent

    async def _check_impl(self) -> HealthCheckResult:
        """Prüfe System-Ressourcen."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }

            # Bestimme Status basierend auf Schwellenwerten
            if (
                cpu_percent > self.max_cpu_percent
                or memory.percent > self.max_memory_percent
            ):
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            elif (
                cpu_percent > self.max_cpu_percent * 0.8
                or memory.percent > self.max_memory_percent * 0.8
            ):
                status = HealthStatus.DEGRADED
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resources normal: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"

            return HealthCheckResult(
                name=self.name, status=status, message=message, details=details
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
            )


class DatabaseHealthChecker(HealthChecker):
    """Health Check für Datenbank."""

    def __init__(self, db_path: str):
        super().__init__("database")
        self.db_path = db_path

    async def _check_impl(self) -> HealthCheckResult:
        """Prüfe Datenbank-Verbindung."""
        try:
            import sqlite3

            # Teste Verbindung
            conn = sqlite3.connect(self.db_path, timeout=5)
            cursor = conn.cursor()

            # Führe einfache Query aus
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            # Prüfe Tabellen-Existenz
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            conn.close()

            if result and result[0] == 1:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    details={"tables_count": len(tables), "db_path": self.db_path},
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database query failed",
                    details={"db_path": self.db_path},
                )

        except sqlite3.OperationalError as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database operational error: {e}",
                details={"error": str(e), "db_path": self.db_path},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {e}",
                details={"error": str(e), "db_path": self.db_path},
            )


class FileSystemHealthChecker(HealthChecker):
    """Health Check für Dateisystem."""

    def __init__(self, required_paths: List[str], min_free_space_gb: float = 1.0):
        super().__init__("filesystem")
        self.required_paths = required_paths
        self.min_free_space_gb = min_free_space_gb

    async def _check_impl(self) -> HealthCheckResult:
        """Prüfe Dateisystem."""
        try:
            details = {}
            issues = []

            # Prüfe erforderliche Pfade
            for path_str in self.required_paths:
                path = Path(path_str)
                if path.exists():
                    details[f"path_{path.name}_exists"] = True
                else:
                    details[f"path_{path.name}_exists"] = False
                    issues.append(f"Required path missing: {path}")

            # Prüfe freien Speicherplatz
            disk_usage = psutil.disk_usage("/")
            free_space_gb = disk_usage.free / (1024**3)
            details["free_space_gb"] = free_space_gb

            if free_space_gb < self.min_free_space_gb:
                issues.append(
                    f"Low disk space: {free_space_gb:.2f}GB (min: {self.min_free_space_gb}GB)"
                )

            # Prüfe Schreibberechtigungen
            try:
                test_file = Path("/tmp/wlan_tool_health_test")
                test_file.write_text("test")
                test_file.unlink()
                details["write_permissions"] = True
            except Exception:
                details["write_permissions"] = False
                issues.append("No write permissions")

            if issues:
                status = HealthStatus.UNHEALTHY
                message = f"File system issues: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "File system healthy"

            return HealthCheckResult(
                name=self.name, status=status, message=message, details=details
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {e}",
                details={"error": str(e)},
            )


class NetworkHealthChecker(HealthChecker):
    """Health Check für Netzwerk."""

    def __init__(self, test_urls: List[str] = None):
        super().__init__("network")
        self.test_urls = test_urls or [
            "https://www.google.com",
            "https://www.cloudflare.com",
        ]

    async def _check_impl(self) -> HealthCheckResult:
        """Prüfe Netzwerk-Verbindung."""
        try:
            import aiohttp

            details = {}
            successful_connections = 0

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                for url in self.test_urls:
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                successful_connections += 1
                                details[f"url_{url}_status"] = "success"
                            else:
                                details[f"url_{url}_status"] = f"http_{response.status}"
                    except Exception as e:
                        details[f"url_{url}_status"] = f"error: {str(e)}"

            success_rate = successful_connections / len(self.test_urls)

            if success_rate >= 0.8:
                status = HealthStatus.HEALTHY
                message = f"Network healthy ({successful_connections}/{len(self.test_urls)} connections successful)"
            elif success_rate >= 0.5:
                status = HealthStatus.DEGRADED
                message = f"Network degraded ({successful_connections}/{len(self.test_urls)} connections successful)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Network unhealthy ({successful_connections}/{len(self.test_urls)} connections successful)"

            details["success_rate"] = success_rate
            details["successful_connections"] = successful_connections
            details["total_connections"] = len(self.test_urls)

            return HealthCheckResult(
                name=self.name, status=status, message=message, details=details
            )

        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="Network check skipped (aiohttp not available)",
                details={"skipped": True},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Network check failed: {e}",
                details={"error": str(e)},
            )


class HardwareHealthChecker(HealthChecker):
    """Health Check für Hardware (WLAN-Interfaces)."""

    def __init__(self, required_interfaces: List[str] = None):
        super().__init__("hardware")
        self.required_interfaces = required_interfaces or ["wlan0", "wlan0mon"]

    async def _check_impl(self) -> HealthCheckResult:
        """Prüfe Hardware-Verfügbarkeit."""
        try:
            import subprocess

            details = {}
            available_interfaces = []

            # Liste verfügbare Netzwerk-Interfaces
            try:
                result = subprocess.run(
                    ["iw", "dev"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Interface" in line:
                            interface = line.split()[-1]
                            available_interfaces.append(interface)
                else:
                    # Fallback: verwende /proc/net/dev
                    with open("/proc/net/dev", "r") as f:
                        for line in f:
                            if ":" in line and not line.startswith("Inter-"):
                                interface = line.split(":")[0].strip()
                                if interface.startswith("wlan"):
                                    available_interfaces.append(interface)
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                pass

            # Prüfe erforderliche Interfaces
            missing_interfaces = []
            for interface in self.required_interfaces:
                if interface in available_interfaces:
                    details[f"interface_{interface}"] = "available"
                else:
                    details[f"interface_{interface}"] = "missing"
                    missing_interfaces.append(interface)

            # Prüfe Monitor-Mode-Fähigkeit
            monitor_capable = False
            try:
                result = subprocess.run(
                    ["iw", "list"], capture_output=True, text=True, timeout=5
                )
                if "monitor" in result.stdout.lower():
                    monitor_capable = True
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                pass

            details["monitor_capable"] = monitor_capable
            details["available_interfaces"] = available_interfaces

            if not missing_interfaces and monitor_capable:
                status = HealthStatus.HEALTHY
                message = "Hardware requirements met"
            elif not missing_interfaces:
                status = HealthStatus.DEGRADED
                message = "Interfaces available but monitor mode not supported"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Missing required interfaces: {missing_interfaces}"

            return HealthCheckResult(
                name=self.name, status=status, message=message, details=details
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Hardware check failed: {e}",
                details={"error": str(e)},
            )


class HealthMonitor:
    """Zentraler Health Monitor."""

    def __init__(self):
        self.checkers: List[HealthChecker] = []
        self.start_time = time.time()
        self._lock = threading.RLock()

    def add_checker(self, checker: HealthChecker) -> None:
        """Füge Health Checker hinzu."""
        with self._lock:
            self.checkers.append(checker)

    def remove_checker(self, name: str) -> bool:
        """Entferne Health Checker."""
        with self._lock:
            for i, checker in enumerate(self.checkers):
                if checker.name == name:
                    del self.checkers[i]
                    return True
            return False

    async def run_checks(self) -> HealthReport:
        """Führe alle Health Checks aus."""
        with self._lock:
            checkers = self.checkers.copy()

        # Führe alle Checks parallel aus
        tasks = [checker.check() for checker in checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verarbeite Ergebnisse
        checks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                checks.append(
                    HealthCheckResult(
                        name=checkers[i].name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check execution failed: {result}",
                        details={"error": str(result)},
                    )
                )
            else:
                checks.append(result)

        # Bestimme Gesamtstatus
        overall_status = self._determine_overall_status(checks)

        return HealthReport(
            overall_status=overall_status,
            checks=checks,
            timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
        )

    def _determine_overall_status(
        self, checks: List[HealthCheckResult]
    ) -> HealthStatus:
        """Bestimme Gesamtstatus basierend auf Einzelchecks."""
        if not checks:
            return HealthStatus.UNKNOWN

        unhealthy_count = sum(1 for check in checks if check.is_unhealthy)
        degraded_count = sum(1 for check in checks if check.is_degraded)

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_quick_status(self) -> HealthStatus:
        """Hole schnellen Status ohne detaillierte Checks."""
        # Einfache Heuristik basierend auf System-Metriken
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 90 or memory_percent > 95:
                return HealthStatus.UNHEALTHY
            elif cpu_percent > 80 or memory_percent > 85:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
        except Exception:
            return HealthStatus.UNKNOWN


# Global Health Monitor
_global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Hole globalen Health Monitor."""
    return _global_health_monitor


def setup_default_health_checks(
    db_path: Optional[str] = None, required_paths: Optional[List[str]] = None
) -> None:
    """Richte Standard-Health-Checks ein."""
    monitor = get_health_monitor()

    # System-Ressourcen
    monitor.add_checker(SystemResourceChecker())

    # Dateisystem
    if required_paths:
        monitor.add_checker(FileSystemHealthChecker(required_paths))

    # Datenbank
    if db_path:
        monitor.add_checker(DatabaseHealthChecker(db_path))

    # Netzwerk
    monitor.add_checker(NetworkHealthChecker())

    # Hardware
    monitor.add_checker(HardwareHealthChecker())


async def get_health_status() -> HealthReport:
    """Hole aktuellen Health-Status."""
    monitor = get_health_monitor()
    return await monitor.run_checks()


def get_quick_health_status() -> HealthStatus:
    """Hole schnellen Health-Status."""
    monitor = get_health_monitor()
    return monitor.get_quick_status()


# Health Check Endpoints für Web-Interface
def get_health_endpoint_data() -> Dict[str, Any]:
    """Hole Daten für Health-Check-Endpoint."""
    monitor = get_health_monitor()
    quick_status = monitor.get_quick_status()

    return {
        "status": quick_status.value,
        "timestamp": time.time(),
        "uptime_seconds": time.time() - monitor.start_time,
        "checkers_count": len(monitor.checkers),
    }


async def get_detailed_health_endpoint_data() -> Dict[str, Any]:
    """Hole detaillierte Daten für Health-Check-Endpoint."""
    report = await get_health_status()

    return {
        "overall_status": report.overall_status.value,
        "timestamp": report.timestamp,
        "uptime_seconds": report.uptime_seconds,
        "summary": report.summary,
        "checks": [
            {
                "name": check.name,
                "status": check.status.value,
                "message": check.message,
                "details": check.details,
                "duration": check.duration,
            }
            for check in report.checks
        ],
    }
