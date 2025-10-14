#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Collection System für das WLAN-Analyse-Tool.
"""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import psutil

from .constants import Constants, ErrorCodes, get_error_message
from .exceptions import ResourceError, ValidationError

T = TypeVar("T")


@dataclass
class MetricPoint:
    """Einzelner Metrik-Punkt."""

    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Serie von Metrik-Punkten."""

    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    metric_type: str = "gauge"  # gauge, counter, histogram
    description: str = ""

    def add_point(
        self, value: Union[int, float], labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Füge Metrik-Punkt hinzu."""
        point = MetricPoint(timestamp=time.time(), value=value, labels=labels or {})
        self.points.append(point)

    def get_latest(self) -> Optional[MetricPoint]:
        """Hole neuesten Punkt."""
        return self.points[-1] if self.points else None

    def get_average(self, duration_seconds: Optional[float] = None) -> Optional[float]:
        """Berechne Durchschnitt über Zeitraum."""
        if not self.points:
            return None

        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        else:
            recent_points = list(self.points)

        if not recent_points:
            return None

        return sum(p.value for p in recent_points) / len(recent_points)

    def get_count(self, duration_seconds: Optional[float] = None) -> int:
        """Hole Anzahl Punkte im Zeitraum."""
        if not self.points:
            return 0

        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            return sum(1 for p in self.points if p.timestamp >= cutoff_time)
        else:
            return len(self.points)


class MetricsCollector:
    """Zentrale Metrik-Sammlung."""

    def __init__(self):
        self._metrics: Dict[str, MetricSeries] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()
        self._system_metrics_enabled = True
        self._cleanup_interval = 300  # 5 Minuten
        self._cleanup_task: Optional[threading.Timer] = None
        self._start_cleanup()

    def _start_cleanup(self) -> None:
        """Starte Cleanup-Task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        self._cleanup_task = threading.Timer(
            self._cleanup_interval, self._cleanup_old_metrics
        )
        self._cleanup_task.daemon = True
        self._cleanup_task.start()

    def _cleanup_old_metrics(self) -> None:
        """Entferne alte Metriken."""
        with self._lock:
            cutoff_time = time.time() - 3600  # 1 Stunde
            for series in self._metrics.values():
                # Entferne alte Punkte
                while series.points and series.points[0].timestamp < cutoff_time:
                    series.points.popleft()

        # Starte nächsten Cleanup
        self._start_cleanup()

    def create_metric(
        self, name: str, metric_type: str = "gauge", description: str = ""
    ) -> MetricSeries:
        """Erstelle neue Metrik."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricSeries(
                    name=name, metric_type=metric_type, description=description
                )
            return self._metrics[name]

    def record_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Recorde Gauge-Metrik."""
        series = self.create_metric(name, "gauge")
        series.add_point(value, labels)

    def record_counter(
        self,
        name: str,
        increment: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Recorde Counter-Metrik."""
        series = self.create_metric(name, "counter")
        series.add_point(increment, labels)

    def record_timing(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Recorde Timing-Metrik."""
        series = self.create_metric(name, "histogram")
        series.add_point(duration, labels)

    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Hole Metrik-Serie."""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Hole alle Metriken."""
        with self._lock:
            return self._metrics.copy()

    def get_metric_summary(
        self, name: str, duration_seconds: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Hole Metrik-Zusammenfassung."""
        series = self.get_metric(name)
        if not series:
            return None

        latest = series.get_latest()
        average = series.get_average(duration_seconds)
        count = series.get_count(duration_seconds)

        return {
            "name": name,
            "type": series.metric_type,
            "description": series.description,
            "latest_value": latest.value if latest else None,
            "latest_timestamp": latest.timestamp if latest else None,
            "average": average,
            "count": count,
            "total_points": len(series.points),
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Hole System-Metriken."""
        if not self._system_metrics_enabled:
            return {}

        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": memory_info.rss / 1024 / 1024,
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "uptime_seconds": time.time() - self._start_time,
            }
        except Exception:
            return {}

    def export_metrics(self, format_type: str = "json") -> str:
        """Exportiere Metriken."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "system_metrics": self.get_system_metrics(),
                "custom_metrics": {},
            }

            for name, series in self._metrics.items():
                data["custom_metrics"][name] = {
                    "type": series.metric_type,
                    "description": series.description,
                    "points": [
                        {"timestamp": p.timestamp, "value": p.value, "labels": p.labels}
                        for p in list(series.points)[-100:]  # Letzte 100 Punkte
                    ],
                }

            if format_type == "json":
                return json.dumps(data, indent=2)
            elif format_type == "prometheus":
                return self._export_prometheus_format(data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Exportiere im Prometheus-Format."""
        lines = []

        # System-Metriken
        system_metrics = data.get("system_metrics", {})
        for name, value in system_metrics.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Custom-Metriken
        for name, metric_data in data.get("custom_metrics", {}).items():
            metric_type = metric_data["type"]
            lines.append(f"# TYPE {name} {metric_type}")

            for point in metric_data["points"]:
                labels_str = ""
                if point["labels"]:
                    labels_str = (
                        "{"
                        + ",".join(f'{k}="{v}"' for k, v in point["labels"].items())
                        + "}"
                    )

                lines.append(
                    f"{name}{labels_str} {point['value']} {int(point['timestamp'] * 1000)}"
                )

        return "\n".join(lines)

    def save_metrics(self, file_path: Union[str, Path]) -> None:
        """Speichere Metriken in Datei."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.export_metrics("json"))

    def clear_metrics(self) -> None:
        """Leere alle Metriken."""
        with self._lock:
            self._metrics.clear()


# Global Metrics Collector
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Hole globalen Metrics Collector."""
    return _global_metrics


# Decorator für automatische Metrik-Sammlung
def record_metrics(
    operation_name: str,
    metric_type: str = "timing",
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[T], T]:
    """Decorator für automatische Metrik-Sammlung."""

    def decorator(func: T) -> T:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Recorde Erfolg
                duration = time.time() - start_time
                if metric_type == "timing":
                    _global_metrics.record_timing(operation_name, duration, labels)
                elif metric_type == "counter":
                    _global_metrics.record_counter(operation_name, 1, labels)

                return result

            except Exception as e:
                # Recorde Fehler
                duration = time.time() - start_time
                error_labels = (labels or {}).copy()
                error_labels["error"] = type(e).__name__

                if metric_type == "timing":
                    _global_metrics.record_timing(
                        f"{operation_name}_error", duration, error_labels
                    )
                elif metric_type == "counter":
                    _global_metrics.record_counter(
                        f"{operation_name}_error", 1, error_labels
                    )

                raise

        return wrapper  # type: ignore

    return decorator


# Spezielle Metrik-Collectors
class DatabaseMetrics:
    """Metriken für Datenbankoperationen."""

    @staticmethod
    def record_query(query_name: str, duration: float, success: bool = True) -> None:
        """Recorde Datenbank-Query."""
        labels = {"query": query_name, "success": str(success)}
        _global_metrics.record_timing("database_query_duration", duration, labels)
        _global_metrics.record_counter("database_query_count", 1, labels)

    @staticmethod
    def record_connection_pool(size: int, active: int) -> None:
        """Recorde Connection Pool-Status."""
        _global_metrics.record_gauge("database_connection_pool_size", size)
        _global_metrics.record_gauge("database_connection_pool_active", active)

    @staticmethod
    def record_transaction(duration: float, success: bool = True) -> None:
        """Recorde Transaktion."""
        labels = {"success": str(success)}
        _global_metrics.record_timing("database_transaction_duration", duration, labels)
        _global_metrics.record_counter("database_transaction_count", 1, labels)


class AnalysisMetrics:
    """Metriken für Analyse-Operationen."""

    @staticmethod
    def record_clustering(
        algorithm: str, duration: float, num_clusters: int, num_points: int
    ) -> None:
        """Recorde Clustering-Operation."""
        labels = {
            "algorithm": algorithm,
            "num_clusters": str(num_clusters),
            "num_points": str(num_points),
        }
        _global_metrics.record_timing("analysis_clustering_duration", duration, labels)
        _global_metrics.record_counter("analysis_clustering_count", 1, labels)

    @staticmethod
    def record_feature_extraction(duration: float, num_features: int) -> None:
        """Recorde Feature-Extraktion."""
        labels = {"num_features": str(num_features)}
        _global_metrics.record_timing(
            "analysis_feature_extraction_duration", duration, labels
        )
        _global_metrics.record_counter("analysis_feature_extraction_count", 1, labels)

    @staticmethod
    def record_inference(duration: float, num_results: int) -> None:
        """Recorde Inferenz-Operation."""
        labels = {"num_results": str(num_results)}
        _global_metrics.record_timing("analysis_inference_duration", duration, labels)
        _global_metrics.record_counter("analysis_inference_count", 1, labels)


class CaptureMetrics:
    """Metriken für Capture-Operationen."""

    @staticmethod
    def record_packet_processed(packet_type: str) -> None:
        """Recorde verarbeitetes Paket."""
        labels = {"packet_type": packet_type}
        _global_metrics.record_counter("capture_packets_processed", 1, labels)

    @staticmethod
    def record_packet_parse_error(error_type: str) -> None:
        """Recorde Paket-Parse-Fehler."""
        labels = {"error_type": error_type}
        _global_metrics.record_counter("capture_packet_parse_errors", 1, labels)

    @staticmethod
    def record_interface_status(interface: str, active: bool) -> None:
        """Recorde Interface-Status."""
        labels = {"interface": interface, "active": str(active)}
        _global_metrics.record_gauge(
            "capture_interface_active", 1 if active else 0, labels
        )


# Async Metrics Support
class AsyncMetricsCollector:
    """Async Metrics Collector."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    async def record_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Async Gauge-Recording."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.collector.record_gauge, name, value, labels
        )

    async def record_counter(
        self,
        name: str,
        increment: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Async Counter-Recording."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.collector.record_counter, name, increment, labels
        )

    async def record_timing(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Async Timing-Recording."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.collector.record_timing, name, duration, labels
        )

    async def get_metric_summary(
        self, name: str, duration_seconds: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Async Metric-Summary."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.collector.get_metric_summary, name, duration_seconds
        )

    async def export_metrics(self, format_type: str = "json") -> str:
        """Async Metrics-Export."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.collector.export_metrics, format_type
        )


# Convenience-Funktionen
def record_timing(
    operation_name: str, duration: float, labels: Optional[Dict[str, str]] = None
) -> None:
    """Recorde Timing-Metrik."""
    _global_metrics.record_timing(operation_name, duration, labels)


def record_counter(
    operation_name: str,
    increment: Union[int, float] = 1,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Recorde Counter-Metrik."""
    _global_metrics.record_counter(operation_name, increment, labels)


def record_gauge(
    operation_name: str,
    value: Union[int, float],
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Recorde Gauge-Metrik."""
    _global_metrics.record_gauge(operation_name, value, labels)


def get_metric_summary(
    operation_name: str, duration_seconds: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Hole Metrik-Zusammenfassung."""
    return _global_metrics.get_metric_summary(operation_name, duration_seconds)
