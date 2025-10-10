#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrales Logging-System für das WLAN-Analyse-Tool.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class WLANToolFormatter(logging.Formatter):
    """Custom Formatter für WLAN-Tool-Logs."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        super().__init__()
    
    def format(self, record):
        # Basis-Format
        if self.include_context:
            format_str = (
                "%(asctime)s | %(levelname)-8s | %(name)-20s | "
                "%(funcName)-15s:%(lineno)-4d | %(message)s"
            )
        else:
            format_str = "%(asctime)s | %(levelname)-8s | %(message)s"
        
        formatter = logging.Formatter(format_str)
        formatted = formatter.format(record)
        
        # Füge Kontext-Informationen hinzu
        if self.include_context and hasattr(record, 'context'):
            context_str = json.dumps(record.context, indent=2)
            formatted += f"\nContext: {context_str}"
        
        return formatted


class ErrorTrackingHandler(logging.Handler):
    """Handler für Error-Tracking und -Aggregation."""
    
    def __init__(self):
        super().__init__()
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            error_key = f"{record.name}:{record.funcName}:{record.msg}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Speichere recent errors
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'logger': record.name,
                'function': record.funcName,
                'message': record.msg,
                'level': record.levelname,
                'pathname': record.pathname,
                'lineno': record.lineno
            }
            
            self.recent_errors.append(error_info)
            if len(self.recent_errors) > self.max_recent_errors:
                self.recent_errors.pop(0)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Erstelle Error-Summary."""
        return {
            'total_unique_errors': len(self.error_counts),
            'error_counts': dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)),
            'recent_errors': self.recent_errors[-10:],  # Letzte 10 Fehler
            'total_errors': sum(self.error_counts.values())
        }


class PerformanceHandler(logging.Handler):
    """Handler für Performance-Monitoring."""
    
    def __init__(self):
        super().__init__()
        self.performance_metrics = []
        self.max_metrics = 1000
    
    def emit(self, record):
        if hasattr(record, 'performance_data'):
            metric = {
                'timestamp': datetime.now().isoformat(),
                'function': record.funcName,
                'duration': record.performance_data.get('duration'),
                'memory_usage': record.performance_data.get('memory_usage'),
                'cpu_usage': record.performance_data.get('cpu_usage'),
                'operation': record.performance_data.get('operation')
            }
            
            self.performance_metrics.append(metric)
            if len(self.performance_metrics) > self.max_metrics:
                self.performance_metrics.pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Erstelle Performance-Summary."""
        if not self.performance_metrics:
            return {'message': 'No performance data available'}
        
        durations = [m['duration'] for m in self.performance_metrics if m['duration']]
        memory_usage = [m['memory_usage'] for m in self.performance_metrics if m['memory_usage']]
        
        return {
            'total_operations': len(self.performance_metrics),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_usage': max(memory_usage) if memory_usage else 0
        }


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_performance_logging: bool = False,
    enable_error_tracking: bool = True
) -> logging.Logger:
    """
    Richte das Logging-System ein.
    
    Args:
        log_level: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Pfad zur Log-Datei (optional)
        max_file_size: Maximale Größe einer Log-Datei in Bytes
        backup_count: Anzahl der Backup-Dateien
        enable_console: Ob Console-Logging aktiviert werden soll
        enable_performance_logging: Ob Performance-Logging aktiviert werden soll
        enable_error_tracking: Ob Error-Tracking aktiviert werden soll
    
    Returns:
        Konfigurierter Logger
    """
    # Root Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Entferne existierende Handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console Handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = WLANToolFormatter(include_context=False)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File logs immer DEBUG
        file_formatter = WLANToolFormatter(include_context=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Error Tracking Handler
    if enable_error_tracking:
        error_handler = ErrorTrackingHandler()
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    # Performance Handler
    if enable_performance_logging:
        perf_handler = PerformanceHandler()
        perf_handler.setLevel(logging.INFO)
        root_logger.addHandler(perf_handler)
    
    # WLAN-Tool Logger
    wlan_logger = logging.getLogger("wlan_tool")
    wlan_logger.setLevel(getattr(logging, log_level.upper()))
    
    return wlan_logger


def get_logger(name: str) -> logging.Logger:
    """Hole Logger für spezifisches Modul."""
    return logging.getLogger(f"wlan_tool.{name}")


def log_performance(logger: logging.Logger, operation: str, duration: float, 
                   memory_usage: Optional[float] = None, cpu_usage: Optional[float] = None):
    """Logge Performance-Metriken."""
    performance_data = {
        'operation': operation,
        'duration': duration,
        'memory_usage': memory_usage,
        'cpu_usage': cpu_usage
    }
    
    logger.info(f"Performance: {operation} took {duration:.3f}s", 
                extra={'performance_data': performance_data})


def log_with_context(logger: logging.Logger, level: int, message: str, 
                    context: Dict[str, Any], **kwargs):
    """Logge mit Kontext-Informationen."""
    logger.log(level, message, extra={'context': context}, **kwargs)


def get_error_summary() -> Dict[str, Any]:
    """Hole Error-Summary von allen Error-Tracking-Handlern."""
    summaries = []
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, ErrorTrackingHandler):
            summaries.append(handler.get_error_summary())
    
    if not summaries:
        return {'message': 'No error tracking handlers found'}
    
    # Kombiniere alle Summaries
    combined = {
        'total_unique_errors': sum(s['total_unique_errors'] for s in summaries),
        'total_errors': sum(s['total_errors'] for s in summaries),
        'recent_errors': []
    }
    
    for summary in summaries:
        combined['recent_errors'].extend(summary['recent_errors'])
    
    # Sortiere nach Timestamp
    combined['recent_errors'].sort(key=lambda x: x['timestamp'], reverse=True)
    combined['recent_errors'] = combined['recent_errors'][:20]  # Top 20
    
    return combined


def get_performance_summary() -> Dict[str, Any]:
    """Hole Performance-Summary von allen Performance-Handlern."""
    summaries = []
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, PerformanceHandler):
            summaries.append(handler.get_performance_summary())
    
    if not summaries:
        return {'message': 'No performance tracking handlers found'}
    
    # Kombiniere alle Summaries
    combined = {
        'total_operations': sum(s.get('total_operations', 0) for s in summaries),
        'avg_duration': 0,
        'max_duration': 0,
        'min_duration': 0,
        'avg_memory_usage': 0,
        'max_memory_usage': 0
    }
    
    # Berechne Durchschnittswerte
    valid_summaries = [s for s in summaries if 'total_operations' in s and s['total_operations'] > 0]
    if valid_summaries:
        combined['avg_duration'] = sum(s.get('avg_duration', 0) for s in valid_summaries) / len(valid_summaries)
        combined['max_duration'] = max(s.get('max_duration', 0) for s in valid_summaries)
        combined['min_duration'] = min(s.get('min_duration', 0) for s in valid_summaries)
        combined['avg_memory_usage'] = sum(s.get('avg_memory_usage', 0) for s in valid_summaries) / len(valid_summaries)
        combined['max_memory_usage'] = max(s.get('max_memory_usage', 0) for s in valid_summaries)
    
    return combined


def create_logging_config() -> Dict[str, Any]:
    """Erstelle Standard-Logging-Konfiguration."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s\nContext: %(context)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'wlan_tool.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'wlan_tool': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }


# Performance-Decorator
def log_performance_metrics(operation_name: Optional[str] = None):
    """Decorator für automatisches Performance-Logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import psutil
            import os
            
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.Process(os.getpid()).memory_info().rss
                
                duration = end_time - start_time
                memory_usage = end_memory - start_memory
                
                logger = get_logger(func.__module__)
                log_performance(
                    logger, 
                    operation_name or func.__name__,
                    duration,
                    memory_usage
                )
        
        return wrapper
    return decorator