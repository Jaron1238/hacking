#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrale Konstanten und Error-Codes für das WLAN-Analyse-Tool.
"""

from enum import Enum
from typing import Dict, Any


class ErrorCodes(Enum):
    """Zentrale Error-Codes für das gesamte System."""
    
    # Database Errors
    DB_CONNECTION_FAILED = "DB_CONNECTION_FAILED"
    DB_QUERY_FAILED = "DB_QUERY_FAILED"
    DB_MIGRATION_FAILED = "DB_MIGRATION_FAILED"
    DB_TRANSACTION_FAILED = "DB_TRANSACTION_FAILED"
    DB_LOCKED = "DB_LOCKED"
    DB_CORRUPT = "DB_CORRUPT"
    DB_PERMISSION_DENIED = "DB_PERMISSION_DENIED"
    
    # File System Errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_PERMISSION_DENIED = "FILE_PERMISSION_DENIED"
    FILE_READ_ERROR = "FILE_READ_ERROR"
    FILE_WRITE_ERROR = "FILE_WRITE_ERROR"
    FILE_CREATE_ERROR = "FILE_CREATE_ERROR"
    FILE_DELETE_ERROR = "FILE_DELETE_ERROR"
    DISK_SPACE_INSUFFICIENT = "DISK_SPACE_INSUFFICIENT"
    
    # Network Errors
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_CONNECTION_FAILED = "NETWORK_CONNECTION_FAILED"
    NETWORK_DNS_FAILED = "NETWORK_DNS_FAILED"
    NETWORK_SSL_ERROR = "NETWORK_SSL_ERROR"
    NETWORK_RATE_LIMITED = "NETWORK_RATE_LIMITED"
    
    # Hardware Errors
    HARDWARE_NOT_FOUND = "HARDWARE_NOT_FOUND"
    HARDWARE_PERMISSION_DENIED = "HARDWARE_PERMISSION_DENIED"
    HARDWARE_INITIALIZATION_FAILED = "HARDWARE_INITIALIZATION_FAILED"
    HARDWARE_TIMEOUT = "HARDWARE_TIMEOUT"
    
    # Analysis Errors
    ANALYSIS_INSUFFICIENT_DATA = "ANALYSIS_INSUFFICIENT_DATA"
    ANALYSIS_INVALID_INPUT = "ANALYSIS_INVALID_INPUT"
    ANALYSIS_CLUSTERING_FAILED = "ANALYSIS_CLUSTERING_FAILED"
    ANALYSIS_FEATURE_EXTRACTION_FAILED = "ANALYSIS_FEATURE_EXTRACTION_FAILED"
    ANALYSIS_MODEL_LOAD_FAILED = "ANALYSIS_MODEL_LOAD_FAILED"
    ANALYSIS_PREDICTION_FAILED = "ANALYSIS_PREDICTION_FAILED"
    
    # Capture Errors
    CAPTURE_INTERFACE_NOT_FOUND = "CAPTURE_INTERFACE_NOT_FOUND"
    CAPTURE_PERMISSION_DENIED = "CAPTURE_PERMISSION_DENIED"
    CAPTURE_PACKET_PARSE_ERROR = "CAPTURE_PACKET_PARSE_ERROR"
    CAPTURE_CHANNEL_HOP_FAILED = "CAPTURE_CHANNEL_HOP_FAILED"
    CAPTURE_MONITOR_MODE_FAILED = "CAPTURE_MONITOR_MODE_FAILED"
    
    # Validation Errors
    VALIDATION_INVALID_MAC = "VALIDATION_INVALID_MAC"
    VALIDATION_INVALID_IP = "VALIDATION_INVALID_IP"
    VALIDATION_INVALID_PATH = "VALIDATION_INVALID_PATH"
    VALIDATION_INVALID_TIMESTAMP = "VALIDATION_INVALID_TIMESTAMP"
    VALIDATION_INVALID_PARAMETER = "VALIDATION_INVALID_PARAMETER"
    VALIDATION_MISSING_REQUIRED_FIELD = "VALIDATION_MISSING_REQUIRED_FIELD"
    VALIDATION_OUT_OF_RANGE = "VALIDATION_OUT_OF_RANGE"
    
    # Configuration Errors
    CONFIG_INVALID_VALUE = "CONFIG_INVALID_VALUE"
    CONFIG_MISSING_KEY = "CONFIG_MISSING_KEY"
    CONFIG_FILE_NOT_FOUND = "CONFIG_FILE_NOT_FOUND"
    CONFIG_PARSE_ERROR = "CONFIG_PARSE_ERROR"
    
    # Resource Errors
    RESOURCE_MEMORY_INSUFFICIENT = "RESOURCE_MEMORY_INSUFFICIENT"
    RESOURCE_CPU_OVERLOAD = "RESOURCE_CPU_OVERLOAD"
    RESOURCE_DISK_FULL = "RESOURCE_DISK_FULL"
    RESOURCE_CONNECTION_POOL_EXHAUSTED = "RESOURCE_CONNECTION_POOL_EXHAUSTED"
    
    # Recovery Errors
    RECOVERY_MAX_ATTEMPTS_EXCEEDED = "RECOVERY_MAX_ATTEMPTS_EXCEEDED"
    RECOVERY_CIRCUIT_BREAKER_OPEN = "RECOVERY_CIRCUIT_BREAKER_OPEN"
    RECOVERY_FALLBACK_FAILED = "RECOVERY_FALLBACK_FAILED"
    
    # Generic Errors
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"
    OPERATION_CANCELLED = "OPERATION_CANCELLED"
    OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"


class Constants:
    """Zentrale Konstanten für das gesamte System."""
    
    # Time Constants (in seconds)
    DEFAULT_TIMEOUT = 30.0
    DB_CONNECTION_TIMEOUT = 30.0
    NETWORK_TIMEOUT = 10.0
    HARDWARE_TIMEOUT = 5.0
    PRUNING_THRESHOLD_S = 7200  # 2 hours
    PRUNING_INTERVAL_S = 3600   # 1 hour
    CACHE_TTL_S = 300           # 5 minutes
    RETRY_DELAY_S = 1.0
    MAX_RETRY_DELAY_S = 60.0
    BACKOFF_FACTOR = 2.0
    
    # Size Constants
    MAX_BATCH_SIZE = 1000
    DEFAULT_BATCH_SIZE = 100
    MAX_FILE_SIZE_MB = 100
    MAX_MEMORY_USAGE_PERCENT = 90
    MAX_CPU_USAGE_PERCENT = 80
    MAX_QUEUE_SIZE = 10000
    MAX_LOG_SIZE_MB = 10
    MAX_LOG_FILES = 5
    
    # Network Constants
    MAX_PACKET_SIZE = 65535
    DEFAULT_CHANNEL = 6
    SUPPORTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165]
    DEFAULT_HOP_INTERVAL_MS = 100
    DEFAULT_DISCOVERY_TIME_S = 60
    
    # Analysis Constants
    MIN_PACKETS_FOR_ANALYSIS = 5
    MIN_CLIENTS_FOR_CLUSTERING = 2
    MAX_CLUSTERS = 20
    DEFAULT_CLUSTERS = 5
    SILHOUETTE_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.7
    
    # Database Constants
    DB_VERSION = 4
    MIGRATION_TIMEOUT_S = 60
    VACUUM_TIMEOUT_S = 300
    WAL_CHECKPOINT_TIMEOUT_S = 30
    
    # Logging Constants
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s"
    DETAILED_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s\nContext: %(context)s"
    
    # Error Recovery Constants
    MAX_RECOVERY_ATTEMPTS = 3
    CIRCUIT_BREAKER_TIMEOUT_S = 300
    FALLBACK_TIMEOUT_S = 10.0
    
    # Performance Constants
    PERFORMANCE_LOG_INTERVAL_S = 60
    MEMORY_CHECK_INTERVAL_S = 30
    CPU_CHECK_INTERVAL_S = 30
    
    # Validation Constants
    MAC_ADDRESS_PATTERN = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    IP_ADDRESS_PATTERN = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    SSID_MAX_LENGTH = 32
    BSSID_LENGTH = 17
    MAC_ADDRESS_LENGTH = 17
    
    # File Extensions
    DATABASE_EXTENSION = '.db'
    LOG_EXTENSION = '.log'
    CONFIG_EXTENSION = '.yaml'
    CSV_EXTENSION = '.csv'
    GEXF_EXTENSION = '.gexf'
    PCAP_EXTENSION = '.pcap'
    
    # Default Paths
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_DATA_DIR = "data"
    DEFAULT_CONFIG_DIR = "config"
    DEFAULT_TEMP_DIR = "temp"
    DEFAULT_BACKUP_DIR = "backups"


class DefaultValues:
    """Standardwerte für Konfigurationen."""
    
    # Database Defaults
    DB_BATCH_SIZE = 100
    DB_FLUSH_INTERVAL_S = 1.0
    DB_CACHE_SIZE = 10000
    DB_JOURNAL_MODE = "WAL"
    DB_SYNCHRONOUS = "NORMAL"
    
    # Capture Defaults
    CAPTURE_DURATION_S = 60
    CAPTURE_INTERFACE = "wlan0mon"
    CAPTURE_CHANNEL_HOP = True
    CAPTURE_LIVE_UI = False
    
    # Analysis Defaults
    ANALYSIS_CLUSTER_ALGORITHM = "kmeans"
    ANALYSIS_FEATURE_WEIGHTS = {
        "vendor": 1.0,
        "standards": 1.0,
        "probe_count": 0.5,
        "seen_with_ap_count": 0.3
    }
    
    # UI Defaults
    UI_DEFAULT_MIN_SCORE = 0.0
    UI_DEFAULT_MAX_SCORE = 1.0
    UI_MAX_CANDIDATES = 100
    UI_AUTO_RETRAIN_MIN_LABELS = 10
    
    # Logging Defaults
    LOG_LEVEL = "INFO"
    LOG_CONSOLE = True
    LOG_FILE = True
    LOG_PERFORMANCE = False
    LOG_ERROR_TRACKING = True


class ErrorMessages:
    """Zentrale Error-Messages für bessere Wartbarkeit."""
    
    # Database Messages
    DB_CONNECTION_FAILED = "Failed to connect to database: {details}"
    DB_QUERY_FAILED = "Database query failed: {query}"
    DB_MIGRATION_FAILED = "Database migration failed: {migration_file}"
    DB_TRANSACTION_FAILED = "Database transaction failed: {operation}"
    
    # File System Messages
    FILE_NOT_FOUND = "File not found: {file_path}"
    FILE_PERMISSION_DENIED = "Permission denied for file: {file_path}"
    FILE_READ_ERROR = "Failed to read file: {file_path}"
    FILE_WRITE_ERROR = "Failed to write file: {file_path}"
    DISK_SPACE_INSUFFICIENT = "Insufficient disk space: {required}MB required, {available}MB available"
    
    # Network Messages
    NETWORK_TIMEOUT = "Network operation timed out after {timeout}s"
    NETWORK_CONNECTION_FAILED = "Failed to establish network connection: {host}:{port}"
    NETWORK_DNS_FAILED = "DNS resolution failed for: {hostname}"
    
    # Hardware Messages
    HARDWARE_NOT_FOUND = "Hardware device not found: {device}"
    HARDWARE_PERMISSION_DENIED = "Permission denied for hardware device: {device}"
    HARDWARE_INITIALIZATION_FAILED = "Failed to initialize hardware: {device}"
    
    # Analysis Messages
    ANALYSIS_INSUFFICIENT_DATA = "Insufficient data for analysis: {required} required, {available} available"
    ANALYSIS_INVALID_INPUT = "Invalid input for analysis: {input_type}"
    ANALYSIS_CLUSTERING_FAILED = "Clustering algorithm failed: {algorithm}"
    
    # Validation Messages
    VALIDATION_INVALID_MAC = "Invalid MAC address format: {mac}"
    VALIDATION_INVALID_IP = "Invalid IP address format: {ip}"
    VALIDATION_INVALID_PATH = "Invalid file path: {path}"
    VALIDATION_MISSING_REQUIRED_FIELD = "Missing required field: {field}"
    VALIDATION_OUT_OF_RANGE = "Value out of range: {value} (min: {min}, max: {max})"
    
    # Resource Messages
    RESOURCE_MEMORY_INSUFFICIENT = "Insufficient memory: {required}MB required, {available}MB available"
    RESOURCE_CPU_OVERLOAD = "CPU overload detected: {usage}% usage"
    RESOURCE_DISK_FULL = "Disk space exhausted: {usage}% used"
    
    # Generic Messages
    UNEXPECTED_ERROR = "Unexpected error occurred: {error_type}: {message}"
    OPERATION_TIMEOUT = "Operation timed out: {operation}"
    OPERATION_CANCELLED = "Operation cancelled: {operation}"


def get_error_message(error_code: ErrorCodes, **kwargs) -> str:
    """Hole formatierte Error-Message für Error-Code."""
    message_template = getattr(ErrorMessages, error_code.value, "Unknown error: {error_code}")
    try:
        return message_template.format(**kwargs)
    except KeyError as e:
        return f"Error formatting message for {error_code.value}: missing {e}"


def get_constant(constant_name: str, default: Any = None) -> Any:
    """Hole Konstante aus Constants-Klasse."""
    return getattr(Constants, constant_name, default)


def validate_constant_value(constant_name: str, value: Any) -> bool:
    """Validiere ob Wert für Konstante gültig ist."""
    constant_value = get_constant(constant_name)
    if constant_value is None:
        return False
    
    # Einfache Typ-Validierung
    if isinstance(constant_value, (int, float)) and isinstance(value, (int, float)):
        return True
    elif isinstance(constant_value, str) and isinstance(value, str):
        return True
    elif isinstance(constant_value, list) and isinstance(value, list):
        return True
    elif isinstance(constant_value, dict) and isinstance(value, dict):
        return True
    
    return False