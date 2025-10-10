#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Umfassendes Input-Validierungs-System für das WLAN-Analyse-Tool.
"""

import re
import socket
import ipaddress
from typing import Any, Optional, Union, List, Dict, Callable, Type, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .constants import ErrorCodes, Constants, ErrorMessages, get_error_message
from .exceptions import ValidationError


class Validator:
    """Basis-Klasse für alle Validatoren."""
    
    @staticmethod
    def validate_not_none(value: Any, field_name: str) -> None:
        """Validiere dass Wert nicht None ist."""
        if value is None:
            raise ValidationError(
                get_error_message(ErrorCodes.VALIDATION_MISSING_REQUIRED_FIELD, field=field_name),
                error_code=ErrorCodes.VALIDATION_MISSING_REQUIRED_FIELD.value,
                details={"field": field_name, "value": value}
            )
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type, field_name: str) -> None:
        """Validiere dass Wert den erwarteten Typ hat."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Invalid type for {field_name}: expected {expected_type.__name__}, got {type(value).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "expected_type": expected_type.__name__, "actual_type": type(value).__name__}
            )
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], field_name: str) -> None:
        """Validiere dass Wert im erwarteten Bereich liegt."""
        if not (min_val <= value <= max_val):
            raise ValidationError(
                get_error_message(ErrorCodes.VALIDATION_OUT_OF_RANGE, value=value, min=min_val, max=max_val),
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"field": field_name, "value": value, "min": min_val, "max": max_val}
            )


class NetworkValidator(Validator):
    """Validatoren für Netzwerk-bezogene Eingaben."""
    
    @staticmethod
    def validate_mac_address(mac: str) -> str:
        """Validiere MAC-Adresse."""
        if not isinstance(mac, str):
            raise ValidationError(
                f"MAC address must be string, got {type(mac).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_MAC.value,
                details={"mac": mac, "type": str(type(mac))}
            )
        
        if len(mac) != Constants.MAC_ADDRESS_LENGTH:
            raise ValidationError(
                f"MAC address must be {Constants.MAC_ADDRESS_LENGTH} characters long, got {len(mac)}",
                error_code=ErrorCodes.VALIDATION_INVALID_MAC.value,
                details={"mac": mac, "expected_length": Constants.MAC_ADDRESS_LENGTH, "actual_length": len(mac)}
            )
        
        if not re.match(Constants.MAC_ADDRESS_PATTERN, mac):
            raise ValidationError(
                get_error_message(ErrorCodes.VALIDATION_INVALID_MAC, mac=mac),
                error_code=ErrorCodes.VALIDATION_INVALID_MAC.value,
                details={"mac": mac, "pattern": Constants.MAC_ADDRESS_PATTERN}
            )
        
        return mac.upper()  # Normalisiere zu Großbuchstaben
    
    @staticmethod
    def validate_ip_address(ip: str) -> str:
        """Validiere IP-Adresse."""
        if not isinstance(ip, str):
            raise ValidationError(
                f"IP address must be string, got {type(ip).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_IP.value,
                details={"ip": ip, "type": str(type(ip))}
            )
        
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError as e:
            raise ValidationError(
                get_error_message(ErrorCodes.VALIDATION_INVALID_IP, ip=ip),
                error_code=ErrorCodes.VALIDATION_INVALID_IP.value,
                details={"ip": ip, "error": str(e)}
            ) from e
    
    @staticmethod
    def validate_ssid(ssid: str) -> str:
        """Validiere SSID."""
        if not isinstance(ssid, str):
            raise ValidationError(
                f"SSID must be string, got {type(ssid).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"ssid": ssid, "type": str(type(ssid))}
            )
        
        if len(ssid) > Constants.SSID_MAX_LENGTH:
            raise ValidationError(
                f"SSID too long: {len(ssid)} characters (max: {Constants.SSID_MAX_LENGTH})",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"ssid": ssid, "length": len(ssid), "max_length": Constants.SSID_MAX_LENGTH}
            )
        
        return ssid
    
    @staticmethod
    def validate_channel(channel: int) -> int:
        """Validiere WiFi-Kanal."""
        if not isinstance(channel, int):
            raise ValidationError(
                f"Channel must be integer, got {type(channel).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"channel": channel, "type": str(type(channel))}
            )
        
        if channel not in Constants.SUPPORTED_CHANNELS:
            raise ValidationError(
                f"Unsupported channel: {channel} (supported: {Constants.SUPPORTED_CHANNELS})",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"channel": channel, "supported_channels": Constants.SUPPORTED_CHANNELS}
            )
        
        return channel


class FileSystemValidator(Validator):
    """Validatoren für Dateisystem-bezogene Eingaben."""
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> Path:
        """Validiere Dateipfad."""
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(
                f"Path must be string or Path, got {type(path).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PATH.value,
                details={"path": path, "type": str(type(path))}
            )
        
        try:
            resolved_path = path.resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(
                get_error_message(ErrorCodes.VALIDATION_INVALID_PATH, path=str(path)),
                error_code=ErrorCodes.VALIDATION_INVALID_PATH.value,
                details={"path": str(path), "error": str(e)}
            ) from e
        
        if must_exist and not resolved_path.exists():
            raise ValidationError(
                get_error_message(ErrorCodes.FILE_NOT_FOUND, file_path=str(resolved_path)),
                error_code=ErrorCodes.FILE_NOT_FOUND.value,
                details={"path": str(resolved_path)}
            )
        
        if must_be_file and not resolved_path.is_file():
            raise ValidationError(
                f"Path must be a file: {resolved_path}",
                error_code=ErrorCodes.VALIDATION_INVALID_PATH.value,
                details={"path": str(resolved_path), "is_file": resolved_path.is_file()}
            )
        
        if must_be_dir and not resolved_path.is_dir():
            raise ValidationError(
                f"Path must be a directory: {resolved_path}",
                error_code=ErrorCodes.VALIDATION_INVALID_PATH.value,
                details={"path": str(resolved_path), "is_dir": resolved_path.is_dir()}
            )
        
        return resolved_path
    
    @staticmethod
    def validate_file_extension(path: Union[str, Path], allowed_extensions: List[str]) -> Path:
        """Validiere Dateiendung."""
        path = FileSystemValidator.validate_file_path(path)
        
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"Invalid file extension: {path.suffix} (allowed: {allowed_extensions})",
                error_code=ErrorCodes.VALIDATION_INVALID_PATH.value,
                details={"path": str(path), "extension": path.suffix, "allowed": allowed_extensions}
            )
        
        return path


class DataValidator(Validator):
    """Validatoren für Daten-bezogene Eingaben."""
    
    @staticmethod
    def validate_timestamp(timestamp: Union[int, float, datetime], field_name: str = "timestamp") -> float:
        """Validiere Zeitstempel."""
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()
        elif isinstance(timestamp, (int, float)):
            pass
        else:
            raise ValidationError(
                f"Timestamp must be int, float, or datetime, got {type(timestamp).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_TIMESTAMP.value,
                details={"field": field_name, "timestamp": timestamp, "type": str(type(timestamp))}
            )
        
        # Validiere vernünftigen Zeitbereich (1970-2100)
        min_timestamp = 0
        max_timestamp = 4102444800  # 2100-01-01
        
        if not (min_timestamp <= timestamp <= max_timestamp):
            raise ValidationError(
                f"Timestamp out of range: {timestamp} (must be between {min_timestamp} and {max_timestamp})",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"field": field_name, "timestamp": timestamp, "min": min_timestamp, "max": max_timestamp}
            )
        
        return float(timestamp)
    
    @staticmethod
    def validate_rssi(rssi: Union[int, float], field_name: str = "rssi") -> float:
        """Validiere RSSI-Wert."""
        if not isinstance(rssi, (int, float)):
            raise ValidationError(
                f"RSSI must be numeric, got {type(rssi).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "rssi": rssi, "type": str(type(rssi))}
            )
        
        # RSSI sollte zwischen -100 und 0 dBm liegen
        if not (-100 <= rssi <= 0):
            raise ValidationError(
                f"RSSI out of range: {rssi} dBm (must be between -100 and 0)",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"field": field_name, "rssi": rssi, "min": -100, "max": 0}
            )
        
        return float(rssi)
    
    @staticmethod
    def validate_dataframe(df: Any, required_columns: Optional[List[str]] = None, field_name: str = "dataframe") -> pd.DataFrame:
        """Validiere DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                f"Data must be pandas DataFrame, got {type(df).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "data_type": str(type(df))}
            )
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(
                    f"DataFrame missing required columns: {missing_columns}",
                    error_code=ErrorCodes.VALIDATION_MISSING_REQUIRED_FIELD.value,
                    details={"field": field_name, "missing_columns": missing_columns, "available_columns": list(df.columns)}
                )
        
        return df
    
    @staticmethod
    def validate_numpy_array(arr: Any, expected_shape: Optional[Tuple[int, ...]] = None, field_name: str = "array") -> np.ndarray:
        """Validiere NumPy-Array."""
        if not isinstance(arr, np.ndarray):
            raise ValidationError(
                f"Data must be numpy array, got {type(arr).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "data_type": str(type(arr))}
            )
        
        if expected_shape and arr.shape != expected_shape:
            raise ValidationError(
                f"Array shape mismatch: expected {expected_shape}, got {arr.shape}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "expected_shape": expected_shape, "actual_shape": arr.shape}
            )
        
        return arr


class ConfigValidator(Validator):
    """Validatoren für Konfigurations-bezogene Eingaben."""
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any], required_keys: List[str], field_name: str = "config") -> Dict[str, Any]:
        """Validiere Konfigurations-Dictionary."""
        if not isinstance(config, dict):
            raise ValidationError(
                f"Config must be dictionary, got {type(config).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "config_type": str(type(config))}
            )
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(
                f"Config missing required keys: {missing_keys}",
                error_code=ErrorCodes.VALIDATION_MISSING_REQUIRED_FIELD.value,
                details={"field": field_name, "missing_keys": missing_keys, "available_keys": list(config.keys())}
            )
        
        return config
    
    @staticmethod
    def validate_log_level(level: str) -> str:
        """Validiere Log-Level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if not isinstance(level, str):
            raise ValidationError(
                f"Log level must be string, got {type(level).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"level": level, "type": str(type(level))}
            )
        
        level_upper = level.upper()
        if level_upper not in valid_levels:
            raise ValidationError(
                f"Invalid log level: {level} (valid: {valid_levels})",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"level": level, "valid_levels": valid_levels}
            )
        
        return level_upper


class BatchValidator(Validator):
    """Validatoren für Batch-Operationen."""
    
    @staticmethod
    def validate_batch_size(batch_size: int, field_name: str = "batch_size") -> int:
        """Validiere Batch-Größe."""
        if not isinstance(batch_size, int):
            raise ValidationError(
                f"Batch size must be integer, got {type(batch_size).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "batch_size": batch_size, "type": str(type(batch_size))}
            )
        
        if not (1 <= batch_size <= Constants.MAX_BATCH_SIZE):
            raise ValidationError(
                f"Batch size out of range: {batch_size} (must be between 1 and {Constants.MAX_BATCH_SIZE})",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"field": field_name, "batch_size": batch_size, "min": 1, "max": Constants.MAX_BATCH_SIZE}
            )
        
        return batch_size
    
    @staticmethod
    def validate_timeout(timeout: Union[int, float], field_name: str = "timeout") -> float:
        """Validiere Timeout-Wert."""
        if not isinstance(timeout, (int, float)):
            raise ValidationError(
                f"Timeout must be numeric, got {type(timeout).__name__}",
                error_code=ErrorCodes.VALIDATION_INVALID_PARAMETER.value,
                details={"field": field_name, "timeout": timeout, "type": str(type(timeout))}
            )
        
        if not (0 < timeout <= Constants.DEFAULT_TIMEOUT * 10):  # Max 5 Minuten
            raise ValidationError(
                f"Timeout out of range: {timeout} (must be between 0 and {Constants.DEFAULT_TIMEOUT * 10})",
                error_code=ErrorCodes.VALIDATION_OUT_OF_RANGE.value,
                details={"field": field_name, "timeout": timeout, "min": 0, "max": Constants.DEFAULT_TIMEOUT * 10}
            )
        
        return float(timeout)


# Convenience-Funktionen für häufige Validierungen
def validate_mac(mac: str) -> str:
    """Validiere MAC-Adresse (Convenience-Funktion)."""
    return NetworkValidator.validate_mac_address(mac)


def validate_ip(ip: str) -> str:
    """Validiere IP-Adresse (Convenience-Funktion)."""
    return NetworkValidator.validate_ip_address(ip)


def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validiere Dateipfad (Convenience-Funktion)."""
    return FileSystemValidator.validate_file_path(path, must_exist=must_exist)


def validate_timestamp(ts: Union[int, float, datetime]) -> float:
    """Validiere Zeitstempel (Convenience-Funktion)."""
    return DataValidator.validate_timestamp(ts)


def validate_rssi(rssi: Union[int, float]) -> float:
    """Validiere RSSI (Convenience-Funktion)."""
    return DataValidator.validate_rssi(rssi)


# Decorator für automatische Validierung
def validate_inputs(**validators: Callable[[Any], Any]):
    """Decorator für automatische Input-Validierung."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validiere alle spezifizierten Parameter
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    kwargs[param_name] = validator(kwargs[param_name])
            
            return func(*args, **kwargs)
        return wrapper
    return decorator