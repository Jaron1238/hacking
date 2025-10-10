#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zentrale Exception-Klassen und Error-Handler für das WLAN-Analyse-Tool.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import functools
import sys


# ==============================================================================
# CUSTOM EXCEPTION CLASSES
# ==============================================================================

class WLANToolError(Exception):
    """Basis-Exception für alle WLAN-Tool-Fehler."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None
    
    def __str__(self):
        base_msg = f"WLANToolError: {self.message}"
        if self.error_code:
            base_msg += f" (Code: {self.error_code})"
        if self.details:
            base_msg += f" | Details: {self.details}"
        return base_msg


class ConfigurationError(WLANToolError):
    """Fehler bei der Konfiguration."""
    pass


class DatabaseError(WLANToolError):
    """Fehler bei Datenbankoperationen."""
    pass


class CaptureError(WLANToolError):
    """Fehler bei der Paket-Erfassung."""
    pass


class AnalysisError(WLANToolError):
    """Fehler bei der Analyse."""
    pass


class NetworkError(WLANToolError):
    """Fehler bei Netzwerkoperationen."""
    pass


class HardwareError(WLANToolError):
    """Fehler bei Hardware-Zugriff."""
    pass


class ValidationError(WLANToolError):
    """Fehler bei der Datenvalidierung."""
    pass


class FileSystemError(WLANToolError):
    """Fehler bei Dateisystemoperationen."""
    pass


class PermissionError(WLANToolError):
    """Fehler bei Berechtigungen."""
    pass


class ResourceError(WLANToolError):
    """Fehler bei Ressourcen (Speicher, CPU, etc.)."""
    pass


# ==============================================================================
# ERROR HANDLER DECORATORS
# ==============================================================================

def handle_errors(
    error_type: type = WLANToolError,
    error_code: Optional[str] = None,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
):
    """
    Decorator für automatische Fehlerbehandlung.
    
    Args:
        error_type: Exception-Typ, der abgefangen werden soll
        error_code: Fehlercode für die Exception
        default_return: Rückgabewert bei Fehler
        log_error: Ob der Fehler geloggt werden soll
        reraise: Ob die Exception erneut geworfen werden soll
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    if isinstance(e, WLANToolError):
                        e.error_code = error_code or e.error_code
                    raise
                else:
                    return default_return
            except Exception as e:
                if log_error:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    # Wandle in WLANToolError um
                    wlan_error = WLANToolError(
                        message=str(e),
                        error_code=error_code or "UNEXPECTED_ERROR",
                        details={"original_error": type(e).__name__}
                    )
                    raise wlan_error
                else:
                    return default_return
        
        return wrapper
    return decorator


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator für Wiederholung bei Fehlern.
    
    Args:
        max_attempts: Maximale Anzahl Versuche
        delay: Verzögerung zwischen Versuchen (Sekunden)
        backoff_factor: Faktor für exponentielle Verzögerung
        exceptions: Exception-Typen, bei denen wiederholt werden soll
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed in {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger = logging.getLogger(func.__module__)
                        logger.error(f"All {max_attempts} attempts failed in {func.__name__}: {e}")
            
            # Alle Versuche fehlgeschlagen
            raise last_exception
        
        return wrapper
    return decorator


def validate_input(
    **validators: Callable[[Any], bool]
):
    """
    Decorator für Eingabevalidierung.
    
    Args:
        **validators: Validator-Funktionen für Parameter
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validiere Parameter
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    if not validator(kwargs[param_name]):
                        raise ValidationError(
                            f"Invalid value for parameter '{param_name}': {kwargs[param_name]}",
                            error_code="INVALID_INPUT",
                            details={"parameter": param_name, "value": kwargs[param_name]}
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ==============================================================================
# ERROR CONTEXT MANAGER
# ==============================================================================

class ErrorContext:
    """Context Manager für strukturierte Fehlerbehandlung."""
    
    def __init__(self, operation: str, error_code: Optional[str] = None, 
                 log_level: int = logging.ERROR):
        self.operation = operation
        self.error_code = error_code
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__module__)
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"Error in {self.operation}: {exc_val}"
            
            if isinstance(exc_val, WLANToolError):
                error_msg += f" (Code: {exc_val.error_code})"
                if exc_val.details:
                    error_msg += f" | Details: {exc_val.details}"
            else:
                error_msg += f" | Type: {exc_type.__name__}"
            
            self.logger.log(self.log_level, error_msg)
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Konvertiere zu WLANToolError falls nötig
            if not isinstance(exc_val, WLANToolError):
                wlan_error = WLANToolError(
                    message=str(exc_val),
                    error_code=self.error_code or "CONTEXT_ERROR",
                    details={"operation": self.operation, "original_type": exc_type.__name__}
                )
                raise wlan_error from exc_val
        
        return False  # Re-raise exception


# ==============================================================================
# ERROR RECOVERY MECHANISMS
# ==============================================================================

class ErrorRecovery:
    """Klasse für Error-Recovery-Strategien."""
    
    @staticmethod
    def safe_file_operation(operation: Callable, file_path: str, 
                          fallback: Optional[Callable] = None) -> Any:
        """Sichere Dateioperation mit Fallback."""
        try:
            return operation()
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"File operation failed for {file_path}: {e}")
            
            if fallback:
                logger.info(f"Using fallback for {file_path}")
                return fallback()
            else:
                raise FileSystemError(
                    f"File operation failed: {file_path}",
                    error_code="FILE_OPERATION_FAILED",
                    details={"file_path": file_path, "original_error": str(e)}
                ) from e
    
    @staticmethod
    def safe_database_operation(operation: Callable, 
                              fallback: Optional[Callable] = None) -> Any:
        """Sichere Datenbankoperation mit Fallback."""
        try:
            return operation()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Database operation failed: {e}")
            
            if fallback:
                logger.info("Using database fallback")
                return fallback()
            else:
                raise DatabaseError(
                    f"Database operation failed: {e}",
                    error_code="DATABASE_OPERATION_FAILED",
                    details={"original_error": str(e)}
                ) from e
    
    @staticmethod
    def safe_network_operation(operation: Callable, 
                             fallback: Optional[Callable] = None) -> Any:
        """Sichere Netzwerkoperation mit Fallback."""
        try:
            return operation()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Network operation failed: {e}")
            
            if fallback:
                logger.info("Using network fallback")
                return fallback()
            else:
                raise NetworkError(
                    f"Network operation failed: {e}",
                    error_code="NETWORK_OPERATION_FAILED",
                    details={"original_error": str(e)}
                ) from e


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_error_logging(log_file: Optional[str] = None, 
                       log_level: int = logging.INFO) -> logging.Logger:
    """Richte Error-Logging ein."""
    logger = logging.getLogger("wlan_tool")
    logger.setLevel(log_level)
    
    # Verhindere doppelte Handler
    if logger.handlers:
        return logger
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_error_summary(exception: Exception) -> Dict[str, Any]:
    """Erstelle eine Zusammenfassung eines Fehlers."""
    summary = {
        "type": type(exception).__name__,
        "message": str(exception),
        "timestamp": None
    }
    
    if isinstance(exception, WLANToolError):
        summary.update({
            "error_code": exception.error_code,
            "details": exception.details
        })
    
    return summary


def log_error_with_context(logger: logging.Logger, error: Exception, 
                          context: Dict[str, Any]) -> None:
    """Logge Fehler mit Kontextinformationen."""
    error_summary = get_error_summary(error)
    error_summary.update(context)
    
    logger.error(f"Error occurred: {error_summary}")
    logger.debug(f"Full traceback: {traceback.format_exc()}")


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_mac_address(mac: str) -> bool:
    """Validiere MAC-Adresse."""
    import re
    pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    return bool(re.match(pattern, mac))


def validate_ip_address(ip: str) -> bool:
    """Validiere IP-Adresse."""
    import socket
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


def validate_file_path(path: str) -> bool:
    """Validiere Dateipfad."""
    try:
        Path(path).resolve()
        return True
    except (OSError, ValueError):
        return False


def validate_positive_number(value: Any) -> bool:
    """Validiere positive Zahl."""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def validate_non_empty_string(value: Any) -> bool:
    """Validiere nicht-leeren String."""
    return isinstance(value, str) and len(value.strip()) > 0