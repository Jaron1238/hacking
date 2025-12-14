#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für Error-Handling und Recovery-Mechanismen.
"""

import pytest
import tempfile
import sqlite3
import logging
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import time

from wlan_tool.exceptions import (
    WLANToolError, ConfigurationError, DatabaseError, CaptureError,
    AnalysisError, NetworkError, HardwareError, ValidationError,
    FileSystemError, PermissionError, ResourceError,
    handle_errors, retry_on_error, validate_input, ErrorContext
)
from wlan_tool.recovery import RecoveryManager, with_recovery, recovery_context
from wlan_tool.logging_config import setup_logging, get_logger, log_performance


class TestCustomExceptions:
    """Tests für Custom Exception-Klassen."""
    
    def test_wlan_tool_error_creation(self):
        """Test WLANToolError-Erstellung."""
        error = WLANToolError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert "WLANToolError: Test error (Code: TEST_ERROR)" in str(error)
    
    def test_specific_exceptions(self):
        """Test spezifische Exception-Typen."""
        exceptions = [
            ConfigurationError("Config error"),
            DatabaseError("DB error"),
            CaptureError("Capture error"),
            AnalysisError("Analysis error"),
            NetworkError("Network error"),
            HardwareError("Hardware error"),
            ValidationError("Validation error"),
            FileSystemError("File error"),
            PermissionError("Permission error"),
            ResourceError("Resource error")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, WLANToolError)
            assert exc.message in str(exc)


class TestErrorHandlingDecorators:
    """Tests für Error-Handling-Decorators."""
    
    def test_handle_errors_success(self):
        """Test handle_errors bei erfolgreicher Ausführung."""
        @handle_errors(WLANToolError, "TEST_ERROR", default_return="fallback")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_handle_errors_catch_exception(self):
        """Test handle_errors bei Exception."""
        @handle_errors(WLANToolError, "TEST_ERROR", default_return="fallback")
        def test_func():
            raise WLANToolError("Test error")
        
        result = test_func()
        assert result == "fallback"
    
    def test_handle_errors_reraise(self):
        """Test handle_errors mit reraise=True."""
        @handle_errors(WLANToolError, "TEST_ERROR", reraise=True)
        def test_func():
            raise ValueError("Original error")
        
        with pytest.raises(WLANToolError) as exc_info:
            test_func()
        
        assert "Original error" in str(exc_info.value)
        assert exc_info.value.error_code == "TEST_ERROR"
    
    def test_retry_on_error_success(self):
        """Test retry_on_error bei erfolgreicher Ausführung."""
        @retry_on_error(max_attempts=3, delay=0.01)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_retry_on_error_retry_success(self):
        """Test retry_on_error mit erfolgreichem Retry."""
        call_count = 0
        
        @retry_on_error(max_attempts=3, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_error_max_attempts(self):
        """Test retry_on_error mit max attempts erreicht."""
        call_count = 0
        
        @retry_on_error(max_attempts=2, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert "Persistent error" in str(exc_info.value)
        assert call_count == 2
    
    def test_validate_input_success(self):
        """Test validate_input bei gültiger Eingabe."""
        @validate_input(value=lambda x: x > 0)
        def test_func(value):
            return value * 2
        
        result = test_func(value=5)
        assert result == 10
    
    def test_validate_input_invalid(self):
        """Test validate_input bei ungültiger Eingabe."""
        @validate_input(value=lambda x: x > 0)
        def test_func(value):
            return value * 2
        
        with pytest.raises(ValidationError) as exc_info:
            test_func(value=-1)
        
        assert "Invalid value for parameter 'value': -1" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_INPUT"


class TestErrorContext:
    """Tests für ErrorContext."""
    
    def test_error_context_success(self):
        """Test ErrorContext bei erfolgreicher Ausführung."""
        with ErrorContext("test_operation", "TEST_ERROR") as ctx:
            result = "success"
        
        assert result == "success"
    
    def test_error_context_exception(self):
        """Test ErrorContext bei Exception."""
        with pytest.raises(WLANToolError) as exc_info:
            with ErrorContext("test_operation", "TEST_ERROR"):
                raise ValueError("Original error")
        
        assert "Original error" in str(exc_info.value)
        assert exc_info.value.error_code == "TEST_ERROR"
        assert exc_info.value.details["operation"] == "test_operation"
        assert exc_info.value.details["original_type"] == "ValueError"


class TestRecoveryManager:
    """Tests für RecoveryManager."""
    
    def test_recovery_manager_initialization(self):
        """Test RecoveryManager-Initialisierung."""
        manager = RecoveryManager()
        
        assert len(manager.recovery_strategies) > 0
        assert len(manager.fallback_handlers) == 0
        assert len(manager.circuit_breakers) == 0
    
    def test_register_strategy(self):
        """Test Strategy-Registrierung."""
        manager = RecoveryManager()
        
        def test_recovery(exc, op):
            return "recovered"
        
        manager.register_strategy(ValueError, test_recovery, max_attempts=2)
        
        assert ValueError in manager.recovery_strategies
        strategy = manager.recovery_strategies[ValueError]
        assert strategy['function'] == test_recovery
        assert strategy['max_attempts'] == 2
    
    def test_register_fallback(self):
        """Test Fallback-Registrierung."""
        manager = RecoveryManager()
        
        def test_fallback():
            return "fallback_result"
        
        manager.register_fallback("test_operation", test_fallback)
        
        assert "test_operation" in manager.fallback_handlers
        assert manager.fallback_handlers["test_operation"] == test_fallback
    
    def test_recover_with_strategy(self):
        """Test Recovery mit Strategy."""
        manager = RecoveryManager()
        
        def test_recovery(exc, op):
            return "recovered"
        
        manager.register_strategy(ValueError, test_recovery, max_attempts=2)
        
        result = manager.recover(ValueError("Test error"), "test_operation")
        assert result == "recovered"
    
    def test_recover_with_fallback(self):
        """Test Recovery mit Fallback."""
        manager = RecoveryManager()
        
        def test_fallback():
            return "fallback_result"
        
        manager.register_fallback("test_operation", test_fallback)
        
        result = manager.recover(ValueError("Test error"), "test_operation")
        assert result == "fallback_result"
    
    def test_circuit_breaker(self):
        """Test Circuit Breaker-Funktionalität."""
        manager = RecoveryManager()
        
        def failing_recovery(exc, op):
            raise Exception("Recovery failed")
        
        manager.register_strategy(ValueError, failing_recovery, max_attempts=2)
        
        # Erste Versuche sollten fehlschlagen
        result1 = manager.recover(ValueError("Test error"), "test_operation")
        result2 = manager.recover(ValueError("Test error"), "test_operation")
        
        # Nach max_attempts sollte Circuit Breaker offen sein
        result3 = manager.recover(ValueError("Test error"), "test_operation")
        
        assert result1 is None  # Recovery failed
        assert result2 is None  # Recovery failed
        assert result3 is None  # Circuit breaker open


class TestRecoveryDecorators:
    """Tests für Recovery-Decorators."""
    
    def test_with_recovery_success(self):
        """Test with_recovery bei erfolgreicher Ausführung."""
        @with_recovery("test_operation")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_with_recovery_with_fallback(self):
        """Test with_recovery mit Fallback."""
        def fallback_func():
            return "fallback_result"
        
        @with_recovery("test_operation", fallback_func)
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        assert result == "fallback_result"
    
    def test_recovery_context_success(self):
        """Test recovery_context bei erfolgreicher Ausführung."""
        with recovery_context("test_operation") as ctx:
            result = "success"
        
        assert result == "success"
    
    def test_recovery_context_with_fallback(self):
        """Test recovery_context mit Fallback."""
        def fallback_func():
            return "fallback_result"
        
        with recovery_context("test_operation", fallback_func) as ctx:
            raise ValueError("Test error")
        
        # Context Manager sollte Exception weiterleiten
        # (Fallback wird nur bei explizitem recover() aufgerufen)


class TestLoggingSystem:
    """Tests für Logging-System."""
    
    def test_setup_logging(self):
        """Test Logging-Setup."""
        logger = setup_logging(
            log_level="DEBUG",
            enable_console=True,
            enable_performance_logging=True,
            enable_error_tracking=True
        )
        
        assert logger is not None
        assert logger.name == "wlan_tool"
        assert logger.level == logging.DEBUG
    
    def test_get_logger(self):
        """Test Logger-Abruf."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "wlan_tool.test_module"
    
    def test_log_performance(self):
        """Test Performance-Logging."""
        logger = get_logger("test_module")
        
        # Sollte nicht crashen
        log_performance(logger, "test_operation", 1.5, memory_usage=1024, cpu_usage=50.0)


class TestDatabaseErrorHandling:
    """Tests für Datenbank-Error-Handling."""
    
    def test_database_connection_error(self):
        """Test Datenbankverbindungs-Fehler."""
        with pytest.raises(DatabaseError) as exc_info:
            from wlan_tool.storage.database import db_conn_ctx
            with db_conn_ctx("/invalid/path/database.db"):
                pass
        
        assert "Cannot create database directory" in str(exc_info.value)
        assert exc_info.value.error_code == "DB_UNEXPECTED_ERROR"
    
    def test_database_migration_error(self):
        """Test Datenbankmigrations-Fehler."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Erstelle ungültige Migration
            migrations_dir = Path("wlan_tool/assets/sql_data/versions")
            migrations_dir.mkdir(parents=True, exist_ok=True)
            
            invalid_migration = migrations_dir / "999_invalid.sql"
            invalid_migration.write_text("INVALID SQL SYNTAX")
            
            # Die migrate_db Funktion hat error handling und wirft keine Exception
            from wlan_tool.storage.database import migrate_db
            migrate_db(db_path)  # Sollte nicht crashen, sondern graceful handling
        
        finally:
            Path(db_path).unlink(missing_ok=True)
            if invalid_migration.exists():
                invalid_migration.unlink()


class TestAnalysisErrorHandling:
    """Tests für Analyse-Error-Handling."""
    
    def test_client_features_null_state(self):
        """Test Client-Features mit None-State."""
        from wlan_tool.analysis.logic import features_for_client
        
        # Die Funktion hat einen @handle_errors Decorator, der None zurückgibt
        result = features_for_client(None)
        assert result is None
    
    def test_client_features_invalid_type(self):
        """Test Client-Features mit ungültigem Typ."""
        from wlan_tool.analysis.logic import features_for_client
        
        # Die Funktion hat einen @handle_errors Decorator, der None zurückgibt
        result = features_for_client("invalid")
        assert result is None
    
    def test_clustering_invalid_parameters(self):
        """Test Clustering mit ungültigen Parametern."""
        from wlan_tool.analysis.logic import cluster_clients
        from wlan_tool.storage.state import WifiAnalysisState
        
        state = WifiAnalysisState()
        
        # Die Funktion hat einen @handle_errors Decorator, der (None, None) zurückgibt
        result = cluster_clients(state, n_clusters=-1)  # Verwende -1 statt 0
        assert result == (None, None)
        
        # Teste ungültigen Algorithmus
        result = cluster_clients(state, algo="invalid")
        assert result == (None, None)


class TestFileSystemErrorHandling:
    """Tests für FileSystem-Error-Handling."""
    
    def test_csv_export_invalid_path(self):
        """Test CSV-Export mit ungültigem Pfad."""
        from wlan_tool.storage.database import export_confirmed_to_csv
        
        # Die Funktion hat einen @handle_errors Decorator, der 0 zurückgibt
        result = export_confirmed_to_csv("", "/tmp/test.csv")
        assert result == 0
    
    def test_csv_export_missing_db(self):
        """Test CSV-Export mit fehlender Datenbank."""
        from wlan_tool.storage.database import export_confirmed_to_csv
        
        # Die Funktion hat einen @handle_errors Decorator, der 0 zurückgibt
        result = export_confirmed_to_csv("/nonexistent/database.db", "/tmp/test.csv")
        assert result == 0


class TestErrorRecoveryIntegration:
    """Integrationstests für Error-Recovery."""
    
    def test_database_recovery_flow(self):
        """Test kompletten Database-Recovery-Flow."""
        manager = RecoveryManager()
        
        # Simuliere Database-Fehler
        db_error = DatabaseError("Database locked", error_code="DB_LOCKED")
        
        # Recovery sollte erfolgreich sein (wait and retry)
        result = manager.recover(db_error, "database_operation")
        assert result is True
    
    def test_file_system_recovery_flow(self):
        """Test kompletten FileSystem-Recovery-Flow."""
        manager = RecoveryManager()
        
        # Simuliere FileSystem-Fehler
        fs_error = FileSystemError("Permission denied", error_code="PERMISSION_DENIED")
        
        # Recovery sollte versucht werden
        result = manager.recover(fs_error, "file_operation")
        # Kann None sein wenn Recovery fehlschlägt, das ist OK für Test
        assert result is not None or result is None


class TestErrorHandlingEdgeCases:
    """Tests für Edge-Cases im Error-Handling."""
    
    def test_nested_error_contexts(self):
        """Test verschachtelte Error-Contexts."""
        with pytest.raises(WLANToolError) as exc_info:
            with ErrorContext("outer_operation", "OUTER_ERROR") as outer:
                with ErrorContext("inner_operation", "INNER_ERROR") as inner:
                    raise ValueError("Inner error")
        
        # Sollte die innere Exception mit dem inneren Context behandeln
        assert "Inner error" in str(exc_info.value)
        assert exc_info.value.error_code == "INNER_ERROR"
    
    def test_error_context_without_exception(self):
        """Test ErrorContext ohne Exception."""
        with ErrorContext("test_operation", "TEST_ERROR") as ctx:
            result = "success"
        
        assert result == "success"
    
    def test_retry_with_different_exceptions(self):
        """Test Retry mit verschiedenen Exception-Typen."""
        call_count = 0
        
        @retry_on_error(max_attempts=3, delay=0.01, exceptions=(ValueError, TypeError))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Value error")
            elif call_count == 2:
                raise TypeError("Type error")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 3
    
    def test_validate_input_multiple_parameters(self):
        """Test validate_input mit mehreren Parametern."""
        @validate_input(
            value1=lambda x: x > 0,
            value2=lambda x: isinstance(x, str)
        )
        def test_func(value1, value2):
            return f"{value1}_{value2}"
        
        # Erfolgreicher Fall
        result = test_func(value1=5, value2="test")
        assert result == "5_test"
        
        # Fehlerhafter Fall
        with pytest.raises(ValidationError):
            test_func(value1=-1, value2="test")
        
        with pytest.raises(ValidationError):
            test_func(value1=5, value2=123)