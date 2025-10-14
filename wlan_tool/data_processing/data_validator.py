"""
Data Validator Module
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Daten-Validator für WLAN-Daten."""
    
    def __init__(self):
        """Initialisiert den Daten-Validator."""
        self.required_columns = [
            'timestamp', 'device_id', 'signal_strength', 'frequency',
            'mac_address', 'ssid', 'encryption', 'channel', 'data_rate',
            'packet_count', 'bytes_transferred'
        ]
        
        # Validierungs-Patterns
        self.mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
        self.sql_injection_patterns = [
            r"('|(\\')|(;)|(--)|(/\*)|(\*/)|(\b(OR|AND)\b)",
            r"(\bUNION\b)",
            r"(\bSELECT\b)",
            r"(\bINSERT\b)",
            r"(\bUPDATE\b)",
            r"(\bDELETE\b)",
            r"(\bDROP\b)"
        ]
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<img[^>]*onerror[^>]*>"
        ]
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validiert WiFi-Daten.
        
        Args:
            data: Zu validierende Daten
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Leere Daten prüfen
        if data.empty:
            errors.append("Empty dataset")
            return False, errors
        
        # Erforderliche Spalten prüfen
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Signal-Stärke validieren
        if 'signal_strength' in data.columns:
            signal_errors = self._validate_signal_strength(data['signal_strength'])
            errors.extend(signal_errors)
        
        # Frequenz validieren
        if 'frequency' in data.columns:
            frequency_errors = self._validate_frequency(data['frequency'])
            errors.extend(frequency_errors)
        
        # MAC-Adressen validieren
        if 'mac_address' in data.columns:
            mac_errors = self._validate_mac_addresses(data['mac_address'])
            errors.extend(mac_errors)
        
        # Sicherheits-Checks
        security_errors = self._check_security(data)
        errors.extend(security_errors)
        
        # Duplicate Timestamps prüfen
        if 'timestamp' in data.columns:
            duplicate_errors = self._check_duplicate_timestamps(data['timestamp'])
            errors.extend(duplicate_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_signal_strength(self, signal_data: pd.Series) -> List[str]:
        """Validiert Signal-Stärke-Daten."""
        errors = []
        
        # NaN-Werte prüfen
        if signal_data.isna().any():
            errors.append("Signal strength contains NaN values")
        
        # Bereich validieren (-100 dBm bis 0 dBm)
        invalid_values = signal_data[(signal_data < -100) | (signal_data > 0)]
        if not invalid_values.empty:
            errors.append(f"Invalid signal strength values: {invalid_values.tolist()}")
        
        return errors
    
    def _validate_frequency(self, frequency_data: pd.Series) -> List[str]:
        """Validiert Frequenz-Daten."""
        errors = []
        
        # Gültige Frequenzen (2.4 GHz und 5.0 GHz)
        valid_frequencies = [2.4, 5.0]
        invalid_frequencies = frequency_data[~frequency_data.isin(valid_frequencies)]
        
        if not invalid_frequencies.empty:
            errors.append(f"Invalid frequency values: {invalid_frequencies.tolist()}")
        
        return errors
    
    def _validate_mac_addresses(self, mac_data: pd.Series) -> List[str]:
        """Validiert MAC-Adressen."""
        errors = []
        
        for idx, mac in mac_data.items():
            if pd.isna(mac):
                continue
            
            if not self.mac_pattern.match(str(mac)):
                errors.append(f"Invalid MAC address at index {idx}: {mac}")
        
        return errors
    
    def _check_security(self, data: pd.DataFrame) -> List[str]:
        """Führt Sicherheits-Checks durch."""
        errors = []
        
        # SQL-Injection-Check
        for col in data.select_dtypes(include=['object']).columns:
            for idx, value in data[col].items():
                if pd.isna(value):
                    continue
                
                value_str = str(value)
                
                # SQL-Injection-Patterns prüfen
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        errors.append(f"Potential SQL injection in {col} at index {idx}")
                        break
                
                # XSS-Patterns prüfen
                for pattern in self.xss_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        errors.append(f"Potential XSS in {col} at index {idx}")
                        break
        
        return errors
    
    def _check_duplicate_timestamps(self, timestamp_data: pd.Series) -> List[str]:
        """Prüft auf doppelte Timestamps."""
        errors = []
        
        duplicates = timestamp_data.duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            errors.append(f"Duplicate timestamps found: {duplicate_count} duplicates")
        
        return errors