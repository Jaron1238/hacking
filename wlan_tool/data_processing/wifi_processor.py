"""
WiFi Data Processor Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WiFiDataProcessor:
    """WiFi-Datenverarbeiter für WLAN-Analyse."""
    
    def __init__(self):
        """Initialisiert den WiFi-Datenverarbeiter."""
        self.required_columns = [
            'timestamp', 'device_id', 'signal_strength', 'frequency',
            'mac_address', 'ssid', 'encryption', 'channel', 'data_rate',
            'packet_count', 'bytes_transferred'
        ]
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verarbeitet WiFi-Daten.
        
        Args:
            data: Rohdaten DataFrame
            
        Returns:
            Verarbeitete Daten
        """
        if data.empty:
            raise ValueError("Empty dataset")
        
        # Erforderliche Spalten prüfen
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Daten kopieren
        processed_data = data.copy()
        
        # Timestamp verarbeiten
        processed_data['processed_timestamp'] = pd.to_datetime(processed_data['timestamp'])
        
        # Signal-Stärke bereinigen
        processed_data['signal_strength'] = self._clean_signal_strength(
            processed_data['signal_strength']
        )
        
        # Frequenz validieren
        processed_data['frequency'] = self._validate_frequency(
            processed_data['frequency']
        )
        
        # Fehlende Werte behandeln
        processed_data = self._handle_missing_values(processed_data)
        
        return processed_data
    
    def _clean_signal_strength(self, signal_data: pd.Series) -> pd.Series:
        """Bereinigt Signal-Stärke-Daten."""
        # Ungültige Werte entfernen
        signal_data = signal_data.replace([np.inf, -np.inf], np.nan)
        
        # Extreme Werte begrenzen
        signal_data = signal_data.clip(-100, 0)
        
        # Fehlende Werte mit Median füllen
        if signal_data.isna().any():
            median_value = signal_data.median()
            signal_data = signal_data.fillna(median_value)
        
        return signal_data
    
    def _validate_frequency(self, frequency_data: pd.Series) -> pd.Series:
        """Validiert Frequenz-Daten."""
        # Nur 2.4 GHz und 5.0 GHz erlauben
        valid_frequencies = [2.4, 5.0]
        frequency_data = frequency_data.where(
            frequency_data.isin(valid_frequencies), 2.4
        )
        
        return frequency_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Behandelt fehlende Werte."""
        # Numerische Spalten mit Median füllen
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isna().any():
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
        
        # Kategorische Spalten mit Modus füllen
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isna().any():
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
        
        return data
    
    def _classify_signal_quality(self, signal_strength: float) -> str:
        """Klassifiziert Signal-Qualität basierend auf Signal-Stärke."""
        if signal_strength >= -30:
            return "excellent"
        elif signal_strength >= -50:
            return "good"
        elif signal_strength >= -70:
            return "fair"
        else:
            return "poor"