"""
Feature Extractor Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Feature-Extraktor für WLAN-Daten."""
    
    def __init__(self):
        """Initialisiert den Feature-Extraktor."""
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extrahiert Features aus WiFi-Daten.
        
        Args:
            data: Verarbeitete WiFi-Daten
            
        Returns:
            Feature-Matrix (n_samples, n_features)
        """
        features = []
        
        # Zeit-Features
        time_features = self.extract_time_features(data)
        features.append(time_features)
        
        # Signal-Features
        signal_features = self.extract_signal_features(data)
        features.append(signal_features)
        
        # Netzwerk-Features
        network_features = self.extract_network_features(data)
        features.append(network_features)
        
        # Features zusammenführen
        all_features = pd.concat(features, axis=1)
        
        # Skalierung
        if not self.is_fitted:
            scaled_features = self.scaler.fit_transform(all_features)
            self.is_fitted = True
        else:
            scaled_features = self.scaler.transform(all_features)
        
        return scaled_features
    
    def extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrahiert Zeit-basierte Features."""
        features = pd.DataFrame(index=data.index)
        
        if 'processed_timestamp' in data.columns:
            timestamp = pd.to_datetime(data['processed_timestamp'])
            features['hour'] = timestamp.dt.hour
            features['day_of_week'] = timestamp.dt.dayofweek
            features['is_weekend'] = (timestamp.dt.dayofweek >= 5).astype(int)
            features['month'] = timestamp.dt.month
        else:
            # Fallback für fehlende Timestamp
            features['hour'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0
            features['month'] = 1
        
        return features
    
    def extract_signal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrahiert Signal-basierte Features."""
        features = pd.DataFrame(index=data.index)
        
        if 'signal_strength' in data.columns:
            signal_data = data['signal_strength']
            features['signal_mean'] = signal_data.rolling(window=5, min_periods=1).mean()
            features['signal_std'] = signal_data.rolling(window=5, min_periods=1).std()
            features['signal_min'] = signal_data.rolling(window=5, min_periods=1).min()
            features['signal_max'] = signal_data.rolling(window=5, min_periods=1).max()
        else:
            features['signal_mean'] = -50
            features['signal_std'] = 0
            features['signal_min'] = -50
            features['signal_max'] = -50
        
        return features
    
    def extract_network_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrahiert Netzwerk-basierte Features."""
        features = pd.DataFrame(index=data.index)
        
        # SSID-Diversität
        if 'ssid' in data.columns:
            features['unique_ssids'] = data.groupby('device_id')['ssid'].transform('nunique')
        else:
            features['unique_ssids'] = 1
        
        # Verschlüsselungs-Typen
        if 'encryption' in data.columns:
            features['encryption_types'] = data.groupby('device_id')['encryption'].transform('nunique')
        else:
            features['encryption_types'] = 1
        
        # Kanal-Diversität
        if 'channel' in data.columns:
            features['channel_diversity'] = data.groupby('device_id')['channel'].transform('nunique')
        else:
            features['channel_diversity'] = 1
        
        # Frequenz-Features
        if 'frequency' in data.columns:
            features['is_5ghz'] = (data['frequency'] == 5.0).astype(int)
        else:
            features['is_5ghz'] = 0
        
        return features
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """
        Skaliert Features.
        
        Args:
            features: Feature-Matrix
            
        Returns:
            Skalierte Features
        """
        if not self.is_fitted:
            scaled_features = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            scaled_features = self.scaler.transform(features)
        
        return scaled_features