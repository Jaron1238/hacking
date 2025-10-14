"""
WiFi Visualizer Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WiFiVisualizer:
    """WiFi-Daten-Visualisierer."""
    
    def __init__(self):
        """Initialisiert den WiFi-Visualisierer."""
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_signal_strength(self, data: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Erstellt Signal-Stärke-Visualisierung.
        
        Args:
            data: WiFi-Daten
            
        Returns:
            Matplotlib Figure
        """
        try:
            if 'signal_strength' not in data.columns:
                logger.warning("Signal strength column not found")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Signal-Stärke über Zeit
            if 'processed_timestamp' in data.columns:
                ax.plot(data['processed_timestamp'], data['signal_strength'], alpha=0.7)
                ax.set_xlabel('Zeit')
            else:
                ax.plot(data['signal_strength'], alpha=0.7)
                ax.set_xlabel('Index')
            
            ax.set_ylabel('Signal-Stärke (dBm)')
            ax.set_title('WiFi Signal-Stärke über Zeit')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating signal strength plot: {e}")
            return None
    
    def plot_clusters(self, features: np.ndarray, labels: np.ndarray) -> Optional[plt.Figure]:
        """
        Erstellt Clustering-Visualisierung.
        
        Args:
            features: Feature-Matrix
            labels: Cluster-Labels
            
        Returns:
            Matplotlib Figure
        """
        try:
            if features.shape[1] < 2:
                logger.warning("Features must have at least 2 dimensions for plotting")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 2D-Scatter-Plot der ersten beiden Features
            scatter = ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('WiFi-Daten Clustering')
            
            # Farbbalken hinzufügen
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster plot: {e}")
            return None
    
    def plot_device_distribution(self, data: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Erstellt Geräte-Verteilungs-Visualisierung.
        
        Args:
            data: WiFi-Daten
            
        Returns:
            Matplotlib Figure
        """
        try:
            if 'device_id' not in data.columns:
                logger.warning("Device ID column not found")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Geräte-Anzahl
            device_counts = data['device_id'].value_counts()
            ax1.bar(range(len(device_counts)), device_counts.values)
            ax1.set_xlabel('Geräte-Index')
            ax1.set_ylabel('Anzahl Verbindungen')
            ax1.set_title('Verbindungen pro Gerät')
            ax1.tick_params(axis='x', rotation=45)
            
            # SSID-Verteilung
            if 'ssid' in data.columns:
                ssid_counts = data['ssid'].value_counts()
                ax2.pie(ssid_counts.values, labels=ssid_counts.index, autopct='%1.1f%%')
                ax2.set_title('SSID-Verteilung')
            else:
                ax2.text(0.5, 0.5, 'SSID-Daten nicht verfügbar', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('SSID-Verteilung')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating device distribution plot: {e}")
            return None
    
    def plot_frequency_analysis(self, data: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Erstellt Frequenz-Analyse-Visualisierung.
        
        Args:
            data: WiFi-Daten
            
        Returns:
            Matplotlib Figure
        """
        try:
            if 'frequency' not in data.columns:
                logger.warning("Frequency column not found")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Frequenz-Verteilung
            frequency_counts = data['frequency'].value_counts()
            ax1.bar(frequency_counts.index, frequency_counts.values)
            ax1.set_xlabel('Frequenz (GHz)')
            ax1.set_ylabel('Anzahl')
            ax1.set_title('Frequenz-Verteilung')
            
            # Kanal-Verteilung
            if 'channel' in data.columns:
                channel_counts = data['channel'].value_counts().sort_index()
                ax2.bar(channel_counts.index, channel_counts.values)
                ax2.set_xlabel('Kanal')
                ax2.set_ylabel('Anzahl')
                ax2.set_title('Kanal-Verteilung')
            else:
                ax2.text(0.5, 0.5, 'Kanal-Daten nicht verfügbar', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Kanal-Verteilung')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating frequency analysis plot: {e}")
            return None