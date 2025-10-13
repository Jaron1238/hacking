"""
Erweiterte Clustering-Algorithmen Plugin.

Bietet verschiedene Clustering-Algorithmen für WiFi-Client-Analyse:
- Spectral Clustering
- Hierarchical Clustering  
- Gaussian Mixture Model
- OPTICS
- HDBSCAN
"""

from .plugin import Plugin

__all__ = ['Plugin']