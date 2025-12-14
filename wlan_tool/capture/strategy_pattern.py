# Strategy Pattern für verschiedene Capture-Modi
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class CaptureStrategy(ABC):
    """Abstract Base Class für Capture-Strategien"""
    
    @abstractmethod
    async def capture(self, interface: str, duration: int, **kwargs) -> Dict[str, Any]:
        """Führt Packet Capture durch"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validiert Konfiguration für diese Strategie"""
        pass

class BasicCaptureStrategy(CaptureStrategy):
    """Basis Capture-Strategie ohne erweiterte Features"""
    
    async def capture(self, interface: str, duration: int, **kwargs) -> Dict[str, Any]:
        logger.info(f"Basic Capture auf {interface} für {duration}s")
        # Basis Packet Capture Logik
        return {"packets": [], "mode": "basic"}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "interface" in config and "duration" in config

class EnhancedCaptureStrategy(CaptureStrategy):
    """Erweiterte Capture-Strategie mit DPI und Metriken"""
    
    async def capture(self, interface: str, duration: int, **kwargs) -> Dict[str, Any]:
        logger.info(f"Enhanced Capture mit DPI auf {interface}")
        # Erweiterte Capture Logik mit Deep Packet Inspection
        return {"packets": [], "dpi_data": {}, "metrics": {}, "mode": "enhanced"}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["interface", "duration", "dpi_enabled", "metrics_enabled"]
        return all(key in config for key in required)

class LiveCaptureStrategy(CaptureStrategy):
    """Live Capture-Strategie mit Real-time TUI"""
    
    async def capture(self, interface: str, duration: int, **kwargs) -> Dict[str, Any]:
        logger.info(f"Live Capture mit TUI auf {interface}")
        # Live Capture mit TUI Updates
        return {"packets": [], "live_updates": True, "mode": "live"}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "interface" in config and "tui_enabled" in config

class CaptureContext:
    """Context-Klasse die verschiedene Capture-Strategien verwendet"""
    
    def __init__(self, strategy: CaptureStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: CaptureStrategy):
        """Wechselt die Capture-Strategie zur Laufzeit"""
        self._strategy = strategy
    
    async def execute_capture(self, interface: str, duration: int, **kwargs) -> Dict[str, Any]:
        """Führt Capture mit aktueller Strategie durch"""
        return await self._strategy.capture(interface, duration, **kwargs)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validiert Konfiguration mit aktueller Strategie"""
        return self._strategy.validate_config(config)

# Factory für Capture-Strategien
class CaptureStrategyFactory:
    """Factory für Capture-Strategien"""
    
    _strategies = {
        'basic': BasicCaptureStrategy,
        'enhanced': EnhancedCaptureStrategy,
        'live': LiveCaptureStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str) -> CaptureStrategy:
        """Erstellt Capture-Strategie basierend auf Typ"""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unbekannte Capture-Strategie: {strategy_type}")
        
        return cls._strategies[strategy_type]()