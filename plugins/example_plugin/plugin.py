"""
Example_Plugin Plugin.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from plugins import BasePlugin, PluginMetadata

logger = logging.getLogger(__name__)

class Plugin(BasePlugin):
    """Example_Plugin Plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Example_Plugin",
            version="1.0.0",
            description="Beschreibung des example_plugin Plugins",
            author="Dein Name",
            dependencies=[]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Hauptfunktion des Plugins.
        """
        console.print(f"\n[bold cyan]Example_Plugin Plugin wird ausgeführt...[/bold cyan]")
        
        try:
            # Plugin-Logik hier implementieren
            console.print(f"[green]Example_Plugin Plugin erfolgreich ausgeführt![/green]")
            
        except Exception as e:
            console.print(f"[red]Fehler im example_plugin Plugin: {e}[/red]")
            logger.error(f"Fehler im example_plugin Plugin: {e}", exc_info=True)
