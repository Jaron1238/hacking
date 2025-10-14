#!/usr/bin/env python3
"""
Plugin Health Check - Überprüft den Zustand aller Plugins.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Füge Projekt-Root zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plugins import load_all_plugins, BasePlugin


class PluginHealthChecker:
    """Health Checker für Plugins."""
    
    def __init__(self, plugin_dir: Path = None):
        self.plugin_dir = plugin_dir or project_root / "plugins"
        self.plugins = {}
        self.health_report = {}
    
    def check_all_plugins(self) -> Dict[str, Any]:
        """Führt einen vollständigen Health Check aller Plugins durch."""
        print("🔍 Plugin Health Check gestartet...")
        
        # Lade alle Plugins
        self.plugins = load_all_plugins(self.plugin_dir)
        
        # Prüfe jeden Plugin
        for name, plugin in self.plugins.items():
            self.health_report[name] = self._check_plugin_health(name, plugin)
        
        # Generiere Gesamtbericht
        total_plugins = len(self.plugins)
        healthy_plugins = sum(1 for report in self.health_report.values() if report['status'] == 'healthy')
        warning_plugins = sum(1 for report in self.health_report.values() if report['status'] == 'warning')
        error_plugins = sum(1 for report in self.health_report.values() if report['status'] == 'error')
        
        self.health_report['_summary'] = {
            'total_plugins': total_plugins,
            'healthy_plugins': healthy_plugins,
            'warning_plugins': warning_plugins,
            'error_plugins': error_plugins,
            'health_percentage': (healthy_plugins / total_plugins * 100) if total_plugins > 0 else 0
        }
        
        return self.health_report
    
    def _check_plugin_health(self, name: str, plugin: BasePlugin) -> Dict[str, Any]:
        """Prüft die Gesundheit eines einzelnen Plugins."""
        report = {
            'name': plugin.get_metadata().name,
            'version': plugin.get_metadata().version,
            'status': 'unknown',
            'issues': [],
            'warnings': [],
            'checks': {}
        }
        
        # 1. Dependencies Check
        deps_ok = plugin.validate_dependencies()
        report['checks']['dependencies'] = deps_ok
        if not deps_ok:
            report['issues'].append("Dependencies fehlen")
        
        # 2. Plugin-Dateien Check
        plugin_dir = self.plugin_dir / name
        files_check = self._check_plugin_files(plugin_dir)
        report['checks']['files'] = files_check
        if not files_check['all_present']:
            report['issues'].extend(files_check['missing_files'])
        
        # 3. Test-Dateien Check
        tests_check = self._check_test_files(plugin_dir)
        report['checks']['tests'] = tests_check
        if not tests_check['has_tests']:
            report['warnings'].append("Keine Tests vorhanden")
        
        # 4. Requirements Check
        req_check = self._check_requirements_file(plugin_dir)
        report['checks']['requirements'] = req_check
        if not req_check['has_requirements']:
            report['warnings'].append("Keine requirements.txt vorhanden")
        
        # 5. Plugin-Funktionalität Check
        func_check = self._check_plugin_functionality(plugin)
        report['checks']['functionality'] = func_check
        if not func_check['run_method_works']:
            report['issues'].append("run() Methode funktioniert nicht")
        
        # Status bestimmen
        if report['issues']:
            report['status'] = 'error'
        elif report['warnings']:
            report['status'] = 'warning'
        else:
            report['status'] = 'healthy'
        
        return report
    
    def _check_plugin_files(self, plugin_dir: Path) -> Dict[str, Any]:
        """Prüft, ob alle notwendigen Plugin-Dateien vorhanden sind."""
        required_files = ['__init__.py', 'plugin.py']
        optional_files = ['requirements.txt', 'README.md']
        
        missing_files = []
        present_files = []
        
        for file in required_files:
            if (plugin_dir / file).exists():
                present_files.append(file)
            else:
                missing_files.append(file)
        
        for file in optional_files:
            if (plugin_dir / file).exists():
                present_files.append(file)
        
        return {
            'all_present': len(missing_files) == 0,
            'missing_files': missing_files,
            'present_files': present_files
        }
    
    def _check_test_files(self, plugin_dir: Path) -> Dict[str, Any]:
        """Prüft Test-Dateien."""
        tests_dir = plugin_dir / "tests"
        
        if not tests_dir.exists():
            return {'has_tests': False, 'test_files': []}
        
        test_files = list(tests_dir.glob("test_*.py"))
        return {
            'has_tests': len(test_files) > 0,
            'test_files': [f.name for f in test_files]
        }
    
    def _check_requirements_file(self, plugin_dir: Path) -> Dict[str, Any]:
        """Prüft requirements.txt Datei."""
        req_file = plugin_dir / "requirements.txt"
        
        if not req_file.exists():
            return {'has_requirements': False, 'dependencies': []}
        
        try:
            content = req_file.read_text()
            dependencies = [line.strip() for line in content.split('\n') 
                          if line.strip() and not line.startswith('#')]
            return {
                'has_requirements': True,
                'dependencies': dependencies
            }
        except Exception:
            return {'has_requirements': False, 'dependencies': []}
    
    def _check_plugin_functionality(self, plugin: BasePlugin) -> Dict[str, Any]:
        """Prüft Plugin-Funktionalität."""
        try:
            # Prüfe Metadaten
            metadata = plugin.get_metadata()
            metadata_ok = all([
                metadata.name,
                metadata.version,
                metadata.description,
                metadata.author
            ])
            
            # Prüfe run() Methode (ohne sie auszuführen)
            run_method_ok = hasattr(plugin, 'run') and callable(getattr(plugin, 'run'))
            
            return {
                'metadata_ok': metadata_ok,
                'run_method_works': run_method_ok
            }
        except Exception as e:
            return {
                'metadata_ok': False,
                'run_method_works': False,
                'error': str(e)
            }
    
    def print_report(self):
        """Druckt einen formatierten Health Report."""
        print("\n" + "="*60)
        print("📊 PLUGIN HEALTH REPORT")
        print("="*60)
        
        summary = self.health_report.get('_summary', {})
        print(f"Gesamt Plugins: {summary.get('total_plugins', 0)}")
        print(f"✅ Gesunde Plugins: {summary.get('healthy_plugins', 0)}")
        print(f"⚠️  Plugins mit Warnungen: {summary.get('warning_plugins', 0)}")
        print(f"❌ Plugins mit Fehlern: {summary.get('error_plugins', 0)}")
        print(f"📈 Health Score: {summary.get('health_percentage', 0):.1f}%")
        
        print("\n" + "-"*60)
        print("📋 DETAILLIERTE BERICHTE")
        print("-"*60)
        
        for name, report in self.health_report.items():
            if name == '_summary':
                continue
            
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️ ',
                'error': '❌'
            }.get(report['status'], '❓')
            
            print(f"\n{status_icon} {report['name']} v{report['version']}")
            print(f"   Status: {report['status'].upper()}")
            
            if report['issues']:
                print(f"   Probleme: {', '.join(report['issues'])}")
            
            if report['warnings']:
                print(f"   Warnungen: {', '.join(report['warnings'])}")
            
            # Checks Details
            checks = report.get('checks', {})
            if checks.get('dependencies'):
                print("   ✅ Dependencies OK")
            else:
                print("   ❌ Dependencies fehlen")
            
            if checks.get('files', {}).get('all_present'):
                print("   ✅ Dateien OK")
            else:
                print("   ❌ Dateien fehlen")
            
            if checks.get('tests', {}).get('has_tests'):
                print("   ✅ Tests vorhanden")
            else:
                print("   ⚠️  Keine Tests")
    
    def save_report(self, output_file: Path):
        """Speichert den Health Report als JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.health_report, f, indent=2, default=str)
        print(f"\n💾 Health Report gespeichert: {output_file}")


def main():
    """Hauptfunktion des Health Checkers."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plugin Health Check")
    parser.add_argument("--output", "-o", help="JSON-Output-Datei")
    parser.add_argument("--quiet", "-q", action="store_true", help="Nur JSON-Output")
    
    args = parser.parse_args()
    
    checker = PluginHealthChecker()
    health_report = checker.check_all_plugins()
    
    if not args.quiet:
        checker.print_report()
    
    if args.output:
        checker.save_report(Path(args.output))
    
    # Exit Code basierend auf Health Score
    summary = health_report.get('_summary', {})
    health_percentage = summary.get('health_percentage', 0)
    
    if health_percentage >= 90:
        sys.exit(0)  # Alles OK
    elif health_percentage >= 70:
        sys.exit(1)  # Warnung
    else:
        sys.exit(2)  # Fehler


if __name__ == "__main__":
    main()