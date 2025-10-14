#!/usr/bin/env python3
"""
Code-Qualitäts-Tools für das WLAN-Tool.
Führt Linting, Type-Checking, Formatierung und Sicherheits-Checks durch.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class CodeQualityChecker:
    """Hauptklasse für Code-Qualitäts-Checks."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.results = {}

    def run_flake8(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt Flake8 Linting durch."""
        print("🔍 Führe Flake8 Linting durch...")

        cmd = [
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            "--exclude=.git,__pycache__,venv,env,.venv",
            "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
            "--statistics",
        ]

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("tests")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["flake8"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ Flake8: Keine Linting-Fehler gefunden")
            else:
                print(f"❌ Flake8: {result.returncode} Fehler gefunden")
                print(result.stdout)

            return self.results["flake8"]

        except FileNotFoundError:
            print("❌ Flake8 nicht installiert. Installiere mit: pip install flake8")
            return {"success": False, "error": "Flake8 not installed"}

    def run_mypy(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt MyPy Type-Checking durch."""
        print("🔍 Führe MyPy Type-Checking durch...")

        cmd = ["mypy", "--config-file=pyproject.toml", "--show-error-codes", "--pretty"]

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("tests")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["mypy"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ MyPy: Keine Type-Checking-Fehler gefunden")
            else:
                print(f"❌ MyPy: {result.returncode} Fehler gefunden")
                print(result.stdout)

            return self.results["mypy"]

        except FileNotFoundError:
            print("❌ MyPy nicht installiert. Installiere mit: pip install mypy")
            return {"success": False, "error": "MyPy not installed"}

    def run_black(
        self, check_only: bool = True, paths: List[str] = None
    ) -> Dict[str, Any]:
        """Führt Black Code-Formatierung durch."""
        print("🔍 Führe Black Code-Formatierung durch...")

        cmd = ["black", "--config=pyproject.toml"]

        if check_only:
            cmd.append("--check")
            cmd.append("--diff")

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("tests")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["black"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ Black: Code ist korrekt formatiert")
            else:
                print(f"❌ Black: Code-Formatierung erforderlich")
                if result.stdout:
                    print(result.stdout)

            return self.results["black"]

        except FileNotFoundError:
            print("❌ Black nicht installiert. Installiere mit: pip install black")
            return {"success": False, "error": "Black not installed"}

    def run_isort(
        self, check_only: bool = True, paths: List[str] = None
    ) -> Dict[str, Any]:
        """Führt isort Import-Sortierung durch."""
        print("🔍 Führe isort Import-Sortierung durch...")

        cmd = ["isort", "--profile=black", "--line-length=88"]

        if check_only:
            cmd.append("--check-only")
            cmd.append("--diff")

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("tests")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["isort"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ isort: Imports sind korrekt sortiert")
            else:
                print(f"❌ isort: Import-Sortierung erforderlich")
                if result.stdout:
                    print(result.stdout)

            return self.results["isort"]

        except FileNotFoundError:
            print("❌ isort nicht installiert. Installiere mit: pip install isort")
            return {"success": False, "error": "isort not installed"}

    def run_bandit(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt Bandit Sicherheits-Scan durch."""
        print("🔍 Führe Bandit Sicherheits-Scan durch...")

        cmd = [
            "bandit",
            "-r",
            "-f",
            "json",
            "-o",
            "bandit-report.json",
            "--exclude",
            "tests",
            "--exclude",
            "venv",
            "--exclude",
            "env",
        ]

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            # Bandit-Report lesen
            report_file = self.project_root / "bandit-report.json"
            bandit_results = {}
            if report_file.exists():
                with open(report_file, "r") as f:
                    bandit_results = json.load(f)

            self.results["bandit"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "results": bandit_results,
            }

            if result.returncode == 0:
                print("✅ Bandit: Keine Sicherheitsprobleme gefunden")
            else:
                print(f"❌ Bandit: {result.returncode} Sicherheitsprobleme gefunden")
                if result.stdout:
                    print(result.stdout)

            return self.results["bandit"]

        except FileNotFoundError:
            print("❌ Bandit nicht installiert. Installiere mit: pip install bandit")
            return {"success": False, "error": "Bandit not installed"}

    def run_safety(self) -> Dict[str, Any]:
        """Führt Safety Dependency-Scan durch."""
        print("🔍 Führe Safety Dependency-Scan durch...")

        cmd = ["safety", "check", "--json"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["safety"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ Safety: Keine bekannten Sicherheitslücken in Dependencies")
            else:
                print(
                    f"❌ Safety: {result.returncode} Sicherheitslücken in Dependencies gefunden"
                )
                if result.stdout:
                    print(result.stdout)

            return self.results["safety"]

        except FileNotFoundError:
            print("❌ Safety nicht installiert. Installiere mit: pip install safety")
            return {"success": False, "error": "Safety not installed"}

    def run_radon(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt Radon Komplexitäts-Analyse durch."""
        print("🔍 Führe Radon Komplexitäts-Analyse durch...")

        cmd = ["radon", "cc", "-a", "-s", "--min", "B"]

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["radon"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ Radon: Komplexitäts-Analyse abgeschlossen")
            else:
                print(f"❌ Radon: Komplexitäts-Probleme gefunden")
                if result.stdout:
                    print(result.stdout)

            return self.results["radon"]

        except FileNotFoundError:
            print("❌ Radon nicht installiert. Installiere mit: pip install radon")
            return {"success": False, "error": "Radon not installed"}

    def run_xenon(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt Xenon Komplexitäts-Monitoring durch."""
        print("🔍 Führe Xenon Komplexitäts-Monitoring durch...")

        cmd = [
            "xenon",
            "--max-average",
            "A",
            "--max-modules",
            "B",
            "--max-absolute",
            "B",
        ]

        if paths:
            cmd.extend(paths)
        else:
            cmd.append("wlan_tool")
            cmd.append("scripts")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            self.results["xenon"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            if result.returncode == 0:
                print("✅ Xenon: Komplexität ist akzeptabel")
            else:
                print(f"❌ Xenon: Komplexitäts-Grenzen überschritten")
                if result.stdout:
                    print(result.stdout)

            return self.results["xenon"]

        except FileNotFoundError:
            print("❌ Xenon nicht installiert. Installiere mit: pip install xenon")
            return {"success": False, "error": "Xenon not installed"}

    def run_all_checks(self, paths: List[str] = None) -> Dict[str, Any]:
        """Führt alle Code-Qualitäts-Checks durch."""
        print("🚀 Führe alle Code-Qualitäts-Checks durch...")

        checks = [
            self.run_flake8,
            self.run_mypy,
            self.run_black,
            self.run_isort,
            self.run_bandit,
            self.run_safety,
            self.run_radon,
            self.run_xenon,
        ]

        all_success = True
        for check in checks:
            try:
                result = check(paths)
                if not result.get("success", False):
                    all_success = False
            except Exception as e:
                print(f"❌ Fehler bei {check.__name__}: {e}")
                all_success = False

        # Zusammenfassung
        print("\n" + "=" * 50)
        print("📊 CODE-QUALITÄTS-ZUSAMMENFASSUNG")
        print("=" * 50)

        for tool, result in self.results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"{status} {tool.upper()}")

        overall_status = (
            "✅ ALLE CHECKS ERFOLGREICH"
            if all_success
            else "❌ EINIGE CHECKS FEHLGESCHLAGEN"
        )
        print(f"\n{overall_status}")

        return {"overall_success": all_success, "results": self.results}

    def fix_formatting(self, paths: List[str] = None) -> Dict[str, Any]:
        """Korrigiert Code-Formatierung automatisch."""
        print("🔧 Korrigiere Code-Formatierung...")

        # Black Formatierung
        black_result = self.run_black(check_only=False, paths=paths)

        # isort Import-Sortierung
        isort_result = self.run_isort(check_only=False, paths=paths)

        return {
            "black": black_result,
            "isort": isort_result,
            "success": black_result.get("success", False)
            and isort_result.get("success", False),
        }


def main():
    """Hauptfunktion für CLI."""
    parser = argparse.ArgumentParser(description="Code-Qualitäts-Tools für WLAN-Tool")
    parser.add_argument(
        "--tool",
        choices=[
            "flake8",
            "mypy",
            "black",
            "isort",
            "bandit",
            "safety",
            "radon",
            "xenon",
            "all",
        ],
        default="all",
        help="Welches Tool ausführen",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Formatierung automatisch korrigieren"
    )
    parser.add_argument("--paths", nargs="+", help="Spezifische Pfade prüfen")
    parser.add_argument("--project-root", type=Path, help="Projekt-Root-Verzeichnis")

    args = parser.parse_args()

    checker = CodeQualityChecker(args.project_root)

    if args.fix and args.tool in ["black", "isort", "all"]:
        result = checker.fix_formatting(args.paths)
        sys.exit(0 if result["success"] else 1)

    if args.tool == "all":
        result = checker.run_all_checks(args.paths)
    else:
        tool_method = getattr(checker, f"run_{args.tool}")
        result = tool_method(args.paths)

    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
