# Makefile für pytest-Tests

.PHONY: help test test-unit test-integration test-performance test-coverage test-fast test-slow clean

help: ## Zeige verfügbare Ziele
	@echo "Verfügbare Test-Ziele:"
	@echo "  test              - Führe alle Tests aus"
	@echo "  test-unit         - Führe nur Unit-Tests aus"
	@echo "  test-integration  - Führe nur Integrationstests aus"
	@echo "  test-performance  - Führe nur Performance-Tests aus"
	@echo "  test-coverage     - Führe Tests mit Coverage-Report aus"
	@echo "  test-fast         - Führe nur schnelle Tests aus"
	@echo "  test-slow         - Führe nur langsame Tests aus"
	@echo "  clean             - Bereinige Test-Artefakte"

test: ## Führe alle Tests aus
	pytest -v

test-unit: ## Führe nur Unit-Tests aus
	pytest -v -m "not integration and not performance"

test-integration: ## Führe nur Integrationstests aus
	pytest -v -m "integration"

test-performance: ## Führe nur Performance-Tests aus
	pytest -v -m "performance"

test-coverage: ## Führe Tests mit Coverage-Report aus
	pytest -v --cov=wlan_tool --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml

test-fast: ## Führe nur schnelle Tests aus
	pytest -v -m "not slow"

test-slow: ## Führe nur langsame Tests aus
	pytest -v -m "slow"

test-network: ## Führe Tests mit Netzwerk-Zugriff aus
	pytest -v -m "network"

test-hardware: ## Führe Tests mit Hardware-Anforderungen aus
	pytest -v -m "hardware"

test-verbose: ## Führe Tests mit detaillierter Ausgabe aus
	pytest -vv --tb=long

test-debug: ## Führe Tests im Debug-Modus aus
	pytest -vv --tb=long --pdb

clean: ## Bereinige Test-Artefakte
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf coverage.xml
	rm -rf *.db
	rm -rf /tmp/test_*.gexf
	rm -rf /tmp/test_*.html
	rm -rf /tmp/test_*.pcap
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

install-deps: ## Installiere Test-Abhängigkeiten
	pip install pytest pytest-cov pytest-mock psutil

# Standard-Ziel
.DEFAULT_GOAL := help