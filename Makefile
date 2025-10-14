# WLAN-Tool Makefile
# Vereinfacht die Ausführung von Tests, Code-Qualitäts-Checks und Performance-Tests

.PHONY: help install install-dev test test-unit test-integration test-performance test-plugins test-security test-docs test-all lint format type-check security-check quality-check benchmark profile clean

# Standard-Ziel
help:
	@echo "WLAN-Tool Makefile"
	@echo "=================="
	@echo ""
	@echo "Installation:"
	@echo "  install      - Installiere Produktions-Dependencies"
	@echo "  install-dev  - Installiere Entwicklungs-Dependencies"
	@echo ""
	@echo "Tests:"
	@echo "  test         - Führe alle Tests aus"
	@echo "  test-unit    - Führe Unit-Tests aus"
	@echo "  test-integration - Führe Integration-Tests aus"
	@echo "  test-performance - Führe Performance-Tests aus"
	@echo "  test-plugins - Führe Plugin-Tests aus"
	@echo "  test-security - Führe Security-Tests aus"
	@echo "  test-docs    - Führe Dokumentations-Tests aus"
	@echo "  test-all     - Führe alle Tests mit Coverage aus"
	@echo ""
	@echo "Code-Qualität:"
	@echo "  lint         - Führe Linting durch (flake8)"
	@echo "  format       - Formatiere Code (black, isort)"
	@echo "  type-check   - Führe Type-Checking durch (mypy)"
	@echo "  security-check - Führe Security-Checks durch"
	@echo "  quality-check - Führe alle Code-Qualitäts-Checks durch"
	@echo ""
	@echo "Performance:"
	@echo "  benchmark    - Führe Benchmark-Tests aus"
	@echo "  profile      - Führe Memory-Profiling aus"
	@echo ""
	@echo "Sonstiges:"
	@echo "  clean        - Bereinige temporäre Dateien"
	@echo "  pre-commit   - Installiere pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Tests
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v --cov=wlan_tool --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v --cov=wlan_tool --cov-report=html --cov-report=term-missing

test-performance:
	pytest tests/performance/ -v --benchmark-only --benchmark-sort=mean

test-plugins:
	pytest tests/plugins/ -v --cov=plugins --cov-report=html --cov-report=term-missing

test-security:
	pytest tests/security/ -v

test-docs:
	pytest tests/documentation/ -v

test-all:
	pytest tests/ -v --cov=wlan_tool --cov-report=html --cov-report=term-missing --cov-report=xml --junitxml=test-results.xml

# Code-Qualität
lint:
	flake8 wlan_tool tests scripts --max-line-length=88 --extend-ignore=E203,W503

format:
	black wlan_tool tests scripts
	isort wlan_tool tests scripts --profile=black

type-check:
	mypy wlan_tool tests scripts --config-file=pyproject.toml

security-check:
	bandit -r wlan_tool scripts --exclude tests
	safety check

quality-check:
	python scripts/code_quality.py --tool all

# Performance
benchmark:
	pytest tests/performance/test_benchmarks.py -v --benchmark-only --benchmark-sort=mean --benchmark-json=benchmark-results.json

profile:
	pytest tests/performance/test_memory_profiling.py -v -m memory

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Plugin-Management
plugin-install:
	python scripts/plugin_manager.py install-all

plugin-test:
	python scripts/plugin_manager.py test-all

plugin-health:
	python scripts/plugin_health_check.py

# Bereinigung
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .benchmarks/
	rm -rf bandit-report.json
	rm -rf safety-report.json
	rm -rf test-results.xml
	rm -rf benchmark-results.json

# Entwicklungs-Workflow
dev-setup: install-dev pre-commit
	@echo "Entwicklungsumgebung eingerichtet!"

dev-test: quality-check test-all
	@echo "Alle Tests und Checks abgeschlossen!"

# CI/CD Simulation
ci-simulate: quality-check test-all benchmark profile
	@echo "CI/CD Pipeline simuliert!"

# Spezielle Test-Marker
test-slow:
	pytest tests/ -v -m slow

test-fast:
	pytest tests/ -v -m "not slow"

test-clustering:
	pytest tests/ -v -m clustering

test-ensemble:
	pytest tests/ -v -m ensemble

test-rl:
	pytest tests/ -v -m rl

# Coverage-Berichte
coverage-html:
	pytest tests/ --cov=wlan_tool --cov-report=html
	@echo "Coverage-Bericht erstellt: htmlcov/index.html"

coverage-xml:
	pytest tests/ --cov=wlan_tool --cov-report=xml
	@echo "Coverage-XML erstellt: coverage.xml"

# Performance-Monitoring
monitor:
	@echo "Starte Performance-Monitoring..."
	watch -n 1 'ps aux | grep python | head -5'

# Dokumentation
docs-build:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Package-Build
build:
	python -m build

build-check:
	twine check dist/*

# Git-Hooks
git-hooks:
	@echo "Git-Hooks werden eingerichtet..."
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

# Docker (falls verwendet)
docker-build:
	docker build -t wlan-tool .

docker-test:
	docker run --rm wlan-tool make test-all

# Spezielle Workflows
quick-check: lint type-check test-unit
	@echo "Schnelle Qualitätsprüfung abgeschlossen!"

full-check: quality-check test-all benchmark profile
	@echo "Vollständige Qualitätsprüfung abgeschlossen!"

# Hilfe für spezifische Bereiche
help-tests:
	@echo "Test-Optionen:"
	@echo "  test-unit        - Nur Unit-Tests"
	@echo "  test-integration - Nur Integration-Tests"
	@echo "  test-performance - Nur Performance-Tests"
	@echo "  test-plugins     - Nur Plugin-Tests"
	@echo "  test-security    - Nur Security-Tests"
	@echo "  test-docs        - Nur Dokumentations-Tests"
	@echo "  test-all         - Alle Tests mit Coverage"

help-quality:
	@echo "Code-Qualitäts-Optionen:"
	@echo "  lint           - Flake8 Linting"
	@echo "  format         - Black + isort Formatierung"
	@echo "  type-check     - MyPy Type-Checking"
	@echo "  security-check - Bandit + Safety"
	@echo "  quality-check  - Alle Qualitäts-Checks"

help-performance:
	@echo "Performance-Optionen:"
	@echo "  benchmark      - Benchmark-Tests"
	@echo "  profile        - Memory-Profiling"
	@echo "  monitor        - Live Performance-Monitoring"