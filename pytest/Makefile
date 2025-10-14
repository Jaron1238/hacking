# Makefile für Plugin-Tests

.PHONY: help test test-plugins test-clustering test-ensemble test-rl test-all clean install-deps

help: ## Zeige verfügbare Ziele
	@echo "Verfügbare Ziele:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Führe alle Tests aus
	pytest -v

test-plugins: ## Führe nur Plugin-Tests aus
	pytest -v -m plugin

test-clustering: ## Führe Clustering-Plugin-Tests aus
	pytest -v -m clustering

test-ensemble: ## Führe Ensemble-Plugin-Tests aus
	pytest -v -m ensemble

test-rl: ## Führe RL-Plugin-Tests aus
	pytest -v -m rl

test-all: ## Führe alle Tests mit Coverage aus
	pytest -v --cov=wlan_tool --cov=plugins --cov-report=html:htmlcov --cov-report=term-missing

test-fast: ## Führe schnelle Tests aus (ohne langsame)
	pytest -v -m "not slow"

test-integration: ## Führe Integration-Tests aus
	pytest -v -m integration

test-unit: ## Führe Unit-Tests aus
	pytest -v -m unit

install-deps: ## Installiere Plugin-Dependencies
	@echo "Installiere Dependencies für alle Plugins..."
	@for plugin_dir in ../plugins/*/; do \
		if [ -f "$$plugin_dir/requirements.txt" ]; then \
			echo "Installing dependencies for $$(basename $$plugin_dir)..."; \
			pip install -r "$$plugin_dir/requirements.txt"; \
		fi; \
	done

clean: ## Bereinige Test-Artefakte
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

lint: ## Führe Linting aus
	flake8 ../plugins/
	black --check ../plugins/
	isort --check-only ../plugins/

format: ## Formatiere Code
	black ../plugins/
	isort ../plugins/

# Plugin-spezifische Ziele
test-clustering-advanced: ## Teste nur Advanced Clustering Plugin
	python -m pytest -v ../plugins/clustering_advanced/tests/

test-ensemble-models: ## Teste nur Ensemble Models Plugin
	python -m pytest -v ../plugins/ensemble_models/tests/

test-reinforcement-learning: ## Teste nur RL Plugin
	python -m pytest -v ../plugins/reinforcement_learning/tests/

test-sankey: ## Teste nur Sankey Plugin
	python -m pytest -v ../plugins/sankey/tests/

test-umap-plot: ## Teste nur UMAP Plot Plugin
	python -m pytest -v ../plugins/umap_plot/tests/

health-check: ## Führe Plugin Health Check aus
	python3 ../scripts/plugin_health_check.py

health-check-json: ## Führe Plugin Health Check aus und speichere als JSON
	python3 ../scripts/plugin_health_check.py --output plugin_health.json

plugin-manager: ## Zeige Plugin Manager Hilfe
	python3 ../scripts/plugin_manager.py --help

plugin-list: ## Liste alle Plugins auf
	python3 ../scripts/plugin_manager.py list

plugin-install: ## Installiere alle Plugin Dependencies
	python3 ../scripts/plugin_manager.py install

plugin-update: ## Aktualisiere alle Plugin Dependencies
	python3 ../scripts/plugin_manager.py update