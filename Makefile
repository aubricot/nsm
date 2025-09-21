# NSM Development Makefile
# Simplifies common development tasks

.PHONY: help install install-dev test test-coverage lint format clean

# Default target
help:
	@echo "NSM Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install NSM package in current environment"
	@echo "  install-dev      Install NSM in development mode with dev dependencies"
	@echo "  test             Run all tests with pytest"
	@echo "  test-loader      Run only model loader tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run code linting (flake8)"
	@echo "  format           Format code with black"
	@echo "  format-check     Check if code formatting is correct"
	@echo "  clean            Clean up temporary files and caches"
	@echo "  env-setup        Setup conda development environment"

# Installation targets
install:
	pip install .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Testing targets
test:
	pytest testing/ -v

test-loader:
	pytest testing/NSM/models/test_loader.py -v

test-coverage:
	pytest testing/ --cov=NSM --cov-report=html --cov-report=term-missing

# TODO: Add parallel testing once pytest-xdist is fully configured
# test-parallel:
#	pytest testing/ -n auto -v

# Code quality targets
lint:
	flake8 NSM/ testing/

format:
	black NSM/ testing/

format-check:
	black NSM/ testing/ --check

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Environment setup
env-setup:
	conda create -n nsm-dev python=3.9 -y
	@echo "Environment created. Activate it with: conda activate nsm-dev"
	@echo "Then run: make install-dev"

# TODO: Documentation targets - considering pdoc vs sphinx
# Currently using pdoc might be easier than sphinx for this project
# docs:
#	pdoc --html --output-dir docs NSM
# OR for sphinx (once configured):
# docs:
#	sphinx-build -b html docs docs/_build/html

# TODO: CI/CD targets - set these up once CI is configured
# ci-test: install-dev test-coverage lint
#	@echo "CI tests completed!"

# Quick development helpers
quick-test: format test-loader
	@echo "Quick development cycle completed!"