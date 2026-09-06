.PHONY: install lint test smoke table clean

install:
	pip install -e ".[dev]"

lint:
	ruff format --check src tests
	ruff check src tests
	mypy src

test:
	pytest

smoke:
	bash scripts/smoke_cpu.sh

table:
	python -m shardkit report --runs runs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
