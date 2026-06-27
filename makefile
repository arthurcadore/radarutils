LOG_DIR := ./logs
LOG_FILE := $(LOG_DIR)/test.log

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

.PHONY: all install install-dev install-additional lock test plots images doc deploy-docs build-doc clean-all clean-packages

all: install

# ==========================================================
# Package Management:

install:
	@echo "Verificando uv..."
	@uv --version || (echo "uv não encontrado. Instale o uv: https://docs.astral.sh/uv/getting-started/installation/" && exit 1)
	@echo "Sincronizando ambiente e dependências..."
	uv sync --all-groups
	@echo "Instalação concluída!"

install-additional:
	@sudo apt update
	@sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

lock:
	@echo "Atualizando uv.lock..."
	uv lock
	@echo "Lockfile atualizado em uv.lock"

clean-all:
	@echo "Executando limpando arquivos..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .venv
	rm -rf assets/example_antenna_patterns*.pdf
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

clean-packages:
	@echo "Executando limpeza de pacotes..."
	rm -rf dist/

# ==========================================================
# Testes:

test:
	@echo "Executando todos os testes..."
	uv run pytest -v -s
	@echo "Testes concluídos!"

# ==========================================================
# Documentação:

plots:
	@echo "Gerando plots..."
	@for script in tests/plots/*.py; do \
		if [ -f "$$script" ]; then \
			echo "Executando $$script..."; \
			uv run python "$$script"; \
		fi; \
	done
	@echo "Plots gerados em ./assets/"

images: plots
	@echo "Gerando imagens para a web..."
	uv run python web/scripts/pdf2svg.py
	@echo "Imagens geradas!"

doc: images build-doc
	@echo "Apresentando doc..."
	uv run mkdocs serve -a 0.0.0.0:8006

deploy-docs: build-doc
	@echo "Fazendo deploy para o GitHub Pages..."
	@uv run mkdocs gh-deploy --force
	@echo "Documentação publicada com sucesso no GitHub Pages!"


build-doc: clean-packages
	@echo "Executando build da documentação..."
	@uv run mkdocs build
