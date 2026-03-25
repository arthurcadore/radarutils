LOG_DIR := ./logs
LOG_FILE := $(LOG_DIR)/test.log

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

.PHONY: all install install-dev test export-pdf export-pdf-show export-all clean clean-test

all: install 

install:
	@echo "Verificando Python..."
	@python3 --version || (echo "Python não encontrado. Por favor, instale o Python primeiro." && exit 1)
	@echo "Criando ambiente virtual..."
	python3 -m venv .venv
	@echo "Instalando dependências..."
	.venv/bin/pip install -r requirements.txt
	@echo "Instalação concluída!"

install-dev:
	@echo "Verificando Python..."
	@python3 --version || (echo "Python não encontrado. Por favor, instale o Python primeiro." && exit 1)
	@echo "Criando ambiente virtual..."
	python3 -m venv .venv
	@echo "Instalando dependências..."
	.venv/bin/pip install -r requirements.txt
	@echo "Instalando dependências de desenvolvimento..."
	.venv/bin/pip install -e .[dev]
	@echo "Instalação concluída!"

test:
	@echo "Executando testes..."
	.venv/bin/pytest

clean-test:
	@echo "Limpando arquivos de teste..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

install-additional:
	@sudo apt update
	@sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

clean:
	rm -rf .venv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

images: remove-images
	@echo "Gerando imagens para a documentação..."
	.venv/bin/python docs/pdf2svg_plots.py
	.venv/bin/python docs/pdf2svg_assets.py
	@echo "Imagens geradas!"

remove-images:
	@echo "Removendo imagens..."
	@find docs/api/assets/ -type f -exec rm -f {} \; || true

doc: 
	@echo "Gerando documentação..."
	mkdocs build
	mkdocs serve -a 0.0.0.0:8006
	
freeze: 
	@echo "Congelando dependências..."
	.venv/bin/pip freeze > requirements.txt
	@echo "Dependências congeladas em requirements.txt"

deploy-docs:
	@echo "Gerando documentação..."
	@if ! command -v mkdocs &> /dev/null; then \
		echo "MkDocs não encontrado. Instalando..."; \
		.venv/bin/pip install mkdocs mkdocs-material; \
	fi
	@echo "Fazendo deploy para o GitHub Pages..."
	@mkdocs gh-deploy --force
	@echo "Documentação publicada com sucesso no GitHub Pages!"

clean-packages:
	@echo "Limpeza de pacotes..."
	rm -rf dist/
	
build: clean-packages
	@echo "Building packages..."
	.venv/bin/python3 -m build

upload: build
	@echo "Uploading to PyPI..."
	.venv/bin/python3 -m twine upload dist/*