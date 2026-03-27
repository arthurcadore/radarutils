LOG_DIR := ./logs
LOG_FILE := $(LOG_DIR)/test.log

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

.PHONY: all install install-dev test export-pdf export-pdf-show export-all clean-all clean-packages

all: install 

# ==========================================================
# Package Management: 

install:
	@echo "Verificando Python..."
	@python3 --version || (echo "Python não encontrado. Por favor, instale o Python primeiro." && exit 1)
	@echo "Criando ambiente virtual..."
	python3 -m venv .venv
	@echo "Instalando dependências..."
	.venv/bin/pip install -r requirements.txt
	@echo "Instalação concluída!"

install-additional:
	@sudo apt update
	@sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

freeze: 
	@echo "Congelando dependências..."
	.venv/bin/pip freeze > requirements.txt
	@echo "Dependências congeladas em requirements.txt"

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
	.venv/bin/pytest -v
	@echo "Testes concluídos!"

generate-plots:
	@echo "Gerando plots..."
	@for script in tests/plots/*.py; do \
		if [ -f "$$script" ]; then \
			echo "Executando $$script..."; \
			.venv/bin/python "$$script"; \
		fi; \
	done
	@echo "Plots gerados em ./assets/"

# ==========================================================
# Documentação: 

images:
	@echo "Gerando imagens para a web..."
	.venv/bin/python web/scripts/pdf2svg.py
	@echo "Imagens geradas!"

doc: images build-doc
	@echo "Apresentando doc..."
	mkdocs serve -a 0.0.0.0:8006
	
deploy-docs: build-doc
	@echo "Fazendo deploy para o GitHub Pages..."
	@mkdocs gh-deploy --force
	@echo "Documentação publicada com sucesso no GitHub Pages!"


build-doc: clean-packages
	@echo "Executando build da documentação..."
	@mkdocs build