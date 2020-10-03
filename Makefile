# "DAGS"
all: install_dependencies run_project

install_dependencies: install_python install_poetry install_project_dependencies
run_project: build_features

# TASKS
install_python:
	@echo "Installing python..."
	sudo apt-get install python3 python3-dev python3-pip -y

install_poetry:
	@echo "Installing poetry..."
	sudo curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

install_project_dependencies:
	@echo "Installing project dependencies..."
	poetry shell
	poetry install

build_features:
	@echo "Building features..."
	python nlp/features/build_features.py
