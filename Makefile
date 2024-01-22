VENV := .venv
REQUIREMENTS := requirements.txt

.PHONY: dev venv requirements
.PHONY: black
.PHONY: clean


dev: setup-submodules venv requirements

	@printf "\n\nDevelopment Environment is now setup\n"
	@printf "Run 'source $(VENV)/bin/activate' to enter virtual environment\n"

setup-submodules:
	@git submodule update --init --recursive

venv:
	@python3.9 -m venv $(VENV)

requirements:
	@. $(VENV)/bin/activate && \
		pip install -r $(REQUIREMENTS)

black:
	@. $(VENV)/bin/activate && \
		black --diff --check . 

isort:
	@. $(VENV)/bin/activate && \
		isort . --profile black

lint: isort black

clean: clean-venv

clean-venv:
	@echo "Cleaning venv"
	rm -rf $(VENV)
	@echo
	@echo "Run 'deactivate' to exit virtual environment"


install:
	@. $(VENV)/bin/activate && \
		pip install -e ./labelsom