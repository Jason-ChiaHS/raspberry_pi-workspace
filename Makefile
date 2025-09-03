PATH_TO_PROJECT_ROOT = ../..

# Help Commands
.PHONY: default
default: help

.PHONY: create-venv
create-venv:
	if [ ! -d ./.venv ]; then \
		python -m venv .venv --system-site-packages; \
	fi

.PHONY: install-sdk
install-sdk:
	./.venv/bin/pip install -r requirements.txt
	./.venv/bin/pip install -e .


.PHONY: install-raspi
install-raspi:
	if [ -f ./raspi-cam-srv-fork/requirements.txt ]; then \
		./.venv/bin/pip install -r raspi-cam-srv-fork/requirements.txt; \
		./.venv/bin/pip install -e raspi-cam-srv-fork; \
	fi

.PHONY: install-webui
install-webui:
	cd scripts/web_ui; \
	npm i;

.PHONY: build-webui
build-webui:
	cd scripts/web_ui; \
	npm run build;

# CH: For some reason, the system numpy package always get override on install
# This removes the installed version so it defaults back to the system numpy package
# This is important because the major version upgrade breaks other system python related packages
# I also do not fully understand how and why this is the case
.PHONY: fix-numpy
fix-numpy:
	./.venv/bin/pip uninstall numpy -y

# Main Commands
.PHONY: install
install: create-venv install-sdk install-raspi install-webui build-webui fix-numpy

.PHONY: install-hdmi # All python packages without Node
install-hdmi: create-venv install-sdk


.PHONY: run-hdmi-gazev3 # All python packages without Node
run-hdmi-gaze-v2:
	./.venv/bin/python scripts/show_result_to_hdmi.py --demo gazev3 --cv2_show_window

.PHONY: run-webui
run-webui:
	npm run --prefix scripts/web_ui/ preview -- --host & \
	./.venv/bin/python scripts/show_result_to_web_ui.py

.PHONY: install-dev
install-dev: create-venv
	./.venv/bin/pip install -r requirements-dev.txt

.PHONY: clean
clean:
	rm -r .venv

.PHONY: help
help:
	@echo 'PLEASE READ THE README.md before running any commands'
	@echo 'Usage: make [command]'
	@echo 'install - Creates the venv and installs the required packages'
	@echo 'clean - Cleans the .venv'
