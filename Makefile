install-deps: bin/pip3
bin/pip3:
	virtualenv . --python=python3

.PHONY: install-deps
