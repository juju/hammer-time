install-deps: .python-canary

.python-canary: requirements.txt bin/pip3
	bin/pip3 install -r requirements.txt
	touch .python-canary

bin/pip3:
	python3 venv .

lint:
	flake8 hammer_time

.PHONY: install-deps
