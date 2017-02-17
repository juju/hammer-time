install-deps: .python-canary

.python-canary: requirements.txt bin/pip3
	bin/pip3 install -r requirements.txt
	touch .python-canary
bin/pip3:
	python3 venv .

.PHONY: install-deps
