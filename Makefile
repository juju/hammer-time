develop: .develop-canary

.develop-canary: .python-canary
	bin/python setup.py develop
	touch .develop-canary

.python-canary: requirements.txt bin/pip3
	bin/pip3 install -r requirements.txt
	touch .python-canary

bin/pip3:
	python3 -m venv .

lint:
	flake8 hammer_time

test:
	bin/nosetests hammer_time -v

.PHONY: lint test develop
