develop: .develop-canary

.develop-canary: .python-canary setup.py
	bin/python setup.py develop
	bin/pip install nose
	touch .develop-canary

.python-canary: requirements.txt bin/pip3
	# When we update revno, pip doesn't consider it an upgrade.  So force a
	# reinstall.
	./remove-jujupy
	bin/pip3 install -r requirements.txt
	touch .python-canary

bin/pip3:
	sudo apt-get install python3-venv -y
	python3 -m venv .

lint:
	flake8 hammer_time

test:
	bin/nosetests hammer_time -v

.PHONY: lint test develop
