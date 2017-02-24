Hammer Time
===========
This is a tool that tests Juju's behaviour under non-ideal conditions, such as
machines rebooting, processes dying, etc.

It uses chaos-generating code from
[Matrix](https://github.com/juju-solutions/matrix), but whereas Matrix is
designed to test charms, Hammer Time tests Juju itself.

Setup
-----
Setup is currently ugly.  You need to manually:

1. Install a virtualenv
2. Get juju-ci-tools (lp:juju-ci-tools)
3. Use "bin/pip $JUJU_CI_TOOLS_PATH" to install it.
4. Get Matrix (https://github.com/juju-solutions/matrix)
5. Use "bin/pip $MATRIX_PATH" to install it.
6. Use bin/python setup.py develop to install Hammer Time.

This will get better as packaging status improves.
