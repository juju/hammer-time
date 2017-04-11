Hammertime
===========
Test Juju's behaviour under non-ideal conditions, such as machines rebooting,
processes dying, etc.

Its chaos-generating code is inspired by
[Matrix](https://github.com/juju-solutions/matrix), but whereas Matrix is
designed to test charms, Hammertime tests Juju itself.

Setup
-----
Setup is developer-oriented.  Run "make develop".  This will create a
virtualenv with Hammer Time installed as bin/hammertime.
