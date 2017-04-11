import os
from setuptools import setup, find_packages

SETUP = {
    'name': "hammertime",
    'packages': find_packages(),
    'version': "0.1.1",
    'entry_points': {
        'console_scripts': [
            'hammertime = hammertime.hammertime:main',
            'hammer-time = hammertime.hammertime:main',
        ]
    },
    # Note: requirements.txt has the correct values to install these packages.
    'install_requires': ['jujupy'],
}


if __name__ == '__main__':
    setup(**SETUP)
