import os
from setuptools import setup, find_packages

SETUP = {
    'name': "hammer-time",
    'packages': find_packages(),
    'version': "0.1.0",
    'entry_points': {
        'console_scripts': [
            'hammer-time = hammer_time.hammer_time:main',
        ]
    },
    # Note: requirements.txt has the correct values to install these packages.
    'install_requires': ['jujupy'],
}


if __name__ == '__main__':
    setup(**SETUP)
