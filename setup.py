import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))
reqs_file = os.path.join(this_dir, 'requirements.txt')
with open(reqs_file) as f:
    reqs = [line for line in f.read().splitlines()
            if not line.startswith('--')]

SETUP = {
    'name': "hammer-time",
    'packages': find_packages(),
    'version': "0.1.0",
    'entry_points': {
        'console_scripts': [
            'hammer-time = hammer_time.hammer_time:main',
            'h-time = hammer_time.hammer_time:main',
        ]
    },
    'install_requires': reqs,
}


if __name__ == '__main__':
    setup(**SETUP)
